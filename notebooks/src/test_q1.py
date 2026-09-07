"""Adversarial checks for frozen splits and sequence integrity."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import q1
except ImportError:
    from notebooks.src import q1


def manifest(positions, genes=None, chromosomes=None):
    n = len(positions)
    chromosomes = chromosomes or ['1'] * n
    return pd.DataFrame({
        'chrom': chromosomes, 'pos': positions, 'ref': ['A'] * n, 'alt': ['C'] * n,
        'gene_ids': genes or [str(i + 1) for i in range(n)],
        'variation_id': [str(i + 100) for i in range(n)],
        'ALLELEID': [str(i + 200) for i in range(n)],
        'variant_key': [f'GRCh38:{c}:{p}:A:C' for c, p in zip(chromosomes, positions)],
        'locus_id': [f'{c}:{p}' for c, p in zip(chromosomes, positions)],
        'start': np.array(positions) - 1 - q1.CONTEXT // 2,
        'end': np.array(positions) - 1 + q1.CONTEXT // 2,
    })



class GroupingTests(unittest.TestCase):
    def test_unselected_multigene_bridge_is_preserved(self):
        full = manifest([2000, 10000, 20000], ['1', '1|2', '2'])
        graph = q1.full_components(full)
        self.assertEqual(graph.find(0), graph.find(2))
        selected_only = q1.full_components(full.iloc[[0, 2]].reset_index(drop=True))
        self.assertNotEqual(selected_only.find(0), selected_only.find(1))

    def test_transitive_windows_touching_boundary_and_chromosomes(self):
        full = manifest([10, 13, 16, 20, 20], chromosomes=['1', '1', '1', '1', '2'])
        graph = q1.full_components(full, context=4)
        self.assertEqual(graph.find(0), graph.find(2))
        self.assertNotEqual(graph.find(2), graph.find(3))
        self.assertNotEqual(graph.find(3), graph.find(4))

    def test_alternate_alleles_and_shared_source_ids(self):
        full = manifest([2000, 2000, 20000, 40000])
        full.loc[1, 'variant_key'] = 'GRCh38:1:2000:A:T'
        full.loc[2, 'ALLELEID'] = '200|202'
        full.loc[3, 'variation_id'] = '102|103'
        graph = q1.full_components(full)
        self.assertEqual(len(set(graph.names(full.variant_key))), 1)

    def test_group_names_and_sampling_do_not_depend_on_order_or_labels(self):
        full = manifest([2000, 10000, 20000], ['1', '1|2', '3'])
        names = dict(zip(full.variant_key, q1.full_components(full).names(full.variant_key)))
        shuffled = full.iloc[::-1].reset_index(drop=True)
        other = dict(zip(shuffled.variant_key, q1.full_components(shuffled).names(shuffled.variant_key)))
        self.assertEqual(names, other)
        a = full.assign(label=[0, 0, 1])
        b = full.assign(label=[1, 1, 0])
        self.assertEqual(q1.sample_indices(a, 2), q1.sample_indices(b, 2))
        self.assertEqual(set(a.iloc[q1.sample_indices(a, 2)].variant_key),
                         set(shuffled.iloc[q1.sample_indices(shuffled, 2)].variant_key))

    def test_reverse_complement_and_ref_alt_identity_merge(self):
        full = manifest([2000, 10000, 20000])
        graph = q1.full_components(full)
        dna = pd.DataFrame([
            [full.variant_key[0], 'AAAC', 'AACC'],
            [full.variant_key[1], 'GGTT', 'AGCC'],  # RC of first ALT
            [full.variant_key[2], 'AGCC', 'CCCC'],
        ], columns=q1.DNA_COLUMNS)
        q1.merge_identical_contexts(graph, full, dna)
        self.assertEqual(len(set(graph.names(full.variant_key))), 1)

    def test_independent_audit_detects_gene_leakage(self):
        full = manifest([2000, 10000], ['1', '1'])
        full['component'] = ['a', 'b']
        full['split'] = ['train', 'validation']
        with self.assertRaisesRegex(AssertionError, 'gene_ids'):
            q1.audit_splits(full, full, pd.DataFrame())

    def test_independent_audit_detects_interval_leakage(self):
        full = manifest([2000, 2500])
        full['component'] = ['a', 'b']
        full['split'] = ['train', 'validation']
        with self.assertRaisesRegex(AssertionError, 'Overlapping'):
            q1.audit_splits(full, full, pd.DataFrame())

    def test_independent_audit_detects_identical_context_leakage(self):
        full = manifest([2000, 10000])
        full['component'] = ['a', 'b']
        full['split'] = ['train', 'validation']
        ref = 'A' * q1.CONTEXT
        alt = ref[:512] + 'C' + ref[513:]
        pilot = full.assign(ref_context_hash=q1.context_hash(ref), alt_context_hash=q1.context_hash(alt))
        dna = pd.DataFrame([[key, ref, alt] for key in full.variant_key], columns=q1.DNA_COLUMNS)
        with self.assertRaisesRegex(AssertionError, 'Identical'):
            q1.audit_splits(full, pilot, dna)



class SequenceTests(unittest.TestCase):
    def test_reference_and_exact_mutation(self):
        ref, alt, reason = q1.mutate_context('A' * 3000, 1500, 'A', 'T')
        self.assertEqual(reason, '')
        self.assertEqual(len(ref), 1024)
        self.assertEqual([i for i, (a, b) in enumerate(zip(ref, alt)) if a != b], [512])
        self.assertEqual(q1.reverse_complement(alt)[511], 'A')
        with self.assertRaisesRegex(ValueError, 'REF mismatch'):
            q1.mutate_context('A' * 3000, 1500, 'C', 'T')



    def test_truncated_and_ambiguous_windows(self):
        self.assertEqual(q1.mutate_context('A' * 3000, 1, 'A', 'T')[2], 'truncated_window')
        chromosome = 'A' * 1400 + 'N' + 'A' * 1600
        self.assertEqual(q1.mutate_context(chromosome, 1500, 'A', 'T')[2], 'non_ACGT_window')



class FrozenDataTests(unittest.TestCase):
    def test_protocol_rejects_changed_data_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {'train': root / 'clinvar-train-pilot.vcf', 'validation': root / 'clinvar-test-pilot.vcf'}
            with patch.object(q1, 'OUTPUT', root), patch.object(q1, 'VCF_FILES', files):
                path = root / 'split_manifest.csv'
                path.write_text('variant_key,split\nv1,train\n')
                for output in files.values():
                    output.write_text('fixture\n')
                q1.write_json(root / 'protocol.json', {
                    'config': q1.protocol_config(), 'implementation_sha256': q1.digest_file(q1.__file__),
                    'artifacts': {path.name: q1.digest_file(path)},
                    'vcf_exports': {output.name: q1.digest_file(output) for output in files.values()},
                })
                q1.verify_protocol()
                with patch.object(q1, 'SEED', 43), self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                    q1.verify_protocol()
                files['validation'].write_text('changed variant\n')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    q1.verify_protocol()
                files['validation'].write_text('fixture\n')
                path.write_text('variant_key,split\nv1,validation\n')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    q1.verify_protocol()

    def test_data_module_imports_without_gpu_stack(self):
        import subprocess
        import sys
        subprocess.run([sys.executable, '-c',
                        "from notebooks.src import q1; import sys; assert not ({'torch', 'evo2', 'vortex'} & sys.modules.keys())"],
                       cwd=q1.ROOT, check=True, capture_output=True)


class VCFExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.files = {'train': self.root / 'clinvar-train-pilot.vcf',
                      'validation': self.root / 'clinvar-test-pilot.vcf'}
        self.source = self.root / 'clinvar.vcf'
        self.header = ('##fileformat=VCFv4.1\n##source=ClinVar\n##reference=GRCh38\n'
                       '##fileDate=2026-09-05\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        self.records = [f'1\t{pos}\t{i+100}\tA\tC\t.\t.\t'
                        f'ALLELEID={i+200};CLNSIG={label};GENEINFO=G{i}:{i+1};'
                        'CLNREVSTAT=reviewed_by_expert_panel\n'
                        for i, (pos, label) in enumerate(zip([2000, 10000, 20000, 40000],
                                                            ['Benign', 'Pathogenic'] * 2))]
        self.source.write_text(self.header + ''.join(self.records))
        self.pilot = manifest([2000, 10000, 20000, 40000]).assign(split=['train'] * 2 + ['validation'] * 2)
        self.pilot = self.pilot.iloc[[1, 0, 3, 2]].reset_index(drop=True)
        self.addCleanup(patch.stopall)
        patch.object(q1, 'VCF_FILES', self.files).start()
        patch.object(q1, 'OUTPUT', self.root).start()

    def export(self):
        return q1.export_vcfs(self.pilot, self.source, q1.digest_file(self.source))

    def test_preserves_headers_records_membership_and_repeated_export(self):
        hashes = self.export()
        self.assertEqual(self.files['train'].read_text(), self.header + ''.join(self.records[:2]))
        self.assertEqual(self.files['validation'].read_text(), self.header + ''.join(self.records[2:]))
        self.assertEqual(self.export(), hashes)
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_missing_and_duplicate_variants_fail_without_publishing(self):
        for rows, message in [(self.records[:3], 'exact split manifest'),
                              (self.records + self.records[:1], 'Duplicate')]:
            with self.subTest(message=message):
                self.source.write_text(self.header + ''.join(rows))
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                self.assertFalse(any(path.exists() for path in self.files.values()))
                self.assertFalse(list(self.root.glob('*.partial')))

    def test_source_checksum_failure_does_not_publish(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            q1.export_vcfs(self.pilot, self.source, 'wrong-checksum')
        self.assertFalse(any(path.exists() for path in self.files.values()))

    def test_changed_export_is_not_overwritten(self):
        self.export()
        self.files['validation'].write_text('corrupted\n')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.export()
        self.assertEqual(self.files['validation'].read_text(), 'corrupted\n')

    def test_vcf_labels_align_to_manifest_and_reject_wrong_membership(self):
        hashes = self.export()
        self.pilot.to_csv(self.root / 'split_manifest.csv', index=False)
        q1.write_json(self.root / 'protocol.json', {
            'config': q1.protocol_config(), 'implementation_sha256': q1.digest_file(q1.__file__),
            'artifacts': {'split_manifest.csv': q1.digest_file(self.root / 'split_manifest.csv')},
            'vcf_exports': hashes,
        })
        for split in q1.SPLITS:
            keys = self.pilot.loc[self.pilot.split.eq(split), 'variant_key'].tolist()
            np.testing.assert_equal(q1.load_partition_labels(split, keys), [1, 0])
            with self.assertRaisesRegex(ValueError, 'membership|order'):
                q1.load_partition_labels(split, keys[::-1])
        with self.assertRaisesRegex(ValueError, 'Unknown split'):
            q1.load_partition_labels('test', [])


if __name__ == '__main__':
    unittest.main()
