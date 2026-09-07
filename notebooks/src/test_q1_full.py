"""Adversarial checks for extending the frozen pilot to every eligible variant."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from . import q1, q1_full, test_q1


class FullCoverageTests(unittest.TestCase):
    def setUp(self):
        self.full = test_q1.manifest([2000, 10000, 20000])
        self.full['MC'] = 'SO:0001583|missense_variant'
        self.full['component'] = self.full.variant_key
        self.full['split'] = ['train', 'validation', 'train']
        rows = []
        for i, key in enumerate(self.full.variant_key):
            ref = 'A' * (1023 - i) + 'G' * (i + 1)
            alt = ref[:512] + 'C' + ref[513:]
            rows.append([key, ref, alt])
        self.dna = pd.DataFrame(rows, columns=q1.DNA_COLUMNS)
        self.retained = self.full.assign(ref_context_hash=self.dna.ref_sequence.map(q1.context_hash),
                                         alt_context_hash=self.dna.alt_sequence.map(q1.context_hash))
        self.exclusions = pd.DataFrame(columns=['variant_key', 'reason'])
        self.pilot = self.retained.iloc[:2].copy()
        self.addCleanup(patch.stopall)
        patch.object(q1, 'FULL_ASSIGNMENTS_SHA256', q1.fingerprint(sorted(zip(self.full.variant_key, self.full.split)))).start()
        patch.object(q1, 'read_csv', return_value=self.pilot).start()

    def test_all_candidates_inherit_pilot_and_group_assignments(self):
        self.assertTrue(all(q1_full.audit(self.full, self.retained, self.dna, self.exclusions).values()))

    def test_silent_downsampling_fails(self):
        with self.assertRaisesRegex(AssertionError, 'every candidate'):
            q1_full.audit(self.full, self.retained.iloc[:2], self.dna.iloc[:2], self.exclusions)

    def test_only_explicit_sequence_quality_exclusions_allowed(self):
        excluded = pd.DataFrame([[self.full.variant_key[2], 'non_ACGT_window']], columns=self.exclusions.columns)
        self.assertTrue(all(q1_full.audit(self.full, self.retained.iloc[:2], self.dna.iloc[:2], excluded).values()))
        excluded.loc[0, 'reason'] = 'rebalance_classes'
        with self.assertRaisesRegex(AssertionError, 'every candidate'):
            q1_full.audit(self.full, self.retained.iloc[:2], self.dna.iloc[:2], excluded)

    def test_lost_or_reassigned_pilot_is_rejected(self):
        self.pilot.loc[0, 'split'] = 'validation'
        with self.assertRaisesRegex(AssertionError, 'Pilot split changed'):
            q1_full.audit(self.full, self.retained, self.dna, self.exclusions)
        excluded = pd.DataFrame([[self.full.variant_key[0], 'truncated_window']], columns=self.exclusions.columns)
        with self.assertRaisesRegex(AssertionError, 'pilot variant was lost'):
            q1_full.audit(self.full, self.retained.iloc[1:], self.dna.iloc[1:], excluded)

    def test_new_identical_context_bridge_cannot_move_frozen_groups(self):
        dna = self.dna.copy()
        dna.loc[1, 'ref_sequence'] = q1.reverse_complement(dna.loc[0, 'alt_sequence'])
        with self.assertRaisesRegex(AssertionError, 'frozen train and validation'):
            q1_full.expand_groups(self.full, dna)
        self.assertEqual(self.full.split.tolist(), ['train', 'validation', 'train'])

    def test_same_split_contexts_merge_for_grouped_evaluation(self):
        dna = self.dna.copy()
        dna.loc[2, 'ref_sequence'] = dna.loc[0, 'alt_sequence']
        expanded = q1_full.expand_groups(self.full, dna)
        self.assertEqual(expanded.component[0], expanded.component[2])
        self.assertEqual(expanded.split.tolist(), self.full.split.tolist())
        reordered = q1_full.expand_groups(self.full.iloc[::-1].reset_index(drop=True), dna.iloc[::-1])
        self.assertEqual(dict(zip(expanded.variant_key, expanded.component)),
                         dict(zip(reordered.variant_key, reordered.component)))


class FullExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {'train': self.root / 'clinvar-train.vcf', 'validation': self.root / 'clinvar-test.vcf'}
        self.source = self.root / 'source.vcf'
        self.header = ('##fileformat=VCFv4.1\n##source=ClinVar\n##reference=GRCh38\n'
                       '##fileDate=2026-09-05\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        self.records = [f'1\t{pos}\t{i+100}\tA\tC\t.\t.\t'
                        f'ALLELEID={i+200};CLNSIG={label};GENEINFO=G{i}:{i+1};'
                        'CLNREVSTAT=reviewed_by_expert_panel;MC=SO:0001583|missense_variant\n'
                        for i, (pos, label) in enumerate(zip([2000, 10000, 20000, 40000], ['Benign', 'Pathogenic'] * 2))]
        self.source.write_text(self.header + ''.join(self.records))
        self.manifest = test_q1.manifest([2000, 10000, 20000, 40000]).assign(split=['train'] * 2 + ['validation'] * 2)
        self.manifest = self.manifest.iloc[[1, 0, 3, 2]].reset_index(drop=True)
        self.addCleanup(patch.stopall)
        patch.object(q1_full, 'VCF_FILES', self.files).start()

    def export(self):
        return q1_full.export_vcfs(self.manifest, self.source, q1.digest_file(self.source))

    def test_records_preserved_and_labels_aligned_without_resampling(self):
        hashes, labels = self.export()
        self.assertEqual(self.files['train'].read_text(), self.header + ''.join(self.records[:2]))
        self.assertEqual(self.files['validation'].read_text(), self.header + ''.join(self.records[2:]))
        self.assertEqual(labels['train'].label.tolist(), [1, 0])
        self.assertEqual(labels['validation'].label.tolist(), [1, 0])
        self.assertEqual(self.export()[0], hashes)

    def test_corrupt_frozen_export_is_never_overwritten(self):
        self.export()
        self.files['validation'].write_text('corrupt\n')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.export()
        self.assertEqual(self.files['validation'].read_text(), 'corrupt\n')

    def test_source_and_membership_errors_publish_neither_partition(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            q1_full.export_vcfs(self.manifest, self.source, 'wrong')
        for rows, error in [(self.records[:3], 'exact split manifest'), (self.records + self.records[:1], 'Duplicate')]:
            self.source.write_text(self.header + ''.join(rows))
            with self.assertRaisesRegex(ValueError, error):
                self.export()
            self.assertFalse(any(path.exists() for path in self.files.values()))
            self.assertFalse(list(self.root.glob('*.partial')))

    def test_original_quality_and_label_filters_rechecked_before_publication(self):
        for old, new in [('SO:0001583|missense_variant', 'SO:0001819|synonymous_variant'),
                         ('reviewed_by_expert_panel', 'criteria_provided,_single_submitter'),
                         ('CLNSIG=Benign', 'CLNSIG=Uncertain_significance')]:
            self.source.write_text((self.header + ''.join(self.records)).replace(old, new, 1))
            with self.assertRaisesRegex((ValueError, AssertionError), 'missense|filters'):
                self.export()
            self.assertFalse(any(path.exists() for path in self.files.values()))


if __name__ == '__main__':
    unittest.main()
