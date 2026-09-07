"""Small adversarial VCF fixtures for cohort and grouping safety."""

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from notebooks import q0


HEADER = ('##fileformat=VCFv4.1\n##fileDate=2026-09-05\n##source=ClinVar\n'
          '##reference=GRCh38\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')


def record(pos=100, identifier='1', ref='A', alt='C', chrom='1', **annotations):
    info = {'ALLELEID': identifier, 'CLNSIG': 'Pathogenic',
            'CLNREVSTAT': 'criteria_provided,_multiple_submitters,_no_conflicts',
            'GENEINFO': 'GENE1:1', 'CLNVC': 'single_nucleotide_variant'}
    info.update(annotations)
    encoded = ';'.join(f'{key}={value}' for key, value in info.items() if value is not None)
    return f'{chrom}\t{pos}\t{identifier}\t{ref}\t{alt}\t.\t.\t{encoded}\n'


class ClinVarTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def load(self, *records):
        path = self.root / 'input.vcf'
        path.write_text(HEADER + ''.join(records))
        return q0.read_clinvar(path)[0]

    def test_labels_are_exact_and_compound_labels_are_not_split(self):
        labels = list(q0.LABELS) + ['Uncertain_significance', 'Pathogenic|risk_factor',
                                   'Benign|Pathogenic', 'Likely_pathogenic,_low_penetrance', None]
        frame = self.load(*(record(i + 1, str(i + 1), CLNSIG=label) for i, label in enumerate(labels)))
        cohort, funnel = q0.propose_cohort(frame, q0.audit_data(frame))
        self.assertEqual(len(cohort), 6)
        self.assertEqual(cohort.label.value_counts().to_dict(), {'benign': 3, 'pathogenic': 3})
        self.assertEqual(funnel.removed.sum() + len(cohort), len(frame))

    def test_review_mapping_and_unknown_status(self):
        statuses = list(q0.REVIEW_STARS) + ['new_unknown_status', None]
        frame = self.load(*(record(i + 1, str(i + 1), CLNREVSTAT=status) for i, status in enumerate(statuses)))
        cohort, _ = q0.propose_cohort(frame, q0.audit_data(frame))
        self.assertEqual(set(cohort.stars), {2, 3, 4})
        self.assertEqual(len(cohort), 3)
        self.assertEqual(frame.stars.eq(-1).sum(), 2)

    def test_missing_and_multiple_gene_identifiers(self):
        frame = self.load(record(GENEINFO=None), record(200, '2', GENEINFO='A:2|B:1|C:2'),
                          record(300, '3', GENEINFO='A:0|B:abc'))
        cohort, _ = q0.propose_cohort(frame, q0.audit_data(frame))
        self.assertEqual(cohort.gene_ids.tolist(), ['1|2'])
        self.assertEqual(cohort.GENEINFO.iloc[0], 'A:2|B:1|C:2')

    def test_conflict_and_somatic_fields_do_not_supply_germline_labels(self):
        frame = self.load(record(CLNSIGCONF='Benign(1)'),
                          record(200, '2', CLNSIG=None, ONC='Oncogenic', SCI='Tier_I'),
                          record(300, '3', CLNREVSTAT='criteria_provided,_conflicting_classifications'))
        cohort, _ = q0.propose_cohort(frame, q0.audit_data(frame))
        self.assertTrue(cohort.empty)

    def test_inconsistent_duplicate_keys_are_excluded_before_filtering(self):
        frame = self.load(record(), record(identifier='2', CLNSIG='Benign'),
                          record(200, '3'), record(200, '4', MC='SO:0001583|missense_variant'))
        audit = q0.audit_data(frame)
        cohort, _ = q0.propose_cohort(frame, audit)
        self.assertEqual(len(audit['inconsistent_keys']), 2)
        self.assertTrue(cohort.empty)

    def test_consistent_duplicates_preserve_source_ids_and_alt_alleles(self):
        first = record(CLNSIGSCV='SCV1')
        frame = self.load(first, first, record(identifier='2', CLNSIGSCV='SCV2'), record(identifier='3', alt='G'))
        audit = q0.audit_data(frame)
        cohort, _ = q0.propose_cohort(frame, audit)
        self.assertEqual(len(cohort), 2)
        self.assertEqual(audit['metrics']['Exact duplicate rows beyond first'], 1)
        self.assertEqual(audit['metrics']['Loci with multiple distinct alleles'], 1)
        merged = cohort.loc[cohort.alt.eq('C')].iloc[0]
        self.assertEqual(merged.variation_id, '1|2')
        self.assertEqual(merged.CLNSIGSCV, 'SCV1|SCV2')
        self.assertEqual(merged.ALLELEID, '1|2')

    def test_non_snv_and_noncanonical_chromosome_exclusions(self):
        frame = self.load(record(alt='C,G'), record(200, '2', ref='AC', alt='A'),
                          record(300, '3', alt='<DEL>'), record(400, '4', alt='N'),
                          record(500, '5', alt='A'), record(600, '6', chrom='MT'),
                          record(700, '7', chrom='X'), record(800, '8', chrom='Y'))
        cohort, _ = q0.propose_cohort(frame, q0.audit_data(frame))
        self.assertEqual(set(cohort.chrom), {'X', 'Y'})

    def test_context_boundaries_transitivity_and_chromosomes(self):
        frame = self.load(record(1), record(10, '2'), record(19, '3'), record(29, '4'),
                          record(29, '5', alt='G'), record(1, '6', chrom='2'))
        loci = q0.context_groups(frame, window_bp=10)
        self.assertEqual(loci.context_group.tolist(), ['1:1', '1:1', '1:1', '1:29', '2:1'])
        self.assertEqual(len(loci), 5)
        with self.assertRaises(ValueError):
            q0.context_groups(frame, 9)

    def test_counts_do_not_double_count_same_annotation_within_a_record(self):
        counts = q0.annotation_counts(pd.Series(['A|B|A', 'A', '']), '|')
        self.assertEqual(counts.to_dict(), {'A': 2, 'B': 1})

    def test_invalid_input_and_checksum_fail(self):
        for line in ['1\t1\n', record(pos=0), record()[:-1] + ';CLNSIG=Benign\n']:
            with self.assertRaises(ValueError):
                q0.parse_record(line, 1)
        self.load(record())
        path = self.root / 'input.vcf'
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            q0.read_clinvar(path, expected_sha256='incorrect')
        with self.assertRaisesRegex(ValueError, 'assembly'):
            q0.read_clinvar(path, expected_reference='GRCh37')
        with self.assertRaisesRegex(ValueError, 'fileDate'):
            q0.read_clinvar(path, expected_date='2000-01-01')

    def test_input_decompression_and_missing_archive(self):
        archive = self.root / 'input.gz'
        with gzip.open(archive, 'wb') as stream:
            stream.write((HEADER + record()).encode())
        target = self.root / 'unpacked.vcf'
        q0.ensure_input(target, archive)
        self.assertEqual(target.read_text(), HEADER + record())
        with self.assertRaises(FileNotFoundError):
            q0.ensure_input(self.root / 'absent.vcf', self.root / 'absent.gz')

    def test_export_invariants_and_determinism(self):
        frame = self.load(record(), record(identifier='2', alt='G'), record(200, '3', GENEINFO='A:1|B:2'))
        audit = q0.audit_data(frame)
        cohort, funnel = q0.propose_cohort(frame, audit)
        # The notebook shows these checks as JSON in a disclosure widget.
        checks = q0.assert_cohort(cohort, audit['inconsistent_keys'])
        self.assertTrue(all(json.loads(json.dumps(checks)).values()))
        provenance = {'reference': 'GRCh38', 'sha256': 'fixture'}
        output = self.root / 'output'
        q0.export_results(frame, cohort, funnel, audit, provenance, output)
        first_hash = hashlib.sha256((output / 'cohort.csv.gz').read_bytes()).hexdigest()
        q0.export_results(frame, cohort, funnel, audit, provenance, output)
        self.assertEqual(first_hash, hashlib.sha256((output / 'cohort.csv.gz').read_bytes()).hexdigest())
        exported = pd.read_csv(output / 'cohort.csv.gz')
        self.assertEqual(len(exported), 3)
        self.assertEqual(exported.context_group.nunique(), 1)
        self.assertNotIn('split', exported.columns)
        corrupt = pd.concat([cohort, cohort.iloc[:1]])
        with self.assertRaisesRegex(AssertionError, 'unique variant keys'):
            q0.assert_cohort(corrupt, audit['inconsistent_keys'])


if __name__ == '__main__':
    unittest.main()
