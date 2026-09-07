"""Adversarial checks for external-score identity, allele matching and evaluation."""

import gzip
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import q8, q8_baseline as baseline
except ImportError:
    from notebooks.src import q8, q8_baseline as baseline


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scores.gz'
        self.data = gzip.compress(b'fixture\n')
        self.source = {'url': 'https://example.invalid/scores.gz',
                       'bytes': len(self.data), 'md5': hashlib.md5(self.data).hexdigest()}

    def test_download_then_verified_cache_requires_no_network(self):
        with patch.object(baseline.urllib.request, 'urlopen', return_value=io.BytesIO(self.data)) as request:
            baseline.ensure_scores(self.path, self.source)
            baseline.ensure_scores(self.path, self.source)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.path.with_name('scores.gz.partial').exists())

    def test_bad_checksum_does_not_publish_download(self):
        damaged = self.data[:-1] + bytes([self.data[-1] ^ 1])
        with patch.object(baseline.urllib.request, 'urlopen', return_value=io.BytesIO(damaged)):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                baseline.ensure_scores(self.path, self.source)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name('scores.gz.partial').exists())

    def test_existing_corrupt_archive_is_preserved_and_rejected(self):
        self.path.write_bytes(b'corrupt')
        with patch.object(baseline.urllib.request, 'urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                baseline.ensure_scores(self.path, self.source)
        request.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'corrupt')

    def test_interrupted_download_cleans_partial_file(self):
        response = io.BytesIO(self.data)
        with patch.object(baseline.urllib.request, 'urlopen', return_value=response), \
                patch.object(response, 'read', side_effect=[b'partial', OSError('disconnected')]):
            with self.assertRaisesRegex(OSError, 'disconnected'):
                baseline.ensure_scores(self.path, self.source)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_name('scores.gz.partial').exists())


class LookupTests(unittest.TestCase):
    def lookup(self, rows, keys, header=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.gz'
            with gzip.open(path, 'wt') as out:
                out.write('# original license\n' + (header or '\t'.join(baseline.COLUMNS)) + '\n')
                for row in rows:
                    out.write('\t'.join(map(str, row)) + '\n')
            return baseline.lookup_scores(path, keys)

    def test_exact_alleles_chr_prefix_and_transcript_maximum(self):
        rows = [
            ['chr1', 10, 'A', 'C', 'hg38', 'P1', 'T1', 'A1C', .2, 'likely_benign'],
            ['1', 10, 'A', 'C', 'hg38', 'P2', 'T2', 'A1C', .8, 'likely_pathogenic'],
            ['chr1', 10, 'A', 'G', 'hg38', 'P1', 'T1', 'A1G', .99, 'likely_pathogenic'],
            ['chr1', 11, 'C', 'T', 'hg38', 'P1', 'T1', 'C2T', .7, 'likely_pathogenic'],
        ]
        keys = ['GRCh38:1:11:A:T', 'GRCh38:1:10:A:C', 'GRCh38:1:12:A:C']
        scores, annotations, scan = self.lookup(rows, keys)
        self.assertEqual(scores.variant_key.tolist(), keys)
        self.assertTrue(np.isnan(scores.am_pathogenicity.iloc[0]))  # REF mismatch
        self.assertEqual(scores.am_pathogenicity.iloc[1], .8)  # ALT G does not match
        self.assertEqual(scores.n_transcripts.tolist(), [0, 2, 0])
        self.assertAlmostEqual(scores.score_range.iloc[1], .6)
        self.assertEqual(scores.status.iloc[2], 'no_exact_allele_match')
        self.assertEqual(len(annotations), 2)
        self.assertEqual(scan['source_rows_scanned'], 4)
        self.assertEqual(scan['archive_header'], ['# original license'])
        self.assertNotIn('label', scores)

    def test_missing_keys_never_get_zero_scores(self):
        scores, annotations, _ = self.lookup([], ['GRCh38:1:10:A:C'])
        self.assertTrue(scores.am_pathogenicity.isna().all())
        self.assertTrue(annotations.empty)
        self.assertEqual(scores.n_annotations.iloc[0], 0)

    def test_duplicate_pilot_keys_fail(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.lookup([], ['GRCh38:1:10:A:C'] * 2)

    def test_wrong_assembly_schema_and_nonfinite_scores_fail(self):
        with self.assertRaisesRegex(ValueError, 'columns'):
            self.lookup([], [], header='#CHROM\tPOS')
        for genome, value in [('hg19', .5), ('hg38', float('nan')), ('hg38', 1.1)]:
            with self.subTest(genome=genome, value=value), self.assertRaises(ValueError):
                self.lookup([['chr1', 10, 'A', 'C', genome, 'P', 'T', 'A1C', value, 'ambiguous']],
                            ['GRCh38:1:10:A:C'])


class EvaluationTests(unittest.TestCase):
    def test_perfect_ranking_and_missing_coverage(self):
        validation = pd.DataFrame({'label': [0, 1, 0, 1, 1],
                                   'am_pathogenicity': [.1, .8, .2, .9, np.nan],
                                   'component': ['a', 'a', 'b', 'b', 'c']})
        report = baseline.metric_summary(validation, repetitions=100, seed=42)
        self.assertEqual((report['total'], report['covered'], report['missing']), (5, 4, 1))
        self.assertEqual(report['components'], 3)
        for metric in report['metrics'].values():
            self.assertEqual(metric, {'value': 1., 'ci95': [1., 1.]})
        self.assertEqual(report, baseline.metric_summary(validation, repetitions=100, seed=42))

    def test_single_component_members_stay_together(self):
        validation = pd.DataFrame({'label': [0, 1, 0, 1],
                                   'am_pathogenicity': [.1, .9, .8, .2],
                                   'component': ['a'] * 4})
        report = baseline.metric_summary(validation, repetitions=20)
        self.assertEqual(report['bootstrap_valid_replicates'], 20)
        self.assertEqual(report['metrics']['auroc']['ci95'], [.75, .75])

    def test_one_covered_class_has_no_misleading_metrics(self):
        validation = pd.DataFrame({'label': [0, 1], 'am_pathogenicity': [.1, np.nan],
                                   'component': ['a', 'b']})
        report = baseline.metric_summary(validation, repetitions=10)
        self.assertEqual(report['status'], 'both_classes_required')
        self.assertFalse(report['metrics'])

    def test_evaluation_aligns_labels_and_reports_each_partition(self):
        pilot = pd.DataFrame({'variant_key': ['a', 'b', 'c', 'd'],
                              'split': ['validation', 'train', 'train', 'validation'],
                              'component': ['v1', 't1', 't2', 'v2']})
        scores = pd.DataFrame({'variant_key': pilot.variant_key, 'am_pathogenicity': [.8, .2, .9, np.nan]})
        def labels(split, keys):
            self.assertEqual(keys, ['b', 'c'] if split == 'train' else ['a', 'd'])
            return np.array([0, 1] if split == 'train' else [1, 0])
        with patch.object(baseline.q1, 'load_partition_labels', side_effect=labels):
            table, coverage, report = baseline.evaluate(pilot, scores)
        self.assertEqual(table.label.tolist(), [1, 0, 1, 0])
        self.assertEqual(len(coverage), 6)
        self.assertEqual(report['total'], 2)
        self.assertEqual(report['covered'], 1)
        with self.assertRaisesRegex(ValueError, 'ordered pilot'):
            baseline.evaluate(pilot, scores.iloc[::-1])

    def test_catalog_only_contains_missense_specialists(self):
        survey = q8.load_survey()
        self.assertEqual(len(survey['tools']), 6)
        self.assertEqual(survey['shortlist'][0]['tools'], ['AlphaMissense'])
        self.assertTrue(all(t['targets'] == ['Missense impact'] for t in survey['tools']))


if __name__ == '__main__':
    unittest.main()
