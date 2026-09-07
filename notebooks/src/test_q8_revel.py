"""REVEL coordinate, transcript aggregation and shared evaluation checks."""

import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from . import q8_baseline as baseline, q8_revel as revel


class RevelTests(unittest.TestCase):
    def lookup(self, rows, keys, header=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'revel.zip'
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(revel.COLUMNS if header is None else header)
            writer.writerows(rows)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(revel.MEMBER, text.getvalue())
                archive.writestr('__MACOSX/._revel.csv', 'ignored resource fork')
            return revel.lookup_scores(path, keys)

    def test_grch38_exact_alleles_transcript_maximum_and_missing_mapping(self):
        rows = [
            ['chr1', 999, 10, 'A', 'C', 'K', 'Q', .2, 'T1;T2'],
            ['1', 999, 10, 'A', 'C', 'N', 'H', .8, 'T2;T3'],
            ['1', 999, 10, 'A', 'G', 'K', 'R', .99, 'T4'],
            ['1', 999, 11, 'C', 'T', 'A', 'V', .9, 'T5'],
            ['1', 12, '.', 'A', 'C', 'K', 'Q', 1, 'T6'],
        ]
        keys = ['GRCh38:1:10:A:C', 'GRCh38:1:11:A:T', 'GRCh38:1:12:A:C', 'GRCh38:1:999:A:C']
        scores, annotations, scan = self.lookup(rows, keys)
        self.assertEqual(scores.variant_key.tolist(), keys)
        self.assertEqual(scores.revel.iloc[0], .8)
        self.assertAlmostEqual(scores.score_range.iloc[0], .6)
        self.assertEqual(scores.n_annotations.tolist(), [2, 0, 0, 0])
        self.assertEqual(scores.n_transcripts.tolist(), [3, 0, 0, 0])
        self.assertTrue(scores.revel.iloc[1:].isna().all())
        self.assertTrue(scores.status.iloc[1:].eq('no_exact_allele_match').all())
        self.assertEqual(annotations.Ensembl_transcriptid.tolist(), ['T1;T2', 'T2;T3'])
        self.assertEqual(scan['rows_without_grch38_position'], 1)
        self.assertEqual(scan['source_rows_scanned'], 5)
        self.assertNotIn('label', scores)
        self.assertNotIn('label', annotations)

    def test_empty_lookup_preserves_unscored_keys(self):
        scores, annotations, _ = self.lookup([], ['GRCh38:1:10:A:C'])
        self.assertTrue(annotations.empty)
        self.assertTrue(scores.revel.isna().all())
        self.assertEqual(scores.n_transcripts.iloc[0], 0)

    def test_invalid_keys_columns_positions_and_scores_fail(self):
        for keys in [['GRCh38:1:10:A:C'] * 2, ['GRCh37:1:10:A:C'], ['GRCh38:1:10:A:AC']]:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.lookup([], keys)
        with self.assertRaisesRegex(ValueError, 'columns'):
            self.lookup([], [], header=revel.COLUMNS[:-1])
        for pos, score in [('zero', .5), (0, .5), (10, 'nan'), (10, 'inf'), (10, 1.1), (10, -.1)]:
            with self.subTest(pos=pos, score=score), self.assertRaises(ValueError):
                self.lookup([['1', 100, pos, 'A', 'C', 'K', 'Q', score, 'T1']], ['GRCh38:1:10:A:C'])

    def test_revel_column_is_used_for_coverage_labels_and_bootstrap(self):
        pilot = pd.DataFrame({'variant_key': ['a', 'b', 'c', 'd', 'e'],
                              'split': ['train', 'train', 'validation', 'validation', 'validation'],
                              'component': ['t', 't', 'v', 'v', 'w']})
        scores = pd.DataFrame({'variant_key': pilot.variant_key, 'revel': [.2, .8, .1, .9, np.nan]})
        def labels(split, keys):
            self.assertEqual(keys, ['a', 'b'] if split == 'train' else ['c', 'd', 'e'])
            return np.array([0, 1] if split == 'train' else [0, 1, 1])
        with patch.object(baseline.q1, 'load_partition_labels', side_effect=labels):
            table, coverage, report = baseline.evaluate(pilot, scores, score_column='revel')
        self.assertEqual(table.label.tolist(), [0, 1, 0, 1, 1])
        self.assertEqual(report['covered'], 2)
        self.assertEqual(report['total'], 3)
        self.assertEqual(report['metrics']['auroc'], {'value': 1., 'ci95': [1., 1.]})
        self.assertEqual(len(coverage), 6)
        self.assertNotIn('am_pathogenicity', table)


if __name__ == '__main__':
    unittest.main()
