"""Full-cohort scoring must retain missing variants and reject pilot/stale exports."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from notebooks.src import comparison as c, q8_full as full


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.validation = pd.DataFrame({'variant_key': ['v1', 'v2', 'v3', 'v4'],
                                       'split': ['validation'] * 4, 'component': ['a', 'a', 'b', 'b']})
        self.scores = pd.DataFrame({'variant_key': self.validation.variant_key,
                                    'score': [.1, .9, np.nan, .8]})

    def test_full_labels_are_loaded_after_complete_score_alignment(self):
        with patch.object(full.q1_full, 'load_partition_labels', return_value=np.array([0, 1, 0, 1])) as labels, \
                patch.dict(full.POLICY, {'bootstrap_repetitions': 20}):
            table, coverage, metrics = full.evaluate(self.validation, self.scores, 'score')
        labels.assert_called_once_with('validation', ['v1', 'v2', 'v3', 'v4'])
        self.assertEqual(table.variant_key.tolist(), self.validation.variant_key.tolist())
        self.assertTrue(np.isnan(table.score.iloc[2]))
        self.assertEqual((metrics['total'], metrics['covered'], metrics['missing']), (4, 3, 1))
        self.assertEqual(metrics['metrics']['auroc']['value'], 1.)
        self.assertEqual(coverage['class'].tolist(), ['all', 'benign', 'pathogenic'])

    def test_partial_reordered_and_duplicate_predictions_fail_before_labels(self):
        for scores in [self.scores.iloc[:3], self.scores.iloc[::-1],
                       self.scores.assign(variant_key=['v1', 'v1', 'v3', 'v4'])]:
            with self.subTest(keys=scores.variant_key.tolist()), \
                    patch.object(full.q1_full, 'load_partition_labels') as labels:
                with self.assertRaisesRegex(ValueError, 'every full validation key'):
                    full.evaluate(self.validation, scores, 'score')
                labels.assert_not_called()

    def test_training_partition_is_rejected_before_labels(self):
        with patch.object(full.q1_full, 'load_partition_labels') as labels:
            with self.assertRaises(ValueError):
                full.evaluate(self.validation.assign(split='train'), self.scores, 'score')
            labels.assert_not_called()


class FullExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'notebooks/results/q8/full'
        self.directory.mkdir(parents=True)
        self.frame = pd.DataFrame({'variant_key': ['v1', 'v2'], 'split': ['validation'] * 2,
                                   'component': ['a', 'b'], 'label': [0, 1], 'am_pathogenicity': [.1, np.nan]})
        self.context = {'scope': 'full', 'protocol_sha256': 'full-cohort', 'vcf_exports': {'full.vcf': 'hash'},
                        'validation': self.frame[['variant_key', 'component', 'label']]}
        self.source = self.root / 'notebooks/src/q8_full.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# fixture\n')
        for name in ['protocol.json', 'leakage_checks.json', 'baseline_protocol.json', 'validation_metrics.json']:
            (self.directory / name).write_text('{}\n')
        self.frame.to_csv(self.directory / 'validation_predictions.csv', index=False)
        self.provenance = {
            'scope': 'full_validation', 'tool': 'AlphaMissense',
            'q1_protocol_sha256': self.context['protocol_sha256'], 'q1_vcf_sha256': self.context['vcf_exports'],
            'sources': {'notebooks/src/q8_full.py': c.digest(self.source)},
            'full_q8_protocol_sha256': c.digest(self.directory / 'protocol.json'),
            'local_split_checks_sha256': c.digest(self.directory / 'leakage_checks.json'),
            'artifacts': {name: c.digest(self.directory / name) for name in
                          ['baseline_protocol.json', 'validation_predictions.csv', 'validation_metrics.json']}}
        self.save()

    def save(self):
        (self.directory / 'baseline_provenance.json').write_text(json.dumps(self.provenance))

    def test_valid_full_export_retains_missing_values_and_is_watched(self):
        scores = c.load_q8(self.root, self.context)['AlphaMissense']
        np.testing.assert_equal(scores, [.1, np.nan])
        self.assertIn(self.directory / 'validation_predictions.csv', c.source_paths(self.root))

    def test_pilot_cohort_and_changed_source_are_rejected(self):
        self.provenance['q1_protocol_sha256'] = 'pilot-cohort'
        self.save()
        with self.assertRaisesRegex(ValueError, 'cohort is stale'):
            c.load_q8(self.root, self.context)
        self.provenance['q1_protocol_sha256'] = self.context['protocol_sha256']
        self.save()
        self.source.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            c.load_q8(self.root, self.context)

    def test_partial_full_export_does_not_fall_back_to_pilot(self):
        (self.directory / 'baseline_provenance.json').unlink()
        with self.assertRaises(FileNotFoundError):
            c.load_q8(self.root, self.context)

    def test_collector_accepts_full_results_without_any_pilot_outputs(self):
        catalog = c.read_json(c.ROOT / 'notebooks/src/q8_catalog.json')
        (self.root / 'notebooks/src/q8_catalog.json').write_text(json.dumps(catalog))
        (self.root / 'notebooks/results/q1/full').mkdir(parents=True)
        context = {**self.context, 'config': {'clinvar_date': '2026-07-06'}}
        self.frame['am_pathogenicity'] = [.1, .9]
        self.frame.to_csv(self.directory / 'validation_predictions.csv', index=False)
        self.provenance['artifacts']['validation_predictions.csv'] = c.digest(self.directory / 'validation_predictions.csv')
        self.save()
        with patch.object(c, 'load_context', return_value=context):
            result = c.collect(self.root, repetitions=20)
        row = next(row for row in result['methods'] if row['id'] == 'q8:AlphaMissense')
        self.assertEqual(row['status'], 'Available')
        self.assertEqual(row['covered'], 2)
        self.assertEqual(row['metrics']['auroc']['value'], 1.)

    def test_wrong_labels_membership_and_score_range_are_rejected(self):
        for frame in [self.frame.assign(label=[1, 0]), self.frame.iloc[:1],
                      self.frame.assign(am_pathogenicity=[1.1, .5])]:
            with self.subTest(frame=frame.to_dict()):
                frame.to_csv(self.directory / 'validation_predictions.csv', index=False)
                self.provenance['artifacts']['validation_predictions.csv'] = c.digest(self.directory / 'validation_predictions.csv')
                self.save()
                with self.assertRaises(ValueError):
                    c.load_q8(self.root, self.context)


if __name__ == '__main__':
    unittest.main()
