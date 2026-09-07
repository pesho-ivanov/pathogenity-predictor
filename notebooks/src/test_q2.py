"""Adversarial checks for model isolation, frozen caches and evaluation provenance."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import q2
except ImportError:
    from notebooks.src import q2


class SequenceTests(unittest.TestCase):
    def test_baseline_uses_both_correct_strand_centers(self):
        ref = 'A' * 512 + 'C' + 'G' * 511
        alt = ref[:512] + 'T' + ref[513:]
        values = q2.sequence_baseline(ref, alt)
        self.assertEqual(len(values), 73)
        np.testing.assert_equal(values[:4], [0, .5, .5, 0])
        np.testing.assert_equal(values[4:8], [.5, 0, 0, .5])
        self.assertEqual(values[8 + q2.TRIPLETS.index('ACG')], .5)
        self.assertEqual(values[8 + q2.TRIPLETS.index('CGT')], .5)
        self.assertEqual(values[-1], .5)



    def test_feature_allowlist_rejects_annotation(self):
        dna = pd.DataFrame([['key', 'A', 'C']], columns=q2.DNA_COLUMNS)
        with patch.object(q2, 'verify_protocol'), patch.object(q2, 'read_csv', return_value=dna):
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                q2.extract_features(dna.assign(CLNSIG='Pathogenic'))



class ModelBoundaryTests(unittest.TestCase):
    def test_scaler_and_coefficients_fit_only_training(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(40, 3))
        target = (train[:, 0] > 0).astype(int)
        validation = rng.normal(size=(12, 3)) + 1000
        y_val = np.tile([0, 1], 6)
        a, scores_a = q2.fit_candidates(train, target, validation, y_val, cs=[.1])
        b, scores_b = q2.fit_candidates(train, target, validation, 1 - y_val, cs=[.1])
        np.testing.assert_allclose(a['mean'], train.mean(axis=0))
        np.testing.assert_equal(a['coef'], b['coef'])
        self.assertAlmostEqual(scores_a[0]['validation_auroc'] + scores_b[0]['validation_auroc'], 1)

    def test_validation_selection_tie_uses_smaller_c(self):
        train = np.array([[-2.], [-1.], [1.], [2.]])
        labels = np.array([0, 0, 1, 1])
        model, scores = q2.fit_candidates(train, labels, train, labels, cs=[10., .01, 1.])
        self.assertEqual(model['C'], .01)
        self.assertTrue(all(score['validation_auroc'] == 1 for score in scores))

    def test_third_split_rejected_and_changed_experiment_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(q2, 'OUTPUT', Path(directory)), patch.object(q2.q1, 'OUTPUT', Path(directory)):
            with self.assertRaisesRegex(ValueError, 'Unknown split'):
                q2.load_labels('test', [])
            q2.write_json(q2.OUTPUT / 'validation_report_started.json', {'identity': {'code': 'original'}}, frozen=True)
            q2.assert_evaluation_identity({'code': 'original'})
            with self.assertRaisesRegex(RuntimeError, 'experiment changed'):
                q2.assert_evaluation_identity({'code': 'tuned'})
            with self.assertRaisesRegex(RuntimeError, 'Frozen artifact'):
                q2.write_json(q2.OUTPUT / 'validation_report_started.json', {'identity': {'code': 'tuned'}}, frozen=True)

    def test_cache_keys_provenance_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunk.npz'
            values = {'keys': np.array(['a', 'b']), 'identity': np.array('frozen'),
                      'evo': np.zeros((2, q2.HIDDEN * 2)), 'sequence': np.zeros((2, 73)), 'zero_shot': np.zeros(2)}
            q2.save_npz(path, **values)
            q2.load_chunk(path, ['a', 'b'], 'frozen')
            with self.assertRaisesRegex(ValueError, 'provenance/key'):
                q2.load_chunk(path, ['b', 'a'], 'frozen')
            with self.assertRaisesRegex(ValueError, 'provenance/key'):
                q2.load_chunk(path, ['a', 'b'], 'different')
            values['evo'][0, 0] = np.nan
            q2.save_npz(path, **values)
            with self.assertRaisesRegex(ValueError, 'Invalid evo'):
                q2.load_chunk(path, ['a', 'b'], 'frozen')

    def test_paired_bootstrap_keeps_method_pairs_and_groups(self):
        labels = np.tile([0, 1], 8)
        score = labels.astype(float)
        result = q2.paired_bootstrap(labels, {'evo': score, 'zero_shot': score},
                                     np.repeat(np.arange(8), 2), repetitions=30)
        self.assertEqual(result['evo_minus_zero_shot_95ci'], [0, 0])
        self.assertEqual(result['auroc_95ci']['evo'], [1, 1])
        self.assertEqual(result['independent_components'], 8)

    def test_train_validation_workflow_reads_canonical_partitions_and_reuses_results(self):
        rng = np.random.default_rng(42)
        keys = np.array([f'v{i}' for i in range(80)])
        splits = ['train'] * 40 + ['validation'] * 40
        features = {'keys': keys, 'evo': rng.normal(size=(80, 3)),
                    'sequence': rng.normal(size=(80, 2)), 'zero_shot': rng.normal(size=80)}
        identity = {'synthetic': True}
        with tempfile.TemporaryDirectory() as directory, patch.object(q2, 'OUTPUT', Path(directory)), patch.object(q2.q1, 'OUTPUT', Path(directory) / 'q1_data'), \
                patch.object(q2, 'verify_protocol'), patch.object(q2, 'load_features', return_value=features), \
                patch.object(q2, 'experiment_identity', return_value=identity), \
                patch.object(q2.q0, 'details'), patch.object(q2.plt, 'show'), patch.object(q2, 'show_results'), \
                patch.object(q2.q1, 'load_partition_labels') as partition_loader:
            q2.q1.OUTPUT.mkdir()
            pd.DataFrame({'variant_key': keys, 'split': splits,
                          'component': [f'g{i // 2}' for i in range(80)]}).to_csv(q2.q1.OUTPUT / 'split_manifest.csv', index=False)
            for split in q2.SPLITS:
                split_keys = keys[np.array(splits) == split]
                pd.DataFrame({'variant_key': split_keys, 'label': np.arange(len(split_keys)) % 2}).to_csv(q2.q1.OUTPUT / f'{split}_labels.csv', index=False)
            q2.write_json(q2.OUTPUT / 'feature_manifest.json', {'fixture': True})
            def partition_labels(split, keys):
                frame = q2.read_csv(q2.q1.OUTPUT / f'{split}_labels.csv')
                self.assertEqual(frame.variant_key.tolist(), list(keys))
                return frame.label.to_numpy(dtype=int)
            partition_loader.side_effect = partition_labels
            with patch.object(q2, 'load_labels', wraps=q2.load_labels) as loader:
                selected = q2.select_models()
                self.assertEqual([call.args[0] for call in loader.call_args_list], ['train', 'validation'])
                self.assertFalse(selected['separate_test_set'])
                result = q2.report_validation()
                self.assertEqual(loader.call_args_list[-1].args[0], 'validation')
                self.assertEqual(result['validation_variants'], 40)
                calls = loader.call_count
                repeated = q2.report_validation()
                self.assertEqual(result, repeated)
                self.assertEqual(loader.call_count, calls)
            q2.plt.close('all')



class ProvenanceTests(unittest.TestCase):
    def test_model_protocol_requires_exact_q1_data_and_model_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(q2, 'OUTPUT', Path(directory)):
            split_protocol = {'artifacts': {'split_manifest.csv': 'frozen'}, 'vcf_exports': {'clinvar-train.vcf': 'a', 'clinvar-test.vcf': 'b'}}
            q2.write_json(q2.OUTPUT / 'protocol.json', {'config': q2.protocol_config(), **split_protocol})
            with patch.object(q2.q1, 'verify_protocol', return_value=split_protocol):
                q2.verify_protocol()
                with patch.object(q2, 'CS', [100.]), self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                    q2.verify_protocol()
            with patch.object(q2.q1, 'verify_protocol', return_value={**split_protocol, 'vcf_exports': {'clinvar-train.vcf': 'changed'}}):
                with self.assertRaisesRegex(RuntimeError, 'exact frozen Q1 inputs'):
                    q2.verify_protocol()
            with patch.object(q2.q1, 'verify_protocol', return_value={**split_protocol, 'artifacts': {'split_manifest.csv': 'reassigned'}}):
                with self.assertRaisesRegex(RuntimeError, 'exact frozen Q1 inputs'):
                    q2.verify_protocol()

    def test_missing_features_cannot_restart_inference_after_evaluation(self):
        dna = pd.DataFrame([['v1', 'A', 'C']], columns=q2.DNA_COLUMNS)
        identity = {'fixture': True}
        with tempfile.TemporaryDirectory() as directory, patch.object(q2, 'OUTPUT', Path(directory)), \
                patch.object(q2, 'verify_protocol'), patch.object(q2, 'read_csv', return_value=dna), \
                patch.object(q2, 'experiment_identity', return_value=identity), patch.object(q2, 'load_model') as loader:
            q2.write_json(q2.OUTPUT / 'validation_report_started.json', {'identity': identity})
            with self.assertRaisesRegex(RuntimeError, 'feature batch is missing'):
                q2.extract_features(dna)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
