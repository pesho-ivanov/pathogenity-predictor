"""Integrity and paired uncertainty checks for the frozen 7B experiment."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import q1, q2, q10
except ImportError:
    from notebooks.src import q1, q2, q10


class Q10Tests(unittest.TestCase):
    def test_feature_reuse_rejects_changed_context_or_backbone(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(q10, 'OUTPUT', Path(temporary)):
            root = q10.OUTPUT
            archive = root / 'archive/producer'
            archive.mkdir(parents=True)
            old_config = dict(q10.CONFIG)
            old_config['classifier'] = old_config['classifier'].replace('max_iter=30000,', 'max_iter=3000,')
            old = {'configuration': old_config,
                   'sources': {'notebooks/src/q10.py': 'old', 'notebooks/src/q10_backend.py': 'same'},
                   'q1_protocol_sha256': 'q1', 'checkpoint_manifest_sha256': 'weights',
                   'environment_sha256': 'runtime', 'upstream_sources': {'nemo': 'pinned'}}
            current = dict(old, configuration=dict(q10.CONFIG),
                           sources={'notebooks/src/q10.py': 'new', 'notebooks/src/q10_backend.py': 'same'})
            q1.write_json(archive / 'protocol.json', old)
            q1.write_json(root / 'feature_manifest.json', {'identity': q1.fingerprint(old)})
            reuse = {'archive': 'archive/producer', 'from_identity': q1.fingerprint(old),
                     'artifacts': {'archive/producer/protocol.json': q1.digest_file(archive / 'protocol.json')}}
            def record(protocol):
                q1.write_json(root / 'protocol.json', protocol)
                q1.write_json(root / 'feature_reuse.json', dict(reuse, to_identity=q1.fingerprint(protocol)))
                return q1.fingerprint(protocol)
            self.assertEqual(q10.feature_identity(record(current)), q1.fingerprint(old))
            changed = dict(current, configuration=dict(current['configuration'], context_bp=2048))
            with self.assertRaisesRegex(ValueError, 'iteration limit'):
                q10.feature_identity(record(changed))
            changed = dict(current, sources=dict(current['sources'], **{'notebooks/src/q10_backend.py': 'changed'}))
            with self.assertRaisesRegex(ValueError, 'backbone'):
                q10.feature_identity(record(changed))

    def test_cached_features_reject_changed_identity_keys_shape_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features.npz'
            keys = ['variant-a', 'variant-b']
            values = np.ones((2, q10.CONFIG['feature_dimension']), dtype=np.float32)
            q2.save_npz(path, keys=np.asarray(keys), evo=values)
            record = {'identity': 'frozen', 'keys': keys, 'sha256': q1.digest_file(path)}
            np.testing.assert_array_equal(q10.read_feature_batch(path, record, keys, 'frozen'), values)
            for expected_keys, identity in [(keys[::-1], 'frozen'), (keys, 'changed')]:
                with self.assertRaises(ValueError):
                    q10.read_feature_batch(path, record, expected_keys, identity)
            q2.save_npz(path, keys=np.asarray(keys[::-1]), evo=values)
            record['sha256'] = q1.digest_file(path)
            with self.assertRaisesRegex(ValueError, 'keys differ'):
                q10.read_feature_batch(path, record, keys, 'frozen')
            q2.save_npz(path, keys=np.asarray(keys), evo=values[:, :-1])
            record['sha256'] = q1.digest_file(path)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                q10.read_feature_batch(path, record, keys, 'frozen')
            path.write_bytes(b'corrupted cache')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                q10.read_feature_batch(path, record, keys, 'frozen')

    def test_identical_predictors_have_exactly_zero_paired_difference(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        values = np.array([.1, .9, .6, .4, .2, .8])
        with patch.dict(q10.CONFIG, bootstrap_repetitions=40):
            result, delta = q10.paired_metrics(labels, {'1b': values, '7b': values},
                                              np.array(['a', 'a', 'b', 'b', 'c', 'c']))
        self.assertEqual(result['1b'], result['7b'])
        for metric in delta.values():
            self.assertEqual(metric, {'value': 0., 'ci95': [0., 0.]})


if __name__ == '__main__':
    unittest.main()
