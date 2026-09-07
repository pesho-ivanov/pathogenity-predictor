"""CPU checks for DNA-only scoring, frozen batches and grouped evaluation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import q1, q2, q2_7b
except ImportError:
    from notebooks.src import q1, q2, q2_7b


class InputTests(unittest.TestCase):
    def test_only_exact_frozen_dna_pairs_are_accepted(self):
        reference = 'A' * 1024
        alternate = reference[:512] + 'C' + reference[513:]
        frame = pd.DataFrame([['v1', reference, alternate]], columns=q1.DNA_COLUMNS)
        q2_7b.assert_dna(frame, ['v1'])
        for invalid in [frame.assign(label=1), frame.assign(alt_sequence=reference),
                        frame.assign(ref_sequence='N'+reference[1:]), frame.assign(ref_sequence=reference[:-1]),
                        frame.assign(alt_sequence='C'+reference[1:])]:
            with self.subTest(columns=invalid.columns.tolist()), self.assertRaises(ValueError):
                q2_7b.assert_dna(invalid, ['v1'])
        with self.assertRaisesRegex(ValueError, 'ordered'):
            q2_7b.assert_dna(frame, ['v2'])

    def test_strand_order_and_memoization_do_not_depend_on_orientation(self):
        sequence = 'A' * 1023 + 'C'
        reverse = q1.reverse_complement(sequence)
        calls = []
        def likelihood(model, value):
            calls.append(value)
            return -2. if value == min(sequence, reverse) else -4.
        with patch.object(q2_7b, 'mean_likelihood', likelihood):
            memo = {}
            self.assertEqual(q2_7b.strand_average(None, sequence, memo), -3.)
            self.assertEqual(q2_7b.strand_average(None, reverse, memo), -3.)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], min(sequence, reverse))
            self.assertEqual(q2_7b.strand_average(None, reverse, {}), -3.)
            self.assertEqual(calls[:2], calls[2:])


class CacheTests(unittest.TestCase):
    def test_corruption_reordering_and_changed_score_definition_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'batch.npz'
            arrays = {'keys': np.array(['v1', 'v2']), 'identity': np.array('fixed'),
                      'reference_ll': np.array([-1., -2.]), 'alternate_ll': np.array([-2., -1.]),
                      'zero_shot': np.array([1., -1.])}
            def save():
                q2.save_npz(path, **arrays)
                return {'identity': 'fixed', 'sha256': q1.digest_file(path)}
            record = save()
            np.testing.assert_equal(q2_7b.read_batch(path, record, ['v1', 'v2'], 'fixed')['zero_shot'], [1., -1.])
            with self.assertRaisesRegex(ValueError, 'membership'):
                q2_7b.read_batch(path, record, ['v2', 'v1'], 'fixed')
            with self.assertRaisesRegex(ValueError, 'identity'):
                q2_7b.read_batch(path, record, ['v1', 'v2'], 'changed')
            arrays['zero_shot'] *= -1
            altered = save()
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                q2_7b.read_batch(path, record, ['v1', 'v2'], 'fixed')
            with self.assertRaisesRegex(ValueError, 'likelihood definition'):
                q2_7b.read_batch(path, altered, ['v1', 'v2'], 'fixed')
            arrays['zero_shot'][0] = np.nan
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                q2_7b.read_batch(path, save(), ['v1', 'v2'], 'fixed')


class EvaluationTests(unittest.TestCase):
    def test_component_bootstrap_preserves_perfect_and_reversed_rankings(self):
        frame = pd.DataFrame({'component': ['a','a','b','b'], 'label': [0,1,0,1],
                              'good': [0.,1.,0.,1.], 'bad': [1.,0.,1.,0.]})
        measured = q2_7b.metric_summary(frame, ['good', 'bad'], repetitions=25)
        self.assertEqual(measured['components'], 2)
        self.assertEqual(measured['prevalence'], .5)
        self.assertEqual(measured['metrics']['good']['auroc'], {'value': 1., 'ci95': [1., 1.]})
        self.assertEqual(measured['metrics']['good']['average_precision']['value'], 1.)
        self.assertEqual(measured['metrics']['bad']['auroc']['value'], 0.)
        self.assertEqual(measured, q2_7b.metric_summary(frame, ['good', 'bad'], repetitions=25))
        with self.assertRaisesRegex(ValueError, 'both classes'):
            q2_7b.metric_summary(frame.assign(label=0), ['good'], repetitions=25)


if __name__ == '__main__':
    unittest.main()
