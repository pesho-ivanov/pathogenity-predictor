"""CPU checks of Q15's training losses and the ranking skip contract."""

import math
import unittest

try:
    import torch
    from notebooks.src.q15_objective import compute_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for training objective tests')
class ObjectiveTests(unittest.TestCase):
    def ranking(self, logits, labels, **kwargs):
        return compute_loss(logits, labels, objective='pairwise_logistic', **kwargs)

    def test_ranking_matches_all_pairs_hand_calculation(self):
        logits = torch.tensor([0., math.log(3), math.log(2), -math.log(2)],
                              dtype=torch.float64)
        result = self.ranking(logits, torch.tensor([0, 1, 1, 0]))
        expected = sum(math.log(x) for x in (4 / 3, 3 / 2, 7 / 6, 5 / 4)) / 4
        self.assertAlmostEqual(result.loss.item(), expected, places=14)
        self.assertTrue(result.has_signal)
        self.assertEqual((result.positives, result.negatives, result.pairs), (2, 2, 4))

    def test_gradient_raises_positive_and_lowers_negative_scores(self):
        logits = torch.tensor([.2, -.8, .4], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 1])
        result = self.ranking(logits, labels)
        result.loss.backward()
        self.assertTrue((logits.grad[labels == 0] > 0).all())
        self.assertTrue((logits.grad[labels == 1] < 0).all())
        self.assertAlmostEqual(logits.grad.sum().item(), 0., places=14)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: self.ranking(value, labels).loss, (logits,)))

    def test_ranking_is_translation_and_class_weight_invariant(self):
        logits = torch.tensor([.2, -.8, .4], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 1])
        original = self.ranking(logits, labels).loss
        translated = self.ranking(logits + 100, labels,
                                  class_weights=torch.tensor([.25, 12.])).loss
        torch.testing.assert_close(original, translated, atol=1e-13, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(original, logits)[0],
                                   torch.autograd.grad(translated, logits)[0],
                                   atol=1e-13, rtol=0)

    def test_single_class_batches_signal_skip_and_have_safe_zero(self):
        for target in (0, 1):
            with self.subTest(target=target):
                logits = torch.tensor([1., -2.], requires_grad=True)
                result = self.ranking(logits, torch.full((2,), target))
                self.assertFalse(result.has_signal)
                self.assertEqual(result.pairs, 0)
                self.assertEqual(result.loss.item(), 0.)
                result.loss.backward()
                torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
                # No requires_grad is also a valid input; the caller still skips.
                detached = self.ranking(logits.detach(), torch.full((2,), target))
                self.assertFalse(detached.loss.requires_grad)
                self.assertFalse(detached.has_signal)

    def test_bce_uses_mean_of_cohort_weighted_examples(self):
        logits = torch.tensor([0., math.log(3), math.log(2)], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1])
        weights = torch.tensor([.5, 2.], dtype=torch.float64)
        result = compute_loss(logits, labels, objective='weighted_bce', class_weights=weights)
        expected = (.5 * math.log(2) + .5 * math.log(4) + 2 * math.log(1.5)) / 3
        self.assertAlmostEqual(result.loss.item(), expected, places=14)
        self.assertTrue(result.has_signal)
        self.assertEqual((result.positives, result.negatives, result.pairs), (1, 2, 2))
        single = compute_loss(logits, torch.ones(3), objective='weighted_bce', class_weights=weights)
        self.assertTrue(single.has_signal)
        self.assertGreater(single.loss.item(), 0)
        with self.assertRaisesRegex(ValueError, 'requires training-cohort'):
            compute_loss(logits, labels, objective='weighted_bce')

    def test_low_precision_inputs_have_finite_float32_ranking_loss(self):
        logits = torch.tensor([60000., -60000.], dtype=torch.float16, requires_grad=True)
        result = self.ranking(logits, torch.tensor([0, 1]))
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertEqual(result.loss.item(), 120000.)
        result.loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([1., -1.], dtype=torch.float16))

    def test_bad_shapes_nonfinite_inputs_and_nonbinary_labels_are_rejected(self):
        invalid = [
            (torch.zeros(2, 1), torch.zeros(2, 1)),
            (torch.zeros(2), torch.zeros(1)),
            (torch.empty(0), torch.empty(0)),
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.tensor([float('nan'), 0.]), torch.tensor([0, 1])),
            (torch.tensor([float('inf'), 0.]), torch.tensor([0, 1])),
            (torch.zeros(2), torch.tensor([float('nan'), 1.])),
            (torch.zeros(2), torch.tensor([0., .5])),
            (torch.zeros(2), torch.tensor([-1, 1])),
            (torch.zeros(2), torch.tensor([0j, 1j])),
        ]
        for logits, labels in invalid:
            with self.subTest(logits=logits, labels=labels), self.assertRaises(ValueError):
                self.ranking(logits, labels)
        with self.assertRaises(TypeError):
            self.ranking([0., 1.], torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            compute_loss(torch.zeros(2), torch.tensor([0, 1]), objective='unknown')

    def test_invalid_class_weights_are_rejected(self):
        for weights in (torch.ones(1), torch.ones(2, 1), torch.tensor([0., 1.]),
                        torch.tensor([-1., 1.]), torch.tensor([float('nan'), 1.]),
                        torch.tensor([float('inf'), 1.]), torch.tensor([1j, 1j])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                compute_loss(torch.zeros(2), torch.tensor([0, 1]),
                             objective='weighted_bce', class_weights=weights)
        with self.assertRaises(TypeError):
            compute_loss(torch.zeros(2), torch.tensor([0, 1]),
                         objective='weighted_bce', class_weights=[1., 1.])


if __name__ == '__main__':
    unittest.main()
