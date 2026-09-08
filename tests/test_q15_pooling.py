"""Canonical mutation positions and differentiable Q15 pooling contracts."""

import unittest

try:
    import torch
    from notebooks.src import q1, q11, q15_pooling as pooling
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for Q15 pooling tests')
class PoolingTests(unittest.TestCase):
    def pair(self):
        reference = 'G' * 512 + 'A' + 'T' * 511
        alternate = reference[:512] + 'C' + reference[513:]
        return reference, alternate

    def test_actual_q11_crop_is_centered_and_strand_invariant(self):
        pair = self.pair()
        reverse = tuple(q1.reverse_complement(sequence) for sequence in pair)
        canonical = pooling.canonical_pair(*pair)
        self.assertEqual(canonical[:2], q11.crop_pair(*pair, length=512))
        self.assertEqual(canonical, pooling.canonical_pair(*reverse))
        self.assertEqual(canonical[2], 256)
        self.assertEqual([i for i, (r, a) in enumerate(zip(*canonical[:2])) if r != a], [256])
        sequences, positions = pooling.canonical_sequences([pair, reverse])
        self.assertEqual(sequences, list(canonical[:2]) * 2)
        self.assertEqual(positions, [256] * 4)

    def test_malformed_or_annotated_pairs_are_rejected(self):
        reference, alternate = self.pair()
        for pairs in [[], [(reference, alternate, 1)], ['ACGT'], [(reference, reference)],
                      [(reference, alternate[:-1])], [(reference, alternate.replace('C', 'N'))],
                      [(reference, 1)]]:
            with self.subTest(pairs=str(pairs)[:30]), self.assertRaises(ValueError):
                pooling.canonical_sequences(pairs)

    def test_local_windows_have_exact_boundaries_and_global_mean(self):
        hidden = torch.arange(512, dtype=torch.float32)[:, None, None].expand(512, 3, 2).clone()
        result = pooling.pool_hidden(hidden, [0, 256, 448], include_global=True)
        torch.testing.assert_close(result['local'], torch.tensor([[31.5] * 2, [287.5] * 2, [479.5] * 2]))
        torch.testing.assert_close(result['global'], torch.full((3, 2), 255.5))
        self.assertEqual(set(pooling.pool_hidden(hidden, [0, 256, 448])), {'local'})

    def test_gradients_reach_exactly_the_selected_tokens(self):
        hidden = torch.zeros(512, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        local = pooling.pool_hidden(hidden, [256, 448])['local']
        self.assertEqual(local.dtype, torch.float32)
        local.sum().backward()
        expected = torch.zeros_like(hidden)
        expected[256:320, 0, :] = 1 / 64
        expected[448:512, 1, :] = 1 / 64
        torch.testing.assert_close(hidden.grad, expected, atol=0, rtol=0)

    def test_invalid_shapes_indices_and_nonfinite_activations_fail(self):
        hidden = torch.zeros(512, 2, 3)
        for indices in [[-1, 256], [256, 449], [256], [[256, 256]], [256., 256.], [True, False]]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                pooling.pool_hidden(hidden, indices)
        for value in [torch.zeros(512, 3), torch.zeros(511, 2, 3), torch.zeros(512, 0, 3),
                      torch.zeros(512, 2, 3, dtype=torch.int32), hidden + float('nan')]:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                pooling.pool_hidden(value, [256, 256])
        for window in [0, 513, 1.5, True]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                pooling.pool_hidden(hidden, [256, 256], window=window)


if __name__ == '__main__':
    unittest.main()
