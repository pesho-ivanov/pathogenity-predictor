"""CPU model doubles verify frozen-prefix replay values, ordering and gradients."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

try:
    import torch
    from notebooks.src import q15_replay as replay
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for replay tests')
class PrefixReplayTests(unittest.TestCase):
    def setUp(self):
        self.config_patch = patch.object(replay.q11, 'CONFIG', dict(replay.q11.CONFIG, hidden_size=4))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.tokenizer = SimpleNamespace(text_to_ids=lambda sequence: list(sequence.encode('ascii')))

        class Embedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = torch.nn.Embedding(256, 4, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.word_embeddings.weight.copy_(torch.arange(1024).reshape(256, 4) / 128.)

            def forward(self, input_ids, position_ids):
                return self.word_embeddings(input_ids).transpose(0, 1).contiguous()

        class Rotary(torch.nn.Module):
            def get_rotary_seq_len(self, inference, decoder, hidden, config, packed):
                assert inference is None and packed is None and decoder.input_tensor is None
                return hidden.shape[0]

            def forward(self, length):
                return torch.arange(length, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 1, 4) / 4096.

        class Layer(torch.nn.Module):
            def __init__(self, number):
                super().__init__()
                self.attention = number in (3, 10, 17, 24, 31)
                self.number = number
                if number in (29, 30):
                    self.adapter_a = torch.nn.Parameter(torch.linspace(-.2, .2, 8).reshape(2, 4).to(torch.bfloat16))
                    self.adapter_b = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))

            def forward(self, hidden, attention_mask, inference_params=None, rotary_pos_emb=None):
                assert attention_mask is None and inference_params is None and rotary_pos_emb.shape == (512, 1, 1, 4)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                if self.number in (29, 30):
                    hidden = hidden + (hidden @ self.adapter_a.T) @ self.adapter_b.T
                if self.attention:
                    hidden = hidden + rotary_pos_emb[:, 0, 0].unsqueeze(1).to(hidden.dtype)
                    return hidden, None
                return hidden

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = Embedding()
                self.rotary_pos_emb = Rotary()
                self.decoder = torch.nn.Module()
                self.decoder.layers = torch.nn.ModuleList([Layer(i) for i in range(32)])
                self.decoder.pre_process, self.decoder.input_tensor = True, None
                self.pre_process, self.position_embedding_type = True, 'rope'
                self.transformer_config = SimpleNamespace(hidden_size=4, params_dtype=torch.bfloat16,
                    fp32_residual_connection=False, sequence_parallel=False, tensor_model_parallel_size=1,
                    context_parallel_size=1, fp8=None)
                self.requires_grad_(False)
                for layer in self.decoder.layers[29:31]:
                    layer.adapter_a.requires_grad_(True); layer.adapter_b.requires_grad_(True)
                self.eval()

            def forward(self, tokens, positions, attention_mask=None):
                hidden = self.embedding(input_ids=tokens, position_ids=positions)
                rotary = self.rotary_pos_emb(hidden.shape[0])
                for layer in self.decoder.layers:
                    hidden = layer(hidden, attention_mask, inference_params=None, rotary_pos_emb=rotary)
                hidden = hidden[0] if isinstance(hidden, tuple) else hidden
                # A deliberately different post-block output proves that replay
                # uses the captured pre-final-norm activations, not model logits.
                return hidden * 7

        self.model = Model()
        self.rows = pd.DataFrame({
            'ref_sequence': ['A' * 1024, 'T' * 1024],
            'alt_sequence': ['A' * 512 + 'C' + 'A' * 511, 'T' * 512 + 'C' + 'T' * 511],
        })

    def reference(self):
        sequences = [sequence for pair in zip(self.rows.ref_sequence, self.rows.alt_sequence)
                     for sequence in replay.q11.crop_pair(*pair)]
        tokens = torch.tensor([self.tokenizer.text_to_ids(sequence) for sequence in sequences])
        positions = torch.arange(512)[None].expand(len(sequences), -1)
        captured = {}
        handles = []
        for block in (28, 31):
            def capture(module, args, output, block=block):
                captured[block] = output[0] if isinstance(output, tuple) else output
            handles.append(self.model.decoder.layers[block].register_forward_hook(capture))
        try:
            self.model(tokens, positions, attention_mask=None)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def gradient_result(self, pooled):
        weights = torch.tensor([.2, -.3, .4, .5])
        ((pooled[:, 0] + 2 * pooled[:, 1]) * weights).sum().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in self.model.named_parameters()
                     if parameter.requires_grad}
        self.model.zero_grad(set_to_none=True)
        return gradients

    def test_full_and_replay_values_and_gradients_match_before_and_after_adapter_update(self):
        cached = replay.extract_prefix(self.model, self.tokenizer, self.rows)
        self.assertEqual(cached.shape, (2, 2, 512, 4))
        self.assertEqual(cached.dtype, torch.bfloat16)
        self.assertEqual(cached.device.type, 'cpu')
        self.assertFalse(cached.requires_grad or cached.is_inference())
        original_cache = cached.clone()
        original_parameters = replay.base.parameter_hash(self.model, frozen_only=True)
        for updated in (False, True):
            with self.subTest(nonzero_adapters=updated):
                reference = self.reference()
                expected_cache = reference[28].detach().permute(1, 0, 2).reshape(2, 2, 512, 4)
                torch.testing.assert_close(cached, expected_cache, atol=0, rtol=0)
                expected = reference[31].float().mean(0).reshape(2, 2, 4)
                gradients = self.gradient_result(expected)
                hidden = replay.replay_tail_hidden(self.model, cached)
                torch.testing.assert_close(hidden, reference[31], atol=0, rtol=0)
                actual = replay.replay_pooled(self.model, cached)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                replay_gradients = self.gradient_result(actual)
                for name in gradients:
                    torch.testing.assert_close(replay_gradients[name], gradients[name], atol=0, rtol=0)
                if not updated:
                    self.assertTrue(all(torch.count_nonzero(value) == 0 for name, value in gradients.items()
                                        if name.endswith('adapter_a')))
                    self.assertTrue(any(torch.count_nonzero(value) > 0 for name, value in gradients.items()
                                        if name.endswith('adapter_b')))
                    optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.01)
                    for name, parameter in self.model.named_parameters():
                        if parameter.requires_grad:
                            parameter.grad = gradients[name].clone()
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    self.assertGreater(torch.count_nonzero(self.model.decoder.layers[29].adapter_b), 0)
                else:
                    self.assertTrue(any(torch.count_nonzero(value) > 0 for name, value in gradients.items()
                                        if name.endswith('adapter_a')))
                torch.testing.assert_close(cached, original_cache, atol=0, rtol=0)
                self.assertEqual(replay.base.parameter_hash(self.model, frozen_only=True), original_parameters)

    def test_reference_alternate_order_and_global_features_are_preserved(self):
        cached = replay.extract_prefix(self.model, self.tokenizer, self.rows)
        # The second input is reverse-complemented canonically: T/C becomes A/G.
        expected_bases = ['A', 'C', 'A', 'G']
        for index, base in enumerate(expected_bases):
            first = self.model.embedding.word_embeddings.weight[ord(base)].detach()
            rotary_add = self.model.rotary_pos_emb(512)[256, 0, 0].to(torch.bfloat16)
            for _ in range(4):
                first = first + rotary_add
            torch.testing.assert_close(cached[index // 2, index % 2, 256], first, atol=0, rtol=0)
        reference = self.reference()[31].float().mean(0).reshape(2, 2, 4)
        expected = replay.q12_backend.features_from_pooled(reference[:, 0], reference[:, 1] - reference[:, 0])
        torch.testing.assert_close(replay.replay_features(self.model, cached), expected, atol=0, rtol=0)

    def test_training_prefix_and_inference_mode_caches_are_rejected(self):
        cached = replay.extract_prefix(self.model, self.tokenizer, self.rows)
        with torch.inference_mode():
            invalid = cached.clone()
            with self.assertRaisesRegex(ValueError, 'inference_mode'):
                replay.extract_prefix(self.model, self.tokenizer, self.rows)
        with self.assertRaisesRegex(ValueError, 'ordinary'):
            replay.replay_tail_hidden(self.model, invalid)
        self.model.embedding.word_embeddings.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'completely frozen'):
            replay.extract_prefix(self.model, self.tokenizer, self.rows)


if __name__ == '__main__':
    unittest.main()
