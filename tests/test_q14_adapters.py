"""CPU contracts for prospective multi-block LoRA attachment, using fake NeMo."""

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

try:
    import torch
    from notebooks.src import q14_adapters
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for adapter attachment tests')
class AdapterAttachmentTests(unittest.TestCase):
    def setUp(self):
        class FakeLoRALinear(torch.nn.Module):
            def __init__(self, projection, rank):
                super().__init__()
                self.to_wrap = projection
                self.adapter = torch.nn.Module()
                self.adapter.linear_in = torch.nn.Linear(projection.in_features, rank, bias=False)
                self.adapter.linear_out = torch.nn.Linear(rank, projection.out_features, bias=False)

            def forward(self, value):
                hidden = self.adapter.linear_in(value.to(self.adapter.linear_in.weight.dtype))
                change = self.adapter.linear_out(hidden).to(value.dtype)
                return self.to_wrap(value) + change

        class FakeLoRA:
            def __init__(self, **kwargs):
                self.options = kwargs

            def transform(self, projection, name, prefix):
                if f'{prefix}.{name}' not in self.options['target_modules']:
                    return projection
                return FakeLoRALinear(projection, self.options['dim'])

        module = ModuleType('nemo.collections.llm.peft.lora')
        module.LoRA, module.LoRALinear = FakeLoRA, FakeLoRALinear
        self.mock_modules = patch.dict(sys.modules, {'nemo.collections.llm.peft.lora': module})
        self.mock_modules.start()
        self.addCleanup(self.mock_modules.stop)

    def model(self):
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(32)])
        for index in (28, 29, 30):
            mixer = torch.nn.Module()
            mixer.dense_projection = torch.nn.Linear(4, 12)
            mixer.dense = torch.nn.Linear(4, 4)
            model.decoder.layers[index].mixer = mixer
        model.register_buffer('unchanged_buffer', torch.tensor([1., 2.]))
        return model

    def test_all_four_adapters_are_retained_and_original_tensors_are_frozen(self):
        model = self.model()
        original = {name: p for name, p in model.named_parameters()}
        before = q14_adapters.base.parameter_hash(model)
        x = torch.randn(3, 4)
        expected = model.decoder.layers[29].mixer.dense_projection(x).detach()
        adapters = q14_adapters.attach(model)
        self.assertEqual(list(adapters), ['block29_dense_projection', 'block29_dense',
                                         'block30_dense_projection', 'block30_dense'])
        self.assertEqual(sum(p.numel() for p in adapters.parameters()), 2 * 8 * (4 + 12 + 4 + 4))
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(trainable, {id(p) for p in adapters.parameters()})
        self.assertTrue(all(not p.requires_grad for p in original.values()))
        self.assertEqual(q14_adapters.base.parameter_hash(model, frozen_only=True), before)
        self.assertIs(model.decoder.layers[29].mixer.dense_projection.to_wrap,
                      model.get_submodule('decoder.layers.29.mixer.dense_projection.to_wrap'))
        self.assertIs(model.decoder.layers[29].mixer.dense_projection.to_wrap.weight,
                      original['decoder.layers.29.mixer.dense_projection.weight'])
        torch.testing.assert_close(model.decoder.layers[29].mixer.dense_projection(x), expected,
                                   atol=0, rtol=0)
        self.assertFalse(hasattr(model.decoder.layers[28].mixer.dense, 'adapter'))
        for adapter in adapters.values():
            self.assertEqual(adapter.linear_in.weight.dtype, torch.bfloat16)
            self.assertGreater(torch.count_nonzero(adapter.linear_in.weight), 0)
            self.assertEqual(torch.count_nonzero(adapter.linear_out.weight), 0)

    def test_seeded_initialization_is_independent_of_caller_random_state(self):
        first = q14_adapters.attach(self.model(), seed=7)
        torch.randn(100)
        repeated = q14_adapters.attach(self.model(), seed=7)
        different = q14_adapters.attach(self.model(), seed=8)
        for name in first:
            torch.testing.assert_close(first[name].linear_in.weight, repeated[name].linear_in.weight,
                                       atol=0, rtol=0)
        self.assertFalse(torch.equal(first['block29_dense'].linear_in.weight,
                                     different['block29_dense'].linear_in.weight))
        self.assertFalse(torch.equal(first['block29_dense'].linear_in.weight,
                                     first['block30_dense'].linear_in.weight))

    def test_invalid_or_missing_block_is_rejected_before_model_changes(self):
        for blocks in [(29, 29), (), (-1,), (32,), (True,), (29, 31)]:
            with self.subTest(blocks=blocks):
                model = self.model()
                before = q14_adapters.base.parameter_hash(model)
                with self.assertRaises(ValueError):
                    q14_adapters.attach(model, blocks=blocks)
                self.assertEqual(q14_adapters.base.parameter_hash(model), before)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertFalse(hasattr(model.decoder.layers[29].mixer.dense, 'adapter'))

    def test_repeated_attachment_and_invalid_parameters_are_rejected(self):
        model = self.model()
        q14_adapters.attach(model)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            q14_adapters.attach(model)
        for options in [{'rank': 0}, {'rank': True}, {'alpha': float('nan')}, {'seed': -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                q14_adapters.attach(self.model(), **options)


if __name__ == '__main__':
    unittest.main()
