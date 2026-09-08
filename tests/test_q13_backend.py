"""CPU checks of Q13's matched objectives, optimizer and checkpoint integrity."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from notebooks.src import q13, q13_backend as backend, q13_heads
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for Q13 optimizer tests')
class OptimizerTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(42)
        adapters = torch.nn.ModuleDict({'adapter': torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)})
        with torch.no_grad():
            adapters['adapter'].weight.zero_()
        head = torch.nn.Linear(2, 1)
        return adapters, head

    def test_large_adapter_gradients_leave_matched_head_step_identical(self):
        adapters, head = self.models()
        control = torch.nn.Linear(2, 1)
        control.load_state_dict(head.state_dict())
        optimizer = backend.AdapterOptimizer(adapters, head, .01)
        control_optimizer = torch.optim.AdamW(control.parameters(), lr=q13.CONFIG['head_lr'],
            weight_decay=0., betas=tuple(q13.CONFIG['betas']), eps=q13.CONFIG['epsilon'])
        features = torch.tensor([[.1, 2.], [3., -.5], [1., 1.]])
        target = torch.tensor([0., 1., 1.])
        weights = torch.tensor([1.5, .75, .75])
        for step in range(1, 4):
            for parameter in adapters.parameters():
                parameter.grad = torch.full_like(parameter, 1e8 / step)
            backend.head_loss(head, features, target, weights, .01).backward()
            backend.head_loss(control, features, target, weights, .01).backward()
            old = adapters['adapter'].weight.detach().clone()
            multiplier = backend.schedule(step, q13.CONFIG['max_steps'])
            diagnostics = optimizer.step(multiplier)
            control_norm = torch.nn.utils.clip_grad_norm_(control.parameters(), q13.CONFIG['clip_grad'])
            control_optimizer.param_groups[0]['lr'] = q13.CONFIG['head_lr'] * multiplier
            control_optimizer.step(); control_optimizer.zero_grad(set_to_none=True)
            for fitted, frozen in zip(head.parameters(), control.parameters()):
                torch.testing.assert_close(fitted, frozen, atol=0, rtol=0)
            self.assertAlmostEqual(diagnostics['head_gradient_norm'], float(control_norm), places=7)
            self.assertLess(diagnostics['adapter_clip_scale'], 1e-6)
            deployed = adapters['adapter'].weight.detach()
            self.assertGreater(diagnostics['adapters']['adapter.weight']['update_norm'], 0.)
            self.assertEqual(diagnostics['adapters']['adapter.weight']['update_norm'],
                             float((deployed.float() - old.float()).norm()))
            self.assertEqual(diagnostics['adapters']['adapter.weight']['changed_fraction'],
                             float((deployed != old).float().mean()))
            torch.testing.assert_close(deployed, optimizer.masters[0].detach().to(deployed.dtype), atol=0, rtol=0)
            self.assertIsNone(adapters['adapter'].weight.grad)

    def test_zero_multiplier_and_warmup_decay_boundaries(self):
        adapters, head = self.models()
        optimizer = backend.AdapterOptimizer(adapters, head, .01)
        originals = [p.detach().clone() for p in list(adapters.parameters()) + list(head.parameters())]
        for p in list(adapters.parameters()) + list(head.parameters()):
            p.grad = torch.ones_like(p)
        diagnostics = optimizer.step(0.)
        self.assertEqual(diagnostics['head_update_norm'], 0.)
        self.assertEqual(diagnostics['adapters']['adapter.weight']['update_norm'], 0.)
        for parameter, original in zip(list(adapters.parameters()) + list(head.parameters()), originals):
            torch.testing.assert_close(parameter, original, atol=0, rtol=0)
        with patch.dict(q13.CONFIG, {'warmup_steps': 4, 'minimum_lr_ratio': .1}):
            self.assertEqual(backend.schedule(1, 12), .25)
            self.assertEqual(backend.schedule(4, 12), 1.)
            self.assertAlmostEqual(backend.schedule(12, 12), .1)
            self.assertTrue(all(backend.schedule(i, 12) > backend.schedule(i + 1, 12) for i in range(4, 12)))
            self.assertEqual(backend.schedule(2, 2), 1.)
            for step in [0, 13]:
                with self.assertRaisesRegex(ValueError, 'out of range'):
                    backend.schedule(step, 12)

    def test_head_loss_matches_converged_fit_objective_and_gradients(self):
        _, head = self.models()
        features = torch.tensor([[.3, -.4], [1., 2.], [-.1, .7]])
        labels = torch.tensor([0., 1., 1.])
        class_weights = torch.tensor([1.5, .75])
        weights = torch.tensor(np.asarray(class_weights)[labels.numpy().astype(int)])
        paired = backend.head_loss(head, features, labels, weights, .03)
        fitted = q13_heads.objective(head.weight.flatten(), head.bias, features, labels, weights, .03)
        self.assertAlmostEqual(float(paired.detach()), float(fitted.detach()), places=6)
        paired_grads = torch.autograd.grad(paired, tuple(head.parameters()))
        fitted_grads = torch.autograd.grad(fitted, tuple(head.parameters()))
        for left, right in zip(paired_grads, fitted_grads):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)

    def test_checkpoint_roundtrip_rejects_corruption_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            state = {'format': 'q13-controlled-v1', 'identity': 'frozen-protocol',
                     'weight': torch.tensor([[.2, -.3]])}
            backend.save_state(path, state)
            restored = backend.load_state(path, 'frozen-protocol')
            torch.testing.assert_close(restored['weight'], state['weight'], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.load_state(path, 'different-protocol')
            payload = path.read_bytes()
            path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                backend.load_state(path, 'frozen-protocol')
            backend.save_state(path, state | {'format': 'different-format'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.load_state(path, 'frozen-protocol')


if __name__ == '__main__':
    unittest.main()
