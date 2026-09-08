"""Fixed-classifier controls must isolate learned changes to Q14 adapters."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from notebooks.src import q14, q14_backend as backend
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for fixed-head tests')
class FixedHeadTests(unittest.TestCase):
    def metrics(self, auroc, ap):
        return {'metrics': {'best_control': {'auroc': {'value': auroc},
                                            'average_precision': {'value': ap}}}}

    def test_initialization_uses_strongest_control_and_retains_q12_on_exact_ties(self):
        q12_result = self.metrics(.83, .72)
        with patch.object(backend.q1, 'read_json') as read:
            for other in [self.metrics(.82, .8), self.metrics(.83, .71), q12_result]:
                source, path, _ = q14.choose_initial_control(q12_result, other)
                self.assertEqual(source, 'q12_control')
                self.assertEqual(path, backend.q12.OUTPUT / 'best_control.pt')
            read.assert_not_called()
        history = {'adapter_winner': {'best_control': {'auroc': .82, 'average_precision': .7}},
                   'control_winner': {'best_control': {'auroc': .83, 'average_precision': .73}}}
        with patch.object(backend.q1, 'read_json', return_value=history):
            source, path, _ = q14.choose_initial_control(q12_result, self.metrics(.83, .73))
            self.assertEqual(source, 'q13_control')
            self.assertEqual(path, backend.q13.OUTPUT / 'control_winner' / 'best_control.pt')
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                q14.choose_initial_control(q12_result, self.metrics(.85, .8))

    def test_loader_preserves_control_head_and_scaler_without_inheriting_adapted_head(self):
        for source in ['q12_control', 'q13_control']:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory)
                stack.enter_context(patch.object(q14, 'OUTPUT', output))
                stack.enter_context(patch.object(q14, 'ROOT', output))
                stack.enter_context(patch.object(q14, 'CONFIG', dict(q14.CONFIG, feature_dimension=2)))
                labels = {'train': np.array([0, 1]), 'validation': np.array([0, 1])}
                values = torch.tensor([0., 1.])
                metrics = {name: {'value': value} for name, value in q14.metric(labels['validation'], values.numpy()).items()}
                old = {'head': {'weight': torch.tensor([[-9., -9.]]), 'bias': torch.tensor([-9.])},
                       'control': {'weight': torch.tensor([[1., 2.]]), 'bias': torch.tensor([3.])},
                       'mean': torch.tensor([4., 5.]), 'scale': torch.tensor([6., 7.]),
                       'control_scores': values, 'strength': .01}
                protocol = {'initialization_checkpoint': 'source.pt', 'initialization_checkpoint_sha256': 'sha',
                            'initialization_source': source, 'initial_control_metrics': metrics}
                stack.enter_context(patch.object(backend.q1, 'read_json', side_effect=lambda path:
                    protocol if path.name == 'protocol.json' else {'class_weights': [1., 1.]}))
                stack.enter_context(patch.object(backend.q1, 'verify_file'))
                stack.enter_context(patch.object(backend.q12, 'verify_protocol', return_value='q12-id'))
                stack.enter_context(patch.object(backend.q13, 'verify_protocol', return_value='q13-id'))
                stack.enter_context(patch.object(backend.previous, 'load_state', return_value=old))
                stack.enter_context(patch.object(backend.q13_backend, 'load_state', return_value=old))
                result = backend.load_head('q14-id')
                backend.validate_fixed_head(result, labels)
                for key in ('weight', 'bias'):
                    torch.testing.assert_close(result[key], old['control'][key], atol=0, rtol=0)
                for key in ('mean', 'scale'):
                    torch.testing.assert_close(result[key], old[key], atol=0, rtol=0)
                self.assertEqual(result['strength'], 0.)
                self.assertTrue(result['head_fixed'])
                bad = dict(result, class_weights=torch.tensor([.8, 1.3]))
                with self.assertRaisesRegex(ValueError, 'Class weights'):
                    backend.validate_fixed_head(bad, labels)

    def test_adapter_updates_flow_through_frozen_head_which_never_enters_optimizer(self):
        torch.manual_seed(7)
        adapters = torch.nn.ModuleDict({'mixer': torch.nn.Linear(2, 2, bias=False)})
        head, control = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        control.load_state_dict(head.state_dict()); control.requires_grad_(False)
        heads = dict(backend.base.cpu_state(head.state_dict()), mean=torch.zeros(2), scale=torch.ones(2))
        mean, scale = heads['mean'].clone(), heads['scale'].clone()
        optimizer = backend.AdapterOptimizer(adapters, head, adapter_lr=.01)
        self.assertTrue(all(not p.requires_grad for p in head.parameters()))
        optimized = {id(p) for group in optimizer.optimizer.param_groups for p in group['params']}
        self.assertFalse(optimized & {id(p) for p in head.parameters()})
        x, y = torch.tensor([[1., 2.], [-1., -2.]]), torch.tensor([1., 0.])
        before = adapters['mixer'].weight.detach().clone()
        for _ in range(3):
            loss = backend.head_loss(head, (adapters['mixer'](x) - mean) / scale, y, torch.ones(2), 0.)
            loss.backward()
            self.assertGreater(adapters['mixer'].weight.grad.abs().sum(), 0.)
            self.assertTrue(all(p.grad is None for p in head.parameters()))
            diagnostic = optimizer.step(.5)
            self.assertEqual(diagnostic['head_update_norm'], 0.)
            backend.verify_fixed_tensors(head, control, mean, scale, heads)
        self.assertFalse(torch.equal(before, adapters['mixer'].weight))
        scale[0] = 2.
        with self.assertRaisesRegex(ValueError, 'scaler changed'):
            backend.verify_fixed_tensors(head, control, mean, scale, heads)


if __name__ == '__main__':
    unittest.main()
