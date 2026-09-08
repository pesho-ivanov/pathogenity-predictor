"""CPU checks for Q16 continuation membership, selection and resumability."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from notebooks.src import q16_backend as backend


class ContinuationTests(unittest.TestCase):
    def test_full_training_plan_keeps_all_three_eight_variant_tails(self):
        orders = backend.epoch_orders(46888, 3, 42)
        plan = list(backend.batches(orders, 32))
        self.assertEqual(len(plan), 4398)
        for epoch in range(1, 4):
            epoch_batches = [chosen for number, _, chosen in plan if number == epoch]
            self.assertEqual(len(epoch_batches[-1]), 8)
            np.testing.assert_array_equal(np.sort(np.concatenate(epoch_batches)), np.arange(46888))
        self.assertEqual(sum(len(chosen) for _, _, chosen in plan), 140664)
        self.assertEqual(backend.order_hashes(orders), backend.order_hashes(backend.epoch_orders(46888, 3, 42)))
        self.assertEqual(len(set(backend.order_hashes(orders))), 3)

    def progress(self, orders, steps, size=8):
        plan = list(backend.batches(orders, size))
        epoch, offset, chosen = plan[steps - 1]
        seen = sum(len(batch) for _, _, batch in plan[:steps])
        return {'steps': steps, 'epoch': epoch, 'offset': offset + len(chosen), 'examples_seen': seen,
                'unique_variants_seen': min(seen, len(orders[0])), 'epoch_order_sha256': backend.order_hashes(orders)}

    def test_resume_rejects_changed_order_and_exposure_cursor(self):
        orders = backend.epoch_orders(23, 3, 42)
        for step in [1, 3, 4, 9]:
            progress = self.progress(orders, step)
            backend.validate_progress(progress, orders, 8)
            for field in ['offset', 'examples_seen', 'unique_variants_seen']:
                with self.subTest(step=step, field=field), self.assertRaises(ValueError):
                    backend.validate_progress(dict(progress, **{field: progress[field] + 1}), orders, 8)
        bad = self.progress(orders, 4)
        bad['epoch_order_sha256'] = backend.order_hashes(backend.epoch_orders(23, 3, 99))
        with self.assertRaisesRegex(ValueError, 'order'):
            backend.validate_progress(bad, orders, 8)

    def test_retention_needs_strict_auc_improvement_without_ap_reduction(self):
        parent = {'auroc': .85, 'average_precision': .77}
        self.assertFalse(backend.retain_continuation(parent, parent))
        self.assertFalse(backend.retain_continuation({'auroc': .86, 'average_precision': .769}, parent))
        self.assertFalse(backend.retain_continuation({'auroc': .85, 'average_precision': .78}, parent))
        self.assertTrue(backend.retain_continuation({'auroc': .850001, 'average_precision': .77}, parent))
        self.assertTrue(backend.better({'auroc': .86, 'average_precision': .76}, parent))

    def test_schedule_preserves_frozen_endpoints(self):
        self.assertEqual(backend.schedule(1), 1 / 64)
        self.assertEqual(backend.schedule(64), 1.)
        self.assertEqual(backend.schedule(4398), .1)
        self.assertGreater(backend.schedule(512), backend.schedule(4096))
        with self.assertRaises(ValueError):
            backend.schedule(0)

    def test_budget_keeps_full_validation_and_reporting_reserve(self):
        with patch.object(backend.q16, 'deadline', return_value=10000.), patch.object(backend.time, 'time', return_value=5700.):
            budget = backend.Budget(0.)
            self.assertEqual(budget.reserve(17927, 2048), 4200.)
            self.assertTrue(budget.can_train(17927, 2048))
            budget.prediction_seconds_per_variant = .08
            self.assertGreater(budget.reserve(17927, 2048), 5600.)
            self.assertFalse(budget.can_train(17927, 2048))

    def test_paired_statistics_use_each_correct_reference_and_selected_alias(self):
        labels = np.array([0, 1, 0, 1])
        groups = np.array([0, 0, 1, 1])
        predictions = {'continuation': np.array([-2., 2., -1., 1.]),
                       'q14_parent': np.array([-1., .1, 1., .2]),
                       'frozen_control': np.array([2., -2., 1., -1.])}
        predictions['selected'] = predictions['q14_parent'].copy()
        with patch.object(backend.q16, 'CONFIG', dict(backend.q16.CONFIG, bootstrap_repetitions=20)):
            result = backend.summarize(labels, predictions, groups)
            self.assertEqual(set(result['metrics']), set(predictions))
            self.assertEqual(result['metrics']['selected'], result['metrics']['q14_parent'])
            self.assertEqual(result['paired_continuation_minus_q14_parent']['auroc']['value'], .5)
            self.assertEqual(result['paired_continuation_minus_frozen_control']['auroc']['value'], 1.)
            predictions['selected'] = np.array([7., 8., 9., 10.])
            with self.assertRaisesRegex(ValueError, 'Selected predictions'):
                backend.summarize(labels, predictions, groups)

    def test_reused_optimizer_and_adapter_configuration_cannot_silently_drift(self):
        backend.verify_optimizer_config()
        changed = dict(backend.q16.CONFIG, lora=dict(backend.q16.CONFIG['lora'], rank=16))
        with patch.object(backend.q16, 'CONFIG', changed), self.assertRaisesRegex(ValueError, 'adapter configuration'):
            backend.verify_optimizer_config()

    def test_operational_batch_guard_retains_tolerance_and_serial_sensitivity_is_measured(self):
        expected = {'features': torch.tensor([[1., 2.], [3., 4.]]),
                    'standardized': torch.tensor([[.5, -.5], [1., -1.]]),
                    'logits': torch.tensor([.1, .2])}
        identical = backend.comparison_statistics(expected, expected)
        backend.require_operational_equivalence(identical, 'batch32_permutation')
        self.assertTrue(all(value['exact_fraction'] == 1. for value in identical.values()))
        different = {key: value + .02 for key, value in expected.items()}
        sensitivity = backend.comparison_statistics(different, expected)
        self.assertAlmostEqual(sensitivity['logits']['max_abs_difference'], .02, places=6)
        self.assertGreater(sensitivity['standardized']['relative_rms_difference'], 0.)
        self.assertFalse(sensitivity['features']['allclose_2e_4'])
        with self.assertRaisesRegex(ValueError, 'operational batch equivalence failed'):
            backend.require_operational_equivalence(sensitivity, 'batch32_vs_batch8')

    def model(self):
        adapters = torch.nn.ModuleDict({'mixer': torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)})
        head = torch.nn.Linear(2, 1)
        with torch.no_grad():
            adapters['mixer'].weight.copy_(torch.tensor([[.25, -.125], [.5, .125]], dtype=torch.bfloat16))
            head.weight.copy_(torch.tensor([[.5, -.25]])); head.bias.fill_(.1)
        return adapters, head

    def update(self, adapters, head, optimizer, step):
        values = torch.tensor([[1., -1.], [.5, 2.]], dtype=torch.bfloat16)
        logits = head(adapters['mixer'](values).float()).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        optimizer.step(backend.schedule(step))

    def test_resume_restores_fp32_masters_moments_and_fixed_head_behavior(self):
        original, head = self.model()
        optimizer = backend.parent_backend.AdapterOptimizer(original, head, .01)
        before_head = backend.base.cpu_state(head.state_dict())
        self.update(original, head, optimizer, 1)
        saved = backend.base.cpu_state({'masters': optimizer.masters, 'optimizer': optimizer.optimizer.state_dict()})
        resumed, resumed_head = self.model()
        backend.previous.restore_adapters(resumed, backend.previous.adapter_state(original))
        resumed_optimizer = backend.parent_backend.AdapterOptimizer(resumed, resumed_head, .01)
        backend.restore_optimizer(resumed_optimizer, saved)
        for step in [2, 3]:
            self.update(original, head, optimizer, step)
            self.update(resumed, resumed_head, resumed_optimizer, step)
        for left, right in zip(optimizer.masters, resumed_optimizer.masters):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            for field in ['exp_avg', 'exp_avg_sq', 'step']:
                torch.testing.assert_close(optimizer.optimizer.state[left][field], resumed_optimizer.optimizer.state[right][field], atol=0, rtol=0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value, before_head[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.parameters()))
        with torch.no_grad():
            resumed['mixer'].weight.fill_(9.)
        with self.assertRaisesRegex(ValueError, 'master does not reproduce'):
            backend.restore_optimizer(resumed_optimizer, saved)

    def test_checkpoint_roundtrip_rejects_corruption_wrong_identity_and_nonfinite_values(self):
        adapters, head = self.model()
        with tempfile.TemporaryDirectory() as directory, patch.object(backend.base, 'rng_state', return_value={}), patch.object(backend.q16, 'deadline', return_value=123.):
            path = Path(directory) / 'state.pt'
            state = backend.checkpoint('identity', adapters, head, torch.zeros(2), torch.ones(2), {'steps': 0})
            backend.previous.save_state(path, state)
            restored = backend.load_state(path, 'identity')
            self.assertEqual(restored['deadline_utc'], 123.)
            torch.testing.assert_close(restored['head']['weight'], head.weight, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.load_state(path, 'other')
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                backend.load_state(path, 'identity')
            state['scale'][0] = float('nan')
            backend.previous.save_state(path, state)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                backend.load_state(path, 'identity')


if __name__ == '__main__':
    unittest.main()
