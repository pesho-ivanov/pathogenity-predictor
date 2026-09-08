"""CPU checks for shared deployment, objective skips, schedules and checkpoints."""

from pathlib import Path
from contextlib import ExitStack
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from notebooks.src import q15_backend as backend, q15_objective


class PairedOptimizerTests(unittest.TestCase):
    def model(self):
        layer = torch.nn.Linear(3, 1, dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[.25, -.125, .375]], dtype=torch.bfloat16))
            layer.bias.fill_(.125)
        return layer

    def update(self, model, arm, step, labels=None):
        arm.activate()
        values = torch.tensor([[1., -2., .5], [.5, 1., -.25], [-1., .5, 2.]], dtype=torch.bfloat16)
        labels = torch.tensor([0., 1., 1.]) if labels is None else labels
        loss = q15_objective.compute_loss(model(values).flatten().float(), labels,
            objective=arm.objective, class_weights=torch.tensor([1.5, .75]))
        return arm.update(loss, step)

    def test_interleaved_arms_match_separate_optimizers(self):
        shared = self.model()
        arms = {name: backend.Arm(shared, name, .01) for name in ['weighted_bce', 'pairwise_logistic']}
        separate = {name: self.model() for name in arms}
        reference = {name: backend.Arm(separate[name], name, .01) for name in arms}
        for step in range(1, 5):
            for name, arm in arms.items():
                self.update(shared, arm, step)
                self.update(separate[name], reference[name], step)
                for actual, expected in zip(arm.masters, reference[name].masters):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for field in ['step', 'exp_avg', 'exp_avg_sq']:
                        torch.testing.assert_close(arm.optimizer.state[actual][field],
                            reference[name].optimizer.state[expected][field], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(arms['weighted_bce'].masters,
                                                               arms['pairwise_logistic'].masters)))
        for name, arm in arms.items():
            arm.activate()
            self.assertEqual(arm.steps, 4)
            for key, value in shared.state_dict().items():
                torch.testing.assert_close(value, separate[name].state_dict()[key], rtol=0, atol=0)

    def test_single_class_skip_changes_neither_parameters_nor_moments(self):
        model = self.model()
        arm = backend.Arm(model, 'pairwise_logistic', .01)
        self.update(model, arm, 1)
        before = [master.detach().clone() for master in arm.masters]
        moments = [{key: value.clone() for key, value in arm.optimizer.state[master].items()} for master in arm.masters]
        deployed = {key: value.clone() for key, value in model.state_dict().items()}
        result = self.update(model, arm, 2, torch.ones(3))
        self.assertFalse(result['updated'])
        self.assertEqual((arm.attempted_steps, arm.steps), (2, 1))
        for master, original, old_state in zip(arm.masters, before, moments):
            torch.testing.assert_close(master, original, rtol=0, atol=0)
            for key, value in old_state.items():
                torch.testing.assert_close(arm.optimizer.state[master][key], value, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, deployed[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, 'cursors'):
            self.update(model, arm, 4)

    def test_chunk_schedule_preserves_membership_and_counts_every_exposure(self):
        exposures = np.zeros(23, dtype=int)
        chunks = list(backend.chunk_orders(23, 8, 3, 42))
        other = list(backend.chunk_orders(23, 8, 3, 42))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(sorted(np.concatenate([indexes for _, indexes, _ in chunks]).tolist()), list(range(23)))
        for (_, indexes, orders), (_, indexes2, orders2) in zip(chunks, other):
            np.testing.assert_array_equal(indexes, indexes2)
            for order, order2 in zip(orders, orders2):
                np.testing.assert_array_equal(order, order2)
                self.assertEqual(sorted(order.tolist()), list(range(len(indexes))))
                exposures[indexes[order]] += 1
        np.testing.assert_array_equal(exposures, np.full(23, 3))

    def test_schedule_uses_attempts_and_prespecified_endpoints(self):
        self.assertEqual(backend.schedule(64), 1.)
        self.assertEqual(backend.schedule(2304), .1)
        self.assertEqual(backend.schedule(1), 1 / 64)

    def test_equivalence_guard_rejects_a_nonzero_adapter_gradient_difference(self):
        expected = torch.tensor([1., .1, 0.])
        backend.assert_equivalent(expected.clone(), expected, 'gradient', gradient=True)
        with self.assertRaisesRegex(ValueError, 'equivalence failed'):
            backend.assert_equivalent(torch.tensor([1., .11, 0.]), expected, 'gradient', gradient=True)

    def test_checkpoint_roundtrip_and_checksum_rejection(self):
        model = self.model()
        head = torch.nn.Linear(3, 1).requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory, patch.object(backend.base, 'rng_state', return_value={}):
            path = Path(directory) / 'state.pt'
            state = backend.checkpoint('fixed-identity', model.state_dict(), head,
                torch.zeros(3), torch.ones(3), {'steps': 0, 'attempted_steps': 0},
                np.array([-.2, 2.], dtype=np.float32), np.array([-.2, 2.], dtype=np.float32))
            backend.previous.save_state(path, state)
            restored = backend.load_state(path, 'fixed-identity')
            torch.testing.assert_close(restored['control_scores'], torch.tensor([-.2, 2.]))
            self.assertEqual(restored['format'], 'q15-cached-v1')
            with self.assertRaisesRegex(ValueError, 'identity'):
                backend.load_state(path, 'changed')
            payload = bytearray(path.read_bytes())
            payload[-10] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                backend.load_state(path, 'fixed-identity')

    def test_cached_predictions_preserve_inherited_logit_scale(self):
        head = torch.nn.Linear(2, 1).requires_grad_(False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2., -1.]]))
            head.bias.fill_(.5)
        raw = torch.tensor([[1., 2.], [-1., 4.]])
        with patch.object(backend, 'cached_raw', return_value=raw):
            scores = backend.score_cache(None, None, head, torch.tensor([0., 2.]), torch.tensor([1., 2.]))
        np.testing.assert_array_equal(scores, np.array([2.5, -2.5], dtype=np.float32))

    def test_deadline_stops_both_arms_at_one_cursor_and_retains_unvisited_rows(self):
        model = self.model()
        head = torch.nn.Linear(3, 1).requires_grad_(False)
        mean, scale = torch.zeros(3), torch.ones(3)
        heads = {**{key: value.clone() for key, value in head.state_dict().items()},
                 'mean': mean.clone(), 'scale': scale.clone(), 'class_weights': torch.ones(2)}
        rows = {'train': pd.DataFrame({'variant_key': [f'train-{i}' for i in range(8)]}),
                'validation': pd.DataFrame({'variant_key': ['v0', 'v1']})}
        labels = {'train': np.zeros(8, dtype=np.int64), 'validation': np.array([0, 1])}
        scores = np.array([-1., 1.], dtype=np.float32)

        class Deadline:
            def __init__(self):
                self.checks = 0

            def elapsed(self):
                return 10.

            def can_work(self, seconds, reserve):
                self.checks += 1
                return self.checks <= 2  # Admit one chunk and one paired attempt.

        config = dict(backend.q15.CONFIG, chunk_variants=4, passes_per_chunk=2,
                      microbatch_variants=2, checkpoint_steps=1, max_attempted_steps=8)
        calibration = {'prefix_seconds_per_variant': .1, 'paired_update_seconds': .1,
                       'tail_eval_seconds_per_variant': .1, 'full_seconds_per_variant': .1}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            for owner, name, value in [(backend.q15, 'OUTPUT', output), (backend.q15, 'CONFIG', config),
                (backend.base, 'rng_state', lambda: {}), (torch.cuda, 'synchronize', lambda: None),
                (backend, 'cache_rows', lambda *args: torch.ones(4, 3)),
                (backend, 'score_cache', lambda *args: scores.copy()),
                (backend.replay, 'replay_features', lambda _, batch: model(batch.to(torch.bfloat16)).float().expand(-1, 3))]:
                stack.enter_context(patch.object(owner, name, value))
            history = backend.train(None, None, model, {key: value.clone() for key, value in model.state_dict().items()},
                head, mean, scale, heads, rows, labels, torch.ones(2, 3), scores,
                'identity', {'adapter_lr': .01}, Deadline(), calibration)
            self.assertEqual(history['attempted_steps'], 1)
            self.assertEqual(history['completed_chunks'], 0)
            self.assertEqual(history['actual_unique_training_variants'], 2)
            self.assertEqual(history['examples_per_arm'], 2)
            self.assertEqual(history['trials']['weighted_bce']['steps'], 1)
            self.assertEqual(history['trials']['pairwise_logistic']['steps'], 0)
            self.assertEqual(len(history['trials']['pairwise_logistic']['skipped_batches']), 1)
            exposures = pd.read_csv(output / 'training_exposures.csv')
            self.assertEqual(exposures.variant_key.tolist(), rows['train'].variant_key.tolist())
            self.assertEqual(int((exposures.exposures_per_arm == 0).sum()), 6)
            for objective in config['objectives']:
                last = backend.load_state(output / objective / 'last_checkpoint.pt', 'identity')
                self.assertEqual(last['progress']['attempted_steps'], 1)


if __name__ == '__main__':
    unittest.main()
