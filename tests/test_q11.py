"""Q11 scientific input, optimizer, resume and delivery contract tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

from notebooks.src import q11, refresh_q11
if torch is not None:
    from notebooks.src import q11_backend as backend


class InputTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame({'variant_key': ['t0', 't1', 'v0', 'v1'],
                                      'split': ['train', 'train', 'validation', 'validation'],
                                      'component': ['t', 't', 'v', 'v']})
        self.dna = pd.DataFrame({'variant_key': ['v1', 't1', 'v0', 't0'],
                                 'ref_sequence': ['AAAA'] * 4, 'alt_sequence': ['ACAA'] * 4})
        self.labels = {s: pd.DataFrame({'variant_key': keys, 'label': [1, 0]}) for s, keys in
                       [('train', ['t1', 't0']), ('validation', ['v1', 'v0'])]}

    def test_labels_and_dna_align_by_key(self):
        indexes, labels = q11.align_inputs(self.manifest, self.dna, self.labels)
        np.testing.assert_array_equal(indexes['train'], [3, 1])
        np.testing.assert_array_equal(labels['train'], [0, 1])
        np.testing.assert_array_equal(labels['validation'], [0, 1])

    def test_annotations_cannot_enter_encoder(self):
        for column in ['label', 'MC', 'CLNSIG', 'gene_ids']:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'DNA-only'):
                q11.align_inputs(self.manifest, self.dna.assign(**{column: 1}), self.labels)

    def test_overlap_and_missing_labels_fail(self):
        with self.assertRaisesRegex(ValueError, 'cross partitions'):
            q11.align_inputs(self.manifest.assign(component='same'), self.dna, self.labels)
        with self.assertRaisesRegex(ValueError, 'label membership'):
            q11.align_inputs(self.manifest, self.dna, {**self.labels, 'train': self.labels['train'].iloc[:1]})
        with self.assertRaisesRegex(ValueError, 'DNA membership'):
            q11.align_inputs(self.manifest, self.dna.iloc[:3], self.labels)

    def test_pilot_protocol_is_rejected_before_labels(self):
        protocol = {'config': {'context_bp': 1024}, 'vcf_exports': {'clinvar-train-pilot.vcf': 'a'}}
        with patch.object(q11.q1_full, 'verify_protocol', return_value=protocol), \
                patch.object(q11.q1_full, 'read_labels') as labels:
            with self.assertRaisesRegex(ValueError, 'pilot'):
                q11.verify_inputs()
            labels.assert_not_called()

    def test_incompatible_context_rejected(self):
        with patch.object(q11.q1_full, 'verify_protocol', return_value={'config': {'context_bp': 8192}}):
            with self.assertRaisesRegex(ValueError, 'context'):
                q11.verify_inputs()

    def test_short_crop_retains_variant_and_is_strand_invariant(self):
        ref = 'AACGTACGTGCA'
        alt = ref[:6] + 'T' + ref[7:]
        pair = q11.crop_pair(ref, alt, length=4)
        self.assertEqual([i for i, (r, a) in enumerate(zip(*pair)) if r != a], [2])
        self.assertEqual(pair, q11.crop_pair(q11.q1.reverse_complement(ref),
                                           q11.q1.reverse_complement(alt), length=4))
        with self.assertRaisesRegex(ValueError, 'single substitution'):
            q11.crop_pair(ref, ref, length=4)
        with patch.dict(q11.CONFIG, {'context_bp': 4}):
            dna = self.dna.assign(ref_sequence=ref, alt_sequence=alt)
            with self.assertRaisesRegex(ValueError, 'cross splits'):
                q11.audit_short_contexts(self.manifest, dna)

    def test_leakage_fails_before_labels(self):
        protocol = {'config': {'context_bp': 1024},
                    'vcf_exports': {'clinvar-train.vcf': 'a', 'clinvar-test.vcf': 'b'}}
        with patch.object(q11.q1_full, 'verify_protocol', return_value=protocol), \
                patch.object(q11.q1, 'read_csv'), \
                patch.object(q11.q1_full, 'audit', side_effect=AssertionError('Shared genes')), \
                patch.object(q11.q1_full, 'read_labels') as labels:
            with self.assertRaisesRegex(AssertionError, 'Shared genes'):
                q11.verify_inputs()
            labels.assert_not_called()


@unittest.skipIf(torch is None, 'Optimizer tests require the GPU profile PyTorch dependency')
class OptimizerTests(unittest.TestCase):
    def test_explicit_initialization_overwrites_uninitialized_adapter_matrices(self):
        adapter = torch.nn.Module()
        adapter.linear_in = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        adapter.linear_out = torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            adapter.linear_in.weight.fill_(float('nan'))
            adapter.linear_out.weight.fill_(float('nan'))
        torch.manual_seed(42)
        backend.initialize_adapter(adapter)
        initial = adapter.linear_in.weight.detach().clone()
        self.assertTrue(torch.isfinite(initial).all())
        self.assertTrue(torch.any(initial != 0))
        self.assertTrue(torch.all(adapter.linear_out.weight == 0))
        torch.manual_seed(42)
        backend.initialize_adapter(adapter)
        torch.testing.assert_close(initial, adapter.linear_in.weight, atol=0, rtol=0)

    def test_fixed_normalization_zero_difference_has_finite_gradients(self):
        reference = torch.tensor([[2., -2.]], requires_grad=True)
        difference = torch.zeros_like(reference, requires_grad=True)
        features = backend.normalize_pair(reference, difference)
        torch.testing.assert_close(features, torch.tensor([[1., -1., 0., 0.]]))
        features.sum().backward()
        self.assertTrue(torch.isfinite(reference.grad).all())
        self.assertTrue(torch.isfinite(difference.grad).all())

    def test_batched_sum_accumulation_matches_individual_updates(self):
        block, head, optimizer = self.model()
        other_block, other_head, other = self.model()
        x = torch.tensor([[.1, .4, .9], [.3, -.2, .7]])
        y = torch.tensor([1., 0.])
        self.update(block, head, optimizer, list(zip(x, y)))
        torch.nn.functional.binary_cross_entropy_with_logits(
            other_head(other_block(x)).flatten(), y, reduction='sum').backward()
        other.accumulate(len(x))
        other.step()
        for a, b in zip(list(block.parameters()) + list(head.parameters()),
                        list(other_block.parameters()) + list(other_head.parameters())):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        self.assertGreater(backend.validation_reserve(.05, 18000), 900)

    def test_fp32_masters_retain_updates_below_bf16_resolution(self):
        block = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            block.weight.fill_(1.)
        optimizer = backend.MasterAdamW(block, head, {**q11.CONFIG, 'weight_decay': 0., 'head_lr': 0.})
        for step in range(300):
            block.weight.grad = torch.full_like(block.weight, .01)
            head.weight.grad = torch.ones_like(head.weight)
            optimizer.accumulate()
            optimizer.step()
            if step == 0:
                self.assertEqual(float(block.weight.detach()), 1.)
                self.assertLess(float(optimizer.masters[0].detach()), 1.)
        self.assertLess(float(block.weight.detach()), 1.)

    def model(self):
        torch.manual_seed(8)
        block, head = torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)
        return block, head, backend.MasterAdamW(block, head)

    def update(self, block, head, optimizer, rows):
        for x, y in rows:
            torch.nn.functional.binary_cross_entropy_with_logits(head(block(x)).reshape(()), y).backward()
            optimizer.accumulate()
        optimizer.step()

    def test_accumulation_and_partial_batch_match_mean_loss(self):
        rows = [(torch.tensor([.1, .4, .9]), torch.tensor(1.)),
                (torch.tensor([.3, -.2, .7]), torch.tensor(0.))]
        block, head, optimizer = self.model()
        other_block, other_head, other = self.model()
        standard = torch.optim.AdamW([
            {'params': other_block.parameters(), 'lr': q11.CONFIG['adapter_lr']},
            {'params': other_head.parameters(), 'lr': q11.CONFIG['head_lr']}],
            weight_decay=.01, betas=(.9, .999), eps=1e-8)
        for batch in [rows, rows[:1]]:
            self.update(block, head, optimizer, batch)
            loss = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(
                other_head(other_block(x)).reshape(()), y) for x, y in batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(other_block.parameters()) + list(other_head.parameters()), 1.)
            standard.step()
            standard.zero_grad(set_to_none=True)
        for a, b in zip(list(block.parameters()) + list(head.parameters()),
                        list(other_block.parameters()) + list(other_head.parameters())):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_resume_reproduces_next_update_and_rejects_corruption(self):
        block, head, optimizer = self.model()
        rows = [(torch.tensor([.1, .4, .9]), torch.tensor(1.))]
        self.update(block, head, optimizer, rows)
        mean, scale = torch.zeros(2), torch.ones(2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(backend, 'rng_state', return_value={}), patch.object(backend, 'restore_rng'):
            path = Path(directory) / 'resume.pt'
            backend.save_checkpoint(path, block, head, optimizer, mean, scale, 'identity', {'offset': 8})
            state = backend.load_checkpoint(path, 'identity')
            restored_block, restored_head, restored_optimizer = self.model()
            backend.restore_checkpoint(state, restored_block, restored_head, restored_optimizer, mean, scale)
            self.update(block, head, optimizer, rows)
            self.update(restored_block, restored_head, restored_optimizer, rows)
            for a, b in zip(list(block.parameters()) + list(head.parameters()),
                            list(restored_block.parameters()) + list(restored_head.parameters())):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'another experiment'):
                backend.load_checkpoint(path, 'changed')
            with patch.dict(q11.CONFIG, {'lora': {**q11.CONFIG['lora'], 'rank': 16}}):
                with self.assertRaisesRegex(ValueError, 'LoRA configuration'):
                    backend.load_checkpoint(path, 'identity')
            with self.assertRaisesRegex(ValueError, 'scaler'):
                backend.restore_checkpoint(state, restored_block, restored_head, None, mean + 1, scale)
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                backend.load_checkpoint(path, 'identity')

    def test_save_refuses_partial_accumulation(self):
        block, head, optimizer = self.model()
        head(block(torch.ones(3))).sum().backward()
        optimizer.accumulate()
        with self.assertRaisesRegex(ValueError, 'optimizer boundary'):
            backend.save_checkpoint(Path('/unused'), block, head, optimizer, torch.zeros(2), torch.ones(2), 'x', {})

    def test_wrapper_hash_and_optimizer_exclude_original_backbone(self):
        model = torch.nn.Module()
        model.projection = torch.nn.Linear(2, 2).requires_grad_(False)
        before = backend.parameter_hash(model, frozen_only=True)
        wrapper = torch.nn.Module()
        wrapper.to_wrap = model.projection
        wrapper.adapter = torch.nn.Linear(2, 2)
        model.projection = wrapper
        self.assertEqual(before, backend.parameter_hash(model, frozen_only=True))
        head = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, 'adapter-only'):
            backend.MasterAdamW(model, head)
        optimizer = backend.MasterAdamW(wrapper.adapter, head)
        self.assertEqual({id(p) for p in optimizer.parameters}, {id(p) for p in wrapper.adapter.parameters()})
        with torch.no_grad():
            wrapper.adapter.weight.add_(1)
        self.assertEqual(before, backend.parameter_hash(model, frozen_only=True))
        with torch.no_grad():
            wrapper.to_wrap.weight.add_(1)
        self.assertNotEqual(before, backend.parameter_hash(model, frozen_only=True))

    def test_shuffle_and_boundary_validation(self):
        progress = {'order': np.random.default_rng(42).permutation(70).tolist(), 'offset': 32, 'epoch': 1}
        backend.validate_progress(progress, 70)
        backend.validate_progress({**progress, 'offset': 70}, 70)
        for invalid in [{'offset': 3}, {'order': list(range(70))}, {'epoch': 2}]:
            with self.assertRaises(ValueError):
                backend.validate_progress({**progress, **invalid}, 70)


class ResultTests(unittest.TestCase):
    def test_full_validation_intervals_and_incomplete_results(self):
        labels = np.array([0, 1] * 4)
        predictions = {'fine_tuned': 1 - labels * .8 - .1}
        result = q11.evaluate(labels, predictions, np.repeat(np.arange(4), 2), repetitions=20)
        self.assertEqual(result['fine_tuned']['auroc']['value'], 0)
        self.assertEqual(result['fine_tuned']['auroc']['ci95'], [0., 0.])
        with self.assertRaisesRegex(ValueError, 'Complete'):
            q11.evaluate(labels, {'fine_tuned': predictions['fine_tuned'][:2]}, np.arange(8), repetitions=20)

    def test_notebook_generator_has_short_calls_and_rejects_unexecuted(self):
        notebook = refresh_q11.build_notebook()
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == 'code']
        self.assertTrue(all(c.source.strip() and len(c.source.splitlines()) <= 2 for c in cells))
        self.assertIn('show_conclusion', cells[-1].source)
        with self.assertRaisesRegex(RuntimeError, 'every code cell'):
            refresh_q11.validate_execution(notebook)


if __name__ == '__main__':
    unittest.main()
