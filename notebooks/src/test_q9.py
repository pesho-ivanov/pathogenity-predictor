"""Q9 must reject changed frozen inputs and annotation-bearing model inputs."""

import unittest
from unittest.mock import patch

import pandas as pd

try:
    from . import q9
except ImportError:
    from notebooks.src import q9


class InputBoundaryTests(unittest.TestCase):
    def test_changed_vcf_is_rejected_before_reading_labels(self):
        with patch.object(q9.q1, 'verify_protocol', return_value={'vcf_exports': {}}), \
                patch.object(q9.q1, 'load_partition_labels') as labels:
            with self.assertRaisesRegex(ValueError, 'approved'):
                q9.verify_inputs()
            labels.assert_not_called()

    def test_changed_context_is_rejected(self):
        protocol = {'vcf_exports': q9.PILOT_SHA256, 'config': {'context_bp': 8192}}
        with patch.object(q9.q1, 'verify_protocol', return_value=protocol):
            with self.assertRaisesRegex(ValueError, 'context'):
                q9.verify_inputs()

    def test_model_inputs_reject_labels_annotations_and_reordered_variants(self):
        dna = pd.DataFrame({'variant_key': ['v1', 'v2'],
                            'ref_sequence': ['ACGT', 'AAAA'],
                            'alt_sequence': ['ATGT', 'AACA']})
        q9.assert_feature_inputs(dna, dna)
        for name in ['label', 'CLNSIG', 'MC', 'gene_ids']:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'DNA-only'):
                q9.assert_feature_inputs(dna.assign(**{name: 'annotation'}), dna)
        with self.assertRaisesRegex(ValueError, 'DNA-only'):
            q9.assert_feature_inputs(dna.iloc[::-1], dna)

    def test_changed_sequence_is_rejected(self):
        dna = pd.DataFrame({'variant_key': ['v1'], 'ref_sequence': ['ACGT'], 'alt_sequence': ['ATGT']})
        with self.assertRaisesRegex(ValueError, 'DNA-only'):
            q9.assert_feature_inputs(dna.assign(alt_sequence='AGGT'), dna)

    def test_changed_experiment_identity_rejects_cached_artifacts(self):
        identity = {'checkpoint': 'a', 'split': 'b', 'code': 'c'}
        for name in identity:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'identity changed'):
                q9.assert_identity({**identity, name: 'changed'}, identity)

    def test_split_overlap_stops_before_any_label_loading(self):
        protocol = {'vcf_exports': q9.PILOT_SHA256, 'config': {'context_bp': 1024}}
        with patch.object(q9.q1, 'verify_protocol', return_value=protocol), \
                patch.object(q9.q1, 'read_csv'), \
                patch.object(q9.q1, 'audit_splits', side_effect=AssertionError('Shared genes')), \
                patch.object(q9.q1, 'load_partition_labels') as labels:
            with self.assertRaisesRegex(AssertionError, 'Shared genes'):
                q9.verify_inputs()
            labels.assert_not_called()

    def test_nonmissense_stops_before_label_loading(self):
        protocol = {'vcf_exports': q9.PILOT_SHA256, 'config': {'context_bp': 1024}}
        with patch.object(q9.q1, 'verify_protocol', return_value=protocol), \
                patch.object(q9.q1, 'read_csv'), patch.object(q9.q1, 'audit_splits', return_value={}), \
                patch.object(q9.q1, 'audit_missense_scope', side_effect=AssertionError('nonmissense')), \
                patch.object(q9.q1, 'load_partition_labels') as labels:
            with self.assertRaisesRegex(AssertionError, 'nonmissense'):
                q9.verify_inputs()
            labels.assert_not_called()

    def test_failed_compatibility_never_launches_training(self):
        from . import q9_environment
        result = {'identity': 'frozen', 'status': 'blocked', 'blockers': ['Backends disagree']}
        with patch.object(q9.q1, 'read_json', return_value=result), \
                patch.object(q9, 'protocol_identity', return_value='frozen'), \
                patch.object(q9_environment, 'run_command') as launch:
            self.assertEqual(q9.run_experiment()['status'], 'blocked')
            with self.assertRaisesRegex(RuntimeError, 'Training blocked'):
                result['artifacts'] = {}
                q9.require_ready()
            launch.assert_not_called()


class ExecutionTests(unittest.TestCase):
    def test_second_training_run_cannot_replace_active_log(self):
        import fcntl
        import tempfile
        from pathlib import Path
        from . import q9_environment
        with tempfile.TemporaryDirectory() as directory, patch.object(q9_environment, 'OUTPUT', Path(directory)):
            log = Path(directory) / 'experiment.log'
            log.write_text('active training')
            with (Path(directory) / 'experiment.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(q9_environment, '_run_command') as launch:
                    with self.assertRaisesRegex(RuntimeError, 'already running'):
                        q9_environment.run_command('experiment', ['unused'])
                    launch.assert_not_called()
            self.assertEqual(log.read_text(), 'active training')

    def test_subprocess_log_is_preserved_on_rerun(self):
        import sys
        import tempfile
        from pathlib import Path
        from . import q9_environment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'environment'
            output.mkdir()
            (output / 'probe.log').write_text('original error')
            with patch.object(q9_environment, 'ROOT', root), patch.object(q9_environment, 'OUTPUT', output):
                log = q9_environment.run_command('probe', [sys.executable, '-c', 'print("successful rerun")'])
            self.assertIn('successful rerun', log.read_text())
            self.assertEqual([p.read_text() for p in (output / 'logs').glob('*.log')], ['original error'])


class NumericalProtocolTests(unittest.TestCase):
    def test_legacy_resume_rejects_changed_inputs_backbone_and_cache(self):
        import copy
        import tempfile
        from pathlib import Path
        from . import q1, q9_experiment
        with tempfile.TemporaryDirectory() as directory, patch.object(q9_experiment, 'OUTPUT', Path(directory)):
            root = Path(directory)
            old = {'configuration': {'seed': 42}, 'parent_protocol_sha256': 'fixed-split',
                   'sources': {'notebooks/src/q9_experiment.py': 'before', 'notebooks/src/q9_backend.py': 'fixed'}}
            current = copy.deepcopy(old)
            current['sources']['notebooks/src/q9_experiment.py'] = 'resume-fix'
            record = {'identity': q1.fingerprint(old)}
            archive = root / 'archive/old'
            q1.write_json(archive / 'protocol.json', old)
            q1.write_json(root / 'cache.json', record)
            migration = {'archive': 'archive/old', 'from_identity': record['identity'],
                         'artifacts': {'protocol.json': q1.digest_file(archive / 'protocol.json'),
                                       'cache.json': q1.digest_file(root / 'cache.json')}}
            def verify(protocol):
                identity = q1.fingerprint(protocol)
                q1.write_json(root / 'protocol.json', protocol)
                q1.write_json(root / 'resume_migration.json', {**migration, 'to_identity': identity})
                q9_experiment.verify_artifact_identity(record, identity, root / 'cache.json')
            verify(current)
            changed = copy.deepcopy(current)
            changed['parent_protocol_sha256'] = 'different-split'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                verify(changed)
            changed = copy.deepcopy(current)
            changed['sources']['notebooks/src/q9_backend.py'] = 'different-backbone'
            with self.assertRaisesRegex(ValueError, 'model, feature or input code'):
                verify(changed)
            (root / 'cache.json').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                verify(current)

    def test_resume_reproduces_the_next_adam_update(self):
        import tempfile
        from pathlib import Path
        import torch
        from .q9_backend import MasterAdamW
        from .q9_experiment import save_checkpoint, restore_adapter
        torch.manual_seed(42)
        block = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        head = torch.nn.Linear(2, 1)
        optimizer = MasterAdamW(block, head)
        mean, scale = torch.zeros(2), torch.ones(2)
        def step():
            score = head(block(torch.tensor([[.5, -.25]], dtype=torch.bfloat16)).float())
            (score-1).square().mean().backward()
            optimizer.accumulate()
            optimizer.step()
        for _ in range(3):
            step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_checkpoint(path, block, head, optimizer, mean, scale, 'identity', 1,
                            {'history': [{'epoch': 0, 'auroc': .6}, {'epoch': 1, 'auroc': .5}]})
            saved = torch.load(path, weights_only=True)
            step()
            expected = [p.detach().clone() for p in list(block.parameters())+list(head.parameters())+optimizer.masters]
            restore_adapter(saved, block, head, optimizer, mean, scale)
            step()
            for actual, wanted in zip(list(block.parameters())+list(head.parameters())+optimizer.masters, expected):
                self.assertTrue(torch.equal(actual, wanted))
            with self.assertRaisesRegex(ValueError, 'scaler changed'):
                restore_adapter(saved, block, head, optimizer, mean+1, scale)

    def test_resume_preserves_early_stopping_and_tie_breaking(self):
        from .q9_experiment import early_stopping_state
        self.assertEqual(early_stopping_state([{'epoch': i, 'auroc': a}
                                              for i, a in enumerate([.7, .6, .7])]), (.7, 0, 2))
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            early_stopping_state([{'epoch': 0, 'auroc': .7}, {'epoch': 2, 'auroc': .8}])

    def test_selected_hyena_block_learns_through_frozen_final_block(self):
        import torch
        from .q9_backend import select_trainable_block
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([
            torch.nn.Linear(1, 1, bias=False) for _ in range(25)])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.)
        block = select_trainable_block(model)
        self.assertIs(block, model.decoder.layers[23])
        value = torch.ones(1, 1)
        for layer in model.decoder.layers:
            value = layer(value)
        value.sum().backward()
        self.assertGreater(float(block.weight.grad.abs().sum()), 0.)
        for i, layer in enumerate(model.decoder.layers):
            if i != 23:
                self.assertFalse(layer.weight.requires_grad)
                self.assertIsNone(layer.weight.grad)

    def test_saved_adapter_identifies_the_selected_block(self):
        import tempfile
        from pathlib import Path
        import torch
        from .q9_backend import MasterAdamW
        from .q9_experiment import save_checkpoint
        block = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        head = torch.nn.Linear(1, 1)
        optimizer = MasterAdamW(block, head)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            save_checkpoint(path, block, head, optimizer, torch.zeros(1), torch.ones(1), 'identity', 0)
            state = torch.load(path, weights_only=True)
            self.assertEqual(state['block_index'], 23)
            self.assertEqual(state['identity'], 'identity')

    def test_trace_handles_tuple_inputs_and_preserves_input_before_mutation(self):
        import torch
        from . import q9_backend, q9_diagnostics
        class Block(torch.nn.Module):
            def forward(self, value):
                return value[0].add_(1)
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([Block()])
        def encode(model, tokenizer, sequence):
            return model.decoder.layers[0]((torch.ones(2, 1, 3), None))
        with patch.object(q9_backend, 'encode', side_effect=encode):
            measured = q9_diagnostics.trace(model, None, 'ACGT')['decoder.layers.0']
        self.assertEqual(measured['input']['rms'], 1.)
        self.assertEqual(measured['output']['rms'], 2.)
        self.assertEqual(measured['equal_fraction'], 0.)

    def test_native_readiness_does_not_require_vortex_agreement(self):
        self.assertEqual(q9.readiness_blockers({'passed': True}, {'status': 'passed'}), [])

    def test_native_conversion_and_gradient_failures_still_block_training(self):
        blockers = q9.readiness_blockers({'passed': False}, {'status': 'blocked', 'blocker': 'No update'})
        self.assertEqual(len(blockers), 2)
        self.assertIn('No update', blockers)

    def test_paired_intervals_are_zero_for_identical_predictors(self):
        import numpy as np
        from .q9_experiment import paired_intervals
        labels = np.tile([0, 1], 4)
        predictions = np.array([.2, .7, .3, .9, .6, .8, .1, .5])
        intervals = paired_intervals(labels, {'fine_tuned': predictions, 'frozen': predictions.copy()},
                                     np.repeat(np.arange(4), 2), repetitions=25)
        for metric in intervals['fine_tuned-minus-frozen'].values():
            self.assertEqual(metric['95_percent_interval'], [0., 0.])
            self.assertEqual(metric['valid_repetitions'], 25)

    def test_fp32_masters_preserve_updates_below_bf16_resolution(self):
        import torch
        from .q9_backend import MasterAdamW
        block = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            block.weight.fill_(1.)
        optimizer = MasterAdamW(block, head)
        for step in range(250):
            block.weight.grad = torch.ones_like(block.weight)
            for parameter in head.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.accumulate()
            optimizer.step()
            if step == 0:
                self.assertEqual(float(block.weight.detach()), 1.)
                self.assertLess(float(optimizer.masters[0].detach()), 1.)
        self.assertLess(float(block.weight.detach()), 1.)


if __name__ == '__main__':
    unittest.main()
