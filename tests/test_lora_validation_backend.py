"""CPU checks for full-validation model resolution, shared inference and reloads."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
    from notebooks.src import lora_validation_backend as backend
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for validation backend tests')
class FullValidationBackendTests(unittest.TestCase):
    def test_best_control_trial_ignores_adapter_winner_and_uses_ap_to_break_ties(self):
        history = {
            'adapter_winner': {'best_lora': {'auroc': 1., 'average_precision': 1.},
                               'best_control': {'auroc': .7, 'average_precision': .6}},
            'first_control': {'best_control': {'auroc': .8, 'average_precision': .65}},
            'best_control': {'best_control': {'auroc': .8, 'average_precision': .7}},
        }
        self.assertEqual(backend.best_control_trial(history), 'best_control')
        for invalid in [{}, {'bad': {'best_control': {'auroc': float('nan'), 'average_precision': .6}}}]:
            with self.assertRaises(ValueError):
                backend.best_control_trial(invalid)

    def test_control_resolver_loads_best_control_checkpoint_for_each_parent(self):
        from notebooks.src import q15, q15_backend
        cases = [('q13', 'best_control', backend.q13, backend.q13_backend, True),
                 ('q14', 'best_control', backend.q14, backend.q14_backend, True),
                 ('q14', 'q13_control', backend.q13, backend.q13_backend, True),
                 ('q14', 'q12_control', backend.q12, backend.previous, False),
                 ('q15', 'best_control', q15, q15_backend, True),
                 ('q15', 'q14_control', backend.q14, backend.q14_backend, True),
                 ('q15', 'q13_control', backend.q13, backend.q13_backend, True),
                 ('q15', 'q12_control', backend.q12, backend.previous, False)]
        for question, source, module, loader, has_trials in cases:
            with self.subTest(question=question, source=source), ExitStack() as stack:
                folder = Path('/fixture') / module.__name__.split('.')[-1]
                stack.enter_context(patch.object(module, 'OUTPUT', folder))
                verified = stack.enter_context(patch.object(module, 'verified_results', return_value={}))
                stack.enter_context(patch.object(module, 'verify_protocol', return_value='identity'))
                stack.enter_context(patch.object(backend.q1, 'read_json', return_value={
                    'winner': {'best_control': {'auroc': .9, 'average_precision': .8}},
                    'loser': {'best_control': {'auroc': .8, 'average_precision': .7}}}))
                state = {'control': {'fixture': True}}
                load = stack.enter_context(patch.object(loader, 'load_state', return_value=state))
                actual, path = backend.load_control(question, source)
                expected = folder / 'winner' / 'best_control.pt' if has_trials else folder / 'best_control.pt'
                self.assertIs(actual, state)
                self.assertEqual(path, expected)
                load.assert_called_once_with(expected, 'identity')
                verified.assert_called_once_with()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            backend.load_control('q14', 'selected_model')

    def test_q15_selected_loader_and_nested_history_preserve_checkpoint_identity(self):
        from notebooks.src import q15, q15_backend
        with ExitStack() as stack:
            stack.enter_context(patch.object(q15, 'OUTPUT', Path('/fixture/q15')))
            stack.enter_context(patch.object(q15, 'verify_protocol', return_value='q15-identity'))
            result = {'promotion_passed': True, 'selected_model': 'lora'}
            stack.enter_context(patch.object(q15, 'verified_results', return_value=result))
            state = {'selected_model': 'lora', 'control': {'fixed': True}}
            load = stack.enter_context(patch.object(q15_backend, 'load_state', return_value=state))
            selected, path, actual = backend.load_selected('q15')
            self.assertIs(selected, state)
            self.assertIs(actual, result)
            load.assert_called_once_with(Path('/fixture/q15/selected_model.pt'), 'q15-identity')
            stack.enter_context(patch.object(backend.q1, 'read_json', return_value={
                'trials': {'weighted_bce': {'best_control': {'auroc': .83, 'average_precision': .72}},
                           'pairwise_logistic': {'best_control': {'auroc': .83, 'average_precision': .72}}},
                'attempted_steps': 2304, 'completed_chunks': 6}))
            selected, path = backend.load_control('q15', 'best_control')
            self.assertEqual(path, Path('/fixture/q15/weighted_bce/best_control.pt'))
            self.assertEqual(load.call_args.args[1], 'q15-identity')

    def test_q14_validation_does_not_import_or_hash_optional_q15_code(self):
        import builtins
        from notebooks.src import refresh_lora_validation
        original = builtins.__import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            optional = {'q15', 'q15_backend', 'q15_replay', 'q15_pooling', 'q15_objective'}
            if name.split('.')[-1] in optional or optional.intersection(fromlist or ()):
                raise AssertionError('Q14 validation must not import optional Q15 modules')
            return original(name, globals, locals, fromlist, level)

        with patch('builtins.__import__', side_effect=guarded):
            self.assertIs(backend.validation.parent('q14'), backend.q14)
            self.assertIs(backend.checkpoint_loader('q14'), backend.q14_backend.load_state)
            sources = backend.validation.sources('q14')
            self.assertFalse(any(Path(name).name.startswith('q15') for name in sources))
            notebook = refresh_lora_validation.build_notebook('q14')
            self.assertIn('Q14-layer-adapters.ipynb', notebook.cells[0].source)

    def state(self, weight=1., mean=0., scale=1.):
        dimension = backend.q12.CONFIG['feature_dimension']
        coefficients = torch.zeros(1, dimension)
        coefficients[0, 0] = weight
        return {'control': {'weight': coefficients, 'bias': torch.zeros(1)},
                'mean': torch.full((dimension,), mean), 'scale': torch.full((dimension,), scale)}

    def test_reloading_clones_scaler_and_erasure_does_not_modify_saved_checkpoint(self):
        state = self.state(mean=1., scale=2.)
        spec = backend.head_spec(state, 'control', device='cpu')
        adapters = torch.nn.ModuleDict({'fixture': torch.nn.Linear(1, 1, bias=False)})
        backend.erase(adapters, {'control': spec})
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in adapters.parameters()))
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in spec['head'].parameters()))
        self.assertEqual(torch.count_nonzero(spec['mean']), 0)
        self.assertEqual(torch.count_nonzero(spec['scale']), 0)
        self.assertTrue(torch.all(state['mean'] == 1.))
        self.assertTrue(torch.all(state['scale'] == 2.))
        self.assertEqual(state['control']['weight'][0, 0], 1.)
        restored = backend.head_spec(state, 'control', device='cpu')
        self.assertEqual(restored['head'].weight[0, 0], 1.)
        for key, value in [('scale', 0.), ('mean', float('nan'))]:
            invalid = self.state()
            invalid[key][0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                backend.head_spec(invalid, 'control', device='cpu')

    def test_one_frozen_feature_pass_scores_heads_with_their_own_scalers(self):
        rows = pd.DataFrame({'variant_key': list('abcde'), 'number': np.arange(5)})
        specs = {'matched_control': backend.head_spec(self.state(), 'control', device='cpu'),
                 'strongest_control': backend.head_spec(self.state(weight=2., mean=1., scale=2.),
                                                        'control', device='cpu')}

        def features(model, tokenizer, batch):
            raw = torch.zeros(len(batch), backend.q12.CONFIG['feature_dimension'])
            raw[:, 0] = torch.tensor(batch.number.to_numpy(), dtype=torch.float32)
            return raw

        with patch.object(backend.validation, 'CONFIG', dict(backend.validation.CONFIG, microbatch_variants=2)), \
                patch.object(backend.previous, 'features', side_effect=features) as extract, redirect_stdout(io.StringIO()):
            scores = backend.predict(None, None, rows, specs, 'fixture')
        self.assertEqual(extract.call_count, 3)
        np.testing.assert_array_equal(scores['matched_control'], np.arange(5))
        np.testing.assert_array_equal(scores['strongest_control'], np.arange(5) - 1)

    def test_unavailable_subsets_are_reported_without_bootstrapping(self):
        labels, groups = np.array([0, 1, 1]), np.array(['a', 'b', 'c'])
        scores = {name: np.array([0., 1., 2.]) for name in ('lora', 'matched_control', 'strongest_control')}
        with patch.object(backend.validation, 'intervals') as intervals:
            result = backend.subset_result(np.array([False, True, True]), labels, scores, groups)
        self.assertEqual(result['status'], 'not_available')
        self.assertEqual(result['variants'], 2)
        self.assertEqual(result['components'], 2)
        self.assertNotIn('metrics', result)
        intervals.assert_not_called()

    def test_sample_score_check_rejects_misassociated_parent_metrics(self):
        labels = np.array([0, 1, 0, 1])
        state = {'control_scores': torch.tensor([0., 1., 0., 1.])}
        expected = {name: {'value': value} for name, value in
                    backend.q12.metric(labels, state['control_scores'].numpy()).items()}
        backend.verify_sample_scores(state, 'control_scores', labels, expected)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            backend.verify_sample_scores(state, 'control_scores', 1 - labels, expected)


if __name__ == '__main__':
    unittest.main()
