"""CPU regression for inherited frozen controls and renamed Q14 adapter states."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
    from notebooks.src import q14, q14_backend as backend
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for checkpoint compatibility tests')
class InheritedControlTests(unittest.TestCase):
    def test_parent_control_selection_uses_frozen_head_and_new_zero_adapter_keys(self):
        for winner in ('q13_control', 'q12_control'):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                for module, name in [(q14, 'q14'), (backend.q13, 'q13'), (backend.q12, 'q12')]:
                    output = root / name
                    output.mkdir()
                    stack.enter_context(patch.object(module, 'OUTPUT', output))
                stack.enter_context(patch.object(q14, 'CONFIG', dict(q14.CONFIG, feature_dimension=2, reload_variants=4)))
                stack.enter_context(patch.object(backend.q13, 'verify_protocol', return_value='parent13'))
                stack.enter_context(patch.object(backend.q12, 'verify_protocol', return_value='parent12'))
                identity = 'q14-test'
                x = torch.tensor([[0., 0.], [1., 2.], [2., 1.], [3., 3.]])
                labels = np.array([0, 0, 1, 1])
                keys = ['a', 'b', 'c', 'd']
                groups = np.array(['g1', 'g1', 'g2', 'g2'])
                good, bad, lora = x[:, 0].numpy(), -x[:, 0].numpy(), x[:, 1].numpy()
                parents = {'q13_control': good if winner == 'q13_control' else bad,
                           'q12_control': good if winner == 'q12_control' else bad}
                for name, module in [('q13_control', backend.q13), ('q12_control', backend.q12)]:
                    # A parent's selected model can differ from its frozen control.
                    # Q14 must consume the explicit best_control column instead.
                    frame = pd.DataFrame({'variant_key': keys, 'label': labels,
                                          'best_control': parents[name], 'selected': -parents[name]})
                    frame.to_csv(module.OUTPUT / 'validation_predictions.csv', index=False)

                def head(weight):
                    return {'weight': torch.tensor([weight], dtype=torch.float32), 'bias': torch.zeros(1)}

                keys_new = ['block29_dense_projection', 'block29_dense', 'block30_dense_projection', 'block30_dense']
                adapters = torch.nn.ModuleDict({name: torch.nn.Linear(1, 1, bias=False) for name in keys_new})
                initial = {name: torch.zeros_like(p) for name, p in adapters.named_parameters()}
                trained = {name: torch.ones_like(p) for name, p in adapters.named_parameters()}
                common = {'format': 'q14-controlled-v1', 'identity': identity,
                          'mean': torch.zeros(2), 'scale': torch.ones(2), 'strength': .01,
                          'progress': {'steps': 256}}
                state = dict(common, adapters=trained, head=head([0., 1.]), control=head([-1., 0.]),
                             lora_scores=torch.from_numpy(lora), control_scores=torch.from_numpy(bad))
                trial = q14.OUTPUT / 'trial'
                trial.mkdir()
                backend.save_state(trial / 'best_lora.pt', state)
                backend.save_state(trial / 'best_control.pt', state)
                histories = {'trial': {'best_lora': q14.metric(labels, lora),
                                       'best_control': q14.metric(labels, bad), 'control_steps': 0,
                                       'head_fixed_verified': True, 'scaler_fixed_verified': True}}
                old_history = {'parent_trial': {'best_control': q14.metric(labels, parents['q13_control'])}}
                backend.q1.write_json(backend.q13.OUTPUT / 'training_history.json', old_history)

                def inherited(parent):
                    sign = 1. if winner == parent else -1.
                    # Old adapters deliberately use incompatible single-block keys;
                    # their adapted head must also be replaced by the control head.
                    return dict(common, format='parent-format', identity='parent-identity',
                                adapters={'dense.linear_out.weight': torch.ones(1, 1)},
                                head=head([-sign, 0.]), control=head([sign, 0.]))

                old13 = stack.enter_context(patch.object(backend.q13_backend, 'load_state', return_value=inherited('q13_control')))
                old12 = stack.enter_context(patch.object(backend.previous, 'load_state', return_value=inherited('q12_control')))
                stack.enter_context(patch.object(backend.previous, 'new_head', side_effect=lambda dim: torch.nn.Linear(dim, 1)))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor))
                stack.enter_context(patch.object(backend.base, 'parameter_hash', return_value='frozen'))
                stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
                stack.enter_context(patch.object(backend.q1, 'digest_file', return_value='fixture-sha'))

                def predict(model, tokenizer, fitted, mean, scale, rows):
                    with torch.no_grad():
                        return fitted((x[:len(rows)] - mean) / scale).flatten().numpy()

                def intervals(targets, scores, components):
                    return ({name: {metric: {'value': value, 'ci95': [value, value]}
                                    for metric, value in q14.metric(targets, values).items()}
                             for name, values in scores.items()}, {})

                stack.enter_context(patch.object(backend, 'predict', side_effect=predict))
                stack.enter_context(patch.object(q14, 'intervals', side_effect=intervals))
                rows = {'validation': pd.DataFrame({'variant_key': keys}), 'train': pd.DataFrame(index=range(8))}
                with redirect_stdout(io.StringIO()):
                    backend.finish(None, None, adapters, initial, rows, {'validation': labels},
                                   {'validation': groups}, histories, identity,
                                   SimpleNamespace(elapsed=lambda: 10.), 'frozen')
                selected = backend.load_state(q14.OUTPUT / 'selected_model.pt', identity)
                self.assertEqual(selected['selected_model'], winner)
                self.assertEqual(set(selected['adapters']), set(initial))
                self.assertTrue(all(torch.count_nonzero(value) == 0 for value in selected['adapters'].values()))
                torch.testing.assert_close(selected['head']['weight'], torch.tensor([[1., 0.]]), atol=0, rtol=0)
                result = pd.read_csv(q14.OUTPUT / 'validation_predictions.csv')
                np.testing.assert_array_equal(result['selected'], good)
                loader = old13 if winner == 'q13_control' else old12
                self.assertEqual(loader.call_args.args[0].name, 'best_control.pt')


if __name__ == '__main__':
    unittest.main()
