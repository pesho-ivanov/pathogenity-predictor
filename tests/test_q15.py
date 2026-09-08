"""CPU contracts for cached-adapter initialization, budget and scientific reporting."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from notebooks.src import q1, q12, q13, q14, q15, refresh_q15


def summary(auroc, ap):
    return {'auroc': {'value': auroc, 'ci95': [auroc - .01, auroc + .01]},
            'average_precision': {'value': ap, 'ci95': [ap - .01, ap + .01]}}


class Q15InitializationTests(unittest.TestCase):
    def test_resolves_each_strongest_frozen_checkpoint_and_prefers_q14_ties(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            for module, name in [(q12, 'q12'), (q13, 'q13'), (q14, 'q14')]:
                stack.enter_context(patch.object(module, 'OUTPUT', root / name))
                q1.write_json(module.OUTPUT / 'training_history.json', {
                    'best': {'best_control': {'auroc': .84, 'average_precision': .74}},
                    'adapter_winner': {'best_control': {'auroc': .82, 'average_precision': .71},
                                       'best_lora': {'auroc': 1., 'average_precision': 1.}}})
            for winner, expected_question in [('best_control', 'q14'), ('q13_control', 'q13'), ('q12_control', 'q12')]:
                with self.subTest(winner=winner):
                    parent = {'metrics': {name: summary(.84, .74) if name == winner else summary(.82, .71)
                                         for name in ['best_control', 'q13_control', 'q12_control']}}
                    question, source, path, metrics = q15.choose_initial_control(parent)
                    self.assertEqual((question, source), (expected_question, winner))
                    folder = root / question
                    expected = folder / 'best_control.pt' if question == 'q12' else folder / 'best/best_control.pt'
                    self.assertEqual(path, expected)
                    self.assertEqual(metrics, parent['metrics'][winner])
            parent = {'metrics': {name: summary(.84, .74) for name in ['best_control', 'q13_control', 'q12_control']}}
            self.assertEqual(q15.choose_initial_control(parent)[:2], ('q14', 'best_control'))
            parent['metrics']['best_control'] = summary(.85, .74)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                q15.choose_initial_control(parent)

    def test_learning_rate_comes_from_verified_winning_trial_not_its_name(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(q14, 'OUTPUT', Path(temporary)):
            history = {'a_name_that_does_not_encode_lr': {'adapter_lr': 1e-4,
                'best_lora': {'auroc': .84, 'average_precision': .74}}}
            parent = {'best_lora_trial': 'a_name_that_does_not_encode_lr', 'metrics': {'lora': summary(.84, .74)}}
            q1.write_json(q14.OUTPUT / 'training_history.json', history)
            self.assertEqual(q15.inherited_learning_rate(parent), ('a_name_that_does_not_encode_lr', 1e-4))
            for corruption in ['score', 'learning_rate', 'missing_trial']:
                with self.subTest(corruption=corruption):
                    changed = deepcopy(history)
                    if corruption == 'score':
                        changed[parent['best_lora_trial']]['best_lora']['auroc'] = .9
                    elif corruption == 'learning_rate':
                        changed[parent['best_lora_trial']]['adapter_lr'] = -1.
                    else:
                        changed = {}
                    q1.write_json(q14.OUTPUT / 'training_history.json', changed)
                    with self.assertRaises(ValueError):
                        q15.inherited_learning_rate(parent)

    def test_promotion_preserves_the_strongest_inherited_control(self):
        protocol = {'initial_control_metrics': summary(.835, .74)}
        weaker = {'auroc': .83, 'average_precision': .73}
        strongest = q15.promotion_control(weaker, protocol)
        self.assertEqual(strongest, {'auroc': .835, 'average_precision': .74})
        self.assertFalse(q15.promote({'auroc': .838, 'average_precision': .75}, strongest))
        self.assertTrue(q15.promote({'auroc': .84, 'average_precision': .74}, strongest))
        self.assertFalse(q15.promote({'auroc': .85, 'average_precision': .739}, strongest))
        fresh = {'auroc': .84, 'average_precision': .75}
        self.assertEqual(q15.promotion_control(fresh, protocol), fresh)

    def test_deadline_reserves_both_arms_without_resetting_elapsed_work(self):
        budget = q15.Budget(started=0.)
        with patch.object(q15.time, 'time', return_value=3500.):
            self.assertEqual(budget.elapsed(), 3500.)
            self.assertEqual(budget.remaining(), 100.)
            self.assertTrue(budget.can_work(49., reserve=50.))
            self.assertFalse(budget.can_work(50., reserve=50.))
            with self.assertRaises(ValueError):
                budget.can_work(-1.)
        with patch.object(q15.time, 'time', return_value=3601.):
            self.assertEqual(budget.remaining(), 0.)
            self.assertFalse(budget.can_work(0.))

    def test_planned_exposures_and_notebook_do_not_depend_on_completed_q14_files(self):
        expected = q15.CONFIG['training_variants'] * q15.CONFIG['passes_per_chunk'] // q15.CONFIG['microbatch_variants']
        self.assertEqual(expected, q15.CONFIG['max_attempted_steps'])
        self.assertEqual(q15.CONFIG['head_lr'], 0.)
        with patch.object(q14, 'verified_results', side_effect=AssertionError('Q14 may still be running')):
            notebook = refresh_q15.build_notebook()
        nbformat.validate(notebook)
        self.assertEqual(len([cell for cell in notebook.cells if cell.cell_type == 'code']), 5)
        self.assertTrue(all(cell.source.strip() for cell in notebook.cells))
        self.assertTrue(all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == 'code'))


class Q15CompletionTests(unittest.TestCase):
    def test_completed_result_requires_replay_fixed_head_and_matched_training_checks(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(q15, 'OUTPUT', Path(temporary)), \
                patch.object(q15, 'verify_protocol', return_value='fixture'):
            artifact = q15.OUTPUT / 'fixture.txt'
            artifact.write_text('fixture checkpoint evidence')
            q1.write_json(q15.OUTPUT / 'protocol.json', {'initial_control_metrics': summary(.835, .74)})
            result = {'status': 'complete', 'scope': 'sampled_validation', 'identity': 'fixture',
                'reloaded_predictions_verified': True, 'frozen_unchanged': True,
                'head_fixed_verified': True, 'scaler_fixed_verified': True, 'control_optimizer_steps': 0,
                'prefix_replay_verified': True, 'replay_gradient_verified': True, 'matched_training_verified': True,
                'artifacts': {'fixture.txt': q1.digest_file(artifact)},
                'metrics': {name: summary(.83, .73) for name in
                            ['weighted_bce', 'pairwise_logistic', 'lora', 'matched_control', 'best_control', 'strongest_control', 'selected']},
                'promotion_passed': True, 'selected_model': 'lora'}
            for name in ['weighted_bce', 'lora', 'selected']:
                result['metrics'][name] = summary(.841, .75)
            q1.write_json(q15.OUTPUT / 'metrics.json', result)
            q15.verified_results()
            for key in ['reloaded_predictions_verified', 'frozen_unchanged', 'head_fixed_verified',
                        'scaler_fixed_verified', 'prefix_replay_verified', 'replay_gradient_verified', 'matched_training_verified']:
                with self.subTest(missing=key):
                    changed = dict(result, **{key: False})
                    q1.write_json(q15.OUTPUT / 'metrics.json', changed)
                    with self.assertRaises(ValueError):
                        q15.verified_results()
            changed = deepcopy(result)
            for name in ['weighted_bce', 'lora', 'selected']:
                changed['metrics'][name] = summary(.838, .75)
            q1.write_json(q15.OUTPUT / 'metrics.json', changed)
            with self.assertRaisesRegex(ValueError, 'promotion'):
                q15.verified_results()


if __name__ == '__main__':
    unittest.main()
