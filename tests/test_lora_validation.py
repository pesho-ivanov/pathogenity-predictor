"""CPU contracts for full LoRA confirmation, cohort integrity and frozen baselines."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import nbformat

from notebooks.src import lora_validation as validation, q1, q12, q13, q14


def summary(auroc, ap):
    return {'auroc': {'value': auroc, 'ci95': [auroc - .01, auroc + .01]},
            'average_precision': {'value': ap, 'ci95': [ap - .01, ap + .01]}}


class ConfirmationRuleTests(unittest.TestCase):
    def candidate(self):
        return {'metrics': {'lora': summary(.84, .75), 'strongest_control': summary(.835, .75)},
                'paired_lora_minus_strongest_control': {'auroc': {'value': .005, 'ci95': [.001, .009]}},
                'subsets': {'outside_selection': {'metrics': {
                    'lora': summary(.841, .751), 'strongest_control': summary(.835, .75)}}}}

    def test_accepts_threshold_auroc_and_equal_ap(self):
        result = self.candidate()
        result['subsets']['outside_selection']['metrics']['lora']['average_precision']['value'] = .75
        self.assertTrue(validation.confirmation_passes(result))

    def test_each_required_gain_and_interval_boundary_is_enforced(self):
        for name in ['full_auroc', 'full_ap', 'interval', 'outside_auroc', 'outside_ap', 'outside_missing']:
            with self.subTest(condition=name):
                result = self.candidate()
                if name == 'full_auroc':
                    result['metrics']['lora']['auroc']['value'] = .839
                elif name == 'full_ap':
                    result['metrics']['lora']['average_precision']['value'] = .749
                elif name == 'interval':
                    result['paired_lora_minus_strongest_control']['auroc']['ci95'][0] = 0.
                elif name == 'outside_auroc':
                    result['subsets']['outside_selection']['metrics']['lora']['auroc']['value'] = .835
                elif name == 'outside_ap':
                    result['subsets']['outside_selection']['metrics']['lora']['average_precision']['value'] = .749
                else:
                    result['subsets']['outside_selection'] = {'status': 'not_available'}
                self.assertFalse(validation.confirmation_passes(result))

    def test_bootstrap_pairs_against_strongest_without_renaming_the_actual_control(self):
        labels, groups = np.tile([0, 1], 8), np.repeat(np.arange(8), 2)
        scores = {'lora': np.tile([.1, .9], 8), 'matched_control': np.tile([.8, .2], 8),
                  'strongest_control': np.tile([.4, .6], 8)}
        with patch.dict(validation.CONFIG, bootstrap_repetitions=20):
            metrics, paired = validation.intervals(labels, scores, groups)
        self.assertEqual(set(metrics), set(scores))
        self.assertEqual(metrics['matched_control']['auroc']['value'], 0.)
        self.assertEqual(metrics['strongest_control']['auroc']['value'], 1.)
        self.assertEqual(paired['auroc']['value'], 0.)
        self.assertEqual(paired['auroc']['ci95'], [0., 0.])

    def test_unpromoted_parent_cannot_start_full_confirmation(self):
        for selected, promoted in [('frozen_control', False), ('lora', False), ('frozen_control', True)]:
            with self.subTest(selected=selected, promoted=promoted), \
                    patch.object(q13, 'verified_results', return_value={
                        'selected_model': selected, 'promotion_passed': promoted}), \
                    patch.object(validation, '_full_inputs') as inputs:
                with self.assertRaisesRegex(ValueError, 'promoted parent'):
                    validation.prepare('q13')
                inputs.assert_not_called()


class FullPredictionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(validation, 'output', return_value=self.directory))
        self.stack.enter_context(patch.object(validation, 'verify_protocol', return_value='fixture'))
        self.stack.enter_context(patch.dict(validation.CONFIG, validation_variants=6))
        self.frame = pd.DataFrame({
            'variant_key': [f'v{i}' for i in range(6)], 'label': [0, 1, 0, 1, 0, 1],
            'component': ['a', 'a', 'b', 'b', 'c', 'c'],
            'lora': [0., 1., 0., 1., 0., 1.],
            'matched_control': [0., 1., .9, .8, .8, .9],
            'strongest_control': [0., 1., .9, .8, .8, .9],
        })
        self.protocol = {
            'validation_variant_keys': self.frame.variant_key.tolist(),
            'labels_sha256': q1.fingerprint(self.frame.label.tolist()),
            'components_sha256': q1.fingerprint(self.frame.component.tolist()),
            'selection_variant_keys': ['v0', 'v1'], 'selection_components': ['a'],
        }
        q1.write_json(self.directory / 'protocol.json', self.protocol)
        self.result = {'status': 'complete', 'scope': 'full_validation', 'identity': 'fixture',
                       'validation_variants': 6, 'reload_verified': True, 'frozen_unchanged': True,
                       'subsets': {}, 'artifacts': {}}
        self.result.update(self.measurements(self.frame))
        for name in ['outside_selection', 'unseen_components']:
            subset = self.frame.iloc[2:]
            self.result['subsets'][name] = dict(self.measurements(subset), status='complete',
                                              variants=len(subset), components=2)
        self.result['confirmation_passed'] = validation.confirmation_passes(self.result)
        self.save()

    def measurements(self, frame):
        metrics = {}
        for model in ['lora', 'matched_control', 'strongest_control']:
            points = validation.metric(frame.label.to_numpy(), frame[model].to_numpy())
            metrics[model] = summary(points['auroc'], points['average_precision'])
        paired = {key: {'value': metrics['lora'][key]['value'] - metrics['strongest_control'][key]['value'],
                        'ci95': [.01, .5]} for key in ['auroc', 'average_precision']}
        return {'metrics': metrics, 'paired_lora_minus_strongest_control': paired}

    def save(self):
        path = self.directory / 'validation_predictions.csv'
        self.frame.to_csv(path, index=False)
        self.result['artifacts']['validation_predictions.csv'] = q1.digest_file(path)
        q1.write_json(self.directory / 'metrics.json', self.result)

    def test_complete_predictions_and_confirmation_reproduce(self):
        result = validation.verified_results('q13')
        self.assertTrue(result['confirmation_passed'])
        self.assertEqual(result['validation_variants'], 6)

    def test_wrong_subset_counts_fail_even_with_valid_prediction_hash(self):
        for name in ['outside_selection', 'unseen_components']:
            with self.subTest(subset=name):
                self.result['subsets'][name]['variants'] = 3
                self.save()
                with self.assertRaisesRegex(ValueError, 'cohort size'):
                    validation.verified_results('q13')
                self.result['subsets'][name]['variants'] = 4

    def test_changed_membership_and_order_fail_even_after_rehashing(self):
        original = self.frame.copy()
        for change in ['unknown', 'duplicate', 'order']:
            with self.subTest(change=change):
                self.frame = original.copy()
                if change == 'unknown':
                    self.frame.loc[0, 'variant_key'] = 'different_variant'
                elif change == 'duplicate':
                    self.frame.loc[0, 'variant_key'] = self.frame.loc[1, 'variant_key']
                else:
                    self.frame = self.frame.iloc[::-1].reset_index(drop=True)
                self.save()
                with self.assertRaisesRegex(ValueError, 'exact full validation order'):
                    validation.verified_results('q13')

    def test_changed_labels_or_components_fail_even_after_rehashing(self):
        original = self.frame.copy()
        for column, changed in [('label', 1), ('component', 'different_component')]:
            with self.subTest(column=column):
                self.frame = original.copy()
                self.frame.loc[0, column] = changed
                self.save()
                with self.assertRaisesRegex(ValueError, 'labels or components'):
                    validation.verified_results('q13')

    def test_available_subset_cannot_be_omitted_or_assigned_different_scores(self):
        original = deepcopy(self.result['subsets']['outside_selection'])
        for change in ['omitted', 'metrics']:
            with self.subTest(change=change):
                self.result['subsets']['outside_selection'] = deepcopy(original)
                subset = self.result['subsets']['outside_selection']
                if change == 'omitted':
                    subset.pop('metrics')
                    subset['status'] = 'not_available'
                    expression = 'cannot be omitted'
                else:
                    subset['metrics']['lora']['auroc']['value'] = .3
                    expression = 'reported metrics'
                self.save()
                with self.assertRaisesRegex(ValueError, expression):
                    validation.verified_results('q13')


class ParentProvenanceTests(unittest.TestCase):
    def test_q13_full_protocol_verification_never_requires_a_completed_q14(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            parent_output, full_output = root / 'q13', root / 'q1_full'
            parent_output.mkdir()
            full_output.mkdir()
            stack.enter_context(patch.object(q13, 'OUTPUT', parent_output))
            stack.enter_context(patch.object(validation.q1_full, 'OUTPUT', full_output))
            stack.enter_context(patch.object(q13, 'sources', return_value={}))
            stack.enter_context(patch.object(q13, 'verify_protocol', return_value='parent-identity'))
            never = [stack.enter_context(patch.object(q14, method, side_effect=AssertionError('Q14 must not be used')))
                     for method in ['sources', 'verify_protocol', 'verified_results']]
            parent_protocol = {'validation_variant_keys': ['v1', 'v2']}
            full_protocol = {'vcf_exports': {'clinvar-test.vcf': 'frozen-vcf'}}
            q1.write_json(parent_output / 'protocol.json', parent_protocol)
            q1.write_json(parent_output / 'metrics.json', {'fixture': 'completed parent'})
            (parent_output / 'selected_model.pt').write_bytes(b'fixture checkpoint')
            q1.write_json(full_output / 'protocol.json', full_protocol)
            stack.enter_context(patch.object(validation.q1_full, 'verify_protocol', return_value=full_protocol))
            value = {'question': 'q13', 'configuration': validation.CONFIG,
                     'sources': validation.sources('q13'), 'parent_identity': 'parent-identity',
                     'parent_metrics_sha256': q1.digest_file(parent_output / 'metrics.json'),
                     'parent_selected_model_sha256': q1.digest_file(parent_output / 'selected_model.pt'),
                     'parent_protocol_sha256': q1.digest_file(parent_output / 'protocol.json'),
                     'q1_protocol_sha256': q1.digest_file(full_output / 'protocol.json'),
                     'q1_identity': q1.fingerprint(full_protocol), 'vcf_exports': full_protocol['vcf_exports'],
                     'selection_variant_keys': parent_protocol['validation_variant_keys']}
            q1.write_json(parent_output / 'full/protocol.json', value)
            self.assertEqual(validation.verify_protocol('q13'), q1.fingerprint(value))
            for mocked in never:
                mocked.assert_not_called()

    def test_parent_selected_adapter_cannot_be_mislabeled_as_a_frozen_control(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            for module, name in [(q12, 'q12'), (q13, 'q13'), (q14, 'q14')]:
                directory = root / name
                directory.mkdir()
                stack.enter_context(patch.object(module, 'OUTPUT', directory))
            stack.enter_context(patch.object(q14, 'verify_protocol', return_value='fixture'))
            parent_metrics = {'best_control': summary(.8, .7), 'selected': summary(.9, .9)}
            q1.write_json(q13.OUTPUT / 'metrics.json', {'metrics': parent_metrics})
            q1.write_json(q12.OUTPUT / 'metrics.json', {'metrics': {'best_control': summary(.79, .69)}})
            artifact = q14.OUTPUT / 'fixture.txt'
            artifact.write_text('completed fixture')
            result = {'status': 'complete', 'scope': 'sampled_validation', 'identity': 'fixture',
                      'reloaded_predictions_verified': True, 'frozen_unchanged': True,
                      'head_fixed_verified': True, 'scaler_fixed_verified': True, 'control_optimizer_steps': 0,
                      'artifacts': {'fixture.txt': q1.digest_file(artifact)},
                      'metrics': {'lora': summary(.86, .76), 'matched_control': summary(.77, .65),
                                  'best_control': summary(.77, .65), 'q13_control': summary(.8, .7),
                                  'q12_control': summary(.79, .69), 'selected': summary(.86, .76)},
                      'promotion_passed': True, 'selected_model': 'lora'}
            q1.write_json(q14.OUTPUT / 'metrics.json', result)
            q14.verified_results()
            result['metrics']['q13_control'] = parent_metrics['selected']
            q1.write_json(q14.OUTPUT / 'metrics.json', result)
            with self.assertRaisesRegex(ValueError, 'parent frozen control'):
                q14.verified_results()


class NotebookPublicationTests(unittest.TestCase):
    def test_q15_notebook_uses_its_exploration_and_full_validation_calls(self):
        from notebooks.src import refresh_lora_validation as runner
        notebook = runner.build_notebook('q15')
        nbformat.validate(notebook)
        self.assertIn('Q15-prefix-ranking.ipynb', notebook.cells[0].source)
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        self.assertEqual(len(cells), 5)
        self.assertEqual(cells[2].source, "result = lora_validation.run_validation('q15')")
        self.assertTrue(all(cell.execution_count is None for cell in cells))

    def test_export_and_readme_refresh_follow_saved_fresh_execution(self):
        from notebooks.src import comparison, refresh_lora_validation as runner
        from notebooks.src.refresh_q11 import validate_execution
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            (root / 'notebooks').mkdir()
            directory = root / 'notebooks/results/q13/full'
            stack.enter_context(patch.object(validation, 'ROOT', root))
            stack.enter_context(patch.object(validation, 'output', return_value=directory))
            stack.enter_context(patch.object(validation, 'verified_results', return_value={'fixture': 'completed'}))
            events = []

            def execute(notebook):
                events.append('execute')
                for number, cell in enumerate([cell for cell in notebook.cells if cell.cell_type == 'code'], 1):
                    cell.execution_count = number
                    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=f'fixture output {number}\n')]

            stack.enter_context(patch.object(runner, 'LoggedClient',
                side_effect=lambda notebook, **kwargs: SimpleNamespace(execute=lambda: execute(notebook))))

            def export(question):
                events.append('export')
                path = root / 'notebooks/Q13-lora-validation.ipynb'
                validate_execution(nbformat.read(path, as_version=4))
                status = q1.read_json(directory / 'run_status.json')
                self.assertEqual(status['status'], 'complete')
                self.assertEqual(status['notebook_sha256'], q1.digest_file(path))

            stack.enter_context(patch.object(validation, 'export_full_comparison', side_effect=export))
            stack.enter_context(patch.object(comparison, 'publish_full_lora', side_effect=lambda question: events.append('publish')))
            stack.enter_context(patch.object(comparison, 'refresh', side_effect=lambda: events.append('refresh')))
            runner.refresh('q13')
            self.assertEqual(events, ['execute', 'export', 'publish', 'refresh'])

    def test_failed_execution_preserves_failure_and_never_exports(self):
        from notebooks.src import comparison, refresh_lora_validation as runner
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            (root / 'notebooks').mkdir()
            directory = root / 'notebooks/results/q13/full'
            stack.enter_context(patch.object(validation, 'ROOT', root))
            stack.enter_context(patch.object(validation, 'output', return_value=directory))
            client = stack.enter_context(patch.object(runner, 'LoggedClient'))
            client.return_value.execute.side_effect = RuntimeError('fixture genuine execution failure')
            export = stack.enter_context(patch.object(validation, 'export_full_comparison'))
            publish = stack.enter_context(patch.object(comparison, 'publish_full_lora'))
            refresh = stack.enter_context(patch.object(comparison, 'refresh'))
            with self.assertRaisesRegex(RuntimeError, 'genuine execution failure'):
                runner.refresh('q13')
            export.assert_not_called()
            publish.assert_not_called()
            refresh.assert_not_called()
            self.assertEqual(q1.read_json(directory / 'run_status.json')['status'], 'failed')
            self.assertEqual(len(list(directory.glob('failed-*.ipynb'))), 1)
            self.assertFalse((root / 'notebooks/Q13-lora-validation.ipynb').exists())


if __name__ == '__main__':
    unittest.main()
