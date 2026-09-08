"""Portable Q16 evidence and preservation of prior comparison results."""

import copy
import hashlib
import html
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import nbformat

from notebooks.src import comparison, q16_report as report
from . import test_comparison as fixtures


class Q16PublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.prior = self.fixture.published_fixture()
        self.old_comparison = '<!-- comparison:start -->\nAll previous numbers: .974 / .962\n<!-- comparison:end -->'
        self.readme = self.root / 'README.md'
        self.readme.write_text('# Existing project\n\n' + self.old_comparison + '\n\nExisting notes.\n')
        self.docs = self.root / 'notebooks/results/README.md'
        self.docs.write_text('# Existing artifact descriptions\n')
        self.directory = self.root / report.OUTPUT
        self.directory.mkdir()
        context = comparison.load_context(self.root)
        source = self.root / 'notebooks/src/q16_fixture.py'
        source.write_text('# Fixture source, never GPU code.\n')
        protocol = {'q1_protocol_sha256': context['protocol_sha256'], 'q1_identity': 'q1-fixture',
            'vcf_exports': context['vcf_exports'],
            'sources': {str(source.relative_to(self.root)): comparison.digest(source)}}
        self.dump('protocol.json', protocol)
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
        frame = context['validation'].assign(continuation=[.1, .9, .2, .8],
            q14_parent=[.1, .8, .7, .6], frozen_control=[.2, .7, .6, .4], selected=[.1, .9, .2, .8])
        frame.to_csv(self.directory / 'validation_predictions.csv', index=False)
        summary = comparison.summarize(frame.label.to_numpy(),
            {branch: frame[branch].to_numpy() for branch in [*report.NAMES, 'selected']},
            frame.component.to_numpy(), repetitions=10)
        metrics = {branch: value['metrics'] for branch, value in summary.items()}
        paired = {name: {'value': metrics['continuation'][name]['value'] - metrics['q14_parent'][name]['value'],
                         'ci95': [-.01, .4]} for name in ['auroc', 'average_precision']}
        sample = {branch: {name: value['value'] for name, value in values.items()} for branch, values in metrics.items()}
        history = {'evaluations': [{'steps': 0, **{key: sample['q14_parent'] for key in report.NAMES}},
                                   {'steps': 32, **{key: sample[key] for key in report.NAMES}}]}
        self.dump('training_history.json', history)
        self.result = {'status': 'complete', 'scope': 'full_validation', 'identity': identity,
            'validation_variants': 4, 'metrics': metrics, 'sample_metrics': sample,
            'selected_model': 'continuation', 'head_fixed_verified': True, 'scaler_fixed_verified': True,
            'frozen_unchanged': True, 'reloaded_predictions_verified': True, 'parent_predictions_reproduced': True,
            'paired_continuation_minus_q14_parent': paired,
            'paired_continuation_minus_frozen_control': paired,
            'subsets': {key: {'variants': 2, 'metrics': metrics,
                'paired_continuation_minus_q14_parent': paired} for key in ['outside_selection', 'unseen_components']},
            'seconds': 1800., 'training_examples_seen': 1024, 'actual_unique_training_variants': 512,
            'training_variants': 1024, 'optimizer_steps': 32,
            'artifacts': {name: comparison.digest(self.directory / name) for name in
                          ['protocol.json', 'training_history.json', 'validation_predictions.csv']}}
        self.dump('metrics.json', self.result)
        provenance = report._provenance(self.result, self.root)
        cells = [nbformat.v4.new_code_cell('run_experiment()', execution_count=1),
                 nbformat.v4.new_code_cell('report.show_results()', execution_count=2, outputs=[
                     nbformat.v4.new_output('display_data', data={'text/html':
                         '<pre>' + html.escape(json.dumps(value)) + '</pre>'})
                     for value in [self.result, provenance]])]
        nbformat.write(nbformat.v4.new_notebook(cells=cells), self.root / report.NOTEBOOK)
        self.dump('run_status.json', {'status': 'complete', 'notebook_execution_seconds': 1200.,
                                     'notebook_sha256': comparison.digest(self.root / report.NOTEBOOK)})

    def dump(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value))
        return path

    def publish(self):
        with patch.object(report, '_verified_results', return_value=self.result):
            return report.publish(self.root)

    def test_publication_preserves_old_numbers_and_copies_original_runtime(self):
        prior_bytes = (self.root / comparison.PUBLISHED).read_bytes()
        before = self.readme.read_text()
        record = self.publish()
        self.assertEqual((self.root / comparison.PUBLISHED).read_bytes(), prior_bytes)
        self.assertTrue(self.readme.read_text().startswith(before))
        self.assertIn(self.old_comparison, self.readme.read_text())
        self.assertIn('Q16. Does longer fine-tuning improve on Q14?', self.readme.read_text())
        self.assertIn('Q14 parent LoRA', self.readme.read_text())
        self.assertIn('20.0 minutes', self.readme.read_text())
        self.assertIn('30.0 minutes', self.readme.read_text())
        self.assertEqual((self.root / record['run_status']['path']).read_bytes(),
                         (self.directory / 'run_status.json').read_bytes())
        self.assertEqual(report.published_results(self.root)['result'], self.result)
        self.assertIn('Q16 artifacts', self.docs.read_text())
        first = self.readme.read_text()
        self.publish()
        self.assertEqual(self.readme.read_text(), first)
        self.assertEqual(self.readme.read_text().count(report.START), 1)

    def test_missing_caches_are_portable_but_partial_results_are_not(self):
        record = self.publish()
        for name in record['local_artifacts']:
            (self.root / name).unlink()
        portable = report.published_results(self.root)
        self.assertEqual(portable['result'], self.result)
        report._publish_readmes(self.root, portable)
        self.dump('metrics.json', {'status': 'running'})
        with self.assertRaises((ValueError, FileNotFoundError)):
            report.published_results(self.root)

    def test_corrupt_status_notebook_and_sources_are_rejected(self):
        record = self.publish()
        status = self.root / record['run_status']['path']
        original = status.read_bytes()
        status.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            report.published_results(self.root)
        status.write_bytes(original)
        notebook = self.root / report.NOTEBOOK
        original_notebook = notebook.read_bytes()
        notebook.write_bytes(original_notebook + b' ')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            report.published_results(self.root)
        notebook.write_bytes(original_notebook)
        source = self.root / next(iter(record['provenance']['sources']))
        source.write_text('# Changed implementation\n')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            report.published_results(self.root)

    def test_runtime_must_equal_the_measured_status(self):
        record = self.publish()
        record['runtime_seconds'] = 99.
        with self.assertRaisesRegex(ValueError, 'runtime'):
            report._verify_record(self.root, record)

    def test_readme_sections_are_inserted_without_editing_historical_details(self):
        historical = '## Experiment details\n\nHistorical Q14 values.\n\n## Research questions\n\nQ14 description.\n\n'
        self.readme.write_text(self.old_comparison + '\n\n' + historical + '## Setup\n\nExisting setup.\n')
        self.publish()
        text = self.readme.read_text()
        self.assertLess(text.index(report.START), text.index('## Experiment details'))
        self.assertLess(text.index('## Research questions'), text.index(report.QUESTION_START))
        self.assertLess(text.index(report.QUESTION_END), text.index('## Setup'))
        self.assertIn(historical, text)
        self.assertIn(self.old_comparison, text)

    def test_unexecuted_notebook_cannot_publish_or_change_readmes(self):
        before = self.readme.read_bytes()
        path = self.root / report.NOTEBOOK
        notebook = nbformat.read(path, as_version=4)
        notebook.cells[1].execution_count = None
        nbformat.write(notebook, path)
        with self.assertRaisesRegex(ValueError, 'not completely executed'):
            self.publish()
        self.assertEqual(self.readme.read_bytes(), before)
        self.assertFalse((self.root / report.PUBLISHED).exists())

    def test_result_selection_and_full_membership_are_required(self):
        changed = copy.deepcopy(self.result)
        changed['selected_model'] = 'q14_parent'
        with self.assertRaisesRegex(ValueError, 'selected metrics'):
            report._validate_result(changed, 4)
        for change in [{'scope': 'sampled_validation'}, {'validation_variants': 3},
                       {'reloaded_predictions_verified': False}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                report._validate_result(self.result | change, 4)

    def test_conclusion_separates_sample_selection_from_full_outcome(self):
        result = copy.deepcopy(self.result)
        result['selected_model'] = 'q14_parent'
        result['paired_continuation_minus_q14_parent']['auroc']['value'] = -.01
        text = report.conclusion(result)
        self.assertIn('did not improve both', text)
        self.assertIn('retains **Q14 parent LoRA**', text)
        self.assertIn('Full results do not change that decision', text)
        self.assertIn('not an untouched test', text)


if __name__ == '__main__':
    unittest.main()
