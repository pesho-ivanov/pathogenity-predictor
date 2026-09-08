"""Cohort integrity, shared coverage and README propagation checks."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

try:
    from . import comparison as c, comparison_watch as watch
except ImportError:
    from notebooks.src import comparison as c, comparison_watch as watch


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.results = self.root / 'notebooks/results'
        (self.root / 'notebooks/src').mkdir(parents=True)
        self.dump('notebooks/src/q8_catalog.json', c.read_json(c.ROOT / 'notebooks/src/q8_catalog.json'))
        for name in ['q2.py', 'q8_baseline.py', 'q8_revel.py', 'q8_remaining.py', 'q8_dbnsfp.py', 'q9.py']:
            (self.root / 'notebooks/src' / name).write_text('# fixture\n')
        self.source_notebook = self.root / 'notebooks/Q8-existing-tools.ipynb'
        nbformat.write(nbformat.v4.new_notebook(), self.source_notebook)
        self.pilot = pd.DataFrame({
            'variant_key': ['t1', 't2', 'v1', 'v2', 'v3', 'v4'],
            'split': ['train', 'train'] + ['validation'] * 4,
            'component': ['t', 't', 'a', 'a', 'b', 'b'],
            'MC': ['SO:0001583|missense_variant'] * 6})
        directory = self.results / 'q1'
        directory.mkdir(parents=True)
        self.pilot.to_csv(directory / 'split_manifest.csv', index=False)
        self.labels = pd.DataFrame({'variant_key': ['v1', 'v2', 'v3', 'v4'], 'label': [0, 1, 0, 1]})
        self.labels.to_csv(directory / 'validation_labels.csv', index=False)
        self.vcfs = {}
        (self.root / 'data').mkdir()
        for name in ['clinvar-train-pilot.vcf', 'clinvar-test-pilot.vcf']:
            path = self.root / 'data' / name
            path.write_text('frozen VCF fixture\n')
            self.vcfs[name] = c.digest(path)
        self.protocol = {'config': {'assembly': 'GRCh38', 'eligibility': 'MC SO:0001583'},
                         'vcf_exports': self.vcfs, 'artifacts': {name: c.digest(directory / name)
                         for name in ['split_manifest.csv', 'validation_labels.csv']}}
        self.dump('notebooks/results/q1/protocol.json', self.protocol)

    def dump(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def export(self, question='q10', scores=(.1, .9, .8, .2), cohort=None):
        directory = self.results / question
        directory.mkdir(exist_ok=True)
        path = directory / 'comparison_predictions.csv'
        self.labels.assign(score=scores).to_csv(path, index=False)
        self.dump(f'notebooks/results/{question}/comparison_results.json', {
            'q1_protocol_sha256': cohort or c.digest(self.results / 'q1/protocol.json'),
            'predictions_sha256': c.digest(path), 'methods': {'score': f'{question} predictor'},
            'limitations': 'Development fixture'})

    def q8(self):
        directory = self.results / 'q8'
        directory.mkdir(exist_ok=True)
        frame = self.pilot.drop(columns='MC').assign(label=[0, 1, 0, 1, 0, 1],
                                                   am_pathogenicity=[.1, .9, .1, .9, .8, np.nan])
        frame.to_csv(directory / 'pilot_evaluation.csv', index=False)
        self.dump('notebooks/results/q8/validation_metrics.json', {'ignored': 'recomputed from predictions'})
        self.dump('notebooks/results/q8/baseline_protocol.json', {})
        self.dump('notebooks/results/q8/baseline_provenance.json', {
            'q1_protocol_sha256': c.digest(self.results / 'q1/protocol.json'), 'q1_vcf_sha256': self.vcfs,
            'implementation_sha256': c.digest(self.root / 'notebooks/src/q8_baseline.py'),
            'artifacts': {name: c.digest(directory / name) for name in
                          ['pilot_evaluation.csv', 'validation_metrics.json', 'baseline_protocol.json']}})

    def publish_full_cohort(self):
        directory = self.results / 'q1/full'
        directory.mkdir()
        self.pilot.to_csv(directory / 'split_manifest.csv', index=False)
        self.labels.to_csv(directory / 'validation_labels.csv', index=False)
        vcfs = {}
        for name in ['clinvar-train.vcf', 'clinvar-test.vcf']:
            path = self.root / 'data' / name
            path.write_text('full July VCF fixture\n')
            vcfs[name] = c.digest(path)
        self.dump('notebooks/results/q1/full/protocol.json', {
            'config': {'assembly': 'GRCh38', 'eligibility': 'MC SO:0001583',
                       'clinvar_date': '2026-07-06'},
            'vcf_exports': vcfs, 'artifacts': {name: c.digest(directory / name)
                for name in ['split_manifest.csv', 'validation_labels.csv']}})
        return directory

    def test_full_cohort_excludes_old_pilot_metrics_and_labels_snapshot(self):
        self.q8()
        self.publish_full_cohort()
        result, section = c.refresh(self.root, repetitions=10)
        self.assertEqual(result['cohort']['scope'], 'full')
        self.assertEqual(result['cohort']['clinvar_date'], '2026-07-06')
        self.assertFalse(any(row.get('metrics') for row in result['methods']))
        self.assertIn('ClinVar 2026-07-06', section)
        self.assertIn('notebooks/results/q1/full/protocol.json', c.source_signature(self.root))
        self.assertIn('data/clinvar-test.vcf', c.source_signature(self.root))

    def published_fixture(self):
        directory = self.publish_full_cohort()
        protocol = c.read_json(directory / 'protocol.json')
        cohort = {'scope': 'full', 'clinvar_date': '2026-07-06', 'validation_variants': 4,
                  'protocol_sha256': c.digest(directory / 'protocol.json'),
                  'vcf_exports': protocol['vcf_exports'], 'artifacts': protocol['artifacts']}
        measured = {'covered': 4, 'total': 4, 'metrics': {
            'auroc': {'value': .875, 'ci95': [.75, 1.]},
            'average_precision': {'value': .833, 'ci95': [.7, 1.]}}}
        q1_name = 'notebooks/Q1-clinvar-split.ipynb'
        q8_name = 'notebooks/Q8-existing-tools.ipynb'
        q1_cell = nbformat.v4.new_code_cell('show_protocol()', execution_count=1, outputs=[
            nbformat.v4.new_output('display_data', data={
                'text/html': '<pre>' + c.html.escape(json.dumps(protocol)) + '</pre>'})])
        q8_cell = nbformat.v4.new_code_cell('show_results()', execution_count=1, outputs=[
            nbformat.v4.new_output('display_data', data={'text/markdown':
                '| AlphaMissense | 4 / 4 | 0.875 [0.750, 1.000] | 0.833 [0.700, 1.000] |'}),
            nbformat.v4.new_output('stream', name='stdout', text='AlphaMissense: completed; 12.0 seconds.\n')])
        for name, cell in [(q1_name, q1_cell), (q8_name, q8_cell)]:
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), self.root / name)
        prior_path = self.root / c.PUBLISHED.parent / 'prior-readme.md'
        prior_path.parent.mkdir(parents=True)
        prior_path.write_text('```json\n' + json.dumps({'cohort': cohort}) + '\n```\n'
            '| AlphaMissense (Q8) | 4 | 0.875 [0.750, 1.000] | 0.833 [0.700, 1.000] |\n')
        record = {**measured, 'id': 'q8:AlphaMissense', 'question': 'Q8', 'method': 'AlphaMissense',
                  'note': 'Completed fixture.', 'notebook': q8_name, 'runtime_seconds': 12.,
                  'local_artifacts': ['notebooks/results/q8/full/baseline_provenance.json',
                                      'notebooks/results/q8/full/validation_predictions.csv'],
                  'evidence': {'kind': 'table', 'cell': 0, 'output': 0, 'row': 'AlphaMissense'}}
        saved = {'schema_version': 1, 'cohort': cohort, 'cohort_notebook': q1_name,
                 'cohort_evidence': {'cell': 0, 'output': 0},
                 'notebook_sha256': {name: c.digest(self.root / name) for name in [q1_name, q8_name]},
                 'methods': [record], 'published_common': {'q8:AlphaMissense': measured},
                 'bootstrap': {'repetitions': 1000, 'seed': 42},
                 'prior_readme': {'path': str(prior_path.relative_to(self.root)), 'sha256': c.digest(prior_path)}}
        self.dump(c.PUBLISHED, saved)
        return saved

    def test_completed_notebook_scores_survive_missing_exports_and_repeated_refresh(self):
        saved = self.published_fixture()
        self.export(cohort=saved['cohort']['protocol_sha256'])
        for _ in range(2):
            result, section = c.refresh(self.root, repetitions=10)
            rows = {row['id']: row for row in result['methods']}
            self.assertEqual(rows['q8:AlphaMissense']['status'], 'Published')
            self.assertEqual(rows['q8:AlphaMissense']['metrics'], saved['methods'][0]['metrics'])
            self.assertEqual(rows['q10:score']['status'], 'Available')
            self.assertEqual(result['common'], {})
            self.assertIn('4 shared variants (Q2/Q8; excludes LoRA)', section)
            self.assertIn('Published executed-notebook result', section)
            self.assertFalse(result['errors'])
        self.assertIn(str(c.PUBLISHED), c.source_signature(self.root))

    def test_published_scores_match_data_even_after_parent_protocol_regeneration(self):
        self.published_fixture()
        path = self.results / 'q1/full/protocol.json'
        path.write_text(json.dumps(c.read_json(path) | {'regenerated_parent': 'different identity'}))
        result = c.collect(self.root, repetitions=10)
        self.assertEqual(result['published_methods'], ['q8:AlphaMissense'])
        self.assertFalse(result['errors'])

    def test_published_scores_cannot_mask_invalid_or_partial_local_exports(self):
        self.published_fixture()
        path = self.dump('notebooks/results/q8/full/baseline_provenance.json', {})
        result = c.collect(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q8:AlphaMissense')
        self.assertEqual(row['status'], 'Invalid / stale')
        self.assertNotIn('metrics', row)
        path.unlink()
        path.with_name('validation_predictions.csv').write_text('partial')
        result = c.collect(self.root, repetitions=10)
        self.assertNotIn('published_methods', result)

    def test_tampered_or_unexecuted_published_notebook_is_rejected(self):
        saved = self.published_fixture()
        notebook = nbformat.read(self.source_notebook, as_version=4)
        notebook.cells[0].execution_count = None
        nbformat.write(notebook, self.source_notebook)
        for update_hash in [False, True]:
            if update_hash:
                saved['notebook_sha256']['notebooks/Q8-existing-tools.ipynb'] = c.digest(self.source_notebook)
                self.dump(c.PUBLISHED, saved)
            result = c.collect(self.root, repetitions=10)
            self.assertIn('Published q8:AlphaMissense', result['errors'])
            self.assertNotIn('published_methods', result)

    def test_published_metrics_must_match_executed_output(self):
        saved = self.published_fixture()
        saved['methods'][0]['metrics']['auroc']['value'] = .99
        self.dump(c.PUBLISHED, saved)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Published q8:AlphaMissense', result['errors'])
        self.assertNotIn('published_methods', result)

    def test_clean_checkout_retains_published_but_corrupt_q1_cannot(self):
        self.published_fixture()
        path = self.results / 'q1/full/protocol.json'
        path.write_text('{}')
        result = c.collect(self.root, repetitions=10)
        self.assertIsNone(result['cohort'])
        self.assertNotIn('published_methods', result)
        path.unlink()
        (self.results / 'q1/protocol.json').unlink()
        result = c.collect(self.root, repetitions=10)
        self.assertEqual(result['published_methods'], ['q8:AlphaMissense'])
        self.assertEqual(result['cohort']['validation_variants'], 4)

    def test_published_cohort_mismatch_is_rejected(self):
        self.published_fixture()
        path = self.results / 'q1/full/protocol.json'
        protocol = c.read_json(path)
        protocol['config']['clinvar_date'] = '2026-08-01'
        path.write_text(json.dumps(protocol))
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Published results', result['errors'])
        self.assertNotIn('published_methods', result)

    def test_diagnostic_notebooks_do_not_replace_main_q11_comparison_link(self):
        for name in ['Q11-evo2-lora.ipynb', 'Q11-gradient-diagnostics.ipynb', 'Q11-lora-diagnostics.ipynb']:
            nbformat.write(nbformat.v4.new_notebook(), self.root / 'notebooks' / name)
        section = c.render(c.collect(self.root, repetitions=10), self.root)
        self.assertIn('[Q11](notebooks/Q11-evo2-lora.ipynb)', section)
        self.assertNotIn('Q11-lora-diagnostics.ipynb', section)

    def test_invalid_full_cohort_cannot_fall_back_to_valid_pilot(self):
        self.q8()
        directory = self.publish_full_cohort()
        (directory / 'validation_labels.csv').write_text('corrupt\n')
        result = c.collect(self.root, repetitions=10)
        self.assertIsNone(result['cohort'])
        self.assertIn('Q1', result['errors'])
        self.assertFalse(any(row.get('metrics') for row in result['methods']))

    def test_export_bound_to_full_protocol_is_accepted(self):
        directory = self.publish_full_cohort()
        self.export(cohort=c.digest(directory / 'protocol.json'))
        result = c.collect(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q10:score')
        self.assertEqual(row['covered'], 4)
        self.assertIn('metrics', row)

    def test_q12_rejects_sample_scope_even_if_csv_contains_full_membership(self):
        directory = self.publish_full_cohort()
        self.export('q12', cohort=c.digest(directory / 'protocol.json'))
        path = self.results / 'q12/comparison_results.json'
        spec = c.read_json(path) | {'scope': 'sampled_validation', 'require_complete': True,
                                   'vcf_exports': c.read_json(directory / 'protocol.json')['vcf_exports']}
        self.dump(str(path.relative_to(self.root)), spec)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Q12', result['errors'])
        self.assertFalse(any(row.get('metrics') for row in result['methods'] if row['question'] == 'Q12'))
        marker = self.dump('notebooks/results/q12/full/metrics.json', {'status':'complete', 'scope':'full_validation',
            'validation_variants':4, 'reloaded_predictions_verified':True})
        spec.update(scope='full_validation', artifacts={'full/metrics.json':c.digest(marker)})
        self.dump(str(path.relative_to(self.root)), spec)
        result = c.collect(self.root, repetitions=10)
        self.assertNotIn('Q12', result['errors'])
        self.assertTrue(any(row.get('metrics') for row in result['methods'] if row['question'] == 'Q12'))

    def full_lora_export(self, question='q13', confirmed=False, publish_context=True):
        """Build small real files and exercise the exporter without model execution."""
        from types import SimpleNamespace
        from notebooks.src import lora_validation

        directory = self.publish_full_cohort() if publish_context else self.results / 'q1/full'
        context = c.load_context(self.root)
        parent = self.results / question
        full = parent / 'full'
        full.mkdir(parents=True, exist_ok=True)
        parent_protocol = self.dump(f'notebooks/results/{question}/protocol.json', {'fixture': question})
        parent_metrics = self.dump(f'notebooks/results/{question}/metrics.json', {
            'identity': question + '-parent', 'status': 'complete', 'scope': 'sampled_validation',
            'promotion_passed': True, 'selected_model': 'lora'})
        parent_notebook = self.root / 'notebooks' / c.LORA_EXPLORATION_NOTEBOOKS[question]
        parent_cell = nbformat.v4.new_code_cell('show_results()', execution_count=1, outputs=[
            nbformat.v4.new_output('display_data', data={
                'text/html': '<pre>' + c.html.escape(json.dumps(c.read_json(parent_metrics))) + '</pre>'})])
        nbformat.write(nbformat.v4.new_notebook(cells=[parent_cell]), parent_notebook)
        self.dump(f'notebooks/results/{question}/run_status.json', {
            'status': 'complete', 'notebook_sha256': c.digest(parent_notebook), 'notebook_execution_seconds': 50.})
        selected = parent / 'selected_model.pt'
        selected.write_bytes(b'fixture selected model')
        source = self.root / 'notebooks/src/full_validation_fixture.py'
        source.write_text('# checksum-bound fixture implementation\n')
        sources = {str(source.relative_to(self.root)): c.digest(source)}
        frame = context['validation'].assign(lora=[.1, .9, .2, .8], matched_control=[.1, .8, .7, .6],
                                              strongest_control=[.1, .8, .7, .6])
        frame.to_csv(full / 'validation_predictions.csv', index=False)
        protocol = {'question': question, 'sources': sources,
            'q1_protocol_sha256': context['protocol_sha256'], 'vcf_exports': context['vcf_exports'],
            'parent_identity': question + '-parent', 'parent_metrics_sha256': c.digest(parent_metrics),
            'parent_selected_model_sha256': c.digest(selected), 'parent_protocol_sha256': c.digest(parent_protocol)}
        self.dump(f'notebooks/results/{question}/full/protocol.json', protocol)
        identity = c.hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
        measured = c.summarize(frame.label.to_numpy(), {name: frame[name].to_numpy() for name in
                              ['lora', 'matched_control', 'strongest_control']}, frame.component.to_numpy(), 10)
        completion = {'identity': identity, 'status': 'complete', 'scope': 'full_validation',
            'validation_variants': 4, 'reload_verified': True, 'frozen_unchanged': True,
            'original_selection_scores_reproduced': True, 'confirmation_passed': confirmed, 'seconds': 10.,
            'metrics': {name: record['metrics'] for name, record in measured.items()},
            'artifacts': {'validation_predictions.csv': c.digest(full / 'validation_predictions.csv')}}
        self.dump(f'notebooks/results/{question}/full/metrics.json', completion)
        notebook_path = self.root / 'notebooks' / c.NOTEBOOKS[question.upper()]
        cell = nbformat.v4.new_code_cell('show_results()', execution_count=1, outputs=[
            nbformat.v4.new_output('display_data', data={
                'text/html': '<pre>' + c.html.escape(json.dumps(completion)) + '</pre>'})])
        nbformat.write(nbformat.v4.new_notebook(cells=[cell]), notebook_path)
        self.dump(f'notebooks/results/{question}/full/run_status.json', {
            'status': 'complete', 'question': question, 'notebook_sha256': c.digest(notebook_path),
            'notebook_execution_seconds': 10.})
        with patch.object(lora_validation, 'ROOT', self.root), \
                patch.object(lora_validation, 'parent', return_value=SimpleNamespace(OUTPUT=parent)), \
                patch.object(lora_validation, 'output', return_value=full), \
                patch.object(lora_validation, 'verified_results', return_value=completion):
            spec = lora_validation.export_full_comparison(question)
        return parent, spec, completion

    def test_q13_q14_q15_completed_full_exports_publish_both_models_even_on_negative_confirmation(self):
        for question, confirmed in [('q13', False), ('q14', True), ('q15', False)]:
            self.full_lora_export(question, confirmed, publish_context=question == 'q13')
        result = c.collect(self.root, repetitions=10)
        self.assertFalse(result['errors'])
        for question in ['q13', 'q14', 'q15']:
            rows = [row for row in result['methods'] if row['question'] == question.upper()]
            self.assertEqual({row['id'] for row in rows}, {f'{question}:lora', f'{question}:strongest_control'})
            self.assertTrue(all(row['covered'] == row['total'] == 4 and row.get('metrics') for row in rows))
            self.assertTrue(all(row['runtime_seconds'] == 60. for row in rows))
            self.assertTrue(all('one-hour target applies only to exploration' in row['runtime_scope'] for row in rows))
            self.assertEqual(result['common'][f'{question}:lora']['total'], 4)
        negative = next(row for row in result['methods'] if row['id'] == 'q13:lora')
        self.assertIn('confirmation rule not met', negative['note'])
        rendered = c.render(result, self.root)
        self.assertIn('(notebooks/Q13-lora-validation.ipynb)', rendered)
        self.assertIn('(notebooks/Q14-lora-validation.ipynb)', rendered)
        self.assertIn('(notebooks/Q15-lora-validation.ipynb)', rendered)

    def test_lora_full_export_rejects_sampled_stale_and_incomplete_contracts(self):
        parent, spec, _ = self.full_lora_export()
        variants = [('sampled', {'scope': 'sampled_validation'}), ('partial', {'require_complete': False}),
                    ('old_cohort', {'q1_protocol_sha256': 'stale'}), ('old_vcf', {'vcf_exports': {}}),
                    ('missing_evidence', {'artifacts': {}}), ('wrong_identity', {'full_identity': 'stale'}),
                    ('missing_model', {'methods': {'lora': spec['methods']['lora']}})]
        for name, change in variants:
            with self.subTest(change=name):
                self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec | change)
                result = c.collect(self.root, repetitions=10)
                self.assertIn('Q13', result['errors'])
                self.assertFalse(any(row.get('metrics') for row in result['methods'] if row['question'] == 'Q13'))

    def test_lora_full_export_requires_all_completion_and_notebook_checks(self):
        parent, spec, completion = self.full_lora_export()
        original = json.loads(json.dumps(spec))
        for key, value in [('status', 'running'), ('scope', 'sampled_validation'), ('validation_variants', 3),
                           ('reload_verified', False), ('frozen_unchanged', False),
                           ('original_selection_scores_reproduced', False)]:
            with self.subTest(field=key):
                changed = self.dump(str((parent / 'full/metrics.json').relative_to(self.root)), completion | {key: value})
                spec = json.loads(json.dumps(original))
                spec['artifacts']['full/metrics.json'] = c.digest(changed)
                self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
                result = c.collect(self.root, repetitions=10)
                self.assertIn('incomplete or failed integrity', result['errors']['Q13'])
        self.dump(str((parent / 'full/metrics.json').relative_to(self.root)), completion)
        path = self.root / original['notebook']['path']
        notebook = nbformat.read(path, as_version=4)
        for change in ['unexecuted', 'error', 'different_result']:
            with self.subTest(notebook=change):
                changed = json.loads(json.dumps(notebook))
                if change == 'unexecuted':
                    changed['cells'][0]['execution_count'] = None
                elif change == 'error':
                    changed['cells'][0]['outputs'].append({'output_type': 'error', 'ename': 'Error', 'evalue': 'failure', 'traceback': []})
                else:
                    changed['cells'][0]['outputs'][0]['data']['text/html'] = '<pre>{"different": true}</pre>'
                path.write_text(json.dumps(changed))
                spec = json.loads(json.dumps(original))
                spec['notebook']['sha256'] = c.digest(path)
                status = self.dump(str((parent / 'full/run_status.json').relative_to(self.root)), {
                    'status': 'complete', 'question': 'q13', 'notebook_sha256': c.digest(path),
                    'notebook_execution_seconds': 10.})
                spec['artifacts']['full/run_status.json'] = c.digest(status)
                self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
                result = c.collect(self.root, repetitions=10)
                self.assertIn('notebook', result['errors']['Q13'])

    def test_lora_full_export_rejects_prediction_replacement_and_stale_parent_checkpoint(self):
        parent, spec, _ = self.full_lora_export()
        predictions = parent / 'comparison_predictions.csv'
        frame = pd.read_csv(predictions)
        frame.loc[0, 'lora'] = .99
        frame.to_csv(predictions, index=False)
        changed = spec | {'predictions_sha256': c.digest(predictions)}
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), changed)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('differ from completed full predictions', result['errors']['Q13'])
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
        checkpoint = parent / 'selected_model.pt'
        checkpoint.write_bytes(b'replacement model')
        spec['artifacts']['selected_model.pt'] = c.digest(checkpoint)
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('promoted parent checkpoint', result['errors']['Q13'])

    def test_lora_exporter_requires_saved_execution_and_matching_result_outputs(self):
        from types import SimpleNamespace
        from notebooks.src import lora_validation
        parent, spec, completion = self.full_lora_export()
        notebook = self.root / spec['notebook']['path']
        notebook.unlink()
        with patch.object(lora_validation, 'ROOT', self.root), \
                patch.object(lora_validation, 'parent', return_value=SimpleNamespace(OUTPUT=parent)), \
                patch.object(lora_validation, 'output', return_value=parent / 'full'), \
                patch.object(lora_validation, 'verified_results', return_value=completion):
            with self.assertRaises(FileNotFoundError):
                lora_validation.export_full_comparison('q13')

    def test_lora_full_export_preserves_published_competitor_and_shared_subset_results(self):
        published = self.published_fixture()
        self.full_lora_export(publish_context=False)
        result = c.collect(self.root, repetitions=10)
        rows = {row['id']: row for row in result['methods']}
        self.assertEqual(rows['q8:AlphaMissense']['metrics'], published['methods'][0]['metrics'])
        self.assertEqual(rows['q8:AlphaMissense']['status'], 'Published')
        self.assertEqual(result['published_common'], published['published_common'])
        self.assertIn('metrics', rows['q13:lora'])
        text = c.render(result, self.root)
        self.assertIn('Published comparison on 4 shared variants (Q2/Q8; excludes LoRA)', text)

    def test_full_lora_publisher_preserves_competitors_and_survives_missing_caches(self):
        prior = self.published_fixture()
        expected = {}
        for question, confirmed in [('q13', False), ('q14', True), ('q15', False)]:
            parent, spec, completion = self.full_lora_export(question, confirmed, publish_context=False)
            records = c.publish_full_lora(question, self.root)
            expected.update({record['id']: record for record in records})
            self.assertEqual({record['id'] for record in records},
                             {f'{question}:lora', f'{question}:strongest_control'})
            self.assertTrue(all(record['runtime_seconds'] == 60. for record in records))
            for entry, source in zip(records[0]['runtime_status_evidence'],
                                     [parent / 'run_status.json', parent / 'full/run_status.json']):
                self.assertEqual((self.root / entry['path']).read_bytes(), source.read_bytes())
                self.assertIn(entry['sha256'][:12], entry['path'])
                self.assertIn(entry['path'], c.source_signature(self.root))
            for record in records:
                branch = record['id'].split(':')[1]
                self.assertEqual(record['evidence']['metrics_path'], ['metrics', branch])
                self.assertEqual(record['evidence']['count_path'], ['validation_variants'])
                self.assertEqual(record['metrics'], completion['metrics'][branch])
                self.assertEqual(record['note'], spec['limitations'])
            # Remove every local export/full-result cache; parent exploration
            # files alone must not prevent these completed full rows surviving.
            for name in records[0]['local_artifacts']:
                (self.root / name).unlink(missing_ok=True)
        saved = c.read_json(self.root / c.PUBLISHED)
        self.assertEqual(saved['methods'][:len(prior['methods'])], prior['methods'])
        for key, value in prior.items():
            if key not in ['methods', 'notebook_sha256']:
                self.assertEqual(saved[key], value)
        for path, checksum in prior['notebook_sha256'].items():
            self.assertEqual(saved['notebook_sha256'][path], checksum)
        for _ in range(2):
            result = c.collect(self.root, repetitions=10)
            self.assertFalse(result['errors'])
            rows = {row['id']: row for row in result['methods']}
            for identifier, record in expected.items():
                self.assertEqual(rows[identifier]['status'], 'Published')
                self.assertEqual(rows[identifier]['metrics'], record['metrics'])
                self.assertEqual(rows[identifier]['runtime_seconds'], 60.)
            self.assertIn('confirmation rule not met', rows['q13:lora']['note'])
            self.assertEqual(result['published_common'], prior['published_common'])
        # The fully portable path also works without generated Q1 protocols.
        (self.results / 'q1/protocol.json').unlink()
        (self.results / 'q1/full/protocol.json').unlink()
        result = c.collect(self.root, repetitions=10)
        self.assertTrue(set(expected) <= set(result['published_methods']))
        self.assertEqual(result['cohort']['validation_variants'], 4)

    def test_full_lora_publisher_validates_before_changing_any_published_files(self):
        self.published_fixture()
        parent, spec, _ = self.full_lora_export(publish_context=False)
        original = (self.root / c.PUBLISHED).read_bytes()
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)),
                  spec | {'scope': 'sampled_validation'})
        with self.assertRaisesRegex(ValueError, 'sampled exploration'):
            c.publish_full_lora('q13', self.root)
        self.assertEqual((self.root / c.PUBLISHED).read_bytes(), original)
        self.assertFalse(list((self.root / c.PUBLISHED.parent).glob('*-status-*.json')))
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
        self.source_notebook.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'competitor evidence'):
            c.publish_full_lora('q13', self.root)
        self.assertEqual((self.root / c.PUBLISHED).read_bytes(), original)
        self.assertFalse(list((self.root / c.PUBLISHED.parent).glob('*-status-*.json')))

    def test_published_full_lora_cannot_mask_partial_local_results(self):
        self.published_fixture()
        parent, _, _ = self.full_lora_export(publish_context=False)
        records = c.publish_full_lora('q13', self.root)
        for name in records[0]['local_artifacts']:
            (self.root / name).unlink(missing_ok=True)
        for relative in ['full/metrics.json', 'full/run_status.json', 'comparison_predictions.csv',
                         'comparison_results.json']:
            with self.subTest(relative=relative):
                path = parent / relative
                path.write_text('{}')
                result = c.collect(self.root, repetitions=10)
                self.assertFalse(any(row.get('metrics') for row in result['methods']
                                     if row['question'] == 'Q13'))
                self.assertNotIn('q13:lora', result.get('published_methods', []))
                path.unlink()

    def test_published_full_lora_rejects_corrupt_status_and_wrong_duration_or_notebook(self):
        self.published_fixture()
        self.full_lora_export(publish_context=False)
        records = c.publish_full_lora('q13', self.root)
        saved = c.read_json(self.root / c.PUBLISHED)
        entry = records[0]['runtime_status_evidence'][0]
        path = self.root / entry['path']
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        result = c.published_results(self.root)
        self.assertIn('Checksum mismatch', result['errors']['q13:lora'])
        for change in [{'status': 'running'}, {'notebook_sha256': 'different'},
                       {'notebook_execution_seconds': 51.}, {'notebook_execution_seconds': -1.},
                       {'notebook_execution_seconds': True}]:
            with self.subTest(change=change):
                path.write_text(json.dumps(json.loads(original) | change))
                updated = json.loads(json.dumps(saved))
                for row in updated['methods']:
                    if row['question'] == 'Q13':
                        row['runtime_status_evidence'][0]['sha256'] = c.digest(path)
                self.dump(c.PUBLISHED, updated)
                result = c.published_results(self.root)
                self.assertIn('q13:lora', result['errors'])
                self.assertIn('q13:strongest_control', result['errors'])
                self.assertIn('q8:AlphaMissense', result['methods'])
        path.write_bytes(original)
        self.dump(c.PUBLISHED, saved)
        notebook_path = self.root / entry['notebook']
        notebook_path.write_text('{}')
        result = c.published_results(self.root)
        self.assertIn('Checksum mismatch', result['errors']['q13:lora'])
        # Even updating the checksum cannot make an unexecuted source eligible.
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('show_results()')])
        nbformat.write(notebook, notebook_path)
        saved['notebook_sha256'][entry['notebook']] = c.digest(notebook_path)
        self.dump(c.PUBLISHED, saved)
        result = c.published_results(self.root)
        self.assertIn('not completely executed', result['errors']['q13:lora'])

    def test_full_lora_rerun_replaces_only_its_two_rows_and_keeps_old_status_bytes(self):
        prior = self.published_fixture()
        self.full_lora_export(confirmed=False, publish_context=False)
        records = c.publish_full_lora('q13', self.root)
        old_statuses = {entry['path']: (self.root / entry['path']).read_bytes()
                        for entry in records[0]['runtime_status_evidence']}
        self.full_lora_export(confirmed=True, publish_context=False)
        self.assertEqual(set(c.published_results(self.root)['errors']),
                         {'q13:lora', 'q13:strongest_control'})
        records = c.publish_full_lora('q13', self.root)
        saved = c.read_json(self.root / c.PUBLISHED)
        self.assertEqual(len(saved['methods']), len(prior['methods']) + 2)
        self.assertEqual(saved['methods'][:len(prior['methods'])], prior['methods'])
        self.assertEqual(saved['published_common'], prior['published_common'])
        self.assertIn('confirmation rule met.', records[0]['note'])
        for relative, payload in old_statuses.items():
            self.assertEqual((self.root / relative).read_bytes(), payload)
        self.assertFalse(c.published_results(self.root)['errors'])

    def test_lora_full_export_rejects_stale_exploration_notebook_or_runtime(self):
        parent, spec, _ = self.full_lora_export()
        bad = json.loads(json.dumps(spec))
        bad['runtime_stages']['exploration_seconds'] = 0.
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), bad)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('runtime stages', result['errors']['Q13'])
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
        notebook = self.root / spec['exploration_notebook']['path']
        notebook.write_text('{}')
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Checksum mismatch', result['errors']['Q13'])

    def test_q15_cannot_export_sampled_or_incomplete_results_as_full_validation(self):
        parent, spec, completion = self.full_lora_export('q15')
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec | {'scope': 'sampled_validation'})
        result = c.collect(self.root, repetitions=10)
        self.assertIn('sampled exploration', result['errors']['Q15'])
        bad = self.dump(str((parent / 'full/metrics.json').relative_to(self.root)), completion | {'reload_verified': False})
        spec['artifacts']['full/metrics.json'] = c.digest(bad)
        self.dump(str((parent / 'comparison_results.json').relative_to(self.root)), spec)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('integrity checks', result['errors']['Q15'])
        self.assertFalse(any(row.get('metrics') for row in result['methods'] if row['question'] == 'Q15'))

    def test_published_runtime_json_changes_and_deletions_change_source_signature(self):
        self.published_fixture()
        parent, _, _ = self.full_lora_export('q15', publish_context=False)
        records = c.publish_full_lora('q15', self.root)
        evidence = records[0]['runtime_status_evidence'][0]
        path = self.root / evidence['path']
        before = c.source_signature(self.root)
        self.assertIn(evidence['path'], before)
        path.write_text(path.read_text() + '\n')
        corrupted = c.source_signature(self.root)
        self.assertNotEqual(before, corrupted)
        path.unlink()
        removed = c.source_signature(self.root)
        self.assertNotEqual(corrupted, removed)

    def test_missing_full_protocol_cannot_fall_back_to_pilot(self):
        self.q8()
        directory = self.publish_full_cohort()
        (directory / 'protocol.json').unlink()
        result = c.collect(self.root, repetitions=10)
        self.assertIsNone(result['cohort'])
        self.assertFalse(any(row.get('metrics') for row in result['methods']))

    def q2(self):
        directory = self.results / 'q2'
        directory.mkdir(exist_ok=True)
        self.labels.assign(evo=[.1, .9, .2, .8], zero_shot=[-4., -1., -3., -2.],
                           sequence=[.8, .2, .9, .1]).to_csv(directory / 'validation_predictions.csv', index=False)
        self.dump('notebooks/results/q2/protocol.json', {})
        self.dump('notebooks/results/q2/validation_report.json', {
            'lock': {'identity': {'split_protocol_sha256': c.digest(self.results / 'q1/protocol.json'),
                     'protocol_sha256': c.digest(directory / 'protocol.json'),
                     'implementation_sha256': c.digest(self.root / 'notebooks/src/q2.py')}},
            'predictions_sha256': c.digest(directory / 'validation_predictions.csv')})

    def revel(self):
        directory = self.results / 'q8/revel'
        directory.mkdir(parents=True, exist_ok=True)
        self.pilot.drop(columns='MC').assign(label=[0, 1, 0, 1, 0, 1],
            revel=[.1, .9, .1, .9, .2, np.nan]).to_csv(directory / 'pilot_evaluation.csv', index=False)
        for name in ['baseline_protocol.json', 'validation_metrics.json']:
            self.dump(f'notebooks/results/q8/revel/{name}', {})
        self.dump('notebooks/results/q8/revel/baseline_provenance.json', {
            'q1_protocol_sha256': c.digest(self.results / 'q1/protocol.json'), 'q1_vcf_sha256': self.vcfs,
            'implementation_sha256': c.digest(self.root / 'notebooks/src/q8_revel.py'),
            'shared_evaluation_sha256': c.digest(self.root / 'notebooks/src/q8_baseline.py'),
            'artifacts': {name: c.digest(directory / name) for name in
                          ['pilot_evaluation.csv', 'validation_metrics.json', 'baseline_protocol.json']}})

    def test_revel_updates_catalog_row_independently_of_alphamissense(self):
        self.revel()
        result = c.collect(self.root, repetitions=10)
        rows = {r['id']: r for r in result['methods']}
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows['q8:REVEL']['covered'], 3)
        self.assertEqual(rows['q8:REVEL']['metrics']['auroc']['value'], 1)
        self.assertNotIn('metrics', rows['q8:AlphaMissense'])
        self.q8()
        self.q2()
        result, section = c.refresh(self.root, repetitions=10)
        self.assertEqual(len(result['common']), 3)
        self.assertIn('| REVEL | [Q8]', section)
        self.assertIn('notebooks/results/q8/revel/pilot_evaluation.csv', c.source_signature(self.root))
        # A corrupt REVEL export must not hide the independently verified AlphaMissense result.
        (self.results / 'q8/revel/pilot_evaluation.csv').write_text('changed')
        result = c.collect(self.root, repetitions=10)
        rows = {r['id']: r for r in result['methods']}
        self.assertNotIn('metrics', rows['q8:REVEL'])
        self.assertIn('metrics', rows['q8:AlphaMissense'])
        self.assertIn('Q8 REVEL', result['errors'])

    def test_changed_revel_implementation_invalidates_only_revel(self):
        self.revel()
        self.q8()
        (self.root / 'notebooks/src/q8_revel.py').write_text('# changed\n')
        result = c.collect(self.root, repetitions=10)
        rows = {r['id']: r for r in result['methods']}
        self.assertNotIn('metrics', rows['q8:REVEL'])
        self.assertIn('metrics', rows['q8:AlphaMissense'])

    def test_dbnsfp_tools_and_primate_access_are_independent(self):
        for tool in ['SIFT4G', 'PolyPhen-2', 'EVE']:
            slug, _, _ = c.Q8_EXPORTS[tool]
            directory = self.results / 'q8' / slug
            directory.mkdir(parents=True, exist_ok=True)
            self.pilot.drop(columns='MC').assign(label=[0, 1, 0, 1, 0, 1],
                score=[.1, .9, .1, .9, .2, np.nan]).to_csv(directory / 'pilot_evaluation.csv', index=False)
            for name in ['baseline_protocol.json', 'validation_metrics.json']:
                (directory / name).write_text('{}')
            self.dump(f'notebooks/results/q8/{slug}/baseline_provenance.json', {
                'q1_protocol_sha256': c.digest(self.results / 'q1/protocol.json'), 'q1_vcf_sha256': self.vcfs,
                'implementation_sha256': c.digest(self.root / 'notebooks/src/q8_remaining.py'),
                'shared_evaluation_sha256': c.digest(self.root / 'notebooks/src/q8_baseline.py'),
                'acquisition_implementation_sha256': c.digest(self.root / 'notebooks/src/q8_dbnsfp.py'),
                'method': {'tool': tool}, 'artifacts': {name: c.digest(directory / name) for name in
                    ['baseline_protocol.json', 'validation_metrics.json', 'pilot_evaluation.csv']}})
        self.dump('notebooks/results/q8/primateai3d/access_status.json', {
            'status': 'blocked', 'reason': 'Licensed data unavailable',
            'q1_protocol_sha256': c.digest(self.results / 'q1/protocol.json')})
        result = c.collect(self.root, repetitions=10)
        rows = {r['id']: r for r in result['methods']}
        self.assertEqual(len(rows), 10)
        for tool in ['SIFT4G', 'PolyPhen-2', 'EVE']:
            self.assertEqual(rows[f'q8:{tool}']['metrics']['auroc']['value'], 1)
        self.assertEqual(rows['q8:PrimateAI-3D']['status'], 'Blocked')
        self.assertNotIn('metrics', rows['q8:PrimateAI-3D'])
        self.assertFalse(result['errors'])
        (self.results / 'q8/eve/pilot_evaluation.csv').write_text('corrupt')
        rows = {r['id']: r for r in c.collect(self.root, repetitions=10)['methods']}
        self.assertNotIn('metrics', rows['q8:EVE'])
        self.assertIn('metrics', rows['q8:SIFT4G'])

    def test_missing_results_stay_missing_and_archives_are_ignored(self):
        self.dump('notebooks/results/archive/q2/validation_report.json', {'auroc': 1})
        result = c.collect(self.root, repetitions=10)
        self.assertEqual(len(result['methods']), 10)
        self.assertTrue(all('metrics' not in row for row in result['methods']))
        self.assertFalse(result['common'])

    def test_q8_matching_and_q2_negative_zero_shot_scores(self):
        self.q8()
        self.q2()
        result = c.collect(self.root, repetitions=20)
        rows = {row['id']: row for row in result['methods']}
        self.assertEqual(rows['q8:AlphaMissense']['covered'], 3)
        self.assertEqual(rows['q2:zero_shot']['metrics']['auroc']['value'], 1)
        self.assertNotIn('q2:evo', rows)
        self.assertNotIn('q2:sequence', rows)
        self.assertEqual(len(result['common']), 2)
        self.assertTrue(all(row['total'] == 3 for row in result['common'].values()))

    def test_shared_subset_is_recomputed_not_copied(self):
        self.export(scores=[.1, .9, .8, .2])
        self.export('q16', scores=[.2, .8, np.nan, np.nan])
        result = c.collect(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q10:score')
        self.assertEqual(row['metrics']['auroc']['value'], .75)
        self.assertEqual(result['common']['q10:score']['metrics']['auroc']['value'], 1)
        self.assertEqual(result['common']['q10:score']['total'], 2)

    def test_q11_requires_complete_current_full_cohort(self):
        self.publish_full_cohort()
        context = c.load_context(self.root)
        directory = self.results / 'q11'
        directory.mkdir()
        path = directory / 'comparison_predictions.csv'
        frame = self.labels.assign(fine_tuned=[.1, .9, .2, .8])
        frame.to_csv(path, index=False)
        marker = self.dump('notebooks/results/q11/metrics.json', {
            'status': 'complete', 'training_mode': 'partial_epoch', 'reloaded_predictions_verified': True})
        spec = {'methods': c.Q11_METHODS, 'require_complete': True,
                'q1_protocol_sha256': context['protocol_sha256'], 'vcf_exports': context['vcf_exports'],
                'predictions_sha256': c.digest(path), 'artifacts': {'metrics.json': c.digest(marker)}}
        self.dump('notebooks/results/q11/comparison_results.json', spec)
        result = c.collect(self.root, repetitions=10)
        rows = {r['id']: r for r in result['methods'] if r['question'] == 'Q11'}
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(r['status'] == 'Available' for r in rows.values()))
        self.assertEqual(rows['q11:fine_tuned']['metrics']['auroc']['value'], 1.)
        for change in [{'q1_protocol_sha256': 'stale'}, {'require_complete': False}]:
            self.dump('notebooks/results/q11/comparison_results.json', {**spec, **change})
            result = c.collect(self.root, repetitions=10)
            self.assertTrue(all('metrics' not in r for r in result['methods'] if r['question'] == 'Q11'))
        frame.loc[0, 'fine_tuned'] = np.nan
        frame.to_csv(path, index=False)
        self.dump('notebooks/results/q11/comparison_results.json', {**spec, 'predictions_sha256': c.digest(path)})
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Q11', result['errors'])
        self.assertTrue(all('metrics' not in r for r in result['methods'] if r['question'] == 'Q11'))

    def test_stale_cohort_bad_hash_and_wrong_labels_never_get_metrics(self):
        self.export(cohort='other cohort')
        result = c.collect(self.root, repetitions=10)
        self.assertIn('stale', result['errors']['Q10'])
        self.export()
        path = self.results / 'q10/comparison_predictions.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'label'] = 1
        frame.to_csv(path, index=False)
        result = c.collect(self.root, repetitions=10)
        self.assertIn('Checksum mismatch', result['errors']['Q10'])
        spec_path = self.results / 'q10/comparison_results.json'
        spec = c.read_json(spec_path)
        spec['predictions_sha256'] = c.digest(path)
        spec_path.write_text(json.dumps(spec))
        result = c.collect(self.root, repetitions=10)
        self.assertIn('labels differ', result['errors']['Q10'])
        self.assertFalse(any(row.get('metrics') for row in result['methods']))

    def test_changed_vcf_invalidates_all_metrics(self):
        self.q8()
        (self.root / 'data/clinvar-test-pilot.vcf').write_text('changed\n')
        result = c.collect(self.root, repetitions=10)
        self.assertIsNone(result['cohort'])
        self.assertTrue(all(row['status'] == 'Unavailable' for row in result['methods']))

    def test_custom_export_rejects_changed_source_or_model_artifact(self):
        self.export()
        source = self.root / 'notebooks/src/q2.py'
        artifact = self.dump('notebooks/results/q10/selection.json', {'C': 1.0})
        spec_path = self.results / 'q10/comparison_results.json'
        spec = c.read_json(spec_path)
        spec.update(sources={'notebooks/src/q2.py': c.digest(source)},
                    artifacts={'selection.json': c.digest(artifact)})
        spec_path.write_text(json.dumps(spec))
        result = c.collect(self.root, repetitions=10)
        self.assertIn('metrics', next(row for row in result['methods'] if row['id'] == 'q10:score'))
        self.assertIn(str(artifact.relative_to(self.root)), c.source_signature(self.root))
        for path in [source, artifact]:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_text('changed')
                result = c.collect(self.root, repetitions=10)
                self.assertIn('Checksum mismatch', result['errors']['Q10'])
                self.assertNotIn('metrics', next(row for row in result['methods'] if row['id'] == 'q10:score'))
                path.write_bytes(original)

    def test_duplicate_unknown_and_cross_partition_keys_rejected(self):
        context = c.load_context(self.root)
        for keys in [['v1', 'v1', 'v3', 'v4'], ['t1', 'v2', 'v3', 'v4']]:
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, 'membership'):
                c.align_predictions(self.labels.assign(variant_key=keys, score=.5), context, ['score'])

    def test_blocked_q9_has_no_performance(self):
        self.dump('notebooks/results/q9/readiness.json', {'status': 'blocked', 'blockers': ['failed parity']})
        result = c.collect(self.root, repetitions=10)
        rows = [r for r in result['methods'] if r['question'] == 'Q9']
        self.assertTrue(all(r['status'] == 'Blocked' and 'metrics' not in r for r in rows))

    def test_completed_q9_reads_selected_methods(self):
        directory = self.results / 'q9'
        directory.mkdir()
        identity = {'parent_protocol_sha256': c.digest(self.results / 'q1/protocol.json'),
                    'sources': {'notebooks/src/q9.py': c.digest(self.root / 'notebooks/src/q9.py')}}
        np.savez(directory / 'validation_predictions.npz', keys=self.labels.variant_key.to_numpy(dtype=str),
                 labels=self.labels.label.to_numpy(),
                 **{name: [.1, .9, .2, .8] for name in ['fine_tuned', 'frozen', 'zero_shot', 'sequence']})
        self.dump('notebooks/results/q9/readiness.json', {'status': 'passed', 'identity': identity})
        self.dump('notebooks/results/q9/metrics.json', {'status': 'complete', 'identity': identity,
                  'artifacts': {'validation_predictions.npz': c.digest(directory / 'validation_predictions.npz')}})
        result = c.collect(self.root, repetitions=10)
        self.assertTrue(all(r['metrics']['auroc']['value'] == 1 for r in result['methods'] if r['question'] == 'Q9'))
        # Current Q9 exports the fingerprint returned by require_ready().
        report = c.read_json(directory / 'metrics.json')
        report['identity'] = c.hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.dump('notebooks/results/q9/metrics.json', report)
        result = c.collect(self.root, repetitions=10)
        self.assertTrue(all(r['metrics']['auroc']['value'] == 1 for r in result['methods'] if r['question'] == 'Q9'))

    def test_single_class_shared_subset_is_not_ranked(self):
        self.export(scores=[.1, .9, np.nan, np.nan])
        self.export('q16', scores=[.2, np.nan, np.nan, .8])
        result = c.collect(self.root, repetitions=10)
        self.assertFalse(result['common'])
        self.assertIn('Common subset', result['errors'])

    def test_notebook_save_and_result_change_propagate_without_execution(self):
        # The watcher runs the real refresh loop, with bootstrap shortened for this fixture.
        self.export()
        readme = self.root / 'README.md'
        readme.write_text('# Project\n\nIntroduction.\n\n## Research questions\n\nKeep these notes.\n')
        original = watch.readme_comparison.refresh
        with patch.object(watch.readme_comparison, 'refresh', side_effect=lambda root: original(root, repetitions=10)):
            worker = threading.Thread(target=watch.watch, args=(self.root, .03), daemon=True)
            worker.start()
            try:
                self.wait_for(lambda: c.SECTION_END in readme.read_text())
                initial = readme.read_text()
                self.assertIn('| q10 predictor | Q10 | — | 4 / 4 | 0.750', initial)
                # Preserve prose edited while the watcher is running.
                edited = initial.replace('Keep these notes.', 'Keep these revised notes.')
                readme.write_text(edited)
                nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise RuntimeError("must never execute")')])
                nbformat.write(nb, self.source_notebook)
                self.wait_for(lambda: readme.read_text() != edited)
                self.export(scores=[.1, .9, .2, .8])
                def latest_score():
                    result = c.read_json(self.results / 'comparison/summary.json')
                    row = next(r for r in result['methods'] if r['id'] == 'q10:score')
                    return (row['metrics']['auroc']['value'] == 1 and
                            '| q10 predictor | Q10 | — | 4 / 4 | 1.000' in readme.read_text())
                self.wait_for(latest_score)
                self.assertTrue(readme.read_text().startswith('# Project\n\nIntroduction.\n\n'))
                self.assertTrue(readme.read_text().endswith('## Research questions\n\nKeep these revised notes.\n'))
                self.assertEqual(readme.read_text().count(c.SECTION_START), 1)
                self.assertNotIn('| Status |', readme.read_text())
                self.assertFalse((self.root / 'assets/comparison.png').exists())
                self.assertFalse((self.root / 'comparison.ipynb').exists())
                signature = c.source_signature(self.root)
                self.assertNotIn('README.md', signature)
                self.assertNotIn('assets/comparison.png', signature)
                saved = nbformat.read(self.source_notebook, as_version=4)
                self.assertEqual(saved.cells[0].source, nb.cells[0].source)
                self.assertIsNone(saved.cells[0].execution_count)
                self.assertEqual(saved.cells[0].outputs, [])
            finally:
                watch.stop(self.root)
                worker.join(timeout=5)
        self.assertFalse(worker.is_alive())

    def test_readme_includes_metrics_common_subset_and_provenance(self):
        self.q2()
        self.q8()
        result, section = c.refresh(self.root, repetitions=10)
        readme = (self.root / 'README.md').read_text()
        self.assertIn(section, readme)
        self.assertIn('**Direct comparison on the same variants**', readme)
        self.assertIn('[Q8](notebooks/Q8-existing-tools.ipynb)', readme)
        for row in result['methods']:
            if row['id'] in c.README_EXCLUDED_METHODS:
                self.assertNotIn(row['method'], readme)
                continue
            self.assertIn(row['method'], readme)
            self.assertIn(c.html.escape(row['note']).replace('|', r'\|').replace('\n', '<br>'), readme)
            for key in ['auroc', 'average_precision']:
                self.assertIn(c.format_metric(row, key), readme)
        for identifier, row in result['common'].items():
            if identifier in c.README_EXCLUDED_METHODS:
                continue
            self.assertIn(c.format_metric(row, 'auroc'), readme)
            self.assertIn(c.format_metric(row, 'average_precision'), readme)
        metadata = json.loads(readme.split('```json\n', 1)[1].split('\n```', 1)[0])
        self.assertEqual(metadata['cohort'], result['cohort'])
        self.assertEqual(metadata['bootstrap'], result['bootstrap'])
        self.assertEqual(metadata['source_errors'], result['errors'])
        self.assertIn('**Conclusion.**', readme)
        self.assertNotIn('![', readme)
        self.assertFalse((self.root / 'assets/comparison.png').exists())
        self.assertFalse((self.results / 'comparison/metrics.png').exists())

    def test_readme_rejects_malformed_markers_without_overwriting(self):
        readme = self.root / 'README.md'
        for text in [c.SECTION_START, c.SECTION_END, c.SECTION_END + c.SECTION_START,
                     c.SECTION_START * 2 + c.SECTION_END]:
            with self.subTest(text=text):
                readme.write_text(text)
                with self.assertRaisesRegex(ValueError, 'markers'):
                    c.update_readme(self.root, 'new results')
                self.assertEqual(readme.read_text(), text)

    def test_lookup_runtime_uses_current_provenance_and_rejects_old_cohorts(self):
        self.q8()
        path = self.results / 'q8/baseline_provenance.json'
        provenance = c.read_json(path) | {'elapsed_seconds': 96.05}
        path.write_text(json.dumps(provenance))
        result, section = c.refresh(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q8:AlphaMissense')
        self.assertEqual(row['runtime_seconds'], 96.05)
        self.assertIn('| Runtime |', section)
        self.assertIn('1.6 min', section)
        self.assertIn('CPU score lookup/evaluation', section)
        self.assertIn('runtime_seconds', pd.read_csv(self.results / 'comparison/methods.csv').columns)
        provenance['q1_protocol_sha256'] = 'older-cohort'
        path.write_text(json.dumps(provenance))
        row = next(row for row in c.collect(self.root, repetitions=10)['methods']
                   if row['id'] == 'q8:AlphaMissense')
        self.assertNotIn('runtime_seconds', row)
        self.assertEqual(c.format_runtime(row), '—')

    def test_optional_export_runtime_does_not_hide_metrics_when_missing_or_invalid(self):
        self.export()
        path = self.results / 'q10/comparison_results.json'
        spec = c.read_json(path)
        for seconds, expected in [(0, '0 s'), (.025, '0.025 s'), (42, '42.0 s'), (7200, '2.0 h'),
                                  (-1, '—'), (float('inf'), '—'), ('12', '—'), (True, '—')]:
            with self.subTest(seconds=seconds):
                spec['runtimes'] = {'score': {'seconds': seconds, 'scope': 'Inference only'}}
                path.write_text(json.dumps(spec))
                row = next(row for row in c.collect(self.root, repetitions=10)['methods']
                           if row['id'] == 'q10:score')
                self.assertIn('metrics', row)
                self.assertEqual(c.format_runtime(row), expected)
        spec.pop('runtimes')
        path.write_text(json.dumps(spec))
        row = next(row for row in c.collect(self.root, repetitions=10)['methods'] if row['id'] == 'q10:score')
        self.assertIn('metrics', row)
        self.assertEqual(c.format_runtime(row), '—')

    def test_7b_runtime_follows_verified_score_manifest_and_propagates(self):
        self.export('q2')
        directory = self.results / 'q2'
        self.labels.assign(zero_shot_7b=[.1, .9, .8, .2]).to_csv(directory / 'comparison_predictions.csv', index=False)
        timing_path = self.dump('notebooks/results/q2/7b/score_manifest.json', {'batch_seconds': 180})
        report_path = self.dump('notebooks/results/q2/7b/metrics.json', {
            'artifacts': {'score_manifest.json': c.digest(timing_path)}})
        spec_path = directory / 'comparison_results.json'
        spec = c.read_json(spec_path) | {'methods': {'zero_shot_7b': '7B zero-shot'},
            'predictions_sha256': c.digest(directory / 'comparison_predictions.csv'),
            'artifacts': {'7b/metrics.json': c.digest(report_path)}}
        spec_path.write_text(json.dumps(spec))
        result, section = c.refresh(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q2:zero_shot_7b')
        self.assertEqual(row['runtime_seconds'], 180)
        self.assertIn('3.0 min', section)
        signature = c.source_signature(self.root)
        self.assertIn('notebooks/results/q2/7b/score_manifest.json', signature)
        timing_path.write_text(json.dumps({'batch_seconds': 999}))
        self.assertNotEqual(signature, c.source_signature(self.root))
        row = next(row for row in c.collect(self.root, repetitions=10)['methods']
                   if row['id'] == 'q2:zero_shot_7b')
        self.assertIn('metrics', row)
        self.assertNotIn('runtime_seconds', row)
        self.assertIn('Checksum mismatch', row['runtime_error'])

    def test_frozen_head_runtime_sums_extraction_and_fitting(self):
        timing = self.dump('notebooks/results/q10/feature_manifest.json', {'seconds': 2000})
        report = self.dump('notebooks/results/q10/metrics.json', {
            'fit_seconds': 80, 'artifacts': {'feature_manifest.json': c.digest(timing)}})
        value = c.load_runtime(self.root, 'q10:frozen_7b', {'artifacts': {'metrics.json': c.digest(report)}})
        self.assertEqual(value['runtime_seconds'], 2080)
        self.assertIn('classifier fitting', value['runtime_scope'])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.03)
        self.fail('Timed out waiting for comparison propagation')


if __name__ == '__main__':
    unittest.main()
