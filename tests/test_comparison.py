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
        self.assertEqual(len(rows), 9)
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
        self.assertEqual(len(rows), 9)
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
        self.assertEqual(len(result['methods']), 9)
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
        self.export('q11', scores=[.2, .8, np.nan, np.nan])
        result = c.collect(self.root, repetitions=10)
        row = next(row for row in result['methods'] if row['id'] == 'q10:score')
        self.assertEqual(row['metrics']['auroc']['value'], .75)
        self.assertEqual(result['common']['q10:score']['metrics']['auroc']['value'], 1)
        self.assertEqual(result['common']['q10:score']['total'], 2)

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
        self.export('q11', scores=[.2, np.nan, np.nan, .8])
        result = c.collect(self.root, repetitions=10)
        self.assertFalse(result['common'])
        self.assertIn('Common subset', result['errors'])

    def test_notebook_save_and_result_change_propagate_without_execution(self):
        # The watcher runs the real refresh loop, with bootstrap shortened for this fixture.
        self.export()
        readme = self.root / 'README.md'
        readme.write_text('# Project\n\nIntroduction.\n\n## Research questions\n\nKeep these notes.\n')
        original = c.refresh
        with patch.object(c, 'refresh', side_effect=lambda root: original(root, repetitions=10)):
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
            self.assertIn(row['method'], readme)
            self.assertIn(c.html.escape(row['note']).replace('|', r'\|').replace('\n', '<br>'), readme)
            for key in ['auroc', 'average_precision']:
                self.assertIn(c.format_metric(row, key), readme)
        for row in result['common'].values():
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
