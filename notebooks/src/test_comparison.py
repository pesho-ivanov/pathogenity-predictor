"""Cohort integrity, shared coverage and notebook propagation checks."""

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

from . import comparison as c, comparison_watch as watch


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.results = self.root / 'notebooks/results'
        (self.root / 'notebooks/src').mkdir(parents=True)
        self.dump('notebooks/src/q8_catalog.json', c.read_json(c.ROOT / 'notebooks/src/q8_catalog.json'))
        for name in ['q2.py', 'q8_baseline.py', 'q9.py']:
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

    def test_missing_results_stay_missing_and_archives_are_ignored(self):
        self.dump('notebooks/results/archive/q2/validation_report.json', {'auroc': 1})
        result = c.collect(self.root, repetitions=10)
        self.assertEqual(len(result['methods']), 13)
        self.assertTrue(all('metrics' not in row for row in result['methods']))
        self.assertFalse(result['common'])

    def test_q8_matching_and_q2_negative_zero_shot_scores(self):
        self.q8()
        self.q2()
        result = c.collect(self.root, repetitions=20)
        rows = {row['id']: row for row in result['methods']}
        self.assertEqual(rows['q8:AlphaMissense']['covered'], 3)
        self.assertEqual(rows['q2:zero_shot']['metrics']['auroc']['value'], 1)
        self.assertEqual(rows['q2:sequence']['metrics']['auroc']['value'], 0)
        self.assertEqual(len(result['common']), 4)
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

    def test_completed_q9_reads_all_four_methods(self):
        directory = self.results / 'q9'
        directory.mkdir()
        identity = {'parent_protocol_sha256': c.digest(self.results / 'q1/protocol.json'),
                    'sources': {'notebooks/src/q9.py': c.digest(self.root / 'notebooks/src/q9.py')}}
        np.savez(directory / 'validation_predictions.npz', keys=self.labels.variant_key.to_numpy(dtype=str),
                 labels=self.labels.label.to_numpy(), **{name: [.1, .9, .2, .8] for name in c.Q9_METHODS})
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
        original = c.refresh
        with patch.object(c, 'refresh', side_effect=lambda root: original(root, repetitions=10)):
            worker = threading.Thread(target=watch.watch, args=(self.root, .03), daemon=True)
            worker.start()
            try:
                self.wait_for(lambda: (self.root / 'comparison.ipynb').exists())
                initial = (self.root / 'comparison.ipynb').read_text()
                nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('raise RuntimeError("must never execute")')])
                nbformat.write(nb, self.source_notebook)
                self.wait_for(lambda: (self.root / 'comparison.ipynb').read_text() != initial)
                self.export(scores=[.1, .9, .2, .8])
                def latest_score():
                    result = c.read_json(self.results / 'comparison/summary.json')
                    row = next(r for r in result['methods'] if r['id'] == 'q10:score')
                    return row['metrics']['auroc']['value'] == 1
                self.wait_for(latest_score)
                notebook = nbformat.read(self.root / 'comparison.ipynb', as_version=4)
                nbformat.validate(notebook)
                self.assertTrue(all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code'))
                self.assertFalse(any(output.output_type == 'error' for cell in notebook.cells for output in cell.get('outputs', [])))
                self.assertFalse(any('comparison.ipynb' == key for key in c.source_signature(self.root)))
            finally:
                watch.stop(self.root)
                worker.join(timeout=5)
        self.assertFalse(worker.is_alive())

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.03)
        self.fail('Timed out waiting for comparison propagation')


if __name__ == '__main__':
    unittest.main()
