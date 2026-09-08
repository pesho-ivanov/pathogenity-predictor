"""One-table presentation without altering frozen scientific evidence."""

import copy
import unittest
from unittest.mock import patch

from notebooks.src import comparison as c, q16_report, readme_comparison as presentation
from . import test_q16_report as fixtures


class ReadmeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Q16PublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.record = self.fixture.publish()
        measured = self.record['result']['metrics']['q14_parent']
        definitions = [
            ('q11:fine_tuned', 'Q11', 'Previous Q11 LoRA', 4),
            ('q14:lora', 'Q14', 'Previous Q14 LoRA', 4),
            ('q14:strongest_control', 'Q14', 'Previous Q14 frozen control', 4),
            ('q2:zero_shot_7b', 'Q2', 'Evo2 7B zero-shot', 4),
            ('q8:SIFT4G', 'Q8', 'SIFT4G', 3),
            ('q8:PolyPhen-2', 'Q8', 'PolyPhen-2', 3),
            ('q8:REVEL', 'Q8', 'REVEL', 4),
            ('q8:AlphaMissense', 'Q8', 'AlphaMissense', 4),
            ('q8:EVE', 'Q8', 'EVE', 2),
            ('q8:PrimateAI-3D', 'Q8', 'PrimateAI-3D', None),
        ]
        methods = []
        for identifier, question, method, covered in definitions:
            row = {'id': identifier, 'question': question, 'method': method,
                   'status': 'Available' if covered is not None else 'Blocked',
                   'note': 'Previous evidence is retained.' if covered is not None else 'Licensed data unavailable.'}
            if covered is not None:
                row.update(covered=covered, total=4, metrics=copy.deepcopy(measured),
                           runtime_seconds=12., runtime_scope='Measured fixture lookup.')
            methods.append(row)
        self.base = {'methods': methods, 'cohort': self.record['provenance']['cohort'],
                     'errors': {}, 'bootstrap': {'repetitions': 10, 'seed': 42},
                     'generated_utc': '2026-09-08T00:00:00+00:00', 'common': {}}
        self.collect_patch = patch.object(presentation.c, 'collect',
            side_effect=lambda *args, **kwargs: copy.deepcopy(self.base))
        self.collect_patch.start()
        self.addCleanup(self.collect_patch.stop)

    def q16_row(self, result):
        return next(row for row in result['methods'] if row['id'] == presentation.Q16_ID)

    def test_verified_q16_is_first_and_all_previous_rows_keep_exact_measurements(self):
        result = presentation.collect(self.root, repetitions=10)
        self.assertEqual(result['methods'][0]['id'], presentation.Q16_ID)
        self.assertEqual(result['methods'][1:], self.base['methods'])
        self.assertEqual(result['q16'], self.record)
        row = self.q16_row(result)
        self.assertEqual(row['covered'], 4)
        self.assertEqual(row['metrics'], self.record['result']['metrics']['continuation'])
        self.assertEqual(row['runtime_seconds'], 1200.)
        section = presentation.render(result, self.root)
        table = next(presentation.TABLE.finditer(section)).group()
        self.assertTrue(table.splitlines()[2].startswith('| Evo2 7B Q16 continued LoRA'))
        self.assertNotIn('| Paper |', table)
        self.assertIn('| [REVEL](https://doi.org/10.1016/j.ajhg.2016.08.016) |', table)
        for old in self.base['methods']:
            prefix = '| [' if old['id'] in c.METHOD_PAPERS else '| '
            self.assertIn(prefix + old['method'], table)
            if 'covered' in old:
                self.assertIn(f'| {old["covered"]} / 4 |', table)
        self.assertIn('Licensed data unavailable.', section)
        self.assertEqual(len(list(presentation.TABLE.finditer(section))), 1)
        self.assertIn('Provenance, missing results and limitations', section)

    def test_corruption_removes_q16_metrics_and_old_prose_without_dropping_other_tools(self):
        for relative in [self.record['run_status']['path'], str(q16_report.PUBLISHED), str(q16_report.NOTEBOOK)]:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_text('{}')
                try:
                    result, section = presentation.refresh(self.root, repetitions=10)
                    self.assertIsNone(result['q16'])
                    row = self.q16_row(result)
                    self.assertEqual(row['status'], 'Invalid / stale')
                    self.assertNotIn('metrics', row)
                    self.assertEqual(result['methods'][1:], self.base['methods'])
                    self.assertIn('Q16', result['errors'])
                    line = next(line for line in section.splitlines() if line.startswith('| Evo2 7B Q16'))
                    self.assertEqual([part.strip() for part in line.split('|')[3:6]], ['—', '—', '—'])
                    q16_text = (self.root / 'README.md').read_text().split(q16_report.START, 1)[1].split(q16_report.END, 1)[0]
                    self.assertIn('results are unavailable', q16_text)
                    self.assertNotIn('Continuation minus Q14:', q16_text)
                finally:
                    path.write_bytes(original)

    def test_different_or_missing_main_cohort_cannot_publish_q16_scores(self):
        original = self.base['cohort']
        for cohort in [dict(original, clinvar_date='different snapshot'), None]:
            with self.subTest(cohort=cohort):
                self.base['cohort'] = cohort
                result = presentation.collect(self.root, repetitions=10)
                self.assertIsNone(result['q16'])
                self.assertNotIn('metrics', self.q16_row(result))
                self.assertIn('main comparison cohort', result['errors']['Q16'])
                self.assertEqual(result['methods'][1:], self.base['methods'])
        self.base['cohort'] = original

    def test_refresh_is_idempotent_and_leaves_frozen_evidence_and_nonresult_text_unchanged(self):
        counts = '| Split | Variants |\n| --- | ---: |\n| Train | 46,888 |\n| Validation | 17,927 |\n'
        settings = '| Setting | Value |\n| --- | --- |\n| LoRA rank | 8 |\n| Learning rate | 3e-5 |\n'
        extra = '| Old sample | AUROC | AP |\n| --- | ---: | ---: |\n| Old model | .832 | .725 |\n'
        path = self.root / 'README.md'
        path.write_text(path.read_text() + '\nUnrelated explanation remains.\n\n'
                        + counts + '\n' + settings + '\n' + extra)
        protected = [self.root / c.PUBLISHED, self.root / q16_report.PUBLISHED,
                     self.root / self.record['run_status']['path'], self.fixture.docs]
        protected += list((self.root / 'notebooks').glob('*.ipynb'))
        protected += [self.root / name for name in self.record['local_artifacts']]
        protected += [self.root / name for name in self.record['provenance']['sources']]
        snapshots = {file: file.read_bytes() for file in protected}
        first, _ = presentation.refresh(self.root, repetitions=10)
        rendered = path.read_text()
        performance = [match for match in presentation.TABLE.finditer(rendered)
                       if 'AUROC' in match.group().splitlines()[0]]
        self.assertEqual(len(performance), 1)
        self.assertIn(counts, rendered)
        self.assertIn(settings, rendered)
        self.assertIn('Unrelated explanation remains.', rendered)
        self.assertNotIn('| Old model |', rendered)
        second, _ = presentation.refresh(self.root, repetitions=10)
        self.assertEqual(path.read_text(), rendered)
        self.assertEqual(first['methods'], second['methods'])
        for file, before in snapshots.items():
            self.assertEqual(file.read_bytes(), before, str(file))
        summary = c.read_json(self.root / 'notebooks/results/comparison/summary.json')
        self.assertEqual({row['id'] for row in summary['methods']},
                         {row['id'] for row in self.base['methods']} | {presentation.Q16_ID})
        exported = (self.root / 'notebooks/results/comparison/methods.csv').read_text()
        for row in first['methods']:
            self.assertIn(row['id'], exported)

    def test_absent_q16_publication_does_not_change_existing_collection(self):
        (self.root / q16_report.PUBLISHED).unlink()
        result = presentation.collect(self.root, repetitions=10)
        self.assertIsNone(result['q16'])
        self.assertEqual(result['methods'], self.base['methods'])
        self.assertEqual(result['errors'], {})


if __name__ == '__main__':
    unittest.main()
