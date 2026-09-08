"""Present verified experiment results in one README comparison table.

The experiment collectors and Q16 publisher are frozen scientific sources.
This presentation layer changes the README without changing their evidence.
Run: python -m notebooks.src.readme_comparison
"""

import html
import json
from pathlib import Path
import re

import pandas as pd

from . import comparison as c, q16_report

ROOT = c.ROOT
Q16_ID = 'q16:continuation'
TABLE = re.compile(r'(?m)^\|[^\n]+\|\n\|[ :|\-]+\|\n(?:\|[^\n]+\|(?:\n|$))+')


def source_signature(root=ROOT, content=False):
    root = Path(root)
    signature = c.source_signature(root, content)
    paths = {Path('notebooks/src/readme_comparison.py'), Path('notebooks/src/comparison_watch.py')}
    record = root / q16_report.PUBLISHED
    if record.exists():
        try:
            paths.update(Path(name) for name in c.read_json(record)['local_artifacts'])
        except (ValueError, KeyError, TypeError):
            pass  # The collector reports invalid evidence; its file is already watched.
    for relative in paths:
        path = root / relative
        try:
            stat = path.stat()
            signature[str(relative)] = c.digest(path) if content else [stat.st_mtime_ns, stat.st_size]
        except FileNotFoundError:
            signature[str(relative)] = None
    return signature


def collect(root=ROOT, repetitions=c.REPETITIONS):
    root = Path(root)
    result = c.collect(root, repetitions)
    result['q16'] = None
    if not (root / q16_report.PUBLISHED).exists():
        return result
    row = {'id': Q16_ID, 'question': 'Q16',
           'method': 'Evo2 7B Q16 continued LoRA (blocks 29 and 30, rank 8, 512 bp)',
           'notebook': str(q16_report.NOTEBOOK), 'status': 'Invalid / stale', 'note': ''}
    result['methods'] = [item for item in result['methods'] if item['id'] != Q16_ID]
    result['methods'].insert(0, row)
    try:
        record = q16_report.published_results(root)
        cohort = record['provenance']['cohort']
        c.require(result['cohort'] is not None and all(
            cohort[key] == result['cohort'][key]
            for key in ['scope', 'clinvar_date', 'validation_variants', 'vcf_exports']),
            'Q16 does not match the main comparison cohort')
        measured = record['result']
        status = c.read_json(root / record['run_status']['path'])
        seconds = status.get('request_elapsed_seconds', record['runtime_seconds'])
        c.require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                  and record['runtime_seconds'] <= seconds < float('inf'), 'Invalid Q16 request runtime')
        scope = ('Original request through the saved notebook, including preparation, the failed preflight, '
                 'training, all three full-validation passes, reload checks and confidence intervals. '
                 'Excludes inherited Q12/Q14 fitting and initial model acquisition.')
        if 'request_elapsed_seconds' not in status:
            scope = 'Fresh notebook execution, including training, full validation and reporting; excludes inherited fitting.'
        row.update(status='Published', covered=measured['validation_variants'],
                   total=measured['validation_variants'], metrics=measured['metrics']['continuation'],
                   runtime_seconds=seconds, runtime_scope=scope, note=q16_report.LIMITATIONS)
        result['q16'] = record
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        row['note'] = str(error)
        result['errors']['Q16'] = str(error)
    return result


def render(result, root=ROOT):
    root = Path(root)
    rows = sorted((row for row in result['methods'] if row['id'] not in c.README_EXCLUDED_METHODS),
                  key=lambda row: (row['id'] != Q16_ID, row['question'] == 'Q8',
                                   row['id'] == 'q8:PrimateAI-3D'))
    table, notes = [], []
    for row in rows:
        relative = row.get('notebook', 'notebooks/' + c.NOTEBOOKS.get(row['question'], ''))
        path = root / relative
        if not path.is_file():
            path = next(iter(sorted((root / 'notebooks').glob(row['question'] + '-*.ipynb'))), None)
        link = f'[{row["question"]}]({path.relative_to(root).as_posix()})' if path else row['question']
        name = html.escape(row['method']) + (' (licensed)' if row['id'] == 'q8:PrimateAI-3D' else '')
        table.append({'Method': name, 'Notebook': link, 'Paper': c.METHOD_PAPERS.get(row['id'], '—'),
                      'Scored / validation': f'{row["covered"]:,} / {row["total"]:,}' if 'covered' in row else '—',
                      'AUROC [95% CI]': c.format_metric(row, 'auroc'),
                      'Average precision [95% CI]': c.format_metric(row, 'average_precision'),
                      'Runtime': c.format_runtime(row)})
        detail = row['note']
        if row.get('runtime_scope'):
            detail += ' Runtime: ' + row['runtime_scope']
        elif row.get('runtime_error'):
            detail += ' Runtime unavailable: ' + row['runtime_error']
        notes.append(f'**{name} ({row["question"]}).** {html.escape(detail)}')
    cohort = result['cohort']
    intro = (f'**{cohort["validation_variants"]:,} missense validation variants · '
             f'ClinVar {cohort["clinvar_date"]} · {cohort["scope"]} cohort**' if cohort
             else '**Frozen validation inputs unavailable**')
    metadata = {key: result.get(key) for key in ['cohort', 'errors', 'bootstrap', 'published_methods',
                                               'published_bootstrap', 'generated_utc']}
    return '\n\n'.join([
        '## Method comparison', intro,
        c.markdown_table(table, separator_before=sum(row['question'] != 'Q8' for row in rows)),
        'Coverage differs among tools: each AUROC and average precision uses the scored variants shown. '
        'These rows do not establish a ranking on identical variants. “—” indicates an unavailable result or timing.',
        'Runtime covers the recorded stages described below; hardware and caching differ. '
        'The Q16 row includes preparation through completion of its saved notebook. '
        'Sampled checkpoint-selection results and shared-variant analyses remain in the linked notebooks '
        'and [preserved comparison records](notebooks/results/comparison/published/results.json).',
        '<details>\n<summary>Provenance, missing results and limitations</summary>\n\n'
        + '\n\n'.join(notes) + '\n\n```json\n' + json.dumps(metadata, indent=2) + '\n```\n\n</details>',
        'These are development results: validation participates in model selection. '
        'The 95% intervals resample whole Q1 components and do not correct selection bias. '
        'Pretraining exposure and external-tool training/calibration overlap with ClinVar remain unresolved; '
        'clinical validity has not been established.',
    ])


def collapse_secondary_tables(text):
    """Remove extra performance tables; retain the main table and data/method tables."""
    def replace(match):
        header = match.group().splitlines()[0]
        performance = 'AUROC' in header or 'Average precision' in header
        main = '| Notebook |' in header and '| Paper |' in header
        return '' if performance and not main else match.group()
    return re.sub(r'\n{3,}', '\n\n', TABLE.sub(replace, text))


def q16_section(result):
    heading = '## Q16. Does longer fine-tuning improve on Q14?\n\n'
    record = result.get('q16')
    if record is None:
        return heading + 'Q16 results are unavailable: ' + html.escape(result['errors'].get('Q16', 'No completed publication.'))
    value = record['result']
    row = next(row for row in result['methods'] if row['id'] == Q16_ID)
    sample = value['sample_metrics']['continuation']
    return (heading + f'[Q16 continued LoRA]({q16_report.NOTEBOOK}) trained on all '
        f'**{value["training_variants"]:,} variants**, processing **{value["training_examples_seen"]:,} examples** '
        f'in **{value["optimizer_steps"]:,} updates**. It continued Q14’s adapters at learning rate **3e-5**, '
        'with the classifier and scaler fixed. Full metrics and coverage appear in the '
        '[main comparison](#method-comparison).\n\n' + q16_report.conclusion(value) + '\n\n'
        f'The 2,048-variant selection sample scored **{sample["auroc"]:.3f} AUROC / '
        f'{sample["average_precision"]:.3f} average precision**. '
        f'Fresh notebook execution took **{record["runtime_seconds"]/60:.1f} minutes**; '
        f'the runtime reported in the main table is **{row["runtime_seconds"]/60:.1f} minutes** '
        'through notebook completion, with its scope recorded in the comparison details.\n\n'
        f'The [preserved Q16 record]({q16_report.PUBLISHED}) pins the executed notebook, '
        'displayed metrics, source hashes, cohort and measured runtime.')


def refresh(root=ROOT, repetitions=c.REPETITIONS):
    import fcntl
    root = Path(root)
    output = root / 'notebooks/results/comparison'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = source_signature(root)
        result = collect(root, repetitions)
        section = render(result, root)
        c.require(before == source_signature(root), 'Sources changed during refresh; retry after writes finish')
        result['sources'] = source_signature(root, content=True)
        path = root / 'README.md'
        original = path.read_text()
        updated = q16_report._replace_section(original, section, before='## Research questions\n',
                                             start=c.SECTION_START, end=c.SECTION_END)
        if result.get('q16') is not None or q16_report.START in updated:
            updated = q16_report._replace_section(updated, q16_section(result), before='## Experiment details\n')
        updated = collapse_secondary_tables(updated)
        c.require(path.read_text() == original, 'README changed during update; retry')
        c.atomic_write(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
        frame = pd.DataFrame([{key: row.get(key) for key in ['id', 'question', 'method', 'status', 'covered', 'total',
                                                           'runtime_seconds', 'runtime_scope']} |
                              {name: row.get('metrics', {}).get(name, {}).get('value')
                               for name in ['auroc', 'average_precision']} for row in result['methods']])
        c.atomic_write(output / 'methods.csv', frame.to_csv(index=False))
        if updated != original:
            c.atomic_write(path, updated)
        return result, section


if __name__ == '__main__':
    refresh()
    print('Refreshed the single README comparison table.')
