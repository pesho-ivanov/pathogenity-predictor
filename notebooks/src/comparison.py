"""Collect verified validation predictions and render the root comparison notebook.

This module never executes a source notebook, imports a model, or reads archives.
Notebook saves trigger refreshes; scientific values come from exported predictions.
"""

import base64
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import tempfile

import nbformat
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPETITIONS = 1000
SEED = 42
Q2_METHODS = {'evo': 'Evo2 + logistic regression', 'zero_shot': 'Evo2 zero-shot',
              'sequence': 'Sequence + logistic regression'}
Q9_METHODS = {'fine_tuned': 'Evo2 fine-tuned (BioNeMo)', 'frozen': 'Evo2 frozen head (BioNeMo)',
              'zero_shot': 'Evo2 zero-shot (BioNeMo)', 'sequence': 'Sequence classifier (Q9)'}
SOURCE_FILES = {
    'q1': ['protocol.json', 'split_manifest.csv', 'validation_labels.csv'],
    'q2': ['protocol.json', 'validation_report.json', 'validation_predictions.csv'],
    'q8': ['baseline_protocol.json', 'baseline_provenance.json', 'validation_metrics.json', 'pilot_evaluation.csv'],
    'q9': ['protocol.json', 'readiness.json', 'metrics.json', 'validation_predictions.npz'],
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content.encode() if isinstance(content, str) else content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified(path, expected):
    require(digest(path) == expected, f'Checksum mismatch: {Path(path).name}')


def source_paths(root=ROOT):
    """Only small result inputs, source notebooks and provenance-relevant code."""
    root = Path(root)
    paths = set((root / 'notebooks').glob('Q*.ipynb'))
    paths.update((root / 'notebooks/src').glob('q*.py'))
    paths.add(root / 'notebooks/src/q8_catalog.json')
    for question, names in SOURCE_FILES.items():
        paths.update(root / 'notebooks/results' / question / name for name in names)
    paths.update(root / 'data' / name for name in ['clinvar-train-pilot.vcf', 'clinvar-test-pilot.vcf'])
    for path in (root / 'notebooks/results').glob('q*/comparison_results.json'):
        paths.add(path)
        # This is the documented filename for future notebooks' prediction exports.
        paths.add(path.with_name('comparison_predictions.csv'))
    return sorted(paths)


def source_signature(root=ROOT, content=False):
    """Missing files are represented too, so creations and deletions refresh."""
    root = Path(root)
    result = {}
    for path in source_paths(root):
        try:
            stat = path.stat()
            result[str(path.relative_to(root))] = digest(path) if content else [stat.st_mtime_ns, stat.st_size]
        except FileNotFoundError:
            result[str(path.relative_to(root))] = None
    return result


def load_context(root):
    directory = root / 'notebooks/results/q1'
    protocol = read_json(directory / 'protocol.json')
    require(protocol['config']['assembly'] == 'GRCh38' and
            'SO:0001583' in protocol['config']['eligibility'], 'Q1 must identify the missense GRCh38 cohort')
    for name in ['split_manifest.csv', 'validation_labels.csv']:
        verified(directory / name, protocol['artifacts'][name])
    for name, checksum in protocol['vcf_exports'].items():
        verified(root / 'data' / name, checksum)
    pilot = pd.read_csv(directory / 'split_manifest.csv')
    require(not pilot.variant_key.duplicated().any(), 'Duplicate Q1 variants')
    require(pilot.MC.map(lambda value: isinstance(value, str) and any(
        term.partition('|')[0] == 'SO:0001583' for term in value.split(','))).all(), 'Nonmissense Q1 variants')
    require(set(pilot.split) == {'train', 'validation'}, 'Unexpected Q1 partition roles')
    require(not set(pilot.loc[pilot.split.eq('train'), 'component']) &
            set(pilot.loc[pilot.split.eq('validation'), 'component']), 'Q1 components cross partitions')
    validation = pilot[pilot.split.eq('validation')][['variant_key', 'component']].copy()
    labels = pd.read_csv(directory / 'validation_labels.csv')
    require(not labels.variant_key.duplicated().any() and set(labels.variant_key) == set(validation.variant_key),
            'Q1 validation label membership differs')
    validation = validation.merge(labels, on='variant_key', validate='one_to_one')
    require(set(validation.label) == {0, 1}, 'Q1 validation needs both ClinVar classes')
    return {'protocol_sha256': digest(directory / 'protocol.json'), 'vcf_exports': protocol['vcf_exports'],
            'validation': validation, 'pilot_keys': pilot.variant_key.tolist(), 'config': protocol['config']}


def align_predictions(frame, context, columns, allow_missing=False):
    """Keys, groups and labels must agree before any ranking metric is computed."""
    expected = context['validation']
    require(not frame.variant_key.duplicated().any() and set(frame.variant_key) == set(expected.variant_key),
            'Predictions do not match the exact Q1 validation membership')
    aligned = frame.set_index('variant_key').loc[expected.variant_key].reset_index()
    require(aligned.label.tolist() == expected.label.tolist(), 'Prediction labels differ from frozen ClinVar labels')
    if 'component' in aligned:
        require(aligned.component.tolist() == expected.component.tolist(), 'Prediction components differ from Q1')
    arrays = {}
    for column in columns:
        values = pd.to_numeric(aligned[column], errors='raise').to_numpy(dtype=float)
        require(not np.isinf(values).any() and (allow_missing or np.isfinite(values).all()),
                f'Invalid or missing scores: {column}')
        arrays[column] = values
    return arrays


def load_q2(root, context):
    directory = root / 'notebooks/results/q2'
    report = read_json(directory / 'validation_report.json')
    identity = report['lock']['identity']
    require(identity['split_protocol_sha256'] == context['protocol_sha256'], 'Q2 cohort is stale')
    verified(directory / 'protocol.json', identity['protocol_sha256'])
    verified(root / 'notebooks/src/q2.py', identity['implementation_sha256'])
    verified(directory / 'validation_predictions.csv', report['predictions_sha256'])
    return align_predictions(pd.read_csv(directory / 'validation_predictions.csv'), context, Q2_METHODS)


def load_q8(root, context):
    directory = root / 'notebooks/results/q8'
    provenance = read_json(directory / 'baseline_provenance.json')
    require(provenance['q1_protocol_sha256'] == context['protocol_sha256'] and
            provenance['q1_vcf_sha256'] == context['vcf_exports'], 'Q8 cohort is stale')
    verified(root / 'notebooks/src/q8_baseline.py', provenance['implementation_sha256'])
    for name in ['baseline_protocol.json', 'pilot_evaluation.csv', 'validation_metrics.json']:
        verified(directory / name, provenance['artifacts'][name])
    frame = pd.read_csv(directory / 'pilot_evaluation.csv')
    require(frame.variant_key.tolist() == context['pilot_keys'], 'Q8 pilot membership or ordering changed')
    arrays = align_predictions(frame[frame.split.eq('validation')], context, ['am_pathogenicity'], allow_missing=True)
    return {'AlphaMissense': arrays['am_pathogenicity']}


def load_q9(root, context):
    directory = root / 'notebooks/results/q9'
    report = read_json(directory / 'metrics.json')
    readiness = read_json(directory / 'readiness.json')
    require(readiness['status'] == 'passed' and report['status'] == 'complete', 'Q9 has no completed, gated evaluation')
    identity = readiness['identity']
    expected_identity = (hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
                         if isinstance(report['identity'], str) else identity)
    require(report['identity'] == expected_identity, 'Q9 results and readiness identity differ')
    require(identity['parent_protocol_sha256'] == context['protocol_sha256'], 'Q9 cohort is stale')
    for name, checksum in identity['sources'].items():
        verified(root / name, checksum)
    verified(directory / 'validation_predictions.npz', report['artifacts']['validation_predictions.npz'])
    with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved:
        frame = pd.DataFrame({'variant_key': saved['keys'], 'label': saved['labels'],
                              **{key: saved[key] for key in Q9_METHODS}})
    return align_predictions(frame, context, Q9_METHODS)


def summarize(labels, predictions, groups, repetitions=REPETITIONS):
    """Same group resamples for every method; condition on each method's coverage."""
    metrics = {'auroc': roc_auc_score, 'average_precision': average_precision_score}
    result, samples = {}, {}
    for name, values in predictions.items():
        covered = np.isfinite(values)
        result[name] = {'covered': int(covered.sum()), 'total': len(labels), 'metrics': {}}
        if len(np.unique(labels[covered])) == 2:
            result[name]['metrics'] = {metric: {'value': float(function(labels[covered], values[covered])), 'ci95': None}
                                       for metric, function in metrics.items()}
            samples[name] = {metric: [] for metric in metrics}
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(SEED)
    for _ in range(repetitions if samples else 0):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        for name, values in predictions.items():
            if name not in samples:
                continue
            selected = index[np.isfinite(values[index])]
            if len(np.unique(labels[selected])) < 2:
                continue
            for metric, function in metrics.items():
                samples[name][metric].append(float(function(labels[selected], values[selected])))
    for name, values in samples.items():
        result[name]['bootstrap_valid_replicates'] = len(values['auroc'])
        for metric, draws in values.items():
            if len(draws) >= 2:
                result[name]['metrics'][metric]['ci95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def collect(root=ROOT, repetitions=REPETITIONS):
    root = Path(root)
    catalog = read_json(root / 'notebooks/src/q8_catalog.json')
    definitions = [
        ('q2', Q2_METHODS, 'Not run', 'Missense feature extraction/evaluation pending. Validation selects C.'),
        ('q8', {tool['name']: tool['name'] for tool in catalog['tools']}, 'Not evaluated',
         'Surveyed tool; no pilot predictions exported.'),
        ('q9', Q9_METHODS, 'Not run', 'Training has not produced a completed evaluation.'),
    ]
    methods = {f'{question}:{key}': {'id': f'{question}:{key}', 'question': question.upper(), 'method': name,
                                   'status': status, 'note': note}
               for question, names, status, note in definitions for key, name in names.items()}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'methods': [], 'common': {},
              'cohort': None, 'bootstrap': {'repetitions': repetitions, 'seed': SEED}, 'errors': {}}
    try:
        context = load_context(root)
    except (OSError, ValueError, KeyError) as error:
        result['errors']['Q1'] = str(error)
        for row in methods.values():
            row.update(status='Unavailable', note='Frozen Q1 inputs unavailable or invalid; no metrics displayed.')
        result['methods'] = list(methods.values())
        return result
    validation = context['validation']
    result['cohort'] = {key: context[key] for key in ['protocol_sha256', 'vcf_exports']}
    result['cohort']['validation_variants'] = len(validation)
    predictions = {}
    for question, loader, required in [('q2', load_q2, 'validation_report.json'),
                                        ('q8', load_q8, 'baseline_provenance.json'), ('q9', load_q9, 'metrics.json')]:
        directory = root / 'notebooks/results' / question
        candidates = [row for row in methods.values() if row['question'].lower() == question and
                      (question != 'q8' or row['method'] == 'AlphaMissense')]
        if not (directory / required).exists():
            if question == 'q9' and (directory / 'readiness.json').exists():
                try:
                    readiness = read_json(directory / 'readiness.json')
                    if readiness['status'] == 'blocked':
                        for row in candidates:
                            row.update(status='Blocked', note='Last Q9 run: ' + '; '.join(readiness['blockers']))
                except (ValueError, KeyError):
                    pass
            continue
        try:
            arrays = loader(root, context)
            for key, values in arrays.items():
                identifier = f'{question}:{key}'
                predictions[identifier] = values
                methods[identifier].update(status='Available', note=(
                    'ClinVar calibration overlap unresolved; maximum matching transcript score.' if question == 'q8'
                    else 'Development result; validation participates in model selection.'))
        except (OSError, ValueError, KeyError, AssertionError) as error:
            result['errors'][question.upper()] = str(error)
            for row in candidates:
                row.update(status='Invalid / stale', note=str(error))
    # Future question notebooks can export this small documented contract.
    for path in sorted((root / 'notebooks/results').glob('q*/comparison_results.json')):
        question = path.parent.name
        candidates = {}
        try:
            spec = read_json(path)
            candidates = {f'{question}:{column}': {'id': f'{question}:{column}', 'question': question.upper(),
                          'method': name, 'status': 'Invalid / stale', 'note': ''}
                          for column, name in spec['methods'].items()}
            require(not set(candidates) & set(methods), 'Duplicate method IDs in custom comparison export')
            methods.update(candidates)
            require(spec['q1_protocol_sha256'] == context['protocol_sha256'], 'Export cohort is stale')
            csv = path.with_name('comparison_predictions.csv')
            verified(csv, spec['predictions_sha256'])
            arrays = align_predictions(pd.read_csv(csv), context, spec['methods'], allow_missing=True)
            for column, values in arrays.items():
                identifier = f'{question}:{column}'
                methods[identifier].update(status='Available', note=spec.get('limitations', ''))
                predictions[identifier] = values
        except (OSError, ValueError, KeyError) as error:
            result['errors'][question.upper()] = str(error)
            for row in candidates.values():
                row['note'] = str(error)
    labels, groups = validation.label.to_numpy(), validation.component.to_numpy()
    summary = summarize(labels, predictions, groups, repetitions)
    for identifier, row in methods.items():
        row.update(summary.get(identifier, {}))
        if identifier in predictions and not row.get('metrics'):
            row.update(status='Insufficient coverage', note='The scored subset needs both ClinVar classes. ' + row['note'])
    result['methods'] = list(methods.values())
    usable = {key: values for key, values in predictions.items() if summary[key]['metrics']}
    if len(usable) >= 2:
        mask = np.logical_and.reduce([np.isfinite(values) for values in usable.values()])
        if len(np.unique(labels[mask])) == 2:
            result['common'] = summarize(labels[mask], {name: values[mask] for name, values in usable.items()},
                                         groups[mask], repetitions)
        else:
            result['errors']['Common subset'] = 'The shared scored subset does not contain both classes.'
    return result


def format_metric(measured, key):
    metric = measured.get('metrics', {}).get(key)
    if not metric:
        return '—'
    interval = metric['ci95']
    suffix = f' [{interval[0]:.3f}, {interval[1]:.3f}]' if interval else ''
    return f'{metric["value"]:.3f}{suffix}'


def render(result, root=ROOT):
    """Return notebook display bundles and export a standalone comparison plot."""
    root = Path(root)
    rows = result['methods']
    notebooks = {p.stem.split('-')[0].upper(): p for p in (root / 'notebooks').glob('Q*.ipynb')}
    table = []
    for row in rows:
        path = notebooks.get(row['question'])
        link = (f'<a href="{html.escape(str(path.relative_to(root)))}">{row["question"]}</a>' if path else row['question'])
        table.append({'Method': html.escape(row['method']), 'Notebook': link,
                      'Scored / validation': f'{row["covered"]:,} / {row["total"]:,}' if 'covered' in row else '—',
                      'AUROC [95% CI]': format_metric(row, 'auroc'),
                      'Average precision [95% CI]': format_metric(row, 'average_precision')})
    cohort = result['cohort']
    intro = (f'**{cohort["validation_variants"]:,} missense validation variants · ClinVar labels**' if cohort
             else '**Frozen validation inputs unavailable**')
    bundles = [{'text/markdown': intro}, {'text/html': pd.DataFrame(table).to_html(index=False, escape=False, border=0)}]
    available = [row for row in rows if row.get('metrics')]
    if available:
        common = result['common']
        plotted = [row for row in available if not common or row['id'] in common]
        fig = Figure(figsize=(10, max(2.5, .55 * len(plotted) + 1.4)), layout='constrained')
        axes = fig.subplots(1, 2)
        for index, row in enumerate(plotted):
            measured = common[row['id']] if common else row
            for ax, key in zip(axes, ['auroc', 'average_precision']):
                metric = measured['metrics'][key]
                low, high = metric['ci95'] or [metric['value']] * 2
                ax.hlines(index, low, high, color='#28866b', linewidth=3)
                ax.plot(metric['value'], index, 'o', color='#17634d')
        for ax, title in zip(axes, ['AUROC', 'Average precision']):
            ax.set(yticks=range(len(plotted)), yticklabels=[f'{r["method"]} ({r["question"]})' for r in plotted],
                   xlim=(0, 1), ylim=(len(plotted) - .5, -.5), xlabel=title)
            ax.grid(axis='x', alpha=.2)
        axes[1].set_yticklabels([])
        title = (f'Common scored subset: {next(iter(common.values()))["total"]:,} variants' if common
                 else 'Available result · covered validation variants')
        fig.suptitle(title)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        atomic_write(root / 'notebooks/results/comparison/metrics.png', buffer.getvalue())
        bundles.append({'image/png': base64.b64encode(buffer.getvalue()).decode()})
    if result['common']:
        table = [{'Method': f'{row["method"]} ({row["question"]})',
                  'Shared variants': result['common'][row['id']]['total'],
                  'AUROC [95% CI]': format_metric(result['common'][row['id']], 'auroc'),
                  'Average precision [95% CI]': format_metric(result['common'][row['id']], 'average_precision')}
                 for row in available if row['id'] in result['common']]
        bundles.extend([{'text/markdown': '**Direct comparison on the same variants**'},
                        {'text/html': pd.DataFrame(table).to_html(index=False, border=0)}])
    notes = pd.DataFrame([{'Method': f'{r["method"]} ({r["question"]})', 'Details': r['note']} for r in rows])
    bundles.append({'text/html': '<details><summary>Provenance, missing results and limitations</summary>' +
                    notes.to_html(index=False, border=0) + '<pre>' + html.escape(json.dumps({
                        'cohort': cohort, 'source_errors': result['errors'], 'bootstrap': result['bootstrap'],
                        'generated_utc': result['generated_utc']}, indent=2)) + '</pre></details>'})
    if len(available) == 1:
        conclusion = f'Only **{available[0]["method"]}** currently has verified metrics. A ranking awaits the other methods’ results.'
    elif not available:
        conclusion = 'No verified metrics are currently available. Complete the source experiments or resolve the listed input checks.'
    elif not result['common']:
        conclusion = 'Several methods have results, but there is no shared scored subset with both classes for a direct comparison.'
    else:
        conclusion = 'Use the common-subset comparison to assess methods; coverage remains a separate limitation.'
    bundles.append({'text/markdown': '**Conclusion.** ' + conclusion + '\n\n'
                    'These are development results: validation participates in model selection. '
                    'AlphaMissense has unresolved ClinVar calibration overlap. The 95% intervals resample whole '
                    'Q1 components and do not correct selection bias or establish clinical validity.'})
    return bundles


def notebook_with_outputs(bundles):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Method comparison\n\n'
            'Compare methods on Q1’s frozen missense validation set. '
            'Run this notebook once to enable automatic refresh, then reload it after source results change.'),
        nbformat.v4.new_code_cell('from notebooks.src import comparison\ncomparison.show()',
                                 outputs=[nbformat.v4.new_output('display_data', data=bundle) for bundle in bundles]),
    ], metadata={'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}, 'comparison': {'generated': True}})
    for index, cell in enumerate(notebook.cells):
        cell.id = f'comparison-{index}'
    return notebook


def refresh(root=ROOT, repetitions=REPETITIONS):
    """Publish an executed comparison; source notebooks are never executed."""
    import fcntl
    from nbclient import NotebookClient
    root = Path(root)
    output = root / 'notebooks/results/comparison'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'refresh.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = source_signature(root)
        result = collect(root, repetitions)
        bundles = render(result, root)
        require(before == source_signature(root), 'Sources changed during refresh; retry after writes finish')
        result['sources'] = source_signature(root, content=True)
        atomic_write(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
        csv = pd.DataFrame([{key: row.get(key) for key in ['id', 'question', 'method', 'status', 'covered', 'total']} |
                            {metric: row.get('metrics', {}).get(metric, {}).get('value')
                             for metric in ['auroc', 'average_precision']} for row in result['methods']])
        atomic_write(output / 'methods.csv', csv.to_csv(index=False))
        # Execute the comparison's display cell in a fresh kernel. The private
        # payload avoids recursively refreshing while that cell is being run.
        payload = output / 'display.json'
        atomic_write(payload, json.dumps(bundles, allow_nan=False) + '\n')
        notebook = notebook_with_outputs([])
        client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        client.execute(env={**os.environ, 'PATHOGENITY_COMPARISON_DISPLAY': str(payload.resolve())})
        require(all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code'),
                'Comparison contains an unexecuted code cell')
        require(before == source_signature(root), 'Sources changed during notebook execution; retry')
        atomic_write(root / 'comparison.ipynb', nbformat.writes(notebook))
        return result, bundles


def show():
    from IPython.display import display
    from .comparison_watch import start
    payload = os.environ.get('PATHOGENITY_COMPARISON_DISPLAY')
    if payload:
        bundles = read_json(payload)
    else:
        start(ROOT)
        _, bundles = refresh()
    for bundle in bundles:
        display(bundle, raw=True)
