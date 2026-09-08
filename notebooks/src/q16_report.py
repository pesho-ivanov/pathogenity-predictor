"""Q16 displays and portable publication, preserving all earlier comparisons."""

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import comparison

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('notebooks/results/q16')
NOTEBOOK = Path('notebooks/Q16-lora-continuation.ipynb')
PUBLISHED = Path('notebooks/results/comparison/published/q16-results.json')
START, END = '<!-- q16-results:start -->', '<!-- q16-results:end -->'
QUESTION_START, QUESTION_END = '<!-- q16-question:start -->', '<!-- q16-question:end -->'
NAMES = {'continuation': 'Q16 continued LoRA', 'q14_parent': 'Q14 parent LoRA',
         'frozen_control': 'Matched frozen classifier'}
LIMITATIONS = (
    'The 2,048-variant sample selects checkpoints; full validation reports the frozen selection. '
    'The full partition and its subsets have been evaluated before and are development data, '
    'not an untouched test set. Component-bootstrap intervals do not correct repeated selection '
    'or establish clinical validity. Pretraining, homology and external-data overlap remain unresolved. '
    'Single-variant BF16 scoring is numerically sensitive; Q16 uses its verified batch-32 workflow.'
)


def _verified_results():
    from . import q16
    return q16.verified_results()


def _point(value):
    return value['value'] if isinstance(value, dict) else value


def _metric(value, signed=False):
    number = _point(value)
    digits = '+.4f' if signed else '.3f'
    text = format(number, digits)
    interval = value.get('ci95') if isinstance(value, dict) else None
    if interval is not None:
        text += f' [{format(interval[0], digits)}, {format(interval[1], digits)}]'
    return text


def _table(metrics):
    lines = ['| Model | AUROC [95% CI] | Average precision [95% CI] |', '| --- | ---: | ---: |']
    for branch, label in NAMES.items():
        lines.append(f'| {label} | {_metric(metrics[branch]["auroc"])} | '
                     f'{_metric(metrics[branch]["average_precision"])} |')
    return '\n'.join(lines)


def _display_json(title, value):
    from IPython.display import HTML, display
    display(HTML(f'<details><summary>{html.escape(title)}</summary><pre>'
                 + html.escape(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
                 + '</pre></details>'))


def _provenance(result, root):
    protocol = comparison.read_json(root / OUTPUT / 'protocol.json')
    context = comparison.load_context(root)
    comparison.require(context['scope'] == 'full'
        and protocol['q1_protocol_sha256'] == context['protocol_sha256']
        and protocol['vcf_exports'] == context['vcf_exports'], 'Q16 Q1 cohort identity differs')
    comparison.require(result['identity'] == hashlib.sha256(json.dumps(
        protocol, sort_keys=True, allow_nan=False).encode()).hexdigest(), 'Q16 protocol identity differs')
    for relative, checksum in protocol['sources'].items():
        comparison.verified(root / relative, checksum)
    cohort = {key: context[key] for key in ['scope', 'protocol_sha256', 'vcf_exports', 'artifacts']}
    cohort.update(validation_variants=len(context['validation']),
                  clinvar_date=context['config']['clinvar_date'])
    return {'identity': result['identity'], 'protocol_sha256': comparison.digest(root / OUTPUT / 'protocol.json'),
            'q1_identity': protocol['q1_identity'], 'cohort': cohort, 'sources': protocol['sources']}


def show_results():
    """Display actual full/sample results and retain the exact publication evidence."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    result = _verified_results()
    history = comparison.read_json(ROOT / OUTPUT / 'training_history.json')
    evaluations = history['evaluations']
    comparison.require(bool(evaluations), 'Q16 training evaluations are missing')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for branch, label in NAMES.items():
        for axis, metric in zip(axes, ['auroc', 'average_precision']):
            axis.plot([row['steps'] for row in evaluations],
                      [_point(row[branch][metric]) for row in evaluations],
                      marker='o' if branch == 'continuation' else None,
                      linestyle='-' if branch == 'continuation' else '--', label=label)
    for axis, label in zip(axes, ['AUROC', 'Average precision']):
        axis.set(xlabel='Additional LoRA updates', ylabel='Sample validation ' + label)
        axis.legend(fontsize=8)
    fig.savefig(ROOT / OUTPUT / 'training_curves.png', dpi=160)
    plt.show()
    display(Markdown(f'**Full validation: {result["validation_variants"]:,} variants**\n\n'
                     + _table(result['metrics'])))
    display(Markdown('<details><summary>Checkpoint-selection sample</summary>\n\n'
                     + _table(result['sample_metrics']) + '\n\n</details>'))
    lines = []
    for label, key in [('Q14 parent LoRA', 'paired_continuation_minus_q14_parent'),
                       ('frozen classifier', 'paired_continuation_minus_frozen_control')]:
        paired = result[key]
        lines.append(f'Continuation minus {label}: AUROC **{_metric(paired["auroc"], True)}**, '
                     f'AP **{_metric(paired["average_precision"], True)}**.')
    for key, label in [('outside_selection', 'Outside the selection sample'),
                       ('unseen_components', 'Components absent from the selection sample')]:
        subset = result['subsets'][key]
        paired = subset.get('paired_continuation_minus_q14_parent')
        if paired:
            count = subset.get('variants', subset.get('validation_variants'))
            count_text = f' ({count:,} variants)' if isinstance(count, int) else ''
            lines.append(f'{label}{count_text}: continuation minus Q14 AUROC '
                         f'**{_metric(paired["auroc"], True)}**, AP '
                         f'**{_metric(paired["average_precision"], True)}**.')
    display(Markdown('\n\n'.join(lines)))
    _display_json('Completed Q16 measurements and integrity checks', result)
    _display_json('Q16 portable provenance', _provenance(result, ROOT))


def conclusion(result):
    parent = result['paired_continuation_minus_q14_parent']
    selected = NAMES[result['selected_model']]
    auroc, ap = parent['auroc']['value'], parent['average_precision']['value']
    if auroc > 0 and ap >= 0:
        observation = 'The tested continuation improved full-validation AUROC without reducing AP versus Q14.'
    else:
        observation = 'The tested continuation did not improve both full-validation ranking metrics versus Q14.'
    return (f'{observation} Continuation minus Q14: AUROC **{_metric(parent["auroc"], True)}**, '
            f'AP **{_metric(parent["average_precision"], True)}**. '
            f'The sample-based decision retains **{selected}**. '
            'The best nonzero-update continuation is chosen by sample AUROC, then AP; it replaces Q14 '
            'only if sample AUROC strictly increases and AP does not decrease. Full results do not change '
            f'that decision. {LIMITATIONS}')


def show_conclusion():
    from IPython.display import Markdown, display
    display(Markdown('**Conclusion.** ' + conclusion(_verified_results())))


def _relative(root, name, directory=None):
    path = Path(name)
    comparison.require(not path.is_absolute() and '..' not in path.parts, 'Q16 evidence path is invalid')
    if directory is not None:
        comparison.require(path.parent == directory, 'Q16 evidence directory is invalid')
    comparison.require((root / path).resolve().is_relative_to(root.resolve()), 'Q16 evidence escapes its root')
    return root / path


def _notebook_payload(root, evidence):
    path = _relative(root, evidence['path'], Path('notebooks'))
    comparison.verified(path, evidence['sha256'])
    notebook = comparison.completed_notebook(path)
    return comparison.notebook_result_payload(notebook, evidence['cell'], evidence['output'])


def _validate_result(result, total):
    comparison.require(result.get('status') == 'complete' and result.get('scope') == 'full_validation'
        and result.get('validation_variants') == total and total > 0, 'Q16 full validation is incomplete')
    for flag in ['head_fixed_verified', 'scaler_fixed_verified', 'frozen_unchanged',
                 'reloaded_predictions_verified', 'parent_predictions_reproduced']:
        comparison.require(result.get(flag) is True, f'Q16 integrity check failed: {flag}')
    comparison.require(result['selected_model'] in ['continuation', 'q14_parent'], 'Invalid Q16 selection')
    comparison.require(result['metrics']['selected'] == result['metrics'][result['selected_model']],
                       'Q16 selected metrics differ from its declared model')
    for branch in [*NAMES, 'selected']:
        for name in ['auroc', 'average_precision']:
            metric = result['metrics'][branch][name]
            values = [metric['value'], *metric['ci95']]
            comparison.require(len(values) == 3 and np.isfinite(values).all()
                and all(0 <= x <= 1 for x in values) and values[1] <= values[2], 'Invalid Q16 metric interval')


def _verify_record(root, record, check_local=True):
    comparison.require(record['schema_version'] == 1 and record['question'] == 'q16', 'Invalid Q16 publication schema')
    result, provenance = record['result'], record['provenance']
    comparison.require(record['notebook']['path'] == str(NOTEBOOK)
        and record['provenance_notebook']['path'] == str(NOTEBOOK)
        and record['notebook']['sha256'] == record['provenance_notebook']['sha256'], 'Q16 source notebook differs')
    comparison.require(_notebook_payload(root, record['notebook']) == result
        and _notebook_payload(root, record['provenance_notebook']) == provenance,
        'Q16 published measurements or provenance differ from executed outputs')
    comparison.require(provenance['identity'] == result['identity']
        and provenance['protocol_sha256'] == record['local_artifacts'][str(OUTPUT / 'protocol.json')],
        'Q16 published protocol identity differs')
    cohort = provenance['cohort']
    published_cohort = comparison.published_results(root)['cohort']
    comparison.require(all(cohort[key] == published_cohort[key] for key in
        ['scope', 'clinvar_date', 'validation_variants', 'vcf_exports'])
        and all(cohort['artifacts'].get(key) == value for key, value in published_cohort['artifacts'].items()),
        'Q16 published cohort differs from completed Q1 evidence')
    for name in ['protocol.json', 'full/protocol.json']:
        if (root / 'notebooks/results/q1' / name).exists():
            context = comparison.load_context(root)
            comparison.require(context['scope'] == 'full' and context['vcf_exports'] == cohort['vcf_exports']
                and context['config']['clinvar_date'] == cohort['clinvar_date']
                and len(context['validation']) == cohort['validation_variants']
                and all(context['artifacts'].get(key) == value for key, value in cohort['artifacts'].items()),
                'Q16 published cohort differs from local Q1 inputs')
            break
    for name, checksum in cohort['artifacts'].items():
        path = _relative(root, str(Path('notebooks/results/q1/full') / name))
        if path.exists():
            comparison.verified(path, checksum)
    _validate_result(result, cohort['validation_variants'])
    for relative, checksum in provenance['sources'].items():
        comparison.verified(_relative(root, relative), checksum)
    timing = record['run_status']
    status_path = _relative(root, timing['path'], PUBLISHED.parent)
    comparison.verified(status_path, timing['sha256'])
    status = comparison.read_json(status_path)
    seconds = status.get('notebook_execution_seconds')
    comparison.require(status.get('status') == 'complete'
        and status.get('notebook_sha256') == record['notebook']['sha256']
        and isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        and np.isfinite(seconds) and seconds >= 0 and seconds == record['runtime_seconds'],
        'Q16 runtime does not match its completed notebook status')
    comparison.require(bool(record['local_artifacts']), 'Q16 local artifact registry is empty')
    if check_local and any(_relative(root, name).exists() for name in record['local_artifacts']):
        for relative, checksum in record['local_artifacts'].items():
            comparison.verified(_relative(root, relative), checksum)
    return record


def published_results(root=ROOT):
    """Verify the portable record; partially present local evidence never falls back."""
    root = Path(root)
    return _verify_record(root, comparison.read_json(root / PUBLISHED))


def _section(record):
    result = record['result']
    return ('## Q16. Does longer fine-tuning improve on Q14?\n\n'
        '[Q16 continued LoRA](notebooks/Q16-lora-continuation.ipynb) continues the successful Q14 adapters '
        'with the original classifier and scaler fixed. Train on all '
        f'**{result["training_variants"]:,} available training variants**, for up to **three epochs** '
        'at learning rate **3e-5**, subject to the original four-hour deadline.\n\n'
        f'**Matched full validation: {result["validation_variants"]:,} missense variants.**\n\n'
        + _table(result['metrics']) + '\n\n' + conclusion(result) + '\n\n'
        f'Fresh notebook execution: **{record["runtime_seconds"]/60:.1f} minutes**. '
        f'Elapsed from the original four-hour request through backend results: **{result["seconds"]/60:.1f} minutes**. '
        'The notebook duration includes its input checks, training, full validation and reporting; '
        'inherited Q14 training and initial environment acquisition are excluded.\n\n'
        '<details><summary>Selection sample and reproducibility</summary>\n\n'
        + _table(result['sample_metrics']) + '\n\n'
        f'Training examples processed: **{result["training_examples_seen"]:,}**; unique training variants: '
        f'**{result["actual_unique_training_variants"]:,} / {result["training_variants"]:,}**; '
        f'optimizer updates: **{result["optimizer_steps"]:,}**. '
        'The preserved [Q16 result record](notebooks/results/comparison/published/q16-results.json) '
        'pins the executed notebook, displayed metrics/provenance, source hashes, Q1 cohort and original '
        'measured run status. Earlier comparison rows remain in their original section.\n\n</details>')


def _replace_section(text, section, *, before=None, start=START, end=END):
    comparison.require(text.count(start) == text.count(end) and text.count(start) <= 1,
                       'Q16 README markers are malformed')
    block = start + '\n' + section + '\n' + end
    if start in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + block + suffix
    elif before is not None and before in text:
        offset = text.index(before)
        text = text[:offset] + block + '\n\n' + text[offset:]
    else:
        text += ('' if text.endswith('\n') else '\n') + '\n' + block + '\n'
    return text


def _publish_readmes(root, record):
    docs = ('## Q16 artifacts: longer LoRA continuation\n\n'
        '[Q16](../Q16-lora-continuation.ipynb) continues Q14 adapters, keeps the classifier/scaler fixed, '
        'selects a candidate on the existing 2,048-variant sample, and compares it with Q14 and the frozen '
        'classifier on all 17,927 validation variants. Full validation never changes the sample decision.\n\n'
        '`q16/` contains the frozen protocol/input checks, training history and exposure records, '
        'selected and best-continuation checkpoints, full `validation_predictions.csv`, `metrics.json`, '
        '`training_curves.png`, and the original `run_status.json`. Actual failed attempts remain preserved.\n\n'
        '`publication_status.json`, written after publication, records the actual end-to-end completion '
        'or publication failure against the original deadline. It is separate from the hash-bound notebook '
        'completion status and is not inserted into the earlier result artifact registry.\n\n'
        '`comparison/published/q16-results.json` retains the full result and displayed provenance with '
        'a checksum-pinned executed notebook and byte-for-byte copy of its measured run status. '
        'Run `python -m notebooks.src.q16_report` to verify that evidence and refresh only Q16’s marked '
        'README sections. A partial or corrupt local Q16 result is an error, never a fallback to cached numbers. '
        'The original comparison registry and its earlier method rows are unchanged.\n\n' + LIMITATIONS)
    question = ('### [Q16. Does longer fine-tuning improve on Q14?](notebooks/Q16-lora-continuation.ipynb)\n\n'
        'Continue Q14’s successful adapters at learning rate **3e-5**, using all **46,888 training variants** '
        'for up to **three epochs** within the original four-hour deadline. Keep the classifier and scaler '
        'fixed. Select on the existing 2,048-variant sample, then score all 17,927 validation variants '
        'with the continuation, Q14 parent and frozen classifier; preserve negative outcomes and paired '
        'component intervals. See the [Q16 results](#q16-does-longer-fine-tuning-improve-on-q14).\n\n'
        'Run `.venv/bin/python -m notebooks.src.refresh_q16` with the prepared **Lida** kernel. '
        'The runner preserves prior attempts and saves only a fully executed notebook.')
    readme = root / 'README.md'
    text = _replace_section(readme.read_text(), _section(record), before='## Experiment details\n')
    text = _replace_section(text, question, before='## Setup\n', start=QUESTION_START, end=QUESTION_END)
    artifact_docs = root / 'notebooks/results/README.md'
    updates = [(readme, text), (artifact_docs, _replace_section(artifact_docs.read_text(), docs))]
    for path, text in updates:
        comparison.atomic_write(path, text)


def publish(root=ROOT):
    """Publish only after a genuinely executed notebook and completed result exist."""
    root = Path(root)
    result = _verified_results()
    provenance = _provenance(result, root)
    _validate_result(result, provenance['cohort']['validation_variants'])
    notebook = comparison.result_notebook_evidence(root, NOTEBOOK, result)
    source_evidence = comparison.result_notebook_evidence(root, NOTEBOOK, provenance)
    directory = root / OUTPUT
    context = comparison.load_context(root)
    frame = pd.read_csv(directory / 'validation_predictions.csv')
    arrays = comparison.align_predictions(frame, context, [*NAMES, 'selected'])
    comparison.require(np.array_equal(arrays['selected'], arrays[result['selected_model']]),
                       'Q16 selected predictions differ from its declared model')
    for branch, values in arrays.items():
        for name, function in [('auroc', roc_auc_score), ('average_precision', average_precision_score)]:
            comparison.require(np.isclose(function(frame.set_index('variant_key').loc[
                context['validation'].variant_key, 'label'], values), result['metrics'][branch][name]['value'],
                atol=1e-12, rtol=0), 'Q16 metrics differ from its full predictions')
    files = dict(result['artifacts'])
    for name in ['metrics.json', 'protocol.json', 'run_status.json', 'validation_predictions.csv', 'training_history.json']:
        actual = comparison.digest(directory / name)
        comparison.require(name not in files or files[name] == actual, 'Q16 artifact registry differs')
        files[name] = actual
    for name, checksum in files.items():
        comparison.verified(_relative(root, str(OUTPUT / name)), checksum)
    payload = (directory / 'run_status.json').read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    status = json.loads(payload)
    seconds = status.get('notebook_execution_seconds')
    comparison.require(status.get('status') == 'complete' and status.get('notebook_sha256') == notebook['sha256']
        and isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        and np.isfinite(seconds) and seconds >= 0,
                       'Q16 notebook completion status is missing or stale')
    status_relative = PUBLISHED.parent / f'q16-status-{checksum[:12]}.json'
    target = root / status_relative
    comparison.require(not target.exists() or target.read_bytes() == payload, 'Immutable Q16 runtime evidence differs')
    record = {'schema_version': 1, 'question': 'q16', 'result': result, 'provenance': provenance,
        'notebook': notebook, 'provenance_notebook': source_evidence,
        'run_status': {'path': str(status_relative), 'sha256': checksum},
        'runtime_seconds': status['notebook_execution_seconds'],
        'local_artifacts': {str(OUTPUT / name): checksum for name, checksum in files.items()}}
    if not target.exists():
        comparison.atomic_write(target, payload)
    _verify_record(root, record)
    comparison.atomic_write(root / PUBLISHED, json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    _publish_readmes(root, record)
    return record


if __name__ == '__main__':
    _publish_readmes(ROOT, published_results())
