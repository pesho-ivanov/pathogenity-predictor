"""Collect verified validation predictions and update the main README comparison.

This module never executes a source notebook or imports a model. Local prediction
exports take precedence; pinned, completed notebook results survive missing caches.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPETITIONS = 1000
SEED = 42
SECTION_START = '<!-- comparison:start -->'
SECTION_END = '<!-- comparison:end -->'
Q2_METHODS = {'zero_shot': 'Evo2 1B base zero-shot (Vortex, FP8)'}
# One BioNeMo representative in the README; the full experiment stays in Q9.
Q9_METHODS = {'frozen': 'Evo2 1B base frozen head (BioNeMo, BF16)',
              'sequence': 'Sequence baseline (Evo2 1B experiment)'}
# Keep the archived 1B experiment in result records, outside the README tables.
README_EXCLUDED_METHODS = {'q2:zero_shot', 'q9:frozen', 'q9:sequence'}
Q11_METHODS = {'fine_tuned': 'Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp)'}
PUBLISHED = Path('notebooks/results/comparison/published/results.json')
NOTEBOOKS = {'Q2': 'Q2-evo2-classifier.ipynb', 'Q8': 'Q8-existing-tools.ipynb',
             'Q11': 'Q11-evo2-lora.ipynb', 'Q12': 'Q12-lora-validation.ipynb',
             'Q13': 'Q13-lora-validation.ipynb', 'Q14': 'Q14-lora-validation.ipynb',
             'Q15': 'Q15-lora-validation.ipynb'}
LORA_FULL_METHODS = {
    question: {'lora': f'Evo2 7B {question.upper()} LoRA ({blocks}, rank 8, 512 bp)',
               'strongest_control': f'Evo2 7B {question.upper()} strongest frozen classifier (magnitude features, 512 bp)'}
    for question, blocks in [('q13', 'block 30'), ('q14', 'blocks 29 and 30'), ('q15', 'blocks 29 and 30')]
}
LORA_EXPLORATION_NOTEBOOKS = {'q13': 'Q13-lora-optimization.ipynb', 'q14': 'Q14-layer-adapters.ipynb',
                            'q15': 'Q15-prefix-ranking.ipynb'}
Q8_NOTES = {
    'AlphaMissense': 'ClinVar calibration overlap unresolved; maximum matching transcript score.',
    'REVEL': 'HGMD and constituent-tool training overlap with ClinVar unresolved; maximum exact-allele score across transcript annotations.',
    'SIFT4G': 'dbNSFP4.9a; 1 minus the minimum raw SIFT4G score. Evolutionary sequence exposure is unaudited.',
    'PolyPhen-2': 'HumVar model from dbNSFP4.9a; maximum raw score. Known disease training variants may overlap ClinVar.',
    'EVE': 'dbNSFP4.9a continuous EVE score; maximum across matches, no confidence-category filtering. Limited protein/position coverage.',
}
Q8_EXPORTS = {
    'AlphaMissense': ('', 'q8_baseline.py', 'am_pathogenicity'),
    'REVEL': ('revel', 'q8_revel.py', 'revel'),
    'SIFT4G': ('sift4g', 'q8_remaining.py', 'score'),
    'PolyPhen-2': ('polyphen2', 'q8_remaining.py', 'score'),
    'EVE': ('eve', 'q8_remaining.py', 'score'),
}
# Original method papers for the Q8 reference predictors.
METHOD_PAPERS = {
    'q8:SIFT4G': '[Vaser et al. (2016)](https://doi.org/10.1038/nprot.2015.123)',
    'q8:PolyPhen-2': '[Adzhubei et al. (2010)](https://doi.org/10.1038/nmeth0410-248)',
    'q8:REVEL': '[Ioannidis et al. (2016)](https://doi.org/10.1016/j.ajhg.2016.08.016)',
    'q8:AlphaMissense': '[Cheng et al. (2023)](https://doi.org/10.1126/science.adg7492)',
    'q8:EVE': '[Frazer et al. (2021)](https://doi.org/10.1038/s41586-021-04043-8)',
    'q8:PrimateAI-3D': '[Gao et al. (2023)](https://doi.org/10.1126/science.abn8197)',
}
SOURCE_FILES = {
    'q1': ['protocol.json', 'split_manifest.csv', 'validation_labels.csv',
           'full/protocol.json', 'full/split_manifest.csv', 'full/validation_labels.csv'],
    'q2': ['protocol.json', 'validation_report.json', 'validation_predictions.csv',
           '7b/score_manifest.json'],
    'q8': [str(Path(folder) / name) for folder, _, _ in Q8_EXPORTS.values()
           for name in ['baseline_protocol.json', 'baseline_provenance.json', 'validation_metrics.json', 'pilot_evaluation.csv']]
          + ['primateai3d/access_status.json', 'full/protocol.json', 'full/leakage_checks.json',
             'full/primateai3d/access_status.json']
          + [str(Path('full') / folder / name) for folder, _, _ in Q8_EXPORTS.values()
             for name in ['baseline_protocol.json', 'baseline_provenance.json',
                          'validation_metrics.json', 'validation_predictions.csv']],
    'q9': ['protocol.json', 'readiness.json', 'metrics.json', 'validation_predictions.npz',
           'baseline_selection.json', 'frozen_features.json'],
    'q10': ['feature_manifest.json'],
    'q11': ['protocol.json', 'run_status.json', 'metrics.json', 'comparison_results.json',
            'comparison_predictions.csv'],
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


def completed_notebook(path):
    """Require actual ordered execution; outputs supply result evidence separately."""
    notebook = read_json(path)
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    require(bool(cells) and all(''.join(cell['source']).strip() for cell in cells)
            and [cell.get('execution_count') for cell in cells] == list(range(1, len(cells) + 1))
            and not any(output.get('output_type') == 'error'
                        for cell in cells for output in cell.get('outputs', [])),
            'Full-validation notebook is not completely executed')
    return notebook


def notebook_result_payload(notebook, cell_index, output_index):
    value = notebook['cells'][cell_index]['outputs'][output_index]['data']['text/html']
    value = ''.join(value) if isinstance(value, list) else value
    match = re.search(r'<pre>(.*?)</pre>', value, re.S)
    require(match is not None, 'Full-validation notebook result evidence is missing')
    return json.loads(html.unescape(match[1]))


def result_notebook_evidence(root, relative, completion):
    """Find the complete result object displayed by the executed source notebook."""
    root = Path(root)
    relative = Path(relative)
    path = root / relative
    notebook = completed_notebook(path)
    for cell_index, cell in enumerate(notebook['cells']):
        for output_index, _ in enumerate(cell.get('outputs', [])):
            try:
                payload = notebook_result_payload(notebook, cell_index, output_index)
            except (ValueError, KeyError, TypeError):
                continue
            if payload == completion:
                return {'path': str(relative), 'sha256': digest(path),
                        'cell': cell_index, 'output': output_index}
    raise ValueError('Executed full-validation notebook does not display the completed result object')


def full_lora_notebook_evidence(root, question, completion):
    return result_notebook_evidence(root, Path('notebooks') / NOTEBOOKS[question.upper()], completion)


def verify_full_lora_export(root, question, directory, spec, context):
    """Bind Q13/Q14/Q15 benchmark rows to full predictions and an executed notebook."""
    label = question.upper()
    require(spec.get('scope') == 'full_validation' and spec.get('require_complete') is True
            and context.get('scope') == 'full' and spec.get('vcf_exports') == context['vcf_exports'],
            f'{label} sampled exploration cannot enter the full-cohort comparison')
    require(spec['methods'] == LORA_FULL_METHODS[question],
            f'{label} must export LoRA and its strongest frozen classifier together')
    required = {'full/metrics.json', 'full/protocol.json', 'full/run_status.json',
                'full/validation_predictions.csv', 'protocol.json', 'metrics.json', 'selected_model.pt', 'run_status.json'}
    require(required <= set(spec.get('artifacts', {})), f'{label} completed full-validation evidence is incomplete')
    completion = read_json(directory / 'full/metrics.json')
    protocol = read_json(directory / 'full/protocol.json')
    parent = read_json(directory / 'metrics.json')
    parent_run = read_json(directory / 'run_status.json')
    run = read_json(directory / 'full/run_status.json')
    require(completion.get('scope') == 'full_validation' and completion.get('status') == 'complete'
            and completion.get('validation_variants') == len(context['validation'])
            and completion.get('reload_verified') is True and completion.get('frozen_unchanged') is True
            and completion.get('original_selection_scores_reproduced') is True,
            f'{label} full validation is incomplete or failed integrity checks')
    require(protocol.get('question') == question and protocol.get('q1_protocol_sha256') == context['protocol_sha256']
            and protocol.get('vcf_exports') == context['vcf_exports']
            and protocol.get('sources') == spec.get('sources') and bool(protocol.get('sources')),
            f'{label} full-validation protocol or sources differ from its export')
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(completion.get('identity') == fingerprint == spec.get('full_identity'),
            f'{label} full-validation identity is stale')
    require(parent.get('status') == 'complete' and parent.get('scope') == 'sampled_validation'
            and parent.get('promotion_passed') is True and parent.get('selected_model') == 'lora'
            and parent.get('identity') == protocol.get('parent_identity') == spec.get('parent_identity')
            and protocol.get('parent_metrics_sha256') == spec['artifacts']['metrics.json']
            and protocol.get('parent_selected_model_sha256') == spec['artifacts']['selected_model.pt']
            and protocol.get('parent_protocol_sha256') == spec['artifacts']['protocol.json'],
            f'{label} full validation is not bound to its promoted parent checkpoint')
    require(completion.get('artifacts', {}).get('validation_predictions.csv')
            == spec['artifacts']['full/validation_predictions.csv'] == spec['predictions_sha256'],
            f'{label} comparison predictions differ from completed full predictions')
    require(all(spec['artifacts'].get('full/' + name) == sha
                for name, sha in completion.get('artifacts', {}).items()),
            f'{label} full-validation artifact registry is incomplete')
    exploration = spec.get('exploration_notebook', {})
    require(exploration.get('path') == str(Path('notebooks') / LORA_EXPLORATION_NOTEBOOKS[question]),
            f'{label} completed exploration notebook is missing')
    verified(root / exploration['path'], exploration['sha256'])
    require(notebook_result_payload(completed_notebook(root / exploration['path']),
                                   exploration['cell'], exploration['output']) == parent
            and parent_run.get('status') == 'complete'
            and parent_run.get('notebook_sha256') == exploration['sha256'],
            f'{label} exploration notebook or completion record is stale')
    evidence = spec.get('notebook', {})
    require(evidence.get('path') == str(Path('notebooks') / NOTEBOOKS[label]),
            f'{label} must cite its full-validation notebook')
    verified(root / evidence['path'], evidence['sha256'])
    notebook = completed_notebook(root / evidence['path'])
    require(notebook_result_payload(notebook, evidence['cell'], evidence['output']) == completion,
            f'{label} notebook evidence differs from completed full metrics')
    require(run.get('status') == 'complete' and run.get('question') == question
            and run.get('notebook_sha256') == evidence['sha256'],
            f'{label} full notebook completion record is missing or stale')
    require(type(completion.get('confirmation_passed')) is bool
            and spec.get('confirmation_passed') == completion['confirmation_passed'],
            f'{label} confirmation outcome differs from completed results')
    stages = spec.get('runtime_stages', {})
    require(stages.get('exploration_seconds') == parent_run.get('notebook_execution_seconds')
            and stages.get('full_validation_seconds') == run.get('notebook_execution_seconds')
            and all(isinstance(stages.get(key), (int, float)) and not isinstance(stages[key], bool)
                    and np.isfinite(stages[key]) and stages[key] >= 0
                    for key in ['exploration_seconds', 'full_validation_seconds']),
            f'{label} runtime stages do not match completed notebook measurements')
    seconds = stages['exploration_seconds'] + stages['full_validation_seconds']
    require(set(spec.get('runtimes', {})) == set(spec['methods'])
            and all(value.get('seconds') == seconds for value in spec['runtimes'].values()),
            f'{label} combined runtime differs from its measured stages')


def publish_full_lora(question, root=ROOT):
    """Preserve two completed full-cohort rows without changing other results.

    Status files are copied byte-for-byte to immutable checksum-named files.
    The registry is replaced last, so an interrupted publication cannot change
    evidence referenced by its preceding version.
    """
    root = Path(root)
    require(question in LORA_FULL_METHODS, 'Unsupported full LoRA publication')
    directory = root / 'notebooks/results' / question
    context = load_context(root)
    spec = read_json(directory / 'comparison_results.json')
    require(spec['q1_protocol_sha256'] == context['protocol_sha256'], 'Export cohort is stale')
    for name, checksum in spec.get('sources', {}).items():
        verified(root / name, checksum)
    for name, checksum in spec.get('artifacts', {}).items():
        verified(directory / name, checksum)
    verify_full_lora_export(root, question, directory, spec, context)
    predictions = directory / 'comparison_predictions.csv'
    verified(predictions, spec['predictions_sha256'])
    arrays = align_predictions(pd.read_csv(predictions), context, spec['methods'])
    completion = read_json(directory / 'full/metrics.json')
    outcome = 'met' if completion['confirmation_passed'] else 'not met'
    require(f'confirmation rule {outcome}.' in spec.get('limitations', ''),
            'Published note must retain the actual full-validation confirmation outcome')
    labels = context['validation'].label.to_numpy()
    for branch, values in arrays.items():
        for metric, function in [('auroc', roc_auc_score), ('average_precision', average_precision_score)]:
            require(np.isclose(completion['metrics'][branch][metric]['value'], function(labels, values),
                               atol=1e-12, rtol=0),
                    'Published full metrics differ from verified predictions')

    saved = read_json(root / PUBLISHED)
    owned = {f'{question}:{branch}' for branch in LORA_FULL_METHODS[question]}
    prior = published_results(root, context)
    # A fresh rerun has already replaced its own canonical notebooks. Stale
    # records for those two IDs are replaceable; all other evidence must pass.
    require(not set(prior['errors']) - owned, 'Existing published competitor evidence is invalid')
    require(len({row['id'] for row in saved['methods']}) == len(saved['methods']),
            'Published method IDs are duplicated')
    require(not owned & set(saved['published_common']),
            'Full LoRA publication cannot replace shared-subset evidence')
    runtime_evidence, copies = [], []
    for stage, source, evidence in [
            ('exploration', directory / 'run_status.json', spec['exploration_notebook']),
            ('full', directory / 'full/run_status.json', spec['notebook'])]:
        payload = source.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        relative = PUBLISHED.parent / f'{question}-{stage}-status-{checksum[:12]}.json'
        target = root / relative
        require(not target.exists() or target.read_bytes() == payload,
                'Immutable published runtime evidence already differs')
        copies.append((target, payload))
        runtime_evidence.append({'path': str(relative), 'sha256': checksum, 'notebook': evidence['path']})
    evidence = spec['notebook']
    local = ['comparison_results.json', 'comparison_predictions.csv'] + sorted(
        name for name in spec['artifacts'] if name.startswith('full/'))
    records = []
    for branch, method in LORA_FULL_METHODS[question].items():
        runtime = load_runtime(root, f'{question}:{branch}', spec)
        require(runtime is not None, 'Published full LoRA runtime is missing')
        records.append({'id': f'{question}:{branch}', 'question': question.upper(), 'method': method,
            'covered': completion['validation_variants'], 'total': completion['validation_variants'],
            'metrics': completion['metrics'][branch], 'note': spec['limitations'],
            'notebook': evidence['path'], **runtime,
            'local_artifacts': [str((directory / name).relative_to(root)) for name in local],
            'evidence': {'kind': 'json', 'cell': evidence['cell'], 'output': evidence['output'],
                         'metrics_path': ['metrics', branch], 'count_path': ['validation_variants']},
            'runtime_status_evidence': runtime_evidence})
    saved['methods'] = [row for row in saved['methods'] if row['id'] not in owned] + records
    for notebook in [spec['exploration_notebook'], spec['notebook']]:
        saved['notebook_sha256'][notebook['path']] = notebook['sha256']
    for target, payload in copies:
        if not target.exists():
            atomic_write(target, payload)
    atomic_write(root / PUBLISHED, json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return records


def source_paths(root=ROOT):
    """Only small result inputs, source notebooks and provenance-relevant code."""
    root = Path(root)
    paths = set((root / 'notebooks').glob('Q*.ipynb'))
    paths.update((root / 'notebooks/src').glob('q*.py'))
    paths.add(root / 'notebooks/src/q8_catalog.json')
    paths.add(root / 'notebooks/src/comparison.py')
    paths.add(root / PUBLISHED)
    paths.update((root / PUBLISHED.parent).glob('*.md'))
    paths.update((root / PUBLISHED.parent).glob('*.json'))
    for question, names in SOURCE_FILES.items():
        paths.update(root / 'notebooks/results' / question / name for name in names)
    paths.update(root / 'data' / name for name in ['clinvar-train-pilot.vcf', 'clinvar-test-pilot.vcf',
                                                'clinvar-train.vcf', 'clinvar-test.vcf'])
    for path in (root / 'notebooks/results').glob('q*/comparison_results.json'):
        paths.add(path)
        # This is the documented filename for future notebooks' prediction exports.
        paths.add(path.with_name('comparison_predictions.csv'))
        try:
            spec = read_json(path)
            paths.update(path.parent / name for name in spec.get('artifacts', {}))
            paths.update(root / name for name in spec.get('sources', {}))
        except (OSError, ValueError):
            pass  # The collector reports malformed exports explicitly.
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
    # New experiments use the full cohort. Never fall back to pilot results when
    # a published full protocol fails validation.
    scope = 'pilot'
    if (directory / 'full').exists():
        directory = directory / 'full'
        scope = 'full'
    protocol = read_json(directory / 'protocol.json')
    require(protocol['config']['assembly'] == 'GRCh38' and
            'SO:0001583' in protocol['config']['eligibility'], 'Q1 must identify the missense GRCh38 cohort')
    for name in ['split_manifest.csv', 'validation_labels.csv']:
        verified(directory / name, protocol['artifacts'][name])
    for name, checksum in protocol['vcf_exports'].items():
        verified(root / 'data' / name, checksum)
    pilot = pd.read_csv(directory / 'split_manifest.csv',
                        usecols=['variant_key', 'split', 'component', 'MC'])
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
            'artifacts': protocol['artifacts'],
            'validation': validation, 'pilot_keys': pilot.variant_key.tolist(),
            'config': protocol['config'], 'scope': scope}


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


def load_q8(root, context, tool='AlphaMissense'):
    require(tool in Q8_EXPORTS, 'Unknown Q8 score export')
    folder, implementation, column = Q8_EXPORTS[tool]
    directory = root / 'notebooks/results/q8' / folder
    full = root / 'notebooks/results/q8/full'
    if context.get('scope') == 'full' and full.exists():
        directory = full / folder
        provenance = read_json(directory / 'baseline_provenance.json')
        require(provenance['scope'] == 'full_validation' and provenance['tool'] == tool,
                'Q8 full export belongs to a different scope or tool')
        require(provenance['q1_protocol_sha256'] == context['protocol_sha256'] and
                provenance['q1_vcf_sha256'] == context['vcf_exports'], 'Q8 full cohort is stale')
        for name, checksum in provenance['sources'].items():
            verified(root / name, checksum)
        verified(full / 'protocol.json', provenance['full_q8_protocol_sha256'])
        verified(full / 'leakage_checks.json', provenance['local_split_checks_sha256'])
        for name in ['baseline_protocol.json', 'validation_predictions.csv', 'validation_metrics.json']:
            verified(directory / name, provenance['artifacts'][name])
        frame = pd.read_csv(directory / 'validation_predictions.csv')
        require(frame.split.eq('validation').all(), 'Q8 full predictions contain another partition')
        arrays = align_predictions(frame, context, [column], allow_missing=True)
        require(not ((arrays[column] < 0) | (arrays[column] > 1)).any(), f'Invalid {tool} score range')
        return {tool: arrays[column]}
    provenance = read_json(directory / 'baseline_provenance.json')
    require(provenance['q1_protocol_sha256'] == context['protocol_sha256'] and
            provenance['q1_vcf_sha256'] == context['vcf_exports'], 'Q8 cohort is stale')
    verified(root / 'notebooks/src' / implementation, provenance['implementation_sha256'])
    if tool != 'AlphaMissense':
        verified(root / 'notebooks/src/q8_baseline.py', provenance['shared_evaluation_sha256'])
    if implementation == 'q8_remaining.py':
        verified(root / 'notebooks/src/q8_dbnsfp.py', provenance['acquisition_implementation_sha256'])
        require(provenance['method']['tool'] == tool, 'Q8 export belongs to a different tool')
    for name in ['baseline_protocol.json', 'pilot_evaluation.csv', 'validation_metrics.json']:
        verified(directory / name, provenance['artifacts'][name])
    frame = pd.read_csv(directory / 'pilot_evaluation.csv')
    require(frame.variant_key.tolist() == context['pilot_keys'], 'Q8 pilot membership or ordering changed')
    arrays = align_predictions(frame[frame.split.eq('validation')], context, [column], allow_missing=True)
    require(not ((arrays[column] < 0) | (arrays[column] > 1)).any(), f'Invalid {tool} score range')
    return {tool: arrays[column]}


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


def timing_artifact(directory, manifest, name):
    """Follow an already verified result's checksum to its timing record."""
    checksum = manifest.get('artifacts', {}).get(name)
    if checksum is None:
        return None
    path = directory / name
    verified(path, checksum)
    return read_json(path)


def load_runtime(root, identifier, export=None):
    """Read measured stages only after this method's cohort/results pass checks."""
    question, method = identifier.split(':', 1)
    directory = root / 'notebooks/results' / question
    value = None
    if export is not None:
        runtimes = export.get('runtimes', {})
        require(isinstance(runtimes, dict), 'Runtimes must be keyed by score column')
        value = runtimes.get(method)
        if value is None and identifier == 'q2:zero_shot_7b':
            report = timing_artifact(directory, export, '7b/metrics.json')
            timing = timing_artifact(directory / '7b', report, 'score_manifest.json') if report else None
            if timing is not None:
                value = {'seconds': timing['batch_seconds'],
                         'scope': 'Validation scoring batches summed across runs; excludes downloads, model loading and evaluation.'}
        elif value is None and identifier == 'q10:frozen_7b':
            report = timing_artifact(directory, export, 'metrics.json')
            timing = timing_artifact(directory, report, 'feature_manifest.json') if report else None
            if timing is not None:
                value = {'seconds': timing['seconds'] + report['fit_seconds'],
                         'scope': 'Train/validation feature extraction plus classifier fitting and selection; excludes setup and bootstrap evaluation.'}
    elif question == 'q8' and method in Q8_EXPORTS:
        folder = Q8_EXPORTS[method][0]
        if (root / 'notebooks/results/q1/full').exists() and (directory / 'full').exists():
            directory = directory / 'full'
        provenance = read_json(directory / folder / 'baseline_provenance.json')
        if provenance.get('elapsed_seconds') is not None:
            scope = ('CPU score lookup/evaluation workflow, including any downloads in that run; excludes upstream model training.'
                     if method in {'AlphaMissense', 'REVEL'} else
                     'CPU score lookup/evaluation; excludes shared dbNSFP acquisition and upstream model training.')
            value = {'seconds': provenance['elapsed_seconds'], 'scope': provenance.get('runtime_scope', scope)}
    elif question == 'q9':
        report = read_json(directory / 'metrics.json')
        selection = timing_artifact(directory, report, 'baseline_selection.json')
        if selection is not None:
            if method == 'sequence':
                value = {'seconds': selection['fit_seconds']['sequence'],
                         'scope': 'Classifier fitting and selection only; feature construction was not timed separately.'}
            elif method == 'frozen':
                timing = timing_artifact(directory, report, 'frozen_features.json')
                if timing is not None:
                    value = {'seconds': timing['seconds'] + selection['fit_seconds']['evo'],
                             'scope': 'Shared train/validation feature extraction plus classifier fitting and selection; excludes setup and bootstrap evaluation.'}
    if value is None:
        return None
    require(isinstance(value, dict), 'Runtime must contain seconds and scope')
    seconds, scope = value['seconds'], value['scope']
    require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
            and np.isfinite(seconds) and seconds >= 0, 'Runtime must be finite, nonnegative seconds')
    require(isinstance(scope, str) and bool(scope.strip()), 'Runtime must identify the measured stages')
    return {'runtime_seconds': float(seconds), 'runtime_scope': scope}


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


def published_results(root, context=None):
    """Read portable measurements, checking their executed-notebook evidence.

    A regenerated parent protocol can differ while the VCFs, labels, components
    and DNA are byte-identical. Match those artifacts, not the parent's hash.
    These records never supply per-variant predictions or new shared-subset scores.
    """
    root = Path(root)
    saved = read_json(root / PUBLISHED)
    require(saved['schema_version'] == 1, 'Unsupported published-result schema')
    notebooks = {}

    def notebook(path):
        if path not in notebooks:
            verified(root / path, saved['notebook_sha256'][path])
            value = read_json(root / path)
            cells = [c for c in value['cells'] if c['cell_type'] == 'code']
            require(cells and all(c['source'] for c in cells)
                    and [c['execution_count'] for c in cells] == list(range(1, len(cells) + 1))
                    and not any(o['output_type'] == 'error' for c in cells for o in c['outputs']),
                    'Published notebook is not completely executed')
            notebooks[path] = value
        return notebooks[path]

    def output(path, evidence, mime):
        data = notebook(path)['cells'][evidence['cell']]['outputs'][evidence['output']]['data'][mime]
        return ''.join(data) if isinstance(data, list) else data

    def payload(path, evidence):
        text = output(path, evidence, 'text/html')
        match = re.search(r'<pre>(.*?)</pre>', text, re.S)
        require(match is not None, 'Published JSON evidence is missing')
        return json.loads(html.unescape(match[1]))

    def select(value, keys):
        for key in keys:
            value = value[key]
        return value

    cohort = saved['cohort']
    original = payload(saved['cohort_notebook'], saved['cohort_evidence'])
    require(cohort['scope'] == 'full' and cohort['clinvar_date'] == original['config']['clinvar_date']
            and cohort['vcf_exports'] == original['vcf_exports']
            and all(original['artifacts'][k] == v for k, v in cohort['artifacts'].items()),
            'Published cohort differs from the completed Q1 notebook')
    if context is not None:
        require(context['scope'] == 'full' and context['vcf_exports'] == cohort['vcf_exports']
                and context['config'].get('clinvar_date') == cohort['clinvar_date']
                and len(context['validation']) == cohort['validation_variants']
                and all(context['artifacts'].get(k) == v for k, v in cohort['artifacts'].items()),
                'Published results do not match the current frozen cohort')
    for name, checksum in cohort['vcf_exports'].items():
        if (root / 'data' / name).exists():
            verified(root / 'data' / name, checksum)
    verified(root / saved['prior_readme']['path'], saved['prior_readme']['sha256'])
    prior = (root / saved['prior_readme']['path']).read_text()
    prior_metadata = json.loads(re.search(r'```json\n(.*?)\n```', prior, re.S)[1])
    require(all(cohort[k] == v for k, v in prior_metadata['cohort'].items()),
            'Published README cohort identity changed')
    rows, errors = {}, {}
    for record in saved['methods']:
        row = dict(record)
        try:
            evidence, path = row['evidence'], row['notebook']
            if evidence['kind'] == 'json':
                value = payload(path, evidence)
                require(select(value, evidence['metrics_path']) == row['metrics']
                        and select(value, evidence['count_path']) == row['covered'] == row['total'],
                        'Published metrics differ from notebook JSON')
                if 'runtime_path' in evidence:
                    require(select(value, evidence['runtime_path']) == row['runtime_seconds'],
                            'Published runtime differs from notebook JSON')
            elif evidence['kind'] == 'table':
                lines = output(path, evidence, 'text/markdown').splitlines()
                line = next(line for line in lines if line.startswith('| ' + evidence['row'] + ' |'))
                require(f'{row["covered"]:,} / {row["total"]:,}' in line
                        and all(format_metric(row, metric) in line for metric in ['auroc', 'average_precision']),
                        'Published metrics differ from notebook table')
                streams = ''.join(''.join(o.get('text', '')) for c in notebook(path)['cells']
                                  for o in c.get('outputs', []))
                match = re.search(re.escape(row['method']) + r': [^\n]*; ([\d.]+) seconds\.', streams)
                require(match is not None and float(match[1]) == row['runtime_seconds'],
                        'Published runtime differs from executed stream')
            else:
                raise ValueError('Unsupported published evidence')
            statuses = row.get('runtime_status_evidence')
            question = row['question'].lower()
            if question in LORA_FULL_METHODS:
                require(isinstance(statuses, list) and len(statuses) == 2
                        and {entry['notebook'] for entry in statuses} == {
                            row['notebook'], 'notebooks/' + LORA_EXPLORATION_NOTEBOOKS[question]},
                        'Published LoRA runtime needs both executed notebook stages')
            if statuses is not None:
                require(isinstance(statuses, list) and bool(statuses)
                        and len({entry['path'] for entry in statuses}) == len(statuses),
                        'Published runtime status evidence must be a nonempty list of distinct files')
                seconds = 0.
                for entry in statuses:
                    relative = Path(entry['path'])
                    require(not relative.is_absolute() and relative.parent == PUBLISHED.parent
                            and relative.suffix == '.json', 'Published runtime evidence path is invalid')
                    verified(root / relative, entry['sha256'])
                    status = read_json(root / relative)
                    notebook(entry['notebook'])
                    duration = status.get('notebook_execution_seconds')
                    require(status.get('status') == 'complete'
                            and status.get('question', question) == question
                            and status.get('notebook_sha256') == saved['notebook_sha256'][entry['notebook']]
                            and isinstance(duration, (int, float)) and not isinstance(duration, bool)
                            and np.isfinite(duration) and duration >= 0,
                            'Published runtime status is incomplete or differs from its executed notebook')
                    seconds += duration
                require(not isinstance(row.get('runtime_seconds'), bool) and seconds == row.get('runtime_seconds'),
                        'Published runtime differs from the measured notebook stage sum')
            if 'runtime_display' in row:
                line = next(line for line in prior.splitlines() if line.startswith('| ' + row['method'] + ' |'))
                require(line.rstrip().endswith('| ' + row['runtime_display'] + ' |'),
                        'Published rounded runtime differs from the saved README')
            row.update(status='Published', note=row['note'] +
                       ' Published executed-notebook result; per-variant export is absent locally.')
            rows[row['id']] = row
        except (OSError, ValueError, KeyError, IndexError, StopIteration, TypeError) as error:
            errors[row['id']] = str(error)
    common = {}
    for identifier, measured in saved['published_common'].items():
        if identifier not in rows:
            continue
        row = rows[identifier]
        expected = (f'| {row["method"]} ({row["question"]}) | {measured["total"]} | '
                    f'{format_metric(measured, "auroc")} | {format_metric(measured, "average_precision")} |')
        require(expected in prior, 'Published shared-subset metrics differ from their source')
        common[identifier] = measured
    return {'cohort': cohort, 'methods': rows, 'common': common, 'errors': errors,
            'bootstrap': saved['bootstrap']}


def retain_published(root, result, methods, context=None):
    """Use a completed record only when this method has no local export at all."""
    if not (Path(root) / PUBLISHED).exists():
        return
    try:
        published = published_results(root, context)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result['errors']['Published results'] = str(error)
        return
    result['errors'].update({'Published ' + k: v for k, v in published['errors'].items()})
    retained = []
    for identifier, row in published['methods'].items():
        current = methods.get(identifier, {})
        if current.get('metrics') or current.get('status') in ['Invalid / stale', 'Blocked']:
            continue
        if any((Path(root) / name).exists() for name in row['local_artifacts']):
            continue
        methods[identifier] = row
        retained.append(identifier)
    if retained:
        result['published_methods'] = retained
        result['published_common'] = published['common']
        result['published_bootstrap'] = published['bootstrap']
        if result['cohort'] is None:
            result['cohort'] = published['cohort']
            result['local_inputs'] = 'Local Q1 inputs absent; showing the pinned completed notebook cohort.'


def collect(root=ROOT, repetitions=REPETITIONS):
    root = Path(root)
    catalog = read_json(root / 'notebooks/src/q8_catalog.json')
    definitions = [
        ('q2', Q2_METHODS, 'Not run', 'Missense feature extraction/evaluation pending. Validation selects C.'),
        ('q8', {tool['name']: tool['name'] for tool in catalog['tools']}, 'Not evaluated',
         'Surveyed tool; no predictions exported for the current cohort.'),
        ('q9', Q9_METHODS, 'Not run', 'Training has not produced a completed evaluation.'),
        ('q11', Q11_METHODS, 'Not run', 'The partial-epoch experiment with full validation has not completed.'),
    ]
    methods = {f'{question}:{key}': {'id': f'{question}:{key}', 'question': question.upper(), 'method': name,
                                   'status': status, 'note': note}
               for question, names, status, note in definitions for key, name in names.items()}
    methods['q8:PrimateAI-3D']['note'] = 'Requires licensed data; no predictions for the current cohort.'
    status_path = root / 'notebooks/results/q11/run_status.json'
    if status_path.exists():
        try:
            status = read_json(status_path)
            if status['status'] == 'blocked':
                for key in Q11_METHODS:
                    methods[f'q11:{key}'].update(status='Blocked', note='Last Q11 execution: ' + status['reason'])
        except (OSError, ValueError, KeyError):
            pass
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'methods': [], 'common': {},
              'cohort': None, 'bootstrap': {'repetitions': repetitions, 'seed': SEED}, 'errors': {}}
    try:
        context = load_context(root)
    except (OSError, ValueError, KeyError) as error:
        result['errors']['Q1'] = str(error)
        for row in methods.values():
            row.update(status='Unavailable', note='Frozen Q1 inputs unavailable or invalid; no metrics displayed.')
        # A clean checkout retains the published cohort. An invalid local Q1
        # protocol must remain an error; published data never masks corruption.
        if not any((root / 'notebooks/results/q1' / name).exists()
                   for name in ['protocol.json', 'full/protocol.json']):
            retain_published(root, result, methods)
        result['methods'] = list(methods.values())
        return result
    validation = context['validation']
    result['cohort'] = {key: context[key] for key in ['protocol_sha256', 'vcf_exports']}
    result['cohort']['validation_variants'] = len(validation)
    result['cohort']['clinvar_date'] = context['config'].get('clinvar_date', 'unspecified snapshot')
    result['cohort']['scope'] = context['scope']
    predictions, exports = {}, {}
    loaders = [('q2', load_q2, 'validation_report.json', None)] + [
        ('q8', load_q8, str(Path(folder) / 'baseline_provenance.json'), tool)
        for tool, (folder, _, _) in Q8_EXPORTS.items()] + [('q9', load_q9, 'metrics.json', None)]
    for question, loader, required, tool in loaders:
        directory = root / 'notebooks/results' / question
        if question == 'q8' and context.get('scope') == 'full' and (directory / 'full').exists():
            directory = directory / 'full'
        candidates = [row for row in methods.values() if row['question'].lower() == question and
                      (question != 'q8' or row['method'] == tool)]
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
            arrays = loader(root, context, tool) if question == 'q8' else loader(root, context)
            for key, values in arrays.items():
                identifier = f'{question}:{key}'
                predictions[identifier] = values
                methods[identifier].update(status='Available', note=(
                    Q8_NOTES[key] if question == 'q8'
                    else 'Development result; validation participates in model selection.'))
        except (OSError, ValueError, KeyError, AssertionError) as error:
            result['errors'][f'Q8 {tool}' if question == 'q8' else question.upper()] = str(error)
            for row in candidates:
                row.update(status='Invalid / stale', note=str(error))
    access_path = root / 'notebooks/results/q8/primateai3d/access_status.json'
    if context.get('scope') == 'full' and (root / 'notebooks/results/q8/full').exists():
        access_path = root / 'notebooks/results/q8/full/primateai3d/access_status.json'
    if access_path.exists():
        try:
            access = read_json(access_path)
            require(access['status'] == 'blocked' and access['q1_protocol_sha256'] == context['protocol_sha256'],
                    'PrimateAI-3D access report is stale or invalid')
            methods['q8:PrimateAI-3D'].update(status='Blocked', note=access['reason'])
        except (OSError, ValueError, KeyError) as error:
            result['errors']['Q8 PrimateAI-3D'] = str(error)
    # Future question notebooks can export this small documented contract.
    for path in sorted((root / 'notebooks/results').glob('q*/comparison_results.json')):
        question = path.parent.name
        candidates = {}
        try:
            spec = read_json(path)
            candidates = {f'{question}:{column}': {'id': f'{question}:{column}', 'question': question.upper(),
                          'method': name, 'status': 'Invalid / stale', 'note': ''}
                          for column, name in spec['methods'].items()}
            if question == 'q11':
                require(spec['methods'] == Q11_METHODS, 'Q11 must export its partial-epoch LoRA method')
                require(spec.get('require_complete') is True, 'Q11 requires complete validation coverage')
                require(context.get('scope') == 'full', 'Q11 requires the full cohort')
                require(spec.get('vcf_exports') == context['vcf_exports'], 'Q11 VCF identity changed')
            else:
                require(not set(candidates) & set(methods), 'Duplicate method IDs in custom comparison export')
            methods.update(candidates)
            require(spec['q1_protocol_sha256'] == context['protocol_sha256'], 'Export cohort is stale')
            for name, checksum in spec.get('sources', {}).items():
                verified(root / name, checksum)
            for name, checksum in spec.get('artifacts', {}).items():
                verified(path.parent / name, checksum)
            if question == 'q12':
                require(spec.get('scope') == 'full_validation' and spec.get('require_complete') is True
                        and context.get('scope') == 'full' and spec.get('vcf_exports') == context['vcf_exports'],
                        'Q12 sampled exploration cannot enter the full-cohort comparison')
                require('full/metrics.json' in spec.get('artifacts', {}), 'Q12 requires completed full-validation evidence')
                completion = read_json(path.parent / 'full/metrics.json')
                require(completion.get('scope') == 'full_validation' and completion.get('status') == 'complete'
                        and completion.get('validation_variants') == len(validation)
                        and completion.get('reloaded_predictions_verified') is True,
                        'Q12 full validation is incomplete')
            if question in LORA_FULL_METHODS:
                verify_full_lora_export(root, question, path.parent, spec, context)
            if question == 'q11':
                require('metrics.json' in spec.get('artifacts', {}), 'Q11 completion marker is required')
                completion = read_json(path.parent / 'metrics.json')
                require(completion.get('status') == 'complete' and completion.get('training_mode') == 'partial_epoch'
                        and completion.get('reloaded_predictions_verified') is True,
                        'Q11 has not completed and verified its partial-epoch model')
            csv = path.with_name('comparison_predictions.csv')
            verified(csv, spec['predictions_sha256'])
            arrays = align_predictions(pd.read_csv(csv), context, spec['methods'],
                                       allow_missing=not spec.get('require_complete', False))
            for column, values in arrays.items():
                identifier = f'{question}:{column}'
                methods[identifier].update(status='Available', note=spec.get('limitations', ''))
                predictions[identifier] = values
                exports[identifier] = spec
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            result['errors'][question.upper()] = str(error)
            for row in candidates.values():
                row['note'] = str(error)
            if question == 'q11':
                for key in Q11_METHODS:
                    methods[f'q11:{key}'].update(status='Invalid / stale', note=str(error))
    for identifier in predictions:
        try:
            timing = load_runtime(root, identifier, exports.get(identifier))
            if timing is not None:
                methods[identifier].update(timing)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # An absent or corrupt optional timing must not discard valid scores.
            methods[identifier]['runtime_error'] = str(error)
    labels, groups = validation.label.to_numpy(), validation.component.to_numpy()
    summary = summarize(labels, predictions, groups, repetitions)
    for identifier, row in methods.items():
        row.update(summary.get(identifier, {}))
        if identifier in predictions and not row.get('metrics'):
            row.update(status='Insufficient coverage', note='The scored subset needs both ClinVar classes. ' + row['note'])
    usable = {key: values for key, values in predictions.items() if summary[key]['metrics']}
    if len(usable) >= 2:
        mask = np.logical_and.reduce([np.isfinite(values) for values in usable.values()])
        if len(np.unique(labels[mask])) == 2:
            result['common'] = summarize(labels[mask], {name: values[mask] for name, values in usable.items()},
                                         groups[mask], repetitions)
        else:
            result['errors']['Common subset'] = 'The shared scored subset does not contain both classes.'
    retain_published(root, result, methods, context)
    result['methods'] = list(methods.values())
    return result


def format_metric(measured, key):
    metric = measured.get('metrics', {}).get(key)
    if not metric:
        return '—'
    interval = metric['ci95']
    suffix = f' [{interval[0]:.3f}, {interval[1]:.3f}]' if interval else ''
    return f'{metric["value"]:.3f}{suffix}'


def format_runtime(row):
    if 'runtime_display' in row:
        return row['runtime_display']
    seconds = row.get('runtime_seconds')
    if seconds is None:
        return '—'
    if seconds < 1:
        return f'{seconds:.3g} s'
    if seconds < 60:
        return f'{seconds:.1f} s'
    if seconds < 3600:
        return f'{seconds / 60:.1f} min'
    return f'{seconds / 3600:.1f} h'


def markdown_table(rows, separator_before=None):
    """Render GitHub tables without an additional formatting dependency."""
    columns = list(rows[0])
    def line(values):
        return '| ' + ' | '.join(str(value).replace('|', r'\|').replace('\n', '<br>') for value in values) + ' |'
    lines = [line(columns), line(['---'] * len(columns))]
    for index, row in enumerate(rows):
        if index == separator_before and index > 0:
            lines.append(line(['<hr>'] * len(columns)))
        lines.append(line(row[column] for column in columns))
    return '\n'.join(lines)


def render(result, root=ROOT):
    """Return the Markdown evaluation tables, paper links and provenance."""
    root = Path(root)
    rows = sorted((row for row in result['methods'] if row['id'] not in README_EXCLUDED_METHODS),
                  key=lambda row: (row['question'] == 'Q8', row['id'] == 'q8:PrimateAI-3D'))
    external_start = sum(row['question'] != 'Q8' for row in rows)
    notebooks = {p.stem.split('-')[0].upper(): p for p in (root / 'notebooks').glob('Q*.ipynb')}
    table = []
    for row in rows:
        preferred = root / 'notebooks' / NOTEBOOKS.get(row['question'], '')
        path = preferred if preferred.is_file() else notebooks.get(row['question'])
        link = f'[{row["question"]}]({path.relative_to(root).as_posix()})' if path else row['question']
        name = html.escape(row['method']) + (' (licensed)' if row['id'] == 'q8:PrimateAI-3D' else '')
        table.append({'Method': name, 'Notebook': link,
                      'Paper': METHOD_PAPERS.get(row['id'], '—'),
                      'Scored / validation': f'{row["covered"]:,} / {row["total"]:,}' if 'covered' in row else '—',
                      'AUROC [95% CI]': format_metric(row, 'auroc'),
                      'Average precision [95% CI]': format_metric(row, 'average_precision'),
                      'Runtime': format_runtime(row)})
    cohort = result['cohort']
    intro = (f'**{cohort["validation_variants"]:,} missense validation variants · '
             f'ClinVar {cohort.get("clinvar_date", "unspecified snapshot")} · {cohort.get("scope", "pilot")} cohort**' if cohort
             else '**Frozen validation inputs unavailable**')
    sections = ['## Method comparison', 'Compare methods on Q1’s current frozen missense validation set. Only results matching its snapshot and complete cohort are included.', intro,
                markdown_table(table, separator_before=external_start),
                'Runtime covers the recorded stages listed in the details below; hardware and caching differ between workflows. '
                '“—” means no verified timing is available for the current cohort.']
    available = [row for row in rows if row.get('metrics')]
    if result.get('published_methods'):
        sections.append('Completed notebook measurements are preserved when local prediction exports are absent. '
                        'Their source notebooks and exact cohort checksums are verified; locally available predictions take precedence.')
    if result['common']:
        common_rows = [row for row in available if row['id'] in result['common']]
        table = [{'Method': f'{row["method"]} ({row["question"]})',
                  'Shared variants': result['common'][row['id']]['total'],
                  'AUROC [95% CI]': format_metric(result['common'][row['id']], 'auroc'),
                  'Average precision [95% CI]': format_metric(result['common'][row['id']], 'average_precision')}
                 for row in common_rows]
        sections.extend(['**Direct comparison on the same variants**',
                         markdown_table(table, separator_before=sum(row['question'] != 'Q8' for row in common_rows))])
    if result.get('published_common'):
        shared = result['published_common']
        shared_rows = [row for row in available if row['id'] in shared]
        table = [{'Method': f'{row["method"]} ({row["question"]})',
                  'Shared variants': shared[row['id']]['total'],
                  'AUROC [95% CI]': format_metric(shared[row['id']], 'auroc'),
                  'Average precision [95% CI]': format_metric(shared[row['id']], 'average_precision')}
                 for row in shared_rows]
        count = next(iter(shared.values()))['total']
        sections.extend([f'**Published comparison on {count:,} shared variants (Q2/Q8; excludes LoRA)**',
                         markdown_table(table, separator_before=sum(row['question'] != 'Q8' for row in shared_rows)),
                         'These shared-subset results come from the completed Q2/Q8 comparison. '
                         'Computing LoRA on that subset requires the original per-variant exports.'])
    notes = []
    for row in rows:
        detail = row['note']
        if row.get('runtime_scope'):
            detail += ' Runtime: ' + row['runtime_scope']
        elif row.get('runtime_error'):
            detail += ' Runtime unavailable: ' + row['runtime_error']
        notes.append({'Method': html.escape(f'{row["method"]} ({row["question"]})'), 'Details': html.escape(detail)})
    sections.append('<details>\n<summary>Provenance, missing results and limitations</summary>\n\n' +
                    markdown_table(notes, separator_before=external_start) + '\n\n```json\n' + json.dumps({
                        'cohort': cohort, 'source_errors': result['errors'], 'bootstrap': result['bootstrap'],
                        'published_methods': result.get('published_methods', []),
                        'published_bootstrap': result.get('published_bootstrap'),
                        'generated_utc': result['generated_utc']}, indent=2) + '\n```\n\n</details>')
    if len(available) == 1:
        conclusion = f'Only **{available[0]["method"]}** currently has verified metrics. A ranking awaits the other methods’ results.'
    elif not available:
        conclusion = 'No verified metrics are currently available. Complete the source experiments or resolve the listed input checks.'
    elif result.get('published_common'):
        conclusion = 'Per-method results retain their reported coverage. The published shared-subset comparison covers Q2/Q8; LoRA has complete validation metrics.'
    elif not result['common']:
        conclusion = 'Several methods have results, but there is no shared scored subset with both classes for a direct comparison.'
    else:
        conclusion = 'Use the common-subset comparison to assess methods; coverage remains a separate limitation.'
    sections.append('**Conclusion.** ' + conclusion + '\n\n'
                    'These are development results: validation participates in model selection. '
                    'AlphaMissense calibration and REVEL/PolyPhen-2 training overlap with ClinVar remain unresolved. '
                    'The 95% intervals resample whole '
                    'Q1 components and do not correct selection bias or establish clinical validity.')
    return '\n\n'.join(sections)


def update_readme(root, section):
    """Replace only the managed section, reading other prose immediately before writing."""
    path = Path(root) / 'README.md'
    original = path.read_text() if path.exists() else '# Pathogenicity predictor\n'
    counts = (original.count(SECTION_START), original.count(SECTION_END))
    require(counts in [(0, 0), (1, 1)], 'README comparison markers are incomplete or duplicated')
    block = SECTION_START + '\n' + section + '\n' + SECTION_END
    if counts == (1, 1):
        start, end = original.index(SECTION_START), original.index(SECTION_END)
        require(start < end, 'README comparison markers are reversed')
        updated = original[:start] + block + original[end + len(SECTION_END):]
    else:
        index = original.find('\n## Research questions\n')
        if index < 0:
            updated = original.rstrip() + '\n\n' + block + '\n'
        else:
            updated = original[:index].rstrip() + '\n\n' + block + '\n' + original[index:]
    require(not path.exists() or path.read_text() == original, 'README changed during update; retry')
    atomic_write(path, updated)


def refresh(root=ROOT, repetitions=REPETITIONS):
    """Publish verified comparison results to README; no notebooks are executed."""
    import fcntl
    root = Path(root)
    output = root / 'notebooks/results/comparison'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'refresh.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = source_signature(root)
        result = collect(root, repetitions)
        section = render(result, root)
        require(before == source_signature(root), 'Sources changed during refresh; retry after writes finish')
        result['sources'] = source_signature(root, content=True)
        atomic_write(output / 'summary.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
        csv = pd.DataFrame([{key: row.get(key) for key in ['id', 'question', 'method', 'status', 'covered', 'total',
                                                         'runtime_seconds', 'runtime_scope']} |
                            {metric: row.get('metrics', {}).get(metric, {}).get('value')
                             for metric in ['auroc', 'average_precision']} for row in result['methods']])
        atomic_write(output / 'methods.csv', csv.to_csv(index=False))
        update_readme(root, section)
        return result, section
