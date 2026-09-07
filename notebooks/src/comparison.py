"""Collect verified validation predictions and update the main README comparison.

This module never executes a source notebook, imports a model, or reads archives.
Notebook saves trigger refreshes; scientific values come from exported predictions.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
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
    paths.update(root / 'data' / name for name in ['clinvar-train-pilot.vcf', 'clinvar-test-pilot.vcf',
                                                'clinvar-train.vcf', 'clinvar-test.vcf'])
    for path in (root / 'notebooks/results').glob('q*/comparison_results.json'):
        paths.add(path)
        # This is the documented filename for future notebooks' prediction exports.
        paths.add(path.with_name('comparison_predictions.csv'))
        try:
            paths.update(path.parent / name for name in read_json(path).get('artifacts', {}))
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


def collect(root=ROOT, repetitions=REPETITIONS):
    root = Path(root)
    catalog = read_json(root / 'notebooks/src/q8_catalog.json')
    definitions = [
        ('q2', Q2_METHODS, 'Not run', 'Missense feature extraction/evaluation pending. Validation selects C.'),
        ('q8', {tool['name']: tool['name'] for tool in catalog['tools']}, 'Not evaluated',
         'Surveyed tool; no predictions exported for the current cohort.'),
        ('q9', Q9_METHODS, 'Not run', 'Training has not produced a completed evaluation.'),
    ]
    methods = {f'{question}:{key}': {'id': f'{question}:{key}', 'question': question.upper(), 'method': name,
                                   'status': status, 'note': note}
               for question, names, status, note in definitions for key, name in names.items()}
    methods['q8:PrimateAI-3D']['note'] = 'Requires licensed data; no predictions for the current cohort.'
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
            require(not set(candidates) & set(methods), 'Duplicate method IDs in custom comparison export')
            methods.update(candidates)
            require(spec['q1_protocol_sha256'] == context['protocol_sha256'], 'Export cohort is stale')
            for name, checksum in spec.get('sources', {}).items():
                verified(root / name, checksum)
            for name, checksum in spec.get('artifacts', {}).items():
                verified(path.parent / name, checksum)
            csv = path.with_name('comparison_predictions.csv')
            verified(csv, spec['predictions_sha256'])
            arrays = align_predictions(pd.read_csv(csv), context, spec['methods'], allow_missing=True)
            for column, values in arrays.items():
                identifier = f'{question}:{column}'
                methods[identifier].update(status='Available', note=spec.get('limitations', ''))
                predictions[identifier] = values
                exports[identifier] = spec
        except (OSError, ValueError, KeyError) as error:
            result['errors'][question.upper()] = str(error)
            for row in candidates.values():
                row['note'] = str(error)
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


def format_runtime(row):
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


def markdown_table(rows):
    """Render GitHub tables without an additional formatting dependency."""
    columns = list(rows[0])
    def line(values):
        return '| ' + ' | '.join(str(value).replace('|', r'\|').replace('\n', '<br>') for value in values) + ' |'
    return '\n'.join([line(columns), line(['---'] * len(columns))] +
                     [line(row[column] for column in columns) for row in rows])


def render(result, root=ROOT):
    """Return the Markdown evaluation tables, paper links and provenance."""
    root = Path(root)
    rows = sorted(result['methods'], key=lambda row: row['id'] == 'q8:PrimateAI-3D')
    notebooks = {p.stem.split('-')[0].upper(): p for p in (root / 'notebooks').glob('Q*.ipynb')}
    table = []
    for row in rows:
        path = notebooks.get(row['question'])
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
                markdown_table(table),
                'Runtime covers the recorded stages listed in the details below; hardware and caching differ between workflows. '
                '“—” means no verified timing is available for the current cohort.']
    available = [row for row in rows if row.get('metrics')]
    if result['common']:
        table = [{'Method': f'{row["method"]} ({row["question"]})',
                  'Shared variants': result['common'][row['id']]['total'],
                  'AUROC [95% CI]': format_metric(result['common'][row['id']], 'auroc'),
                  'Average precision [95% CI]': format_metric(result['common'][row['id']], 'average_precision')}
                 for row in available if row['id'] in result['common']]
        sections.extend(['**Direct comparison on the same variants**', markdown_table(table)])
    notes = []
    for row in rows:
        detail = row['note']
        if row.get('runtime_scope'):
            detail += ' Runtime: ' + row['runtime_scope']
        elif row.get('runtime_error'):
            detail += ' Runtime unavailable: ' + row['runtime_error']
        notes.append({'Method': html.escape(f'{row["method"]} ({row["question"]})'), 'Details': html.escape(detail)})
    sections.append('<details>\n<summary>Provenance, missing results and limitations</summary>\n\n' +
                    markdown_table(notes) + '\n\n```json\n' + json.dumps({
                        'cohort': cohort, 'source_errors': result['errors'], 'bootstrap': result['bootstrap'],
                        'generated_utc': result['generated_utc']}, indent=2) + '\n```\n\n</details>')
    if len(available) == 1:
        conclusion = f'Only **{available[0]["method"]}** currently has verified metrics. A ranking awaits the other methods’ results.'
    elif not available:
        conclusion = 'No verified metrics are currently available. Complete the source experiments or resolve the listed input checks.'
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
