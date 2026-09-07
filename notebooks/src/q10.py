"""Q10: frozen Evo2 7B BioNeMo features on Q1's fixed missense pilot."""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from . import q0, q1, q2, q9, q9_environment

ROOT = q1.ROOT
OUTPUT = ROOT / 'notebooks/results/q10'
CHECKPOINT = ROOT / 'data/evo2-savanna-7b/savanna_evo2_7b_base.pt'
CONVERTED = OUTPUT / 'base_checkpoint_zarr'
MODEL_NAME = 'Evo2 7B base frozen head (BioNeMo, BF16)'
CONFIG = {
    'model_repo': 'arcinstitute/savanna_evo2_7b_base',
    'model_revision': 'eb0a7478e5f3c291f31e2b3d9ec14fc067f9982a',
    'model_sha256': 'ed8d264c14fea3c6305b475122e068e09009cbdabc05e1653764147c1b294cd4',
    'model_bytes': 15759263071,
    'context_bp': 1024, 'hidden_size': 4096, 'layers': 32, 'seed': 42,
    'precision': 'BF16 backbone; FP32 pooling; FP64 sklearn classifier',
    'fp32_residual_connection': False, 'fp8': False, 'microbatch_sequences': 1,
    'features': 'final block before final normalization; mean positions, then strands; [reference, alternate-reference]',
    'feature_dimension': 8192, 'cache_batch_variants': 100,
    'classifier': 'train-only StandardScaler and balanced L2 logistic regression; lbfgs, max_iter=30000, tol=1e-5',
    'C_grid': [0.01, 0.1, 1.0, 10.0],
    'selection': 'validation AUROC; ties select smaller C',
    'bootstrap_repetitions': 1000,
    'evaluation': 'fixed validation development comparison; no untouched test',
}
LIMITATIONS = ('Validation selects C; pretraining and homology overlap remain unresolved. '
               'The 1B baseline uses the original BF16-sensitive checkpoint, so this is '
               'a comparison of configurations, not an isolated model-size effect.')


def sources():
    names = ['q0.py', 'q1.py', 'q2.py', 'q9.py', 'q9_environment.py', 'q10.py', 'q10_backend.py']
    paths = [ROOT / 'notebooks/src' / name for name in names] + [ROOT / 'requirements.txt']
    return {str(path.relative_to(ROOT)): q1.digest_file(path) for path in paths}


def prepare_inputs():
    if not (q1.OUTPUT / 'protocol.json').exists():
        q1.prepare()
    protocol, pilot, dna, _, counts, checks = q9.verify_inputs()
    q9.assert_feature_inputs(dna, dna)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {'counts': counts, 'checks': checks, 'configuration': CONFIG,
              'vcf_exports': protocol['vcf_exports'],
              'q1_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
              'manifest_sha256': protocol['artifacts']['split_manifest.csv'],
              'sequences_sha256': protocol['artifacts']['sequences.csv.gz'],
              'limitations': LIMITATIONS}
    q1.write_json(OUTPUT / 'input_checks.json', record, frozen=True)
    print(f"Verified {counts['train']:,} training / {counts['validation']:,} validation missense variants.")
    q0.details('Frozen inputs, leakage checks and experiment settings', record)
    return record


def run_command(name, arguments):
    folder = OUTPUT / 'environment'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A Q10 process is already running; see its environment log.') from None
        path = folder / f'{name}.log'
        if path.exists():
            archive = folder / 'logs'
            archive.mkdir(exist_ok=True)
            path.rename(archive / f'{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        args = list(map(str, arguments))
        q1.write_json(folder / f'{name}-command.json', {'args': args, 'cwd': str(ROOT)})
        print(f'{name}: {path.relative_to(ROOT)}', flush=True)
        with path.open('w') as stream:
            result = subprocess.run(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    pass_fds=(lock.fileno(),))
        if result.returncode:
            tail = '\n'.join(path.read_text(errors='replace').splitlines()[-25:])
            raise RuntimeError(f'Q10 {name} failed ({result.returncode}): {path}\n{tail}')
    return path


def establish_environment():
    from huggingface_hub import hf_hub_download
    environment = q9_environment.setup()
    q1.write_json(OUTPUT / 'environment/runtime.json', environment, frozen=True)
    source = hf_hub_download(CONFIG['model_repo'], CHECKPOINT.name, revision=CONFIG['model_revision'],
                             local_dir=CHECKPOINT.parent)
    q1.verify_file(source, CONFIG['model_sha256'])
    q1.write_json(OUTPUT / 'checkpoint_source.json', {
        'repo': CONFIG['model_repo'], 'revision': CONFIG['model_revision'],
        'filename': CHECKPOINT.name, 'sha256': CONFIG['model_sha256'], 'bytes': CHECKPOINT.stat().st_size}, frozen=True)
    marker = OUTPUT / 'converted_checkpoint.json'
    if not marker.exists():
        if CONVERTED.exists():
            raise RuntimeError('Incomplete Q10 conversion exists; preserve it before retrying.')
        if shutil.disk_usage(ROOT).free < 18 * 1024**3:
            raise RuntimeError('Checkpoint conversion needs at least 18 GiB of free disk space.')
        run_command('convert', [q9_environment.PYTHON, '-m', 'notebooks.src.q10_backend', 'convert'])
        if not (CONVERTED / 'weights/metadata.json').exists():
            raise RuntimeError('Conversion did not produce a complete Zarr checkpoint.')
        files = {str(p.relative_to(CONVERTED)): q1.digest_file(p)
                 for p in sorted(CONVERTED.rglob('*')) if p.is_file()}
        q1.write_json(marker, {'source_sha256': CONFIG['model_sha256'], 'format': 'zarr', 'files': files}, frozen=True)
    converted = q1.read_json(marker)
    if converted['source_sha256'] != CONFIG['model_sha256']:
        raise ValueError('Converted checkpoint source changed')
    for name, checksum in converted['files'].items():
        q1.verify_file(CONVERTED / name, checksum)
    protocol = {'configuration': CONFIG, 'sources': sources(),
                'q1_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
                'input_checks_sha256': q1.digest_file(OUTPUT / 'input_checks.json'),
                'environment_sha256': q1.digest_file(OUTPUT / 'environment/runtime.json'),
                'checkpoint_manifest_sha256': q1.digest_file(marker),
                'upstream_sources': q1.read_json(q9_environment.OUTPUT / 'setup.json')['identity']['sources']}
    q1.write_json(OUTPUT / 'protocol.json', protocol, frozen=True)
    q0.details('Pinned 7B checkpoint and shared BioNeMo runtime', {
        'checkpoint': q1.read_json(OUTPUT / 'checkpoint_source.json'), 'runtime': environment})
    return protocol


def verify_protocol():
    protocol = q1.read_json(OUTPUT / 'protocol.json')
    if protocol['configuration'] != CONFIG or protocol['sources'] != sources():
        raise ValueError('Q10 code or settings changed; preserve the existing experiment.')
    q1.verify_file(q1.OUTPUT / 'protocol.json', protocol['q1_protocol_sha256'])
    for name, key in [('input_checks.json', 'input_checks_sha256'),
                      ('environment/runtime.json', 'environment_sha256'),
                      ('converted_checkpoint.json', 'checkpoint_manifest_sha256')]:
        q1.verify_file(OUTPUT / name, protocol[key])
    q9.verify_inputs()
    return q1.fingerprint(protocol)


def read_feature_batch(path, record, keys, identity):
    if record['identity'] != identity or record['keys'] != list(keys):
        raise ValueError('Feature batch identity or variant ordering changed')
    q1.verify_file(path, record['sha256'])
    with np.load(path, allow_pickle=False) as saved:
        values = saved['evo']
        if saved['keys'].tolist() != list(keys):
            raise ValueError('Feature batch keys differ from the frozen DNA input')
    if values.shape != (len(keys), CONFIG['feature_dimension']) or not np.isfinite(values).all():
        raise ValueError('Invalid 7B feature dimensions or non-finite values')
    return values


def feature_identity(identity):
    """Allow the explicitly audited reuse after the classifier convergence repair."""
    producer = q1.read_json(OUTPUT / 'feature_manifest.json')['identity']
    if producer == identity:
        return identity
    reuse = q1.read_json(OUTPUT / 'feature_reuse.json')
    if reuse['from_identity'] != producer or reuse['to_identity'] != identity:
        raise ValueError('Feature reuse does not match its producer and consuming experiment')
    for name, checksum in reuse['artifacts'].items():
        q1.verify_file(OUTPUT / name, checksum)
    old = q1.read_json(OUTPUT / reuse['archive'] / 'protocol.json')
    current = q1.read_json(OUTPUT / 'protocol.json')
    if q1.fingerprint(old) != producer:
        raise ValueError('Archived feature producer identity changed')
    old_config = dict(old['configuration'])
    old_config['classifier'] = old_config['classifier'].replace('max_iter=3000,', 'max_iter=30000,')
    if old_config != current['configuration']:
        raise ValueError('Reuse changes more than the classifier iteration limit')
    changed = {name for name in old['sources'].keys() | current['sources'].keys()
               if old['sources'].get(name) != current['sources'].get(name)}
    if changed != {'notebooks/src/q10.py'}:
        raise ValueError('Feature-producing backbone or shared input code changed')
    for key in ['q1_protocol_sha256', 'checkpoint_manifest_sha256', 'environment_sha256', 'upstream_sources']:
        if old[key] != current[key]:
            raise ValueError(f'Feature reuse changed {key}')
    return producer


def extract_features():
    verify_protocol()
    run_command('features', [q9_environment.PYTHON, '-m', 'notebooks.src.q10_backend', 'extract'])
    manifest = q1.read_json(OUTPUT / 'feature_manifest.json')
    print(f"Cached {manifest['variants']:,} variants × {CONFIG['feature_dimension']:,} features; "
          f"{manifest['seconds']/60:.1f} min, {manifest['peak_gpu_gib']:.1f} GiB peak GPU memory.")
    q0.details('Conversion, repeatability and DNA-only feature checks', q1.read_json(OUTPUT / 'preflight.json'))
    return manifest


def load_features():
    identity = feature_identity(verify_protocol())
    manifest = q1.read_json(OUTPUT / 'feature_manifest.json')
    if manifest['identity'] != identity:
        raise ValueError('Feature manifest belongs to another experiment')
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz')
    arrays = []
    for offset in range(0, len(dna), CONFIG['cache_batch_variants']):
        name = f'features/{offset:05d}.npz'
        keys = dna.variant_key.iloc[offset:offset+CONFIG['cache_batch_variants']].tolist()
        arrays.append(read_feature_batch(OUTPUT / name, manifest['batches'][name], keys, identity))
    values = np.concatenate(arrays)
    if len(values) != len(dna) or manifest['variants'] != len(dna):
        raise ValueError('Feature coverage differs from Q1')
    return dna.variant_key.to_numpy(), values


def paired_metrics(labels, predictions, groups):
    from sklearn.metrics import roc_auc_score, average_precision_score
    functions = {'auroc': roc_auc_score, 'average_precision': average_precision_score}
    result = {name: {metric: {'value': float(function(labels, values))}
                    for metric, function in functions.items()} for name, values in predictions.items()}
    draws = {name: {metric: [] for metric in functions} for name in predictions}
    differences = {metric: [] for metric in functions}
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(CONFIG['seed'])
    for _ in range(CONFIG['bootstrap_repetitions']):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[index])) < 2:
            continue
        for metric, function in functions.items():
            scores = {name: float(function(labels[index], values[index])) for name, values in predictions.items()}
            for name, value in scores.items():
                draws[name][metric].append(value)
            differences[metric].append(scores['7b']-scores['1b'])
    if len(differences['auroc']) < .9 * CONFIG['bootstrap_repetitions']:
        raise ValueError('Too few valid component-bootstrap samples')
    for name in predictions:
        for metric in functions:
            result[name][metric]['ci95'] = np.quantile(draws[name][metric], [.025, .975]).tolist()
    delta = {metric: {'value': result['7b'][metric]['value']-result['1b'][metric]['value'],
                      'ci95': np.quantile(values, [.025, .975]).tolist()}
             for metric, values in differences.items()}
    return result, delta


def baseline_1b(keys, labels):
    from . import comparison
    context = comparison.load_context(ROOT)
    values = comparison.load_q9(ROOT, context)['frozen']
    if context['validation'].variant_key.tolist() != list(keys) or context['validation'].label.tolist() != list(labels):
        raise ValueError('Q9 baseline keys or labels do not match Q10 validation')
    return values


def export_comparison(result):
    q1.write_json(OUTPUT / 'comparison_results.json', {
        'q1_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
        'predictions_sha256': result['artifacts']['comparison_predictions.csv'],
        'methods': {'frozen_7b': MODEL_NAME}, 'limitations': LIMITATIONS, 'sources': sources(),
        'artifacts': {name: q1.digest_file(OUTPUT / name) for name in ['protocol.json', 'metrics.json']}}, frozen=True)


def fit_candidates(x_train, y_train, x_validation, y_validation):
    """Q9's same objective and C grid, with a larger convergence limit for 7B."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    train, validation = scaler.transform(x_train), scaler.transform(x_validation)
    candidates, best, best_auc = [], None, -np.inf
    for c in sorted(CONFIG['C_grid']):
        model = LogisticRegression(C=c, penalty='l2', class_weight='balanced', solver='lbfgs',
                                   max_iter=30000, tol=1e-5, random_state=CONFIG['seed']).fit(train, y_train)
        iterations = int(model.n_iter_.max())
        if iterations >= model.max_iter:
            raise RuntimeError(f'C={c:g} failed to converge after {iterations} iterations')
        score = float(roc_auc_score(y_validation, model.decision_function(validation)))
        candidates.append({'C': c, 'validation_auroc': score, 'iterations': iterations})
        print(f'C={c:g}: converged in {iterations} iterations.', flush=True)
        if score > best_auc:
            best_auc = score
            best = {'C': c, 'mean': scaler.mean_, 'scale': scaler.scale_,
                    'coef': model.coef_[0], 'intercept': model.intercept_[0]}
    return best, candidates


def fit_and_evaluate():
    from threadpoolctl import threadpool_limits
    identity = verify_protocol()
    path = OUTPUT / 'metrics.json'
    if path.exists():
        result = q1.read_json(path)
        if result['identity'] != identity:
            raise ValueError('Recorded Q10 results belong to another experiment')
        for name, checksum in result['artifacts'].items():
            q1.verify_file(OUTPUT / name, checksum)
        q1.verify_file(q9.OUTPUT / 'validation_predictions.npz', result['baseline_predictions_sha256'])
        export_comparison(result)
        print('Verified completed 7B classifier and validation results.')
        return result
    keys, values = load_features()
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv').set_index('variant_key').loc[keys]
    masks = {split: pilot.split.eq(split).to_numpy() for split in q1.SPLITS}
    labels = {split: q1.load_partition_labels(split, keys[mask]) for split, mask in masks.items()}
    reference = baseline_1b(keys[masks['validation']], labels['validation'])
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        model, candidates = fit_candidates(values[masks['train']], labels['train'],
            values[masks['validation']], labels['validation'])
        scores = q2.predict_saved(model, values[masks['validation']])
    fit_seconds = time.perf_counter()-started
    q2.save_npz(OUTPUT / 'classifier.npz', **model)
    with np.load(OUTPUT / 'classifier.npz', allow_pickle=False) as saved, threadpool_limits(limits=4):
        if not np.array_equal(scores, q2.predict_saved(saved, values[masks['validation']])):
            raise ValueError('Reloaded classifier predictions differ')
    q1.write_json(OUTPUT / 'selection.json', {'identity': identity, 'candidates': candidates,
        'chosen_C': model['C'], 'fitted_on': 'train', 'selected_on': 'validation',
        'fit_seconds': fit_seconds, 'features_sha256': q1.digest_file(OUTPUT / 'feature_manifest.json')}, frozen=True)
    groups = pilot.loc[masks['validation'], 'component'].to_numpy()
    metrics, delta = paired_metrics(labels['validation'], {'7b': scores, '1b': reference}, groups)
    table = pd.DataFrame({'variant_key': keys[masks['validation']], 'component': groups,
                          'label': labels['validation'], 'frozen_7b': scores, 'frozen_1b': reference})
    table.to_csv(OUTPUT / 'validation_predictions.csv', index=False)
    table.drop(columns='frozen_1b').to_csv(OUTPUT / 'comparison_predictions.csv', index=False)
    artifacts = ['classifier.npz', 'selection.json', 'validation_predictions.csv',
                 'comparison_predictions.csv', 'feature_manifest.json', 'preflight.json']
    if (OUTPUT / 'feature_reuse.json').exists():
        artifacts.append('feature_reuse.json')
    result = {'status': 'complete', 'identity': identity, 'chosen_C': model['C'],
              'feature_producer_identity': feature_identity(identity),
              'train_variants': int(masks['train'].sum()), 'validation_variants': len(scores),
              'metrics': metrics, '7b_minus_1b': delta, 'fit_seconds': fit_seconds,
              'baseline_predictions_sha256': q1.digest_file(q9.OUTPUT / 'validation_predictions.npz'),
              'artifacts': {name: q1.digest_file(OUTPUT / name) for name in artifacts}, 'limitations': LIMITATIONS}
    q1.write_json(path, result, frozen=True)
    export_comparison(result)
    print(f"Selected C={model['C']:g}: validation AUROC {metrics['7b']['auroc']['value']:.3f}, "
          f"average precision {metrics['7b']['average_precision']['value']:.3f}.")
    return result


def show_results():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    result = fit_and_evaluate()
    table = q1.read_csv(OUTPUT / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for column, label in [('frozen_1b', '1B frozen head'), ('frozen_7b', '7B frozen head')]:
        fpr, tpr, _ = roc_curve(table.label, table[column])
        precision, recall, _ = precision_recall_curve(table.label, table[column])
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Validation precision–recall')
    for ax in axes:
        ax.legend()
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=150)
    plt.show()
    q0.details('Metrics, paired 95% component-bootstrap intervals and compute', result)


def show_conclusion():
    from IPython.display import Markdown, display
    result = q1.read_json(OUTPUT / 'metrics.json')
    delta = result['7b_minus_1b']['auroc']
    display(Markdown(f"The frozen **7B** classifier reaches **{result['metrics']['7b']['auroc']['value']:.3f} AUROC**, "
        f"a **{delta['value']:+.3f}** change versus 1B (paired 95% interval "
        f"[{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]). "
        'This compares the two BioNeMo configurations; the 1B checkpoint’s BF16 sensitivity prevents '
        'attributing the difference solely to size. Validation selected C; no untouched test was evaluated.'))
