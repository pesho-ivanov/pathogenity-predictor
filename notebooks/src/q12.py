"""Q12: budgeted magnitude-aware LoRA exploration and separate full validation."""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import q0, q1, q1_full, q11, q9_environment

ROOT = q11.ROOT
OUTPUT = ROOT / 'notebooks/results/q12'
CONFIG = {
    'seed': 42, 'training_variants': 24576, 'validation_variants': 2048,
    'context_bp': 512, 'feature_dimension': 8194,
    'forms': ['original', 'with_magnitude', 'difference_magnitude'],
    'features': 'Q11 unit-RMS reference/difference vectors plus log difference RMS and log relative difference RMS',
    'rms_floor': 1e-12, 'standardization': 'All means/stds fitted on selected training rows only; std floor 1e-6; frozen thereafter',
    'head_epochs': 20, 'head_batch': 512, 'head_initial_lr': 1e-3,
    'head_lr': 3e-5, 'adapter_lr': 1e-4, 'weight_decay': .01, 'clip_grad': 1.,
    'betas': [.9, .999], 'epsilon': 1e-8, 'lora': q11.CONFIG['lora'],
    'microbatch_variants': 32, 'max_steps': 768, 'checkpoint_steps': 32,
    'evaluation_steps': 256, 'patience': 2, 'minimum_progress': .002,
    'promotion_auroc': .005, 'total_seconds': 3600,
    'report_seconds': 180, 'validation_margin': 1.3, 'reload_variants': 64,
    'bootstrap_repetitions': 1000,
    'selection': 'Head: AUROC, AP, fewer features. LoRA: best AUROC/AP; promote only above best control AUROC by >=0.005 with nondecreasing AP.',
}
LIMITATIONS = ('User-authorized exploratory training subset and fixed validation sample. '
    'Validation chooses features, checkpoints and early stopping; bootstrap intervals do not correct selection bias. '
    'The matched control uses the same initialization, examples, batches and continuation update count as LoRA. '
    'The best control can come from another monitored step and is reported separately. '
    'No untouched test or clinical validation; pretraining/homology/shared-patient overlap remain unresolved.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def columns(form, hidden=4096):
    if form == 'original':
        return np.arange(2*hidden)
    if form == 'with_magnitude':
        return np.arange(2*hidden+2)
    if form == 'difference_magnitude':
        return np.arange(hidden, 2*hidden+2)
    raise ValueError('Unknown feature form')


def fit_scaler(training):
    require(training.ndim == 2 and np.isfinite(training).all(), 'Invalid training features')
    return (training.mean(axis=0, dtype=np.float64).astype(np.float32),
            np.maximum(training.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32))


def metric(labels, scores):
    require(len(labels) == len(scores) and set(np.unique(labels)) == {0, 1}
            and np.isfinite(scores).all(), 'Metrics require finite scores and both classes')
    return {'auroc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores))}


def better(a, b):
    return (a['auroc'], a['average_precision']) > (b['auroc'], b['average_precision'])


def select_form(results):
    return min(results, key=lambda form: (-results[form]['auroc'],
        -results[form]['average_precision'], len(columns(form)), form))


def promote(lora, control):
    return (lora['auroc']-control['auroc'] >= CONFIG['promotion_auroc']-1e-12
            and lora['average_precision'] >= control['average_precision'])


def monitor(progress, value):
    """Update patience independently of the best individual checkpoint."""
    if value >= progress['monitor_auroc'] + CONFIG['minimum_progress']:
        progress['monitor_auroc'], progress['bad_checks'] = value, 0
    else:
        progress['bad_checks'] += 1
    return progress['bad_checks'] >= CONFIG['patience']


class Budget:
    def __init__(self, started, seconds_per_variant=.05):
        self.started, self.seconds_per_variant = started, seconds_per_variant

    def elapsed(self):
        return time.time()-self.started

    def reserve(self):
        return (self.seconds_per_variant*(CONFIG['validation_variants']+CONFIG['reload_variants'])
                * CONFIG['validation_margin'] + CONFIG['report_seconds'])

    def can_train(self, next_seconds=5.):
        return self.elapsed()+self.reserve()+next_seconds < CONFIG['total_seconds']


def sources():
    paths = [ROOT/'notebooks/src'/name for name in ['q12.py', 'q12_backend.py']]
    return q11.sources() | {str(p.relative_to(ROOT)): q1.digest_file(p) for p in paths}


def prepare():
    """Freeze Q1 inputs and the authorized subset before any fitting."""
    q11.verified_results()
    protocol, manifest, dna, indexes, labels, checks = q11.verify_inputs()
    history = q1.read_json(q11.OUTPUT/'training_history.json')
    order = np.asarray(history['order'])
    expected = np.random.default_rng(CONFIG['seed']).permutation(len(indexes['train']))
    require(np.array_equal(order, expected) and history['offset'] == CONFIG['training_variants'],
            'Q11 training membership differs from the approved Q12 comparison')
    train = dna.iloc[indexes['train']].iloc[order[:CONFIG['training_variants']]]
    seen = pd.read_csv(q11.OUTPUT/'training_seen.csv')
    require(train.variant_key.tolist() == seen.variant_key.tolist(), 'Q11 seen keys changed')
    val = dna.iloc[indexes['validation']]
    selected = np.sort(np.random.default_rng(CONFIG['seed']).choice(len(val), CONFIG['validation_variants'], replace=False))
    val = val.iloc[selected]
    diagnostic = q1.read_json(q11.OUTPUT/'diagnostics/lora_quality/protocol.json')
    require(val.variant_key.tolist() == diagnostic['validation_variant_keys'], 'The existing validation sample changed')
    value = {'configuration': CONFIG, 'sources': sources(), 'q11_identity': q11.verify_protocol(),
        'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT/'protocol.json'), 'vcf_exports': protocol['vcf_exports'],
        'training_variant_keys': train.variant_key.tolist(), 'validation_variant_keys': val.variant_key.tolist(),
        'training_pool': len(indexes['train']), 'full_validation': len(indexes['validation']),
        'q11_seen_sha256': q1.digest_file(q11.OUTPUT/'training_seen.csv'),
        'sample_protocol_sha256': q1.digest_file(q11.OUTPUT/'diagnostics/lora_quality/protocol.json'),
        'original_preflight_sha256': q1.digest_file(q11.OUTPUT/'preflight.json'), 'limitations': LIMITATIONS}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q1.write_json(OUTPUT/'protocol.json', value, frozen=True)
    q1.write_json(OUTPUT/'input_checks.json', checks, frozen=True)
    print(f"Frozen {len(train):,}/{len(indexes['train']):,} training rows and {len(val):,}/{len(indexes['validation']):,} validation rows.", flush=True)
    q0.details('Q12 configuration, fixed cohorts and leakage checks', {'configuration': CONFIG, 'checks': checks,
               'training_variants': len(train), 'validation_variants': len(val), 'limitations': LIMITATIONS})
    return value


def verify_protocol():
    p = q1.read_json(OUTPUT/'protocol.json')
    require(p['configuration'] == CONFIG and p['sources'] == sources(), 'Q12 sources or configuration changed; archive before a new experiment')
    require(p['q11_identity'] == q11.verify_protocol(), 'Q11 source identity changed')
    q1.verify_file(q1_full.OUTPUT/'protocol.json', p['q1_protocol_sha256'])
    q1.verify_file(q11.OUTPUT/'training_seen.csv', p['q11_seen_sha256'])
    q1.verify_file(q11.OUTPUT/'diagnostics/lora_quality/protocol.json', p['sample_protocol_sha256'])
    for name, sha in p['vcf_exports'].items():
        q1.verify_file(ROOT/'data'/name, sha)
    return q1.fingerprint(p)


def inputs():
    verify_protocol()
    _, manifest, dna, _, _, checks = q11.verify_inputs()
    p = q1.read_json(OUTPUT/'protocol.json')
    rows, labels, groups = {}, {}, {}
    for split, key in [('train', 'training_variant_keys'), ('validation', 'validation_variant_keys')]:
        rows[split] = dna.set_index('variant_key').loc[p[key]].reset_index()
        frame = q1_full.read_labels(q1_full.VCF_FILES[split], p['vcf_exports'][q1_full.VCF_FILES[split].name], manifest, split)
        labels[split] = frame.set_index('variant_key').loc[p[key], 'label'].to_numpy(dtype=int)
        groups[split] = manifest.set_index('variant_key').loc[p[key], 'component'].to_numpy()
    require(not set(groups['train']) & set(groups['validation']), 'Q12 components cross partitions')
    return rows, labels, groups, checks


def intervals(labels, predictions, groups, repetitions=None):
    repetitions = CONFIG['bootstrap_repetitions'] if repetitions is None else repetitions
    points = {k: metric(labels, v) for k, v in predictions.items()}
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(CONFIG['seed'])
    draws = {k: {m: [] for m in ['auroc', 'average_precision']} for k in predictions}
    differences = {m: [] for m in ['auroc', 'average_precision']}
    for _ in range(repetitions):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[idx])) < 2:
            continue
        scores = {k: metric(labels[idx], v[idx]) for k, v in predictions.items()}
        for k, values in scores.items():
            for m, value in values.items():
                draws[k][m].append(value)
        if {'lora', 'matched_control'} <= set(scores):
            for m in differences:
                differences[m].append(scores['lora'][m]-scores['matched_control'][m])
    require(all(len(v['auroc']) >= .9*repetitions for v in draws.values()), 'Too few valid component bootstrap draws')
    metrics = {k: {m: {'value': points[k][m], 'ci95': np.quantile(v, [.025,.975]).tolist()}
                   for m, v in values.items()} for k, values in draws.items()}
    paired = {m: {'value': points['lora'][m]-points['matched_control'][m],
                  'ci95': np.quantile(v, [.025,.975]).tolist()} for m, v in differences.items() if v}
    return metrics, paired


def run_gpu(action):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUTPUT/f'{action}.log'
        if path.exists():
            old = OUTPUT/'logs'; old.mkdir(exist_ok=True)
            path.rename(old/f'{action}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q12_backend', action]
        print(f'{action}: {path.relative_to(ROOT)}', flush=True)
        with path.open('w') as stream:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     pass_fds=(lock.fileno(),))
            try:
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        require(status == 0, f'Q12 {action} failed: '+ '\n'.join(path.read_text(errors='replace').splitlines()[-25:]))


def verified_results(full=False):
    directory = OUTPUT/'full' if full else OUTPUT
    result = q1.read_json(directory/'metrics.json')
    require(result['status'] == 'complete' and result['identity'] == verify_protocol(), 'Incomplete or stale Q12 results')
    require(result['scope'] == ('full_validation' if full else 'sampled_validation'), 'Q12 evaluation scope mismatch')
    for name, sha in result['artifacts'].items():
        q1.verify_file(directory/name, sha)
    if full:
        q1.verify_file(OUTPUT/'selected_model.pt', result['selected_model_sha256'])
        q1.verify_file(OUTPUT/'metrics.json', result['exploration_sha256'])
    return result


def run_experiment():
    if not (OUTPUT/'metrics.json').exists():
        path = OUTPUT/'execution.json'
        if not path.exists():
            q1.write_json(path, {'started_utc': float(os.environ.get('Q12_RUN_STARTED_UTC', time.time()))}, frozen=True)
        run_gpu('explore')
    return verified_results()


def export_full_comparison(result):
    require(result.get('scope') == 'full_validation' and result.get('status') == 'complete'
            and result.get('validation_variants') == q1.read_json(OUTPUT/'protocol.json')['full_validation']
            and result.get('reloaded_predictions_verified') is True, 'Sampled Q12 results cannot enter the full comparison')
    directory = OUTPUT/'full'
    verified_results(full=True)
    p = q1.read_json(OUTPUT/'protocol.json')
    frame = pd.read_csv(directory/'validation_predictions.csv')
    require(len(frame) == result['validation_variants'] and not frame.variant_key.duplicated().any(), 'Incomplete full predictions')
    frame.to_csv(OUTPUT/'comparison_predictions.csv', index=False)
    name = f"Evo2 7B Q12 selected {result['selected_model']} ({result['feature_form']})"
    spec = {'scope': 'full_validation', 'require_complete': True, 'q1_protocol_sha256': p['q1_protocol_sha256'],
        'vcf_exports': p['vcf_exports'], 'methods': {'selected': name},
        'sources': p['sources'], 'predictions_sha256': q1.digest_file(OUTPUT/'comparison_predictions.csv'),
        'artifacts': {'full/metrics.json': q1.digest_file(directory/'metrics.json')},
        'runtimes': {'selected': {'seconds': result['seconds'], 'scope': 'Full-validation loading, scoring, reload check and bootstrap; excludes exploratory selection.'}},
        'limitations': LIMITATIONS + ' Selected using the Q12 development sample; evaluated on complete validation.'}
    q1.write_json(OUTPUT/'comparison_results.json', spec, frozen=True)


def run_full_validation():
    verified_results()
    if not (OUTPUT/'full/metrics.json').exists():
        run_gpu('full-validation')
    result = verified_results(full=True)
    return result


def show_results(full=False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    r = verified_results(full)
    directory = OUTPUT/'full' if full else OUTPUT
    frame = pd.read_csv(directory/'validation_predictions.csv')
    fig, axes = plt.subplots(1, 1 if full else 2, figsize=(6,3.5) if full else (10,3.5), layout='constrained')
    axes = np.atleast_1d(axes)
    for key in (['selected'] if full else ['lora','matched_control','best_control']):
        precision, recall, _ = precision_recall_curve(frame.label, frame[key])
        axes[0].plot(recall, precision, label=key.replace('_',' '))
    axes[0].axhline(frame.label.mean(), color='gray', linestyle='--')
    axes[0].set(xlabel='Recall', ylabel='Precision'); axes[0].legend(fontsize=8)
    if not full:
        history = q1.read_json(OUTPUT/'training_history.json')['evaluations']
        for key in ['lora','control']:
            axes[1].plot([v['steps'] for v in history], [v[key]['auroc'] for v in history], marker='o', label=key)
        axes[1].set(xlabel='Matched continuation updates', ylabel='Sampled validation AUROC'); axes[1].legend()
    fig.savefig(directory/'validation_curves.png', dpi=160); plt.show()
    q0.details('Completed Q12 measurements, uncertainty and provenance', r)


def show_conclusion(full=False):
    from IPython.display import Markdown, display
    r = verified_results(full)
    m = r['metrics']['selected']
    message = (f"**Conclusion.** Selected **{r['selected_model']}**, using **{r['feature_form']}** features. "
        f"AUROC **{m['auroc']['value']:.3f}**, average precision **{m['average_precision']['value']:.3f}**, "
        f"on **{r['validation_variants']:,}** validation variants. ")
    if not full:
        paired = r['paired_lora_minus_matched_control']['auroc']
        message += (f"Best LoRA minus its control at the same update count: AUROC **{paired['value']:+.3f}** "
            f"(95% component interval **[{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}]**). "
            f"The selection rule {'promoted LoRA' if r['selected_model']=='lora' else 'retained the frozen-backbone control'}. "
            f"Continuation stopped after **{r['optimizer_steps']}** updates: **{r['stop_reason']}**. "
            'These sampled development results do not replace full-cohort benchmark measurements. ')
    display(Markdown(message + LIMITATIONS))
