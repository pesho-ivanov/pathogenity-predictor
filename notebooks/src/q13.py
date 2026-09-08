"""Q13: converged classifier fitting and gentler, matched Evo2 adaptation."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import subprocess
import time

import numpy as np

from . import q0, q1, q11, q12, q9_environment

ROOT = q11.ROOT
OUTPUT = ROOT / 'notebooks/results/q13'
CONFIG = {
    'seed': 42, 'training_variants': 24576, 'validation_variants': 2048,
    'context_bp': 512, 'feature_dimension': 8194,
    'feature_form': 'with_magnitude',
    'head_strengths': [1e-4, 1e-3, 1e-2], 'head_max_iter': 1000,
    'head_tolerance': 1e-6, 'adapter_lrs': [3e-5, 1e-5], 'head_lr': 1e-5,
    'adapter_weight_decay': .01, 'clip_grad': 1.,
    'betas': [.9, .999], 'epsilon': 1e-8, 'lora': q11.CONFIG['lora'],
    'microbatch_variants': 32, 'max_steps': 512, 'checkpoint_steps': 32,
    'evaluation_steps': 256, 'warmup_steps': 32, 'minimum_lr_ratio': .1,
    'total_seconds': 3600, 'report_seconds': 180, 'validation_margin': 1.3,
    'reload_variants': 64, 'bootstrap_repetitions': 1000,
    'promotion_auroc': .005,
    'head_objective': 'FP64 full-batch fitting of balanced training-subset BCE plus explicit L2 weight penalty; unpenalized intercept. Verify FP32 deployment equivalence and use the same objective during both FP32 continuation branches.',
    'gradient_clipping': 'Separate adapter and head gradient-norm clipping; head clipping is identical to the frozen control.',
    'selection': 'Choose converged fitted head by sampled AUROC, then AP, then stronger L2. Choose checkpoints by AUROC, then AP. Retain step zero among control candidates. Promote LoRA only if AUROC improves by at least 0.005 with nondecreasing AP against the strongest current or Q12 frozen control.',
}
LIMITATIONS = (
    'User-authorized exploratory subsets: the same 24,576 training variants and '
    '2,048 validation variants as Q12. Validation selects regularization, adapter '
    'learning rate and checkpoints; repeated development selection and bootstrap '
    'intervals do not establish performance on an untouched holdout. The paired '
    'effect compares LoRA with its control after the same examples and update count; '
    'promotion also requires improvement over the strongest current or Q12 frozen '
    'control. No clinical validation; pretraining exposure, homology and shared-patient '
    'overlap remain unresolved.'
)

require = q12.require
metric = q12.metric
better = q12.better


def intervals(labels, predictions, groups, repetitions=None):
    return q12.intervals(labels, predictions, groups,
                        CONFIG['bootstrap_repetitions'] if repetitions is None else repetitions)


def promote(lora, control):
    return (lora['auroc'] - control['auroc'] >= CONFIG['promotion_auroc'] - 1e-12
            and lora['average_precision'] >= control['average_precision'])


class Budget:
    def __init__(self, started, seconds_per_variant=.05):
        self.started, self.seconds_per_variant = started, seconds_per_variant

    def elapsed(self):
        return time.time() - self.started

    def reserve(self):
        return (self.seconds_per_variant * (CONFIG['validation_variants'] + CONFIG['reload_variants'])
                * CONFIG['validation_margin'] + CONFIG['report_seconds'])

    def can_train(self, next_seconds=5.):
        return self.elapsed() + self.reserve() + next_seconds < CONFIG['total_seconds']


def sources():
    names = ['q13.py', 'q13_backend.py', 'q13_heads.py', 'refresh_q13.py']
    return q12.sources() | {
        f'notebooks/src/{name}': q1.digest_file(ROOT / 'notebooks/src' / name)
        for name in names
    }


def prepare():
    """Freeze verified parent features, fixed cohorts and this implementation."""
    parent_result = q12.verified_results()
    rows, labels, groups, checks = q12.inputs()
    parent = q1.read_json(q12.OUTPUT / 'protocol.json')
    require(len(rows['train']) == CONFIG['training_variants']
            and len(rows['validation']) == CONFIG['validation_variants'],
            'Q13 must retain the exact authorized Q12 subsets')
    require(parent_result['feature_form'] == CONFIG['feature_form'],
            'Q12 selected feature form differs from the Q13 design')
    value = {
        'configuration': CONFIG, 'sources': sources(),
        'q12_identity': q12.verify_protocol(),
        'q12_metrics_sha256': q1.digest_file(q12.OUTPUT / 'metrics.json'),
        'q12_feature_manifest_sha256': q1.digest_file(q12.OUTPUT / 'feature_manifest.json'),
        'q1_protocol_sha256': parent['q1_protocol_sha256'],
        'vcf_exports': parent['vcf_exports'],
        'training_variant_keys': rows['train'].variant_key.tolist(),
        'validation_variant_keys': rows['validation'].variant_key.tolist(),
        'training_pool': parent['training_pool'], 'full_validation': parent['full_validation'],
        'labels_sha256': {key: q1.fingerprint(labels[key].tolist()) for key in labels},
        'components_sha256': {key: q1.fingerprint(groups[key].tolist()) for key in groups},
        'limitations': LIMITATIONS,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q1.write_json(OUTPUT / 'protocol.json', value, frozen=True)
    q1.write_json(OUTPUT / 'input_checks.json', checks, frozen=True)
    for relative, sha in value['sources'].items():
        source = ROOT / relative
        target = OUTPUT / 'source_snapshot' / relative
        q1.verify_file(source, sha)
        if target.exists():
            q1.verify_file(target, sha)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            q1.verify_file(target, sha)
    print(f"Frozen {len(rows['train']):,}/{parent['training_pool']:,} training rows and "
          f"{len(rows['validation']):,}/{parent['full_validation']:,} validation rows; "
          'verified Q12 frozen features will be reused.', flush=True)
    q0.details('Q13 configuration, inherited leakage checks and parent provenance', {
        'configuration': CONFIG, 'checks': checks, 'q12_identity': value['q12_identity'],
        'q12_metrics_sha256': value['q12_metrics_sha256'],
        'q12_feature_manifest_sha256': value['q12_feature_manifest_sha256'],
        'training_variants': len(rows['train']), 'validation_variants': len(rows['validation']),
        'validation_components': len(np.unique(groups['validation'])),
        'limitations': LIMITATIONS,
    })
    return value


def verify_protocol():
    value = q1.read_json(OUTPUT / 'protocol.json')
    require(value['configuration'] == CONFIG and value['sources'] == sources(),
            'Q13 sources or configuration changed; archive before starting another experiment')
    require(value['q12_identity'] == q12.verify_protocol(), 'Q12 source identity changed')
    q1.verify_file(q12.OUTPUT / 'metrics.json', value['q12_metrics_sha256'])
    q1.verify_file(q12.OUTPUT / 'feature_manifest.json', value['q12_feature_manifest_sha256'])
    parent = q1.read_json(q12.OUTPUT / 'protocol.json')
    for key in ['training_variant_keys', 'validation_variant_keys', 'q1_protocol_sha256', 'vcf_exports']:
        require(value[key] == parent[key], f'Q13 fixed input differs from Q12: {key}')
    return q1.fingerprint(value)


def inputs():
    verify_protocol()
    rows, labels, groups, checks = q12.inputs()
    value = q1.read_json(OUTPUT / 'protocol.json')
    for split, key in [('train', 'training_variant_keys'), ('validation', 'validation_variant_keys')]:
        require(rows[split].variant_key.tolist() == value[key], 'Q13 cohort membership changed')
        require(q1.fingerprint(labels[split].tolist()) == value['labels_sha256'][split],
                'Q13 labels changed')
        require(q1.fingerprint(groups[split].tolist()) == value['components_sha256'][split],
                'Q13 components changed')
    require(not set(groups['train']) & set(groups['validation']), 'Q13 components cross partitions')
    return rows, labels, groups, checks


def run_experiment():
    """Execute training, never substituting completed metrics for an actual run."""
    verify_protocol()
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (OUTPUT / 'metrics.json').exists(),
                'Q13 already has completed results. Use refresh_q13 to archive them and execute a fresh run.')
        path = OUTPUT / 'execution.json'
        if not path.exists():
            q1.write_json(path, {'started_utc': float(os.environ.get('Q13_RUN_STARTED_UTC', time.time()))},
                          frozen=True)
        log = OUTPUT / 'explore.log'
        if log.exists():
            archive = OUTPUT / 'logs'
            archive.mkdir(exist_ok=True)
            log.rename(archive / f'explore-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q13_backend', 'explore']
        print(f"Explore: {log.relative_to(ROOT)}", flush=True)
        with log.open('w') as stream:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            try:
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        require(status == 0, 'Q13 exploration failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
    return verified_results()


def verified_results():
    result = q1.read_json(OUTPUT / 'metrics.json')
    require(result.get('status') == 'complete' and result.get('identity') == verify_protocol(),
            'Incomplete or stale Q13 results')
    require(result.get('scope') == 'sampled_validation', 'Q13 evaluation scope mismatch')
    require(result.get('reloaded_predictions_verified') is True
            and result.get('frozen_unchanged') is True, 'Q13 checkpoint or backbone verification failed')
    require(bool(result.get('artifacts')), 'Q13 result artifact hashes are missing')
    for name, sha in result['artifacts'].items():
        path = OUTPUT / name
        require(path.resolve().is_relative_to(OUTPUT.resolve()), 'Q13 artifact path escapes the result directory')
        q1.verify_file(path, sha)
    required = {'lora', 'matched_control', 'best_control', 'q12_control', 'selected'}
    require(required <= set(result['metrics']), 'Q13 comparison results are incomplete')
    points = {name: {key: value['value'] for key, value in result['metrics'][name].items()}
              for name in required}
    strongest = max(['best_control', 'q12_control'],
                    key=lambda name: (points[name]['auroc'], points[name]['average_precision']))
    require(bool(result['promotion_passed']) == promote(points['lora'], points[strongest]),
            'Q13 promotion result does not match its frozen rule')
    require((result['selected_model'] == 'lora') == result['promotion_passed'],
            'Q13 selected model disagrees with promotion')
    return result


def show_results():
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    result = verified_results()
    history = q1.read_json(OUTPUT / 'training_history.json')
    trials = history.get('trials', history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for name, progress in trials.items():
        if not isinstance(progress, dict) or not progress.get('evaluations'):
            continue
        evaluations = progress['evaluations']
        for branch, style in [('lora', '-'), ('control', '--')]:
            for axis, key in zip(axes, ['auroc', 'average_precision']):
                axis.plot([row['steps'] for row in evaluations],
                          [row[branch][key] for row in evaluations],
                          linestyle=style, marker='o', label=f'{name}: {branch}')
    for axis, key, label in zip(axes, ['auroc', 'average_precision'], ['AUROC', 'Average precision']):
        axis.axhline(result['metrics']['q12_control'][key]['value'], color='gray',
                     linestyle=':', label='Q12 frozen control')
        axis.set(xlabel='Matched continuation updates', ylabel=f'Sampled validation {label}')
        axis.legend(fontsize=7)
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    rows = ['| Model | AUROC (95% interval) | AP (95% interval) |',
            '| --- | ---: | ---: |']
    names = {'q12_control': 'Q12 frozen control', 'matched_control': 'Matched frozen control',
             'best_control': 'Best Q13 frozen control', 'lora': 'Best Q13 LoRA', 'selected': 'Selected model'}
    for name, label in names.items():
        values = result['metrics'][name]
        formatted = [f"{values[key]['value']:.3f} [{values[key]['ci95'][0]:.3f}, {values[key]['ci95'][1]:.3f}]"
                     for key in ['auroc', 'average_precision']]
        rows.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    display(Markdown('\n'.join(rows)))
    q0.details('Head optimization candidates and convergence', q1.read_json(OUTPUT / 'head_results.json'))
    q0.details('Completed Q13 measurements, integrity checks and provenance', result)


def show_conclusion():
    from IPython.display import Markdown, display

    result = verified_results()
    selected = result['metrics']['selected']
    paired = result['paired_lora_minus_matched_control']['auroc']
    decision = ('LoRA met the prespecified promotion rule' if result['promotion_passed']
                else 'LoRA did not meet the prespecified promotion rule; retain the strongest frozen classifier')
    message = (
        f"**Conclusion.** {decision}. Selected **{result['selected_model']}**: "
        f"AUROC **{selected['auroc']['value']:.3f}**, AP **{selected['average_precision']['value']:.3f}**, "
        f"on **{CONFIG['validation_variants']:,} development validation variants**. "
        f"Best adapter trial: **{result['best_lora_trial']}**, checkpoint **{result['best_steps']} updates**. "
        f"LoRA minus its matched control: AUROC **{paired['value']:+.4f}** "
        f"(95% component interval **[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]**). "
        'Promotion requires at least **0.005 AUROC** improvement and no AP reduction against the '
        'strongest Q13 or Q12 frozen control, including the initial fitted head. '
        f"The measured experiment took **{result['seconds']/60:.1f} minutes**; "
        f"{'within' if result['within_one_hour'] else 'exceeding'} the one-hour target. "
        'These sampled results do not replace the complete-cohort benchmark. ' + LIMITATIONS
    )
    display(Markdown(message))
