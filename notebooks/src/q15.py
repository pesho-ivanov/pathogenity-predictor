"""Q15: cached frozen-prefix replay for paired BCE and ranking-loss adaptation."""

from datetime import datetime, timezone
import fcntl
import os
import shutil
import subprocess
import time

import numpy as np

from . import q0, q1, q11, q12, q13, q14, q9_environment

ROOT = q11.ROOT
OUTPUT = ROOT / 'notebooks/results/q15'
CONFIG = {
    'seed': 42, 'training_variants': 24576, 'validation_variants': 2048,
    'context_bp': 512, 'feature_dimension': 8194,
    'feature_form': 'with_magnitude',
    'prefit_head': 'Strongest frozen control recorded by completed Q14; exact control-head checkpoint and scaler held fixed',
    'objectives': ['weighted_bce', 'pairwise_logistic'], 'head_lr': 0.,
    'adapter_lr_selection': 'Actual adapter_lr from Q14 best_lora_trial; frozen in Q15 protocol before training',
    'adapter_weight_decay': .01, 'clip_grad': 1.,
    'betas': [.9, .999], 'epsilon': 1e-8,
    'lora': {'blocks': [29, 30], 'rank': 8, 'alpha': 16, 'dropout': 0.,
             'A_init': 'xavier', 'B_init': 'zero',
             'targets': ['decoder.layers.29.mixer.dense_projection', 'decoder.layers.29.mixer.dense',
                         'decoder.layers.30.mixer.dense_projection', 'decoder.layers.30.mixer.dense']},
    'chunk_variants': 4096, 'passes_per_chunk': 3,
    'microbatch_variants': 32, 'max_attempted_steps': 2304, 'max_steps': 2304, 'checkpoint_steps': 32,
    'evaluation_cadence': 'Initial, after each training chunk (384 attempted batches), and after a final partial chunk',
    'warmup_steps': 64, 'minimum_lr_ratio': .1,
    'total_seconds': 3600, 'report_seconds': 180, 'validation_margin': 1.3,
    'reload_variants': 64, 'bootstrap_repetitions': 1000,
    'promotion_auroc': .005,
    'training_objectives': 'Weighted BCE with verified Q12 training-class weights versus mean within-batch positive-negative pairwise logistic loss. Ranking batches with one class skip the optimizer; attempted cursors and real update counts remain separate.',
    'prefix_cache': 'CPU BF16 frozen-prefix activations before block29: complete 2048-variant validation cache (~16GiB), one 4096-variant training chunk at a time (~32GiB). Replay the trainable tail for three passes per chunk.',
    'gradient_clipping': 'Clip adapter gradient norm at 1.0. Classifier parameters remain fixed and are checked unchanged.',
    'selection': 'Initialize from the strongest frozen control recorded by Q14 by AUROC/AP, preferring Q14 on exact ties. Compare two objectives at matched attempted-example cursors. Hold the classifier fixed and select adapter checkpoints by sampled AUROC/AP. Promote only with AUROC gain >=0.005 and nondecreasing AP against the strongest freshly reproduced or inherited frozen control.',
}
LIMITATIONS = (
    'User-authorized exploratory subsets: the same 24,576 training variants and '
    '2,048 validation variants as Q12/Q13/Q14. Earlier validation selected the inherited '
    'classifier and learning rate; Q15 validation selects objective and checkpoints. '
    'Repeated development selection and bootstrap '
    'intervals do not establish performance on an untouched holdout. '
    'Both arms use matched attempted-example cursors; one-class ranking batches skip '
    'optimizer updates and must be reported separately. Prefix extraction, cache '
    'transfer and replay count toward this experiment\'s one-hour budget. '
    'Promotion requires improvement over the strongest tested frozen '
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


def promotion_control(fresh_points, protocol):
    inherited = {key: protocol['initial_control_metrics'][key]['value']
                 for key in ['auroc', 'average_precision']}
    return max([fresh_points, inherited], key=lambda value: (value['auroc'], value['average_precision']))


class Budget:
    def __init__(self, started, seconds_per_variant=.05):
        self.started, self.seconds_per_variant = started, seconds_per_variant

    def elapsed(self):
        return time.time() - self.started

    def remaining(self):
        return max(0., CONFIG['total_seconds'] - self.elapsed())

    def can_work(self, seconds, reserve=0.):
        require(seconds >= 0 and reserve >= 0 and np.isfinite(seconds + reserve), 'Invalid replay budget estimate')
        return self.elapsed() + seconds + reserve < CONFIG['total_seconds']

    def reserve(self):
        return (self.seconds_per_variant * (CONFIG['validation_variants'] + CONFIG['reload_variants'])
                * CONFIG['validation_margin'] + CONFIG['report_seconds'])

    def can_train(self, next_seconds=5.):
        return self.elapsed() + self.reserve() + next_seconds < CONFIG['total_seconds']


def sources():
    names = ['q15.py', 'q15_backend.py', 'q15_replay.py', 'q15_pooling.py', 'q15_objective.py', 'refresh_q15.py']
    return q14.sources() | {
        f'notebooks/src/{name}': q1.digest_file(ROOT / 'notebooks/src' / name)
        for name in names
    }


def choose_initial_control(q14_result):
    """Resolve the strongest Q14-recorded control, preferring Q14 on exact ties."""
    candidates = {name: q14_result['metrics'][name]
                  for name in ['best_control', 'q13_control', 'q12_control']}
    source = max(candidates, key=lambda name: (candidates[name]['auroc']['value'],
                                              candidates[name]['average_precision']['value']))
    question = {'best_control': 'q14', 'q13_control': 'q13', 'q12_control': 'q12'}[source]
    module = {'q14': q14, 'q13': q13, 'q12': q12}[question]
    if question == 'q12':
        path = module.OUTPUT / 'best_control.pt'
    else:
        history = q1.read_json(module.OUTPUT / 'training_history.json')
        trials = history.get('trials', history)
        require(bool(trials), f'{question.upper()} frozen control history is empty')
        trial = max(trials, key=lambda name: (trials[name]['best_control']['auroc'],
                                              trials[name]['best_control']['average_precision']))
        require(all(np.isclose(trials[trial]['best_control'][key], candidates[source][key]['value'],
                               atol=1e-12, rtol=0.) for key in ['auroc', 'average_precision']),
                'Strongest frozen checkpoint disagrees with completed Q14 metrics')
        path = module.OUTPUT / trial / 'best_control.pt'
    return question, source, path, candidates[source]


def inherited_learning_rate(q14_result):
    history = q1.read_json(q14.OUTPUT / 'training_history.json')
    trials = history.get('trials', history)
    name = q14_result['best_lora_trial']
    require(name in trials, 'Q14 winning adapter trial is missing')
    trial = trials[name]
    require(all(np.isclose(trial['best_lora'][key], q14_result['metrics']['lora'][key]['value'],
                           atol=1e-12, rtol=0.) for key in ['auroc', 'average_precision']),
            'Q14 winning adapter trial disagrees with completed metrics')
    lr = trial['adapter_lr']
    require(isinstance(lr, (int, float)) and np.isfinite(lr) and lr > 0
            and lr in q14.CONFIG['adapter_lrs'], 'Q14 winning trial has an invalid actual adapter learning rate')
    return name, float(lr)


def prepare():
    """Freeze the strongest completed frozen classifier, Q12 caches and inherited cohorts."""
    parent_result = q12.verified_results()
    q14_result = q14.verified_results()
    initialization_question, initialization_source, initialization_path, initial_metrics = choose_initial_control(q14_result)
    inherited_trial, adapter_lr = inherited_learning_rate(q14_result)
    rows, labels, groups, checks = q12.inputs()
    parent = q1.read_json(q12.OUTPUT / 'protocol.json')
    require(len(rows['train']) == CONFIG['training_variants']
            and len(rows['validation']) == CONFIG['validation_variants'],
            'Q15 must retain the exact authorized Q12 subsets')
    require(parent_result['feature_form'] == CONFIG['feature_form'],
            'Q12 selected feature form differs from the Q15 design')
    value = {
        'configuration': CONFIG, 'sources': sources(),
        'q14_identity': q14.verify_protocol(),
        'q14_metrics_sha256': q1.digest_file(q14.OUTPUT / 'metrics.json'),
        'q14_history_sha256': q1.digest_file(q14.OUTPUT / 'training_history.json'),
        'q14_best_control_metrics': q14_result['metrics']['best_control'],
        'initialization_question': initialization_question,
        'initialization_source': initialization_source,
        'initialization_checkpoint': str(initialization_path.relative_to(ROOT)),
        'initialization_checkpoint_sha256': q1.digest_file(initialization_path),
        'initial_control_metrics': initial_metrics,
        'q14_best_lora_trial': inherited_trial, 'adapter_lr': adapter_lr,
        'q12_identity': q12.verify_protocol(),
        'q12_metrics_sha256': q1.digest_file(q12.OUTPUT / 'metrics.json'),
        'q12_head_results_sha256': q1.digest_file(q12.OUTPUT / 'head_results.json'),
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
          f'verified Q12 features and fixed {initialization_source} classifier will be reused.', flush=True)
    q0.details('Q15 configuration, inherited leakage checks and parent provenance', {
        'configuration': CONFIG, 'checks': checks, 'q12_identity': value['q12_identity'],
        'q14_identity': value['q14_identity'],
        'q14_metrics_sha256': value['q14_metrics_sha256'],
        'q14_best_lora_trial': value['q14_best_lora_trial'], 'adapter_lr': value['adapter_lr'],
        'initialization_question': value['initialization_question'],
        'initialization_source': value['initialization_source'],
        'initialization_checkpoint': value['initialization_checkpoint'],
        'initialization_checkpoint_sha256': value['initialization_checkpoint_sha256'],
        'initial_control_metrics': value['initial_control_metrics'],
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
            'Q15 sources or configuration changed; archive before starting another experiment')
    require(value['q12_identity'] == q12.verify_protocol(), 'Q12 source identity changed')
    require(value['q14_identity'] == q14.verify_protocol(), 'Q14 source identity changed')
    q1.verify_file(q14.OUTPUT / 'metrics.json', value['q14_metrics_sha256'])
    q1.verify_file(q14.OUTPUT / 'training_history.json', value['q14_history_sha256'])
    q1.verify_file(q12.OUTPUT / 'metrics.json', value['q12_metrics_sha256'])
    q1.verify_file(q12.OUTPUT / 'head_results.json', value['q12_head_results_sha256'])
    q1.verify_file(q12.OUTPUT / 'feature_manifest.json', value['q12_feature_manifest_sha256'])
    initialization_path = ROOT / value['initialization_checkpoint']
    require(initialization_path.resolve().is_relative_to(ROOT.resolve()),
            'Q15 initialization checkpoint escapes the repository')
    q1.verify_file(initialization_path, value['initialization_checkpoint_sha256'])
    parent_result = q1.read_json(q14.OUTPUT / 'metrics.json')
    question, source, path, metrics = choose_initial_control(parent_result)
    require(question == value['initialization_question'] and source == value['initialization_source'] and path == initialization_path
            and metrics == value['initial_control_metrics'], 'Q15 frozen initialization changed')
    trial, lr = inherited_learning_rate(parent_result)
    require(trial == value['q14_best_lora_trial'] and lr == value['adapter_lr'], 'Q15 inherited learning rate changed')
    parent = q1.read_json(q12.OUTPUT / 'protocol.json')
    for key in ['training_variant_keys', 'validation_variant_keys', 'q1_protocol_sha256', 'vcf_exports']:
        require(value[key] == parent[key], f'Q15 fixed input differs from Q12: {key}')
    return q1.fingerprint(value)


def inputs():
    verify_protocol()
    rows, labels, groups, checks = q12.inputs()
    value = q1.read_json(OUTPUT / 'protocol.json')
    for split, key in [('train', 'training_variant_keys'), ('validation', 'validation_variant_keys')]:
        require(rows[split].variant_key.tolist() == value[key], 'Q15 cohort membership changed')
        require(q1.fingerprint(labels[split].tolist()) == value['labels_sha256'][split],
                'Q15 labels changed')
        require(q1.fingerprint(groups[split].tolist()) == value['components_sha256'][split],
                'Q15 components changed')
    require(not set(groups['train']) & set(groups['validation']), 'Q15 components cross partitions')
    return rows, labels, groups, checks


def run_experiment():
    """Execute training, never substituting completed metrics for an actual run."""
    verify_protocol()
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (OUTPUT / 'metrics.json').exists(),
                'Q15 already has completed results. Use refresh_q15 to archive them and execute a fresh run.')
        path = OUTPUT / 'execution.json'
        if not path.exists():
            q1.write_json(path, {'started_utc': float(os.environ.get('Q15_RUN_STARTED_UTC', time.time()))},
                          frozen=True)
        log = OUTPUT / 'explore.log'
        if log.exists():
            archive = OUTPUT / 'logs'
            archive.mkdir(exist_ok=True)
            log.rename(archive / f'explore-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q15_backend', 'explore']
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
        require(status == 0, 'Q15 exploration failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
    return verified_results()


def verified_results():
    result = q1.read_json(OUTPUT / 'metrics.json')
    require(result.get('status') == 'complete' and result.get('identity') == verify_protocol(),
            'Incomplete or stale Q15 results')
    require(result.get('scope') == 'sampled_validation', 'Q15 evaluation scope mismatch')
    require(result.get('reloaded_predictions_verified') is True
            and result.get('frozen_unchanged') is True, 'Q15 checkpoint or backbone verification failed')
    require(result.get('head_fixed_verified') is True and result.get('scaler_fixed_verified') is True
            and result.get('control_optimizer_steps') == 0, 'Q15 fixed classifier verification failed')
    require(result.get('prefix_replay_verified') is True and result.get('replay_gradient_verified') is True
            and result.get('matched_training_verified') is True, 'Q15 replay or matched-training verification failed')
    require(bool(result.get('artifacts')), 'Q15 result artifact hashes are missing')
    for name, sha in result['artifacts'].items():
        path = OUTPUT / name
        require(path.resolve().is_relative_to(OUTPUT.resolve()), 'Q15 artifact path escapes the result directory')
        q1.verify_file(path, sha)
    required = {'weighted_bce', 'pairwise_logistic', 'lora', 'matched_control', 'best_control', 'strongest_control', 'selected'}
    require(required <= set(result['metrics']), 'Q15 comparison results are incomplete')
    points = {name: {key: value['value'] for key, value in result['metrics'][name].items()}
              for name in required}
    protocol = q1.read_json(OUTPUT / 'protocol.json')
    strongest = promotion_control(points['strongest_control'], protocol)
    require(bool(result['promotion_passed']) == promote(points['lora'], strongest),
            'Q15 promotion result does not match its frozen rule')
    require((result['selected_model'] == 'lora') == result['promotion_passed'],
            'Q15 selected model disagrees with promotion')
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
                axis.plot([row.get('attempted_steps', row.get('steps')) for row in evaluations],
                          [row[branch][key] for row in evaluations],
                          linestyle=style, marker='o', label=f'{name}: {branch}')
    for axis, key, label in zip(axes, ['auroc', 'average_precision'], ['AUROC', 'Average precision']):
        axis.axhline(result['metrics']['strongest_control'][key]['value'], color='black',
                     linestyle=':', label='Strongest fixed control')
        axis.set(xlabel='Matched attempted batches', ylabel=f'Sampled validation {label}')
        axis.legend(fontsize=7)
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    rows = ['| Model | AUROC (95% interval) | AP (95% interval) |',
            '| --- | ---: | ---: |']
    names = {'strongest_control': 'Strongest fixed control', 'weighted_bce': 'Weighted BCE adapters',
             'pairwise_logistic': 'Pairwise logistic adapters', 'selected': 'Selected model'}
    for name, label in names.items():
        values = result['metrics'][name]
        formatted = [f"{values[key]['value']:.3f} [{values[key]['ci95'][0]:.3f}, {values[key]['ci95'][1]:.3f}]"
                     for key in ['auroc', 'average_precision']]
        rows.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    display(Markdown('\n'.join(rows)))
    q0.details('Fixed classifier initialization and integrity', q1.read_json(OUTPUT / 'head_results.json'))
    q0.details('Completed Q15 measurements, integrity checks and provenance', result)


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
        f"The paired arms attempted **{result['attempted_steps']} batches** and covered "
        f"**{result['actual_unique_training_variants']:,} unique training variants**. "
        f"Recorded examples per arm: **{result['examples_per_arm']}**; ranking optimizer skips are retained in the history. "
        f"LoRA minus its matched control: AUROC **{paired['value']:+.4f}** "
        f"(95% component interval **[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]**). "
        'Promotion requires at least **0.005 AUROC** improvement and no AP reduction against the '
        'strongest tested frozen control. The classifier and scaler stayed fixed; head fitting cannot account for an adapter gain. '
        f"The measured experiment took **{result['seconds']/60:.1f} minutes**; "
        f"{'within' if result['within_one_hour'] else 'exceeding'} the one-hour target. "
        'These sampled results do not replace the complete-cohort benchmark. ' + LIMITATIONS
    )
    display(Markdown(message))
