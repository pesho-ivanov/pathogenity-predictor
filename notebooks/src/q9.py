"""Q9: fixed missense inputs and a gated BioNeMo fine-tuning experiment."""

from pathlib import Path
import numpy as np

from . import q0, q1, q2

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'notebooks/results/q9'
PILOT_SHA256 = {
    'clinvar-train-pilot.vcf': 'db7bf9cd1b743e2a9f7045a7053f268b9f568dd9726347af286e224c1b572ea6',
    'clinvar-test-pilot.vcf': '1007a3b5229199203c1174e78d430ad0195b9f33ea258af1829077223e72f4eb',
}
CONFIG = {
    'seed': 42, 'context_bp': 1024,
    'model_repo': 'arcinstitute/savanna_evo2_1b_base',
    'model_revision': '7217626d9f843e1830a5de1f5209c046570b6856',
    'bionemo_revision': 'ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69',
    'trainable': 'Hyena block 23 and linear binary classification head; attention block 24 stays frozen',
    'trainable_block': 23,
    'readiness_optimizer_steps': 8,
    'precision': 'BF16 with FP32 optimizer master weights',
    'fp32_residual_connection': False,
    'microbatch_variants': 1, 'gradient_accumulation': 8,
    'optimizer': 'AdamW', 'backbone_lr': 1e-5, 'head_lr': 1e-4,
    'betas': [.9, .999], 'epsilon': 1e-8, 'scheduler': 'constant',
    'weight_decay': .01, 'clip_grad': 1., 'max_epochs': 5, 'patience': 2,
    'loss': 'binary cross entropy; class weights from training labels only',
    'initialization': 'training-only scaler and fitted frozen logistic head',
    'selection': 'validation AUROC after each epoch including epoch 0; ties prefer earlier',
    'baselines': ['frozen Evo2 logistic regression', 'zero-shot Evo2', 'sequence logistic regression'],
    'baseline_C': [.01, .1, 1., 10.], 'bootstrap_repetitions': 1000,
    'evaluation': 'validation development comparison; no independent final test',
    'comparison_backend': 'BioNeMo for frozen baseline and fine-tuned model; Vortex is diagnostic only',
}


def verify_inputs():
    """Verify the exact approved files and the complete parent split protocol."""
    protocol = q1.verify_protocol()
    if protocol['vcf_exports'] != PILOT_SHA256:
        raise ValueError('Q9 requires the approved 5,000-missense pilot VCF hashes')
    if protocol['config']['context_bp'] != CONFIG['context_bp']:
        raise ValueError('Q9 context differs from the frozen split context')
    full = q1.read_csv(q1.OUTPUT / 'full_cohort_groups.csv.gz')
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz')
    checks = q1.audit_splits(full, pilot, dna)
    checks.update(q1.audit_missense_scope(full, pilot))
    if len(pilot) != 5000 or len(dna) != 5000:
        raise ValueError('Q9 requires exactly 5,000 frozen missense variants')
    labels, counts = {}, {}
    for split in q1.SPLITS:
        keys = pilot.loc[pilot.split.eq(split), 'variant_key'].tolist()
        labels[split] = q1.load_partition_labels(split, keys)
        if set(labels[split]) != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
        counts[split] = len(keys)
    return protocol, pilot, dna, labels, counts, checks


def assert_feature_inputs(values, expected):
    if list(values.columns) != q1.DNA_COLUMNS or not values.equals(expected):
        raise ValueError('Model inputs must be exactly the frozen DNA-only allowlist')


def prepare_inputs():
    protocol, pilot, dna, labels, counts, checks = verify_inputs()
    assert_feature_inputs(dna, dna)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        'status': 'inputs_verified', 'counts': counts, 'checks': checks,
        'vcf_exports': protocol['vcf_exports'],
        'parent_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
        'manifest_sha256': protocol['artifacts']['split_manifest.csv'],
        'sequences_sha256': protocol['artifacts']['sequences.csv.gz'],
        'configuration': CONFIG,
        'limitations': ['Validation is used for selection; no untouched test estimate',
                        'Homology, shared patients and pretraining contamination remain unresolved'],
    }
    q1.write_json(OUTPUT / 'input_checks.json', record)
    print(f"Verified {sum(counts.values()):,} missense variants: "
          f"{counts['train']:,} training / {counts['validation']:,} validation.")
    q0.details('Frozen inputs, split checks and planned training settings', record)
    return record


def source_hashes():
    paths = [ROOT / 'requirements.txt'] + sorted((ROOT / 'notebooks/src').glob('q9*.py'))
    paths += [ROOT / 'notebooks/src' / f'q{i}.py' for i in range(3)]
    return {str(p.relative_to(ROOT)): q1.digest_file(p) for p in paths}


def assert_identity(actual, expected):
    if actual != expected:
        raise ValueError('Q9 artifact identity changed; preserve the previous run before rerunning')


def establish_environment():
    from . import q9_environment as environment
    verify_inputs()
    result = environment.setup()
    checkpoint = environment.checkpoint()
    q0.details('Isolated BioNeMo environment and checkpoint', {
        **result, 'source_sha256': checkpoint['source_sha256'], 'checkpoint_format': checkpoint['format'],
        'conversion_note': checkpoint['reason']})
    return result


def protocol_identity():
    verify_inputs()
    return {'configuration': CONFIG, 'sources': source_hashes(),
            'parent_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
            'checkpoint_manifest_sha256': q1.digest_file(OUTPUT / 'converted_checkpoint.json'),
            'environment_sha256': q1.digest_file(OUTPUT / 'environment/setup.json')}


def compatibility_checks():
    """Run real probes before fitting a classifier or fine-tuning the backbone."""
    from . import q9_environment as environment
    identity = protocol_identity()
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists():
        assert_identity(q1.read_json(protocol_path), identity)
    else:
        q1.write_json(protocol_path, identity)
    cached = OUTPUT / 'readiness.json'
    if cached.exists():
        record = q1.read_json(cached)
        assert_identity(record['identity'], identity)
        for name, digest in record['artifacts'].items():
            q1.verify_file(OUTPUT / name, digest)
    else:
        environment.run_command('parity-nemo', [environment.PYTHON, '-m', 'notebooks.src.q9_parity', 'nemo'])
        environment.run_command('gradient-preflight', [environment.PYTHON, '-m', 'notebooks.src.q9_backend', 'preflight'])
        conversion = q1.read_json(OUTPUT / 'conversion_audit.json')
        smoke = q1.read_json(OUTPUT / 'preflight.json')
        blockers = readiness_blockers(conversion, smoke)
        artifacts = ['conversion_audit.json', 'parity_nemo.npz', 'preflight.json', 'smoke_adapter.pt']
        record = {'status': 'blocked' if blockers else 'passed', 'identity': identity,
                  'comparison': CONFIG['comparison_backend'],
                  'blockers': blockers, 'artifacts': {n: q1.digest_file(OUTPUT / n) for n in artifacts}}
        q1.write_json(cached, record)
    print('Compatibility: ' + record['status'].upper())
    for reason in record['blockers']:
        print('• ' + reason)
    return record


def readiness_blockers(conversion, smoke):
    """Only the producing BioNeMo model determines readiness; Vortex cannot veto it."""
    blockers = []
    if not conversion['passed']:
        blockers.append('Converted BioNeMo weights do not match their source tensors')
    if smoke['status'] != 'passed':
        blockers.append(smoke['blocker'])
    return blockers


def require_ready():
    record = q1.read_json(OUTPUT / 'readiness.json')
    assert_identity(record['identity'], protocol_identity())
    for name, digest in record['artifacts'].items():
        q1.verify_file(OUTPUT / name, digest)
    if record['status'] != 'passed':
        raise RuntimeError('Training blocked: ' + '; '.join(record['blockers']))
    return q1.fingerprint(record['identity'])


def run_experiment():
    from . import q9_environment as environment
    record = q1.read_json(OUTPUT / 'readiness.json')
    assert_identity(record['identity'], protocol_identity())
    if record['status'] != 'passed':
        print('Training and performance evaluation were skipped because compatibility checks failed.')
        return {'status': 'blocked', 'blockers': record['blockers']}
    require_ready()
    environment.run_command('experiment', [environment.PYTHON, '-m', 'notebooks.src.q9_experiment'])
    return q1.read_json(OUTPUT / 'metrics.json')


def show_results():
    import matplotlib.pyplot as plt
    record = q1.read_json(OUTPUT / 'readiness.json')
    smoke = q1.read_json(OUTPUT / 'preflight.json')
    if record['status'] == 'passed' and (OUTPUT / 'metrics.json').exists():
        result = q1.read_json(OUTPUT / 'metrics.json')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        names = list(result['metrics'])
        for ax, metric in zip(axes[:2], ['auroc', 'average_precision']):
            ax.barh(names, [result['metrics'][n][metric] for n in names])
            ax.set_xlim(0, 1)
            ax.set_title('Validation ' + metric.replace('_', ' '))
        history = result['history']
        axes[2].plot([r['epoch'] for r in history], [r['auroc'] for r in history], marker='o')
        axes[2].set(xlabel='Epoch (0 = frozen)', ylabel='Validation AUROC', title='Checkpoint selection')
        fig.tight_layout()
        fig.savefig(OUTPUT / 'performance.png', dpi=150)
        plt.show()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].bar([r['epoch'] for r in history[1:]], [r['seconds']/60 for r in history[1:]])
        axes[0].set(xlabel='Epoch', ylabel='Minutes', title='Training + validation time')
        axes[1].plot([r['epoch'] for r in history], [r['peak_gpu_gib'] for r in history], marker='o')
        axes[1].set(xlabel='Epoch', ylabel='Peak allocated GPU GiB', title='GPU memory')
        fig.tight_layout()
        fig.savefig(OUTPUT / 'compute.png', dpi=150)
        plt.show()
        q0.details('Metrics and paired component-bootstrap intervals', result)
    elif record['status'] != 'passed' and (OUTPUT / 'investigation/manifest.json').exists():
        from . import q9_diagnostics
        q9_diagnostics.show_results()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].bar(['Before updates', 'After updates'],
                    [smoke['initial_fixed_head_score'], smoke['final_fixed_head_score']])
        axes[0].set(ylabel='Synthetic-target logit', title='Backbone probe with a fixed head')
        axes[1].bar(['Selected block', 'Frozen backbone'], [len(smoke['updated_block_parameters']), 0])
        axes[1].set(ylabel='Changed parameter tensors', title=f"Train block {smoke['trainable_block']}")
        fig.tight_layout()
        fig.savefig(OUTPUT / 'readiness.png', dpi=150)
        plt.show()
    print(f"Readiness probes: {smoke['variant_step_seconds']:.2f} s, {smoke['peak_gpu_gib']:.2f} GiB peak allocated GPU memory.")
    q0.details('Checkpoint, numerical and optimizer checks', {
        'status': record['status'], 'blockers': record['blockers'], 'preflight': smoke})


def show_conclusion():
    from IPython.display import Markdown, display
    record = q1.read_json(OUTPUT / 'readiness.json')
    if record['status'] != 'passed':
        text = ('**Predictive improvement remains unmeasured.** Training is blocked by: '
                + '; '.join(record['blockers']) + '. Both comparison models use BioNeMo.')
    elif not (OUTPUT / 'metrics.json').exists():
        text = ('**Predictive improvement remains unmeasured.** Hyena block 23 passes the training-only update, '
                'frozen-weight and reload checks; attention block 24 stays frozen. '
                'The full BioNeMo baseline and fine-tuning comparison has not run yet.')
    else:
        result = q1.read_json(OUTPUT / 'metrics.json')
        delta = result['metrics']['fine_tuned']['auroc'] - result['metrics']['frozen']['auroc']
        text = (f"The selected checkpoint changes validation AUROC by **{delta:+.3f}** versus the frozen BioNeMo "
                f"classifier (epoch {result['selected_epoch']}). Validation guided selection; this is a development "
                'comparison, and its bootstrap intervals do not remove selection bias. No untouched test was evaluated.')
    display(Markdown(text))
