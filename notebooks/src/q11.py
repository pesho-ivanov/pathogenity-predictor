"""Q11: one-hour, partial-epoch Evo2 7B LoRA on the frozen Q1 partitions.

CPU orchestration and reporting; GPU operations run in the pinned BioNeMo venv.
Neither historical Q9/Q10 protocols nor their pilot artifacts are consumed.
"""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from . import q0, q1, q1_full, q9_environment, q10

ROOT = q1.ROOT
OUTPUT = ROOT / 'notebooks/results/q11'
CHECKPOINT = q10.CHECKPOINT
CONVERTED = OUTPUT / 'base_checkpoint_zarr'
CONFIG = {
    **{key: q10.CONFIG[key] for key in ['model_repo', 'model_revision', 'model_sha256',
                                      'model_bytes', 'context_bp', 'hidden_size', 'layers']},
    'seed': 42, 'feature_dimension': 8192, 'source_context_bp': 1024, 'context_bp': 512,
    'lora': {'block': 30, 'rank': 8, 'alpha': 16, 'dropout': 0.,
             'A_init': 'xavier', 'B_init': 'zero',
             'targets': ['decoder.layers.30.mixer.dense_projection', 'decoder.layers.30.mixer.dense']},
    'features': 'canonical paired strand; centered 512-base crop; final block mean positions; [reference, alternate-reference]',
    'normalization': 'separate per-example RMS normalization of reference and difference; epsilon 1e-6; no fitted preprocessing',
    'precision': 'BF16 frozen backbone and LoRA computation; FP32 pooling, classifier, accumulation and AdamW masters',
    'fp8': False, 'fp32_residual_connection': False,
    'microbatch_variants': 32, 'gradient_accumulation': 1,
    'optimizer': 'AdamW', 'adapter_lr': 1e-4, 'head_lr': 1e-4,
    'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': .01,
    'clip_grad': 1., 'scheduler': 'constant', 'max_steps': 768,
    'training_seconds': 1800, 'total_seconds': 3600, 'report_reserve_seconds': 240,
    'validation_time_margin': 1.3, 'reload_variants': 64,
    'loss': 'binary cross entropy with training-only balanced class weights',
    'classifier': 'jointly trained FP32 linear head, PyTorch default seeded initialization; no frozen baseline or hyperparameter search',
    'checkpoint_steps': 32,
    'preflight_steps': 8, 'bootstrap_repetitions': 1000,
    'evaluation': 'partial epoch; complete development validation; no frozen-baseline improvement claim or untouched test',
}
METHODS = {'fine_tuned': 'Evo2 7B base LoRA (block 30, rank 8, partial epoch, 512 bp)'}
LIMITATIONS = ('User-authorized partial-epoch training for a one-hour run; full frozen validation is retained. '
              'No fitted frozen baseline or improvement claim; these are development results. '
              'Pretraining sequence exposure, homology and shared-patient overlap remain unresolved. '
              'The final attention block is numerically inactive in this BF16 configuration; '
              'LoRA targets the preceding Hyena mixer. No clinical validity or independent '
              'final-test performance is established.')


def sources():
    names = ['q0.py', 'q1.py', 'q1_full.py', 'q2.py', 'q9_environment.py', 'q10.py',
             'q11.py', 'q11_backend.py']
    paths = [ROOT / 'notebooks/src' / name for name in names] + [ROOT / 'requirements.txt']
    return {str(path.relative_to(ROOT)): q1.digest_file(path) for path in paths}


def align_inputs(manifest, dna, label_frames):
    """Enforce DNA-only inputs and key-based alignment independent of file order."""
    if list(dna.columns) != q1.DNA_COLUMNS or dna.variant_key.duplicated().any():
        raise ValueError('Expected unique frozen DNA-only variant inputs')
    if manifest.variant_key.duplicated().any() or set(manifest.variant_key) != set(dna.variant_key):
        raise ValueError('DNA membership differs from the full manifest')
    if set(manifest.split) != set(q1.SPLITS):
        raise ValueError('Only training and validation partitions are allowed')
    if set(manifest.loc[manifest.split.eq('train'), 'component']) & set(
            manifest.loc[manifest.split.eq('validation'), 'component']):
        raise ValueError('Components cross partitions')
    labels, indexes = {}, {}
    positions = pd.Series(np.arange(len(dna)), index=dna.variant_key)
    for split in q1.SPLITS:
        keys = manifest.loc[manifest.split.eq(split), 'variant_key']
        frame = label_frames[split]
        if frame.variant_key.duplicated().any() or set(frame.variant_key) != set(keys):
            raise ValueError(f'{split} label membership differs from manifest')
        values = frame.set_index('variant_key').loc[keys, 'label'].to_numpy()
        if set(values) != {0, 1}:
            raise ValueError(f'{split} requires both unambiguous classes')
        labels[split], indexes[split] = values.astype(int), positions.loc[keys].to_numpy()
    return indexes, labels


def verify_inputs():
    protocol = q1_full.verify_protocol()
    if protocol['config']['context_bp'] != CONFIG['source_context_bp']:
        raise ValueError('Q11 context is incompatible with the frozen split')
    expected_vcfs = {p.name for p in q1_full.VCF_FILES.values()}
    if set(protocol['vcf_exports']) != expected_vcfs:
        raise ValueError('Q11 requires both full-cohort VCFs; pilot inputs are forbidden')
    directory = q1_full.OUTPUT
    manifest = q1.read_csv(directory / 'split_manifest.csv')
    dna = q1.read_csv(directory / 'sequences.csv.gz')
    checks = q1_full.audit(q1.read_csv(directory / 'full_cohort_groups.csv.gz'), manifest, dna,
                           q1.read_csv(directory / 'sequence_exclusions.csv'))
    frames = {split: q1_full.read_labels(path, protocol['vcf_exports'][path.name], manifest, split)
              for split, path in q1_full.VCF_FILES.items()}
    indexes, labels = align_inputs(manifest, dna, frames)
    checks = {**checks, 'short_contexts': audit_short_contexts(manifest, dna)}
    return protocol, manifest, dna, indexes, labels, checks


def crop_pair(reference, alternate, length=None):
    """Canonicalize the full pair before cropping, preserving strand invariance."""
    length = CONFIG['context_bp'] if length is None else length
    if len(reference) != len(alternate) or set(reference + alternate) - set('ACGT'):
        raise ValueError('Expected equal-length unambiguous DNA alleles')
    reference, alternate = min((reference, alternate),
                              (q1.reverse_complement(reference), q1.reverse_complement(alternate)))
    changes = [i for i, (r, a) in enumerate(zip(reference, alternate)) if r != a]
    if len(changes) != 1 or length <= 0 or length % 2:
        raise ValueError('Expected a single substitution and positive even context length')
    start = changes[0] - length // 2
    if start < 0 or start + length > len(reference):
        raise ValueError('Crop would extend beyond the frozen context')
    return reference[start:start + length], alternate[start:start + length]


def audit_short_contexts(manifest, dna):
    splits = manifest.set_index('variant_key').split
    contexts = {split: set() for split in q1.SPLITS}
    for row in dna.itertuples():
        for sequence in crop_pair(row.ref_sequence, row.alt_sequence):
            contexts[splits.at[row.variant_key]].add(min(sequence, q1.reverse_complement(sequence)))
    overlap = contexts['train'] & contexts['validation']
    if overlap:
        raise ValueError(f'Shortened allele contexts cross splits: {len(overlap)}')
    return {'context_bp': CONFIG['context_bp'], 'cross_split_identical_or_reverse_complement': 0,
            'coordinates': 'Every crop is contained in its previously audited, non-overlapping 1024-base context.'}


def prepare_inputs():
    if not (q1_full.OUTPUT / 'protocol.json').exists():
        q1_full.prepare()
    protocol, manifest, _, _, labels, checks = verify_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {'configuration': CONFIG, 'sources': sources(),
              'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
              'vcf_exports': protocol['vcf_exports'], 'checks': checks,
              'counts': {s: len(labels[s]) for s in q1.SPLITS}, 'limitations': LIMITATIONS}
    q1.write_json(OUTPUT / 'input_checks.json', record, frozen=True)
    print(f"Verified {record['counts']['train']:,} training / {record['counts']['validation']:,} validation variants.")
    q0.details('Frozen full cohort, leakage audits and settings', record)
    return record


def run_command(action):
    directory = OUTPUT / 'environment'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Q11 process is running; inspect environment logs') from None
        path = directory / f'{action}.log'
        if path.exists():
            archive = directory / 'logs'
            archive.mkdir(exist_ok=True)
            path.rename(archive / f'{action}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q11_backend', action]
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        q1.write_json(directory / f'{action}-command.json', {'args': args, 'cwd': str(ROOT)})
        print(f'{action}: {path.relative_to(ROOT)}', flush=True)
        with path.open('w') as stream:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            try:
                result = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if result:
            tail = '\n'.join(path.read_text(errors='replace').splitlines()[-25:])
            raise RuntimeError(f'Q11 {action} failed ({result}): {path}\n{tail}')


def establish_environment():
    from huggingface_hub import hf_hub_download
    # Budget remaining raw/converted weights plus features and recoverable states.
    required = (0 if CHECKPOINT.exists() else CONFIG['model_bytes'])
    required += (0 if CONVERTED.exists() else CONFIG['model_bytes']) + 3 * 1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError(f'Q11 needs at least {required / 1024**3:.1f} GiB free for remaining weights and run artifacts')
    runtime = q9_environment.setup()
    q1.write_json(OUTPUT / 'environment/runtime.json', runtime, frozen=True)
    path = hf_hub_download(CONFIG['model_repo'], CHECKPOINT.name, revision=CONFIG['model_revision'],
                            local_dir=CHECKPOINT.parent)
    q1.verify_file(path, CONFIG['model_sha256'])
    q1.write_json(OUTPUT / 'checkpoint_source.json', {
        'repo': CONFIG['model_repo'], 'revision': CONFIG['model_revision'],
        'sha256': CONFIG['model_sha256'], 'bytes': CHECKPOINT.stat().st_size}, frozen=True)
    marker = OUTPUT / 'converted_checkpoint.json'
    if not marker.exists():
        if CONVERTED.exists():
            raise RuntimeError('Incomplete Q11 conversion exists; preserve it before retrying')
        run_command('convert')
        if not (CONVERTED / 'weights/metadata.json').exists():
            raise RuntimeError('Q11 conversion produced no complete Zarr checkpoint')
        q1.write_json(marker, {'source_sha256': CONFIG['model_sha256'],
                              'files': {str(p.relative_to(CONVERTED)): q1.digest_file(p)
                                        for p in sorted(CONVERTED.rglob('*')) if p.is_file()}}, frozen=True)
    converted = q1.read_json(marker)
    if converted['source_sha256'] != CONFIG['model_sha256']:
        raise ValueError('Converted model has another source checkpoint')
    for name, digest in converted['files'].items():
        q1.verify_file(CONVERTED / name, digest)
    record = {'configuration': CONFIG, 'sources': sources(),
              'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
              'artifacts': {name: q1.digest_file(OUTPUT / name) for name in
                            ['input_checks.json', 'environment/runtime.json', 'checkpoint_source.json',
                             'converted_checkpoint.json']},
              'upstream': q1.read_json(q9_environment.OUTPUT / 'setup.json')['identity']}
    q1.write_json(OUTPUT / 'protocol.json', record, frozen=True)
    q0.details('Pinned checkpoint and compute environment', runtime)
    return record


def verify_protocol():
    record = q1.read_json(OUTPUT / 'protocol.json')
    if record['configuration'] != CONFIG or record['sources'] != sources():
        raise ValueError('Q11 code or configuration changed; preserve the existing experiment')
    q1.verify_file(q1_full.OUTPUT / 'protocol.json', record['q1_protocol_sha256'])
    for name, digest in record['artifacts'].items():
        q1.verify_file(OUTPUT / name, digest)
    return q1.fingerprint(record)


def run_experiment():
    verify_protocol()
    run_command('run')
    return verified_results()


def verified_results():
    identity = verify_protocol()
    result = q1.read_json(OUTPUT / 'metrics.json')
    if (result['identity'] != identity or result['status'] != 'complete'
            or result.get('training_mode') != 'partial_epoch' or not result.get('reloaded_predictions_verified')):
        raise ValueError('Q11 has no completed partial-epoch result for this protocol')
    for name, digest in result['artifacts'].items():
        q1.verify_file(OUTPUT / name, digest)
    return result


def evaluate(labels, predictions, groups, repetitions=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    repetitions = CONFIG['bootstrap_repetitions'] if repetitions is None else repetitions
    functions = {'auroc': roc_auc_score, 'average_precision': average_precision_score}
    if set(predictions) != set(METHODS) or any(
            len(values) != len(labels) or not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('Complete fine-tuned predictions are required')
    result = {name: {metric: {'value': float(function(labels, values))}
                    for metric, function in functions.items()} for name, values in predictions.items()}
    draws = {name: {metric: [] for metric in functions} for name in predictions}
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(CONFIG['seed'])
    for _ in range(repetitions):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[index])) < 2:
            continue
        for metric, function in functions.items():
            values = {n: float(function(labels[index], p[index])) for n, p in predictions.items()}
            for name, value in values.items():
                draws[name][metric].append(value)
    if len(draws['fine_tuned']['auroc']) < .9 * repetitions:
        raise ValueError('Too few valid whole-component bootstrap draws')
    for name in predictions:
        for metric in functions:
            result[name][metric]['ci95'] = np.quantile(draws[name][metric], [.025, .975]).tolist()
    return result


def export_comparison():
    result = verified_results()
    protocol, manifest, _, _, labels, _ = verify_inputs()
    from .comparison import align_predictions
    frame = pd.read_csv(OUTPUT / 'comparison_predictions.csv')
    validation = manifest.loc[manifest.split.eq('validation'), ['variant_key', 'component']].copy()
    validation['label'] = labels['validation']
    align_predictions(frame, {'validation': validation}, METHODS)
    spec = {'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
            'vcf_exports': protocol['vcf_exports'], 'methods': METHODS, 'sources': sources(),
            'predictions_sha256': q1.digest_file(OUTPUT / 'comparison_predictions.csv'),
            'artifacts': {'metrics.json': q1.digest_file(OUTPUT / 'metrics.json'), **result['artifacts']},
            'require_complete': True, 'limitations': LIMITATIONS, 'runtimes': result['runtimes']}
    q1.write_json(OUTPUT / 'comparison_results.json', spec, frozen=True)


def show_results():
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    result = verified_results()
    frame = pd.read_csv(OUTPUT / 'comparison_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for name, label in [('fine_tuned', 'LoRA · partial epoch')]:
        RocCurveDisplay.from_predictions(frame.label, frame[name], name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(frame.label, frame[name], name=label, ax=axes[1])
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    q0.details('Metrics, component-bootstrap intervals, training coverage and compute', result)


def show_conclusion():
    from IPython.display import Markdown, display
    result = verified_results()
    scores = result['metrics']['fine_tuned']
    display(Markdown(
        f"**Conclusion.** The partial-epoch LoRA run trained on **{result['training_variants_seen']:,}** "
        f"variants ({result['epoch_fraction']:.1%} of the frozen training pool), in "
        f"{result['wall_seconds_to_metrics']/60:.1f} minutes through evaluation. "
        f"On all **{result['validation_variants']:,}** validation variants, AUROC was "
        f"**{scores['auroc']['value']:.3f}** (95% interval "
        f"[{scores['auroc']['ci95'][0]:.3f}, {scores['auroc']['ci95'][1]:.3f}]) and average precision "
        f"was **{scores['average_precision']['value']:.3f}** "
        f"(95% interval [{scores['average_precision']['ci95'][0]:.3f}, "
        f"{scores['average_precision']['ci95'][1]:.3f}]). {LIMITATIONS}"))
