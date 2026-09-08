"""Q16: larger Q14 continuation with full validation inside one four-hour budget."""

import fcntl
import os
import shutil
import subprocess
import time

import numpy as np

from . import q0, q1, q1_full, q11, q12, q14, lora_validation, q9_environment

ROOT = q11.ROOT
OUTPUT = ROOT / 'notebooks/results/q16'
NOTEBOOK = ROOT / 'notebooks/Q16-lora-continuation.ipynb'
CONFIG = {
    'seed': 42, 'training_variants': 46888, 'validation_variants': 2048,
    'full_validation_variants': 17927, 'context_bp': 512, 'feature_dimension': 8194,
    'epochs': 3, 'max_steps': 4398, 'microbatch_variants': 32,
    'adapter_lr': 3e-5, 'head_lr': 0., 'adapter_weight_decay': .01,
    'clip_grad': 1., 'betas': [.9, .999], 'epsilon': 1e-8,
    'warmup_steps': 64, 'minimum_lr_ratio': .1,
    'evaluation_steps': 512, 'checkpoint_steps': 32,
    'reload_variants': 64, 'bootstrap_repetitions': 1000,
    'total_seconds': 14400, 'full_validation_reserve_seconds': 3900,
    'report_seconds': 300,
    'lora': {'blocks': [29, 30], 'rank': 8, 'alpha': 16, 'dropout': 0.},
    'initialization': 'Completed Q14 selected adapters and exact fixed classifier/scaler; fresh AdamW and FP32 masters initialized from deployed BF16 adapters.',
    'loss': 'Balanced BCE with logits; class weights recomputed from all 46,888 training labels. Only adapters update.',
    'selection': 'Monitor the inherited 2,048 development variants. Keep the best newly trained checkpoint by AUROC, then AP, separately. Retain it only for strictly higher sampled AUROC and nondecreasing AP versus the Q14 parent; otherwise retain the parent. Always fully evaluate the best continuation, parent and original frozen control.',
    'confirmation': 'Report full-cohort and paired effects versus Q14 and the original frozen control, plus outside-selection and unseen-selection-component subsets. No selection or refitting after full validation.',
    'numerical_preflight': 'Enforce the existing 2e-4 tolerance for operational batch comparisons, repeatability, permutations and strand invariance. Measure single-variant BF16 sensitivity separately; batch-one equivalence is not claimed.',
}
LIMITATIONS = (
    'Development validation, including repeated model selection on the inherited '
    '2,048-variant sample and earlier evaluations of the full partition. There is '
    'no untouched test set; component-bootstrap intervals do not remove selection '
    'bias. Pretraining exposure, homology and shared-patient overlap remain '
    'unresolved. No clinical validation. The four-hour budget includes preparation '
    'for this request, continuation, full validation and reporting, but excludes '
    'historical Q14/Q12 training and initial model acquisition. Single-variant '
    'BF16 scoring is numerically sensitive; this experiment uses the declared '
    'batch-32 workflow and records its operational batch checks.'
)
require = q12.require
metric = q12.metric
better = q12.better


def deadline():
    return float(q1.read_json(OUTPUT / 'execution.json')['deadline_utc'])


def remaining():
    return deadline() - time.time()


def sources():
    names = ['q16.py', 'q16_backend.py', 'q16_report.py', 'refresh_q16.py']
    return lora_validation.sources('q14') | {
        f'notebooks/src/{name}': q1.digest_file(ROOT / 'notebooks/src' / name)
        for name in names
    }


def _inputs():
    full, manifest, dna, indexes, full_labels, checks = q11.verify_inputs()
    rows = {key: dna.iloc[indexes[key]].reset_index(drop=True) for key in ('train', 'validation')}
    rows['full_validation'] = rows.pop('validation')
    keys = q1.read_json(q14.OUTPUT / 'protocol.json')['validation_variant_keys']
    require(len(keys) == CONFIG['validation_variants'] and len(set(keys)) == len(keys),
            'Selection keys must be the fixed 2,048 unique Q14 variants')
    indexed = rows['full_validation'].set_index('variant_key', drop=False)
    require(set(keys) <= set(indexed.index), 'Q14 selection keys escaped full validation')
    rows['validation'] = indexed.loc[keys].reset_index(drop=True)
    mapping = dict(zip(rows['full_validation'].variant_key, full_labels['validation']))
    labels = {'train': full_labels['train'], 'full_validation': full_labels['validation'],
              'validation': np.asarray([mapping[key] for key in keys], dtype=int)}
    components = manifest.set_index('variant_key').component
    groups = {key: components.loc[frame.variant_key].to_numpy() for key, frame in rows.items()}
    expected = {'train': (46888, 18173), 'validation': (2048, 688), 'full_validation': (17927, 6523)}
    for key, (count, positives) in expected.items():
        require(len(rows[key]) == count and int(labels[key].sum()) == positives,
                f'Frozen {key} membership or labels changed')
        require(not rows[key].variant_key.duplicated().any(), 'Duplicate variant membership')
    require(not set(groups['train']) & set(groups['full_validation']), 'Components cross splits')
    return full, rows, labels, groups, checks


def prepare():
    """Freeze full training, unchanged selection membership, parents and budget."""
    parent = q14.verified_results()
    parent_full = lora_validation.verified_results('q14')
    require(parent['selected_model'] == 'lora', 'Q16 requires the successful Q14 adapter')
    full, rows, labels, groups, checks = _inputs()
    execution = q1.read_json(OUTPUT / 'execution.json')
    require(0 < execution['deadline_utc'] - execution['started_utc'] <= CONFIG['total_seconds'],
            'Invalid four-hour cumulative deadline')
    require(remaining() > CONFIG['full_validation_reserve_seconds'] + CONFIG['report_seconds'],
            'Insufficient time remains for training and full validation')
    checkpoint = q14.OUTPUT / 'selected_model.pt'
    parent_files = ['protocol.json', 'metrics.json', 'selected_model.pt', 'validation_predictions.csv',
                    'full/protocol.json', 'full/metrics.json', 'full/validation_predictions.csv']
    value = {
        'configuration': CONFIG, 'sources': sources(), 'budget': execution,
        'parent_identity': parent['identity'], 'parent_full_identity': parent_full['identity'],
        'parent_artifacts': {str((q14.OUTPUT / name).relative_to(ROOT)): q1.digest_file(q14.OUTPUT / name)
                             for name in parent_files},
        'initialization_checkpoint': str(checkpoint.relative_to(ROOT)),
        'initialization_checkpoint_sha256': q1.digest_file(checkpoint),
        'q1_identity': q1.fingerprint(full),
        'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
        'vcf_exports': full['vcf_exports'],
        'training_variant_keys': rows['train'].variant_key.tolist(),
        'validation_variant_keys': rows['validation'].variant_key.tolist(),
        'full_validation_variant_keys': rows['full_validation'].variant_key.tolist(),
        'labels_sha256': {key: q1.fingerprint(values.tolist()) for key, values in labels.items()},
        'components_sha256': {key: q1.fingerprint(values.tolist()) for key, values in groups.items()},
        'counts': {key: {'total': len(values), 'pathogenic': int(values.sum()),
                         'benign': int(len(values) - values.sum())} for key, values in labels.items()},
        'limitations': LIMITATIONS,
    }
    q1.write_json(OUTPUT / 'protocol.json', value, frozen=True)
    q1.write_json(OUTPUT / 'input_checks.json', checks, frozen=True)
    for relative, checksum in value['sources'].items():
        source, target = ROOT / relative, OUTPUT / 'source_snapshot' / relative
        q1.verify_file(source, checksum)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        q1.verify_file(target, checksum)
    print('Frozen all 46,888 training variants, the existing 2,048 selection variants, '
          'and all 17,927 full-validation variants.', flush=True)
    q0.details('Q16 protocol, counts and leakage checks', {
        'configuration': CONFIG, 'counts': value['counts'], 'budget': execution,
        'checks': checks, 'parent_identity': parent['identity'], 'limitations': LIMITATIONS})
    return value


def verify_protocol():
    value = q1.read_json(OUTPUT / 'protocol.json')
    require(value['configuration'] == CONFIG and value['sources'] == sources(),
            'Q16 sources or settings changed; preserve this attempt before starting a new one')
    require(value['budget'] == q1.read_json(OUTPUT / 'execution.json'), 'Q16 deadline changed')
    require(value['parent_identity'] == q14.verify_protocol(), 'Q14 parent identity changed')
    require(value['parent_full_identity'] == lora_validation.verify_protocol('q14'),
            'Q14 full-validation identity changed')
    for relative, checksum in value['parent_artifacts'].items():
        require((ROOT / relative).resolve().is_relative_to(ROOT.resolve()), 'Invalid parent path')
        q1.verify_file(ROOT / relative, checksum)
    q1.verify_file(q1_full.OUTPUT / 'protocol.json', value['q1_protocol_sha256'])
    full = q1_full.verify_protocol()
    require(value['q1_identity'] == q1.fingerprint(full) and value['vcf_exports'] == full['vcf_exports'],
            'Frozen full-cohort identity changed')
    require(value['validation_variant_keys'] == q1.read_json(q14.OUTPUT / 'protocol.json')['validation_variant_keys'],
            'Q16 changed the inherited selection membership')
    return q1.fingerprint(value)


def inputs():
    verify_protocol()
    _, rows, labels, groups, checks = _inputs()
    value = q1.read_json(OUTPUT / 'protocol.json')
    for split, key in [('train', 'training_variant_keys'), ('validation', 'validation_variant_keys'),
                       ('full_validation', 'full_validation_variant_keys')]:
        require(rows[split].variant_key.tolist() == value[key], f'{split} membership changed')
        require(q1.fingerprint(labels[split].tolist()) == value['labels_sha256'][split], 'Labels changed')
        require(q1.fingerprint(groups[split].tolist()) == value['components_sha256'][split], 'Components changed')
    return rows, labels, groups, checks


def run_experiment():
    """Execute continuation and full inference; never replace them with old metrics."""
    verify_protocol()
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (OUTPUT / 'metrics.json').exists(), 'Q16 already completed; archive before a new run')
        log = OUTPUT / 'run.log'
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        args = [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q16_backend']
        if os.environ.get('Q16_RESUME') == '1':
            args.append('--resume')
        print(f'Continuation and full validation: {log.relative_to(ROOT)}', flush=True)
        with log.open('a') as stream:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            try:
                status = child.wait(timeout=max(1., remaining()))
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        require(status == 0, 'Q16 failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
    return verified_results()


def verified_results():
    result = q1.read_json(OUTPUT / 'metrics.json')
    require(result.get('status') == 'complete' and result.get('identity') == verify_protocol(),
            'Incomplete or stale Q16 results')
    require(result.get('scope') == 'full_validation', 'Q16 must complete full validation')
    require(result.get('validation_variants') == CONFIG['full_validation_variants'],
            'Q16 validation cohort is incomplete')
    require({'continuation', 'q14_parent', 'frozen_control', 'selected'} <= set(result['metrics']),
            'Q16 is missing a registered comparison')
    for key in ('reloaded_predictions_verified', 'frozen_unchanged', 'head_fixed_verified', 'scaler_fixed_verified'):
        require(result.get(key) is True, f'Q16 integrity check failed: {key}')
    require(bool(result.get('artifacts')), 'Q16 has no artifact checksums')
    for name, checksum in result['artifacts'].items():
        require((OUTPUT / name).resolve().is_relative_to(OUTPUT.resolve()), 'Invalid artifact path')
        q1.verify_file(OUTPUT / name, checksum)
    return result
