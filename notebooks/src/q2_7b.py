"""Q2: frozen Evo2 7B zero-shot scores on the full Q1 validation partition.

Use the same Vortex/FP8 likelihood definition as the preserved 1B experiment.
No classifier, scaling, threshold, score direction or hyperparameter is fitted.
The README export uses the complete current validation cohort, without mixing
predictions or labels from the archived September experiments.
"""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import q0, q1, q1_full, q2

OUTPUT = q2.OUTPUT / '7b'
MODEL_NAME = 'Evo2 7B base zero-shot (Vortex, FP8)'
CONFIG = {
    'model_repo': 'arcinstitute/evo2_7b_base',
    'model_revision': '074097e9dc788e8bfe045d6495b9f6153a7c6bfc',
    'model_sha256': 'd8a0e775a5d849921b8725837c6a3cbc71fa15e712f4189a2ed52ef955aad29b',
    'model_bytes': 13006429947, 'evo2_commit': q2.EVO2_COMMIT,
    'context_bp': 1024, 'seed': 42, 'batch_sequences': 1, 'cache_batch_variants': 100,
    'precision': 'BF16 weights, upstream frozen FP8 input projections, FP32 log_softmax',
    'inference': 'eval + inference_mode; no opt-in kernels; independent sequences',
    'score': 'negative (alternate-reference) mean next-token log likelihood, mean strands, no BOS',
    'strand_order': 'canonical sequence then its reverse complement, independently for each allele',
    'fitting': 'None; no learned preprocessing, classifier, threshold or validation selection',
    'evaluation': 'All current full Q1 validation variants; complete matching cohort for the README',
    'bootstrap_repetitions': 1000,
}
LIMITATIONS = ('Zero-shot inference on the complete current full validation cohort. '
              'No parameters or thresholds fitted. Archived September results are not mixed with this run. Pretraining, homology and annotation overlap '
              'remain unresolved; validation is development data, not an untouched final test.')


def sources():
    paths = [Path(__file__), Path(q0.__file__), Path(q1.__file__), Path(q1_full.__file__),
             Path(q2.__file__), q1.ROOT / 'requirements.txt']
    return {str(p.relative_to(q1.ROOT)): q1.digest_file(p) for p in paths}


def identity():
    return {'config': CONFIG, 'sources': sources(), 'environment': q2.environment(),
            'full_q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json')}


def prepare():
    full, dna = q1_full.prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = identity()
    q1.write_json(OUTPUT / 'protocol.json', protocol, frozen=True)
    rows = full.loc[full.split.eq('validation')]
    inputs = dna.set_index('variant_key').loc[rows.variant_key].reset_index()
    assert_dna(inputs, rows.variant_key.tolist())
    print(f'{MODEL_NAME}: {len(inputs):,} full validation variants; no fitting or selection.')
    q0.details('Pinned 7B model, full inputs and scoring rules', protocol)
    return inputs


def verify_protocol():
    q1_full.verify_protocol()
    recorded = q1.read_json(OUTPUT / 'protocol.json')
    if recorded != identity():
        raise ValueError('Frozen 7B experiment changed; preserve the recorded run')
    return q1.fingerprint(recorded)


def assert_dna(inputs, keys):
    if list(inputs.columns) != q1.DNA_COLUMNS or inputs.variant_key.tolist() != list(keys):
        raise ValueError('7B inputs must contain only the ordered frozen DNA allowlist')
    for row in inputs.itertuples(index=False):
        if (len(row.ref_sequence) != CONFIG['context_bp'] or len(row.alt_sequence) != CONFIG['context_bp']
                or set(row.ref_sequence + row.alt_sequence) - set('ACGT')
                or [i for i, (a, b) in enumerate(zip(row.ref_sequence, row.alt_sequence)) if a != b] != [512]):
            raise ValueError('Invalid reference/alternate DNA context')


def validation_inputs():
    full = q1.read_csv(q1_full.OUTPUT / 'split_manifest.csv')
    rows = full.loc[full.split.eq('validation')].reset_index(drop=True)
    dna = q1.read_csv(q1_full.OUTPUT / 'sequences.csv.gz').set_index('variant_key').loc[rows.variant_key].reset_index()
    assert_dna(dna, rows.variant_key)
    return rows, dna


def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    checkout = q1.ROOT / 'evo2'
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != CONFIG['evo2_commit']:
        raise ValueError('Evo2 submodule revision changed')
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--', 'evo2'], check=True)
    sys.path.insert(0, str(checkout))
    import evo2
    if Path(evo2.__file__).resolve().parent != checkout / 'evo2':
        raise ValueError('Evo2 was imported from another checkout')
    checkpoint = hf_hub_download(CONFIG['model_repo'], 'evo2_7b_base.pt', revision=CONFIG['model_revision'])
    q1.verify_file(checkpoint, CONFIG['model_sha256'])
    if Path(checkpoint).stat().st_size != CONFIG['model_bytes'] or not torch.cuda.is_available():
        raise ValueError('Expected the pinned 7B checkpoint and a CUDA GPU')
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with (OUTPUT / 'model_load.log').open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
        model = evo2.Evo2('evo2_7b_base', local_path=checkpoint, use_kernels=False)
    model.model.eval().requires_grad_(False)
    return model


def mean_likelihood(model, sequence):
    import torch
    if len(sequence) != CONFIG['context_bp'] or set(sequence) - set('ACGT'):
        raise ValueError('Expected exactly 1,024 A/C/G/T bases')
    tokens = torch.tensor([model.tokenizer.tokenize(sequence)], device='cuda', dtype=torch.long)
    with torch.inference_mode():
        raw, _ = model(tokens)
        logits = raw[0] if isinstance(raw, tuple) else raw
        value = torch.log_softmax(logits.float(), dim=-1)[:, :-1].gather(2, tokens[:, 1:, None]).mean().item()
    if not np.isfinite(value):
        raise ValueError('Nonfinite 7B likelihood')
    return value


def strand_average(model, sequence, memo):
    canonical = min(sequence, q1.reverse_complement(sequence))
    key = q1.context_hash(canonical)
    if key not in memo:
        memo[key] = (mean_likelihood(model, canonical) + mean_likelihood(model, q1.reverse_complement(canonical))) / 2
    return memo[key]


def preflight(model, experiment):
    import torch
    full = q1.read_csv(q1_full.OUTPUT / 'split_manifest.csv')
    keys = full.loc[full.split.eq('train'), 'variant_key'].iloc[:2]
    train = q1.read_csv(q1_full.OUTPUT / 'sequences.csv.gz').set_index('variant_key').loc[keys]
    scales = q2.fp8_scales(model)
    a, b = train.iloc[0].ref_sequence, train.iloc[1].ref_sequence
    first = mean_likelihood(model, a)
    start = time.monotonic()
    mean_likelihood(model, b)
    repeated = mean_likelihood(model, a)
    q2.assert_scales_unchanged(model, scales)
    if first != repeated:
        raise AssertionError('7B A/B/A repeatability failed')
    forward = strand_average(model, a, {})
    reverse = strand_average(model, q1.reverse_complement(a), {})
    q2.assert_scales_unchanged(model, scales)
    if forward != reverse or any(p.requires_grad for p in model.model.parameters()):
        raise AssertionError('Strand invariance or frozen-weight check failed')
    record = {'identity': experiment, 'A_B_A_exact': True, 'strand_invariance_exact': True,
              'fp8_scales_unchanged': True, 'fp8_scale_tensors': len(scales),
              'probe_split': 'train', 'probe_keys': keys.tolist(), 'backbone_frozen': True,
              'seconds_six_sequences': time.monotonic() - start,
              'gpu': torch.cuda.get_device_name(), 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3}
    q1.write_json(OUTPUT / 'preflight.json', record)
    return scales


def read_batch(path, record, keys, experiment):
    q1.verify_file(path, record['sha256'])
    if record['identity'] != experiment:
        raise ValueError('7B batch identity changed')
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    if values['keys'].tolist() != list(keys) or str(values['identity']) != experiment:
        raise ValueError('7B batch membership or order changed')
    for name in ['reference_ll', 'alternate_ll', 'zero_shot']:
        if values[name].shape != (len(keys),) or not np.isfinite(values[name]).all():
            raise ValueError('Invalid 7B batch scores')
    if not np.array_equal(values['zero_shot'], -(values['alternate_ll'] - values['reference_ll'])):
        raise ValueError('7B score differs from the frozen likelihood definition')
    return values


def load_scores():
    experiment = verify_protocol()
    manifest, dna = validation_inputs()
    record = q1.read_json(OUTPUT / 'score_manifest.json')
    if record['identity'] != experiment:
        raise ValueError('7B score manifest is stale')
    q1.verify_file(OUTPUT / 'preflight.json', record['preflight_sha256'])
    preflight_record = q1.read_json(OUTPUT / 'preflight.json')
    if preflight_record['identity'] != experiment or not all(preflight_record[key] for key in [
            'A_B_A_exact', 'strand_invariance_exact', 'fp8_scales_unchanged', 'backbone_frozen']):
        raise ValueError('7B inference checks differ from this experiment')
    size = CONFIG['cache_batch_variants']
    names = [f'scores/{offset:05d}.npz' for offset in range(0, len(dna), size)]
    if set(record['batches']) != set(names):
        raise ValueError('Full 7B score coverage is incomplete')
    batches = [read_batch(OUTPUT / name, record['batches'][name], dna.variant_key.iloc[i*size:(i+1)*size], experiment)
               for i, name in enumerate(names)]
    scores = np.concatenate([batch['zero_shot'] for batch in batches])
    return manifest, scores


def score_validation():
    import torch
    experiment = verify_protocol()
    _, dna = validation_inputs()
    if (OUTPUT / 'score_manifest.json').exists():
        load_scores()
        print('Verified all completed full-validation 7B scores.', flush=True)
        return
    if (OUTPUT / 'metrics.json').exists():
        raise ValueError('Restore the missing score manifest of the evaluated 7B run before continuing')
    (OUTPUT / 'scores').mkdir(exist_ok=True)
    model = load_model()
    scales = preflight(model, experiment)
    torch.cuda.reset_peak_memory_stats()
    start, memo, batches = time.monotonic(), {}, {}
    size = CONFIG['cache_batch_variants']
    for offset in range(0, len(dna), size):
        rows = dna.iloc[offset:offset+size]
        name = f'scores/{offset:05d}.npz'
        path = OUTPUT / name
        metadata = path.with_suffix('.json')
        if metadata.exists():
            record = q1.read_json(metadata)
            read_batch(path, record, rows.variant_key, experiment)
        else:
            started = time.monotonic()
            reference = np.array([strand_average(model, row.ref_sequence, memo) for row in rows.itertuples()])
            alternate = np.array([strand_average(model, row.alt_sequence, memo) for row in rows.itertuples()])
            q2.assert_scales_unchanged(model, scales)
            q2.save_npz(path, keys=rows.variant_key.to_numpy(dtype=str), identity=np.array(experiment),
                        reference_ll=reference, alternate_ll=alternate, zero_shot=-(alternate-reference))
            record = {'identity': experiment, 'sha256': q1.digest_file(path),
                      'seconds': time.monotonic() - started, 'variants': len(rows)}
            q1.write_json(metadata, record)
        batches[name] = record
        print(f'{min(offset+size,len(dna)):,}/{len(dna):,} full validation variants cached; {(time.monotonic()-start)/60:.1f} min this run.', flush=True)
    q1.write_json(OUTPUT / 'score_manifest.json', {
        'identity': experiment, 'batches': batches, 'variants': len(dna),
        'batch_seconds': sum(record['seconds'] for record in batches.values()),
        'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3,
        'preflight_sha256': q1.digest_file(OUTPUT / 'preflight.json')}, frozen=True)


def run():
    verify_protocol()
    if (OUTPUT / 'score_manifest.json').exists():
        load_scores()
        print('Verified completed 7B full-validation inference.')
        return
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = OUTPUT / 'inference.log'
        if log.exists():
            log.rename(OUTPUT / f'inference-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
        print('7B full-validation inference; resumable progress: results/q2/7b/inference.log', flush=True)
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, '-m', 'notebooks.src.q2_7b', 'score'], cwd=q1.ROOT,
                env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'), stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
        if result.returncode:
            raise RuntimeError(f'7B inference failed: {log}\n' + '\n'.join(log.read_text().splitlines()[-20:]))
    load_scores()
    print('Completed and verified all 7B full-validation scores.')


def metric_summary(table, columns, repetitions=1000):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, groups = table.label.to_numpy(), table.component.to_numpy()
    functions = {'auroc': roc_auc_score, 'average_precision': average_precision_score}
    predictions = {column: table[column].to_numpy() for column in columns}
    if set(labels) != {0, 1} or not all(np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Finite scores and both classes are required')
    result = {'variants': len(table), 'pathogenic': int(labels.sum()), 'prevalence': float(labels.mean()),
              'components': len(np.unique(groups)), 'metrics': {}, 'bootstrap_repetitions': repetitions, 'seed': 42}
    draws = {name: {metric: [] for metric in functions} for name in columns}
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(42)
    for _ in range(repetitions):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[index])) < 2:
            continue
        for name, scores in predictions.items():
            for metric, function in functions.items():
                draws[name][metric].append(float(function(labels[index], scores[index])))
    for name, scores in predictions.items():
        if len(draws[name]['auroc']) < repetitions*.9:
            raise ValueError('Too few valid component bootstrap resamples')
        result['metrics'][name] = {metric: {'value': float(function(labels, scores)),
            'ci95': np.quantile(draws[name][metric], [.025, .975]).tolist()} for metric, function in functions.items()}
    return result


def evaluate():
    experiment = verify_protocol()
    path = OUTPUT / 'metrics.json'
    if path.exists():
        result = q1.read_json(path)
        if result['identity'] != experiment:
            raise ValueError('7B evaluation identity changed')
        for name, checksum in result['artifacts'].items():
            q1.verify_file(OUTPUT / name, checksum)
    else:
        manifest, scores = load_scores()
        labels = q1_full.load_partition_labels('validation', manifest.variant_key)
        table = manifest[['variant_key', 'component']].assign(label=labels, zero_shot_7b=scores)
        table.to_csv(OUTPUT / 'validation_predictions.csv', index=False)
        result = {'identity': experiment,
                  'full': metric_summary(table, ['zero_shot_7b'], CONFIG['bootstrap_repetitions']),
                  'clinvar_date': q1.CLINVAR_DATE,
                  'artifacts': {name: q1.digest_file(OUTPUT / name) for name in [
                      'validation_predictions.csv', 'score_manifest.json']},
                  'limitations': LIMITATIONS}
        q1.write_json(path, result, frozen=True)
    table = q1.read_csv(OUTPUT / 'validation_predictions.csv')
    table.to_csv(q2.OUTPUT / 'comparison_predictions.csv', index=False)
    q1.write_json(q2.OUTPUT / 'comparison_results.json', {
        'q1_protocol_sha256': q1.digest_file(q1_full.OUTPUT / 'protocol.json'),
        'predictions_sha256': q1.digest_file(q2.OUTPUT / 'comparison_predictions.csv'),
        'methods': {'zero_shot_7b': MODEL_NAME}, 'limitations': LIMITATIONS, 'sources': sources(),
        'artifacts': {str(p.relative_to(q2.OUTPUT)): q1.digest_file(p) for p in [
            OUTPUT / 'protocol.json', OUTPUT / 'metrics.json', OUTPUT / 'validation_predictions.csv']},
    }, frozen=True)
    return result


def show_results(result):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from IPython.display import Markdown, display
    table = q1.read_csv(OUTPUT / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    fpr, tpr, _ = roc_curve(table.label, table.zero_shot_7b)
    precision, recall, _ = precision_recall_curve(table.label, table.zero_shot_7b)
    axes[0].plot(fpr, tpr, label='Evo2 7B zero-shot')
    axes[0].plot([0, 1], [0, 1], ':', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Full validation ROC')
    axes[1].plot(recall, precision, label='Evo2 7B zero-shot')
    axes[1].axhline(result['full']['prevalence'], ls=':', color='gray', label='Pathogenic fraction')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Full validation precision–recall')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    plt.close(fig)
    measured = result['full']['metrics']['zero_shot_7b']
    values = []
    for metric in ['auroc', 'average_precision']:
        m = measured[metric]
        values.append(f"{metric.replace('_', ' ')} **{m['value']:.3f}** [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]")
    display(Markdown(f"**7B · {result['clinvar_date']} · {result['full']['variants']:,} validation variants**: " + '; '.join(values) + '.'))
    q0.details('Full metrics, component-bootstrap intervals and limitations', result)


def show_conclusion(result):
    from IPython.display import Markdown, display
    full = result['full']['metrics']['zero_shot_7b']
    display(Markdown(
        f"**Evo2 7B zero-shot** reaches AUROC **{full['auroc']['value']:.3f}** "
        f"and average precision **{full['average_precision']['value']:.3f}** on "
        f"**{result['full']['variants']:,}** full ClinVar {result['clinvar_date']} validation variants. "
        'This zero-shot extension does not establish an advantage for a fitted classifier; '
        'the earlier 1B classifier comparison belongs to the archived September pilot. '
        'These are development results with unresolved pretraining and homology overlap.'))


if __name__ == '__main__':
    if sys.argv[1:] != ['score']:
        raise SystemExit('Use the Q2 notebook, or: python -m notebooks.src.q2_7b score')
    score_validation()
