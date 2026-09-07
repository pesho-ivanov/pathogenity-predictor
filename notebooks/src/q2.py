"""Q2: logistic regression on frozen Evo2 features versus zero-shot scoring.

Q1 owns data preparation and frozen split manifests. Fitting uses training data;
validation selects C and compares models. There is no separate final test stage.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import io
import itertools
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import q0, q1
from .q1 import (ROOT, SEED, CONTEXT, SPLITS, DNA_COLUMNS, BASES,
                 digest_file, fingerprint, read_json, write_json, verify_file,
                 read_csv, reverse_complement, context_hash)

OUTPUT = ROOT / 'notebooks/results/q2'
MODEL_REPO = 'arcinstitute/evo2_1b_base'
MODEL_REVISION = '2279e1df422c991037470302360edd40d0d2ea1e'
MODEL_SHA256 = '8ffba7d0e6445a8f2c92d9ff1c4e772c7f73ca9179f5e0c699b8b0ca1b966f64'
EVO2_COMMIT = '53f195997257c56c00e5ef8d33a54f5baad143a6'
LAYER = 'blocks.24'
HIDDEN = 1920
CS = [0.01, 0.1, 1.0, 10.0]
TRIPLETS = [''.join(p) for p in itertools.product(BASES, repeat=3)]
FEATURE_NAMES = ([f'ref_{b}' for b in BASES] + [f'alt_{b}' for b in BASES]
                 + [f'ref_triplet_{s}' for s in TRIPLETS] + ['ref_gc'])


def protocol_config():
    config = q1.protocol_config()
    config.update({
        'checkpoint_repo': MODEL_REPO, 'checkpoint_revision': MODEL_REVISION, 'checkpoint_sha256': MODEL_SHA256,
        'evo2_commit': EVO2_COMMIT, 'layer': LAYER, 'evo_feature_dimension': HIDDEN * 2,
        'embedding': 'mean positions then mean strands; concat(reference, alternate-reference)',
        'zero_shot': 'negative (alt-ref) mean next-token log likelihood, mean strands, no BOS, FP32 log_softmax',
        'inference': 'eval + inference_mode, upstream frozen FP8 scales, BF16 weights, batch 1, no opt-in kernels',
        'baseline_features': FEATURE_NAMES, 'baseline_strands': 'mean forward and reverse complement (centers 512, 511)',
        'classifiers': 'train-only StandardScaler and L2 balanced LogisticRegression, lbfgs, max_iter=3000, tol=1e-5',
        'C_grid': CS, 'selection': 'validation AUROC; ties use smaller C',
        'metrics': ['AUROC (primary)', 'average precision (secondary)'],
        'uncertainty': '1000 paired component bootstrap resamples, seed 42, percentile 95% CI',
        'status': 'Validation comparison used for model selection; no final test evaluation',
        'limitations': ['Validation selects C, so its performance is selection-biased; bootstrap intervals do not correct that bias',
                        'Validation includes previously evaluated variants; no untouched final holdout is claimed',
                        'ClinVar original acquisition chain is unverified; selection/ascertainment bias remains',
                        'Evo2 pretraining can include human reference DNA; exact checkpoint training membership and label contamination unverified',
                        'homologs, distant regulatory relationships, and shared patients are not identified by these checks',
                        'Clinical annotation circularity remains possible even though annotations are excluded from features'],
    })
    return config


def verify_protocol():
    split_protocol = q1.verify_protocol()
    protocol = read_json(OUTPUT / 'protocol.json')
    # Interpretation text may be corrected; scientific settings and data cannot change.
    settings = lambda config: {k: v for k, v in config.items() if k not in {'status', 'limitations'}}
    if settings(protocol['config']) != settings(protocol_config()):
        raise RuntimeError('Frozen model or split configuration changed')
    if (protocol['artifacts'] != split_protocol['artifacts']
            or protocol['vcf_exports'] != split_protocol['vcf_exports']):
        raise RuntimeError('Q2 does not reference the exact frozen Q1 inputs')
    return protocol


def consumer_identity():
    """The code and data currently consuming the frozen computation."""
    return {'protocol_sha256': digest_file(OUTPUT / 'protocol.json'),
            'implementation_sha256': digest_file(__file__),
            'split_implementation_sha256': digest_file(q1.__file__),
            'split_protocol_sha256': digest_file(q1.OUTPUT / 'protocol.json'),
            'environment': environment()}


def experiment_identity():
    """Bind saved features and validation results to the exact code and inputs."""
    return consumer_identity()


def load_inputs():
    """Consume Q1's frozen data; prepare it on CPU if the notebook runs standalone."""
    pilot, sequences = q1.prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'protocol.json'
    if not path.exists():
        split_protocol = q1.verify_protocol()
        write_json(path, {'config': protocol_config(), 'input': split_protocol['input'],
                         'artifacts': split_protocol['artifacts'],
                         'vcf_exports': split_protocol['vcf_exports']}, frozen=True)
    verify_protocol()
    assert_evaluation_identity(experiment_identity())
    print(f"Loaded {len(pilot):,} variants from Q1's frozen manifests; model outputs: results/q2/.")
    return sequences


def settings():
    print('Q1 frozen splits · 1,024 bases · Evo2 1B · validation selects C · seed 42')
    q0.details('Model, features, selection rules and limitations', protocol_config())


def load_model():
    """Use the pinned submodule and checkpoint, with upstream FP8 inference."""
    import torch
    from huggingface_hub import hf_hub_download

    commit = subprocess.check_output(['git', '-C', str(ROOT / 'evo2'), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EVO2_COMMIT:
        raise RuntimeError('Evo2 submodule revision differs from the protocol')
    # Notebook output edits in the submodule are irrelevant; model code edits are not.
    subprocess.run(['git', '-C', str(ROOT / 'evo2'), 'diff', '--exit-code', 'HEAD', '--', 'evo2'], check=True)
    sys.path.insert(0, str(ROOT / 'evo2'))
    from evo2 import Evo2
    import evo2
    if Path(evo2.__file__).resolve().parent != ROOT / 'evo2/evo2':
        raise RuntimeError('Restart kernel: Evo2 was imported from a different source')
    checkpoint = hf_hub_download(MODEL_REPO, 'evo2_1b_base.pt', revision=MODEL_REVISION)
    verify_file(checkpoint, MODEL_SHA256)
    if not torch.cuda.is_available():
        raise RuntimeError('Q2 needs a CUDA GPU and the Evo2/Transformer Engine software stack')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with redirect_stdout(io.StringIO()) as log:
        model = Evo2('evo2_1b_base', local_path=checkpoint, use_kernels=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'model_load.log').write_text(log.getvalue())
    model.model.eval()
    model.model.requires_grad_(False)
    return model



def encode_sequence(model, sequence):
    """One independent pass; FP8 scales do not update under inference_mode."""
    import torch
    tokens = torch.tensor([model.tokenizer.tokenize(sequence)], dtype=torch.long, device='cuda')
    with torch.inference_mode():
        raw_logits, embeddings = model(tokens, return_embeddings=True, layer_names=[LAYER])
        logits = raw_logits[0] if isinstance(raw_logits, tuple) else raw_logits
        pooled = embeddings[LAYER].float().mean(dim=1)[0].cpu().numpy()
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        likelihood = log_probs[:, :-1].gather(2, tokens[:, 1:, None]).mean().item()
    if pooled.shape != (HIDDEN,) or not np.isfinite(pooled).all() or not np.isfinite(likelihood):
        raise ValueError('Invalid Evo2 representation or log likelihood')
    return pooled, likelihood



def fp8_scales(model):
    """Read only scales, not amax observation buffers, which inference never applies."""
    return {name: module.fp8_meta['scaling_fwd'].scale.detach().clone()
            for name, module in model.model.named_modules()
            if hasattr(module, 'fp8_meta') and 'scaling_fwd' in module.fp8_meta}



def assert_scales_unchanged(model, expected):
    import torch
    current = fp8_scales(model)
    if not expected or current.keys() != expected.keys() or any(
            not torch.equal(current[name], scale) for name, scale in expected.items()):
        raise AssertionError('FP8 scaling changed across sequences; extraction stopped')



def benchmark(model):
    """Synthetic A/B/A checks precede all real sequences and never read labels."""
    import torch
    rng = np.random.default_rng(SEED)
    a, b = [''.join(rng.choice(list(BASES), CONTEXT)) for _ in range(2)]
    before = fp8_scales(model)
    first = encode_sequence(model, a)  # initializes kernels, but must preserve checkpoint scales
    assert_scales_unchanged(model, before)
    torch.cuda.synchronize()
    start = time.monotonic()
    encode_sequence(model, b)
    repeated = encode_sequence(model, a)
    torch.cuda.synchronize()
    seconds = (time.monotonic() - start) / 2
    assert_scales_unchanged(model, before)
    if not np.array_equal(first[0], repeated[0]) or first[1] != repeated[1]:
        raise AssertionError('Frozen encoder is order-dependent on the synthetic A/B/A check')
    result = {'seconds_per_sequence': seconds, 'estimated_5000_variant_minutes': seconds * 20000 / 60,
              'fp8_scale_tensors_checked': len(before), 'synthetic_order_independence': True,
              'gpu': torch.cuda.get_device_name(), 'batch_size': 1,
              'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3}
    write_json(OUTPUT / 'benchmark.json', result)
    return result



def sequence_baseline(reference, alternate):
    """73 fixed DNA-only features, strand averaged; no vocabulary is learned."""
    def one_strand(ref, alt, center):
        if len(ref) != CONTEXT or len(alt) != CONTEXT or set(ref + alt) - set(BASES):
            raise ValueError('Invalid baseline DNA')
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        values[BASES.index(ref[center])] = 1
        values[4 + BASES.index(alt[center])] = 1
        values[8 + TRIPLETS.index(ref[center - 1:center + 2])] = 1
        values[-1] = (ref.count('G') + ref.count('C')) / len(ref)
        return values
    return (one_strand(reference, alternate, CONTEXT // 2)
            + one_strand(reverse_complement(reference), reverse_complement(alternate), CONTEXT // 2 - 1)) / 2



def environment():
    import torch
    packages = ['numpy', 'pandas', 'scikit-learn', 'biopython', 'evo2', 'vtx',
                'torch', 'transformer_engine', 'flash-attn', 'huggingface-hub']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in packages},
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}



def assert_evaluation_identity(identity):
    lock = OUTPUT / 'validation_report_started.json'
    if lock.exists() and read_json(lock)['identity'] != identity:
        raise RuntimeError('The recorded validation experiment changed. Preserve its provenance and use a new output directory.')



def save_npz(path, **arrays):
    temporary = Path(path).with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)



def load_chunk(path, keys, identity_hash):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if str(values['identity']) != identity_hash or values['keys'].tolist() != list(keys):
        raise ValueError(f'Feature cache provenance/key mismatch: {path}')
    for name, width in [('evo', HIDDEN * 2), ('sequence', len(FEATURE_NAMES)), ('zero_shot', None)]:
        shape = (len(keys), width) if width else (len(keys),)
        if values[name].shape != shape or not np.isfinite(values[name]).all():
            raise ValueError(f'Invalid {name} cache: {path}')
    return values



def extract_features(sequences):
    """Extract resumable batches of 50 variants. No labels are loaded here."""
    verify_protocol()
    frozen_dna = read_csv(q1.OUTPUT / 'sequences.csv.gz')
    if list(sequences.columns) != DNA_COLUMNS or not sequences.equals(frozen_dna):
        raise ValueError('Feature inputs must be exactly the frozen DNA allowlist')
    identity = experiment_identity()
    assert_evaluation_identity(identity)
    identity_hash = fingerprint(identity)
    cache = OUTPUT / 'features' / identity_hash
    cache.mkdir(parents=True, exist_ok=True)
    model, scales, memo = None, None, {}
    start = time.monotonic()
    files = {}
    for offset in range(0, len(sequences), 50):
        batch = sequences.iloc[offset:offset + 50]
        path = cache / f'{offset:05d}.npz'
        if not path.exists():
            if (OUTPUT / 'validation_report_started.json').exists():
                raise RuntimeError('A frozen feature batch is missing; restore it before resuming the evaluated run.')
            if model is None:
                model = load_model()
                measured = benchmark(model)
                scales = fp8_scales(model)
                print(f"Synthetic checks passed; estimated full inference {measured['estimated_5000_variant_minutes']:.1f} min.", flush=True)

            def strand_average(sequence):
                key = context_hash(sequence)
                if key not in memo:
                    # Canonical order also makes independent resumptions bitwise repeatable.
                    canonical = min(sequence, reverse_complement(sequence))
                    forward = encode_sequence(model, canonical)
                    reverse = encode_sequence(model, reverse_complement(canonical))
                    memo[key] = ((forward[0] + reverse[0]) / 2, (forward[1] + reverse[1]) / 2)
                return memo[key]

            evo, baseline, zero = [], [], []
            for row in batch.itertuples(index=False):
                ref, alt = strand_average(row.ref_sequence), strand_average(row.alt_sequence)
                evo.append(np.concatenate([ref[0], alt[0] - ref[0]]))
                baseline.append(sequence_baseline(row.ref_sequence, row.alt_sequence))
                zero.append(-(alt[1] - ref[1]))
            assert_scales_unchanged(model, scales)
            save_npz(path, keys=batch.variant_key.to_numpy(dtype=str), identity=np.array(identity_hash),
                     evo=np.asarray(evo, dtype=np.float32), sequence=np.asarray(baseline, dtype=np.float32),
                     zero_shot=np.asarray(zero, dtype=np.float64))
        load_chunk(path, batch.variant_key, identity_hash)
        files[str(path.relative_to(OUTPUT))] = digest_file(path)
        if offset % 250 == 0 or offset + len(batch) == len(sequences):
            print(f'{offset + len(batch):,}/{len(sequences):,} variants cached ({(time.monotonic() - start) / 60:.1f} min this run)', flush=True)
    feature_manifest = {'identity': identity, 'identity_hash': identity_hash, 'files': files,
                        'rows': len(sequences), 'feature_inputs': DNA_COLUMNS,
                        'dimensions': {'evo': HIDDEN * 2, 'sequence': len(FEATURE_NAMES), 'zero_shot': 1}}
    write_json(OUTPUT / 'feature_manifest.json', feature_manifest, frozen=True)
    if model is not None:
        import torch
        write_json(OUTPUT / 'compute.json', {'extraction_seconds': time.monotonic() - start,
                   'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
                   'environment': identity['environment'], 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
        del model, scales, memo
        torch.cuda.empty_cache()
    return load_features()



def load_features():
    manifest = read_json(OUTPUT / 'feature_manifest.json')
    identity = experiment_identity()
    assert_evaluation_identity(identity)
    if manifest['identity'] != identity:
        raise ValueError('Feature implementation/environment differs; cached features cannot be mixed')
    pilot = read_csv(q1.OUTPUT / 'split_manifest.csv')
    batches, offset = [], 0
    for name, digest in sorted(manifest['files'].items()):
        verify_file(OUTPUT / name, digest)
        keys = pilot.variant_key.iloc[offset:offset + 50]
        batches.append(load_chunk(OUTPUT / name, keys, manifest['identity_hash']))
        offset += len(keys)
    if offset != len(pilot) or offset != manifest['rows']:
        raise AssertionError('Feature manifest does not cover the pilot')
    return {key: np.concatenate([batch[key] for batch in batches]) for key in
            ['keys', 'evo', 'sequence', 'zero_shot']}



def load_labels(split, keys):
    """Load train/validation outcomes from the two canonical Q1 VCF files."""
    return q1.load_partition_labels(split, keys)


def fit_candidates(x_train, y_train, x_validation, y_validation, cs=CS):
    """Only these four arrays enter selection. There is no test argument."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    train, validation = scaler.transform(x_train), scaler.transform(x_validation)
    candidates, best, best_auc = [], None, -np.inf
    for c in sorted(cs):
        model = LogisticRegression(C=c, penalty='l2', class_weight='balanced', solver='lbfgs',
                                   max_iter=3000, tol=1e-5, random_state=SEED).fit(train, y_train)
        if int(model.n_iter_.max()) >= model.max_iter:
            raise RuntimeError('Classifier did not converge; stop before reporting results')
        score = float(roc_auc_score(y_validation, model.decision_function(validation)))
        candidates.append({'C': c, 'validation_auroc': score, 'iterations': int(model.n_iter_.max())})
        if score > best_auc:
            best_auc = score
            best = {'C': c, 'mean': scaler.mean_, 'scale': scaler.scale_,
                    'coef': model.coef_[0], 'intercept': model.intercept_[0]}
    return best, candidates



def predict_saved(model, values):
    score = ((values - model['mean']) / model['scale']) @ model['coef'] + model['intercept']
    if not np.isfinite(score).all():
        raise ValueError('Non-finite classifier predictions')
    return score



def select_models():
    """Train-only fits; select C using clinvar-test.vcf as validation."""
    verify_protocol()
    features = load_features()
    pilot = read_csv(q1.OUTPUT / 'split_manifest.csv')
    identity = experiment_identity()
    assert_evaluation_identity(identity)
    selection_path = OUTPUT / 'selection.json'
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection['identity'] != identity or selection['features_sha256'] != digest_file(OUTPUT / 'feature_manifest.json'):
            raise RuntimeError('Frozen selection provenance changed')
        for name, digest in selection['models'].items():
            verify_file(OUTPUT / name, digest)
    else:
        if (OUTPUT / 'validation_report_started.json').exists():
            raise RuntimeError('Cannot refit an already recorded validation experiment')
        train, validation = [pilot.split.eq(split).to_numpy() for split in ['train', 'validation']]
        y_train = load_labels('train', features['keys'][train])
        y_validation = load_labels('validation', features['keys'][validation])
        selection = {'identity': identity, 'features_sha256': digest_file(OUTPUT / 'feature_manifest.json'),
                     'candidates': {}, 'chosen_C': {}, 'models': {},
                     'fitting_splits': ['train'], 'selection_splits': ['validation'], 'evaluation_split': 'validation', 'separate_test_set': False}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            for name in ['evo', 'sequence']:
                model, candidates = fit_candidates(features[name][train], y_train, features[name][validation], y_validation)
                path = OUTPUT / f'{name}_classifier.npz'
                save_npz(path, **model)
                selection['models'][path.name] = digest_file(path)
                selection['candidates'][name] = candidates
                selection['chosen_C'][name] = model['C']
        write_json(selection_path, selection, frozen=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, candidates in selection['candidates'].items():
        ax.plot([r['C'] for r in candidates], [r['validation_auroc'] for r in candidates], 'o-', label=name)
    ax.set(xscale='log', xlabel='C', ylabel='Validation AUROC', title='Validation-only selection')
    ax.legend(); fig.tight_layout(); plt.show()
    q0.details('Selected models and train/validation audit', selection)
    return selection



def paired_bootstrap(labels, predictions, groups, repetitions=1000, seed=SEED):
    """Resample components together for every method; skip one-class resamples."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    samples = {name: [] for name in predictions}
    differences = []
    for _ in range(repetitions):
        index = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        if len(np.unique(labels[index])) < 2:
            continue
        aucs = {name: float(roc_auc_score(labels[index], values[index])) for name, values in predictions.items()}
        for name, value in aucs.items():
            samples[name].append(value)
        differences.append(aucs['evo'] - aucs['zero_shot'])
    if len(differences) < repetitions * .9:
        raise ValueError('Too few valid group bootstrap resamples')
    return {'auroc_95ci': {name: np.quantile(values, [.025, .975]).tolist() for name, values in samples.items()},
            'evo_minus_zero_shot_95ci': np.quantile(differences, [.025, .975]).tolist(),
            'valid_resamples': len(differences), 'requested_resamples': repetitions,
            'independent_components': len(unique), 'seed': seed}



def report_validation():
    """Report the same validation partition used for C selection.

    These are development results, not an independent final test estimate.
    Unchanged reruns reuse the recorded predictions and model provenance.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    verify_protocol()
    identity = experiment_identity()
    assert_evaluation_identity(identity)
    features = load_features()
    selection = read_json(OUTPUT / 'selection.json')
    if selection['identity'] != identity or selection['features_sha256'] != digest_file(OUTPUT / 'feature_manifest.json'):
        raise ValueError('Selection does not match this experiment')
    for name, digest in selection['models'].items():
        verify_file(OUTPUT / name, digest)
    lock = {'identity': identity, 'selection_sha256': digest_file(OUTPUT / 'selection.json'),
            'features_sha256': digest_file(OUTPUT / 'feature_manifest.json')}
    write_json(OUTPUT / 'validation_report_started.json', lock, frozen=True)
    result_path = OUTPUT / 'validation_report.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['lock'] != lock:
            raise ValueError('Saved evaluation belongs to another experiment')
        verify_file(OUTPUT / 'validation_predictions.csv', result['predictions_sha256'])
    else:
        pilot = read_csv(q1.OUTPUT / 'split_manifest.csv')
        validation = pilot.split.eq('validation').to_numpy()
        labels = load_labels('validation', features['keys'][validation])
        predictions = {'zero_shot': features['zero_shot'][validation]}
        for name in ['evo', 'sequence']:
            with np.load(OUTPUT / f'{name}_classifier.npz', allow_pickle=False) as archive:
                predictions[name] = predict_saved(archive, features[name][validation])
        if not all(np.isfinite(value).all() for value in predictions.values()):
            raise ValueError('Non-finite validation predictions')
        groups = pilot.loc[validation, 'component'].to_numpy()
        table = pd.DataFrame({'variant_key': features['keys'][validation], 'component': groups,
                              'label': labels, **predictions})
        table.to_csv(OUTPUT / 'validation_predictions.csv', index=False)
        result = {'lock': lock, 'status': 'Validation comparison; the same partition selects C, so results are selection-biased',
                  'validation_variants': len(labels), 'validation_pathogenic': int(labels.sum()),
                  'validation_prevalence': float(labels.mean()),
                  'metrics': {name: {'auroc': float(roc_auc_score(labels, values)),
                                     'average_precision': float(average_precision_score(labels, values))}
                              for name, values in predictions.items()},
                  'bootstrap': paired_bootstrap(labels, predictions, groups),
                  'predictions_sha256': digest_file(OUTPUT / 'validation_predictions.csv')}
        result['evo_minus_zero_shot_auroc'] = result['metrics']['evo']['auroc'] - result['metrics']['zero_shot']['auroc']
        write_json(result_path, result, frozen=True)
    show_results(result)
    return result



def show_results(result):
    from sklearn.metrics import precision_recall_curve, roc_curve
    table = read_csv(OUTPUT / 'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = {'evo': 'Evo2 + logistic regression', 'zero_shot': 'Zero-shot Evo2', 'sequence': 'Sequence + logistic regression'}
    for key, title in names.items():
        fpr, tpr, _ = roc_curve(table.label, table[key])
        precision, recall, _ = precision_recall_curve(table.label, table[key])
        metric = result['metrics'][key]
        axes[0].plot(fpr, tpr, label=f"{title}: {metric['auroc']:.3f}")
        axes[1].plot(recall, precision, label=f"{title}: {metric['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], ':', color='gray')
    axes[1].axhline(result['validation_prevalence'], ls=':', color='gray', label='Prevalence')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC · AUROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Validation PR · average precision')
    for ax in axes:
        ax.legend(fontsize=7, loc='lower right'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.savefig(OUTPUT / 'validation_curves.png', dpi=160)
    plt.show()
    fig, ax = plt.subplots(figsize=(7, 2.6), layout='constrained')
    for i, key in enumerate(['evo', 'zero_shot', 'sequence']):
        point = result['metrics'][key]['auroc']
        low, high = result['bootstrap']['auroc_95ci'][key]
        ax.plot([low, high], [i, i], lw=2)
        ax.plot(point, i, 'o', color='black')
    ax.set(yticks=range(3), yticklabels=[names[k] for k in ['evo', 'zero_shot', 'sequence']],
           xlabel='AUROC · 95% component bootstrap interval', xlim=(0, 1))
    fig.savefig(OUTPUT / 'auroc_intervals.png', dpi=160)
    plt.show()
    delta = result['evo_minus_zero_shot_auroc']
    low, high = result['bootstrap']['evo_minus_zero_shot_95ci']
    conclusion = 'higher than' if low > 0 else 'lower than' if high < 0 else 'not clearly different from'
    print(f'Evo2 classifier AUROC is {conclusion} zero-shot in this pilot: Δ={delta:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}].')
    print('Validation also selects C. These development metrics and intervals are not independent final test estimates.')
    q0.details('Validation metrics and paired component bootstrap', result)



def show_conclusion(result):
    from IPython.display import Markdown, display
    evo = result['metrics']['evo']['auroc']
    zero = result['metrics']['zero_shot']['auroc']
    low, high = result['bootstrap']['evo_minus_zero_shot_95ci']
    answer = ('The Evo2 classifier has higher validation AUROC than zero-shot.' if low > 0
              else 'This pilot does not show a clear validation improvement over zero-shot.')
    display(Markdown(f'**{answer}** AUROC: **{evo:.3f}** versus **{zero:.3f}** '
                     f'(difference 95% CI **[{low:+.3f}, {high:+.3f}]**). '
                     'The same validation data selects C; an independent final test is still needed.'))


def run_checks():
    """Run meaningful synthetic leakage/model tests in the notebook workflow."""
    import unittest
    from . import test_q2
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_q2)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    print(f'{result.testsRun} model, feature and cache isolation tests passed.')

