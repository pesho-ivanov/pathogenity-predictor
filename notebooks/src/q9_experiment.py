"""Fixed supervised BioNeMo experiment. Runs only after every Q9 gate passes.

The loop uses NeMo's actual Hyena backbone, replacing the tutorial's language-model
objective with paired missense classification. There is no alternative backend.
"""

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from . import q1, q2, q9
from .q9_backend import MasterAdamW, load_backbone, parameter_hash, select_trainable_block, variant_features

OUTPUT = q9.OUTPUT


def verify_artifact_identity(record, identity, path):
    """Permit only the explicitly archived, reviewed resume-only code migration."""
    if record['identity'] == identity:
        return
    migration = q1.read_json(OUTPUT / 'resume_migration.json')
    q9.assert_identity(migration['to_identity'], identity)
    q9.assert_identity(record['identity'], migration['from_identity'])
    old_protocol = q1.read_json(OUTPUT / migration['archive'] / 'protocol.json')
    q1.verify_file(OUTPUT / migration['archive'] / 'protocol.json', migration['artifacts']['protocol.json'])
    q9.assert_identity(q1.fingerprint(old_protocol), migration['from_identity'])
    current = q1.read_json(OUTPUT / 'protocol.json')
    q9.assert_identity({k: v for k, v in old_protocol.items() if k != 'sources'},
                       {k: v for k, v in current.items() if k != 'sources'})
    changed = {n for n in old_protocol['sources'].keys() | current['sources'].keys()
               if old_protocol['sources'].get(n) != current['sources'].get(n)}
    if changed - {'notebooks/src/q9_experiment.py', 'notebooks/src/q9_environment.py'}:
        raise ValueError('Resume migration changes model, feature or input code')
    q1.verify_file(path, migration['artifacts'][path.name])


def load_adapter(path, identity):
    state = torch.load(path, map_location='cpu', weights_only=True)
    verify_artifact_identity(state, identity, path)
    if state['block_index'] != q9.CONFIG['trainable_block']:
        raise ValueError('Adapter targets the wrong trainable block')
    if any(not torch.isfinite(p).all() for group in ['block', 'head'] for p in state[group].values()):
        raise ValueError('Non-finite saved adapter')
    return state


def restore_adapter(state, block, head, optimizer, mean, scale):
    """Restore BF16 weights, FP32 masters and Adam moments at an epoch boundary."""
    if set(state['block']) != dict(block.named_parameters()).keys():
        raise ValueError('Saved backbone parameters differ from the selected block')
    if not torch.equal(state['mean'], mean.cpu()) or not torch.equal(state['scale'], scale.cpu()):
        raise ValueError('Saved training scaler changed')
    if len(state['masters']) != len(optimizer.masters):
        raise ValueError('Saved optimizer parameter count changed')
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.copy_(state['block'][name])
        head.load_state_dict(state['head'])
        for master, saved in zip(optimizer.masters, state['masters']):
            master.copy_(saved)
    optimizer.optimizer.load_state_dict(state['optimizer'])
    optimizer.optimizer.zero_grad(set_to_none=True)
    for accumulator in optimizer.accumulated:
        accumulator.zero_()


def early_stopping_state(history):
    if [row['epoch'] for row in history] != list(range(len(history))):
        raise ValueError('Checkpoint history is not contiguous from epoch 0')
    if not history or any(not np.isfinite(row['auroc']) for row in history):
        raise ValueError('Invalid checkpoint history')
    # np.argmax retains the earlier epoch on ties, including the frozen model.
    best_epoch = int(np.argmax([row['auroc'] for row in history]))
    return history[best_epoch]['auroc'], best_epoch, history[-1]['epoch']-best_epoch


def predict_variants(model, tokenizer, head, mean, scale, dna):
    predictions = []
    with torch.no_grad():
        for i, row in enumerate(dna.itertuples()):
            features, _ = variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence)
            predictions.append(float(head((features-mean)/scale).reshape(())))
            if (i+1) % 500 == 0:
                print(f'Validation inference: {i+1}/{len(dna)}', flush=True)
    return np.asarray(predictions)


def metric_values(labels, scores):
    return {'auroc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores))}


def paired_intervals(labels, predictions, groups, repetitions=1000):
    """Resample whole validation components, identically for all predictors."""
    rng = np.random.default_rng(42)
    members = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    samples = {f'fine_tuned-minus-{name}': {'auroc': [], 'average_precision': []}
               for name in predictions if name != 'fine_tuned'}
    for _ in range(repetitions):
        selected = rng.integers(0, len(members), size=len(members))
        index = np.concatenate([members[i] for i in selected])
        if len(np.unique(labels[index])) < 2:
            continue
        scores = {n: metric_values(labels[index], p[index]) for n, p in predictions.items()}
        for name in predictions:
            if name != 'fine_tuned':
                for metric in ['auroc', 'average_precision']:
                    samples[f'fine_tuned-minus-{name}'][metric].append(
                        scores['fine_tuned'][metric]-scores[name][metric])
    if any(len(x) < .9 * repetitions for row in samples.values() for x in row.values()):
        raise ValueError('Too few valid component-bootstrap samples')
    return {name: {metric: {'95_percent_interval': np.quantile(values, [.025, .975]).tolist(),
                           'valid_repetitions': len(values)} for metric, values in row.items()}
            for name, row in samples.items()}


def extract(model, tokenizer, dna, identity):
    path, marker = OUTPUT / 'frozen_features.npz', OUTPUT / 'frozen_features.json'
    if path.exists() or marker.exists():
        record = q1.read_json(marker)
        verify_artifact_identity(record, identity, marker)
        q1.verify_file(path, record['sha256'])
        with np.load(path, allow_pickle=False) as saved:
            if saved['keys'].tolist() != dna.variant_key.tolist():
                raise ValueError('Cached feature order differs from frozen inputs')
            values = {name: saved[name] for name in ['evo', 'zero_shot', 'sequence']}
    else:
        values = {'evo': [], 'zero_shot': [], 'sequence': []}
        start = time.perf_counter()
        for i, row in enumerate(dna.itertuples()):
            features, score = variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence)
            values['evo'].append(features.cpu().numpy())
            values['zero_shot'].append(float(score))
            values['sequence'].append(q2.sequence_baseline(row.ref_sequence, row.alt_sequence))
            if (i+1) % 100 == 0:
                print(f'Frozen BioNeMo features: {i+1}/{len(dna)}', flush=True)
        values = {name: np.asarray(rows) for name, rows in values.items()}
        temporary = path.with_suffix('.partial.npz')
        np.savez(temporary, keys=dna.variant_key.to_numpy(dtype=str), **values)
        temporary.replace(path)
        q1.write_json(marker, {'identity': identity, 'sha256': q1.digest_file(path),
                              'seconds': time.perf_counter()-start,
                              'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3})
    if any(not np.isfinite(array).all() or len(array) != len(dna) for array in values.values()):
        raise ValueError('Invalid frozen feature cache')
    if values['evo'].shape != (len(dna), 3840) or values['sequence'].shape != (len(dna), 73):
        raise ValueError('Feature dimensions changed')
    return values


def save_checkpoint(path, block, head, optimizer, mean, scale, identity, epoch, progress=None):
    temporary = path.with_suffix('.partial.pt')
    torch.save({'identity': identity, 'epoch': epoch, 'block_index': q9.CONFIG['trainable_block'],
                'block': {name: p.detach().cpu() for name, p in block.named_parameters()},
                'head': {name: p.detach().cpu() for name, p in head.state_dict().items()},
                'masters': [p.detach().cpu() for p in optimizer.masters],
                'optimizer': optimizer.optimizer.state_dict(),
                'mean': mean.cpu(), 'scale': scale.cpu(), 'progress': progress}, temporary)
    temporary.replace(path)


def run():
    identity = q9.require_ready()  # Always before model loading, feature fitting or training.
    finished = OUTPUT / 'metrics.json'
    if finished.exists():
        result = q1.read_json(finished)
        q9.assert_identity(result['identity'], identity)
        for name, digest in result['artifacts'].items():
            q1.verify_file(OUTPUT / name, digest)
        print('Verified completed Q9 experiment.', flush=True)
        return
    resume = load_adapter(OUTPUT / 'last_adapter.pt', identity) if (OUTPUT / 'last_adapter.pt').exists() else None
    _, pilot, dna, labels, _, _ = q9.verify_inputs()
    q9.assert_feature_inputs(dna, dna)
    # Split labels are aligned to manifest keys, independently of the DNA file's row order.
    split_rows = {s: pilot.loc[pilot.split.eq(s)] for s in ['train', 'validation']}
    positions = {key: i for i, key in enumerate(dna.variant_key)}
    index = {s: np.array([positions[k] for k in rows.variant_key]) for s, rows in split_rows.items()}
    model, tokenizer = load_backbone()
    try:
        values = extract(model, tokenizer, dna, identity)
        baselines, candidates, fit_seconds = {}, {}, {}
        baseline_path = OUTPUT / 'baseline_selection.json'
        if baseline_path.exists():
            saved_baselines = q1.read_json(baseline_path)
            verify_artifact_identity(saved_baselines, identity, baseline_path)
            if saved_baselines['fitted_on'] != 'train':
                raise ValueError('Baseline was not fitted on training data only')
            baselines = {n: {k: np.asarray(v) if isinstance(v, list) else v for k, v in row.items()}
                         for n, row in saved_baselines['models'].items()}
        else:
            for name in ['evo', 'sequence']:
                start = time.perf_counter()
                baselines[name], candidates[name] = q2.fit_candidates(
                    values[name][index['train']], labels['train'],
                    values[name][index['validation']], labels['validation'], cs=q9.CONFIG['baseline_C'])
                fit_seconds[name] = time.perf_counter()-start
            serial = {name: {key: val.tolist() if isinstance(val, np.ndarray) else float(val)
                             for key, val in fitted.items()} for name, fitted in baselines.items()}
            q1.write_json(baseline_path, {'identity': identity,
                          'candidates': candidates, 'fitted_on': 'train', 'models': serial,
                          'fit_seconds': fit_seconds})
        block = select_trainable_block(model)
        frozen_hash = parameter_hash(model, frozen_only=True)
        head = torch.nn.Linear(3840, 1, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(baselines['evo']['coef'], device='cuda')[None])
            head.bias.fill_(baselines['evo']['intercept'])
        mean = torch.tensor(baselines['evo']['mean'], device='cuda', dtype=torch.float32)
        scale = torch.tensor(baselines['evo']['scale'], device='cuda', dtype=torch.float32)
        optimizer = MasterAdamW(block, head)
        train = dna.iloc[index['train']]
        validation = dna.iloc[index['validation']]
        class_counts = np.bincount(labels['train'], minlength=2)
        class_weights = len(train)/(2*class_counts)
        if resume is not None:
            settings = q1.read_json(OUTPUT / 'training_settings.json')
            q9.assert_identity({k: settings[k] for k in q9.CONFIG}, q9.CONFIG)
            if settings['frozen_sha256'] != frozen_hash or settings['class_weights'] != class_weights.tolist():
                raise ValueError('Frozen backbone or training class weights changed before resume')
        q1.write_json(OUTPUT / 'training_settings.json', {**q9.CONFIG,
                      'class_weights': class_weights.tolist(), 'frozen_sha256': frozen_hash})
        # Use the actual FP32 deployed head for epoch 0 as well as later epochs.
        with torch.no_grad():
            cached = torch.tensor(values['evo'][index['validation']], device='cuda')
            frozen_predictions = head((cached-mean)/scale).flatten().cpu().numpy()
        history = [{'epoch': 0, **metric_values(labels['validation'], frozen_predictions), 'train_loss': None,
                    'seconds': 0., 'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3}]
        best_auc, best_epoch, stale = history[0]['auroc'], 0, 0
        best_predictions = frozen_predictions.copy()
        best_path = OUTPUT / 'best_adapter.pt'
        if resume is not None:
            best_state = load_adapter(best_path, identity)
            # An interruption after saving a newly best epoch but before saving
            # last_adapter can still recover that completed epoch.
            if best_state.get('progress') and best_state['epoch'] > resume['epoch']:
                resume = best_state
            progress = resume.get('progress')
            if not progress:
                migration = q1.read_json(OUTPUT / 'resume_migration.json')
                q1.verify_file(OUTPUT / 'history.json', migration['artifacts']['history.json'])
            history = progress['history'] if progress else q1.read_json(OUTPUT / 'history.json')
            if history[-1]['epoch'] != resume['epoch']:
                raise ValueError('Saved history and checkpoint epochs differ')
            best_auc, best_epoch, stale = early_stopping_state(history)
            if best_state['epoch'] != best_epoch:
                raise ValueError('Best checkpoint disagrees with the preserved selection history')
            if progress:
                best_predictions = np.asarray(progress['best_predictions'])
            elif best_epoch != 0:
                raise ValueError('Legacy checkpoint has no saved best predictions')
            if best_predictions.shape != frozen_predictions.shape or not np.isfinite(best_predictions).all():
                raise ValueError('Invalid saved validation predictions')
            restore_adapter(resume, block, head, optimizer, mean, scale)
            print(f'Resuming after completed epoch {resume["epoch"]}; best epoch {best_epoch}, patience {stale}/{q9.CONFIG["patience"]}.', flush=True)
        else:
            save_checkpoint(best_path, block, head, optimizer, mean, scale, identity, 0,
                            {'history': history, 'best_predictions': best_predictions.tolist()})
        first_epoch = history[-1]['epoch']+1
        final_epoch = q9.CONFIG['max_epochs'] if stale < q9.CONFIG['patience'] else first_epoch-1
        for epoch in range(first_epoch, final_epoch+1):
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            order = np.random.default_rng(42+epoch).permutation(len(train))
            losses = []
            for offset in range(0, len(order), q9.CONFIG['gradient_accumulation']):
                batch = order[offset:offset+q9.CONFIG['gradient_accumulation']]
                for j in batch:
                    row = train.iloc[j]
                    features, _ = variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence, True)
                    target = torch.tensor(float(labels['train'][j]), device='cuda')
                    score = head((features-mean)/scale).reshape(())
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, target)
                    loss = loss * float(class_weights[labels['train'][j]])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    loss.backward()
                    losses.append(float(loss.detach()))
                    optimizer.accumulate()
                optimizer.step(denominator=len(batch))
                if offset % 200 == 0:
                    print(f'Epoch {epoch}: {min(offset+len(batch), len(train))}/{len(train)}', flush=True)
            if parameter_hash(model, frozen_only=True) != frozen_hash:
                raise RuntimeError('Frozen backbone changed during training')
            predictions = predict_variants(model, tokenizer, head, mean, scale, validation)
            measured = metric_values(labels['validation'], predictions)
            history.append({'epoch': epoch, **measured, 'train_loss': float(np.mean(losses)),
                            'seconds': time.perf_counter()-start,
                            'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3})
            if measured['auroc'] > best_auc:
                best_auc, best_epoch, stale = measured['auroc'], epoch, 0
                best_predictions = predictions.copy()
                save_checkpoint(best_path, block, head, optimizer, mean, scale, identity, epoch,
                                {'history': history, 'best_predictions': best_predictions.tolist()})
            else:
                stale += 1
            save_checkpoint(OUTPUT / 'last_adapter.pt', block, head, optimizer, mean, scale, identity, epoch,
                            {'history': history, 'best_predictions': best_predictions.tolist()})
            q1.write_json(OUTPUT / 'history.json', history)
            print('EPOCH', history[-1], flush=True)
            if stale >= q9.CONFIG['patience']:
                break
        best_state = load_adapter(best_path, identity)
        restore_adapter(best_state, block, head, optimizer, mean, scale)
        print(f'Verifying predictions from reloaded best checkpoint (epoch {best_epoch}).', flush=True)
        reloaded_predictions = predict_variants(model, tokenizer, head, mean, scale, validation)
        if not np.allclose(reloaded_predictions, best_predictions, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Reloaded best checkpoint does not reproduce selected predictions')
        predictions = {'fine_tuned': best_predictions, 'frozen': frozen_predictions,
                       'zero_shot': values['zero_shot'][index['validation']],
                       'sequence': q2.predict_saved(baselines['sequence'], values['sequence'][index['validation']])}
        np.savez(OUTPUT / 'validation_predictions.npz',
                 keys=validation.variant_key.to_numpy(dtype=str), labels=labels['validation'], **predictions)
        artifacts = ['best_adapter.pt', 'last_adapter.pt', 'baseline_selection.json', 'training_settings.json',
                     'history.json', 'validation_predictions.npz', 'frozen_features.json', 'frozen_features.npz']
        if (OUTPUT / 'resume_migration.json').exists():
            artifacts.append('resume_migration.json')
        result = {'status': 'complete', 'identity': identity, 'selected_epoch': best_epoch,
                  'reloaded_predictions_verified': True,
                  'metrics': {n: metric_values(labels['validation'], p) for n, p in predictions.items()},
                  'paired_differences': paired_intervals(labels['validation'], predictions,
                       split_rows['validation'].component.to_numpy(), q9.CONFIG['bootstrap_repetitions']),
                  'history': history, 'evaluation': q9.CONFIG['evaluation'],
                  'artifacts': {n: q1.digest_file(OUTPUT / n) for n in artifacts}}
        q1.write_json(finished, result)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    run()
