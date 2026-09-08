"""Q14: matched adaptation of the final two active Hyena mixers."""

import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import q1, q11, q11_backend as base, q12, q12_backend as previous, q14, q13, q13_backend, q14_adapters


def load_state(path, identity):
    with Path(path).open('rb') as stream:
        checksum, payload = stream.readline(65).strip().decode(), stream.read()
    q14.require(hashlib.sha256(payload).hexdigest() == checksum, 'Q14 checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    q14.require(state['identity'] == identity and state['format'] == 'q14-controlled-v1', 'Q14 checkpoint identity mismatch')
    return state


def save_state(path, state):
    previous.save_state(path, state)


def schedule(step, maximum):
    """One-based update schedule; warm up, then decay to 10% of peak LR."""
    q14.require(1 <= step <= maximum, 'Schedule step is out of range')
    warmup = min(q14.CONFIG['warmup_steps'], maximum)
    if step <= warmup:
        return step / warmup
    fraction = (step - warmup) / max(1, maximum - warmup)
    floor = q14.CONFIG['minimum_lr_ratio']
    return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2


def head_loss(head, values, targets, weights, strength):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(head(values).flatten(), targets, reduction='none')
    return (bce * weights).mean() + strength * head.weight.square().sum() / 2


class AdapterOptimizer:
    """FP32 adapter masters; classifier parameters never enter the optimizer."""

    def __init__(self, adapters, head, adapter_lr):
        q14.require(q14.CONFIG['head_lr'] == 0., 'Q14 requires a fixed classifier')
        self.named = list(adapters.named_parameters())
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.named]
        self.head = head.requires_grad_(False)
        self.optimizer = torch.optim.AdamW([
            {'params': self.masters, 'lr': adapter_lr, 'weight_decay': q14.CONFIG['adapter_weight_decay']}],
            betas=tuple(q14.CONFIG['betas']), eps=q14.CONFIG['epsilon'])

    def step(self, multiplier):
        diagnostics = {}
        before = [p.detach().clone() for _, p in self.named]
        head_before = [p.detach().clone() for p in self.head.parameters()]
        for (name, p), master in zip(self.named, self.masters):
            q14.require(p.grad is not None and torch.isfinite(p.grad).all(), 'Missing or nonfinite adapter gradient')
            master.grad = p.grad.detach().float().clone()
            diagnostics[name] = {'gradient_norm': float(master.grad.norm())}
            p.grad = None
        q14.require(all(not p.requires_grad and p.grad is None for p in self.head.parameters()),
                    'The fixed classifier unexpectedly has parameter gradients')
        adapter_norm = float(torch.nn.utils.clip_grad_norm_(self.masters, q14.CONFIG['clip_grad'], error_if_nonfinite=True))
        for group in self.optimizer.param_groups:
            group['lr'] = group.setdefault('peak_lr', group['lr']) * multiplier
        self.optimizer.step()
        with torch.no_grad():
            for (name, p), master, old in zip(self.named, self.masters, before):
                p.copy_(master)
                q14.require(torch.isfinite(p).all(), 'Nonfinite deployed adapter')
                diagnostics[name].update(update_norm=float((p.float() - old.float()).norm()),
                                         changed_fraction=float((p != old).float().mean()))
        head_update = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(self.head.parameters(), head_before)))
        q14.require(head_update == 0., 'The fixed classifier changed during an adapter update')
        self.optimizer.zero_grad(set_to_none=True)
        return {'adapter_gradient_norm': adapter_norm,
                'adapter_clip_scale': min(1., q14.CONFIG['clip_grad'] / (adapter_norm + 1e-6)),
                'head_update_norm': head_update, 'adapters': diagnostics}


def cached_features(rows):
    manifest = q1.read_json(q12.OUTPUT / 'feature_manifest.json')
    q14.require(manifest['identity'] == q12.verify_protocol(), 'Q12 cache identity changed')
    result = {}
    for split, frame in rows.items():
        blocks = []
        for offset in range(0, len(frame), 512):
            name = f'{split}-{offset:05d}.npz'
            path = q12.OUTPUT / 'features' / name
            q1.verify_file(path, manifest['batches'][name]['sha256'])
            with np.load(path, allow_pickle=False) as block:
                q14.require(block['keys'].tolist() == frame.iloc[offset:offset+512].variant_key.tolist(), 'Cached feature membership changed')
                values = block['features'].copy()
            q14.require(np.isfinite(values).all() and values.shape[1] == q14.CONFIG['feature_dimension'], 'Invalid cached feature values')
            blocks.append(values)
        result[split] = np.concatenate(blocks)
    print('Verified and reused Q12 frozen feature cache.', flush=True)
    return result


def predict(model, tokenizer, head, mean, scale, rows):
    values = []
    size = q14.CONFIG['microbatch_variants']
    for offset in range(0, len(rows), size):
        raw = previous.features(model, tokenizer, rows.iloc[offset:offset+size])
        values.append(previous.score_head(head, (raw - mean) / scale))
    return np.concatenate(values)


def checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer=None, control_optimizer=None):
    value = {'format': 'q14-controlled-v1', 'identity': identity, 'adapters': previous.adapter_state(adapters),
             'head': base.cpu_state(head.state_dict()), 'control': base.cpu_state(control.state_dict()),
             'mean': mean, 'scale': scale, 'strength': strength, 'progress': progress, 'rng': base.rng_state()}
    if optimizer is not None:
        value.update(masters=optimizer.masters, optimizer=optimizer.optimizer.state_dict())
    q14.require(control_optimizer is None, 'A fixed control must not have an optimizer')
    return base.cpu_state(value)


def load_head(identity):
    protocol = q1.read_json(q14.OUTPUT / 'protocol.json')
    path = q14.ROOT / protocol['initialization_checkpoint']
    q1.verify_file(path, protocol['initialization_checkpoint_sha256'])
    if protocol['initialization_source'] == 'q13_control':
        old = q13_backend.load_state(path, q13.verify_protocol())
    else:
        q14.require(protocol['initialization_source'] == 'q12_control', 'Unknown frozen head initializer')
        old = previous.load_state(path, q12.verify_protocol())
    result = {'format': 'q14-controlled-v1', 'identity': identity,
        'weight': old['control']['weight'], 'bias': old['control']['bias'],
        'mean': old['mean'], 'scale': old['scale'], 'strength': 0.,
        'class_weights': torch.tensor(q1.read_json(q12.OUTPUT / 'head_results.json')['class_weights'], dtype=torch.float32),
        'initialization_source': protocol['initialization_source'],
        'initialization_checkpoint': protocol['initialization_checkpoint'],
        'initialization_checkpoint_sha256': protocol['initialization_checkpoint_sha256'],
        'initial_control_scores': old['control_scores'],
        'initial_control_metrics': protocol['initial_control_metrics'],
        'head_fixed': True}
    save_state(q14.OUTPUT / 'heads.pt', result)
    public = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
    public.update(class_weights=result['class_weights'].tolist(),
                  head_fitting='No fitting: strongest prior frozen classifier is held fixed')
    q1.write_json(q14.OUTPUT / 'head_results.json', public)
    return result


def validate_fixed_head(heads, labels):
    dimension = q14.CONFIG['feature_dimension']
    shapes = {'weight': (1, dimension), 'bias': (1,), 'mean': (dimension,),
              'scale': (dimension,), 'class_weights': (2,),
              'initial_control_scores': (len(labels['validation']),)}
    for name, shape in shapes.items():
        value = heads[name]
        q14.require(isinstance(value, torch.Tensor) and value.shape == shape
                    and value.dtype == torch.float32 and torch.isfinite(value).all(),
                    f'Invalid inherited fixed classifier {name}')
    q14.require((heads['scale'] > 0).all() and heads['strength'] == 0.,
                'Fixed classifier requires positive scales and no head penalty')
    expected = torch.tensor(len(labels['train']) / (2 * np.bincount(labels['train'], minlength=2)),
                            dtype=torch.float32)
    q14.require(torch.equal(heads['class_weights'], expected), 'Class weights differ from frozen training labels')
    measured = q14.metric(labels['validation'], heads['initial_control_scores'].numpy())
    q14.require(all(np.isclose(measured[key], heads['initial_control_metrics'][key]['value'], atol=1e-12, rtol=0.)
                    for key in measured), 'Inherited control scores disagree with frozen initial metrics')


def verify_fixed_tensors(head, control, mean, scale, heads):
    q14.require(all(not p.requires_grad and p.grad is None for branch in (head, control)
                    for p in branch.parameters()), 'Classifier parameters must remain frozen')
    q14.require(all(torch.equal(value.cpu(), heads[name]) for branch in (head, control)
                    for name, value in branch.state_dict().items()), 'The fixed classifier head changed')
    q14.require(torch.equal(mean.cpu(), heads['mean']) and torch.equal(scale.cpu(), heads['scale']),
                'The fixed classifier scaler changed')


def run_trial(name, adapter_lr, model, tokenizer, adapters, initial, rows, labels, cached, heads, identity, budget):
    directory = q14.OUTPUT / name
    directory.mkdir(exist_ok=True)
    previous.restore_adapters(adapters, initial)
    torch.manual_seed(q14.CONFIG['seed'])
    head, control = previous.new_head(q14.CONFIG['feature_dimension']), previous.new_head(q14.CONFIG['feature_dimension'])
    weights = {'weight': heads['weight'], 'bias': heads['bias']}
    head.load_state_dict(weights); control.load_state_dict(weights)
    head.requires_grad_(False); control.requires_grad_(False)
    mean, scale = heads['mean'].cuda().clone(), heads['scale'].cuda().clone()
    strength = heads['strength']
    q14.require(strength == 0., 'Q14 does not refit or regularize classifier weights')
    optimizer = AdapterOptimizer(adapters, head, adapter_lr)
    targets = torch.tensor(labels['train'], device='cuda', dtype=torch.float32)
    sample_weights = torch.tensor(np.asarray(heads['class_weights'])[labels['train']], device='cuda', dtype=torch.float32)
    order = np.random.default_rng(q14.CONFIG['seed']).permutation(len(targets))
    size = q14.CONFIG['microbatch_variants']
    progress = {'trial': name, 'adapter_lr': adapter_lr, 'steps': 0, 'control_steps': 0, 'offset': 0,
                'evaluations': [], 'loss_history': [], 'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                'stop_reason': None, 'best_lora': None, 'best_control': None}

    def evaluate(initial=False):
        verify_fixed_tensors(head, control, mean, scale, heads)
        control_scores = previous.score_head(control, cached['validation'])
        q14.require(np.allclose(control_scores, heads['initial_control_scores'].numpy(), atol=2e-4, rtol=2e-4),
                    'Fixed control no longer reproduces the inherited checkpoint')
        scores = control_scores.copy() if initial else predict(model, tokenizer, head, mean, scale, rows['validation'])
        a, b = q14.metric(labels['validation'], scores), q14.metric(labels['validation'], control_scores)
        q14.require(all(np.isclose(b[key], heads['initial_control_metrics'][key]['value'], atol=1e-12, rtol=0.)
                        for key in b), 'Fixed control metrics differ from frozen initialization')
        progress['evaluations'].append({'steps': progress['steps'], 'elapsed_seconds': budget.elapsed(), 'lora': a, 'control': b})
        new_a = progress['best_lora'] is None or q14.better(a, progress['best_lora'])
        new_b = progress['best_control'] is None or q14.better(b, progress['best_control'])
        if new_a: progress['best_lora'] = a
        if new_b: progress['best_control'] = b
        state = checkpoint(identity, adapters, head, control, mean, scale, strength, progress)
        state.update(lora_scores=torch.from_numpy(scores), control_scores=torch.from_numpy(control_scores))
        if new_a: save_state(directory / 'best_lora.pt', state)
        if new_b: save_state(directory / 'best_control.pt', state)
        print(f'{name} step {progress["steps"]}: LoRA {a}; control {b}; elapsed {budget.elapsed()/60:.1f} min', flush=True)

    evaluate(initial=True)
    for step in range(1, q14.CONFIG['max_steps'] + 1):
        if not budget.can_train(5.):
            progress['stop_reason'] = 'time_budget'
            break
        chosen = order[(step-1)*size:step*size]
        q14.require(len(chosen) == size, 'Matched batch exceeds frozen training membership')
        batch = torch.tensor(chosen, device='cuda')
        raw = previous.features(model, tokenizer, rows['train'].iloc[chosen], gradients=True)
        loss = head_loss(head, (raw - mean) / scale, targets[batch], sample_weights[batch], strength)
        q14.require(torch.isfinite(loss), 'Nonfinite fine-tuning loss')
        loss.backward()
        multiplier = schedule(step, q14.CONFIG['max_steps'])
        diagnostic = optimizer.step(multiplier)
        with torch.no_grad():
            loss_control = head_loss(control, cached['train'][batch], targets[batch], sample_weights[batch], strength)
        q14.require(torch.isfinite(loss_control), 'Nonfinite fixed-control loss')
        progress.update(steps=step, offset=step*size)
        if step % q14.CONFIG['checkpoint_steps'] == 0:
            progress['loss_history'].append({'steps': step, 'loss': float(loss.detach()), 'control_loss': float(loss_control.detach()),
                'lr_multiplier': multiplier, **diagnostic})
            print(f'{name}: {step}/{q14.CONFIG["max_steps"]} updates, elapsed {budget.elapsed()/60:.1f} min', flush=True)
        if step % q14.CONFIG['evaluation_steps'] == 0:
            evaluate()
        if step % q14.CONFIG['checkpoint_steps'] == 0:
            save_state(directory / 'last_checkpoint.pt', checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer))
    if progress['stop_reason'] is None: progress['stop_reason'] = 'max_steps'
    if progress['evaluations'][-1]['steps'] != progress['steps']: evaluate()
    verify_fixed_tensors(head, control, mean, scale, heads)
    progress.update(head_fixed_verified=True, scaler_fixed_verified=True)
    q14.require(progress['control_steps'] == 0 and progress['offset'] == progress['steps'] * size, 'Fixed-control trial cursors differ')
    q14.require(any(not torch.equal(p.cpu(), initial[n]) for n, p in adapters.named_parameters()), 'Adapters did not update')
    save_state(directory / 'last_checkpoint.pt', checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer))
    q1.write_json(directory / 'history.json', progress)
    del optimizer, head, control
    return progress


def finish(model, tokenizer, adapters, initial, rows, labels, groups, histories, identity, budget, frozen):
    q14.require(all(history.get('head_fixed_verified') is True and history.get('scaler_fixed_verified') is True
                    and history.get('control_steps') == 0 for history in histories.values()),
                'A trial did not preserve its fixed classifier and scaler')
    best_name = max(histories, key=lambda k: (histories[k]['best_lora']['auroc'], histories[k]['best_lora']['average_precision']))
    control_name = max(histories, key=lambda k: (histories[k]['best_control']['auroc'], histories[k]['best_control']['average_precision']))
    best = load_state(q14.OUTPUT / best_name / 'best_lora.pt', identity)
    control = load_state(q14.OUTPUT / control_name / 'best_control.pt', identity)
    prior = pd.read_csv(q12.OUTPUT / 'validation_predictions.csv')
    q14.require(prior.variant_key.tolist() == rows['validation'].variant_key.tolist() and np.array_equal(prior.label, labels['validation']), 'Q12 comparison membership changed')
    previous_trial = pd.read_csv(q13.OUTPUT / 'validation_predictions.csv')
    q14.require(previous_trial.variant_key.tolist() == prior.variant_key.tolist()
                and np.array_equal(previous_trial.label, prior.label), 'Q13 comparison membership changed')
    scores = {'lora': best['lora_scores'].numpy(), 'matched_control': best['control_scores'].numpy(),
              'best_control': control['control_scores'].numpy(), 'q12_control': prior['best_control'].to_numpy(),
              'q13_control': previous_trial['best_control'].to_numpy()}
    points = {k: q14.metric(labels['validation'], v) for k, v in scores.items()}
    strongest = max(['best_control', 'q13_control', 'q12_control'], key=lambda k: (points[k]['auroc'], points[k]['average_precision']))
    promoted = q14.promote(points['lora'], points[strongest])
    selected_name = 'lora' if promoted else strongest
    if selected_name == 'lora':
        selected = best
    elif selected_name == 'best_control':
        selected = dict(control, adapters=initial, head=control['control'])
    elif selected_name == 'q13_control':
        old_history = q1.read_json(q13.OUTPUT / 'training_history.json')
        old_name = max(old_history, key=lambda k: (old_history[k]['best_control']['auroc'], old_history[k]['best_control']['average_precision']))
        old = q13_backend.load_state(q13.OUTPUT / old_name / 'best_control.pt', q13.verify_protocol())
        selected = dict(old, format='q14-controlled-v1', identity=identity, adapters=initial, head=old['control'])
    else:
        old = previous.load_state(q12.OUTPUT / 'best_control.pt', q12.verify_protocol())
        selected = dict(old, format='q14-controlled-v1', identity=identity, strength=None,
                        adapters=initial, head=old['control'])
    selected = dict(selected, selected_model='frozen_control' if selected_name == 'best_control' else selected_name,
                    selected_trial=best_name if promoted else control_name if selected_name == 'best_control' else 'Q13' if selected_name == 'q13_control' else 'Q12')
    save_state(q14.OUTPUT / 'selected_model.pt', selected)
    # Verify both the best LoRA and the selected model after erasing learned tensors.
    head = previous.new_head(q14.CONFIG['feature_dimension'])
    n = q14.CONFIG['reload_variants']
    for state, expected in [(best, scores['lora']), (load_state(q14.OUTPUT / 'selected_model.pt', identity), scores[selected_name])]:
        with torch.no_grad():
            for p in list(adapters.parameters()) + list(head.parameters()): p.zero_()
        previous.restore_adapters(adapters, state['adapters']); head.load_state_dict(state['head'])
        actual = predict(model, tokenizer, head, state['mean'].cuda(), state['scale'].cuda(), rows['validation'].iloc[:n])
        q14.require(np.allclose(actual, expected[:n], atol=2e-4, rtol=2e-4), 'Checkpoint reload predictions differ')
    q14.require(base.parameter_hash(model, frozen_only=True) == frozen, 'Original backbone changed')
    scores['selected'] = scores[selected_name]
    frame = pd.DataFrame({'variant_key': rows['validation'].variant_key, 'label': labels['validation'], 'component': groups['validation'], **scores})
    frame.to_csv(q14.OUTPUT / 'validation_predictions.csv', index=False)
    metrics, paired = q14.intervals(labels['validation'], scores, groups['validation'])
    files = ['protocol.json', 'input_checks.json', 'execution.json', 'heads.pt', 'head_results.json', 'preflight.json', 'calibration.json',
             'training_history.json', 'validation_predictions.csv', 'selected_model.pt']
    files += [f'{name}/{f}' for name in histories for f in ['best_lora.pt', 'best_control.pt', 'last_checkpoint.pt', 'history.json']]
    result = {'status': 'complete', 'scope': 'sampled_validation', 'identity': identity,
        'metrics': metrics, 'paired_lora_minus_matched_control': paired, 'promotion_passed': promoted,
        'selected_model': selected['selected_model'], 'selected_trial': selected['selected_trial'],
        'best_lora_trial': best_name, 'best_steps': best['progress']['steps'], 'strongest_control_source': strongest,
        'head_selection_strength': best['strength'], 'training_variants': len(rows['train']), 'validation_variants': len(frame),
        'full_validation_variants': 17927, 'seconds': budget.elapsed(), 'within_one_hour': budget.elapsed() <= q14.CONFIG['total_seconds'],
        'reloaded_predictions_verified': True, 'frozen_unchanged': True, 'matched_training_verified': True,
        'head_fixed_verified': True, 'scaler_fixed_verified': True, 'control_optimizer_steps': 0,
        'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3,
        'artifacts': {name: q1.digest_file(q14.OUTPUT / name) for name in files}, 'limitations': q14.LIMITATIONS}
    q1.write_json(q14.OUTPUT / 'metrics.json', result, frozen=True)
    print(f'Completed Q14: promotion={promoted}; selected {selected["selected_model"]}; {metrics["selected"]}', flush=True)


def explore():
    identity = q14.verify_protocol()
    budget = q14.Budget(q1.read_json(q14.OUTPUT / 'execution.json')['started_utc'])
    rows, labels, groups, _ = q14.inputs()
    raw = cached_features(rows)
    print('Loading the strongest verified frozen classifier; its head will stay fixed.', flush=True)
    heads = load_head(identity)
    validate_fixed_head(heads, labels)
    q14.require(budget.can_train(120.), 'Initialization exhausted the exploration budget')
    model, tokenizer = base.load_backbone()
    try:
        adapters = q14_adapters.attach(model, blocks=tuple(q14.CONFIG['lora']['blocks']),
            rank=q14.CONFIG['lora']['rank'], alpha=q14.CONFIG['lora']['alpha'], seed=q14.CONFIG['seed'])
        initial = previous.adapter_state(adapters)
        frozen = base.parameter_hash(model, frozen_only=True)
        probe = rows['train'].iloc[:q14.CONFIG['microbatch_variants']]
        live = previous.features(model, tokenizer, probe)
        q14.require(np.allclose(live.cpu().numpy(), raw['train'][:len(probe)], atol=2e-4, rtol=2e-4), 'Zero-adapter live features disagree with verified cache')
        reverse = probe.iloc[:2].copy()
        for name in ['ref_sequence', 'alt_sequence']: reverse[name] = reverse[name].map(q1.reverse_complement)
        q14.require(torch.equal(previous.features(model, tokenizer, reverse), previous.features(model, tokenizer, probe.iloc[:2])), 'Strand invariance failed')
        torch.cuda.synchronize(); tick = time.perf_counter()
        for _ in range(3): previous.features(model, tokenizer, probe)
        torch.cuda.synchronize()
        budget.seconds_per_variant = (time.perf_counter() - tick)/(3*len(probe))
        q1.write_json(q14.OUTPUT / 'preflight.json', {'probe_split': 'train', 'probe_keys': probe.variant_key.tolist(),
            'cache_live_equivalent': True, 'strand_invariance_exact': True, 'frozen_sha256': frozen,
            'original_q12_preflight_sha256': q1.digest_file(q12.OUTPUT / 'preflight.json')})
        q1.write_json(q14.OUTPUT / 'calibration.json', {'seconds_per_variant': budget.seconds_per_variant, 'report_reserve_seconds': budget.reserve()})
        mean, scale = heads['mean'].cuda(), heads['scale'].cuda()
        cached = {k: (torch.from_numpy(v).cuda() - mean)/scale for k, v in raw.items()}
        histories = {}
        for lr in q14.CONFIG['adapter_lrs']:
            q14.require(budget.can_train(60.), 'No remaining time for the next registered trial')
            name = f'lr_{lr:.0e}'
            histories[name] = run_trial(name, lr, model, tokenizer, adapters, initial, rows, labels, cached, heads, identity, budget)
            q1.write_json(q14.OUTPUT / 'training_history.json', histories)
        finish(model, tokenizer, adapters, initial, rows, labels, groups, histories, identity, budget, frozen)
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['explore'])
    parser.parse_args()
    explore()
