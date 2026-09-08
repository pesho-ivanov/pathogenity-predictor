"""Q13: converged cached heads and controlled, gentle Evo2 adapter updates."""

import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import q1, q11, q11_backend as base, q12, q12_backend as previous, q13, q13_heads


def load_state(path, identity):
    with Path(path).open('rb') as stream:
        checksum, payload = stream.readline(65).strip().decode(), stream.read()
    q13.require(hashlib.sha256(payload).hexdigest() == checksum, 'Q13 checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    q13.require(state['identity'] == identity and state['format'] == 'q13-controlled-v1', 'Q13 checkpoint identity mismatch')
    return state


def save_state(path, state):
    previous.save_state(path, state)


def schedule(step, maximum):
    """One-based update schedule; warm up, then decay to 10% of peak LR."""
    q13.require(1 <= step <= maximum, 'Schedule step is out of range')
    warmup = min(q13.CONFIG['warmup_steps'], maximum)
    if step <= warmup:
        return step / warmup
    fraction = (step - warmup) / max(1, maximum - warmup)
    floor = q13.CONFIG['minimum_lr_ratio']
    return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2


def head_loss(head, values, targets, weights, strength):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(head(values).flatten(), targets, reduction='none')
    return (bce * weights).mean() + strength * head.weight.square().sum() / 2


class AdapterOptimizer:
    """FP32 masters with independent adapter/head clipping and explicit head L2."""

    def __init__(self, adapters, head, adapter_lr):
        self.named = list(adapters.named_parameters())
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.named]
        self.head = head
        self.optimizer = torch.optim.AdamW([
            {'params': self.masters, 'lr': adapter_lr, 'weight_decay': q13.CONFIG['adapter_weight_decay']},
            {'params': head.parameters(), 'lr': q13.CONFIG['head_lr'], 'weight_decay': 0.}],
            betas=tuple(q13.CONFIG['betas']), eps=q13.CONFIG['epsilon'])

    def step(self, multiplier):
        diagnostics = {}
        before = [p.detach().clone() for _, p in self.named]
        head_before = [p.detach().clone() for p in self.head.parameters()]
        for (name, p), master in zip(self.named, self.masters):
            q13.require(p.grad is not None and torch.isfinite(p.grad).all(), 'Missing or nonfinite adapter gradient')
            master.grad = p.grad.detach().float().clone()
            diagnostics[name] = {'gradient_norm': float(master.grad.norm())}
            p.grad = None
        for p in self.head.parameters():
            q13.require(p.grad is not None and torch.isfinite(p.grad).all(), 'Invalid head gradient')
        adapter_norm = float(torch.nn.utils.clip_grad_norm_(self.masters, q13.CONFIG['clip_grad'], error_if_nonfinite=True))
        head_norm = float(torch.nn.utils.clip_grad_norm_(self.head.parameters(), q13.CONFIG['clip_grad'], error_if_nonfinite=True))
        for group in self.optimizer.param_groups:
            group['lr'] = group.setdefault('peak_lr', group['lr']) * multiplier
        self.optimizer.step()
        with torch.no_grad():
            for (name, p), master, old in zip(self.named, self.masters, before):
                p.copy_(master)
                q13.require(torch.isfinite(p).all(), 'Nonfinite deployed adapter')
                diagnostics[name].update(update_norm=float((p.float() - old.float()).norm()),
                                         changed_fraction=float((p != old).float().mean()))
        head_update = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(self.head.parameters(), head_before)))
        self.optimizer.zero_grad(set_to_none=True)
        return {'adapter_gradient_norm': adapter_norm, 'head_gradient_norm': head_norm,
                'adapter_clip_scale': min(1., q13.CONFIG['clip_grad'] / (adapter_norm + 1e-6)),
                'head_clip_scale': min(1., q13.CONFIG['clip_grad'] / (head_norm + 1e-6)),
                'head_update_norm': head_update, 'adapters': diagnostics}


def cached_features(rows):
    manifest = q1.read_json(q12.OUTPUT / 'feature_manifest.json')
    q13.require(manifest['identity'] == q12.verify_protocol(), 'Q12 cache identity changed')
    result = {}
    for split, frame in rows.items():
        blocks = []
        for offset in range(0, len(frame), 512):
            name = f'{split}-{offset:05d}.npz'
            path = q12.OUTPUT / 'features' / name
            q1.verify_file(path, manifest['batches'][name]['sha256'])
            with np.load(path, allow_pickle=False) as block:
                q13.require(block['keys'].tolist() == frame.iloc[offset:offset+512].variant_key.tolist(), 'Cached feature membership changed')
                values = block['features'].copy()
            q13.require(np.isfinite(values).all() and values.shape[1] == q13.CONFIG['feature_dimension'], 'Invalid cached feature values')
            blocks.append(values)
        result[split] = np.concatenate(blocks)
    print('Verified and reused Q12 frozen feature cache.', flush=True)
    return result


def predict(model, tokenizer, head, mean, scale, rows):
    values = []
    size = q13.CONFIG['microbatch_variants']
    for offset in range(0, len(rows), size):
        raw = previous.features(model, tokenizer, rows.iloc[offset:offset+size])
        values.append(previous.score_head(head, (raw - mean) / scale))
    return np.concatenate(values)


def checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer=None, control_optimizer=None):
    value = {'format': 'q13-controlled-v1', 'identity': identity, 'adapters': previous.adapter_state(adapters),
             'head': base.cpu_state(head.state_dict()), 'control': base.cpu_state(control.state_dict()),
             'mean': mean, 'scale': scale, 'strength': strength, 'progress': progress, 'rng': base.rng_state()}
    if optimizer is not None:
        value.update(masters=optimizer.masters, optimizer=optimizer.optimizer.state_dict(),
                     control_optimizer=control_optimizer.state_dict())
    return base.cpu_state(value)


def fit_cached_heads(raw, labels, identity):
    result = q13_heads.fit_heads(raw['train'], labels['train'], raw['validation'], labels['validation'],
        device='cuda', strengths=tuple(q13.CONFIG['head_strengths']), max_iter=q13.CONFIG['head_max_iter'],
        tolerance=q13.CONFIG['head_tolerance'])
    result.update(format='q13-controlled-v1', identity=identity)
    save_state(q13.OUTPUT / 'heads.pt', result)
    public = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
    public['class_weights'] = result['class_weights'].tolist()
    q1.write_json(q13.OUTPUT / 'head_results.json', public)
    return result


def run_trial(name, adapter_lr, model, tokenizer, adapters, initial, rows, labels, cached, heads, identity, budget):
    directory = q13.OUTPUT / name
    directory.mkdir(exist_ok=True)
    previous.restore_adapters(adapters, initial)
    torch.manual_seed(q13.CONFIG['seed'])
    head, control = previous.new_head(q13.CONFIG['feature_dimension']), previous.new_head(q13.CONFIG['feature_dimension'])
    weights = {'weight': heads['weight'], 'bias': heads['bias']}
    head.load_state_dict(weights); control.load_state_dict(weights)
    mean, scale = heads['mean'].cuda(), heads['scale'].cuda()
    strength = heads['strength']
    optimizer = AdapterOptimizer(adapters, head, adapter_lr)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=q13.CONFIG['head_lr'], weight_decay=0.,
        betas=tuple(q13.CONFIG['betas']), eps=q13.CONFIG['epsilon'])
    targets = torch.tensor(labels['train'], device='cuda', dtype=torch.float32)
    sample_weights = torch.tensor(np.asarray(heads['class_weights'])[labels['train']], device='cuda', dtype=torch.float32)
    order = np.random.default_rng(q13.CONFIG['seed']).permutation(len(targets))
    size = q13.CONFIG['microbatch_variants']
    progress = {'trial': name, 'adapter_lr': adapter_lr, 'steps': 0, 'control_steps': 0, 'offset': 0,
                'evaluations': [], 'loss_history': [], 'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                'stop_reason': None, 'best_lora': None, 'best_control': None}

    def evaluate(initial=False):
        control_scores = previous.score_head(control, cached['validation'])
        scores = control_scores.copy() if initial else predict(model, tokenizer, head, mean, scale, rows['validation'])
        a, b = q13.metric(labels['validation'], scores), q13.metric(labels['validation'], control_scores)
        progress['evaluations'].append({'steps': progress['steps'], 'elapsed_seconds': budget.elapsed(), 'lora': a, 'control': b})
        new_a = progress['best_lora'] is None or q13.better(a, progress['best_lora'])
        new_b = progress['best_control'] is None or q13.better(b, progress['best_control'])
        if new_a: progress['best_lora'] = a
        if new_b: progress['best_control'] = b
        state = checkpoint(identity, adapters, head, control, mean, scale, strength, progress)
        state.update(lora_scores=torch.from_numpy(scores), control_scores=torch.from_numpy(control_scores))
        if new_a: save_state(directory / 'best_lora.pt', state)
        if new_b: save_state(directory / 'best_control.pt', state)
        print(f'{name} step {progress["steps"]}: LoRA {a}; control {b}; elapsed {budget.elapsed()/60:.1f} min', flush=True)

    evaluate(initial=True)
    for step in range(1, q13.CONFIG['max_steps'] + 1):
        if not budget.can_train(5.):
            progress['stop_reason'] = 'time_budget'
            break
        chosen = order[(step-1)*size:step*size]
        q13.require(len(chosen) == size, 'Matched batch exceeds frozen training membership')
        batch = torch.tensor(chosen, device='cuda')
        raw = previous.features(model, tokenizer, rows['train'].iloc[chosen], gradients=True)
        loss = head_loss(head, (raw - mean) / scale, targets[batch], sample_weights[batch], strength)
        q13.require(torch.isfinite(loss), 'Nonfinite fine-tuning loss')
        loss.backward()
        multiplier = schedule(step, q13.CONFIG['max_steps'])
        diagnostic = optimizer.step(multiplier)
        loss_control = head_loss(control, cached['train'][batch], targets[batch], sample_weights[batch], strength)
        loss_control.backward()
        control_norm = float(torch.nn.utils.clip_grad_norm_(control.parameters(), q13.CONFIG['clip_grad'], error_if_nonfinite=True))
        control_optimizer.param_groups[0]['lr'] = q13.CONFIG['head_lr'] * multiplier
        control_optimizer.step(); control_optimizer.zero_grad(set_to_none=True)
        progress.update(steps=step, control_steps=step, offset=step*size)
        if step % q13.CONFIG['checkpoint_steps'] == 0:
            progress['loss_history'].append({'steps': step, 'loss': float(loss.detach()), 'control_loss': float(loss_control.detach()),
                'control_gradient_norm': control_norm, 'lr_multiplier': multiplier, **diagnostic})
            print(f'{name}: {step}/{q13.CONFIG["max_steps"]} updates, elapsed {budget.elapsed()/60:.1f} min', flush=True)
        if step % q13.CONFIG['evaluation_steps'] == 0:
            evaluate()
        if step % q13.CONFIG['checkpoint_steps'] == 0:
            save_state(directory / 'last_checkpoint.pt', checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer, control_optimizer))
    if progress['stop_reason'] is None: progress['stop_reason'] = 'max_steps'
    if progress['evaluations'][-1]['steps'] != progress['steps']: evaluate()
    q13.require(progress['steps'] == progress['control_steps'] and progress['offset'] == progress['steps'] * size, 'Matched trial cursors differ')
    q13.require(any(not torch.equal(p.cpu(), initial[n]) for n, p in adapters.named_parameters()), 'Adapters did not update')
    save_state(directory / 'last_checkpoint.pt', checkpoint(identity, adapters, head, control, mean, scale, strength, progress, optimizer, control_optimizer))
    q1.write_json(directory / 'history.json', progress)
    del optimizer, control_optimizer, head, control
    return progress


def finish(model, tokenizer, adapters, initial, rows, labels, groups, histories, identity, budget, frozen):
    best_name = max(histories, key=lambda k: (histories[k]['best_lora']['auroc'], histories[k]['best_lora']['average_precision']))
    control_name = max(histories, key=lambda k: (histories[k]['best_control']['auroc'], histories[k]['best_control']['average_precision']))
    best = load_state(q13.OUTPUT / best_name / 'best_lora.pt', identity)
    control = load_state(q13.OUTPUT / control_name / 'best_control.pt', identity)
    prior = pd.read_csv(q12.OUTPUT / 'validation_predictions.csv')
    q13.require(prior.variant_key.tolist() == rows['validation'].variant_key.tolist() and np.array_equal(prior.label, labels['validation']), 'Q12 comparison membership changed')
    scores = {'lora': best['lora_scores'].numpy(), 'matched_control': best['control_scores'].numpy(),
              'best_control': control['control_scores'].numpy(), 'q12_control': prior['selected'].to_numpy()}
    points = {k: q13.metric(labels['validation'], v) for k, v in scores.items()}
    strongest = max(['best_control', 'q12_control'], key=lambda k: (points[k]['auroc'], points[k]['average_precision']))
    promoted = q13.promote(points['lora'], points[strongest])
    selected_name = 'lora' if promoted else strongest
    if selected_name == 'lora':
        selected = best
    elif selected_name == 'best_control':
        selected = dict(control, adapters=initial, head=control['control'])
    else:
        old = previous.load_state(q12.OUTPUT / 'selected_model.pt', q12.verify_protocol())
        selected = dict(old, format='q13-controlled-v1', identity=identity, strength=None)
    selected = dict(selected, selected_model='frozen_control' if selected_name == 'best_control' else selected_name,
                    selected_trial=best_name if promoted else control_name if selected_name == 'best_control' else 'Q12')
    save_state(q13.OUTPUT / 'selected_model.pt', selected)
    # Verify both the best LoRA and the selected model after erasing learned tensors.
    head = previous.new_head(q13.CONFIG['feature_dimension'])
    n = q13.CONFIG['reload_variants']
    for state, expected in [(best, scores['lora']), (load_state(q13.OUTPUT / 'selected_model.pt', identity), scores[selected_name])]:
        with torch.no_grad():
            for p in list(adapters.parameters()) + list(head.parameters()): p.zero_()
        previous.restore_adapters(adapters, state['adapters']); head.load_state_dict(state['head'])
        actual = predict(model, tokenizer, head, state['mean'].cuda(), state['scale'].cuda(), rows['validation'].iloc[:n])
        q13.require(np.allclose(actual, expected[:n], atol=2e-4, rtol=2e-4), 'Checkpoint reload predictions differ')
    q13.require(base.parameter_hash(model, frozen_only=True) == frozen, 'Original backbone changed')
    scores['selected'] = scores[selected_name]
    frame = pd.DataFrame({'variant_key': rows['validation'].variant_key, 'label': labels['validation'], 'component': groups['validation'], **scores})
    frame.to_csv(q13.OUTPUT / 'validation_predictions.csv', index=False)
    metrics, paired = q13.intervals(labels['validation'], scores, groups['validation'])
    files = ['protocol.json', 'input_checks.json', 'execution.json', 'heads.pt', 'head_results.json', 'preflight.json', 'calibration.json',
             'training_history.json', 'validation_predictions.csv', 'selected_model.pt']
    files += [f'{name}/{f}' for name in histories for f in ['best_lora.pt', 'best_control.pt', 'last_checkpoint.pt', 'history.json']]
    result = {'status': 'complete', 'scope': 'sampled_validation', 'identity': identity,
        'metrics': metrics, 'paired_lora_minus_matched_control': paired, 'promotion_passed': promoted,
        'selected_model': selected['selected_model'], 'selected_trial': selected['selected_trial'],
        'best_lora_trial': best_name, 'best_steps': best['progress']['steps'], 'strongest_control_source': strongest,
        'head_selection_strength': best['strength'], 'training_variants': len(rows['train']), 'validation_variants': len(frame),
        'full_validation_variants': 17927, 'seconds': budget.elapsed(), 'within_one_hour': budget.elapsed() <= q13.CONFIG['total_seconds'],
        'reloaded_predictions_verified': True, 'frozen_unchanged': True, 'matched_training_verified': True,
        'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3,
        'artifacts': {name: q1.digest_file(q13.OUTPUT / name) for name in files}, 'limitations': q13.LIMITATIONS}
    q1.write_json(q13.OUTPUT / 'metrics.json', result, frozen=True)
    print(f'Completed Q13: promotion={promoted}; selected {selected["selected_model"]}; {metrics["selected"]}', flush=True)


def explore():
    identity = q13.verify_protocol()
    budget = q13.Budget(q1.read_json(q13.OUTPUT / 'execution.json')['started_utc'])
    rows, labels, groups, _ = q13.inputs()
    raw = cached_features(rows)
    print('Fitting converged heads on cached features.', flush=True)
    heads = fit_cached_heads(raw, labels, identity)
    q13.require(budget.can_train(120.), 'Head fitting exhausted the exploration budget')
    model, tokenizer = base.load_backbone()
    try:
        adapters = base.attach_lora(model)
        initial = previous.adapter_state(adapters)
        frozen = base.parameter_hash(model, frozen_only=True)
        probe = rows['train'].iloc[:q13.CONFIG['microbatch_variants']]
        live = previous.features(model, tokenizer, probe)
        q13.require(np.allclose(live.cpu().numpy(), raw['train'][:len(probe)], atol=2e-4, rtol=2e-4), 'Zero-adapter live features disagree with verified cache')
        reverse = probe.iloc[:2].copy()
        for name in ['ref_sequence', 'alt_sequence']: reverse[name] = reverse[name].map(q1.reverse_complement)
        q13.require(torch.equal(previous.features(model, tokenizer, reverse), previous.features(model, tokenizer, probe.iloc[:2])), 'Strand invariance failed')
        torch.cuda.synchronize(); tick = time.perf_counter()
        for _ in range(3): previous.features(model, tokenizer, probe)
        torch.cuda.synchronize()
        budget.seconds_per_variant = (time.perf_counter() - tick)/(3*len(probe))
        q1.write_json(q13.OUTPUT / 'preflight.json', {'probe_split': 'train', 'probe_keys': probe.variant_key.tolist(),
            'cache_live_equivalent': True, 'strand_invariance_exact': True, 'frozen_sha256': frozen,
            'original_q12_preflight_sha256': q1.digest_file(q12.OUTPUT / 'preflight.json')})
        q1.write_json(q13.OUTPUT / 'calibration.json', {'seconds_per_variant': budget.seconds_per_variant, 'report_reserve_seconds': budget.reserve()})
        mean, scale = heads['mean'].cuda(), heads['scale'].cuda()
        cached = {k: (torch.from_numpy(v).cuda() - mean)/scale for k, v in raw.items()}
        histories = {}
        for lr in q13.CONFIG['adapter_lrs']:
            q13.require(budget.can_train(60.), 'No remaining time for the next registered trial')
            name = f'lr_{lr:.0e}'
            histories[name] = run_trial(name, lr, model, tokenizer, adapters, initial, rows, labels, cached, heads, identity, budget)
            q1.write_json(q13.OUTPUT / 'training_history.json', histories)
        finish(model, tokenizer, adapters, initial, rows, labels, groups, histories, identity, budget, frozen)
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['explore'])
    parser.parse_args()
    explore()
