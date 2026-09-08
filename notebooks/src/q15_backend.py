"""Q15: paired adapter objectives using volatile, frozen-prefix CPU caches."""

import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import (q1, q11, q11_backend as base, q12, q12_backend as previous,
               q13, q13_backend, q14, q14_backend, q14_adapters, q15,
               q15_objective, q15_pooling, q15_replay as replay)


def load_state(path, identity):
    with Path(path).open('rb') as stream:
        checksum, payload = stream.readline(65).strip().decode(), stream.read()
    q15.require(hashlib.sha256(payload).hexdigest() == checksum, 'Q15 checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    q15.require(state['format'] == 'q15-cached-v1' and state['identity'] == identity,
                'Q15 checkpoint identity mismatch')
    for name in ['adapters', 'head', 'control']:
        q15.require(bool(state[name]) and all(torch.isfinite(x).all() for x in state[name].values()),
                    f'Invalid Q15 checkpoint {name}')
    q15.require(torch.isfinite(state['mean']).all() and torch.isfinite(state['scale']).all()
                and (state['scale'] > 0).all(), 'Invalid Q15 checkpoint scaler')
    q15.require(set(state['head']) == set(state['control'])
                and all(torch.equal(value, state['control'][name]) for name, value in state['head'].items()),
                'Q15 checkpoint classifier differs from its fixed control')
    def check(value):
        if isinstance(value, torch.Tensor):
            q15.require(torch.isfinite(value).all(), 'Nonfinite Q15 checkpoint tensor')
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
    check(state)
    return state


def schedule(step):
    maximum, warmup = q15.CONFIG['max_attempted_steps'], q15.CONFIG['warmup_steps']
    q15.require(1 <= step <= maximum, 'Attempted update is outside the registered schedule')
    if step <= warmup:
        return step / warmup
    fraction = (step - warmup) / max(1, maximum - warmup)
    floor = q15.CONFIG['minimum_lr_ratio']
    return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2


class Arm:
    """Independent FP32 masters/moments sharing only deployed BF16 adapters."""

    def __init__(self, adapters, objective, peak_lr):
        q15.require(objective in q15.CONFIG['objectives'], 'Unknown Q15 objective')
        self.objective, self.peak_lr = objective, peak_lr
        self.named = list(adapters.named_parameters())
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.named]
        self.optimizer = torch.optim.AdamW(self.masters, lr=peak_lr,
            weight_decay=q15.CONFIG['adapter_weight_decay'],
            betas=tuple(q15.CONFIG['betas']), eps=q15.CONFIG['epsilon'])
        self.attempted_steps = self.steps = 0

    def activate(self):
        with torch.no_grad():
            for (_, parameter), master in zip(self.named, self.masters):
                parameter.copy_(master)
                parameter.grad = None

    def state(self):
        return {name: master.detach().to(dtype=p.dtype, device='cpu').clone()
                for (name, p), master in zip(self.named, self.masters)}

    def update(self, loss_result, attempted_step):
        q15.require(attempted_step == self.attempted_steps + 1, 'Objective cursors diverged')
        self.attempted_steps = attempted_step
        diagnostics = {'attempted_steps': attempted_step, 'steps': self.steps,
                       'loss': float(loss_result.loss.detach()), 'updated': loss_result.has_signal,
                       'positives': loss_result.positives, 'negatives': loss_result.negatives,
                       'pairs': loss_result.pairs, 'lr': self.peak_lr * schedule(attempted_step)}
        if not loss_result.has_signal:
            # AdamW must not see a zero-gradient step: it would still decay weights.
            return diagnostics
        loss_result.loss.backward()
        for (name, parameter), master in zip(self.named, self.masters):
            q15.require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                        f'Missing or nonfinite adapter gradient: {name}')
            master.grad = parameter.grad.detach().float().clone()
            parameter.grad = None
        norm = float(torch.nn.utils.clip_grad_norm_(self.masters, q15.CONFIG['clip_grad'], error_if_nonfinite=True))
        for group in self.optimizer.param_groups:
            group['lr'] = diagnostics['lr']
        self.optimizer.step()
        q15.require(all(torch.isfinite(p).all() for p in self.masters), 'Nonfinite adapter masters')
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1
        diagnostics.update(steps=self.steps, adapter_gradient_norm=norm)
        self.activate()
        return diagnostics


def chunk_orders(count, chunk_size, passes, seed):
    """Every registered variant appears once per pass in one random chunk."""
    order = np.random.default_rng(seed).permutation(count)
    for number, offset in enumerate(range(0, count, chunk_size)):
        indexes = order[offset:offset + chunk_size]
        rng = np.random.default_rng(np.random.SeedSequence([seed, number, 15]))
        yield number, indexes, [rng.permutation(len(indexes)) for _ in range(passes)]


def load_head(protocol, labels):
    modules = {'q12': (q12, previous), 'q13': (q13, q13_backend), 'q14': (q14, q14_backend)}
    question = protocol['initialization_question']
    q15.require(question in modules, 'Unknown Q15 classifier source')
    module, backend = modules[question]
    path = q15.ROOT / protocol['initialization_checkpoint']
    q1.verify_file(path, protocol['initialization_checkpoint_sha256'])
    old = backend.load_state(path, module.verify_protocol())
    heads = {'weight': old['control']['weight'].clone(), 'bias': old['control']['bias'].clone(),
             'mean': old['mean'].clone(), 'scale': old['scale'].clone(), 'strength': 0.,
             'class_weights': torch.tensor(q1.read_json(q12.OUTPUT / 'head_results.json')['class_weights'], dtype=torch.float32),
             'initial_control_scores': old['control_scores'].clone(),
             'initial_control_metrics': protocol['initial_control_metrics']}
    q14_backend.validate_fixed_head(heads, labels)
    public = {key: protocol[key] for key in ['initialization_question', 'initialization_source',
        'initialization_checkpoint', 'initialization_checkpoint_sha256', 'initial_control_metrics',
        'q14_best_control_metrics', 'q14_best_lora_trial', 'adapter_lr']}
    public.update(head_fixed=True, scaler_fixed=True, head_fitting='No fitting; exact inherited control head and scaler',
                  class_weights=heads['class_weights'].tolist())
    q1.write_json(q15.OUTPUT / 'head_results.json', public)
    return heads


def verify_fixed(head, mean, scale, heads):
    q15.require(all(not p.requires_grad and p.grad is None for p in head.parameters()), 'Classifier acquired gradients')
    q15.require(all(torch.equal(value.cpu(), heads[name]) for name, value in head.state_dict().items()),
                'Fixed classifier changed')
    q15.require(torch.equal(mean.cpu(), heads['mean']) and torch.equal(scale.cpu(), heads['scale']),
                'Fixed scaler changed')


def assert_equivalent(actual, expected, name, *, gradient=False):
    """Predeclared tolerances; report errors and fail before training on mismatch."""
    a, b = actual.detach().float(), expected.detach().float()
    q15.require(a.shape == b.shape, f'Full/replay shapes differ for {name}')
    absolute = float((a - b).abs().max())
    relative = float((a - b).norm() / b.norm().clamp_min(1e-12))
    atol, rtol = (1e-7, 2e-3) if gradient else (2e-4, 2e-4)
    q15.require(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                and torch.allclose(a, b, atol=atol, rtol=rtol)
                and (not gradient or relative <= 2e-3), f'Full/replay equivalence failed for {name}: abs={absolute}, relL2={relative}')
    return {'maximum_absolute_error': absolute, 'relative_l2_error': relative,
            'actual_norm': float(a.norm()), 'reference_norm': float(b.norm()), 'atol': atol, 'rtol': rtol}


def logits(head, mean, scale, raw):
    return head((raw - mean) / scale).flatten()


def preflight(model, tokenizer, adapters, initial, head, mean, scale, weights, rows, labels, peak_lr, frozen):
    # A deterministic balanced training-only probe guarantees both objectives have signal.
    size = q15.CONFIG['microbatch_variants']
    indexes = np.column_stack([np.flatnonzero(labels == 0)[:size // 2],
                               np.flatnonzero(labels == 1)[:size // 2]]).reshape(-1)
    q15.require(len(indexes) == size, 'Insufficient classes for replay preflight')
    probe, targets = rows.iloc[indexes], torch.tensor(labels[indexes], device='cuda', dtype=torch.float32)
    rng = base.rng_state()
    parameters = list(adapters.parameters())
    sequences, _ = q15_pooling.canonical_sequences(base.pairs_from(probe))
    torch.cuda.synchronize(); tick = time.perf_counter()
    cached = replay.extract_prefix(model, tokenizer, probe)
    torch.cuda.synchronize(); prefix_seconds = time.perf_counter() - tick
    cache_sha256 = hashlib.sha256(cached.view(torch.uint8).numpy().tobytes()).hexdigest()
    record = {'probe_split': 'train', 'probe_keys': probe.variant_key.tolist(),
              'probe_labels': labels[indexes].tolist(), 'frozen_sha256': frozen, 'states': {}}
    try:
        for phase in ['zero', 'after_update']:
            record['states'][phase] = {}
            for objective in q15.CONFIG['objectives']:
                full = base.encode_batch(model, tokenizer, sequences, gradients=True).reshape(size, 2, q11.CONFIG['hidden_size'])
                raw = previous.features_from_pooled(full[:, 0], full[:, 1] - full[:, 0])
                loss = q15_objective.compute_loss(logits(head, mean, scale, raw), targets,
                    objective=objective, class_weights=weights)
                full_grads = [g.detach().clone() for g in torch.autograd.grad(loss.loss, parameters)]
                pooled_expected, raw_expected, loss_expected = full.detach(), raw.detach(), loss.loss.detach()
                del full, raw, loss
                pooled = replay.replay_pooled(model, cached)
                raw = previous.features_from_pooled(pooled[:, 0], pooled[:, 1] - pooled[:, 0])
                loss = q15_objective.compute_loss(logits(head, mean, scale, raw), targets,
                    objective=objective, class_weights=weights)
                replay_grads = torch.autograd.grad(loss.loss, parameters)
                measured = {'pooled': assert_equivalent(pooled, pooled_expected, phase + '/pooled'),
                            'features': assert_equivalent(raw, raw_expected, phase + '/features'),
                            'loss': assert_equivalent(loss.loss, loss_expected, phase + '/loss'),
                            'gradients': {name: assert_equivalent(a, b, phase + '/' + name, gradient=True)
                                for (name, _), a, b in zip(adapters.named_parameters(), replay_grads, full_grads)}}
                record['states'][phase][objective] = measured
                del pooled, raw, loss, replay_grads, full_grads
            if phase == 'zero':
                probe_arm = Arm(adapters, 'weighted_bce', peak_lr)
                probe_arm.activate()
                loss = q15_objective.compute_loss(logits(head, mean, scale, replay.replay_features(model, cached)),
                    targets, objective='weighted_bce', class_weights=weights)
                record['probe_update'] = probe_arm.update(loss, 1)
                q15.require(any(torch.count_nonzero(p).item() and not torch.count_nonzero(initial[name]).item()
                                for name, p in adapters.named_parameters()),
                            'Preflight did not produce a real nonzero adapter update')
                del probe_arm, loss
        previous.restore_adapters(adapters, initial)
        calibration = {}
        for objective in q15.CONFIG['objectives']:
            previous.restore_adapters(adapters, initial)
            probe_arm = Arm(adapters, objective, peak_lr)
            times = []
            for step in range(1, 4):
                torch.cuda.synchronize(); tick = time.perf_counter()
                probe_arm.activate()
                loss = q15_objective.compute_loss(logits(head, mean, scale, replay.replay_features(model, cached)),
                    targets, objective=objective, class_weights=weights)
                probe_arm.update(loss, step)
                torch.cuda.synchronize(); times.append(time.perf_counter() - tick)
            calibration[objective] = max(times)
            del probe_arm, loss
        torch.cuda.synchronize(); tick = time.perf_counter()
        previous.features(model, tokenizer, probe)
        torch.cuda.synchronize(); full_seconds = time.perf_counter() - tick
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.no_grad():
            replay.replay_features(model, cached)
        torch.cuda.synchronize(); tail_eval_seconds = time.perf_counter() - tick
        calibration.update(prefix_seconds_per_variant=prefix_seconds / size,
            full_seconds_per_variant=full_seconds / size, tail_eval_seconds_per_variant=tail_eval_seconds / size,
            paired_update_seconds=sum(calibration.values()))
        record.update(prefix_replay_verified=True, replay_gradient_verified=True)
    finally:
        previous.restore_adapters(adapters, initial)
        base.restore_rng(rng)
    q15.require(base.parameter_hash(model, frozen_only=True) == frozen, 'Preflight changed original backbone')
    q15.require(hashlib.sha256(cached.view(torch.uint8).numpy().tobytes()).hexdigest() == cache_sha256,
                'Tail replay mutated its frozen prefix cache')
    record.update(prefix_cache_unchanged=True, prefix_cache_sha256=cache_sha256)
    q1.write_json(q15.OUTPUT / 'preflight.json', record)
    q1.write_json(q15.OUTPUT / 'calibration.json', calibration)
    print('Training-only full/replay values and all adapter gradients verified at zero and nonzero adapters.', flush=True)
    return calibration


def reserve_seconds(calibration):
    margin = q15.CONFIG['validation_margin']
    return margin * (2 * q15.CONFIG['validation_variants'] * calibration['tail_eval_seconds_per_variant']
                    + 2 * q15.CONFIG['reload_variants'] * calibration['full_seconds_per_variant']) + q15.CONFIG['report_seconds']


def cache_rows(model, tokenizer, rows, budget, calibration, reserve):
    size = q15.CONFIG['microbatch_variants']
    if not budget.can_work(min(size, len(rows)) * calibration['prefix_seconds_per_variant'] * 1.3, reserve):
        return None
    result = torch.empty((len(rows), 2, 512, q11.CONFIG['hidden_size']), dtype=torch.bfloat16, device='cpu')
    for offset in range(0, len(rows), size):
        count = min(size, len(rows) - offset)
        if not budget.can_work(count * calibration['prefix_seconds_per_variant'] * 1.3, reserve):
            return None
        result[offset:offset + count].copy_(replay.extract_prefix(model, tokenizer, rows.iloc[offset:offset + count]))
    return result


def cached_raw(model, cache):
    blocks = []
    with torch.no_grad():
        for offset in range(0, len(cache), q15.CONFIG['microbatch_variants']):
            blocks.append(replay.replay_features(model, cache[offset:offset + q15.CONFIG['microbatch_variants']]).cpu())
    return torch.cat(blocks)


def score_cache(model, cache, head, mean, scale):
    # Score all feature rows together, matching the inherited classifier GEMM.
    with torch.no_grad():
        return logits(head, mean, scale, cached_raw(model, cache).to(mean.device)).cpu().numpy()


def checkpoint(identity, adapter_state, head, mean, scale, progress, control_scores, lora_scores, arm=None):
    state = {'format': 'q15-cached-v1', 'identity': identity, 'adapters': adapter_state,
             'head': head.state_dict(), 'control': head.state_dict(), 'mean': mean, 'scale': scale,
             'strength': 0., 'progress': progress, 'rng': base.rng_state(),
             'control_scores': torch.from_numpy(control_scores),
             'lora_scores': None if lora_scores is None else torch.from_numpy(lora_scores)}
    if arm is not None:
        state.update(masters=arm.masters, optimizer=arm.optimizer.state_dict())
    return base.cpu_state(state)


def train(model, tokenizer, adapters, initial, head, mean, scale, heads, rows, labels,
          validation_cache, control_scores, identity, protocol, budget, calibration):
    previous.restore_adapters(adapters, initial)
    arms = {name: Arm(adapters, name, protocol['adapter_lr']) for name in q15.CONFIG['objectives']}
    control = q15.metric(labels['validation'], control_scores)
    histories = {name: {'trial': name, 'adapter_lr': protocol['adapter_lr'], 'steps': 0, 'attempted_steps': 0,
        'control_steps': 0, 'evaluations': [], 'loss_history': [], 'skipped_batches': [],
        'best_lora': None, 'best_control': control, 'examples_processed': 0, 'examples_used': 0}
        for name in arms}
    exposures = np.zeros(len(rows['train']), dtype=np.int64)
    weights = heads['class_weights'].to(mean.device)
    targets = torch.tensor(labels['train'], dtype=torch.float32, device=mean.device)
    global_history = {'trials': histories, 'completed_chunks': 0, 'attempted_steps': 0,
                      'evaluation_cadence': 'initial, every completed chunk, and final partial chunk',
                      'chunk_schedule': [], 'stop_reason': None}
    reserve = reserve_seconds(calibration)
    size = q15.CONFIG['microbatch_variants']

    def save_history():
        global_history.update(actual_unique_training_variants=int(np.count_nonzero(exposures)),
                              examples_per_arm=int(exposures.sum()))
        for name in arms:
            q1.write_json(q15.OUTPUT / name / 'history.json', histories[name])
        q1.write_json(q15.OUTPUT / 'training_history.json', global_history)

    def evaluate(initial_evaluation=False):
        torch.cuda.synchronize(); tick = time.perf_counter()
        for name, arm in arms.items():
            arm.activate()
            verify_fixed(head, mean, scale, heads)
            scores = control_scores.copy() if initial_evaluation else score_cache(model, validation_cache, head, mean, scale)
            points = q15.metric(labels['validation'], scores)
            history = histories[name]
            history.update(steps=arm.steps, attempted_steps=arm.attempted_steps)
            history['evaluations'].append({'steps': arm.steps, 'attempted_steps': arm.attempted_steps,
                'elapsed_seconds': budget.elapsed(), 'lora': points, 'control': control,
                'examples_processed': history['examples_processed'],
                'unique_training_variants': int(np.count_nonzero(exposures))})
            if history['best_lora'] is None or q15.better(points, history['best_lora']):
                history['best_lora'] = points
                previous.save_state(q15.OUTPUT / name / 'best_lora.pt', checkpoint(identity, arm.state(), head,
                    mean, scale, history, control_scores, scores))
            previous.save_state(q15.OUTPUT / name / 'last_checkpoint.pt', checkpoint(identity, arm.state(), head,
                mean, scale, history, control_scores, scores, arm))
            print(f'{name} attempted {arm.attempted_steps}, updates {arm.steps}: {points}; {budget.elapsed()/60:.1f} min', flush=True)
        torch.cuda.synchronize()
        if not initial_evaluation:
            calibration['tail_eval_seconds_per_variant'] = max(calibration['tail_eval_seconds_per_variant'],
                (time.perf_counter() - tick) / (2 * len(validation_cache)))
        save_history()

    for name in arms:
        (q15.OUTPUT / name).mkdir(exist_ok=True)
    baseline = checkpoint(identity, initial, head, mean, scale, {'steps': 0, 'attempted_steps': 0}, control_scores, control_scores)
    previous.save_state(q15.OUTPUT / 'best_control.pt', baseline)
    previous.save_state(q15.OUTPUT / 'heads.pt', baseline)
    for name in arms:
        previous.save_state(q15.OUTPUT / name / 'best_control.pt', baseline)
    evaluate(initial_evaluation=True)
    for number, indexes, passes in chunk_orders(len(exposures), q15.CONFIG['chunk_variants'], q15.CONFIG['passes_per_chunk'], q15.CONFIG['seed']):
        reserve = reserve_seconds(calibration)
        extraction = len(indexes) * calibration['prefix_seconds_per_variant'] * 1.3
        if not budget.can_work(extraction + calibration['paired_update_seconds'] * 1.3, reserve):
            global_history['stop_reason'] = 'budget_before_next_chunk'
            break
        tick = time.perf_counter()
        cache = cache_rows(model, tokenizer, rows['train'].iloc[indexes], budget, calibration, reserve)
        if cache is None:
            global_history['stop_reason'] = 'budget_during_chunk_extraction'
            break
        calibration['prefix_seconds_per_variant'] = max(calibration['prefix_seconds_per_variant'],
                                                        (time.perf_counter() - tick) / len(indexes))
        global_history['chunk_schedule'].append({'chunk': number, 'variant_keys': rows['train'].iloc[indexes].variant_key.tolist(),
            'pass_order_sha256': [hashlib.sha256(order.tobytes()).hexdigest() for order in passes]})
        stopped = False
        for pass_number, order in enumerate(passes):
            for offset in range(0, len(order), size):
                if (global_history['attempted_steps'] >= q15.CONFIG['max_attempted_steps']
                        or not budget.can_work(calibration['paired_update_seconds'] * 1.3, reserve)):
                    stopped = True
                    global_history['stop_reason'] = 'budget_or_registered_attempt_limit'
                    break
                local = order[offset:offset + size]
                selected = indexes[local]
                keys = rows['train'].iloc[selected].variant_key.tolist()
                # One immutable CPU batch is presented to both objectives at the same cursor.
                batch = cache[torch.from_numpy(local)]
                attempt = global_history['attempted_steps'] + 1
                torch.cuda.synchronize(); tick = time.perf_counter()
                for name, arm in arms.items():
                    arm.activate()
                    loss = q15_objective.compute_loss(logits(head, mean, scale, replay.replay_features(model, batch)),
                        targets[selected], objective=name, class_weights=weights)
                    diagnostic = arm.update(loss, attempt)
                    history = histories[name]
                    history['loss_history'].append(diagnostic)
                    history['examples_processed'] += len(selected)
                    history['examples_used'] += len(selected) if loss.has_signal else 0
                    if not loss.has_signal:
                        history['skipped_batches'].append({'attempted_steps': attempt, 'variant_keys': keys,
                            'positives': loss.positives, 'negatives': loss.negatives})
                    history.update(steps=arm.steps, attempted_steps=arm.attempted_steps,
                                   cursor={'chunk': number, 'pass': pass_number, 'offset': offset + len(local)})
                    del loss
                torch.cuda.synchronize()
                calibration['paired_update_seconds'] = max(calibration['paired_update_seconds'], time.perf_counter() - tick)
                exposures[selected] += 1
                global_history['attempted_steps'] = attempt
                q15.require(len({arm.attempted_steps for arm in arms.values()}) == 1, 'Paired objectives diverged')
                if attempt % q15.CONFIG['checkpoint_steps'] == 0:
                    for name, arm in arms.items():
                        previous.save_state(q15.OUTPUT / name / 'last_checkpoint.pt', checkpoint(identity, arm.state(), head,
                            mean, scale, histories[name], control_scores, None, arm))
                    save_history()
                del batch
            if stopped:
                break
        del cache
        if not stopped:
            global_history['completed_chunks'] += 1
        evaluate()
        if stopped:
            break
    global_history['stop_reason'] = global_history['stop_reason'] or 'registered_schedule_complete'
    q15.require(all(arm.attempted_steps == global_history['attempted_steps'] for arm in arms.values()), 'Unmatched training cursors')
    for name, arm in arms.items():
        histories[name].update(head_fixed_verified=True, scaler_fixed_verified=True,
                               skipped_ranking_batches=len(histories[name]['skipped_batches']))
        q15.require(histories[name]['examples_processed'] == int(exposures.sum()), 'Unmatched example exposures')
    verify_fixed(head, mean, scale, heads)
    pd.DataFrame({'variant_key': rows['train'].variant_key, 'exposures_per_arm': exposures}).to_csv(
        q15.OUTPUT / 'training_exposures.csv', index=False)
    save_history()
    q1.write_json(q15.OUTPUT / 'calibration.json', calibration)
    return global_history


def finish(model, tokenizer, adapters, initial, head, mean, scale, heads, rows, labels, groups,
           identity, protocol, history, budget, frozen):
    trials = history['trials']
    name = max(trials, key=lambda key: (trials[key]['best_lora']['auroc'], trials[key]['best_lora']['average_precision']))
    states = {key: load_state(q15.OUTPUT / key / 'best_lora.pt', identity) for key in trials}
    best = states[name]
    control = load_state(q15.OUTPUT / 'best_control.pt', identity)
    scores = {key: state['lora_scores'].numpy() for key, state in states.items()}
    scores.update(lora=best['lora_scores'].numpy(), matched_control=control['control_scores'].numpy(),
                  best_control=control['control_scores'].numpy(), strongest_control=control['control_scores'].numpy())
    parent = pd.read_csv(q14.OUTPUT / 'validation_predictions.csv')
    q15.require(parent.variant_key.tolist() == rows['validation'].variant_key.tolist()
                and np.array_equal(parent.label.to_numpy(), labels['validation'])
                and np.array_equal(parent.component.to_numpy(), groups['validation']),
                'Q14 comparison membership changed')
    scores.update(q14_control=parent.best_control.to_numpy(), q13_control=parent.q13_control.to_numpy(),
                  q12_control=parent.q12_control.to_numpy())
    points = {key: q15.metric(labels['validation'], value) for key, value in scores.items()}
    promoted = q15.promote(points['lora'], q15.promotion_control(points['strongest_control'], protocol))
    selected = dict(best if promoted else control, selected_model='lora' if promoted else 'frozen_control',
                    selected_trial=name if promoted else protocol['initialization_source'])
    previous.save_state(q15.OUTPUT / 'selected_model.pt', selected)
    scores['selected'] = scores['lora'] if promoted else scores['strongest_control']
    n = q15.CONFIG['reload_variants']
    # Both checks reload from disk after erasing adapters, head and scaler, and use FULL model forwards.
    for path, expected in [(q15.OUTPUT / name / 'best_lora.pt', scores['lora']),
                           (q15.OUTPUT / 'selected_model.pt', scores['selected'])]:
        with torch.no_grad():
            for parameter in list(adapters.parameters()) + list(head.parameters()):
                parameter.zero_()
            mean.zero_(); scale.zero_()
        state = load_state(path, identity)
        previous.restore_adapters(adapters, state['adapters'])
        head.load_state_dict(state['head'])
        mean.copy_(state['mean']); scale.copy_(state['scale'])
        actual = q14_backend.predict(model, tokenizer, head, mean, scale, rows['validation'].iloc[:n])
        q15.require(np.allclose(actual, expected[:n], atol=2e-4, rtol=2e-4), 'Fresh full-forward checkpoint predictions differ')
        verify_fixed(head, mean, scale, heads)
    q15.require(base.parameter_hash(model, frozen_only=True) == frozen, 'Original backbone changed')
    q15.require(q15.verify_protocol() == identity, 'Experiment provenance changed during execution')
    pd.DataFrame({'variant_key': rows['validation'].variant_key, 'label': labels['validation'],
                  'component': groups['validation'], **scores}).to_csv(q15.OUTPUT / 'validation_predictions.csv', index=False)
    metrics, paired = q15.intervals(labels['validation'], scores, groups['validation'])
    files = ['protocol.json', 'input_checks.json', 'execution.json', 'heads.pt', 'head_results.json', 'preflight.json',
             'calibration.json', 'training_history.json', 'training_exposures.csv', 'validation_predictions.csv',
             'best_control.pt', 'selected_model.pt']
    files += [f'{key}/{file}' for key in trials for file in ['best_lora.pt', 'best_control.pt', 'last_checkpoint.pt', 'history.json']]
    result = {'status': 'complete', 'scope': 'sampled_validation', 'identity': identity, 'metrics': metrics,
        'paired_lora_minus_matched_control': paired, 'promotion_passed': promoted,
        'selected_model': selected['selected_model'], 'selected_trial': selected['selected_trial'],
        'best_lora_trial': name, 'best_steps': best['progress']['steps'],
        'best_attempted_steps': best['progress']['attempted_steps'],
        'strongest_control_source': 'best_control', 'initialization_source': protocol['initialization_source'],
        'attempted_steps': history['attempted_steps'], 'completed_chunks': history['completed_chunks'],
        'actual_unique_training_variants': history['actual_unique_training_variants'],
        'examples_per_arm': history['examples_per_arm'], 'training_variants': len(rows['train']),
        'validation_variants': len(rows['validation']), 'full_validation_variants': 17927,
        'stop_reason': history['stop_reason'], 'reloaded_predictions_verified': True, 'frozen_unchanged': True,
        'head_fixed_verified': True, 'scaler_fixed_verified': True, 'control_optimizer_steps': 0,
        'matched_training_verified': True, 'prefix_replay_verified': True, 'replay_gradient_verified': True,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'artifacts': {file: q1.digest_file(q15.OUTPUT / file) for file in files}, 'limitations': q15.LIMITATIONS}
    result.update(seconds=budget.elapsed(), within_one_hour=budget.elapsed() <= q15.CONFIG['total_seconds'])
    q1.write_json(q15.OUTPUT / 'metrics.json', result, frozen=True)
    print(f'Completed Q15: promotion={promoted}; selected {selected["selected_model"]}; {metrics["selected"]}', flush=True)


def explore():
    identity = q15.verify_protocol()
    protocol = q1.read_json(q15.OUTPUT / 'protocol.json')
    budget = q15.Budget(q1.read_json(q15.OUTPUT / 'execution.json')['started_utc'])
    rows, labels, groups, _ = q15.inputs()
    q14.verified_results()
    heads = load_head(protocol, labels)
    raw_expected = q14_backend.cached_features({'validation': rows['validation']})['validation']
    q15.require(budget.can_work(120., q15.CONFIG['report_seconds']), 'Initialization exhausted budget')
    model, tokenizer = base.load_backbone()
    try:
        config = q15.CONFIG['lora']
        adapters = q14_adapters.attach(model, blocks=tuple(config['blocks']), rank=config['rank'], alpha=config['alpha'], seed=q15.CONFIG['seed'])
        model.eval()
        initial = previous.adapter_state(adapters)
        frozen = base.parameter_hash(model, frozen_only=True)
        q15.require(frozen == q1.read_json(q14.OUTPUT / 'preflight.json')['frozen_sha256'], 'Original backbone differs from Q14')
        head = previous.new_head(q15.CONFIG['feature_dimension']).requires_grad_(False)
        head.load_state_dict({name: heads[name] for name in ['weight', 'bias']})
        mean, scale = heads['mean'].cuda().clone(), heads['scale'].cuda().clone()
        calibration = preflight(model, tokenizer, adapters, initial, head, mean, scale, heads['class_weights'].cuda(),
            rows['train'], labels['train'], protocol['adapter_lr'], frozen)
        verify_fixed(head, mean, scale, heads)
        cache = cache_rows(model, tokenizer, rows['validation'], budget, calibration, reserve_seconds(calibration))
        q15.require(cache is not None, 'Budget exhausted before complete validation-prefix verification')
        raw = cached_raw(model, cache)
        equivalence = assert_equivalent(raw, torch.from_numpy(raw_expected), 'all validation cached features')
        with torch.no_grad():
            control_scores = logits(head, mean, scale, raw.cuda()).cpu().numpy()
        q15.require(np.allclose(control_scores, heads['initial_control_scores'].numpy(), atol=2e-4, rtol=2e-4),
                    'Zero-adapter replay control differs from inherited scores')
        measured = q15.metric(labels['validation'], control_scores)
        q15.require(all(np.isclose(measured[key], protocol['initial_control_metrics'][key]['value'], atol=1e-12, rtol=0.)
                        for key in measured), 'Zero-adapter replay control metrics differ from inherited control')
        check = q1.read_json(q15.OUTPUT / 'preflight.json')
        check.update(validation_cache_variants=len(cache), validation_cache_equivalence=equivalence,
                     validation_cache_bytes=cache.numel() * cache.element_size(), initial_control_metrics=measured)
        q1.write_json(q15.OUTPUT / 'preflight.json', check)
        del raw, raw_expected
        history = train(model, tokenizer, adapters, initial, head, mean, scale, heads, rows, labels,
                        cache, control_scores, identity, protocol, budget, calibration)
        del cache
        finish(model, tokenizer, adapters, initial, head, mean, scale, heads, rows, labels, groups,
               identity, protocol, history, budget, frozen)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['explore'])
    parser.parse_args()
    explore()
