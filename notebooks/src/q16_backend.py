"""Larger Q14 continuation, with a fixed classifier and fresh full validation."""

import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import q1, q11_backend as base, q12, q12_backend as previous
from . import q14, q14_backend as parent_backend, q14_adapters, q16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def better(candidate, current):
    return (candidate['auroc'], candidate['average_precision']) > (current['auroc'], current['average_precision'])


def retain_continuation(candidate, parent):
    return candidate['auroc'] > parent['auroc'] and candidate['average_precision'] >= parent['average_precision']


def schedule(step, maximum=None):
    maximum = q16.CONFIG['max_steps'] if maximum is None else maximum
    require(1 <= step <= maximum, 'Schedule step is outside the frozen plan')
    warmup = min(q16.CONFIG['warmup_steps'], maximum)
    if step <= warmup:
        return step / warmup
    fraction = (step - warmup) / (maximum - warmup)
    floor = q16.CONFIG['minimum_lr_ratio']
    return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2


def epoch_orders(count, epochs, seed):
    require(count > 0 and epochs > 0, 'Training plan must contain examples and epochs')
    rng = np.random.default_rng(seed)
    return [rng.permutation(count) for _ in range(epochs)]


def batches(orders, size):
    require(size > 0, 'Batch size must be positive')
    for epoch, order in enumerate(orders, 1):
        for offset in range(0, len(order), size):
            yield epoch, offset, order[offset:offset + size]


def order_hashes(orders):
    return [hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest() for order in orders]


def validate_progress(progress, orders, size):
    plan = list(batches(orders, size))
    steps = progress['steps']
    require(isinstance(steps, int) and 0 <= steps <= len(plan), 'Invalid resume step count')
    require(progress['epoch_order_sha256'] == order_hashes(orders), 'Checkpoint epoch order changed')
    exposures = sum(len(batch) for _, _, batch in plan[:steps])
    require(progress['examples_seen'] == exposures, 'Checkpoint exposure count disagrees with cursor')
    if steps:
        epoch, offset, chosen = plan[steps - 1]
        require(progress['epoch'] == epoch and progress['offset'] == offset + len(chosen),
                'Checkpoint epoch or offset is not at the recorded optimizer boundary')
    else:
        require(progress['epoch'] == 1 and progress['offset'] == 0, 'Invalid initial cursor')
    require(progress['unique_variants_seen'] == min(exposures, len(orders[0])),
            'Checkpoint unique-example count is inconsistent')
    return plan


def load_state(path, identity):
    with Path(path).open('rb') as stream:
        checksum, payload = stream.readline(65).strip().decode(), stream.read()
    require(hashlib.sha256(payload).hexdigest() == checksum, 'Q16 checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    require(state.get('format') == 'q16-continuation-v1' and state.get('identity') == identity,
            'Q16 checkpoint identity or format mismatch')

    def finite(value):
        if isinstance(value, torch.Tensor):
            require(torch.isfinite(value).all(), 'Q16 checkpoint contains a nonfinite tensor')
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
    finite(state)
    return state


def checkpoint(identity, adapters, head, mean, scale, progress, optimizer=None, scores=None):
    state = {'format': 'q16-continuation-v1', 'identity': identity,
             'adapters': previous.adapter_state(adapters), 'head': base.cpu_state(head.state_dict()),
             'control': base.cpu_state(head.state_dict()), 'mean': mean, 'scale': scale,
             'progress': progress, 'rng': base.rng_state(), 'deadline_utc': q16.deadline()}
    if optimizer is not None:
        state.update(masters=optimizer.masters, optimizer=optimizer.optimizer.state_dict())
    if scores is not None:
        state['lora_scores'] = torch.as_tensor(scores, dtype=torch.float32)
    return base.cpu_state(state)


def restore_optimizer(optimizer, state):
    require(len(state['masters']) == len(optimizer.masters), 'Optimizer master count changed')
    with torch.no_grad():
        for (name, deployed), master, saved in zip(optimizer.named, optimizer.masters, state['masters']):
            require(master.shape == saved.shape, f'Optimizer master shape changed: {name}')
            master.copy_(saved)
            require(torch.equal(master.to(dtype=deployed.dtype), deployed),
                    f'Optimizer master does not reproduce deployed adapter: {name}')
    optimizer.optimizer.load_state_dict(state['optimizer'])


def verify_optimizer_config():
    for key in ['head_lr', 'adapter_weight_decay', 'clip_grad', 'betas', 'epsilon']:
        require(q16.CONFIG[key] == q14.CONFIG[key], f'Q14 optimizer reuse configuration mismatch: {key}')
    require(all(q14.CONFIG['lora'].get(key) == value for key, value in q16.CONFIG['lora'].items()),
            'Q16 adapter configuration differs from the inherited adapter loader')
    for key in ['feature_dimension', 'context_bp', 'seed']:
        require(q16.CONFIG[key] == q12.CONFIG[key], f'Inherited feature/bootstrap configuration mismatch: {key}')


def verify_fixed(head, mean, scale, parent):
    require(all(not parameter.requires_grad and parameter.grad is None for parameter in head.parameters()),
            'Fixed classifier unexpectedly has gradients')
    require(all(torch.equal(value.cpu(), parent['head'][name]) for name, value in head.state_dict().items()),
            'Fixed classifier parameters changed')
    require(torch.equal(mean.cpu(), parent['mean']) and torch.equal(scale.cpu(), parent['scale']),
            'Fixed classifier scaler changed')


def make_head(state):
    dimension = q16.CONFIG['feature_dimension']
    for name, shape in [('weight', (1, dimension)), ('bias', (1,))]:
        value = state['head'][name]
        require(value.dtype == torch.float32 and value.shape == shape and torch.isfinite(value).all(),
                f'Invalid inherited head {name}')
    for name in ['mean', 'scale']:
        value = state[name]
        require(value.dtype == torch.float32 and value.shape == (dimension,) and torch.isfinite(value).all(),
                f'Invalid inherited scaler {name}')
    require((state['scale'] > 0).all(), 'Inherited scale must be positive')
    head = previous.new_head(dimension)
    head.load_state_dict(state['head'])
    head.eval().requires_grad_(False)
    return head, state['mean'].cuda().clone(), state['scale'].cuda().clone()


class Budget:
    def __init__(self, started):
        self.started = started
        self.prediction_seconds_per_variant = 0.
        self.training_seconds_per_step = 5.

    def elapsed(self):
        return time.time() - self.started

    def remaining(self):
        return q16.deadline() - time.time()

    def reserve(self, full_count, selection_count):
        observed = self.prediction_seconds_per_variant
        measured = observed * (3 * full_count + selection_count + 3 * q16.CONFIG['reload_variants']) * 1.25
        return max(q16.CONFIG['full_validation_reserve_seconds'], measured) + q16.CONFIG['report_seconds']

    def can_train(self, full_count, selection_count):
        return self.remaining() > self.reserve(full_count, selection_count) + self.training_seconds_per_step * 1.5


def predict(model, tokenizer, head, mean, scale, rows, budget, label):
    chunks = []
    started = time.perf_counter()
    size = q16.CONFIG['microbatch_variants']
    for offset in range(0, len(rows), size):
        require(budget.remaining() > 0, 'Frozen four-hour deadline reached during inference')
        raw = previous.features(model, tokenizer, rows.iloc[offset:offset + size])
        chunks.append(previous.score_head(head, (raw - mean) / scale))
        count = min(offset + size, len(rows))
        if count % 512 == 0 or count == len(rows):
            print(f'{label}: {count}/{len(rows)} variants; {budget.remaining()/60:.1f} minutes remain', flush=True)
    seconds = time.perf_counter() - started
    budget.prediction_seconds_per_variant = max(budget.prediction_seconds_per_variant, seconds / len(rows))
    return np.concatenate(chunks)


def assert_scores(actual, expected, label):
    expected = np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, atol=2e-4, rtol=2e-4), f'{label} predictions do not reproduce the frozen source')


def feature_outputs(raw, head, mean, scale):
    standardized = (raw - mean) / scale
    return {'features': raw.detach().cpu(), 'standardized': standardized.detach().cpu(),
            'logits': torch.as_tensor(previous.score_head(head, standardized))}


def comparison_statistics(actual, expected):
    """Record errors separately for Q12 features, inherited scaling and head logits."""
    require(set(actual) == set(expected) == {'features', 'standardized', 'logits'},
            'Batch diagnostic representations are incomplete')
    result = {}
    for name in actual:
        left, right = actual[name].double(), expected[name].double()
        require(left.shape == right.shape and torch.isfinite(left).all() and torch.isfinite(right).all(),
                'Invalid batch diagnostic values')
        error = left - right
        rms = error.square().mean().sqrt()
        result[name] = {'max_abs_difference': float(error.abs().max()), 'rms_difference': float(rms),
            'max_relative_to_reference': float((error.abs() / right.abs().clamp_min(1e-12)).max()),
            'max_symmetric_relative': float((error.abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)).max()),
            'reference_rms': float(right.square().mean().sqrt()),
            'relative_rms_difference': float(rms / right.square().mean().sqrt().clamp_min(1e-12)),
            'allclose_2e_4': bool(torch.allclose(left, right, atol=2e-4, rtol=2e-4)),
            'close_fraction_2e_4': float(torch.isclose(left, right, atol=2e-4, rtol=2e-4).double().mean()),
            'exact_fraction': float((left == right).double().mean())}
    return result


def require_operational_equivalence(statistics, label):
    require(set(statistics) == {'features', 'standardized', 'logits'}
            and all(value['allclose_2e_4'] for value in statistics.values()),
            f'Q16 operational batch equivalence failed: {label}')


def training_preflight(model, tokenizer, adapters, parent, head, mean, scale, rows, labels, class_weights):
    """Check operational batches; quantify the unsupported serial-one kernel separately."""
    original, rng = previous.adapter_state(adapters), base.rng_state()
    require(q16.CONFIG['microbatch_variants'] == 32 and len(rows) >= 64, 'Q16 preflight requires operational batch 32')
    probe = rows.iloc[:32]
    before = previous.features(model, tokenizer, probe)
    before_outputs = feature_outputs(before, head, mean, scale)
    previous.features(model, tokenizer, rows.iloc[32:64])
    repeated = previous.features(model, tokenizer, probe)
    require(torch.equal(before, repeated), 'Q16 repeatability failed')
    comparisons = {'batch32_repeat': comparison_statistics(feature_outputs(repeated, head, mean, scale), before_outputs)}
    reverse = probe.copy()
    for name in ['ref_sequence', 'alt_sequence']:
        reverse[name] = reverse[name].map(q1.reverse_complement)
    reversed_values = previous.features(model, tokenizer, reverse)
    require(torch.equal(before, reversed_values), 'Q16 strand invariance failed')
    comparisons['batch32_reverse_complement'] = comparison_statistics(
        feature_outputs(reversed_values, head, mean, scale), before_outputs)
    permutation = np.random.default_rng(q16.CONFIG['seed']).permutation(32)
    permuted = feature_outputs(previous.features(model, tokenizer, probe.iloc[permutation]), head, mean, scale)
    comparisons['batch32_permutation'] = comparison_statistics(
        {key: value[np.argsort(permutation)] for key, value in permuted.items()}, before_outputs)
    for size in [4, 8]:
        small = feature_outputs(previous.features(model, tokenizer, probe.iloc[:size]), head, mean, scale)
        comparisons[f'batch32_vs_batch{size}'] = comparison_statistics(
            {key: value[:size] for key, value in before_outputs.items()}, small)
        permutation = np.random.default_rng(q16.CONFIG['seed'] + size).permutation(size)
        shuffled = feature_outputs(previous.features(model, tokenizer, probe.iloc[permutation]), head, mean, scale)
        comparisons[f'batch{size}_permutation'] = comparison_statistics(
            {key: value[np.argsort(permutation)] for key, value in shuffled.items()}, small)
    for name, statistics in comparisons.items():
        require_operational_equivalence(statistics, name)
    serial = [feature_outputs(previous.features(model, tokenizer, probe.iloc[i:i+1]), head, mean, scale) for i in range(4)]
    comparisons['batch32_vs_serial1_descriptive'] = comparison_statistics(
        {key: value[:4] for key, value in before_outputs.items()},
        {key: torch.cat([value[key] for value in serial]) for key in before_outputs})
    optimizer = parent_backend.AdapterOptimizer(adapters, head, q16.CONFIG['adapter_lr'])
    targets = torch.as_tensor(labels[:32], dtype=torch.float32, device='cuda')
    weights = torch.as_tensor(class_weights[labels[:32]], dtype=torch.float32, device='cuda')
    maxima = {name: 0. for name in original}
    before_scores = previous.score_head(head, (before - mean) / scale)
    try:
        for _ in range(8):
            raw = previous.features(model, tokenizer, probe, gradients=True)
            loss = parent_backend.head_loss(head, (raw - mean) / scale, targets, weights, 0.)
            require(torch.isfinite(loss), 'Nonfinite preflight loss')
            loss.backward()
            for name, parameter in adapters.named_parameters():
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'Invalid preflight adapter gradient')
                maxima[name] = max(maxima[name], float(parameter.grad.abs().max()))
            optimizer.step(1.)
        after = previous.score_head(head, (previous.features(model, tokenizer, probe) - mean) / scale)
        require(all(maxima.values()) and not np.array_equal(before_scores, after),
                'Continued adapters did not change a fixed-head prediction')
        verify_fixed(head, mean, scale, parent)
        changed = {name: int((parameter.cpu() != original[name]).sum()) for name, parameter in adapters.named_parameters()}
        require(any(changed.values()), 'No deployed adapter elements changed in preflight')
    finally:
        previous.restore_adapters(adapters, original)
        base.restore_rng(rng)
    require(torch.equal(before, previous.features(model, tokenizer, probe)), 'Restoring probe adapters changed features')
    return {'status': 'passed', 'training_variant_keys': probe.variant_key.tolist(),
            'gradient_maxima': maxima, 'changed_adapter_elements': changed,
            'repeatability_verified': True, 'batch_equivalence_verified': True,
            'strand_invariance_verified': True, 'restored_predictions_verified': True,
            'batch_comparisons': comparisons, 'operational_training_batch_variants': 32,
            'operational_epoch_tail_variants': 8, 'gradient_probe_batch_variants': 32,
            'serial1_equivalence_claimed': False,
            'serial1_scope': 'Descriptive training-only sensitivity probe. Serial-one execution is not an equivalent supported feature definition; operational batch32 and batch8 checks retain the original 2e-4 tolerance.',
            'feature_spaces': {'features': 'Q12 unit-RMS reference and difference vectors plus two log magnitudes',
                               'standardized': 'Those 8194 features transformed by the inherited fixed mean and scale',
                               'logits': 'Actual fixed classifier outputs'}}


def train(model, tokenizer, adapters, parent, head, mean, scale, rows, labels, identity,
          budget, sample_parent, sample_frozen, resume):
    count, size = len(rows['train']), q16.CONFIG['microbatch_variants']
    orders = epoch_orders(count, q16.CONFIG['epochs'], q16.CONFIG['seed'])
    require(q16.CONFIG['max_steps'] == math.ceil(count / size) * q16.CONFIG['epochs'], 'Training step plan omits an epoch tail')
    parent_point = q12.metric(labels['validation'], sample_parent)
    frozen_point = q12.metric(labels['validation'], sample_frozen)
    class_weights = count / (2 * np.bincount(labels['train'], minlength=2))
    targets = torch.as_tensor(labels['train'], dtype=torch.float32, device='cuda')
    weights = torch.as_tensor(class_weights[labels['train']], dtype=torch.float32, device='cuda')
    optimizer = parent_backend.AdapterOptimizer(adapters, head, q16.CONFIG['adapter_lr'])
    progress = {'steps': 0, 'epoch': 1, 'offset': 0, 'epochs_completed': 0, 'examples_seen': 0,
                'unique_variants_seen': 0, 'epoch_order_sha256': order_hashes(orders),
                'evaluations': [], 'loss_history': [], 'stop_reason': None,
                'best_continuation': None, 'best_continuation_step': None,
                'selected_model': 'q14_parent', 'selected_step': 0,
                'class_weights': class_weights.tolist(), 'optimizer_reset_from_parent': True}

    def save_last():
        previous.save_state(q16.OUTPUT / 'last_checkpoint.pt',
            checkpoint(identity, adapters, head, mean, scale, progress, optimizer))
        q1.write_json(q16.OUTPUT / 'training_history.json', progress)

    def evaluate(initial=False):
        verify_fixed(head, mean, scale, parent)
        scores = sample_parent.copy() if initial else predict(model, tokenizer, head, mean, scale,
            rows['validation'], budget, f'Selection step {progress["steps"]}')
        point = q12.metric(labels['validation'], scores)
        progress['evaluations'].append({'steps': progress['steps'], 'epoch': progress['epoch'],
            'examples_seen': progress['examples_seen'], 'elapsed_seconds': budget.elapsed(),
            'continuation': point, 'q14_parent': parent_point, 'frozen_control': frozen_point})
        if not initial and (progress['best_continuation'] is None or better(point, progress['best_continuation'])):
            progress['best_continuation'] = point
            progress['best_continuation_step'] = progress['steps']
            previous.save_state(q16.OUTPUT / 'best_continuation.pt',
                checkpoint(identity, adapters, head, mean, scale, progress, scores=scores))
        print(f'Selection step {progress["steps"]}: {point}; parent {parent_point}', flush=True)

    if resume:
        saved = load_state(q16.OUTPUT / 'last_checkpoint.pt', identity)
        require(saved['deadline_utc'] == q16.deadline(), 'Resume cannot reset the frozen deadline')
        progress = saved['progress']
        validate_progress(progress, orders, size)
        previous.restore_adapters(adapters, saved['adapters'])
        head.load_state_dict(saved['head'])
        require(torch.equal(saved['mean'], parent['mean']) and torch.equal(saved['scale'], parent['scale']),
                'Resume scaler differs from parent')
        restore_optimizer(optimizer, saved)
        base.restore_rng(saved['rng'])
        verify_fixed(head, mean, scale, parent)
        progress['stop_reason'] = None
        print(f'Resumed at step {progress["steps"]} with the original deadline.', flush=True)
    else:
        evaluate(initial=True)
        previous.save_state(q16.OUTPUT / 'parent_model.pt',
            checkpoint(identity, adapters, head, mean, scale, progress, scores=sample_parent))
        save_last()
    plan = validate_progress(progress, orders, size)
    for index, (epoch, offset, chosen) in enumerate(plan, 1):
        if index <= progress['steps']:
            continue
        if not budget.can_train(len(rows['full_validation']), len(rows['validation'])):
            progress['stop_reason'] = 'full_validation_deadline_reserve'
            break
        started = time.perf_counter()
        batch = torch.as_tensor(chosen, dtype=torch.long, device='cuda')
        raw = previous.features(model, tokenizer, rows['train'].iloc[chosen], gradients=True)
        loss = parent_backend.head_loss(head, (raw - mean) / scale, targets[batch], weights[batch], 0.)
        require(torch.isfinite(loss), 'Nonfinite continuation loss')
        loss.backward()
        multiplier = schedule(index)
        diagnostic = optimizer.step(multiplier)
        elapsed = time.perf_counter() - started
        budget.training_seconds_per_step = max(elapsed, .9 * budget.training_seconds_per_step)
        exposures = (epoch - 1) * count + offset + len(chosen)
        progress.update(steps=index, epoch=epoch, offset=offset + len(chosen),
            epochs_completed=epoch - 1 + int(offset + len(chosen) == count),
            examples_seen=exposures, unique_variants_seen=min(count, exposures))
        if index % q16.CONFIG['checkpoint_steps'] == 0 or len(chosen) != size:
            progress['loss_history'].append({'steps': index, 'epoch': epoch, 'batch_variants': len(chosen),
                'loss': float(loss.detach()), 'lr_multiplier': multiplier, 'seconds': elapsed, **diagnostic})
            print(f'Q16 step {index}/{len(plan)}, epoch {epoch}, examples {exposures}; '
                  f'{budget.remaining()/60:.1f} minutes remain, reserve '
                  f'{budget.reserve(len(rows["full_validation"]), len(rows["validation"]))/60:.1f}', flush=True)
        if index % q16.CONFIG['evaluation_steps'] == 0:
            evaluate()
        if index % q16.CONFIG['checkpoint_steps'] == 0 or offset + len(chosen) == count:
            save_last()
    if progress['stop_reason'] is None:
        progress['stop_reason'] = 'max_epochs'
    if progress['evaluations'][-1]['steps'] != progress['steps']:
        evaluate()
    require(progress['best_continuation'] is not None, 'Deadline left no completed continuation update')
    chosen_continuation = retain_continuation(progress['best_continuation'], parent_point)
    progress['selected_model'] = 'continuation' if chosen_continuation else 'q14_parent'
    progress['selected_step'] = progress['best_continuation_step'] if chosen_continuation else 0
    selected = load_state(q16.OUTPUT / ('best_continuation.pt' if chosen_continuation else 'parent_model.pt'), identity)
    selected.update(selected_model=progress['selected_model'], selected_steps=progress['selected_step'])
    previous.save_state(q16.OUTPUT / 'selected_model.pt', selected)
    previous.save_state(q16.OUTPUT / 'best_including_parent.pt', selected)
    verify_fixed(head, mean, scale, parent)
    validate_progress(progress, orders, size)
    save_last()
    exposures = np.zeros(count, dtype=int)
    for _, _, chosen in plan[:progress['steps']]:
        exposures[chosen] += 1
    pd.DataFrame({'variant_key': rows['train'].variant_key.to_numpy(), 'label': labels['train'],
                  'exposures': exposures}).to_csv(q16.OUTPUT / 'training_exposures.csv', index=False)
    return progress


def summarize(labels, predictions, groups):
    """Use identical seeded component draws for both prespecified paired effects."""
    aliases = {'lora': predictions['continuation'], 'matched_control': predictions['q14_parent'],
               'frozen_control': predictions['frozen_control']}
    raw, versus_parent = q12.intervals(labels, aliases, groups, q16.CONFIG['bootstrap_repetitions'])
    metrics = {'continuation': raw['lora'], 'q14_parent': raw['matched_control'],
               'frozen_control': raw['frozen_control']}
    selected = next((name for name in ['continuation', 'q14_parent']
                     if np.array_equal(predictions['selected'], predictions[name])), None)
    require(selected is not None, 'Selected predictions differ from both frozen selection candidates')
    metrics['selected'] = metrics[selected]
    _, versus_frozen = q12.intervals(labels,
        {'lora': predictions['continuation'], 'matched_control': predictions['frozen_control']},
        groups, q16.CONFIG['bootstrap_repetitions'])
    return {'metrics': metrics,
            'paired_continuation_minus_q14_parent': versus_parent,
            'paired_continuation_minus_frozen_control': versus_frozen}


def subset(mask, labels, predictions, groups):
    result = {'variants': int(mask.sum()), 'components': int(len(np.unique(groups[mask])))}
    if set(np.unique(labels[mask])) != {0, 1}:
        return dict(result, status='not_available', reason='Subset does not contain both classes')
    return dict(result, status='complete', **summarize(labels[mask],
        {key: values[mask] for key, values in predictions.items()}, groups[mask]))


def finish(model, tokenizer, adapters, zero, parent, head, mean, scale, rows, labels, groups,
           identity, progress, budget, frozen, backend_started, parent_hash):
    best = load_state(q16.OUTPUT / 'best_continuation.pt', identity)
    previous.restore_adapters(adapters, best['adapters'])
    verify_fixed(head, mean, scale, parent)
    full = rows['full_validation']
    predictions = {'continuation': predict(model, tokenizer, head, mean, scale, full, budget, 'Full continuation')}
    previous.restore_adapters(adapters, parent['adapters'])
    predictions['q14_parent'] = predict(model, tokenizer, head, mean, scale, full, budget, 'Full Q14 parent')
    previous.restore_adapters(adapters, zero)
    predictions['frozen_control'] = predict(model, tokenizer, head, mean, scale, full, budget, 'Full frozen control')
    predictions['selected'] = predictions[progress['selected_model']].copy()
    old = pd.read_csv(q14.OUTPUT / 'full/validation_predictions.csv')
    require(old.variant_key.tolist() == full.variant_key.tolist()
            and np.array_equal(old.label, labels['full_validation'])
            and np.array_equal(old.component.astype(str), np.asarray(groups['full_validation']).astype(str)),
            'Inherited full validation membership changed')
    for name, column in [('q14_parent', 'lora'), ('frozen_control', 'strongest_control')]:
        assert_scores(predictions[name], old[column].to_numpy(), name)
    indexes = pd.Series(np.arange(len(full)), index=full.variant_key).loc[rows['validation'].variant_key].to_numpy()
    assert_scores(predictions['continuation'][indexes], best['lora_scores'].numpy(), 'Continuation selection sample')
    assert_scores(predictions['q14_parent'][indexes], parent['lora_scores'].numpy(), 'Parent selection sample')
    assert_scores(predictions['frozen_control'][indexes], parent['control_scores'].numpy(), 'Frozen selection sample')
    n = q16.CONFIG['reload_variants']
    for name in ['continuation', 'q14_parent', 'frozen_control']:
        with torch.no_grad():
            for parameter in list(adapters.parameters()) + list(head.parameters()):
                parameter.zero_()
            mean.zero_(); scale.zero_()
        if name == 'continuation':
            reloaded = load_state(q16.OUTPUT / 'best_continuation.pt', identity)
        else:
            protocol = q1.read_json(q16.OUTPUT / 'protocol.json')
            path = q16.ROOT / protocol['initialization_checkpoint']
            q1.verify_file(path, parent_hash)
            reloaded = parent_backend.load_state(path, q14.verify_protocol())
        previous.restore_adapters(adapters, zero if name == 'frozen_control' else reloaded['adapters'])
        head.load_state_dict(reloaded['head'])
        mean.copy_(reloaded['mean']); scale.copy_(reloaded['scale'])
        actual = predict(model, tokenizer, head, mean, scale, full.iloc[:n], budget, f'{name} reload')
        assert_scores(actual, predictions[name][:n], f'{name} fresh checkpoint reload')
        verify_fixed(head, mean, scale, parent)
    require(base.parameter_hash(model, frozen_only=True) == frozen, 'Original frozen backbone changed')
    require(q16.verify_protocol() == identity, 'Q16 protocol changed during execution')
    q14.verified_results()
    pd.DataFrame({'variant_key': full.variant_key.to_numpy(), 'label': labels['full_validation'],
                  'component': groups['full_validation'], **predictions}).to_csv(q16.OUTPUT / 'validation_predictions.csv', index=False)
    sampled = {key: values[indexes] for key, values in predictions.items()}
    pd.DataFrame({'variant_key': rows['validation'].variant_key.to_numpy(), 'label': labels['validation'],
                  'component': groups['validation'], **sampled}).to_csv(q16.OUTPUT / 'sampled_validation_predictions.csv', index=False)
    q1.write_json(q16.OUTPUT / 'prediction_provenance.json', {'identity': identity,
        'best_continuation_sha256': q1.digest_file(q16.OUTPUT / 'best_continuation.pt'),
        'selected_model_sha256': q1.digest_file(q16.OUTPUT / 'selected_model.pt'),
        'parent_checkpoint_sha256': parent_hash, 'frozen_sha256': frozen,
        'reload_variant_keys': full.variant_key.iloc[:n].tolist(),
        'all_three_models_fresh_full_forward': True, 'parent_full_predictions_reproduced': True,
        'all_selection_predictions_reproduced': True, 'checkpoint_files_reread_after_erasure': True})
    print('Computing full and subset paired component-bootstrap intervals.', flush=True)
    outside = ~full.variant_key.isin(rows['validation'].variant_key).to_numpy()
    unseen = ~np.isin(groups['full_validation'], np.unique(groups['validation']))
    result = {'status': 'complete', 'scope': 'full_validation', 'identity': identity,
        'validation_variants': len(full), 'selection_variants': len(rows['validation']),
        'training_variants': len(rows['train']), 'actual_unique_training_variants': progress['unique_variants_seen'],
        'training_examples_seen': progress['examples_seen'], 'optimizer_steps': progress['steps'],
        'epochs_completed': progress['epochs_completed'], 'stop_reason': progress['stop_reason'],
        'best_continuation_steps': progress['best_continuation_step'], 'selected_steps': progress['selected_step'],
        'selected_model': progress['selected_model'], 'sample_metrics': {key: q12.metric(labels['validation'], values)
            for key, values in sampled.items()}, **summarize(labels['full_validation'], predictions, groups['full_validation']),
        'subsets': {'outside_selection': subset(outside, labels['full_validation'], predictions, groups['full_validation']),
                    'unseen_components': subset(unseen, labels['full_validation'], predictions, groups['full_validation'])},
        'head_fixed_verified': True, 'scaler_fixed_verified': True, 'frozen_unchanged': True,
        'control_optimizer_steps': 0, 'reloaded_predictions_verified': True,
        'parent_predictions_reproduced': True, 'original_selection_scores_reproduced': True,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'deadline_utc': q16.deadline(), 'seconds': budget.elapsed(), 'backend_seconds': time.perf_counter() - backend_started,
        'within_four_hours': time.time() <= q16.deadline()}
    files = ['protocol.json', 'input_checks.json', 'execution.json', 'preflight.json', 'training_history.json',
             'training_exposures.csv', 'best_continuation.pt', 'best_including_parent.pt', 'selected_model.pt',
             'last_checkpoint.pt', 'parent_model.pt', 'validation_predictions.csv',
             'sampled_validation_predictions.csv', 'prediction_provenance.json']
    result['artifacts'] = {name: q1.digest_file(q16.OUTPUT / name) for name in files}
    q1.write_json(q16.OUTPUT / 'metrics.json', result, frozen=True)
    print(f'Q16 complete: selected {result["selected_model"]}; {result["seconds"]/60:.1f} minutes '
          f'from the original start; within deadline={result["within_four_hours"]}', flush=True)
    return result


def run(resume=False):
    backend_started = time.perf_counter()
    identity = q16.verify_protocol()
    protocol = q1.read_json(q16.OUTPUT / 'protocol.json')
    execution = q1.read_json(q16.OUTPUT / 'execution.json')
    budget = Budget(float(execution['started_utc']))
    require(budget.remaining() > q16.CONFIG['full_validation_reserve_seconds'] + q16.CONFIG['report_seconds'],
            'Too little time remains for a new continuation and full validation')
    rows, labels, groups, checks = q16.inputs()
    require(set(rows) == set(labels) == set(groups) == {'train', 'validation', 'full_validation'}, 'Unexpected Q16 input splits')
    for split in rows:
        labels[split], groups[split] = np.asarray(labels[split]), np.asarray(groups[split])
        require(len(rows[split]) == len(labels[split]) == len(groups[split])
                and not rows[split].variant_key.duplicated().any() and set(np.unique(labels[split])) == {0, 1},
                f'Invalid Q16 {split} cohort')
    require(len(rows['train']) == q16.CONFIG['training_variants'] and len(rows['full_validation']) == 17927
            and len(rows['validation']) == 2048, 'Q16 full or selection cohort is incomplete')
    q1.write_json(q16.OUTPUT / 'input_checks.json', checks, frozen=True)
    verify_optimizer_config()
    parent_result = q14.verified_results()
    require(parent_result['selected_model'] == 'lora' and parent_result['promotion_passed'], 'Q16 requires a promoted Q14 adapter')
    path = q16.ROOT / protocol['initialization_checkpoint']
    parent_hash = protocol['initialization_checkpoint_sha256']
    q1.verify_file(path, parent_hash)
    parent = parent_backend.load_state(path, q14.verify_protocol())
    require(parent['selected_model'] == 'lora', 'Initializer is not the selected Q14 adapter')
    require(all(torch.equal(parent['head'][key], parent['control'][key]) for key in ['weight', 'bias']),
            'Q14 parent head differs from its frozen control')
    model, tokenizer = base.load_backbone()
    try:
        config = q14.CONFIG['lora']
        adapters = q14_adapters.attach(model, blocks=tuple(config['blocks']), rank=config['rank'],
                                       alpha=config['alpha'], seed=q16.CONFIG['seed'])
        zero = previous.adapter_state(adapters)
        frozen = base.parameter_hash(model, frozen_only=True)
        require(frozen == q1.read_json(q14.OUTPUT / 'preflight.json')['frozen_sha256'], 'Q16 backbone differs from verified parent')
        previous.restore_adapters(adapters, parent['adapters'])
        head, mean, scale = make_head(parent)
        class_weights = len(rows['train']) / (2 * np.bincount(labels['train'], minlength=2))
        preflight = training_preflight(model, tokenizer, adapters, parent, head, mean, scale,
                                      rows['train'], labels['train'], class_weights)
        preflight.update(identity=identity, frozen_sha256=frozen, class_weights=class_weights.tolist(),
                         parent_checkpoint_sha256=parent_hash)
        if not resume:
            q1.write_json(q16.OUTPUT / 'preflight.json', preflight)
        else:
            q1.write_json(q16.OUTPUT / f'resume_preflight_{int(time.time())}.json', preflight)
        sample_parent = predict(model, tokenizer, head, mean, scale, rows['validation'], budget, 'Initial Q14 sample')
        assert_scores(sample_parent, parent['lora_scores'].numpy(), 'Initial Q14 sample')
        previous.restore_adapters(adapters, zero)
        sample_frozen = predict(model, tokenizer, head, mean, scale, rows['validation'], budget, 'Initial frozen sample')
        assert_scores(sample_frozen, parent['control_scores'].numpy(), 'Initial frozen sample')
        previous.restore_adapters(adapters, parent['adapters'])
        progress = train(model, tokenizer, adapters, parent, head, mean, scale, rows, labels,
                         identity, budget, sample_parent, sample_frozen, resume)
        return finish(model, tokenizer, adapters, zero, parent, head, mean, scale, rows, labels, groups,
                      identity, progress, budget, frozen, backend_started, parent_hash)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run'], nargs='?', default='run')
    parser.add_argument('--resume', action='store_true')
    run(resume=parser.parse_args().resume)
