"""Full-cohort confirmation of a selected Q13/Q14/Q15 LoRA and frozen controls."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import (q1, q11, q11_backend as base, q12, q12_backend as previous,
               q13, q13_backend, q14, q14_backend, q14_adapters,
               lora_validation as validation)


def best_control_trial(history):
    """Select the recorded best frozen head, independently of selected adapters."""
    trials = history.get('trials', history)
    validation.require(bool(trials), 'No recorded frozen control trials')
    for trial in trials.values():
        values = trial.get('best_control', {})
        validation.require(all(key in values and np.isfinite(values[key])
                               and 0 <= values[key] <= 1
                               for key in ('auroc', 'average_precision')),
                           'Invalid recorded frozen control metrics')
    return max(trials, key=lambda name: (trials[name]['best_control']['auroc'],
                                        trials[name]['best_control']['average_precision']))


def checkpoint_loader(question):
    """Import optional Q15 code only when its own experiment is being validated."""
    if question == 'q15':
        from . import q15_backend
        return q15_backend.load_state
    validation.require(question in {'q13', 'q14'}, 'Unknown LoRA checkpoint question')
    return q13_backend.load_state if question == 'q13' else q14_backend.load_state


def load_selected(question):
    module = validation.parent(question)
    result = module.verified_results()
    validation.require(result['promotion_passed'] and result['selected_model'] == 'lora',
                       'Full confirmation requires a selected, promoted LoRA')
    loader = checkpoint_loader(question)
    path = module.OUTPUT / 'selected_model.pt'
    state = loader(path, module.verify_protocol())
    validation.require(state.get('selected_model') == 'lora', 'Selected checkpoint is not LoRA')
    return state, path, result


def load_control(question, source):
    """Load the exact strongest frozen head named by the completed parent run."""
    if source == 'q12_control':
        module, loader = q12, previous.load_state
    elif source == 'q13_control':
        module, loader = q13, q13_backend.load_state
    elif source == 'q14_control':
        module, loader = q14, q14_backend.load_state
    elif source == 'best_control':
        module = validation.parent(question)
        loader = checkpoint_loader(question)
    else:
        raise ValueError(f'Unknown strongest frozen control source: {source}')
    module.verified_results()
    if module is q12:
        path = module.OUTPUT / 'best_control.pt'
    else:
        history = q1.read_json(module.OUTPUT / 'training_history.json')
        path = module.OUTPUT / best_control_trial(history) / 'best_control.pt'
    state = loader(path, module.verify_protocol())
    validation.require('control' in state, 'Frozen control checkpoint has no control head')
    return state, path


def head_spec(state, key, device='cuda'):
    """Validate inherited parameters and scaler; no preprocessing is fitted here."""
    dimension = q12.CONFIG['feature_dimension']
    saved = state[key]
    for name, shape in [('weight', (1, dimension)), ('bias', (1,))]:
        value = saved[name]
        validation.require(isinstance(value, torch.Tensor) and value.shape == shape
                           and torch.isfinite(value).all(), f'Invalid inherited classifier {name}')
    for name in ('mean', 'scale'):
        value = state[name]
        validation.require(isinstance(value, torch.Tensor) and value.shape == (dimension,)
                           and torch.isfinite(value).all(), f'Invalid inherited {name}')
    validation.require((state['scale'] > 0).all(), 'Inherited feature scale must be positive')
    head = torch.nn.Linear(dimension, 1, dtype=torch.float32, device=device)
    head.load_state_dict(saved)
    head.eval().requires_grad_(False)
    return {'head': head, 'mean': state['mean'].to(device=device, dtype=torch.float32).clone(),
            'scale': state['scale'].to(device=device, dtype=torch.float32).clone()}


def score(raw, spec):
    with torch.no_grad():
        return spec['head']((raw - spec['mean']) / spec['scale']).flatten().cpu().numpy()


def predict(model, tokenizer, rows, specs, label):
    """One backbone pass scores all supplied frozen heads/scalers together."""
    chunks = {name: [] for name in specs}
    size = validation.CONFIG['microbatch_variants']
    for offset in range(0, len(rows), size):
        raw = previous.features(model, tokenizer, rows.iloc[offset:offset + size])
        for name, spec in specs.items():
            chunks[name].append(score(raw, spec))
        completed = min(offset + size, len(rows))
        if completed % 512 == 0 or completed == len(rows):
            print(f'{label}: {completed}/{len(rows)} variants', flush=True)
    return {name: np.concatenate(values) for name, values in chunks.items()}


def erase(adapters, specs):
    """Erase every deployed learned tensor before loading checkpoints afresh."""
    with torch.no_grad():
        for parameter in adapters.parameters():
            parameter.zero_()
        for spec in specs.values():
            for parameter in spec['head'].parameters():
                parameter.zero_()
            spec['mean'].zero_()
            spec['scale'].zero_()


def subset_result(mask, labels, predictions, groups):
    labels, groups = labels[mask], groups[mask]
    result = {'variants': int(mask.sum()), 'components': int(len(np.unique(groups)))}
    if set(np.unique(labels)) != {0, 1}:
        return dict(result, status='not_available', reason='Subset does not contain both label classes')
    metrics, paired = validation.intervals(labels, {name: values[mask]
                                                   for name, values in predictions.items()}, groups)
    return dict(result, status='complete', metrics=metrics,
                paired_lora_minus_strongest_control=paired)


def verify_sample_scores(state, key, labels, expected):
    values = state[key]
    validation.require(isinstance(values, torch.Tensor) and values.shape == (len(labels),)
                       and torch.isfinite(values).all(), 'Invalid checkpoint selection-sample scores')
    measured = q12.metric(labels, values.numpy())
    validation.require(all(np.isclose(measured[name], expected[name]['value'], atol=1e-12, rtol=0.)
                           for name in measured), 'Checkpoint scores disagree with completed parent results')


def run(question):
    started = time.perf_counter()
    identity = validation.verify_protocol(question)
    output = validation.output(question)
    protocol = q1.read_json(output / 'protocol.json')
    rows, labels, groups, _ = validation.inputs(question)
    labels, groups = np.asarray(labels), np.asarray(groups)
    validation.require(len(rows) == 17927 and len(labels) == len(groups) == len(rows)
                       and not rows.variant_key.duplicated().any(), 'Full frozen validation cohort is incomplete')
    selected, selected_path, parent_result = load_selected(question)
    control_source = parent_result['strongest_control_source']
    strongest, strongest_path = load_control(question, control_source)
    positions = pd.Series(np.arange(len(rows)), index=rows.variant_key)
    selection_keys = protocol['selection_variant_keys']
    validation.require(len(selection_keys) == 2048 and len(set(selection_keys)) == 2048
                       and set(selection_keys) <= set(rows.variant_key), 'Invalid inherited selection sample')
    selection_indexes = positions.loc[selection_keys].to_numpy()
    sample_labels = labels[selection_indexes]
    verify_sample_scores(selected, 'lora_scores', sample_labels, parent_result['metrics']['lora'])
    verify_sample_scores(selected, 'control_scores', sample_labels, parent_result['metrics']['matched_control'])
    verify_sample_scores(strongest, 'control_scores', sample_labels, parent_result['metrics'][control_source])
    checkpoint_hashes = {'selected': q1.digest_file(selected_path), 'strongest_control': q1.digest_file(strongest_path)}

    model, tokenizer = base.load_backbone()
    try:
        module = validation.parent(question)
        if question == 'q13':
            validation.require(module.CONFIG['lora'] == q11.CONFIG['lora'], 'Q13 adapter configuration differs from its loader')
            adapters = base.attach_lora(model)
        else:
            config = module.CONFIG['lora']
            adapters = q14_adapters.attach(model, blocks=tuple(config['blocks']), rank=config['rank'],
                                           alpha=config['alpha'], seed=module.CONFIG['seed'])
        initial = previous.adapter_state(adapters)
        frozen = base.parameter_hash(model, frozen_only=True)
        parent_preflight = q1.read_json(module.OUTPUT / 'preflight.json')
        validation.require(frozen == parent_preflight['frozen_sha256'], 'Frozen backbone differs from the parent run')
        specs = {'matched_control': head_spec(selected, 'control'),
                 'strongest_control': head_spec(strongest, 'control')}
        predictions = predict(model, tokenizer, rows, specs, 'Frozen controls')
        previous.restore_adapters(adapters, selected['adapters'])
        specs['lora'] = head_spec(selected, 'head')
        predictions.update(predict(model, tokenizer, rows, {'lora': specs['lora']}, 'Selected LoRA'))
        for name, expected in [('lora', selected['lora_scores']),
                               ('matched_control', selected['control_scores']),
                               ('strongest_control', strongest['control_scores'])]:
            validation.require(np.allclose(predictions[name][selection_indexes], expected.numpy(),
                                           atol=2e-4, rtol=2e-4),
                               f'Full-pass {name} scores do not reproduce the original selection sample')

        # Reload original files, including scalers, after destroying deployed tensors.
        n = min(validation.CONFIG['reload_variants'], len(rows))
        erase(adapters, specs)
        reloaded, path, _ = load_selected(question)
        q1.verify_file(path, checkpoint_hashes['selected'])
        previous.restore_adapters(adapters, reloaded['adapters'])
        reloaded_specs = {'lora': head_spec(reloaded, 'head')}
        actual = predict(model, tokenizer, rows.iloc[:n], reloaded_specs, 'LoRA reload')['lora']
        validation.require(np.allclose(actual, predictions['lora'][:n], atol=2e-4, rtol=2e-4),
                           'Reloaded LoRA predictions differ')
        erase(adapters, reloaded_specs)
        previous.restore_adapters(adapters, initial)
        reloaded, path = load_control(question, control_source)
        q1.verify_file(path, checkpoint_hashes['strongest_control'])
        actual = predict(model, tokenizer, rows.iloc[:n],
                         {'strongest_control': head_spec(reloaded, 'control')}, 'Frozen control reload')['strongest_control']
        validation.require(np.allclose(actual, predictions['strongest_control'][:n], atol=2e-4, rtol=2e-4),
                           'Reloaded strongest frozen control predictions differ')
        validation.require(base.parameter_hash(model, frozen_only=True) == frozen, 'Original backbone tensors changed')
        peak_gpu_gib = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    validation.require(validation.verify_protocol(question) == identity, 'Validation protocol changed during inference')
    module.verified_results()
    q1.verify_file(selected_path, checkpoint_hashes['selected'])
    q1.verify_file(strongest_path, checkpoint_hashes['strongest_control'])
    frame = pd.DataFrame({'variant_key': rows.variant_key.to_numpy(), 'label': labels, 'component': groups,
                          **{name: predictions[name] for name in ('lora', 'matched_control', 'strongest_control')}})
    temporary = output / 'validation_predictions.csv.partial'
    frame.to_csv(temporary, index=False)
    temporary.replace(output / 'validation_predictions.csv')
    q1.write_json(output / 'prediction_provenance.json', {
        'question': question, 'identity': identity, 'checkpoint_sha256': checkpoint_hashes,
        'selected_checkpoint': str(selected_path.relative_to(q1.ROOT)),
        'matched_control_checkpoint': str(selected_path.relative_to(q1.ROOT)),
        'strongest_control_checkpoint': str(strongest_path.relative_to(q1.ROOT)),
        'strongest_control_source': control_source, 'frozen_sha256': frozen,
        'microbatch_variants': validation.CONFIG['microbatch_variants'],
        'reload_variant_keys': rows.variant_key.iloc[:n].tolist(),
        'selected_head_key': 'head', 'frozen_head_key': 'control',
        'both_controls_scored_in_one_original_backbone_pass': True,
        'checkpoint_files_reread_after_erasure': True,
        'original_selection_scores_reproduced': True,
    }, frozen=True)
    print('Computing full and post-selection-subset component bootstrap intervals.', flush=True)
    metrics, paired = validation.intervals(labels, predictions, groups)
    outside = ~rows.variant_key.isin(selection_keys).to_numpy()
    selection_components = np.unique(groups[selection_indexes])
    validation.require(set(selection_components) == set(protocol['selection_components']),
                       'Selection component membership changed')
    unseen = ~np.isin(groups, selection_components)
    result = {
        'status': 'complete', 'scope': 'full_validation', 'question': question, 'identity': identity,
        'validation_variants': len(rows), 'selection_variants': len(selection_keys),
        'validation_components': int(len(np.unique(groups))), 'metrics': metrics,
        'paired_lora_minus_strongest_control': paired,
        'subsets': {'outside_selection': subset_result(outside, labels, predictions, groups),
                    'unseen_components': subset_result(unseen, labels, predictions, groups)},
        'strongest_control_source': control_source,
        'reload_verified': True, 'reloaded_predictions_verified': True, 'frozen_unchanged': True,
        'original_selection_scores_reproduced': True,
        'peak_gpu_gib': peak_gpu_gib,
    }
    validation.require(result['subsets']['outside_selection']['variants'] == 15879,
                       'Outside-selection membership is incomplete')
    result['confirmation_passed'] = validation.confirmation_passes(result)
    result['seconds'] = time.perf_counter() - started
    files = ['protocol.json', 'input_checks.json', 'prediction_provenance.json', 'validation_predictions.csv']
    if (output / 'execution.json').exists():
        files.append('execution.json')
    result['artifacts'] = {name: q1.digest_file(output / name) for name in files}
    q1.write_json(output / 'metrics.json', result, frozen=True)
    print(f'{question.upper()} full confirmation completed: confirmed={result["confirmation_passed"]}; '
          f'{result["seconds"]/60:.1f} minutes.', flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('question', choices=('q13', 'q14', 'q15'))
    run(parser.parse_args().question)
