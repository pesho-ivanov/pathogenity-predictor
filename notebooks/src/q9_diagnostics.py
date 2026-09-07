"""Training-only numerical probes for Q9; no validation-based selection."""

import argparse
from pathlib import Path
import tarfile
import time
from urllib.request import urlopen

import numpy as np
import torch

from . import q1, q9_backend

OUTPUT = Path(__file__).resolve().parents[2] / 'notebooks/results/q9/investigation'
ROOT = Path(__file__).resolve().parents[2]
BF16_ARCHIVE_SHA256 = 'ea4a3f5c9c26d5edc10bdc85165c090ad0ff23ac2670d4f61244f5f0d9d5e817'


def download_bf16_checkpoint():
    """Public NVIDIA precision-adapted weights, downloaded only for this investigation."""
    directory = ROOT / 'data/evo2-bf16-1b'
    directory.mkdir(parents=True, exist_ok=True)
    filename = 'nemo2_evo2_1b_8k_bf16.tar.gz'
    url = 'https://api.ngc.nvidia.com/v2/models/nvidia/clara/evo2-1b-8k-bf16-nemo2/versions/1.0/files/' + filename
    archive = directory / filename
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with urlopen(url, timeout=60) as response, partial.open('wb') as stream:
            total = 0
            while block := response.read(8*1024**2):
                stream.write(block)
                total += len(block)
                if total % (128*1024**2) == 0:
                    print('BF16 checkpoint download MiB:', total//1024**2, flush=True)
        q1.verify_file(partial, BF16_ARCHIVE_SHA256)
        partial.replace(archive)
    q1.verify_file(archive, BF16_ARCHIVE_SHA256)
    if not (directory / 'extracted').exists():
        partial_directory = directory / 'extracting'
        if partial_directory.exists():
            raise RuntimeError('Incomplete checkpoint extraction exists; preserve it before retrying')
        with tarfile.open(archive) as bundle:
            bundle.extractall(partial_directory, filter='data')
        partial_directory.rename(directory / 'extracted')
    files = {str(p.relative_to(directory / 'extracted')): q1.digest_file(p)
             for p in sorted((directory / 'extracted').rglob('*')) if p.is_file()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {'url': url, 'sha256': BF16_ARCHIVE_SHA256, 'files': files,
              'nvidia_notes': 'https://docs.nvidia.com/bionemo-framework/2.7.1/main/developer-guide/bionemo-evo2/bionemo-evo2-Overview/index.html#available-models-in-ngc'}
    q1.write_json(OUTPUT / 'bf16_checkpoint.json', record)
    print('Downloaded and verified BF16-compatible checkpoint:', directory, flush=True)
    return directory / 'extracted'


def summary(value):
    if isinstance(value, tuple):
        value = value[0]
    x = value.detach().double()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'rms': float(x.square().mean().sqrt()), 'max_abs': float(x.abs().max())}


def trace(model, tokenizer, sequence):
    records, handles, block_inputs = {}, [], {}
    for name, module in model.named_modules():
        parts = name.split('.')
        is_block = len(parts) == 3 and parts[:2] == ['decoder', 'layers']
        is_tail_module = name.startswith(('decoder.layers.23.', 'decoder.layers.24.'))
        if is_block or is_tail_module:
            if is_block:
                def capture_input(mod, inputs, name=name):
                    value = inputs[0]
                    while isinstance(value, tuple):
                        value = value[0]
                    block_inputs[name] = value.detach().clone()
                handles.append(module.register_forward_pre_hook(capture_input))
            def collect(mod, inputs, output, name=name):
                if isinstance(output, torch.Tensor) or isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    records[name] = {'output': summary(output)}
                    previous = block_inputs.pop(name, inputs[0] if inputs else None)
                    while isinstance(previous, tuple):
                        previous = previous[0]
                    if isinstance(previous, torch.Tensor):
                        records[name]['input'] = summary(previous)
                        out = output[0] if isinstance(output, tuple) else output
                        if out.shape == previous.shape:
                            records[name]['change_rms'] = summary(out.double()-previous.double())['rms']
                            records[name]['equal_fraction'] = float((out == previous).double().mean())
            handles.append(module.register_forward_hook(collect))
    try:
        q9_backend.encode(model, tokenizer, sequence)
    finally:
        for handle in handles:
            handle.remove()
    return records


def probe(mode):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = q1.verify_protocol()
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
    keys = pilot.loc[pilot.split.eq('train'), 'variant_key'].iloc[:3].tolist()
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz').set_index('variant_key').loc[keys]
    model, tokenizer = q9_backend.load_backbone(fp8=mode == 'fp8',
                                                checkpoint='bf16_adapted' if mode == 'bf16_adapted' else 'original')
    records = {'mode': mode, 'training_keys': keys, 'vcf_exports': protocol['vcf_exports'],
               'description': 'Three predetermined training pairs; synthetic target; no model selection', 'cases': []}
    try:
        block = model.decoder.layers[-1].requires_grad_(True)
        head = torch.nn.Linear(3840, 1, dtype=torch.float32, device='cuda')
        for key, row in dna.iterrows():
            # FP8 diagnostic: initial pass and two train-only warmups, not a production calibration protocol.
            if mode == 'fp8':
                records.setdefault('fp8_warmups', {})[key] = []
                for _ in range(3):
                    f, ll = q9_backend.encode(model, tokenizer, row.ref_sequence)
                    records['fp8_warmups'][key].append({'feature_rms': summary(f)['rms'], 'likelihood': float(ll)})
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            layers = trace(model, tokenizer, row.ref_sequence)
            f, _ = q9_backend.variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence, True)
            scale = f.detach().double().std().float().clamp_min(1e-6)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f/scale).reshape(()),
                                                                         torch.ones((), device='cuda'))
            try:
                loss.backward()
            except RuntimeError as error:
                records['failure'] = {'phase': 'backward', 'error': str(error), 'layers': layers,
                                      'feature_scale': float(scale), 'loss': float(loss.detach())}
                q1.write_json(OUTPUT / f'{mode}_trace.json', records)
                print('DIAGNOSTIC_FAILURE', mode, str(error), flush=True)
                return
            case = {'key': key, 'feature_scale': float(scale), 'loss': float(loss.detach()), 'layers': layers,
                    'gradients': {n: summary(p.grad) if p.grad is not None else None for n, p in block.named_parameters()}}
            records['cases'].append(case)
            print('DIAGNOSTIC', mode, key, 'scale', float(scale), 'block23', layers['decoder.layers.23']['output']['rms'],
                  'block24_equal', layers['decoder.layers.24']['equal_fraction'], flush=True)
        q1.write_json(OUTPUT / f'{mode}_trace.json', records)
    finally:
        torch.distributed.destroy_process_group()


def run_investigation():
    """Reproduce every displayed diagnostic, without fitting on validation labels."""
    from . import q9, q9_environment
    q9.verify_inputs()
    download_bf16_checkpoint()
    identity = {'sources': q9.source_hashes(), 'optimizer': q9.CONFIG,
                'parent_protocol_sha256': q1.digest_file(q1.OUTPUT / 'protocol.json'),
                'original_checkpoint_sha256': q1.digest_file(q9.OUTPUT / 'converted_checkpoint.json'),
                'bf16_checkpoint_sha256': q1.digest_file(OUTPUT / 'bf16_checkpoint.json')}
    marker = OUTPUT / 'manifest.json'
    if marker.exists():
        record = q1.read_json(marker)
        q9.assert_identity(record['identity'], identity)
        for name, digest in record['artifacts'].items():
            q1.verify_file(OUTPUT / name, digest)
        print('Verified cached training-only gradient investigation.')
        return record
    for mode in ['bf16', 'bf16_adapted', 'fp8', 'tail', 'tail_adapted', 'hyena_adapted']:
        q9_environment.run_command('investigate-' + mode,
                                    [q9_environment.PYTHON, '-m', 'notebooks.src.q9_diagnostics', mode])
    files = ['bf16_trace.json', 'bf16_adapted_trace.json', 'fp8_trace.json',
             'original_tail_updates.json', 'bf16_adapted_tail_updates.json',
             'bf16_adapted_hyena_updates.json', 'bf16_checkpoint.json']
    train_keys = set(q1.read_csv(q1.OUTPUT / 'split_manifest.csv').query("split == 'train'").variant_key)
    for filename in files[:-1]:
        record = q1.read_json(OUTPUT / filename)
        if set(record['training_keys']) - train_keys:
            raise ValueError('Non-training variant entered the gradient investigation')
        if filename.endswith('_updates.json') and not all(record[k] for k in
                ['frozen_unchanged', 'head_unchanged', 'original_weights_restored']):
            raise ValueError('Gradient investigation failed its isolation checks')
    record = {'identity': identity, 'scope': 'Training-only numerical diagnostics; no validation performance',
              'artifacts': {n: q1.digest_file(OUTPUT / n) for n in files}}
    q1.write_json(marker, record)
    print('Training-only gradient investigation completed.')
    return record


def show_results():
    from . import q0
    import matplotlib.pyplot as plt
    base = q1.read_json(OUTPUT / 'bf16_trace.json')
    adapted = q1.read_json(OUTPUT / 'bf16_adapted_trace.json')
    updates = q1.read_json(OUTPUT / 'bf16_adapted_hyena_updates.json')
    original_updates = q1.read_json(OUTPUT / 'original_tail_updates.json')
    adapted_updates = q1.read_json(OUTPUT / 'bf16_adapted_tail_updates.json')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for run, label in [(base, 'Original checkpoint'), (adapted, 'BF16-compatible')]:
        layers = run['cases'][0]['layers']
        axes[0].semilogy(range(25), [layers[f'decoder.layers.{i}']['output']['rms'] for i in range(25)],
                         marker='.', label=label)
    axes[0].set(xlabel='Block (0-based)', ylabel='Activation RMS', title='Large jump at block 23')
    axes[0].legend(fontsize=8)
    gradients = original_updates['steps'][0]['gradient_max']
    axes[1].bar(['Hyena block 23', 'Attention block 24'], [gradients['23'], gradients['24']],
                color=['#247ba0', '#d1495b'])
    axes[1].set(yscale='log', ylabel='Largest absolute gradient', title='Usable gradient in block 23')
    for run, label in [(adapted_updates, 'Train blocks 23–24'), (updates, 'Train block 23 only')]:
        axes[2].plot([0]+[s['step'] for s in run['steps']],
                     [0]+[s['score']-run['initial_score'] for s in run['steps']], marker='.', label=label)
    axes[2].set(xlabel='Diagnostic optimizer step', ylabel='Change in score (fixed head)',
                title='Backbone updates change the output')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'gradient_investigation.png', dpi=150)
    plt.show()
    report = {'original_checkpoint': {'cases': [
        {'key': c['key'], 'block23_rms': c['layers']['decoder.layers.23']['output']['rms'],
         'block24_attention_rms': c['layers']['decoder.layers.24.self_attention']['output']['rms'],
         'block24_equal_fraction': c['layers']['decoder.layers.24']['equal_fraction']} for c in base['cases']]},
        'alternative_test': {'checkpoint': updates['checkpoint'], 'trainable_blocks': updates['trainable_blocks'],
             'changed_elements': sum(updates['changed_elements'].values()),
             'initial_score': updates['initial_score'], 'final_score': updates['steps'][-1]['score'],
             'frozen_unchanged': updates['frozen_unchanged'], 'head_unchanged': updates['head_unchanged'],
             'original_weights_restored': updates['original_weights_restored'],
             'peak_gpu_gib': updates['peak_gpu_gib'], 'seconds': updates['seconds']},
        'fp8_diagnostic': q1.read_json(OUTPUT / 'fp8_trace.json').get('failure', 'No runtime failure'),
        'interpretation': 'Synthetic training-only update probes; these scores are not pathogenicity evaluation',
        'precision_adaptation_provenance': 'NVIDIA documents precision adaptation; its exact repair-training data have not been independently audited'}
    # The full FP8 tensor trace is kept on disk, not duplicated inside the notebook.
    if isinstance(report['fp8_diagnostic'], dict):
        report['fp8_diagnostic'] = {k: v for k, v in report['fp8_diagnostic'].items() if k != 'layers'}
    q1.write_json(OUTPUT / 'summary.json', report)
    q0.details('Training-only probes, parameter updates and limitations', report)


def update_tail(checkpoint='original', single=False):
    """Eight optimizer steps on fixed training pairs, with an unchanged diagnostic head."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q1.verify_protocol()
    pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
    keys = pilot.loc[pilot.split.eq('train'), 'variant_key'].iloc[:3].tolist()
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz').set_index('variant_key').loc[keys]
    model, tokenizer = q9_backend.load_backbone(checkpoint=checkpoint)
    try:
        blocks = [23] if single else [23, 24]
        tail = torch.nn.ModuleList([model.decoder.layers[i] for i in blocks]).requires_grad_(True)
        before = {n: p.detach().clone() for n, p in tail.named_parameters()}
        frozen = q9_backend.parameter_hash(model, frozen_only=True)
        head = torch.nn.Linear(3840, 1, dtype=torch.float32, device='cuda')
        head_state = {n: p.detach().clone() for n, p in head.named_parameters()}
        optimizer = q9_backend.MasterAdamW(tail, head)
        # Keep the head fixed so score changes must come from backbone updates.
        optimizer.optimizer.param_groups[1]['lr'] = 0.
        row0 = dna.iloc[0]
        initial, _ = q9_backend.variant_features(model, tokenizer, row0.ref_sequence, row0.alt_sequence)
        scale = initial.detach().double().std().float().clamp_min(1e-6)
        score0 = float(head(initial/scale).detach())
        steps = []
        start = time.perf_counter()
        for step in range(8):
            gradient_max = {23: 0., 24: 0.}
            losses = []
            for i in range(8):
                row = dna.iloc[i % len(dna)]
                f, _ = q9_backend.variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence, True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f/scale).reshape(()),
                                                                             torch.ones((), device='cuda'))
                loss.backward()
                losses.append(float(loss.detach()))
                for b in [23, 24]:
                    gradient_max[b] = max(gradient_max[b], max((float(p.grad.abs().max())
                        for p in model.decoder.layers[b].parameters() if p.grad is not None), default=0.))
                optimizer.accumulate()
            norm = optimizer.step(denominator=8)
            f, _ = q9_backend.variant_features(model, tokenizer, row0.ref_sequence, row0.alt_sequence)
            record = {'step': step+1, 'loss': float(np.mean(losses)), 'gradient_norm': norm,
                      'gradient_max': gradient_max, 'score': float(head(f/scale).detach()),
                      'feature_relative_change': float((f-initial).double().norm()/initial.double().norm())}
            steps.append(record)
            print('TAIL_UPDATE', checkpoint, record, flush=True)
        changes = {n: int((p.detach() != before[n]).sum()) for n, p in tail.named_parameters()}
        assert q9_backend.parameter_hash(model, frozen_only=True) == frozen
        assert all(torch.equal(p, head_state[n]) for n, p in head.named_parameters())
        record = {'checkpoint': checkpoint, 'trainable_blocks': blocks, 'training_keys': keys, 'steps': steps,
                  'initial_score': score0, 'changed_elements': changes, 'frozen_unchanged': True,
                  'head_unchanged': True, 'seconds': time.perf_counter()-start,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3,
                  'description': 'Diagnostic synthetic target 1; head fixed, eight accumulated pairs per step; no validation used'}
        with torch.no_grad():
            for n, p in tail.named_parameters():
                p.copy_(before[n])
        assert all(torch.equal(p, before[n]) for n, p in tail.named_parameters())
        record['original_weights_restored'] = True
        suffix = 'hyena_updates' if single else 'tail_updates'
        q1.write_json(OUTPUT / f'{checkpoint}_{suffix}.json', record)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['bf16', 'fp8', 'bf16_adapted', 'download',
                                        'tail', 'tail_adapted', 'hyena', 'hyena_adapted'])
    mode = parser.parse_args().mode
    if mode == 'download':
        download_bf16_checkpoint()
    elif mode.startswith('tail'):
        update_tail('bf16_adapted' if mode == 'tail_adapted' else 'original')
    elif mode.startswith('hyena'):
        update_tail('bf16_adapted' if mode == 'hyena_adapted' else 'original', single=True)
    else:
        probe(mode)
