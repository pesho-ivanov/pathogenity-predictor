"""Reproduce the original (non-LoRA) Q11 numerical failure without fitting a model.

Uses the checksum-pinned archived backend and protocol. Probe only the first
training variant with a synthetic target; restore all modified weights. The live
Q11 implementation, protocol and checkpoints are never rewritten.
"""

import importlib.util
from types import SimpleNamespace
import time

from . import q1, q1_full

ROOT = q1.ROOT
ARCHIVE = ROOT / 'notebooks/results/q11/archive/block31_preflight'
OUTPUT = ROOT / 'notebooks/results/q11/diagnostics/original_numerics'


def archived_backend():
    """Bind archived GPU code to its original configuration and existing inputs."""
    manifest = q1.read_json(ARCHIVE / 'manifest.json')
    for name, checksum in manifest['artifacts'].items():
        q1.verify_file(ARCHIVE / name, checksum)
    for module in ['q0', 'q1', 'q1_full', 'q2', 'q9_environment', 'q10']:
        name = f'notebooks/src/{module}.py'
        q1.verify_file(ROOT / name, manifest['artifacts'][name])
    protocol = q1.read_json(ARCHIVE / 'notebooks/results/q11/protocol.json')
    q1.verify_file(q1_full.OUTPUT / 'protocol.json', protocol['q1_protocol_sha256'])
    directory = ROOT / 'notebooks/results/q11'
    for name in ['checkpoint_source.json', 'converted_checkpoint.json', 'environment/runtime.json']:
        q1.verify_file(directory / name, protocol['artifacts'][name])
    path = ARCHIVE / 'notebooks/src/q11_backend.py'
    spec = importlib.util.spec_from_file_location('notebooks.src._q11_original_backend', path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.q11 = SimpleNamespace(
        CONFIG=protocol['configuration'], OUTPUT=directory,
        CONVERTED=directory / 'base_checkpoint_zarr',
        CHECKPOINT=ROOT / 'data/evo2-savanna-7b/savanna_evo2_7b_base.pt')
    return backend, protocol


def summary(value):
    x = value.detach().double()
    return {'rms': float(x.square().mean().sqrt()), 'max_abs': float(x.abs().max())}


def trace(model, backend, tokenizer, sequence):
    """Observe residuals and actual mixer/MLP outputs before residual addition."""
    import torch
    records, hooks, saved = {}, [], {}
    for name, module in model.named_modules():
        parts = name.split('.')
        is_block = len(parts) == 3 and parts[:2] == ['decoder', 'layers']
        is_tail = name.startswith(('decoder.layers.30.', 'decoder.layers.31.'))
        if not (is_block or is_tail):
            continue
        def capture(mod, args, output, name=name):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            item = {'output': summary(value)}
            if args and isinstance(args[0], torch.Tensor):
                item['input'] = summary(args[0])
                if args[0].shape == value.shape:
                    item['equal_fraction'] = float((args[0] == value).double().mean())
                    item['change'] = summary(value.double() - args[0].double())
            records[name] = item
            if name == 'decoder.layers.31':
                saved['residual'] = args[0].detach().clone()
            if name in ['decoder.layers.31.self_attention', 'decoder.layers.31.mlp']:
                branch = value.detach().clone()
                if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                    branch = branch + output[1]
                saved[name.rsplit('.', 1)[-1]] = branch
        hooks.append(module.register_forward_hook(capture))
    try:
        backend.encode(model, tokenizer, sequence)
    finally:
        for hook in hooks:
            hook.remove()
    residual, attention, mlp = (saved[key] for key in ['residual', 'self_attention', 'mlp'])
    additions = {}
    for dtype in [torch.bfloat16, torch.float32, torch.float64]:
        x, a, m = [value.to(dtype) for value in [residual, attention, mlp]]
        additions[str(dtype)] = {
            'changed_elements': int(((x + a + m) != x).sum()),
            'elements': x.numel(),
            'change': summary((x + a + m).double() - x.double())}
    return {'modules': records, 'attention_with_bias': summary(attention),
            'mlp_branch': summary(mlp), 'residual_addition_replay': additions}


def optimizer_probe(model, backend, tokenizer, row):
    """Measure actual FP32 and BF16 changes, including the fixed head's clipping."""
    import torch
    block = backend.select_trainable_block(model)
    original = backend.cpu_state(dict(block.named_parameters()))
    frozen = backend.parameter_hash(model, frozen_only=True)
    head = torch.nn.Linear(8192, 1, device='cuda', dtype=torch.float32)
    head_state = backend.cpu_state(head.state_dict())
    optimizer = backend.MasterAdamW(block, head)
    optimizer.optimizer.param_groups[1]['lr'] = 0.
    optimizer.optimizer.param_groups[0]['weight_decay'] = 0.
    features = backend.variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence)
    scale = features.std().clamp_min(1e-6)
    initial = float(head(features / scale).detach())
    record = {'feature_scale': float(scale), 'parameters': {}}
    try:
        current = backend.variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence, True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(current / scale).reshape(()), torch.ones((), device='cuda'))
        loss.backward()
        head_norm = torch.cat([p.grad.flatten() for p in head.parameters()]).norm()
        for name, p in block.named_parameters():
            g = p.grad.detach().double()
            record['parameters'][name] = {'weight': summary(p), 'gradient': summary(g),
                'first_step_unclipped_adam_bound': float((1e-5 * g.abs() / (g.abs() + 1e-8)).max())}
        optimizer.accumulate()
        record['joint_gradient_norm_before_clip'] = optimizer.step()
        record['head_gradient_norm_before_clip'] = float(head_norm)
        for (name, p), master in zip(block.named_parameters(), optimizer.masters):
            saved = original[name].to(p.device)
            record['parameters'][name].update(
                changed_fp32_master_elements=int((master != saved.float()).sum()),
                master_delta=summary(master - saved.float()),
                changed_bf16_elements=int((p != saved).sum()))
        after = backend.variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence)
        record.update(loss=float(loss.detach()), score_change=float(head(after / scale).detach()) - initial,
                      feature_max_change=float((after - features).abs().max()),
                      frozen_unchanged=backend.parameter_hash(model, frozen_only=True) == frozen,
                      head_unchanged=all(torch.equal(p.cpu(), head_state[n]) for n, p in head.state_dict().items()))
    finally:
        with torch.no_grad():
            for name, p in block.named_parameters():
                p.copy_(original[name])
        record['original_weights_restored'] = all(torch.equal(p.cpu(), original[n]) for n, p in block.named_parameters())
    if not all(record[k] for k in ['frozen_unchanged', 'head_unchanged', 'original_weights_restored']):
        raise RuntimeError('Diagnostic isolation check failed')
    return record


def run():
    import torch
    start = time.perf_counter()
    backend, protocol = archived_backend()
    parent = q1.read_json(q1_full.OUTPUT / 'protocol.json')
    for name in ['split_manifest.csv', 'sequences.csv.gz']:
        q1.verify_file(q1_full.OUTPUT / name, parent['artifacts'][name])
    manifest = q1.read_csv(q1_full.OUTPUT / 'split_manifest.csv')
    key = manifest.loc[manifest.split.eq('train'), 'variant_key'].iloc[0]
    dna = q1.read_csv(q1_full.OUTPUT / 'sequences.csv.gz')
    row = dna.loc[dna.variant_key.eq(key)].iloc[0]
    record = {'scope': 'Original non-LoRA Q11; training-only synthetic-target numerical diagnosis; no performance estimate',
              'source_sha256': q1.digest_file(__file__), 'training_key': key,
              'original_protocol_sha256': q1.digest_file(ARCHIVE / 'notebooks/results/q11/protocol.json'),
              'configuration': protocol['configuration']}
    free, total = torch.cuda.mem_get_info()
    if free < 45 * 1024**3:
        raise RuntimeError('The original-block diagnostic needs 45 GiB free GPU memory before starting')
    record['actual_runtime'] = {
        'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'free_gpu_gib_before_load': free / 1024**3, 'total_gpu_gib': total / 1024**3,
        'timing_limit': 'Other processes may share the GPU; elapsed time is not a throughput benchmark'}
    model, tokenizer = backend.load_backbone()
    try:
        record['conversion_audit'] = backend.audit_conversion(model)
        print('Original source-to-loaded tensor conversion verified.', flush=True)
        record['trace'] = trace(model, backend, tokenizer, row.ref_sequence)
        print('Tail submodule and residual precision traces collected.', flush=True)
        record['optimizer'] = optimizer_probe(model, backend, tokenizer, row)
        if (record['trace']['modules']['decoder.layers.31']['equal_fraction'] != 1.
                or any(p['changed_bf16_elements'] for p in record['optimizer']['parameters'].values())
                or record['optimizer']['score_change'] != 0.):
            raise RuntimeError('Fresh diagnostic differs from the archived failure; investigate before reporting')
        record['tail_parameters'] = {str(i): {n: summary(p) for n, p in model.decoder.layers[i].named_parameters()}
                                     for i in [29, 30, 31]}
        record['seconds'] = time.perf_counter() - start
        record['peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        record['runtime'] = q1.read_json(backend.q11.OUTPUT / 'environment/runtime.json')
        q1.write_json(OUTPUT / 'probe.json', record)
        print('Saved original Q11 numerical probe:', OUTPUT / 'probe.json', flush=True)
        return record
    finally:
        torch.distributed.destroy_process_group()


def reproduce():
    """Always launch a fresh GPU process; never substitute a cached probe."""
    from datetime import datetime, timezone
    import fcntl
    import os
    import subprocess
    from . import q9_environment
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log = OUTPUT / f'run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log'
    # The isolated probe only reads production inputs and writes in OUTPUT.
    # Its own lock permits unrelated Q11 work when sufficient GPU memory is free.
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Recomputing original checkpoint trace and optimizer probe on the GPU.', flush=True)
        with log.open('w') as stream:
            completed = subprocess.run(
                [str(q9_environment.PYTHON), '-u', '-m', 'notebooks.src.q11_numerics'],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),),
                env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'))
        if completed.returncode:
            raise RuntimeError(f'Original Q11 numerical probe failed; inspect {log}')
    record = q1.read_json(OUTPUT / 'probe.json')
    if record['source_sha256'] != q1.digest_file(__file__):
        raise ValueError('Diagnostic source changed during execution')
    print(f"Completed in {record['seconds']:.1f} s; peak GPU allocation {record['peak_gpu_gib']:.1f} GiB.")
    return record


def show(record):
    from IPython.display import Markdown, display
    import matplotlib.pyplot as plt
    from . import q0
    modules = record['trace']['modules']
    parameters = record['optimizer']['parameters']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].semilogy(range(32), [modules[f'decoder.layers.{i}']['output']['rms'] for i in range(32)], '.-')
    axes[0].set(xlabel='Block (zero based)', ylabel='Output RMS', title='Residual scale jumps in block 30')
    names = ['self_attention.linear_qkv.weight', 'self_attention.linear_proj.weight',
             'self_attention.linear_proj.bias', 'mlp.linear_fc1.weight', 'mlp.linear_fc2.weight']
    axes[1].bar(['QKV', 'Attn out', 'Attn bias', 'MLP in', 'MLP out'],
                [parameters[n]['gradient']['max_abs'] for n in names])
    axes[1].set(yscale='log', ylabel='Maximum absolute gradient', title='Final-block gradients are tiny')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'numerics.png', dpi=160)
    plt.show()
    optimizer = record['optimizer']
    total_masters = sum(p['changed_fp32_master_elements'] for p in parameters.values())
    total_bf16 = sum(p['changed_bf16_elements'] for p in parameters.values())
    display(Markdown(
        f"The first training reference enters block 31 with RMS **{modules['decoder.layers.31']['input']['rms']:.3g}**. "
        f"Its attention branch including bias has RMS **{record['trace']['attention_with_bias']['rms']:.3g}**. "
        f"The complete block leaves **{100 * modules['decoder.layers.31']['equal_fraction']:.0f}%** of elements unchanged.\n\n"
        f"One synthetic-target step changes **{total_masters:,} FP32 master elements**, "
        f"but **{total_bf16:,} deployed BF16 elements**, with fixed-head score change "
        f"**{optimizer['score_change']:.3g}**. Frozen weights, head and restored probe weights all pass equality checks."))
    q0.details('Measured submodule scales, gradients, precision replay and provenance', record)


def conclusion(record):
    from IPython.display import Markdown, display
    norm = record['optimizer']['joint_gradient_norm_before_clip']
    display(Markdown(
        "**Conclusion.** Original Q11 stopped at its preflight before the supervised epoch. "
        "Its final attention block was numerically inactive in this checkpoint/configuration: "
        "native QKV normalization weights have RMS about 4.4e-4 and QKV projection weights about 4.4e-6. "
        "These very small weights produce a tiny branch beside the enormous "
        "block-30 residual. Residual addition loses that contribution, while feature scaling makes "
        "the already small gradients smaller. With AdamW's `lr=1e-5` and `eps=1e-8`, the update for "
        "a gradient much smaller than epsilon is approximately `lr * gradient / eps`. "
        f"The fixed diagnostic head also participates in joint clipping (pre-clip norm **{norm:.3g}**), "
        "although its learning rate is zero. These effects explain unchanged deployed weights despite "
        "a connected gradient graph and FP32 master weights.\n\n"
        "The earlier three-training-variant diagnostic observed zero deployed changes after 32 steps "
        "in block 31; block 30 changed 92,362 elements after eight steps. That supports probing "
        "block 30 as a different training target, but establishes no pathogenicity performance. "
        "Lowering epsilon or increasing precision alone has not been shown to fix final-block training.\n\n"
        "Every loaded tensor was compared with the original Savanna checkpoint; tiny final-block "
        "weights are already present in that source. The checkpoint alone cannot establish why those "
        "weights became tiny during pretraining. Native Savanna/Vortex forward parity is still needed "
        "to determine how much of the block-30 activation scale is checkpoint behavior versus backend "
        "numerics. The precision replay only recomputes the captured residual addition; it is not "
        "a complete FP32/FP64 model run. These training-only synthetic probes do not read validation "
        "labels or alter the live Q11 experiment.\n\n"
        "The update equation follows the [PyTorch 2.7 AdamW documentation]"
        "(https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html). "
        "[NVIDIA's model notes](https://docs.nvidia.com/bionemo-framework/2.7.1/main/developer-guide/"
        "bionemo-evo2/bionemo-evo2-Overview/index.html) distinguish checkpoint-specific precision "
        "behavior; the 1B/40B precision issues should not automatically be assigned to this 7B model."))


def write_notebook():
    """Generate, execute in a fresh kernel, validate, and only then save."""
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            '# Q11. Why did the original final-block fine-tuning fail?\n\n'
            'This investigates the original full-block experiment, separate from the later LoRA work. '
            'It reloads the archived protocol and original checkpoint, traces all 32 blocks, audits '
            'source tensor conversion, and measures one actual optimizer step on the first frozen '
            'training variant with a synthetic target. Earlier 32-step diagnostics remain archived. '
            'No validation labels or performance estimates are used.\n\n'
            'Requires the existing checksum-pinned Q1/Q11 artifacts, `.venv/q9` BioNeMo environment '
            'and an H100 GPU. Implementation: [q11_numerics.py](src/q11_numerics.py). '
            'Regenerate and execute with `.venv/bin/python -m notebooks.src.q11_numerics --notebook`.'),
        nbformat.v4.new_code_cell(
            'import sys\nfrom pathlib import Path\n'
            'sys.path.insert(0, str(Path.cwd().parent) if Path.cwd().name == "notebooks" else str(Path.cwd()))\n'
            'from notebooks.src import q11_numerics'),
        nbformat.v4.new_code_cell('result = q11_numerics.reproduce()'),
        nbformat.v4.new_code_cell('q11_numerics.show(result)'),
        nbformat.v4.new_code_cell('q11_numerics.conclusion(result)'),
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(notebook, timeout=1200, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if [cell.execution_count for cell in cells] != list(range(1, len(cells) + 1)) or any(
            output.output_type == 'error' for cell in cells for output in cell.outputs):
        raise RuntimeError('Notebook did not finish cleanly in order')
    nbformat.validate(notebook)
    path = ROOT / 'notebooks/Q11-gradient-diagnostics.ipynb'
    nbformat.write(notebook, path)
    print('Saved fully executed notebook:', path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', action='store_true')
    args = parser.parse_args()
    write_notebook() if args.notebook else run()
