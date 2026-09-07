"""BioNeMo/NeMo model operations, executed only in Q9's isolated environment."""

from pathlib import Path
import os
import tempfile
import hashlib
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'notebooks/results/q9'


def load_backbone(fp8=False, checkpoint='original'):
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from nemo.collections.llm.gpt.model.hyena import Hyena1bConfig
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    from . import q1

    if checkpoint == 'original':
        conversion = q1.read_json(OUTPUT / 'converted_checkpoint.json')
        checkpoint_root = OUTPUT / 'base_checkpoint_zarr'
    elif checkpoint == 'bf16_adapted':
        conversion = q1.read_json(OUTPUT / 'investigation/bf16_checkpoint.json')
        checkpoint_root = ROOT / 'data/evo2-bf16-1b/extracted'
    else:
        raise ValueError('Unknown checkpoint')
    for name, digest in conversion['files'].items():
        q1.verify_file(checkpoint_root / name, digest)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        handle, rendezvous = tempfile.mkstemp(prefix='q9-rendezvous-')
        os.close(handle)
        os.unlink(rendezvous)
        torch.distributed.init_process_group('nccl', init_method='file://' + rendezvous,
                                             rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                                  pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)
    tokenizer = get_nmt_tokenizer('byte-level')
    config = Hyena1bConfig(seq_length=1024, recompute_granularity=None,
                          recompute_method=None, perform_initialization=False,
                          use_cpu_initialization=True, gradient_accumulation_fusion=False,
                          fp32_residual_connection=False,
                          fp8='hybrid' if fp8 else None,
                          fp8_amax_history_len=16, fp8_amax_compute_algo='max', fp8_wgrad=False)
    model = config.configure_model(tokenizer).cuda().to(torch.bfloat16)
    sharded = {'state_dict': model.sharded_state_dict(prefix='module.')}
    restored = dist_checkpointing.load(sharded, checkpoint_root / 'weights')
    state = {name.removeprefix('module.'): value for name, value in restored['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if len(model.decoder.layers) != 25:
        raise RuntimeError('Unexpected Evo2 backbone depth')
    return model, tokenizer


def encode(model, tokenizer, sequence, gradients=False):
    """Pool the final block before final normalization, matching Q2's feature layer."""
    if len(sequence) != 1024 or set(sequence) - set('ACGT'):
        raise ValueError('Expected exactly 1,024 A/C/G/T bases')
    tokens = torch.tensor([tokenizer.text_to_ids(sequence)], dtype=torch.long, device='cuda')
    positions = torch.arange(len(sequence), device='cuda').unsqueeze(0)
    captured = []

    def collect(module, inputs, output):
        values = output[0] if isinstance(output, tuple) else output
        captured.append(values.float().mean(dim=0)[0])

    hook = model.decoder.layers[-1].register_forward_hook(collect)
    try:
        with torch.set_grad_enabled(gradients):
            logits = model(tokens, positions, attention_mask=None)
            pooled = captured[-1]
            likelihood = torch.log_softmax(logits.float(), dim=-1)[:, :-1].gather(
                2, tokens[:, 1:, None]).mean()
        if pooled.shape != (1920,) or not torch.isfinite(pooled).all() or not torch.isfinite(likelihood):
            raise ValueError('Non-finite or malformed Evo2 output')
        return pooled, likelihood
    finally:
        hook.remove()


def smoke_forward():
    model, tokenizer = load_backbone()
    sequence = ''.join(np.random.default_rng(42).choice(list('ACGT'), 1024))
    pooled, likelihood = encode(model, tokenizer, sequence)
    print('FORWARD_OK', list(pooled.shape), float(likelihood),
          'gpu_gib', torch.cuda.max_memory_allocated() / 1024**3, flush=True)
    torch.distributed.destroy_process_group()


def variant_features(model, tokenizer, reference, alternate, gradients=False):
    from .q1 import reverse_complement
    pooled, scores = [], []
    for sequence in (reference, alternate):
        canonical = min(sequence, reverse_complement(sequence))
        forward = encode(model, tokenizer, canonical, gradients)
        reverse = encode(model, tokenizer, reverse_complement(canonical), gradients)
        pooled.append((forward[0] + reverse[0]) / 2)
        scores.append((forward[1] + reverse[1]) / 2)
    return torch.cat((pooled[0], pooled[1] - pooled[0])), -(scores[1] - scores[0])


def parameter_hash(model, frozen_only=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def select_trainable_block(model):
    """Use the same explicit backbone block in readiness and supervised training."""
    from .q9 import CONFIG
    index = CONFIG['trainable_block']
    if len(model.decoder.layers) != 25 or not 0 <= index < 25:
        raise ValueError('Unexpected Evo2 block configuration')
    model.requires_grad_(False)
    return model.decoder.layers[index].requires_grad_(True)


class MasterAdamW:
    """FP32 optimizer weights prevent small updates from vanishing in BF16 rounding."""

    def __init__(self, block, head):
        from .q9 import CONFIG
        self.clip_grad = CONFIG['clip_grad']
        self.parameters = list(block.parameters())
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.head = head
        self.optimizer = torch.optim.AdamW([
            {'params': self.masters, 'lr': CONFIG['backbone_lr']},
            {'params': head.parameters(), 'lr': CONFIG['head_lr']}],
            betas=tuple(CONFIG['betas']), eps=CONFIG['epsilon'], weight_decay=CONFIG['weight_decay'])
        self.accumulated = [torch.zeros_like(p) for p in self.masters]

    def accumulate(self):
        # Accumulate in FP32 after each variant, then release its BF16 gradients.
        for parameter, accumulator in zip(self.parameters, self.accumulated):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Missing or non-finite trainable backbone gradient')
            accumulator.add_(parameter.grad.float())
            parameter.grad = None

    def step(self, denominator=1):
        for master, accumulator in zip(self.masters, self.accumulated):
            master.grad = accumulator / denominator
        for parameter in self.head.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Missing or non-finite classification gradient')
            parameter.grad.div_(denominator)
        norm = torch.nn.utils.clip_grad_norm_(self.masters + list(self.head.parameters()),
                                              self.clip_grad, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master, accumulator in zip(self.parameters, self.masters, self.accumulated):
                parameter.copy_(master)
                accumulator.zero_()
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm)


def preflight():
    from . import q1, q9
    model, tokenizer = load_backbone()
    try:
        block = select_trainable_block(model)
        block_index = q9.CONFIG['trainable_block']
        original = {n: p.detach().clone() for n, p in block.named_parameters()}
        frozen = parameter_hash(model, frozen_only=True)
        head = torch.nn.Linear(3840, 1, device='cuda', dtype=torch.float32)
        head_before = head.weight.detach().clone()
        bias_before = head.bias.detach().clone()
        optimizer = MasterAdamW(block, head)
        original_masters = [p.detach().clone() for p in optimizer.masters]
        pilot = q1.read_csv(q1.OUTPUT / 'split_manifest.csv')
        train_keys = pilot.loc[pilot.split.eq('train'), 'variant_key'].iloc[:3].tolist()
        inputs = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz')
        dna = inputs.set_index('variant_key').loc[train_keys]
        first = dna.iloc[0]
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        initial, _ = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
        smoke_scale = initial.detach().double().std().float().clamp_min(1e-6)
        initial_score = head(initial / smoke_scale).detach()
        gradient_maxima = {n: 0. for n, _ in block.named_parameters()}
        # A fixed head makes a changed score evidence of a backbone update.
        optimizer.optimizer.param_groups[1]['lr'] = 0.
        for _ in range(q9.CONFIG['readiness_optimizer_steps']):
            for i in range(q9.CONFIG['gradient_accumulation']):
                row = dna.iloc[i % len(dna)]
                features, _ = variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence, True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head(features / smoke_scale).reshape(()), torch.ones((), device='cuda'))
                loss.backward()
                for name, parameter in block.named_parameters():
                    if parameter.grad is not None:
                        gradient_maxima[name] = max(gradient_maxima[name], float(parameter.grad.abs().max()))
                optimizer.accumulate()
            norm = optimizer.step(denominator=q9.CONFIG['gradient_accumulation'])
        features, _ = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
        fixed_head_score = head(features / smoke_scale).detach()
        fixed_head_unchanged = torch.equal(head.weight, head_before) and torch.equal(head.bias, bias_before)
        assert fixed_head_unchanged, 'The diagnostic head changed during the backbone probe'
        backbone_effective = not torch.equal(features, initial) and not torch.equal(fixed_head_score, initial_score)
        feature_relative_change = float((features-initial).double().norm()/initial.double().norm())
        # Separately verify the head also updates at its configured learning rate.
        optimizer.optimizer.param_groups[1]['lr'] = q9.CONFIG['head_lr']
        features, _ = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence, True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(features / smoke_scale).reshape(()), torch.ones((), device='cuda'))
        loss.backward()
        optimizer.accumulate()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        changes = [n for n, p in block.named_parameters() if not torch.equal(p, original[n])]
        print('SMOKE_DIAGNOSTIC', float(loss.detach()), float(smoke_scale), norm,
              changes, bool(torch.equal(head.weight, head_before)), gradient_maxima, flush=True)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen gradient'
        assert parameter_hash(model, frozen_only=True) == frozen, 'Frozen weights changed'
        saved = {n: p.detach().cpu() for n, p in block.named_parameters()}
        path = OUTPUT / 'smoke_adapter.pt'
        torch.save({'block_index': block_index, 'block': saved, 'head': head.state_dict()}, path)
        expected = head(variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)[0] / smoke_scale).detach()
        with torch.no_grad():
            for n, p in block.named_parameters():
                p.copy_(original[n])
        restored = torch.load(path, weights_only=True)
        assert restored['block_index'] == block_index, 'Adapter targets the wrong block'
        with torch.no_grad():
            for n, p in block.named_parameters():
                p.copy_(restored['block'][n])
        head.load_state_dict(restored['head'])
        actual = head(variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)[0] / smoke_scale).detach()
        assert torch.equal(actual, expected), 'Adapter reload changed predictions'
        with torch.no_grad():
            for n, p in block.named_parameters():
                p.copy_(original[n])
        assert all(torch.equal(p, original[n]) for n, p in block.named_parameters())
        master_changes = sum(not torch.equal(a, b) for a, b in zip(optimizer.masters, original_masters))
        head_changed = not torch.equal(head.weight, head_before) or not torch.equal(head.bias, bias_before)
        failures = []
        if not changes or not master_changes:
            failures.append(f'The smoke optimizer changed no effective BF16 block-{block_index} weights')
        if not backbone_effective:
            failures.append(f'Block {block_index} updates did not change features and the fixed-head score')
        if not head_changed:
            failures.append('The classification head did not update')
        record = {'status': 'blocked' if failures else 'passed',
                  'blocker': '; '.join(failures) if failures else None,
                  'trainable_block': block_index,
                  'frozen_final_block': all(not p.requires_grad for p in model.decoder.layers[-1].parameters()),
                  'fixed_head_unchanged': fixed_head_unchanged,
                  'backbone_effective': backbone_effective,
                  'initial_fixed_head_score': float(initial_score), 'final_fixed_head_score': float(fixed_head_score),
                  'feature_relative_change': feature_relative_change,
                  'backbone_optimizer_steps': q9.CONFIG['readiness_optimizer_steps'],
                  'gradient_accumulation': q9.CONFIG['gradient_accumulation'],
                  'head_optimizer_steps': 1,
                  'loss': float(loss.detach()), 'gradient_norm': norm,
                  'variant_step_seconds': elapsed,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3,
                  'frozen_sha256': frozen, 'frozen_unchanged': True,
                  'reload_identical': True, 'original_weights_restored': True,
                  'trainable_parameters': {n: p.numel() for n, p in block.named_parameters()},
                  'backbone_parameters': sum(p.numel() for p in model.parameters()),
                  'head_parameters': sum(p.numel() for p in head.parameters()),
                  'updated_block_parameters': changes, 'updated_master_tensors': master_changes,
                  'head_updated': head_changed, 'gradient_maxima': gradient_maxima,
                  'smoke_feature_scale': float(smoke_scale),
                  'smoke_inputs': train_keys,
                  'smoke_head': 'seeded linear head; features divided by training-input standard deviation',
                  'smoke_target': 'fixed synthetic target 1; no validation label used'}
        q1.write_json(OUTPUT / 'preflight.json', record)
        print('PREFLIGHT', record, flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import sys
    preflight() if 'preflight' in sys.argv else smoke_forward()
