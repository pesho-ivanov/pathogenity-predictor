"""Q11 BioNeMo 7B backbone, pinned NeMo LoRA adapters and budgeted partial-epoch training."""

import os
import tempfile
import time

import numpy as np
import torch

from . import q1, q2, q10, q11


def convert():
    from nemo.collections.llm.gpt.model.hyena import PyTorchHyenaImporter, Hyena7bConfig
    q1.verify_file(q11.CHECKPOINT, q11.CONFIG['model_sha256'])
    PyTorchHyenaImporter(str(q11.CHECKPOINT), model_config=Hyena7bConfig()).apply(
        str(q11.CONVERTED), checkpoint_format='zarr')


def load_backbone():
    import importlib.metadata
    import random
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from nemo.collections.llm.gpt.model.hyena import Hyena7bConfig
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    environment = q1.read_json(q11.OUTPUT / 'environment/runtime.json')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('The BioNeMo inference runtime differs from the frozen environment')
    record = q1.read_json(q11.OUTPUT / 'converted_checkpoint.json')
    for name, checksum in record['files'].items():
        q1.verify_file(q11.CONVERTED / name, checksum)
    torch.set_num_threads(4)
    torch.manual_seed(q11.CONFIG['seed'])
    np.random.seed(q11.CONFIG['seed'])
    random.seed(q11.CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    handle, rendezvous = tempfile.mkstemp(prefix='q11-rendezvous-')
    os.close(handle)
    os.unlink(rendezvous)
    torch.distributed.init_process_group('nccl', init_method='file://' + rendezvous, rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(q11.CONFIG['seed'])
    tokenizer = get_nmt_tokenizer('byte-level')
    config = Hyena7bConfig(seq_length=q11.CONFIG['source_context_bp'], recompute_granularity=None,
        recompute_method=None, perform_initialization=False, use_cpu_initialization=True,
        gradient_accumulation_fusion=False, fp32_residual_connection=False, fp8=None)
    model = config.configure_model(tokenizer).cuda().to(torch.bfloat16)
    restored = dist_checkpointing.load({'state_dict': model.sharded_state_dict(prefix='module.')},
                                      q11.CONVERTED / 'weights')
    model.load_state_dict({name.removeprefix('module.'): value
                           for name, value in restored['state_dict'].items()}, strict=True)
    model.eval().requires_grad_(False)
    if len(model.decoder.layers) != q11.CONFIG['layers']:
        raise ValueError('Unexpected 7B backbone depth')
    return model, tokenizer


def encode_batch(model, tokenizer, sequences, gradients=False):
    length = q11.CONFIG['context_bp']
    if not sequences or any(len(s) != length or set(s)-set('ACGT') for s in sequences):
        raise ValueError('Expected the audited 512-base DNA contexts')
    ids = [tokenizer.text_to_ids(s) for s in sequences]
    if any(row != list(s.encode('ascii')) for row, s in zip(ids, sequences)):
        raise ValueError('Byte tokenizer changed')
    tokens = torch.tensor(ids, device='cuda', dtype=torch.long)
    positions = torch.arange(length, device='cuda')[None].expand(len(sequences), -1)
    pooled = []
    def capture(module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        pooled.append(value.float().mean(dim=0))
    hook = model.decoder.layers[-1].register_forward_hook(capture)
    try:
        with torch.set_grad_enabled(gradients):
            logits = model(tokens, positions, attention_mask=None)
            del logits
        value = pooled[-1]
        if value.shape != (len(sequences), q11.CONFIG['hidden_size']) or not torch.isfinite(value).all():
            raise ValueError('Invalid 7B activations')
        return value
    finally:
        hook.remove()


def encode(model, tokenizer, sequence, gradients=False):
    return encode_batch(model, tokenizer, [sequence], gradients)[0]


def normalize_pair(reference, difference):
    """Fixed per-example scaling; no statistics learned from either partition."""
    def normalize(value):
        return value / value.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    return torch.cat([normalize(reference), normalize(difference)], dim=-1)


def batch_features(model, tokenizer, pairs, gradients=False):
    sequences = [s for ref, alt in pairs for s in q11.crop_pair(ref, alt)]
    values = encode_batch(model, tokenizer, sequences, gradients)
    return normalize_pair(values[0::2], values[1::2] - values[0::2])


def variant_features(model, tokenizer, reference, alternate, gradients=False):
    return batch_features(model, tokenizer, [(reference, alternate)], gradients)[0]


def audit_conversion(model):
    """Check every loaded parameter against the independently named source tensors."""
    q1.verify_file(q11.CHECKPOINT, q11.CONFIG['model_sha256'])
    source = torch.load(q11.CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)['module']
    names = {'mixer.dense_projection.layer_norm_weight': 'input_layernorm.weight',
             'self_attention.linear_qkv.layer_norm_weight': 'input_layernorm.weight',
             'self_attention.linear_qkv.weight': 'mixer.dense_projection.weight',
             'self_attention.linear_proj.weight': 'mixer.dense.weight',
             'self_attention.linear_proj.bias': 'mixer.dense.bias',
             'mlp.linear_fc1.layer_norm_weight': 'pre_mlp_layernorm.weight',
             'mlp.linear_fc2.weight': 'mlp.w3.weight'}
    rows = []
    tensors = list(model.named_parameters()) + [(n, p) for n, p in model.named_buffers() if n.endswith('filter.decay')]
    for name, value in tensors:
        if name == 'embedding.word_embeddings.weight':
            expected = source['sequential.0.word_embeddings.weight']
        elif name == 'decoder.final_norm.weight':
            expected = source[f"sequential.{q11.CONFIG['layers']+3}.norm.weight"]
        else:
            _, _, layer, suffix = name.split('.', 3)
            prefix = f'sequential.{int(layer)+2}.'
            if suffix == 'mlp.linear_fc1.weight':
                expected = torch.cat([source[prefix+'mlp.w1.weight'], source[prefix+'mlp.w2.weight']], dim=0)
            else:
                expected = source[prefix+names.get(suffix, suffix)]
            if suffix.endswith(('filter.h', 'filter.decay')):
                expected = expected[:, :value.shape[-1]]
        equal = torch.equal(value.detach().cpu(), expected.to(value.dtype))
        rows.append({'name': name, 'elements': value.numel(), 'exact': equal})
        if not equal:
            raise ValueError(f'Converted tensor differs from source: {name}')
    return {'passed': True, 'tensors_checked': len(rows), 'tensors': rows}


def initialize_adapter(adapter):
    """The base loader disables initialization; initialize new matrices explicitly."""
    if q11.CONFIG['lora']['A_init'] != 'xavier' or q11.CONFIG['lora']['B_init'] != 'zero':
        raise ValueError('This experiment requires Xavier A and zero B initialization')
    with torch.no_grad():
        torch.nn.init.xavier_normal_(adapter.linear_in.weight)
        torch.nn.init.zeros_(adapter.linear_out.weight)


def attach_lora(model):
    """Wrap only the two selected mixer projections; freeze every base tensor."""
    from nemo.collections.llm.peft.lora import LoRA, LoRALinear
    if len(model.decoder.layers) != q11.CONFIG['layers']:
        raise ValueError('Unexpected backbone depth')
    model.requires_grad_(False)
    before = parameter_hash(model)
    config = q11.CONFIG['lora']
    transform = LoRA(target_modules=config['targets'], dim=config['rank'], alpha=config['alpha'],
                     dropout=config['dropout'], lora_A_init_method=config['A_init'],
                     lora_B_init_method=config['B_init'], lora_dtype=torch.bfloat16)
    adapters = torch.nn.ModuleDict()
    expected_count = 0
    for target in config['targets']:
        prefix, _, name = target.rpartition('.')
        parent = model.get_submodule(prefix)
        base = getattr(parent, name)
        if hasattr(base, 'adapter'):
            raise ValueError('LoRA is already attached')
        expected_count += config['rank'] * (base.in_features + base.out_features)
        wrapped = transform.transform(base, name=name, prefix=prefix)
        if not isinstance(wrapped, LoRALinear):
            raise RuntimeError(f'Pinned NeMo did not wrap {target}')
        wrapped.to_wrap.requires_grad_(False)
        wrapped.adapter.to(device=base.weight.device, dtype=torch.bfloat16).requires_grad_(True)
        initialize_adapter(wrapped.adapter)
        setattr(parent, name, wrapped)
        adapters[name] = wrapped.adapter
    parameters = dict(adapters.named_parameters())
    if sum(p.numel() for p in parameters.values()) != expected_count:
        raise RuntimeError('Adapter parameter count differs from rank × projection dimensions')
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters.values()}:
        raise RuntimeError('Trainable parameters are not exactly the selected LoRA adapters')
    if any(not torch.equal(adapter.linear_out.weight, torch.zeros_like(adapter.linear_out.weight))
           for adapter in adapters.values()):
        raise RuntimeError('LoRA B must initialize to zero')
    if parameter_hash(model, frozen_only=True) != before:
        raise RuntimeError('Adding adapters changed the original backbone tensors')
    return adapters


def parameter_hash(model, frozen_only=False):
    import hashlib
    digest = hashlib.sha256()
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if frozen_only and p.requires_grad:
            continue
        # Pinned NeMo wraps the same original tensor under `.to_wrap`.
        digest.update(name.replace('.to_wrap.', '.').encode())
        digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class MasterAdamW:
    """Accumulate variant gradients in FP32, including the final partial batch."""

    def __init__(self, adapters, head, config=None):
        config = q11.CONFIG if config is None else config
        self.parameters = list(adapters.parameters())
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError('Optimizer requires an explicit trainable adapter-only module')
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.accumulated = [torch.zeros_like(p) for p in self.masters]
        self.head, self.clip_grad, self.pending = head, config['clip_grad'], 0
        self.optimizer = torch.optim.AdamW([
            {'params': self.masters, 'lr': config['adapter_lr']},
            {'params': head.parameters(), 'lr': config['head_lr']}],
            betas=tuple(config['betas']), eps=config['epsilon'], weight_decay=config['weight_decay'])

    def accumulate(self, variants=1):
        if not isinstance(variants, int) or variants <= 0:
            raise ValueError("Positive variant count required")
        for p, accumulator in zip(self.parameters, self.accumulated):
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError('Missing or non-finite LoRA gradient')
            accumulator.add_(p.grad.float())
            p.grad = None
        self.pending += variants

    def step(self):
        if self.pending == 0:
            raise ValueError('No variants accumulated')
        for master, accumulator in zip(self.masters, self.accumulated):
            master.grad = accumulator / self.pending
        for p in self.head.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError('Missing or non-finite classifier gradient')
            p.grad.div_(self.pending)
        norm = torch.nn.utils.clip_grad_norm_(self.masters + list(self.head.parameters()),
                                              self.clip_grad, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for p, master, accumulator in zip(self.parameters, self.masters, self.accumulated):
                p.copy_(master)
                if not torch.isfinite(p).all():
                    raise RuntimeError('Non-finite updated LoRA weight')
                accumulator.zero_()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        return float(norm)


def rng_state():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    import random
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'megatron': get_cuda_rng_tracker().get_states()}


def restore_rng(state):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    import random
    random.setstate(state['python'])
    name, values, *tail = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), *tail))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    get_cuda_rng_tracker().set_states({k: v.cpu() for k, v in state['megatron'].items()})


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(v) for v in value)
    return value


def save_checkpoint(path, block, head, optimizer, mean, scale, identity, progress):
    import hashlib
    import io
    if optimizer is not None and optimizer.pending:
        raise ValueError('Checkpoint must be at an optimizer boundary')
    state = {'format': 'q11-lora-v2-partial', 'identity': identity, 'lora': q11.CONFIG['lora'],
             'adapters': cpu_state(dict(block.named_parameters())), 'head': cpu_state(head.state_dict()),
             'mean': mean.cpu(), 'scale': scale.cpu(), 'progress': progress,
             'rng': cpu_state(rng_state())}
    if optimizer is not None:
        state.update(masters=cpu_state(optimizer.masters), optimizer=cpu_state(optimizer.optimizer.state_dict()))
    temporary = path.with_suffix('.partial.pt')
    payload = io.BytesIO()
    torch.save(state, payload)
    with temporary.open('wb') as stream:
        stream.write(hashlib.sha256(payload.getbuffer()).hexdigest().encode() + b'\n')
        stream.write(payload.getbuffer())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_checkpoint(path, identity):
    import hashlib
    import io
    with path.open('rb') as stream:
        expected = stream.readline(65).strip().decode('ascii')
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (state.get('format') != 'q11-lora-v2-partial' or state['identity'] != identity
            or state.get('lora') != q11.CONFIG['lora']):
        raise ValueError('Checkpoint belongs to another experiment or LoRA configuration; legacy block checkpoints are rejected')
    def check(value):
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError('Non-finite checkpoint tensor')
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                check(v)
    check(state)
    return state


def restore_checkpoint(state, block, head, optimizer, mean, scale):
    if set(state['adapters']) != set(dict(block.named_parameters())):
        raise ValueError('Checkpoint parameter names changed')
    if not torch.equal(state['mean'], mean.cpu()) or not torch.equal(state['scale'], scale.cpu()):
        raise ValueError('Training-only scaler changed')
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.copy_(state['adapters'][name])
        head.load_state_dict(state['head'], strict=True)
        if optimizer is not None:
            if len(state['masters']) != len(optimizer.masters):
                raise ValueError('Optimizer parameter count changed')
            for p, master, saved in zip(optimizer.parameters, optimizer.masters, state['masters']):
                if not torch.equal(p.cpu(), saved.to(p.dtype)):
                    raise ValueError('Saved master and deployed weights disagree')
                master.copy_(saved)
            optimizer.optimizer.load_state_dict(state['optimizer'])
            optimizer.optimizer.zero_grad(set_to_none=True)
            for accumulator in optimizer.accumulated:
                accumulator.zero_()
            optimizer.pending = 0
    restore_rng(state['rng'])


def preflight(model, tokenizer, dna, train_indexes, identity):
    conversion = audit_conversion(model)
    first, second = dna.iloc[train_indexes[:2]].itertuples(index=False)
    first_ref = q11.crop_pair(first.ref_sequence, first.alt_sequence)[0]
    second_ref = q11.crop_pair(second.ref_sequence, second.alt_sequence)[0]
    before = encode(model, tokenizer, first_ref)
    encode(model, tokenizer, second_ref)
    after = encode(model, tokenizer, first_ref)
    features = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
    reverse = variant_features(model, tokenizer, q1.reverse_complement(first.ref_sequence),
                               q1.reverse_complement(first.alt_sequence))
    if not torch.equal(before, after) or not torch.equal(features, reverse):
        raise RuntimeError('A/B/A inference or strand invariance failed')
    base_parameters = sum(p.numel() for p in model.parameters())
    block = attach_lora(model)
    zero_features = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
    if not torch.equal(features, zero_features):
        raise RuntimeError('Zero-initialized LoRA changes frozen features')
    original = cpu_state(dict(block.named_parameters()))
    frozen = parameter_hash(model, frozen_only=True)
    head = torch.nn.Linear(q11.CONFIG['feature_dimension'], 1, device='cuda', dtype=torch.float32)
    initial_rng = rng_state()
    optimizer = MasterAdamW(block, head)
    # Hold the head fixed to prove that the backbone itself can change a score.
    optimizer.optimizer.param_groups[1]['lr'] = 0.
    optimizer.optimizer.param_groups[0]['weight_decay'] = 0.
    scale = features.std().clamp_min(1e-6)
    initial_score = head(features / scale).detach()
    maxima = {n: 0. for n, _ in block.named_parameters()}
    first_gradients = {}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for step in range(q11.CONFIG['preflight_steps']):
            value = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence, True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                head(value / scale).reshape(()), torch.ones((), device='cuda'))
            loss.backward()
            for name, p in block.named_parameters():
                if p.grad is not None:
                    maxima[name] = max(maxima[name], float(p.grad.abs().max()))
                    if step == 0:
                        first_gradients[name] = float(p.grad.abs().max())
            optimizer.accumulate()
            optimizer.step()
        changed_score = head(variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence) / scale)
        if not torch.isfinite(changed_score).all() or torch.equal(initial_score, changed_score):
            raise RuntimeError('LoRA updates do not change a fixed-head prediction')
        if any(value == 0 for value in maxima.values()):
            raise RuntimeError('A selected LoRA parameter has no nonzero gradient across the probe steps')
        if all(torch.equal(p.cpu(), original[n]) for n, p in block.named_parameters()):
            raise RuntimeError('No LoRA weights changed')
        head_before = cpu_state(head.state_dict())
        optimizer.optimizer.param_groups[1]['lr'] = q11.CONFIG['head_lr']
        value = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence, True)
        torch.nn.functional.binary_cross_entropy_with_logits(
            head(value / scale).reshape(()), torch.ones((), device='cuda')).backward()
        optimizer.accumulate()
        optimizer.step()
        if all(torch.equal(p.cpu(), head_before[n]) for n, p in head.named_parameters()):
            raise RuntimeError('Classifier did not update')
        if parameter_hash(model, frozen_only=True) != frozen:
            raise RuntimeError('Frozen model state changed during preflight')
        result = {'identity': identity, 'status': 'passed', 'conversion': conversion,
                  'probe_keys': [first.variant_key, second.variant_key], 'probe_split': 'train',
                  'A_B_A_exact': True, 'strand_invariance_exact': True, 'frozen_unchanged': True,
                  'zero_adapter_features_exact': True, 'lora': q11.CONFIG['lora'],
                  'first_step_gradients': first_gradients, 'gradient_maxima': maxima,
                  'fixed_head_score_change': float((changed_score - initial_score).detach()),
                  'trainable_adapter_parameters': sum(p.numel() for p in block.parameters()),
                  'total_backbone_parameters': base_parameters,
                  'seconds': time.perf_counter() - start,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
                  'limits': 'Tensor mapping and execution checked; upstream numerical parity and pretraining independence unresolved.'}
    finally:
        with torch.no_grad():
            for name, p in block.named_parameters():
                p.copy_(original[name])
        restore_rng(initial_rng)
        del optimizer, head
    restored = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
    if not torch.equal(restored, features):
        raise RuntimeError('Restoring probe adapters did not restore frozen features')
    result['probe_adapters_restored'] = True
    q1.write_json(q11.OUTPUT / 'preflight.json', result)
    print('Conversion, repeatability, strand, gradient and actual update checks passed.', flush=True)
    return result, block


def pairs_from(rows):
    return list(rows[['ref_sequence', 'alt_sequence']].itertuples(index=False, name=None))


def predict(model, tokenizer, head, mean, scale, dna, progress=True):
    values = []
    size = q11.CONFIG['microbatch_variants']
    with torch.no_grad():
        for offset in range(0, len(dna), size):
            rows = dna.iloc[offset:offset + size]
            features = batch_features(model, tokenizer, pairs_from(rows))
            value = head((features - mean) / scale).flatten()
            if not torch.isfinite(value).all():
                raise RuntimeError('Non-finite validation score')
            values.append(value.cpu().numpy())
            if progress and (offset // size % 16 == 0 or offset + len(rows) == len(dna)):
                print(f'Validation: {offset + len(rows)}/{len(dna)}', flush=True)
    return np.concatenate(values)


def validation_reserve(seconds_per_variant, count):
    return (seconds_per_variant * (count + q11.CONFIG['reload_variants']) *
            q11.CONFIG['validation_time_margin'] + q11.CONFIG['report_reserve_seconds'])


def calibrate(model, tokenizer, dna, indexes):
    rows = dna.iloc[indexes[:q11.CONFIG['microbatch_variants']]]
    pairs = pairs_from(rows)
    batch_features(model, tokenizer, pairs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        batch_features(model, tokenizer, pairs)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / (3 * len(rows))
    # Batching must preserve the feature definition before any training updates.
    batched = batch_features(model, tokenizer, pairs[:2])
    serial = torch.stack([variant_features(model, tokenizer, *pair) for pair in pairs[:2]])
    if not torch.allclose(batched, serial, atol=2e-4, rtol=2e-4):
        raise RuntimeError('Batched and single-variant features disagree')
    return {'seconds_per_variant': seconds, 'probe_keys': rows.variant_key.tolist(),
            'probe_split': 'train', 'batch_equivalence_verified': True}


def validate_progress(progress, count):
    order = np.asarray(progress['order'], dtype=int)
    offset = progress['offset']
    size = q11.CONFIG['microbatch_variants']
    if sorted(order.tolist()) != list(range(count)) or not 0 <= offset <= count:
        raise ValueError('Invalid checkpoint permutation or offset')
    if offset != count and offset % size:
        raise ValueError('Checkpoint is not at an optimizer boundary')
    expected = np.random.default_rng(q11.CONFIG['seed']).permutation(count)
    if not np.array_equal(order, expected) or progress['epoch'] != 1:
        raise ValueError('Checkpoint shuffle or epoch changed')
    if progress.get('steps', (offset + size - 1)//size) != (offset + size - 1)//size:
        raise ValueError('Checkpoint update count differs from its cursor')
    return order, offset


def train_limited(model, tokenizer, adapters, head, mean, scale, train, labels, identity, deadline):
    import gc
    optimizer = MasterAdamW(adapters, head)
    frozen = parameter_hash(model, frozen_only=True)
    initial = parameter_hash(adapters)
    weights = len(labels) / (2 * np.bincount(labels, minlength=2))
    path = q11.OUTPUT / 'last_checkpoint.pt'
    progress = {'epoch': 1, 'order': np.random.default_rng(q11.CONFIG['seed']).permutation(len(train)).tolist(),
                'offset': 0, 'steps': 0, 'loss_sum': 0., 'seconds': 0., 'frozen_sha256': frozen,
                'adapter_initial_sha256': initial, 'class_weights': weights.tolist(), 'peak_gpu_gib': 0.,
                'history': []}
    if path.exists():
        state = load_checkpoint(path, identity)
        progress = state['progress']
        if (progress['frozen_sha256'] != frozen or progress['class_weights'] != weights.tolist()
                or progress['adapter_initial_sha256'] != initial):
            raise ValueError('Frozen backbone, initial adapters or class weights changed')
        restore_checkpoint(state, adapters, head, optimizer, mean, scale)
        del state
    order, offset = validate_progress(progress, len(train))
    torch.cuda.reset_peak_memory_stats()
    size = q11.CONFIG['microbatch_variants']
    # Cumulative compute cannot exceed the training budget across a resume.
    train_deadline = min(deadline, time.time() + max(0, q11.CONFIG['training_seconds']-progress['seconds']))
    print(f'Train: resume at {offset}/{len(train)}; maximum {q11.CONFIG["max_steps"]} updates.', flush=True)
    next_seconds = 5.
    reason = 'training_pool_exhausted'
    for start in range(offset, len(train), size):
        if progress['steps'] >= q11.CONFIG['max_steps']:
            reason = 'max_steps'
            break
        if time.time() + next_seconds >= train_deadline:
            reason = 'time_budget'
            break
        tick = time.perf_counter()
        batch = order[start:start + size]
        features = batch_features(model, tokenizer, pairs_from(train.iloc[batch]), True)
        targets = torch.tensor(labels[batch], dtype=torch.float32, device='cuda')
        sample_weights = torch.tensor(weights[labels[batch]], dtype=torch.float32, device='cuda')
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            head((features - mean) / scale).flatten(), targets, reduction='none') * sample_weights
        loss = losses.sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.accumulate(len(batch))
        norm = optimizer.step()
        progress['loss_sum'] += float(loss.detach())
        progress['offset'] = start + len(batch)
        progress['steps'] += 1
        elapsed = time.perf_counter() - tick
        progress['seconds'] += elapsed
        next_seconds = max(elapsed * 1.5, 5.)
        progress['peak_gpu_gib'] = max(progress['peak_gpu_gib'], torch.cuda.max_memory_allocated() / 1024**3)
        if progress['steps'] % q11.CONFIG['checkpoint_steps'] == 0:
            progress['history'].append({'steps': progress['steps'], 'variants': progress['offset'],
                                        'mean_loss': progress['loss_sum']/progress['offset'],
                                        'last_gradient_norm': norm, 'seconds': progress['seconds']})
            save_checkpoint(path, adapters, head, optimizer, mean, scale, identity, progress)
            print(f'Train: {progress["steps"]}/{q11.CONFIG["max_steps"]} updates; '
                  f'{progress["offset"]}/{len(train)} variants; '
                  f'loss={progress["loss_sum"]/progress["offset"]:.4f}; '
                  f'{progress["seconds"]/60:.1f} min', flush=True)
    progress['stop_reason'] = reason
    if progress['offset'] == 0 or parameter_hash(adapters) == initial:
        raise RuntimeError('Budget ended without any LoRA weight changes')
    if parameter_hash(model, frozen_only=True) != frozen:
        raise RuntimeError('Frozen backbone changed during training')
    save_checkpoint(path, adapters, head, optimizer, mean, scale, identity, progress)
    save_checkpoint(q11.OUTPUT / 'final_adapter.pt', adapters, head, None, mean, scale, identity, progress)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return progress


def run():
    import pandas as pd
    wall_start = float(os.environ.get('Q11_RUN_STARTED_UTC', time.time()))
    identity = q11.verify_protocol()
    _, manifest, dna, indexes, labels, _ = q11.verify_inputs()
    if (q11.OUTPUT / 'metrics.json').exists():
        q11.verified_results()
        print('Verified completed Q11 experiment; every notebook cell still executes.', flush=True)
        return
    model, tokenizer = load_backbone()
    try:
        preflight_result, adapters = preflight(model, tokenizer, dna, indexes['train'], identity)
        calibration = calibrate(model, tokenizer, dna, indexes['train'])
        reserve = validation_reserve(calibration['seconds_per_variant'], len(labels['validation']))
        train_deadline = wall_start + q11.CONFIG['total_seconds'] - reserve
        calibration.update(validation_and_report_reserve_seconds=reserve,
                           seconds_remaining_for_training=max(0, train_deadline-time.time()))
        q1.write_json(q11.OUTPUT/'calibration.json', calibration)
        print(f'Calibration: {calibration["seconds_per_variant"]:.4f} s/variant; '
              f'reserve {reserve/60:.1f} min for full validation and reporting.', flush=True)
        if train_deadline - time.time() < 60:
            raise RuntimeError('Insufficient one-hour budget after measured full-validation reserve')
        head = torch.nn.Linear(q11.CONFIG['feature_dimension'], 1, device='cuda', dtype=torch.float32)
        mean = torch.zeros(q11.CONFIG['feature_dimension'], device='cuda')
        scale = torch.ones_like(mean)
        progress = train_limited(model, tokenizer, adapters, head, mean, scale,
                                 dna.iloc[indexes['train']], labels['train'], identity, train_deadline)
        validation = dna.iloc[indexes['validation']]
        start = time.perf_counter()
        predictions = predict(model, tokenizer, head, mean, scale, validation)
        state = load_checkpoint(q11.OUTPUT / 'final_adapter.pt', identity)
        with torch.no_grad():
            for p in list(adapters.parameters()) + list(head.parameters()):
                p.zero_()
        restore_checkpoint(state, adapters, head, None, mean, scale)
        count = min(q11.CONFIG['reload_variants'], len(validation))
        restored = predict(model, tokenizer, head, mean, scale, validation.iloc[:count], progress=False)
        if not np.allclose(predictions[:count], restored, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Reloaded checkpoint does not reproduce the fixed 64-variant validation probe')
        inference_seconds = time.perf_counter() - start
        scores = {'fine_tuned': predictions}
        groups = manifest.loc[manifest.split.eq('validation'), 'component'].to_numpy()
        metrics = q11.evaluate(labels['validation'], scores, groups)
        pd.DataFrame({'variant_key': validation.variant_key.to_numpy(), 'component': groups,
                      'label': labels['validation'], **scores}).to_csv(q11.OUTPUT/'comparison_predictions.csv', index=False)
        train = dna.iloc[indexes['train']]
        train.iloc[np.asarray(progress['order'][:progress['offset']])][['variant_key']].to_csv(
            q11.OUTPUT/'training_seen.csv', index=False)
        q1.write_json(q11.OUTPUT/'training_history.json', progress)
        artifact_names = ['protocol.json', 'preflight.json', 'calibration.json', 'final_adapter.pt',
                          'last_checkpoint.pt', 'training_history.json', 'training_seen.csv', 'comparison_predictions.csv']
        wall_seconds = time.time() - wall_start
        result = {'status': 'complete', 'identity': identity, 'training_mode': 'partial_epoch',
                  'epoch_fraction': progress['offset']/len(train), 'metrics': metrics,
                  'training_variants_seen': progress['offset'], 'training_pool_variants': len(train),
                  'validation_variants': len(validation), 'optimizer_steps': progress['steps'],
                  'train_loss': progress['loss_sum']/progress['offset'], 'stop_reason': progress['stop_reason'],
                  'lora': q11.CONFIG['lora'], 'wall_seconds_to_metrics': wall_seconds,
                  'within_one_hour_to_metrics': wall_seconds <= q11.CONFIG['total_seconds'],
                  'trainable_parameters': preflight_result['trainable_adapter_parameters'] + sum(p.numel() for p in head.parameters()),
                  'peak_gpu_gib': max(progress['peak_gpu_gib'], preflight_result['peak_gpu_gib']),
                  'reloaded_predictions_verified': True, 'reload_variants': count, 'frozen_unchanged': True,
                  'runtimes': {'fine_tuned': {'seconds': wall_seconds,
                      'scope': 'Fresh notebook start through input checks, cached setup verification, preflight, partial-epoch training, full validation, 64-variant reload check and bootstrap; excludes initial environment/model acquisition and final notebook/README export.'}},
                  'inference_seconds': inference_seconds, 'limitations': q11.LIMITATIONS,
                  'artifacts': {n: q1.digest_file(q11.OUTPUT/n) for n in artifact_names}}
        q1.write_json(q11.OUTPUT/'metrics.json', result, frozen=True)
        print(f'Completed partial-epoch experiment in {wall_seconds/60:.1f} minutes.', flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['convert', 'run'])
    arguments = parser.parse_args()
    {'convert': convert, 'run': run}[arguments.action]()
