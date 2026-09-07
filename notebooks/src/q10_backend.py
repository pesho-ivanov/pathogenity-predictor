"""BioNeMo 7B conversion and resumable frozen inference in Q9's isolated runtime."""

import os
import tempfile
import time

import numpy as np
import torch

from . import q1, q2, q10


def convert():
    from nemo.collections.llm.gpt.model.hyena import PyTorchHyenaImporter, Hyena7bConfig
    q1.verify_file(q10.CHECKPOINT, q10.CONFIG['model_sha256'])
    PyTorchHyenaImporter(str(q10.CHECKPOINT), model_config=Hyena7bConfig()).apply(
        str(q10.CONVERTED), checkpoint_format='zarr')


def load_backbone():
    import importlib.metadata
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from nemo.collections.llm.gpt.model.hyena import Hyena7bConfig
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    environment = q1.read_json(q10.OUTPUT / 'environment/runtime.json')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('The BioNeMo inference runtime differs from the frozen environment')
    record = q1.read_json(q10.OUTPUT / 'converted_checkpoint.json')
    for name, checksum in record['files'].items():
        q1.verify_file(q10.CONVERTED / name, checksum)
    torch.set_num_threads(4)
    torch.manual_seed(q10.CONFIG['seed'])
    np.random.seed(q10.CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    handle, rendezvous = tempfile.mkstemp(prefix='q10-rendezvous-')
    os.close(handle)
    os.unlink(rendezvous)
    torch.distributed.init_process_group('nccl', init_method='file://' + rendezvous, rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(q10.CONFIG['seed'])
    tokenizer = get_nmt_tokenizer('byte-level')
    config = Hyena7bConfig(seq_length=q10.CONFIG['context_bp'], recompute_granularity=None,
        recompute_method=None, perform_initialization=False, use_cpu_initialization=True,
        gradient_accumulation_fusion=False, fp32_residual_connection=False, fp8=None)
    model = config.configure_model(tokenizer).cuda().to(torch.bfloat16)
    restored = dist_checkpointing.load({'state_dict': model.sharded_state_dict(prefix='module.')},
                                      q10.CONVERTED / 'weights')
    model.load_state_dict({name.removeprefix('module.'): value
                           for name, value in restored['state_dict'].items()}, strict=True)
    model.eval().requires_grad_(False)
    if len(model.decoder.layers) != q10.CONFIG['layers']:
        raise ValueError('Unexpected 7B backbone depth')
    return model, tokenizer


def encode(model, tokenizer, sequence):
    if len(sequence) != q10.CONFIG['context_bp'] or set(sequence)-set('ACGT'):
        raise ValueError('Expected the frozen 1,024-base DNA context')
    tokens = torch.tensor([tokenizer.text_to_ids(sequence)], device='cuda', dtype=torch.long)
    if tokens[0].tolist() != list(sequence.encode('ascii')):
        raise ValueError('Byte tokenizer changed')
    positions = torch.arange(len(sequence), device='cuda').unsqueeze(0)
    pooled = []
    def capture(module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        pooled.append(value.float().mean(dim=0)[0])
    hook = model.decoder.layers[-1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            logits = model(tokens, positions, attention_mask=None)
            likelihood = torch.log_softmax(logits.float(), dim=-1)[:, :-1].gather(2, tokens[:, 1:, None]).mean()
        value = pooled[-1]
        if value.shape != (q10.CONFIG['hidden_size'],) or not torch.isfinite(value).all() or not torch.isfinite(likelihood):
            raise ValueError('Invalid 7B activations or log likelihood')
        return value, likelihood
    finally:
        hook.remove()


def variant_features(model, tokenizer, reference, alternate):
    pooled = []
    for sequence in [reference, alternate]:
        canonical = min(sequence, q1.reverse_complement(sequence))
        forward, _ = encode(model, tokenizer, canonical)
        reverse, _ = encode(model, tokenizer, q1.reverse_complement(canonical))
        pooled.append((forward+reverse)/2)
    return torch.cat([pooled[0], pooled[1]-pooled[0]])


def audit_conversion(model):
    """Check every loaded parameter against the independently named source tensors."""
    q1.verify_file(q10.CHECKPOINT, q10.CONFIG['model_sha256'])
    source = torch.load(q10.CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)['module']
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
            expected = source[f"sequential.{q10.CONFIG['layers']+3}.norm.weight"]
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


def preflight(model, tokenizer, identity, dna, pilot):
    conversion = audit_conversion(model)
    train = dna.set_index('variant_key').loc[pilot.loc[pilot.split.eq('train'), 'variant_key'].iloc[:2]]
    first, second = train.iloc[0], train.iloc[1]
    before, likelihood = encode(model, tokenizer, first.ref_sequence)
    encode(model, tokenizer, second.ref_sequence)
    after, repeated_likelihood = encode(model, tokenizer, first.ref_sequence)
    if not torch.equal(before, after) or not torch.equal(likelihood, repeated_likelihood):
        raise ValueError('A/B/A inference repeatability failed')
    paired = variant_features(model, tokenizer, first.ref_sequence, first.alt_sequence)
    reverse = variant_features(model, tokenizer, q1.reverse_complement(first.ref_sequence),
                               q1.reverse_complement(first.alt_sequence))
    if not torch.equal(paired, reverse) or paired.shape != (q10.CONFIG['feature_dimension'],):
        raise ValueError('Strand invariance or paired feature dimensions failed')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen backbone contains trainable parameters')
    result = {'identity': identity, 'status': 'passed', 'conversion': conversion,
              'A_B_A_exact': True, 'strand_invariance_exact': True, 'backbone_frozen': True,
              'probe_split': 'train', 'probe_keys': train.index.tolist(),
              'mean_log_likelihood': float(likelihood),
              'pooled_rms': float(before.double().square().mean().sqrt()),
              'limits': 'Checks tensor mapping and deterministic execution, not upstream numerical parity or pretraining contamination.'}
    q1.write_json(q10.OUTPUT / 'preflight.json', result)
    print('7B conversion and frozen inference checks passed.', flush=True)


def extract():
    identity = q10.verify_protocol()
    _, pilot, dna, _, _, _ = q10.q9.verify_inputs()
    q10.q9.assert_feature_inputs(dna, dna)
    marker = q10.OUTPUT / 'feature_manifest.json'
    if marker.exists():
        q10.load_features()
        print('Verified completed 7B feature cache.', flush=True)
        return
    folder = q10.OUTPUT / 'features'
    folder.mkdir(exist_ok=True)
    model, tokenizer = load_backbone()
    try:
        preflight(model, tokenizer, identity, dna, pilot)
        torch.cuda.reset_peak_memory_stats()
        batches = {}
        size = q10.CONFIG['cache_batch_variants']
        for offset in range(0, len(dna), size):
            rows = dna.iloc[offset:offset+size]
            name = f'features/{offset:05d}.npz'
            path = q10.OUTPUT / name
            record_path = path.with_suffix('.json')
            if record_path.exists():
                record = q1.read_json(record_path)
                q10.read_feature_batch(path, record, rows.variant_key.tolist(), identity)
            else:
                started = time.perf_counter()
                values = np.stack([variant_features(model, tokenizer, row.ref_sequence, row.alt_sequence).cpu().numpy()
                                   for row in rows.itertuples()])
                q2.save_npz(path, keys=rows.variant_key.to_numpy(dtype=str), evo=values)
                record = {'identity': identity, 'keys': rows.variant_key.tolist(), 'sha256': q1.digest_file(path),
                          'seconds': time.perf_counter()-started,
                          'peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3}
                q1.write_json(record_path, record)
            batches[name] = record
            print(f"7B features: {offset+len(rows)}/{len(dna)}; batch {record['seconds']:.1f} s", flush=True)
        q1.write_json(marker, {'identity': identity, 'variants': len(dna), 'batches': batches,
                              'seconds': sum(r['seconds'] for r in batches.values()),
                              'peak_gpu_gib': max(r['peak_gpu_gib'] for r in batches.values())}, frozen=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['convert', 'extract'])
    args = parser.parse_args()
    {'convert': convert, 'extract': extract}[args.action]()
