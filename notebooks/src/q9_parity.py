"""Pre-update cross-backend checks, in separate processes to avoid operator conflicts."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from . import q1, q2

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'notebooks/results/q9'


def audit_parameters():
    """Audit all Vortex checkpoint tensors against the pinned Savanna source."""
    from huggingface_hub import hf_hub_download
    source_path = ROOT / 'data/evo2-savanna-1b/savanna_evo2_1b_base.pt'
    vortex_path = hf_hub_download(q2.MODEL_REPO, 'evo2_1b_base.pt', revision=q2.MODEL_REVISION)
    q1.verify_file(source_path, '7bb731473c99db72aba34e7b0df443f2f422e12b0669bea6d78415dc87057a08')
    q1.verify_file(vortex_path, q2.MODEL_SHA256)
    # These two hash-verified official checkpoints contain trusted BytesIO metadata.
    source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=False)['module']
    vortex = torch.load(vortex_path, map_location='cpu', mmap=True, weights_only=False)
    compared, differences = [], []
    simple = {'pre_norm.scale': 'input_layernorm.weight', 'post_norm.scale': 'pre_mlp_layernorm.weight',
              'mlp.l1.weight': 'mlp.w1.weight', 'mlp.l2.weight': 'mlp.w2.weight',
              'mlp.l3.weight': 'mlp.w3.weight', 'projections.weight': 'mixer.dense_projection.weight',
              'out_filter_dense.weight': 'mixer.dense.weight', 'out_filter_dense.bias': 'mixer.dense.bias',
              'filter.D': 'mixer.mixer.conv_bias', 'filter.residues': 'mixer.mixer.filter.R',
              'inner_mha_cls.out_proj.weight': 'mixer.dense.weight',
              'inner_mha_cls.out_proj.bias': 'mixer.dense.bias',
              'inner_mha_cls.Wqkv.weight': 'mixer.dense_projection.weight',
              'inner_mha_cls.rotary_emb.inv_freq': 'mixer.rotary_emb.inv_freq'}
    for name, value in vortex.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name in ['embedding_layer.weight', 'unembed.weight']:
            expected = source['sequential.0.word_embeddings.weight']
        elif name == 'norm.scale':
            expected = source['sequential.28.norm.weight']
        else:
            _, layer, suffix = name.split('.', 2)
            prefix = f'sequential.{int(layer)+2}.'
            if suffix in simple:
                expected = source[prefix + simple[suffix]]
            elif suffix == 'filter.short_filter_weight':
                expected = source[prefix + 'mixer.hyena_proj_conv.short_conv_weight'].unsqueeze(1)
            elif suffix == 'filter.log_poles':
                p = source[prefix + 'mixer.mixer.filter.p'].float()
                gamma = source[prefix + 'mixer.mixer.filter.gamma'].float()
                expected = (-torch.exp(p) * torch.exp(gamma)).unsqueeze(-1)
            elif suffix == 'filter.h':
                short = prefix + 'mixer.mixer.short_conv.short_conv_weight'
                if short in source:
                    expected = source[short]
                else:
                    h = source[prefix + 'mixer.mixer.filter.h'][:, :128]
                    decay = source[prefix + 'mixer.mixer.filter.decay'][:, :128]
                    expected = (h * decay).unsqueeze(1)
            else:
                raise ValueError(f'Unaudited checkpoint tensor: {name}')
        equal = torch.equal(value.float(), expected.float())
        row = {'name': name, 'elements': value.numel(), 'exact': equal,
               'max_abs_difference': float((value.float()-expected.float()).abs().max())}
        compared.append(row)
        if not equal:
            differences.append(row)
    record = {'tensors_checked': len(compared), 'differences': differences, 'tensors': compared}
    record['passed'] = all(row['name'].endswith('filter.log_poles') and
                           row['max_abs_difference'] <= 3e-7 for row in differences)
    q1.write_json(OUTPUT / 'parameter_audit.json', record)
    print('PARAMETER_AUDIT', len(compared), 'tensors,', len(differences), 'differences', flush=True)
    return record


def sequences():
    dna = q1.read_csv(q1.OUTPUT / 'sequences.csv.gz')
    result = []
    for row in dna.iloc[:2].itertuples():
        result += [row.ref_sequence, q1.reverse_complement(row.ref_sequence),
                   row.alt_sequence, q1.reverse_complement(row.alt_sequence)]
    return result


def audit_conversion(model):
    """Check every loaded BioNeMo parameter against its intended source tensor."""
    source_path = ROOT / 'data/evo2-savanna-1b/savanna_evo2_1b_base.pt'
    q1.verify_file(source_path, '7bb731473c99db72aba34e7b0df443f2f422e12b0669bea6d78415dc87057a08')
    source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=False)['module']
    names = {'mixer.dense_projection.layer_norm_weight': 'input_layernorm.weight',
             'self_attention.linear_qkv.layer_norm_weight': 'input_layernorm.weight',
             'self_attention.linear_qkv.weight': 'mixer.dense_projection.weight',
             'self_attention.linear_proj.weight': 'mixer.dense.weight',
             'self_attention.linear_proj.bias': 'mixer.dense.bias',
             'mlp.linear_fc1.layer_norm_weight': 'pre_mlp_layernorm.weight',
             'mlp.linear_fc2.weight': 'mlp.w3.weight'}
    rows = []
    tensors = list(model.named_parameters())
    tensors += [(n, p) for n, p in model.named_buffers() if n.endswith('filter.decay')]
    for name, value in tensors:
        if name == 'embedding.word_embeddings.weight':
            expected = source['sequential.0.word_embeddings.weight']
        elif name == 'decoder.final_norm.weight':
            expected = source['sequential.28.norm.weight']
        else:
            _, _, layer, suffix = name.split('.', 3)
            prefix = f'sequential.{int(layer)+2}.'
            if suffix == 'mlp.linear_fc1.weight':
                expected = torch.cat([source[prefix+'mlp.w1.weight'], source[prefix+'mlp.w2.weight']], dim=0)
            else:
                expected = source[prefix+names.get(suffix, suffix)]
            if suffix.endswith(('filter.h', 'filter.decay')):
                expected = expected[:, :128]
        equal = torch.equal(value.detach().cpu(), expected.to(value.dtype))
        rows.append({'name': name, 'exact': equal, 'elements': value.numel()})
    record = {'passed': all(r['exact'] for r in rows), 'tensors_checked': len(rows), 'tensors': rows}
    q1.write_json(OUTPUT / 'conversion_audit.json', record)
    return record


def measure(backend):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if backend == 'vortex':
        # Match Evo2's import order: TE/FlashAttention before Vortex's wrappers.
        import transformer_engine
        from huggingface_hub import hf_hub_download
        from vortex.model.model import StripedHyena
        from vortex.model.tokenizer import CharLevelTokenizer
        from vortex.model.utils import dotdict, load_checkpoint
        checkpoint = hf_hub_download(q2.MODEL_REPO, 'evo2_1b_base.pt', revision=q2.MODEL_REVISION)
        q1.verify_file(checkpoint, q2.MODEL_SHA256)
        config = yaml.safe_load((ROOT / 'evo2/evo2/configs/evo2-1b-8k.yml').read_text())
        config['use_fp8_input_projections'] = False
        model = StripedHyena(dotdict(config))
        load_checkpoint(model, checkpoint)
        model = model.cuda().eval().requires_grad_(False)
        tokenizer = CharLevelTokenizer(512)
        from vortex.model.layers import RMSNorm
        norm = RMSNorm(model.config).cuda()
    else:
        from . import q9_backend
        model, tokenizer = q9_backend.load_backbone()
        audit_conversion(model)
        from transformer_engine.pytorch import RMSNorm
        norm = RMSNorm(1920, eps=1e-6, params_dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        norm_probe = float(norm(torch.full((1, 1, 1920), .001, dtype=torch.bfloat16,
                                          device='cuda')).float().mean())
    features, likelihoods, encoded, seconds = [], [], [], []
    trace = []
    hooks = []
    layers = model.blocks if backend == 'vortex' else model.decoder.layers
    for layer in layers:
        hooks.append(layer.register_forward_hook(
            lambda module, inputs, out: trace.append(
                (out[0] if isinstance(out, tuple) else out).float().mean(
                    dim=1 if backend == 'vortex' else 0)[0].detach().cpu().numpy())))
    try:
        for sequence in sequences():
            start = time.perf_counter()
            if backend == 'nemo':
                pooled, likelihood = q9_backend.encode(model, tokenizer, sequence)
                token_ids = tokenizer.text_to_ids(sequence)
            else:
                token_ids = tokenizer.tokenize(sequence)
                tokens = torch.tensor([token_ids], device='cuda', dtype=torch.long)
                captured = []
                hook = model.blocks[24].register_forward_hook(
                    lambda module, inputs, out: captured.append((out[0] if isinstance(out, tuple) else out)
                                                                .float().mean(dim=1)[0]))
                with torch.inference_mode():
                    logits = model(tokens)[0]
                    pooled = captured[-1]
                    likelihood = torch.log_softmax(logits.float(), dim=-1)[:, :-1].gather(
                        2, tokens[:, 1:, None]).mean()
                hook.remove()
            features.append(pooled.cpu().numpy())
            likelihoods.append(float(likelihood))
            encoded.append(token_ids)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter()-start)
            print(backend, len(features), float(likelihood), seconds[-1], flush=True)
            if hooks:
                for hook in hooks:
                    hook.remove()
                hooks = []
        np.savez(OUTPUT / f'parity_{backend}.npz', features=features, likelihoods=likelihoods,
                 tokens=encoded, seconds=seconds, layers=trace, normalization_probe=norm_probe)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('backend', choices=['vortex', 'nemo', 'parameters'])
    backend = parser.parse_args().backend
    audit_parameters() if backend == 'parameters' else measure(backend)
