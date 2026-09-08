"""Draft Q15 frozen-prefix cache and direct replay of Evo2 blocks 29–31."""

import torch

from . import q11, q11_backend as base, q12_backend, q15_pooling

FIRST_TAIL_BLOCK = 29
CONTEXT_BP = 512


def _check_model(model):
    config = model.transformer_config
    if (model.training or any(layer.training for layer in model.decoder.layers)
            or len(model.decoder.layers) != q11.CONFIG['layers']
            or config.hidden_size != q11.CONFIG['hidden_size']
            or config.params_dtype != torch.bfloat16 or config.fp32_residual_connection
            or config.sequence_parallel or config.tensor_model_parallel_size != 1
            or config.context_parallel_size != 1 or config.fp8 is not None
            or not model.pre_process or not model.decoder.pre_process
            or model.decoder.input_tensor is not None or model.position_embedding_type != 'rope'):
        raise ValueError('Replay requires the pinned BF16, single-rank Evo2 model in evaluation mode')
    if any(p.requires_grad for p in model.embedding.parameters()) or any(
            p.requires_grad for layer in model.decoder.layers[:FIRST_TAIL_BLOCK] for p in layer.parameters()):
        raise ValueError('Embedding and blocks 0–28 must be completely frozen')


def _rotary(model, hidden):
    length = model.rotary_pos_emb.get_rotary_seq_len(
        None, model.decoder, hidden, model.transformer_config, None)
    if length != CONTEXT_BP:
        raise ValueError('Replay rotary sequence length differs from the frozen 512-base context')
    return model.rotary_pos_emb(length)


def extract_prefix(model, tokenizer, rows):
    """Return CPU BF16 [variants, 2, 512, hidden] inputs to block 29.

    Alleles remain in [reference, alternate] order after the existing canonical
    crop. Use a small input batch (32 variants is 256 MiB at width 4096); callers
    may copy batches into a preallocated volatile CPU chunk. This writes no disk
    cache and does not run the trainable tail. Ordinary no_grad tensors can later
    be used by autograd; inference-mode tensors are deliberately rejected.
    """
    _check_model(model)
    if torch.is_inference_mode_enabled():
        raise ValueError('Prefix extraction must use no_grad, not inference_mode')
    sequences, _ = q15_pooling.canonical_sequences(base.pairs_from(rows))
    ids = [tokenizer.text_to_ids(sequence) for sequence in sequences]
    if any(tokens != list(sequence.encode('ascii')) for tokens, sequence in zip(ids, sequences)):
        raise ValueError('Byte tokenizer changed')
    device = model.embedding.word_embeddings.weight.device
    with torch.no_grad():
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        positions = torch.arange(CONTEXT_BP, device=device)[None].expand(len(sequences), -1)
        hidden = model.embedding(input_ids=tokens, position_ids=positions)
        # Native embedding returns a contiguous, viewless tensor, so the stack's
        # make_viewless_tensor call returns this same tensor without modification.
        if hidden._base is not None:
            raise ValueError('Unexpected view-backed native embedding output')
        rotary = _rotary(model, hidden)
        for layer in model.decoder.layers[:FIRST_TAIL_BLOCK]:
            hidden = layer(hidden, None, inference_params=None, rotary_pos_emb=rotary)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        expected = (CONTEXT_BP, len(sequences), q11.CONFIG['hidden_size'])
        if hidden.shape != expected or hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all():
            raise ValueError('Invalid frozen block-29 input')
        cached = hidden.detach().cpu().permute(1, 0, 2).contiguous()
        return cached.reshape(len(rows), 2, CONTEXT_BP, q11.CONFIG['hidden_size'])


def replay_tail_hidden(model, cached):
    """Replay original layers 29–31 and return pre-final-norm [512, 2N, hidden].

    Adapter gradients stay connected. A private contiguous copy isolates the
    immutable CPU cache from any tail operation. RoPE and layer arguments match
    HyenaModel/HyenaStack; no recurrent state, final normalization or output
    projection is used. Call under no_grad only when evaluating.
    """
    _check_model(model)
    if (not isinstance(cached, torch.Tensor) or cached.ndim != 4 or not len(cached)
            or cached.shape[1:] != (2, CONTEXT_BP, q11.CONFIG['hidden_size'])
            or cached.dtype != torch.bfloat16 or cached.device.type != 'cpu'
            or cached.requires_grad or cached.is_inference()):
        raise ValueError('Expected ordinary, detached CPU BF16 paired prefix states')
    native = cached.reshape(2 * len(cached), CONTEXT_BP, q11.CONFIG['hidden_size']).permute(1, 0, 2).contiguous()
    hidden = native.to(model.embedding.word_embeddings.weight.device,
                       non_blocking=native.is_pinned(), copy=True)
    if not torch.isfinite(hidden).all():
        raise ValueError('Cached prefix states contain nonfinite values')
    rotary = _rotary(model, hidden)
    for layer in model.decoder.layers[FIRST_TAIL_BLOCK:]:
        hidden = layer(hidden, None, inference_params=None, rotary_pos_emb=rotary)
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    if hidden.shape != (CONTEXT_BP, 2 * len(cached), q11.CONFIG['hidden_size']) or not torch.isfinite(hidden).all():
        raise ValueError('Invalid tail activations')
    return hidden


def replay_pooled(model, cached):
    """Q11/Q12 global FP32 mean, returned as [variants, ref/alt, hidden]."""
    hidden = replay_tail_hidden(model, cached)
    return hidden.float().mean(dim=0).reshape(len(cached), 2, q11.CONFIG['hidden_size'])


def replay_features(model, cached):
    """The existing global Q12 magnitude features; no new pooling or fitting."""
    pooled = replay_pooled(model, cached)
    return q12_backend.features_from_pooled(pooled[:, 0], pooled[:, 1] - pooled[:, 0])
