"""DNA-only canonical crops and differentiable variant-local pooling for Q15.

Q11 canonicalizes the full reference/alternate pair before centering its crop.
Its 512-base crop therefore places the sole substitution at zero-based position
256 on either input strand. The causal 64-position pooling window is [256, 320),
including the mutation token and 63 downstream tokens. Windows never shift or clip.
"""

import torch

from . import q11

CONTEXT_BP = 512
WINDOW_BP = 64


def canonical_pair(reference, alternate):
    """Return (reference_crop, alternate_crop, mutation_index), without annotations."""
    if not isinstance(reference, str) or not isinstance(alternate, str):
        raise ValueError('Alleles must be DNA strings')
    reference, alternate = q11.crop_pair(reference, alternate, length=CONTEXT_BP)
    changes = [index for index, (ref, alt) in enumerate(zip(reference, alternate)) if ref != alt]
    if len(reference) != CONTEXT_BP or len(alternate) != CONTEXT_BP or changes != [CONTEXT_BP // 2]:
        raise ValueError('Canonical crop must retain one substitution at position 256')
    return reference, alternate, changes[0]


def canonical_sequences(pairs):
    """Flatten allele pairs as [ref0, alt0, ref1, alt1, ...], with one index per sequence."""
    sequences, indices = [], []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('Each example must contain only a reference and alternate DNA string')
        reference, alternate, position = canonical_pair(*pair)
        sequences.extend([reference, alternate])
        indices.extend([position, position])
    if not sequences:
        raise ValueError('At least one allele pair is required')
    return sequences, indices


def pool_hidden(hidden, mutation_indices, *, window=WINDOW_BP, include_global=False):
    """Pool Megatron [sequence, batch, hidden] activations into FP32 [batch, hidden].

    Returns {'local': tensor}, optionally with 'global' from the same activations.
    Each local window is [mutation_indices[b], mutation_indices[b] + window).
    Gradients remain connected to the input. Local-only pooling casts just the
    selected window to FP32; requesting global pooling also computes Q11's mean.
    """
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
            or any(size <= 0 for size in hidden.shape) or not hidden.is_floating_point()):
        raise ValueError('Expected nonempty floating [sequence, batch, hidden] activations')
    if hidden.shape[0] != CONTEXT_BP:
        raise ValueError('Q15 pooling requires the preserved 512-base sequence context')
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= CONTEXT_BP:
        raise ValueError('Window must be a positive integer within the sequence context')
    if not isinstance(include_global, bool):
        raise ValueError('include_global must be boolean')
    indices = torch.as_tensor(mutation_indices, device=hidden.device)
    if indices.shape != (hidden.shape[1],) or indices.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Mutation indices must be an integer vector with one entry per sequence')
    indices = indices.long()
    if (indices < 0).any() or (indices + window > hidden.shape[0]).any():
        raise ValueError('Local window extends beyond the sequence; clipping and shifting are forbidden')
    if not torch.isfinite(hidden).all():
        raise ValueError('Activations must be finite')
    positions = indices[None, :] + torch.arange(window, device=hidden.device)[:, None]
    batches = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    result = {'local': hidden[positions, batches, :].float().mean(dim=0)}
    if include_global:
        result['global'] = hidden.float().mean(dim=0)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError('FP32 pooled activations overflowed')
    return result
