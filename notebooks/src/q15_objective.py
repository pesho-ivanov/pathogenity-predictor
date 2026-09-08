"""Training-only objectives for the same fixed Q15 classifier.

The caller must skip backward *and* optimizer.step when ``has_signal`` is false;
an AdamW step on a zero gradient would still apply weight decay.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
    has_signal: bool
    positives: int
    negatives: int
    pairs: int


def compute_loss(logits, labels, *, objective, class_weights=None):
    """Return balanced BCE or the mean logistic loss over all class pairs.

    ``logits`` and binary training ``labels`` are matching nonempty 1-D tensors.
    BCE averages ``class_weights[label] * BCE(logit, label)`` over examples;
    these two positive weights must be fixed from the training cohort, not
    recomputed within each batch. Ranking is unweighted and averages
    ``softplus(negative_logit - positive_logit)`` over every positive/negative
    pair in this batch. Its single-class batches have no ranking signal.
    """
    if objective not in ('weighted_bce', 'pairwise_logistic'):
        raise ValueError('Unknown training objective')
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError('Logits and labels must be tensors')
    if logits.ndim != 1 or logits.numel() == 0 or labels.shape != logits.shape:
        raise ValueError('Logits and labels must be matching nonempty 1-D tensors')
    if not logits.is_floating_point() or labels.is_complex():
        raise ValueError('Logits must be floating point and labels must be real')
    if logits.device != labels.device:
        raise ValueError('Logits and labels must share a device')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError('Logits and labels must be finite')
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Training labels must be binary 0 or 1')
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            raise TypeError('Class weights must be a tensor')
        if class_weights.shape != (2,) or class_weights.is_complex():
            raise ValueError('Class weights must be a real tensor with shape (2,)')
        if class_weights.device != logits.device:
            raise ValueError('Class weights and logits must share a device')
        if not torch.isfinite(class_weights).all() or not (class_weights > 0).all():
            raise ValueError('Class weights must be finite and positive')

    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(logits) - positives
    pairs = positives * negatives
    # Avoid half-precision overflow in pair differences and loss reduction.
    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if objective == 'weighted_bce':
        if class_weights is None:
            raise ValueError('Weighted BCE requires training-cohort class weights')
        weights = class_weights.to(scores.dtype)[labels.long()]
        loss = (F.binary_cross_entropy_with_logits(scores, labels.to(scores.dtype),
                                                  reduction='none') * weights).mean()
        has_signal = True
    elif pairs:
        differences = scores[~positive, None] - scores[positive][None, :]
        loss = F.softplus(differences).mean()
        has_signal = True
    else:
        # One finite element gives a graph-connected zero without sum overflow.
        loss = scores[0] * 0
        has_signal = False
    if not torch.isfinite(loss):
        raise ValueError('Training loss is nonfinite')
    return LossResult(loss, has_signal, positives, negatives, pairs)
