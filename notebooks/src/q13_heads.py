"""Converged, magnitude-aware linear controls for Q13's adapter experiments."""

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def objective(weight, bias, features, labels, sample_weights, strength):
    """Balanced mean BCE plus strength/2 times squared weights; bias is unpenalized.

    The caller chooses fitting or deployment tensor precision. Scalar reductions
    use FP64 in both cases so the two objectives can be compared accurately.
    """
    losses = binary_cross_entropy_with_logits(features @ weight + bias, labels, reduction='none')
    return (losses.double() * sample_weights.double()).mean() + strength * weight.double().square().sum() / 2


def select_candidate(candidates):
    """Choose a converged fit by AUROC, AP, then stronger regularization."""
    eligible = [candidate for candidate in candidates if candidate['converged']]
    if not eligible:
        details = [(candidate['strength'], candidate['gradient_norm']) for candidate in candidates]
        raise RuntimeError(f'No linear head converged; (strength, full gradient norm): {details}')
    return max(eligible, key=lambda candidate: (candidate['auroc'],
        candidate['average_precision'], candidate['strength']))


def _validate(training, labels, validation, validation_labels, strengths, max_iter, tolerance):
    training, validation = np.asarray(training), np.asarray(validation)
    if (training.ndim != 2 or validation.ndim != 2 or not training.size
            or not validation.size or training.shape[1] != validation.shape[1]
            or not np.isfinite(training).all() or not np.isfinite(validation).all()):
        raise ValueError('Features must be finite, nonempty matrices with matching columns')
    arrays = []
    for features, target in [(training, labels), (validation, validation_labels)]:
        target = np.asarray(target)
        if target.shape != (len(features),) or set(np.unique(target)) != {0, 1}:
            raise ValueError('Labels must be a vector matching its feature rows and contain both binary classes')
        arrays.extend([features.astype(np.float32, copy=False), target.astype(np.float32, copy=False)])
    if not strengths or len(set(strengths)) != len(strengths) or any(
            not np.isfinite(value) or value <= 0 for value in strengths):
        raise ValueError('Regularization strengths must be distinct, finite and positive')
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    return arrays


def fit_heads(training, labels, validation, validation_labels, *, device='cpu',
              strengths=(1e-4, 1e-3, 1e-2), max_iter=1000, tolerance=1e-6):
    """Fit FP64 heads to convergence, then select using deployed FP32 validation logits.

    Each strength starts from zeros. A nonconverged fit gets one continuation of
    at most 2*max_iter iterations, making the total bound 3*max_iter. A line-search
    stop or an iteration limit alone never counts as convergence. The result's
    weight [1, d], bias [1], mean [d], scale [d], and class_weights [2] are CPU
    FP32 tensors; candidates contain JSON-compatible diagnostics. Convergence
    means the fitted FP64 full gradient L2 norm meets tolerance. The deployed
    FP32 gradient norm is recorded separately and does not redefine convergence.
    Training logits must agree across the cast within atol=rtol=2e-4.
    Validation is used only for selection, never for scaling or the objective.
    """
    strengths = tuple(float(value) for value in strengths)
    training, labels, validation, validation_labels = _validate(
        training, labels, validation, validation_labels, strengths, max_iter, tolerance)
    mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(training.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    deployed_x = torch.tensor((training - mean) / scale, dtype=torch.float32, device=device)
    xv = torch.tensor((validation - mean) / scale, dtype=torch.float32, device=device)
    x = deployed_x.double()
    y = torch.tensor(labels, dtype=torch.float64, device=device)
    class_weights = torch.tensor(len(labels) / (2 * np.bincount(labels.astype(int))),
                                 dtype=torch.float64, device=device)
    sample_weights = class_weights[y.long()]
    if not torch.isfinite(x).all() or not torch.isfinite(xv).all():
        raise ValueError('Standardized features overflow FP32')

    candidates, states = [], {}
    for strength in strengths:
        weight = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64, device=device))
        bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device=device))
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=max_iter,
            tolerance_grad=tolerance / math.sqrt(x.shape[1] + 1), tolerance_change=0.,
            history_size=50, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective(weight, bias, x, y, sample_weights, strength)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite linear-head objective at strength {strength}')
            loss.backward()
            if not torch.isfinite(weight.grad).all() or not torch.isfinite(bias.grad).all():
                raise RuntimeError(f'Nonfinite linear-head gradient at strength {strength}')
            return loss

        for attempt, iterations in enumerate((max_iter, 2 * max_iter), start=1):
            optimizer.param_groups[0].update(max_iter=iterations, max_eval=iterations * 5 // 4)
            optimizer.step(closure)
            loss = float(closure().detach())
            gradient_norm = float(torch.linalg.vector_norm(torch.cat([weight.grad, bias.grad])))
            if gradient_norm <= tolerance:
                break
        with torch.no_grad():
            fitted_logits = x @ weight + bias
            penalty = float(strength * weight.double().square().sum() / 2)
            weighted_bce = float((binary_cross_entropy_with_logits(
                fitted_logits, y, reduction='none') * sample_weights).mean())
        deployed_weight = weight.detach().float().requires_grad_()
        deployed_bias = bias.detach().float().requires_grad_()
        deployed_loss = objective(deployed_weight, deployed_bias, deployed_x, y.float(),
                                  sample_weights.float(), strength)
        deployed_grads = torch.autograd.grad(deployed_loss, (deployed_weight, deployed_bias))
        deployed_gradient_norm = float(torch.linalg.vector_norm(torch.cat(deployed_grads)))
        with torch.no_grad():
            deployed_logits = torch.nn.functional.linear(deployed_x, deployed_weight[None], deployed_bias).flatten()
            difference = float((deployed_logits.double() - fitted_logits).abs().max())
            if not torch.allclose(deployed_logits.double(), fitted_logits, atol=2e-4, rtol=2e-4):
                raise RuntimeError(f'FP32 deployment logits differ from FP64 fit at strength {strength}: max difference {difference}')
            scores = torch.nn.functional.linear(xv, deployed_weight[None], deployed_bias).flatten().cpu().numpy()
        if not np.isfinite(deployed_gradient_norm):
            raise RuntimeError(f'Nonfinite deployed linear-head gradient at strength {strength}')
        if not np.isfinite(scores).all():
            raise RuntimeError(f'Nonfinite linear-head validation scores at strength {strength}')
        candidate = {'strength': strength, 'auroc': float(roc_auc_score(validation_labels, scores)),
            'average_precision': float(average_precision_score(validation_labels, scores)),
            'objective': loss, 'weighted_bce': weighted_bce, 'penalty': penalty,
            'gradient_norm': gradient_norm, 'converged': gradient_norm <= tolerance,
            'fitted_gradient_norm': gradient_norm, 'deployed_gradient_norm': deployed_gradient_norm,
            'deployed_objective': float(deployed_loss.detach()),
            'max_training_logit_cast_difference': difference, 'deployment_logits_verified': True,
            'iterations': optimizer.state[weight]['n_iter'], 'attempts': attempt,
            'function_evaluations': optimizer.state[weight]['func_evals']}
        candidates.append(candidate)
        states[strength] = (deployed_weight.detach().cpu().unsqueeze(0).clone(), deployed_bias.detach().cpu().clone())
        print(f'Head L2={strength:g}: converged={candidate["converged"]}, '
              f'FP64 gradient={gradient_norm:.3g}, FP32 gradient={deployed_gradient_norm:.3g}, '
              f'iterations={candidate["iterations"]}, '
              f'AUROC={candidate["auroc"]:.6f}, AP={candidate["average_precision"]:.6f}', flush=True)
    chosen = select_candidate(candidates)
    weight, bias = states[chosen['strength']]
    return {'weight': weight, 'bias': bias, 'mean': torch.from_numpy(mean),
            'scale': torch.from_numpy(scale), 'class_weights': class_weights.float().cpu(),
            'strength': chosen['strength'], 'candidates': candidates,
            'convergence_tolerance': tolerance, 'max_iterations_per_fit': 3 * max_iter,
            'fitting_precision': 'float64', 'deployment_precision': 'float32',
            'deployment_logit_tolerance': {'atol': 2e-4, 'rtol': 2e-4}}
