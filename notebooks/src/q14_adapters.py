"""Prospective Q14 adapters across active Evo2 Hyena blocks; no training here."""

import math

import torch

from . import q11, q11_backend as base


def attach(model, blocks=(29, 30), rank=8, alpha=16, seed=42):
    """Attach independent zero-output LoRA adapters without changing base tensors.

    The pinned Evo2 7B tail has medium Hyena at block 29 and long Hyena at
    block 30. Every selected block must expose both mixer projections. This
    helper does not change Q11's configuration or its single-block adapter code.
    """
    blocks = tuple(blocks)
    if (not blocks or any(isinstance(i, bool) or not isinstance(i, int) for i in blocks)
            or len(set(blocks)) != len(blocks)):
        raise ValueError('Blocks must be distinct integer indexes')
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError('LoRA rank must be a positive integer')
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('LoRA alpha must be finite and positive')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('Seed must be an integer in [0, 2**63)')
    if len(model.decoder.layers) != q11.CONFIG['layers']:
        raise ValueError('Unexpected Evo2 7B backbone depth')
    if any(i < 0 or i >= len(model.decoder.layers) for i in blocks):
        raise ValueError('Selected block is outside the backbone')

    # Resolve every target before changing the model so invalid selections fail
    # without leaving earlier blocks wrapped or changing their trainable flags.
    targets = []
    for block in blocks:
        for name in ('dense_projection', 'dense'):
            prefix = f'decoder.layers.{block}.mixer'
            path = f'{prefix}.{name}'
            try:
                parent = model.get_submodule(prefix)
                projection = getattr(parent, name)
            except AttributeError as error:
                raise ValueError(f'Selected block lacks Hyena projection: {path}') from error
            if hasattr(projection, 'adapter'):
                raise ValueError(f'LoRA is already attached: {path}')
            if not all(hasattr(projection, field) for field in ('weight', 'in_features', 'out_features')):
                raise ValueError(f'Selected projection is not a supported linear layer: {path}')
            targets.append((block, name, prefix, path, parent, projection))

    from nemo.collections.llm.peft.lora import LoRA, LoRALinear

    before = base.parameter_hash(model)
    model.requires_grad_(False)
    transform = LoRA(target_modules=[item[3] for item in targets], dim=rank, alpha=alpha,
                     dropout=0., lora_A_init_method='xavier', lora_B_init_method='zero',
                     lora_dtype=torch.bfloat16)
    adapters = torch.nn.ModuleDict()
    generators = {}
    expected_count = 0
    for block, name, prefix, path, parent, projection in targets:
        wrapped = transform.transform(projection, name=name, prefix=prefix)
        if not isinstance(wrapped, LoRALinear) or wrapped.to_wrap is not projection:
            raise RuntimeError(f'Pinned NeMo did not preserve and wrap {path}')
        wrapped.to_wrap.requires_grad_(False)
        adapter = wrapped.adapter.to(device=projection.weight.device, dtype=torch.bfloat16)
        adapter.requires_grad_(True)
        device = projection.weight.device
        if device not in generators:
            generators[device] = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            torch.nn.init.xavier_normal_(adapter.linear_in.weight, generator=generators[device])
            torch.nn.init.zeros_(adapter.linear_out.weight)
        setattr(parent, name, wrapped)
        adapters[f'block{block}_{name}'] = adapter
        expected_count += rank * (projection.in_features + projection.out_features)

    parameters = list(adapters.parameters())
    if len(adapters) != 2 * len(blocks) or sum(p.numel() for p in parameters) != expected_count:
        raise RuntimeError('Adapter count differs from selected blocks and projection dimensions')
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}:
        raise RuntimeError('Trainable parameters are not exactly the selected LoRA adapters')
    if any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError('Adapter initialization contains nonfinite values')
    if any(torch.count_nonzero(adapter.linear_out.weight) for adapter in adapters.values()):
        raise RuntimeError('LoRA output matrices must initialize to zero')
    if base.parameter_hash(model, frozen_only=True) != before:
        raise RuntimeError('Adding adapters changed original backbone tensors')
    return adapters
