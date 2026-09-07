#!/usr/bin/env python3
"""Check imports, the notebook kernel and GPU kernels without downloading models."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

try:
    from .setup import ROOT, ensure_submodule, gpu_base, requirements
except ImportError:
    from setup import ROOT, ensure_submodule, gpu_base, requirements


def check(profile):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required.')
    versions = {}
    for requirement in requirements():
        name, expected = requirement.split('==', 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}. Rerun setup.')
        versions[name] = actual
    for executable in ['git', 'curl']:
        if shutil.which(executable) is None:
            raise RuntimeError(f'Missing executable: {executable}')
    sys.path.insert(0, str(ROOT))
    for module in ['numpy', 'pandas', 'matplotlib', 'sklearn', 'Bio', 'nbclient', 'nbconvert',
                   'notebooks.src.q0', 'notebooks.src.q1_full', 'notebooks.src.comparison']:
        importlib.import_module(module)
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
    try:
        kernel = KernelSpecManager().get_kernel_spec('gamow')
    except NoSuchKernel:
        raise RuntimeError('The gamow notebook kernel is missing. Rerun setup.') from None
    if Path(kernel.argv[0]).absolute() != Path(sys.executable).absolute():
        raise RuntimeError('The gamow kernel points to another Python. Rerun setup.')
    result = {'profile': profile, 'python': sys.executable, 'packages': versions,
              'free_disk_gib': round(shutil.disk_usage(ROOT).free / 1024**3, 1)}
    if profile == 'gpu':
        versions.update(gpu_base(sys.executable))
        ensure_submodule()
        import torch
        import transformer_engine.pytorch as te
        sys.path.insert(0, str(ROOT / 'evo2'))
        from evo2 import Evo2  # noqa: F401: checks the actual model import chain
        # Vortex registers its own Flash Attention wrappers. Probe that API,
        # as the model does; importing the other wrapper first leaves stale
        # operator handles after Vortex registers the same operator names.
        from vortex.ops import local_flash_attn_func
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Expose the GPUs to the container (see SETUP.md).')
        if shutil.which('nvcc') is None:
            raise RuntimeError('nvcc is required to compile BioNeMo CUDA extensions.')
        result['gpus'] = []
        with torch.no_grad():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    q = torch.randn(1, 128, 4, 64, device=f'cuda:{index}', dtype=torch.bfloat16)
                    attention = local_flash_attn_func(q, q, q, causal=True)
                    layer = te.Linear(256, 256, params_dtype=torch.bfloat16, device=f'cuda:{index}')
                    output = layer(q.reshape(128, 256))
                    if not torch.isfinite(attention).all() or not torch.isfinite(output).all():
                        raise RuntimeError(f'Non-finite GPU kernel output on device {index}.')
                    torch.cuda.synchronize(index)
                    result['gpus'].append({'name': torch.cuda.get_device_name(index),
                        'capability': list(torch.cuda.get_device_capability(index)),
                        'memory_gib': round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 1)})
        versions['vtx'] = importlib.metadata.version('vtx')
        if versions['vtx'] != '1.1.0':
            raise RuntimeError('Expected vtx==1.1.0.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    try:
        result = check(args.profile)
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f'Environment check failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
