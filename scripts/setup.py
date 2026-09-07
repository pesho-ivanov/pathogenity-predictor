#!/usr/bin/env python3
"""Install the shared requirements into a local CPU or NVIDIA GPU environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GPU_REQUIREMENTS = {'vtx==1.1.0', '-e ./evo2'}
GPU_BASE = {
    'torch': '2.7.0a0+79aa17489c.nv25.4',
    'transformer-engine': '2.2.0+c55e425',
    'flash-attn': '2.7.3',
}
IMAGE = 'nvcr.io/nvidia/pytorch:25.04-py3'


def requirements(root=ROOT):
    """Keep requirements.txt unchanged: it is part of frozen result identities."""
    lines = [line.strip() for line in (root / 'requirements.txt').read_text().splitlines()]
    lines = [line for line in lines if line and not line.startswith('#')]
    if not GPU_REQUIREMENTS <= set(lines):
        raise RuntimeError('GPU requirements changed; update the setup profile explicitly.')
    return [line for line in lines if line not in GPU_REQUIREMENTS]


def run(args, *, env=None):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env, check=True)


def gpu_base(python):
    code = ('import importlib.metadata as m, json; '
            f'print(json.dumps({{p: m.version(p) for p in {list(GPU_BASE)!r}}}))')
    result = subprocess.run([str(python), '-c', code], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'GPU setup needs the compiled stack in {IMAGE}. See README.md#setup.')
    actual = json.loads(result.stdout)
    if actual != GPU_BASE:
        raise RuntimeError(f'GPU stack differs from {IMAGE}: {actual}. See README.md#setup.')
    return actual


def ensure_submodule(initialize=False):
    path = ROOT / 'evo2'
    if not (path / '.git').exists():
        if not initialize:
            raise RuntimeError('Evo2 submodule is missing. Run GPU setup or git submodule update --init --recursive.')
        run(['git', 'submodule', 'update', '--init', '--recursive', '--', 'evo2'])
    expected = subprocess.check_output(['git', 'ls-tree', 'HEAD', 'evo2'], cwd=ROOT, text=True).split()[2]
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if expected != actual:
        raise RuntimeError('Evo2 submodule is at another revision; preserve local work before updating it.')
    run(['git', '-C', path, 'diff', '--exit-code', 'HEAD', '--', 'evo2', 'pyproject.toml'])


def install(profile, destination):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12, the version used for the recorded experiments.')
    for executable in ['git', 'curl']:
        if shutil.which(executable) is None:
            raise RuntimeError(f'Missing {executable}. Install the prerequisites in README.md#setup.')
    gpu = profile == 'gpu'
    if gpu:
        gpu_base(sys.executable)
        ensure_submodule(initialize=True)
    config = destination / 'pyvenv.cfg'
    if destination.exists():
        if not config.exists():
            raise RuntimeError(f'{destination} exists but is not a virtual environment.')
        inherited = 'include-system-site-packages = true' in config.read_text()
        if inherited != gpu:
            raise RuntimeError(f'{destination} has another profile. Use a fresh --venv path; preserve the existing environment.')
    else:
        args = [sys.executable, '-m', 'venv']
        if gpu:
            args.append('--system-site-packages')
        run(args + [destination])
    python = destination / 'bin/python'
    # NGC's global pip constraints pin unrelated notebook packages. Use our
    # requirements in this venv while explicitly protecting the compiled stack.
    env = dict(os.environ)
    env.pop('PIP_CONSTRAINT', None)
    env.pop('PIP_BUILD_CONSTRAINT', None)
    with tempfile.TemporaryDirectory(prefix='gamow-install-') as temporary:
        constraints = Path(temporary) / 'constraints.txt'
        constraints.write_text(''.join(f'{name}=={version}\n' for name, version in GPU_BASE.items()))
        pip = [python, '-m', 'pip', 'install', '--disable-pip-version-check']
        if gpu:
            pip += ['-c', constraints]
        run(pip + requirements(), env=env)
        if gpu:
            run(pip + ['--no-build-isolation', 'vtx==1.1.0', '-e', './evo2'], env=env)
            gpu_base(python)
    run([python, '-m', 'ipykernel', 'install', '--sys-prefix', '--name', 'gamow',
         '--display-name', 'Gamow'])
    return python


def run_tests(python, profile):
    modules = [f'tests.{path.stem}' for path in sorted((ROOT / 'tests').glob('test_*.py'))]
    if profile == 'cpu':
        modules.remove('tests.test_q9')
        print('CPU profile: omitting test_q9, which requires PyTorch; the GPU profile runs it.', flush=True)
    run([python, '-m', 'unittest', *modules])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--venv', type=Path, default=ROOT / '.venv', help='Environment path (default: repo/.venv).')
    parser.add_argument('--check', action='store_true', help='Check an existing environment without installing.')
    parser.add_argument('--test', action='store_true', help='Also run unit tests; no data or models are downloaded.')
    args = parser.parse_args()
    destination = args.venv.expanduser().absolute()
    python = destination / 'bin/python'
    try:
        if not args.check:
            python = install(args.profile, destination)
        if not python.exists():
            raise RuntimeError(f'Environment missing: {destination}. Run setup without --check first.')
        run([python, ROOT / 'scripts/check_environment.py', '--profile', args.profile])
        if args.test:
            run_tests(python, args.profile)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Setup failed: {error}\n')
    print(f'Ready. Start Jupyter with: {destination}/bin/jupyter lab', flush=True)


if __name__ == '__main__':
    main()
