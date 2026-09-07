"""Reproduce Q9's isolated BioNeMo setup without changing Q2's packages."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from contextlib import nullcontext
import fcntl

from . import q1

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/bionemo-recipes'
PYTHON = ROOT / '.venv/q9/bin/python'
OUTPUT = ROOT / 'notebooks/results/q9/environment'
REPOSITORIES = [
    (SOURCE, 'https://github.com/NVIDIA-BioNeMo/bionemo-recipes.git',
     'ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69'),
    (SOURCE / '3rdparty/Megatron-LM', 'https://github.com/NVIDIA/Megatron-LM.git',
     '62529f1d8e3d76f45ba5c0b4d7791566055d3eee'),
    (SOURCE / '3rdparty/NeMo', 'https://github.com/NVIDIA/NeMo.git',
     '2be3af56a4eb57a4892c55b66e29a4a918a0867c'),
]
PROTECTED = {
    'torch': '2.7.0a0+79aa17489c.nv25.4',
    'numpy': '1.26.4', 'transformer-engine': '2.2.0+c55e425',
}
PROBE = '''
import importlib.metadata as m
import json
import torch
import bionemo.evo2.run.train
from nemo.collections.llm.gpt.model.hyena import Hyena1bConfig
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Q9_ENVIRONMENT=' + json.dumps({
    'gpu': torch.cuda.get_device_name(0),
    'packages': {p: m.version(p) for p in ['torch', 'numpy', 'transformer-engine',
        'bionemo-evo2', 'nemo-toolkit', 'megatron-core', 'lightning',
        'transformers', 'causal-conv1d', 'huggingface-hub']}}))
'''


def run_command(name, args, extra_env=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # A second notebook must not overwrite the active run's files or its log.
    with (OUTPUT / 'experiment.lock').open('a') if name == 'experiment' else nullcontext() as lock:
        if lock is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Q9 training is already running; follow environment/experiment.log') from None
        return _run_command(name, args, extra_env, lock.fileno() if lock is not None else None)


def _run_command(name, args, extra_env, lock_fd=None):
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', MAX_JOBS='4')
    if name.startswith('install-'):
        # NGC constrains its global Python. This isolated venv uses the shared
        # Q9 pins and explicit protected-package constraints instead.
        env.pop('PIP_CONSTRAINT', None)
        env.pop('PIP_BUILD_CONSTRAINT', None)
    env.update(extra_env or {})
    q1.write_json(OUTPUT / f'{name}-command.json', {'args': list(map(str, args)),
                                                   'extra_env': extra_env or {}})
    path = OUTPUT / f'{name}.log'
    if path.exists():
        archive = OUTPUT / 'logs'
        archive.mkdir(exist_ok=True)
        path.rename(archive / f'{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.log')
    print(f'{name}: log → {path.relative_to(ROOT)}', flush=True)
    with path.open('w') as stream:
        process = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            result = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if result:
        tail = '\n'.join(path.read_text(errors='replace').splitlines()[-25:])
        raise RuntimeError(f'{name} failed (exit {result}). {path}\n{tail}')
    return path


def checkout_sources():
    for path, url, revision in REPOSITORIES:
        if not (path / '.git').exists():
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f'Refusing to overwrite {path}')
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'init', '-q', str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', url], check=True)
            subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', revision], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'Source revision changed: {path}')
        subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--'], check=True)


def runtime_requirements():
    prefix = '# Q9-RUNTIME: '
    values = [line[len(prefix):] for line in (ROOT / 'requirements.txt').read_text().splitlines()
              if line.startswith(prefix)]
    if not values:
        raise RuntimeError('Q9 runtime requirements are missing from the shared requirements file')
    return values


def probe():
    path = run_command('probe', [PYTHON, '-c', PROBE])
    rows = [line for line in path.read_text().splitlines() if line.startswith('Q9_ENVIRONMENT=')]
    if len(rows) != 1:
        raise RuntimeError('Missing BioNeMo environment probe result')
    result = json.loads(rows[0].split('=', 1)[1])
    for package, version in PROTECTED.items():
        if result['packages'][package] != version:
            raise RuntimeError(f'Protected GPU environment changed: {package}')
    return result


def cuda_arch_list():
    """Compile for this machine's visible GPUs, or an explicit deployment target."""
    override = os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip()
    if override:
        return override
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('BioNeMo setup needs a visible CUDA GPU or TORCH_CUDA_ARCH_LIST.')
    capabilities = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    return ';'.join(f'{major}.{minor}' for major, minor in sorted(capabilities))


def setup():
    """Install only into Q9's venv; required Evo2 imports are checked afterward."""
    checkout_sources()
    requirements = runtime_requirements()
    architecture = cuda_arch_list()
    identity = {'sources': {str(p.relative_to(ROOT)): sha for p, _, sha in REPOSITORIES},
                'requirements': requirements, 'protected_packages': PROTECTED,
                'cuda_arch_list': architecture}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    marker = OUTPUT / 'setup.json'
    if marker.exists() and q1.read_json(marker)['identity'] == identity:
        result = probe()
        print('Existing BioNeMo environment passed its import and CUDA checks.')
        return result
    for package, version in PROTECTED.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Q9 setup requires the documented GPU base: {package}=={version}')
    if not PYTHON.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                        str(PYTHON.parent.parent)], check=True)
    constraints = OUTPUT / 'protected-packages.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in PROTECTED.items()))
    sources = [SOURCE / p for p in ['3rdparty/Megatron-LM', '3rdparty/NeMo',
               'sub-packages/bionemo-core', 'sub-packages/bionemo-llm', 'sub-packages/bionemo-evo2']]
    pip = [PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check']
    run_command('install-sources', pip + ['--no-deps', '--no-build-isolation'] + sources)
    runtime = [r for r in requirements if not r.startswith('causal-conv1d ')]
    upstream = SOURCE / '3rdparty/NeMo/requirements'
    dependency_files = ['requirements.txt', 'requirements_lightning.txt',
                        'requirements_common.txt', 'requirements_nlp.txt']
    args = pip + ['-c', constraints]
    for filename in dependency_files:
        args += ['-r', upstream / filename]
    run_command('install-runtime', args + runtime)
    causal = [r for r in requirements if r.startswith('causal-conv1d ')]
    run_command('install-causal', pip + ['--no-deps', '--no-build-isolation',
                                       '--force-reinstall', '--no-cache-dir'] + causal,
                {'CAUSAL_CONV1D_FORCE_BUILD': 'TRUE', 'TORCH_CUDA_ARCH_LIST': architecture})
    result = probe()
    q1.write_json(marker, {'identity': identity, 'result': result})
    run_command('installed-packages', [PYTHON, '-m', 'pip', 'list', '--format=json'])
    return result


def checkpoint():
    """Download hash-pinned source weights and use the supported Zarr converter."""
    from huggingface_hub import hf_hub_download
    target = ROOT / 'notebooks/results/q9/base_checkpoint_zarr'
    folder = ROOT / 'notebooks/results/q9'
    source = hf_hub_download('arcinstitute/savanna_evo2_1b_base', 'savanna_evo2_1b_base.pt',
                            revision='7217626d9f843e1830a5de1f5209c046570b6856',
                            local_dir=ROOT / 'data/evo2-savanna-1b')
    expected = '7bb731473c99db72aba34e7b0df443f2f422e12b0669bea6d78415dc87057a08'
    q1.verify_file(source, expected)
    q1.write_json(folder / 'checkpoint_source.json', {
        'repo': 'arcinstitute/savanna_evo2_1b_base',
        'revision': '7217626d9f843e1830a5de1f5209c046570b6856',
        'filename': Path(source).name, 'bytes': Path(source).stat().st_size, 'sha256': expected})
    if not target.exists():
        script = '''
from nemo.collections.llm.gpt.model.hyena import PyTorchHyenaImporter, Hyena1bConfig
PyTorchHyenaImporter('data/evo2-savanna-1b/savanna_evo2_1b_base.pt',
                    model_config=Hyena1bConfig()).apply(
                    'notebooks/results/q9/base_checkpoint_zarr', checkpoint_format='zarr')
'''
        run_command('convert-zarr', [PYTHON, '-c', script], {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
    if not (target / 'weights/metadata.json').exists() or not (target / 'context').exists():
        raise RuntimeError(f'Incomplete converted checkpoint: {target}; preserve it before retrying conversion')
    files = {str(p.relative_to(target)): q1.digest_file(p) for p in sorted(target.rglob('*')) if p.is_file()}
    record = {'source_sha256': expected, 'format': 'zarr', 'files': files,
              'reason': 'Pinned Megatron torch_dist writer is incompatible with PyTorch 2.7; public Zarr path works'}
    marker = folder / 'converted_checkpoint.json'
    if marker.exists() and q1.read_json(marker) != record:
        raise RuntimeError('Converted checkpoint files changed')
    q1.write_json(marker, record)
    return record
