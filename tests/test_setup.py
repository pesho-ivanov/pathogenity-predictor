"""Portable setup must preserve the compiled stack and select the actual GPUs."""

import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import setup
from notebooks.src import q9_environment


class SetupTests(unittest.TestCase):
    def test_cpu_requirements_preserve_shared_pins_without_gpu_install(self):
        original = (setup.ROOT / 'requirements.txt').read_bytes()
        requirements = setup.requirements()
        self.assertIn('numpy==1.26.4', requirements)
        self.assertIn('huggingface-hub==1.30.0', requirements)
        self.assertTrue(all('==' in item for item in requirements))
        self.assertFalse(any('vtx' in item or 'evo2' in item for item in requirements))
        self.assertEqual((setup.ROOT / 'requirements.txt').read_bytes(), original)

    def test_existing_cpu_environment_is_not_silently_changed_to_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            config = destination / 'pyvenv.cfg'
            config.write_text('include-system-site-packages = false\n')
            with patch.object(setup, 'gpu_base'), patch.object(setup, 'ensure_submodule'), \
                    patch.object(setup, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'another profile'):
                    setup.install('gpu', destination)
                run.assert_not_called()
            self.assertEqual(config.read_text(), 'include-system-site-packages = false\n')

    def test_gpu_requirement_changes_are_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'requirements.txt').write_text('numpy==1.26.4\nvtx==2.0\n-e ./evo2\n')
            with self.assertRaisesRegex(RuntimeError, 'GPU requirements changed'):
                setup.requirements(root)

    def test_missing_submodule_check_does_not_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(setup, 'ROOT', Path(temporary)), \
                patch.object(setup, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'submodule is missing'):
                setup.ensure_submodule()
            run.assert_not_called()

    def test_compiler_targets_all_visible_architectures(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 3,
                               get_device_capability=lambda index: [(9, 0), (8, 9), (9, 0)][index])
        with patch.dict(os.environ, {'TORCH_CUDA_ARCH_LIST': ''}), \
                patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}):
            self.assertEqual(q9_environment.cuda_arch_list(), '8.9;9.0')

    def test_explicit_compiler_target_is_respected(self):
        with patch.dict(os.environ, {'TORCH_CUDA_ARCH_LIST': '9.0+PTX'}):
            self.assertEqual(q9_environment.cuda_arch_list(), '9.0+PTX')

    def test_missing_gpu_does_not_fall_back_to_l40s(self):
        cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(os.environ, {'TORCH_CUDA_ARCH_LIST': ''}), \
                patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError, 'visible CUDA GPU'):
                q9_environment.cuda_arch_list()

    def test_new_gpu_architecture_forces_uncached_extension_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / 'python'
            python.touch()
            with patch.object(q9_environment, 'OUTPUT', root / 'environment'), \
                    patch.object(q9_environment, 'PYTHON', python), \
                    patch.object(q9_environment, 'checkout_sources'), \
                    patch.object(q9_environment, 'runtime_requirements', return_value=[
                        'causal-conv1d @ git+https://example.test/source@fixed']), \
                    patch.object(q9_environment.importlib.metadata, 'version',
                                 side_effect=q9_environment.PROTECTED.__getitem__), \
                    patch.object(q9_environment, 'probe', return_value={'passed': True}), \
                    patch.object(q9_environment, 'run_command') as run, \
                    patch.object(q9_environment, 'cuda_arch_list', side_effect=['8.9', '9.0']):
                q9_environment.setup()
                run.reset_mock()
                q9_environment.setup()
                build = next(call for call in run.call_args_list if call.args[0] == 'install-causal')
                self.assertIn('--force-reinstall', build.args[1])
                self.assertIn('--no-cache-dir', build.args[1])
                self.assertEqual(build.args[2]['TORCH_CUDA_ARCH_LIST'], '9.0')

    def test_ngc_constraints_do_not_override_isolated_bionemo_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(q9_environment, 'ROOT', root), \
                    patch.object(q9_environment, 'OUTPUT', root / 'environment'), \
                    patch.dict(os.environ, {'PIP_CONSTRAINT': '/container/constraints.txt'}):
                log = q9_environment.run_command('install-probe', [sys.executable, '-c',
                    'import os; print(os.environ.get("PIP_CONSTRAINT", "isolated"))'])
                self.assertEqual(log.read_text().strip(), 'isolated')
                self.assertEqual(os.environ['PIP_CONSTRAINT'], '/container/constraints.txt')


if __name__ == '__main__':
    unittest.main()
