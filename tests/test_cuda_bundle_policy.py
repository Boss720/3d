"""Reject CUDA bundles whose installed cuDNN drops supported legacy GPUs."""
import importlib.metadata
import runpy
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ReachedAnalysis(Exception):
    pass


class CudaBundlePolicyTests(unittest.TestCase):
    def run_spec(self, cudnn_version, *, check_backups=False):
        work = ROOT / '.runtime' / 'cuda-bundle-policy-tests' / uuid.uuid4().hex
        site = work / 'site'
        binaries = site / 'nvidia' / 'cudnn' / 'bin'
        binaries.mkdir(parents=True)
        for name in ('cublas64_12.dll', 'cudart64_12.dll', 'cudnn64_9.dll'):
            (binaries / name).write_bytes(b'test library')
        backup = site / 'nvidia' / 'cudnn' / '~in'
        backup.mkdir()
        (backup / 'cudnn64_9.dll').write_bytes(b'stale incompatible cuDNN')
        hooks = types.ModuleType('PyInstaller.utils.hooks')
        hooks.collect_dynamic_libs = lambda name: [('onnxruntime_providers_cuda.dll', 'onnxruntime/capi')]
        hooks.collect_submodules = lambda name: []
        config = types.ModuleType('PyInstaller.config')
        config.CONF = {'workpath': str(work / 'build')}
        ort = types.ModuleType('onnxruntime')
        ort.get_available_providers = lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider']
        def stop_analysis(*args, **kwargs):
            if check_backups:
                self.assertFalse(any('~in' in source for source, _ in kwargs['binaries']),
                                 'pip uninstall backup DLLs must not enter the CUDA bundle')
            raise ReachedAnalysis('Preflight accepted; no real build executed')
        modules = {'PyInstaller': types.ModuleType('PyInstaller'),
                   'PyInstaller.utils': types.ModuleType('PyInstaller.utils'),
                   'PyInstaller.utils.hooks': hooks, 'PyInstaller.config': config,
                   'onnxruntime': ort}
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'STEREOLAB_CUDA_BUNDLE': '1'}), \
                patch('site.getsitepackages', return_value=[str(site)]), \
                patch('importlib.metadata.version', return_value=cudnn_version):
            runpy.run_path(str(ROOT / 'StereoLab.spec'), init_globals={'SPECPATH': str(ROOT), 'Analysis': stop_analysis})

    def test_modern_cudnn_cannot_silently_drop_legacy_gpu_support(self):
        with self.assertRaisesRegex(RuntimeError, '9.10.2.21'):
            self.run_spec('9.26.1.1')

    def test_qualified_cudnn_reaches_analysis(self):
        with self.assertRaises(ReachedAnalysis):
            self.run_spec('9.10.2.21')

    def test_pip_uninstall_backup_dlls_do_not_enter_bundle(self):
        with self.assertRaises(ReachedAnalysis):
            self.run_spec('9.10.2.21', check_backups=True)
