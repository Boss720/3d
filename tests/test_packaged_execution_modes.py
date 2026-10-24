"""Keep packaged acceptance strict for each explicitly selected GPU backend."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts import verify_packaged_release as verifier


class PackagedExecutionModeTests(unittest.TestCase):
    def test_launcher_identity_fills_null_path_without_changing_pid_or_start(self):
        expected = Path('candidate/StereoLab.exe').resolve()
        original = {'pid': 41264, 'path': None, 'started': '2026-10-23T04:00:00Z', 'parent_pid': 100}
        normalized = verifier.canonical_launcher_identity(original, expected)
        self.assertEqual(normalized, {**original, 'path': str(expected)})
        self.assertIsNone(original['path'])

    def test_launcher_identity_preserves_correct_path_and_rejects_wrong_path(self):
        expected = Path('candidate/StereoLab.exe').resolve()
        original = {'pid': 41264, 'path': str(expected), 'started': '2026-10-23T04:00:00Z'}
        self.assertEqual(verifier.canonical_launcher_identity(original, expected), original)
        with self.assertRaises(AssertionError):
            verifier.canonical_launcher_identity({**original, 'path': str(expected.parent / 'other.exe')}, expected)

    def test_cli_default_and_explicit_cuda(self):
        parser = verifier.build_parser()
        required = ['--exe', 'app.exe', '--data', 'data', '--report', 'report.json', '--ffmpeg', 'ffmpeg.exe']
        self.assertEqual(parser.parse_args(required).execution_mode, 'gpu')
        self.assertEqual(parser.parse_args(required + ['--execution-mode', 'cuda']).execution_mode, 'cuda')

    def test_submission_uses_selected_strict_mode(self):
        for mode in ('gpu', 'cuda'):
            with self.subTest(mode=mode):
                instance = object.__new__(verifier.Verifier)
                instance.args = SimpleNamespace(layout='SBS', execution_mode=mode)
                instance.source = Mock()
                instance.source.open.return_value = io.BytesIO(b'fixture')
                captured = {}

                def request(method, path, **kwargs):
                    if method == 'GET':
                        return SimpleNamespace(json=lambda: [])
                    captured.update(kwargs['data'])
                    raise InterruptedError('stop before admission')

                instance.request = request
                with self.assertRaises(InterruptedError):
                    instance.submit('fixture')
                self.assertEqual(captured['execution_mode'], mode)

    def test_fresh_execution_requires_selected_provider_and_no_fallback(self):
        for mode, provider in (('gpu', 'DmlExecutionProvider'), ('cuda', 'CUDAExecutionProvider')):
            good = {'active_provider': provider, 'gpu_verified': True, 'fallback_reason': None}
            verifier.validate_execution(good, mode)
            for mutation in ({'active_provider': 'CPUExecutionProvider'},
                             {'active_provider': 'CUDAExecutionProvider' if mode == 'gpu' else 'DmlExecutionProvider'},
                             {'gpu_verified': False}, {'fallback_reason': 'backend failed'}):
                with self.subTest(mode=mode, mutation=mutation), self.assertRaises(AssertionError):
                    verifier.validate_execution({**good, **mutation}, mode)

    def test_cache_requires_cpu_execution_and_selected_backend_provenance(self):
        for mode, provider in (('gpu', 'DmlExecutionProvider'), ('cuda', 'CUDAExecutionProvider')):
            good = {'device': 'cpu', 'active_provider': None, 'gpu_verified': False,
                    'depth_cache_hit': True, 'cached_execution': {'active_provider': provider}}
            verifier.validate_execution(good, mode, cached=True)
            for mutation in ({'device': 'gpu'}, {'active_provider': provider}, {'gpu_verified': True},
                             {'depth_cache_hit': False}, {'cached_execution': {}},
                             {'cached_execution': {'active_provider': 'CUDAExecutionProvider' if mode == 'gpu' else 'DmlExecutionProvider'}}):
                with self.subTest(mode=mode, mutation=mutation), self.assertRaises(AssertionError):
                    verifier.validate_execution({**good, **mutation}, mode, cached=True)
