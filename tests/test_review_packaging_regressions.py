"""Behavioral regressions for release identity and isolated packaging."""
import shutil
import subprocess
import uuid
import unittest
from pathlib import Path

from build_identity import SOURCE_COMPONENTS, build_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class PackagingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = ROOT / ".runtime" / "packaging-regression-tests" / uuid.uuid4().hex
        self.tmp_path.mkdir(parents=True)

    def test_gpu_adapter_code_changes_invalidate_build_identity(self):
        tmp_path = self.tmp_path
        for name in set(SOURCE_COMPONENTS) | {'gpu_hardware.py'}:
            (tmp_path / name).write_text(name, encoding='utf-8')
        before = build_fingerprint(tmp_path)
        self.assertIsNotNone(before)
        (tmp_path / 'gpu_hardware.py').write_text('changed GPU selection', encoding='utf-8')
        self.assertNotEqual(build_fingerprint(tmp_path), before)


    def test_build_script_honors_explicit_toolchain_and_isolated_paths(self):
        tmp_path = self.tmp_path
        shell = shutil.which('pwsh') or shutil.which('powershell')
        if not shell:
            self.skipTest('PowerShell is required to execute the Windows build script')
        fake_python = tmp_path / 'fake-python.ps1'
        fake_python.write_text('''
    $global:LASTEXITCODE = 0
    if ($args -contains 'PyInstaller') {
        $distIndex = [array]::IndexOf($args, '--distpath')
        $workIndex = [array]::IndexOf($args, '--workpath')
        if ($distIndex -lt 0 -or $workIndex -lt 0) { throw 'Missing isolated paths' }
        $dist = $args[$distIndex + 1]
        $work = $args[$workIndex + 1]
        New-Item -ItemType Directory -Path $dist, $work -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $dist 'StereoLab.exe') -Value 'mock packaged EXE'
        Set-Content -LiteralPath (Join-Path $work 'toolchain-used.txt') -Value 'explicit Python'
    }
    ''', encoding='utf-8')
        dist = tmp_path / 'isolated dist'
        work = tmp_path / 'isolated work'
        script_literal = str(ROOT / 'build-exe.ps1').replace("'", "''")
        metadata = subprocess.run([
            shell, '-NoProfile', '-Command',
            f"$p = (Get-Command '{script_literal}').Parameters; "
            "if (-not $p -or -not $p.ContainsKey('PythonPath') -or "
            "-not $p.ContainsKey('DistPath') -or -not $p.ContainsKey('WorkPath')) { exit 99 }",
        ], capture_output=True, text=True, timeout=15)
        self.assertEqual(metadata.returncode, 0, 'Build script cannot select an isolated toolchain/output')
        result = subprocess.run([
            shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            str(ROOT / 'build-exe.ps1'), '-PythonPath', str(fake_python),
            '-DistPath', str(dist), '-WorkPath', str(work),
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((dist / 'StereoLab.exe').read_text().strip(), 'mock packaged EXE')
        self.assertTrue((work / 'toolchain-used.txt').exists())
        self.assertEqual((dist / 'LICENSE').read_bytes(), (ROOT / 'LICENSE').read_bytes())
        self.assertEqual((dist / 'Start-StereoLab.ps1').read_bytes(), (ROOT / 'Start-StereoLab.ps1').read_bytes())
