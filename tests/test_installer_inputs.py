"""Exercise installer input validation without invoking the real Inno compiler."""
import re
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallerInputTests(unittest.TestCase):
    def setUp(self):
        self.shell = shutil.which('pwsh') or shutil.which('powershell')
        if not self.shell:
            self.skipTest('PowerShell is required')
        self.work = ROOT / '.runtime/beta6-fixes/installer' / uuid.uuid4().hex
        self.work.mkdir(parents=True)
        self.package = self.work / 'candidate with spaces'
        self.package.mkdir()
        (self.package / 'StereoLab.exe').write_bytes(b'candidate executable')
        (self.package / 'LICENSE').write_bytes(b'candidate license')
        (self.work / 'Start-StereoLab.ps1').write_bytes(b'checkout launcher')
        shutil.copyfile(ROOT / 'StereoLab.iss', self.work / 'StereoLab.iss')
        compiler = self.work / 'fake-compiler.ps1'
        compiler.write_text('''
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'compiler-invoked.txt') -Value 'called'
$global:LASTEXITCODE = 0
''', encoding='utf-8')
        script = (ROOT / 'build-installer.ps1').read_text(encoding='utf-8')
        script, count = re.subn(r"(?m)^\$iscc = '[^']+'", lambda match: "$iscc = '" + str(compiler).replace("'", "''") + "'", script)
        self.assertEqual(count, 1, 'Only replace the compiler location for this harness')
        (self.work / 'build-installer.ps1').write_text(script, encoding='utf-8')

    def run_builder(self):
        return subprocess.run([
            self.shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            str(self.work / 'build-installer.ps1'), '-PackageDir', str(self.package),
        ], capture_output=True, text=True, timeout=30)

    def test_selected_package_launcher_bytes_are_used(self):
        candidate_bytes = b'previously verified candidate launcher'
        (self.package / 'Start-StereoLab.ps1').write_bytes(candidate_bytes)
        result = self.run_builder()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.work / 'compiler-invoked.txt').exists())
        # Resolve the actual Inno Source rule, with the exact PackageDir supplied
        # above, so a checkout-relative launcher demonstrably selects wrong bytes.
        rule = re.search(r'^Source: "([^"]*Start-StereoLab\.ps1)";',
                         (self.work / 'StereoLab.iss').read_text(), re.M).group(1)
        source = rule.replace('{#PackageDir}', str(self.package)).replace('\\', '/')
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = self.work / source_path
        self.assertEqual(source_path.read_bytes(), candidate_bytes)

    def test_missing_launcher_rejected_before_compiler(self):
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Start-StereoLab.ps1', result.stdout + result.stderr)
        self.assertFalse((self.work / 'compiler-invoked.txt').exists())
