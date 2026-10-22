#!/usr/bin/env python3
"""Record and validate the ELF dependency closure of a Linux package."""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def inspect_package(package: Path) -> dict:
    if not sys.platform.startswith('linux'):
        raise RuntimeError('ELF dependency inspection must run on Linux.')
    ldd = shutil.which('ldd')
    if not ldd:
        raise RuntimeError('ldd was not found; install the system libc utilities.')

    files = []
    missing = []
    for path in sorted(package.rglob('*')):
        if not path.is_file():
            continue
        try:
            with path.open('rb') as stream:
                if stream.read(4) != b'\x7fELF':
                    continue
        except OSError as exc:
            raise RuntimeError(f'Could not inspect package file {path}: {exc}') from exc

        result = subprocess.run([ldd, str(path)], capture_output=True, text=True,
                                timeout=20, check=False)
        output = (result.stdout + result.stderr).strip()
        unresolved = [line.strip() for line in output.splitlines()
                      if re.search(r'\bnot found\b', line)]
        if result.returncode and 'not a dynamic executable' not in output.lower():
            raise RuntimeError(f'ldd failed for {path}: {output or result.returncode}')
        rel = path.relative_to(package).as_posix()
        files.append({'path': rel, 'ldd_exit_code': result.returncode, 'ldd': output})
        missing.extend({'path': rel, 'dependency': line} for line in unresolved)

    if not files:
        raise RuntimeError(f'No ELF files were found in {package}.')
    return {
        'schema_version': 1,
        'architecture': platform.machine(),
        'platform': platform.platform(),
        'libc': platform.libc_ver(),
        'elf_count': len(files),
        'files': files,
        'missing_dependencies': missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    if not package.is_dir():
        parser.error(f'package directory does not exist: {package}')
    report = inspect_package(package)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding='utf-8')
    print(json.dumps({key: report[key] for key in
                      ('architecture', 'libc', 'elf_count', 'missing_dependencies')},
                     sort_keys=True))
    return 1 if report['missing_dependencies'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
