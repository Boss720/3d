"""Local disk inventory, explicitly selected temporary cleanup, and desktop actions."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from media_process import external_process_environment


TEMP_CATEGORIES = frozenset({'uploads', 'previews'})


def _plain_path(value: str | Path) -> Path:
    """Reject links/junctions before resolution, including parent components."""
    path = Path(os.path.abspath(value))
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Links and junctions are not supported for this action.')
    return path


def _protected(path: Path, protected_paths: Iterable[str | Path]) -> bool:
    for value in protected_paths:
        other = Path(os.path.abspath(value))
        if path == other or other in path.parents:
            return True
    return False


def storage_inventory(roots: Mapping[str, Path], protected_paths=()) -> dict:
    """Count regular files without traversing links; list inactive temporary files."""
    protected_paths = (*protected_paths, *(value for name, value in roots.items() if name not in TEMP_CATEGORIES))
    result = {'categories': {}, 'candidates': [], 'errors': []}
    for name, value in roots.items():
        category = {'path': str(Path(os.path.abspath(value))), 'bytes': 0, 'files': 0, 'errors': []}
        result['categories'][name] = category
        try:
            root = _plain_path(value)
            if not root.exists():
                continue
            if not root.is_dir():
                raise ValueError('Storage root must be a directory.')
            pending = [root]
            while pending:
                folder = pending.pop()
                try:
                    # Recheck directory ancestry immediately before scanning it.
                    entries = list(_plain_path(folder).iterdir())
                except (OSError, ValueError) as exc:
                    category['errors'].append(str(exc))
                    continue
                for entry in entries:
                    try:
                        path = _plain_path(entry)
                        info = path.lstat()
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                        elif stat.S_ISREG(info.st_mode):
                            category['bytes'] += info.st_size
                            category['files'] += 1
                            if name in TEMP_CATEGORIES and not _protected(path, protected_paths):
                                result['candidates'].append({'category': name, 'path': str(path), 'bytes': info.st_size})
                    except (OSError, ValueError) as exc:
                        category['errors'].append(f'{entry.name}: {exc}')
        except (OSError, ValueError) as exc:
            category['errors'].append(str(exc))
        result['errors'].extend(f'{name}: {error}' for error in category['errors'])
    result['candidates'].sort(key=lambda item: item['path'])
    return result


def cleanup_temporary_files(selected_paths, roots: Mapping[str, Path], protected_paths=()) -> dict:
    """Delete only selected regular upload/preview files; never delete directories."""
    result = {'deleted': [], 'errors': []}
    protected_paths = tuple(protected_paths)
    for value in dict.fromkeys(map(str, selected_paths)):
        try:
            path = _plain_path(value)
            eligible = False
            for name, root_value in roots.items():
                root = _plain_path(root_value)
                if path == root or root in path.parents:
                    # A protected category wins even when roots overlap.
                    if name not in TEMP_CATEGORIES:
                        raise ValueError('Models and outputs cannot be removed by temporary cleanup.')
                    if path != root:
                        eligible = True
            if not eligible:
                raise ValueError('Select a file inside the uploads or previews folder.')
            if _protected(path, protected_paths):
                raise ValueError('This file is in use by an active job.')
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('Only individual regular files can be removed.')
            _plain_path(path).unlink()
            result['deleted'].append(str(path))
        except (OSError, ValueError) as exc:
            result['errors'].append({'path': value, 'error': str(exc)})
    return result


def validate_output_directory(value: str | Path) -> Path:
    """Require an existing ordinary directory and verify a file can be created."""
    path = _plain_path(value)
    if not path.is_dir():
        raise ValueError('Choose an existing output directory.')
    with tempfile.TemporaryFile(dir=path):
        pass
    return path


def open_folder(value: str | Path) -> None:
    path = _plain_path(value)
    if not path.is_dir():
        raise ValueError('Folder does not exist.')
    if os.name == 'nt':
        subprocess.Popen(['explorer.exe', str(path)], shell=False)
        return
    _open_linux_location(path)


def reveal_file(value: str | Path) -> None:
    path = _plain_path(value)
    if not path.is_file():
        raise ValueError('Output file does not exist.')
    if os.name == 'nt':
        subprocess.Popen(['explorer.exe', '/select,', str(path)], shell=False)
        return
    # Linux desktop environments do not share a portable file-selection API.
    # Opening the containing folder is safe and works with the common openers.
    _open_linux_location(path.parent)


def _open_linux_location(path: Path) -> None:
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Desktop folder actions are supported on Windows and Linux.')
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('Cannot open a folder: no graphical desktop session is available.')
    opener = shutil.which('xdg-open') or shutil.which('gio')
    if not opener:
        raise RuntimeError('Cannot open a folder: install xdg-open or gio and try again.')
    command = [opener, 'open', str(path)] if Path(opener).name == 'gio' else [opener, str(path)]
    subprocess.Popen(command, shell=False, env=external_process_environment())
