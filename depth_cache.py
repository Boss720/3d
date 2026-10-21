"""Persistent, exact float32 depth frames with bounded, validated storage.

Each entry contains a header, contiguous little-endian float32 frames, a JSON
manifest, and a length/digest footer. No manifest value is ever used as a path.
Publication uses an atomic rename in the cache directory. Leases and prospective
manifest reservations are shared by all instances in this Python process. A
nonblocking operating system lock excludes other processes while any lease is
held. The persistent single-byte lock file is excluded from payload quota.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import threading
import uuid

import numpy as np


DEFAULT_QUOTA_BYTES = 2 * 1024**3
MAX_ENTRY_BYTES = 16 * 1024**3
MAX_MANIFEST_BYTES = 64 * 1024**2
MAX_METADATA_BYTES = 256 * 1024
MAX_FRAME_DIMENSION = 16384
MAX_FRAME_ELEMENTS = 16 * 1024**2
MAX_FRAMES = 500000
_HEADER = b'STEREOLABDEPTH1\n'
_FOOTER = struct.Struct('<8sQ32s')
_FOOTER_MAGIC = b'SLDCEND1'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_PART = re.compile(r'([0-9a-f]{64})\.[0-9a-f]+\.part\Z')
_CHUNK = 1024 * 1024
_LOCK_NAME = '.depthcache.lock'


class CacheError(Exception):
    """Invalid, corrupt, unavailable, busy, or over-quota cache storage."""


@dataclass
class _RootState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    readers: dict[str, int] = field(default_factory=dict)
    writers: dict[str, int] = field(default_factory=dict)
    os_lock: object = None
    operations: int = 0


_STATES: dict[str, _RootState] = {}
_STATES_LOCK = threading.Lock()


def _key(value: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise CacheError('Cache keys and frame digests must be 64 lowercase hex characters')
    return value


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)
    )


def _regular(info: os.stat_result) -> None:
    if _is_link(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CacheError('Cache files must be regular files without links or reparse points')


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _safe_open(path: Path, writable=False):
    before = path.lstat()
    _regular(before)
    flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        actual = os.fstat(descriptor)
        _regular(actual)
        after = path.lstat()
        _regular(after)
        if _identity(actual) != _identity(before) or _identity(actual) != _identity(after):
            raise CacheError('Cache file changed while opening it')
        return os.fdopen(descriptor, 'r+b' if writable else 'rb', buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _encode(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('ascii')
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CacheError(f'Cache metadata must be bounded JSON: {exc}') from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CacheError('Duplicate JSON manifest property')
        result[key] = value
    return result


def _bad_constant(value):
    raise CacheError(f'Invalid JSON number: {value}')


def _shape(shape) -> tuple[int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise CacheError('Depth frames must have two dimensions')
    if any(type(n) is not int or n <= 0 or n > MAX_FRAME_DIMENSION for n in shape):
        raise CacheError('Depth frame dimensions exceed cache bounds')
    height, width = shape
    if height * width > MAX_FRAME_ELEMENTS:
        raise CacheError('Depth frame exceeds cache element limit')
    return height, width


def _manifest(key, frames, metadata):
    encoded = _encode({'version': 1, 'key': key, 'frames': frames, 'metadata': metadata})
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise CacheError('Cache manifest exceeds size limit')
    return encoded


def _validate(handle, key):
    """Validate all payload bytes before a cache hit; never load an entire entry."""
    size = os.fstat(handle.fileno()).st_size
    if size < len(_HEADER) + _FOOTER.size or size > MAX_ENTRY_BYTES:
        raise CacheError('Cache entry has an invalid or truncated size')
    handle.seek(-_FOOTER.size, os.SEEK_END)
    footer = handle.read(_FOOTER.size)
    if len(footer) != _FOOTER.size:
        raise CacheError('Cache footer is truncated')
    magic, manifest_size, expected_hash = _FOOTER.unpack(footer)
    if magic != _FOOTER_MAGIC or not 0 < manifest_size <= MAX_MANIFEST_BYTES:
        raise CacheError('Cache footer or manifest length is invalid')
    manifest_offset = size - _FOOTER.size - manifest_size
    if manifest_offset < len(_HEADER):
        raise CacheError('Cache manifest overlaps the header')
    handle.seek(manifest_offset)
    encoded = handle.read(manifest_size)
    if len(encoded) != manifest_size:
        raise CacheError('Cache manifest is truncated')
    try:
        manifest = json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CacheError(f'Invalid cache JSON manifest: {exc}') from exc
    if not isinstance(manifest, dict) or type(manifest.get('version')) is not int or manifest['version'] != 1:
        raise CacheError('Unsupported cache manifest version')
    if manifest.get('key') != key or not isinstance(manifest.get('metadata'), dict):
        raise CacheError('Cache key or metadata is invalid')
    if len(_encode(manifest['metadata'])) > MAX_METADATA_BYTES:
        raise CacheError('Cache metadata exceeds size limit')
    frames = manifest.get('frames')
    if not isinstance(frames, list) or not 0 < len(frames) <= MAX_FRAMES:
        raise CacheError('Cache frame count is invalid')
    offset = len(_HEADER)
    for frame in frames:
        if not isinstance(frame, dict):
            raise CacheError('Cache frame record is invalid')
        _key(frame.get('digest'))
        _key(frame.get('sha256'))
        height, width = _shape(frame.get('shape'))
        byte_count = height * width * 4
        if type(frame.get('offset')) is not int or frame['offset'] != offset:
            raise CacheError('Cache frame offsets must be contiguous')
        if type(frame.get('bytes')) is not int or frame['bytes'] != byte_count:
            raise CacheError('Cache frame byte count is invalid')
        offset += byte_count
        if offset > manifest_offset:
            raise CacheError('Cache frame extends beyond payload')
    if offset != manifest_offset:
        raise CacheError('Cache payload has unreferenced bytes')
    handle.seek(0)
    header = handle.read(len(_HEADER))
    if header != _HEADER:
        raise CacheError('Cache header is invalid')
    whole_hash = hashlib.sha256(header)
    for frame in frames:
        frame_hash = hashlib.sha256()
        remaining = frame['bytes']
        while remaining:
            chunk = handle.read(min(remaining, _CHUNK))
            if not chunk:
                raise CacheError('Cache frame is truncated')
            remaining -= len(chunk)
            whole_hash.update(chunk)
            frame_hash.update(chunk)
        if frame_hash.hexdigest() != frame['sha256']:
            raise CacheError('Cache frame digest does not match payload')
    whole_hash.update(encoded)
    if whole_hash.digest() != expected_hash:
        raise CacheError('Cache entry digest does not match content')
    return manifest


class DepthCache:
    def __init__(self, root: Path, quota_bytes: int = DEFAULT_QUOTA_BYTES):
        if type(quota_bytes) is not int or quota_bytes < 0:
            raise CacheError('Cache quota must be a nonnegative integer')
        self.root = Path(os.path.abspath(root))
        self.quota_bytes = quota_bytes
        self._check_root(allow_missing=True)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._check_root()
        except OSError as exc:
            raise CacheError(f'Cannot create cache directory: {exc}') from exc
        with _STATES_LOCK:
            self._state = _STATES.setdefault(os.path.normcase(str(self.root)), _RootState())

    def _acquire_os_lock(self):
        if self._state.os_lock is not None:
            self._check_root()
            try:
                info = (self.root / _LOCK_NAME).lstat()
                _regular(info)
                if _identity(info) != _identity(os.fstat(self._state.os_lock.fileno())):
                    raise CacheError('Cache ownership file changed during use')
            except OSError as exc:
                raise CacheError(f'Cannot verify cache ownership file: {exc}') from exc
            return
        self._check_root()
        path = self.root / _LOCK_NAME
        handle = None
        locked = False
        try:
            try:
                handle = _safe_open(path, writable=True)
            except FileNotFoundError:
                try:
                    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
                    handle = os.fdopen(descriptor, 'r+b', buffering=0)
                except FileExistsError:
                    handle = _safe_open(path, writable=True)
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            info = path.lstat()
            _regular(info)
            if _identity(info) != _identity(os.fstat(handle.fileno())):
                raise CacheError('Cache lock file changed while acquiring ownership')
            if info.st_size == 0:
                handle.seek(0)
                handle.write(b'\0')
                handle.flush()
            elif info.st_size != 1:
                raise CacheError('Cache lock file is invalid')
            self._state.os_lock = handle
        except (OSError, CacheError) as exc:
            if handle is not None:
                if locked:
                    self._unlock_handle(handle)
                else:
                    handle.close()
            raise CacheError(f'Depth cache is unavailable or owned by another process: {exc}') from exc

    @staticmethod
    def _unlock_handle(handle):
        try:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise CacheError(f'Cannot release cache ownership: {exc}') from exc
        finally:
            handle.close()

    def _release_idle_lock(self):
        if not self._state.operations and not self._state.readers and not self._state.writers:
            handle, self._state.os_lock = self._state.os_lock, None
            if handle is not None:
                self._unlock_handle(handle)

    @contextmanager
    def _operation(self):
        with self._state.lock:
            self._acquire_os_lock()
            self._state.operations += 1
            try:
                yield
            finally:
                self._state.operations -= 1
                self._release_idle_lock()

    def _check_root(self, allow_missing=False):
        try:
            for component in reversed((self.root, *self.root.parents)):
                try:
                    info = component.lstat()
                except FileNotFoundError:
                    if allow_missing:
                        continue
                    raise
                if _is_link(info) or not stat.S_ISDIR(info.st_mode):
                    raise CacheError('Cache root and its parents must be directories without links or reparse points')
        except OSError as exc:
            raise CacheError(f'Cannot inspect cache directory: {exc}') from exc

    def _files(self):
        self._check_root()
        return sorted(self.root.iterdir(), key=lambda p: p.name)

    def _usage(self):
        total = 0
        for path in self._files():
            info = path.lstat()
            _regular(info)
            if path.name == _LOCK_NAME:
                continue  # The single byte ownership file is not cached payload.
            total += info.st_size
        return total

    def _capacity(self, extra=0, reservation_delta=0):
        if self._usage() + sum(self._state.writers.values()) + extra + reservation_delta > self.quota_bytes:
            raise CacheError('Depth cache quota exceeded')

    def _active(self, key):
        return key in self._state.writers or self._state.readers.get(key, 0) > 0

    def open_reader(self, key: str) -> CacheReader | None:
        key = _key(key)
        with self._operation():
            self._check_root()
            if key in self._state.writers:
                return None
            handle = None
            try:
                handle = _safe_open(self.root / (key + '.depthcache'))
                manifest = _validate(handle, key)
                self._state.readers[key] = self._state.readers.get(key, 0) + 1
                return CacheReader(self, key, handle, manifest)
            except FileNotFoundError:
                if handle is not None:
                    handle.close()
                return None
            except (OSError, CacheError) as exc:
                if handle is not None:
                    handle.close()
                raise CacheError(f'Cannot open depth cache entry: {exc}') from exc

    def begin(self, key: str) -> CacheWriter:
        key = _key(key)
        with self._operation():
            if self._active(key):
                raise CacheError('Cache entry is active')
            self._check_root()
            existing = self.root / (key + '.depthcache')
            try:
                if existing.exists() or existing.is_symlink():
                    _regular(existing.lstat())
                reservation = len(_manifest(key, [], {})) + _FOOTER.size
                self._capacity(extra=len(_HEADER), reservation_delta=reservation)
                part = self.root / (key + '.' + uuid.uuid4().hex + '.part')
                descriptor = os.open(part, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, 'O_BINARY', 0), 0o600)
                handle = os.fdopen(descriptor, 'w+b', buffering=0)
                writer = CacheWriter(self, key, part, handle)
                self._state.writers[key] = reservation
                writer._owns_lease = True
                try:
                    writer._write(_HEADER)
                except BaseException:
                    writer.abort()
                    raise
                return writer
            except (OSError, CacheError) as exc:
                raise CacheError(f'Cannot begin depth cache entry: {exc}') from exc

    def inventory(self) -> dict:
        try:
            with self._operation():
                return self._inventory_locked()
        except (OSError, CacheError) as exc:
            return {'quota_bytes': self.quota_bytes, 'bytes': 0, 'entries': [],
                    'errors': [{'key': '', 'error': str(exc)}]}

    def _inventory_locked(self):
        with self._state.lock:
            result = {'quota_bytes': self.quota_bytes, 'bytes': 0, 'entries': [], 'errors': []}
            try:
                paths = self._files()
            except (OSError, CacheError) as exc:
                result['errors'].append({'key': '', 'error': str(exc)})
                return result
            entries = {}
            for path in paths:
                key = path.name[:64] if _HEX.fullmatch(path.name[:64]) else ''
                try:
                    info = path.lstat()
                    _regular(info)
                    if path.name == _LOCK_NAME:
                        continue
                    result['bytes'] += info.st_size
                    is_part = _PART.fullmatch(path.name) is not None
                    if path.name != key + '.depthcache' and not is_part:
                        raise CacheError('Unrecognized cache file')
                    entry = entries.setdefault(key, {'key': key, 'bytes': 0, 'frames': 0, 'active': self._active(key)})
                    entry['bytes'] += info.st_size
                    if is_part:
                        if key not in self._state.writers:
                            raise CacheError('Incomplete cache file')
                        continue
                    with _safe_open(path) as handle:
                        entry['frames'] = len(_validate(handle, key)['frames'])
                except (OSError, CacheError) as exc:
                    result['errors'].append({'key': key, 'error': str(exc)})
            result['entries'] = [entries[key] for key in sorted(entries)]
            return result

    def delete(self, keys: list[str]) -> dict:
        try:
            with self._operation():
                return self._delete_locked(keys)
        except (OSError, CacheError) as exc:
            return {'deleted': [], 'errors': [{'key': key, 'error': str(exc)} for key in keys]}

    def _delete_locked(self, keys):
        result = {'deleted': [], 'errors': []}
        with self._state.lock:
            for value in dict.fromkeys(keys):
                try:
                    key = _key(value)
                    if self._active(key):
                        raise CacheError('Cache entry is active')
                    paths = [p for p in self._files() if p.name == key + '.depthcache' or (
                        _PART.fullmatch(p.name) and p.name.startswith(key + '.')
                    )]
                    # Validate the entire selection before removing any of it.
                    for path in paths:
                        _regular(path.lstat())
                    for path in paths:
                        _regular(path.lstat())
                        os.unlink(path)
                    if paths:
                        result['deleted'].append(key)
                except (OSError, CacheError) as exc:
                    result['errors'].append({'key': value, 'error': str(exc)})
        return result


class CacheReader:
    def __init__(self, cache, key, handle, manifest):
        self.metadata = manifest['metadata']
        self.frame_count = len(manifest['frames'])
        self._cache, self._key, self._handle = cache, key, handle
        self._frames = manifest['frames']
        self._position = 0
        self._lock = threading.RLock()

    def read(self, frame_digest: str) -> np.ndarray:
        with self._lock:
            if self._handle is None:
                raise CacheError('Cache reader is closed')
            _key(frame_digest)
            if self._position >= self.frame_count:
                raise CacheError('Cache reader has no remaining frames')
            frame = self._frames[self._position]
            if frame_digest != frame['digest']:
                raise CacheError('Decoded source frame digest or ordering does not match cache')
            try:
                self._handle.seek(frame['offset'])
                payload = self._handle.read(frame['bytes'])
                if len(payload) != frame['bytes'] or hashlib.sha256(payload).hexdigest() != frame['sha256']:
                    raise CacheError('Cache frame changed or became truncated after validation')
                # The copy owns its data; callers can mutate it without aliasing cache data.
                restored = np.empty(tuple(frame['shape']), dtype=np.float32)
                restored[:] = np.frombuffer(payload, dtype='<f4').reshape(tuple(frame['shape']))
                self._position += 1
                return restored
            except OSError as exc:
                raise CacheError(f'Cannot read cached depth frame: {exc}') from exc

    def close(self):
        with self._lock:
            if self._handle is None:
                return
            handle, self._handle = self._handle, None
            try:
                handle.close()
            finally:
                with self._cache._state.lock:
                    count = self._cache._state.readers.get(self._key, 0)
                    if count <= 1:
                        self._cache._state.readers.pop(self._key, None)
                    else:
                        self._cache._state.readers[self._key] = count - 1
                    self._cache._release_idle_lock()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CacheWriter:
    def __init__(self, cache, key, path, handle):
        self._cache, self._key, self._path, self._handle = cache, key, path, handle
        self._identity = _identity(os.fstat(handle.fileno()))
        self._frames = []
        self._hasher = hashlib.sha256()
        self._bytes = 0
        self._done = False
        self._owns_lease = False
        self._manifest_size = len(_manifest(key, [], {}))

    def _write(self, payload):
        view = memoryview(payload)
        while view:
            written = self._handle.write(view)
            if not written:
                raise CacheError('Cannot write complete cache payload')
            self._hasher.update(view[:written])
            self._bytes += written
            view = view[written:]

    def _check(self):
        if self._done or self._handle is None:
            raise CacheError('Cache writer is closed')
        self._cache._check_root()
        info = self._path.lstat()
        _regular(info)
        if _identity(info) != self._identity or info.st_size != self._bytes:
            raise CacheError('Cache temporary file changed during writing')

    def append(self, depth: np.ndarray, frame_digest: str):
        with self._cache._operation():
            try:
                self._check()
                _key(frame_digest)
                if not isinstance(depth, np.ndarray) or depth.dtype != np.dtype(np.float32):
                    raise CacheError('Cached depth must be a float32 NumPy array')
                shape = _shape(tuple(depth.shape))
                if len(self._frames) >= MAX_FRAMES:
                    raise CacheError('Cache frame count exceeds limit')
                payload = depth.astype('<f4', copy=False).tobytes(order='C')
                frame = {'digest': frame_digest, 'shape': list(shape), 'offset': self._bytes,
                         'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                prospective_size = self._manifest_size + len(_encode(frame)) + bool(self._frames)
                if prospective_size > MAX_MANIFEST_BYTES:
                    raise CacheError('Cache manifest exceeds size limit')
                reservation = prospective_size + _FOOTER.size
                if self._bytes + len(payload) + reservation > MAX_ENTRY_BYTES:
                    raise CacheError('Cache entry exceeds size limit')
                self._cache._capacity(extra=len(payload), reservation_delta=reservation - self._cache._state.writers[self._key])
                self._cache._state.writers[self._key] = reservation
                self._write(payload)
                self._frames.append(frame)
                self._manifest_size = prospective_size
            except OSError as exc:
                raise CacheError(f'Cannot append cached depth frame: {exc}') from exc

    def finish(self, metadata: dict):
        try:
            with self._cache._operation():
                self._check()
                if not self._frames:
                    raise CacheError('Cannot publish an empty depth cache entry')
                if not isinstance(metadata, dict):
                    raise CacheError('Cache metadata must be a bounded JSON object')
                metadata_bytes = _encode(metadata)
                if len(metadata_bytes) > MAX_METADATA_BYTES:
                    raise CacheError('Cache metadata must be a bounded JSON object')
                metadata = json.loads(metadata_bytes, object_pairs_hook=_unique_object,
                                      parse_constant=_bad_constant)
                encoded = _manifest(self._key, self._frames, metadata)
                added = len(encoded) + _FOOTER.size
                if self._bytes + added > MAX_ENTRY_BYTES:
                    raise CacheError('Cache entry exceeds size limit')
                self._cache._capacity(extra=added, reservation_delta=-self._cache._state.writers[self._key])
                self._write(encoded)
                footer = _FOOTER.pack(_FOOTER_MAGIC, len(encoded), self._hasher.digest())
                self._write(footer)
                self._cache._state.writers[self._key] = 0
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._check()
                self._handle.close()
                self._handle = None
                destination = self._cache.root / (self._key + '.depthcache')
                if destination.exists() or destination.is_symlink():
                    _regular(destination.lstat())
                os.replace(self._path, destination)
                self._done = True
                self._cache._state.writers.pop(self._key, None)
                self._owns_lease = False
        except (OSError, CacheError) as exc:
            try:
                self.abort()
            except CacheError as cleanup:
                raise CacheError(f'Cannot finish depth cache: {exc}; cleanup failed: {cleanup}') from exc
            raise CacheError(f'Cannot finish depth cache: {exc}') from exc

    def abort(self):
        # Releasing an owned handle/lease must not depend on the filesystem
        # inspection that guards unlink. It may be the very operation failing.
        with self._cache._state.lock:
            if self._done:
                return
            failure = None
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError as exc:
                    failure = exc
                self._handle = None
            try:
                with self._cache._operation():
                    self._cache._check_root()
                    info = self._path.lstat()
                    _regular(info)
                    if _identity(info) != self._identity:
                        raise CacheError('Cache temporary file was replaced; refusing cleanup')
                    os.unlink(self._path)
            except FileNotFoundError:
                pass
            except (OSError, CacheError) as exc:
                failure = exc
            finally:
                if self._owns_lease:
                    self._cache._state.writers.pop(self._key, None)
                    self._owns_lease = False
                try:
                    self._cache._release_idle_lock()
                except (OSError, CacheError) as exc:
                    failure = exc
            if failure is not None:
                raise CacheError(f'Cannot clean up depth cache temporary file: {failure}') from failure
            self._done = True

    def __del__(self):
        try:
            self.abort()
        except Exception:
            pass
