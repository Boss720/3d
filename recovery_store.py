"""Opt-in source retention metadata; conversion is always explicit.

The manifest contains metadata only. The application owns source cleanup and
must protect source_path() values from automatic storage cleanup. A recovery
checkpoint records diagnostic progress; restart always processes the full
source from its beginning. No converter is called by this module.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
import threading
import time
import unicodedata
import uuid
from pathlib import Path

from depth_cache_identity import hash_source
from job_settings import validate_settings
from output_naming import validate_output_template


SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 60 * 1024
_STATES = {'queued', 'processing', 'interrupted', 'claimed', 'completed', 'failed', 'canceled'}
_TERMINAL = {'completed', 'failed', 'canceled', 'interrupted'}
_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.ts', '.mts',
               '.m2ts', '.m4v', '.jpg', '.jpeg', '.png', '.webp'}
_FIELDS = {'job_id', 'filename', 'source_sha256', 'source_size_bytes', 'settings',
           'output_name_template', 'state', 'frames_processed', 'updated_at',
           'claimed_by', 'previous_state', 'previous_frames_processed'}


def _uuid(value):
    if type(value) is str:
        try:
            if str(uuid.UUID(value)) == value:
                return value
        except ValueError:
            pass
    raise ValueError('Recovery identities must be canonical UUIDs.')


def _filename(value):
    if (type(value) is not str or not 1 <= len(value) <= 255
            or any(char in value for char in '/\\:') or value in {'.', '..'}
            or any(unicodedata.category(char) in {'Cc', 'Cf', 'Cs'} for char in value)
            or Path(value).suffix.lower() not in _EXTENSIONS):
        raise ValueError('Recovery filename must be a supported media basename.')


def _count(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'{name} must be an integer from 0 to {maximum}.')


def _regular(path):
    details = path.lstat()
    if (not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or path.is_symlink()
            or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RecoveryUnavailable('Recovery requires regular files without symbolic or hard links.')
    return details


def _namespace(path):
    if path.resolve() != path.absolute():
        raise RecoveryUnavailable('Recovery paths cannot traverse filesystem links.')
    for part in (path, *path.parents):
        if part.exists():
            details = part.lstat()
            if (part.is_symlink() or getattr(details, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                raise RecoveryUnavailable('Recovery paths cannot traverse filesystem links.')


class RecoveryUnavailable(RuntimeError):
    pass


class RecoveryLeaseError(RecoveryUnavailable):
    pass


class RecoveryCapacityError(RecoveryUnavailable):
    pass


class RecoveryConflict(RecoveryUnavailable):
    pass


class RecoverySourceError(RecoveryUnavailable):
    pass


class RecoveryStore:
    def __init__(self, directory, uploads_root, *, max_records=32, max_bytes=16 * 1024**3,
                 clock=time.time):
        _count(max_records, 'Retained job limit', 32)
        _count(max_bytes, 'Retained byte limit', 16 * 1024**3)
        if max_records < 1 or max_bytes < 1:
            raise ValueError('Recovery capacity limits must be positive.')
        self.directory, self.uploads_root = Path(directory).absolute(), Path(uploads_root).absolute()
        self.manifest_path = self.directory / 'recovery.json'
        self.max_records, self.max_bytes = max_records, max_bytes
        self._clock = clock
        self._lock = threading.RLock()
        self._lease_fd = None
        self._records = {}

    @property
    def enabled(self):
        return self._lease_fd is not None

    def _now(self):
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 253402300799:
            raise ValueError('Recovery timestamps must be finite nonnegative numbers.')
        return value

    def _require_open(self):
        if not self.enabled:
            raise RecoveryUnavailable('Source recovery is not open.')

    def _acquire(self):
        _namespace(self.directory)
        _namespace(self.uploads_root)
        if os.name == 'nt':
            import ctypes
            if str(self.directory).startswith('\\\\') or ctypes.windll.kernel32.GetDriveTypeW(self.directory.anchor) == 4:
                raise RecoveryUnavailable('Recovery requires a local filesystem.')
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / 'owner.lock'
        if path.is_symlink():
            raise RecoveryUnavailable('Recovery lease cannot be a filesystem link.')
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            details = os.fstat(fd)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise RecoveryUnavailable('Recovery lease must be a regular file with one link.')
            os.set_inheritable(fd, False)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RecoveryLeaseError('Source recovery is already owned by another process or cannot be locked.') from exc
        except BaseException:
            os.close(fd)
            raise
        self._lease_fd = fd

    def _validate_record(self, record):
        if type(record) is not dict or set(record) != _FIELDS:
            raise ValueError('Unsupported recovery record fields.')
        _uuid(record['job_id'])
        _filename(record['filename'])
        if type(record['source_sha256']) is not str or re.fullmatch('[0-9a-f]{64}', record['source_sha256']) is None:
            raise ValueError('Recovery source digest must be lowercase SHA256.')
        _count(record['source_size_bytes'], 'Recovery source bytes', 8 * 1024**3)
        if not record['source_size_bytes']:
            raise ValueError('Recovery source cannot be empty.')
        if type(record['state']) is not str or record['state'] not in _STATES:
            raise ValueError('Unsupported recovery state.')
        _count(record['frames_processed'], 'Recovered frame count', 2**63 - 1)
        _count(record['previous_frames_processed'], 'Previous recovered frame count', 2**63 - 1)
        updated = record['updated_at']
        if type(updated) not in (int, float) or not math.isfinite(updated) or not 0 <= updated <= 253402300799:
            raise ValueError('Recovery timestamp is invalid.')
        # Preserve the exact admitted settings, while validating the complete
        # canonical representation and bounded serialized size.
        validate_settings(record['settings'])
        validate_output_template(record['output_name_template'])
        if record['claimed_by'] is not None:
            _uuid(record['claimed_by'])
            if record['claimed_by'] == record['job_id'] or record['previous_state'] not in _TERMINAL:
                raise ValueError('Recovery restart claim is invalid.')
            if record['state'] not in {'claimed', 'queued', 'processing'}:
                raise ValueError('A recovery claim must belong to an active restart.')
        elif (record['previous_state'] is not None or record['state'] == 'claimed'
              or record['previous_frames_processed'] != 0):
            raise ValueError('Recovery claim state is inconsistent.')
        encoded = json.dumps(record, allow_nan=False, ensure_ascii=False).encode('utf-8')
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError('Recovery record exceeds its metadata limit.')

    def _load(self):
        try:
            details = _regular(self.manifest_path)
        except FileNotFoundError:
            return {}
        if details.st_size > MAX_MANIFEST_BYTES:
            raise RecoveryUnavailable('Recovery manifest exceeds its metadata limit; preserve it for repair.')
        def reject_constant(_):
            raise ValueError('Nonfinite recovery JSON.')
        def unique_object(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate recovery JSON key.')
                result[key] = value
            return result
        data = json.loads(self.manifest_path.read_text(encoding='utf-8'), parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
        if (type(data) is not dict or set(data) != {'schema_version', 'records'}
                or type(data['schema_version']) is not int or data['schema_version'] != SCHEMA_VERSION
                or type(data['records']) is not list):
            raise ValueError('Unsupported recovery manifest schema.')
        result = {}
        for record in data['records']:
            self._validate_record(record)
            if record['job_id'] in result:
                raise ValueError('Duplicate recovery record identity.')
            result[record['job_id']] = record
        self._capacity(result)
        return result

    def _capacity(self, records):
        if len(records) > self.max_records:
            raise RecoveryCapacityError(f'Recovery retains at most {self.max_records} sources. Forget a retained source first.')
        if sum(row['source_size_bytes'] for row in records.values()) > self.max_bytes:
            raise RecoveryCapacityError(f'Recovery source quota is {self.max_bytes} bytes. Forget a retained source first.')

    def _persist(self, records):
        self._require_open()
        self._capacity(records)
        for record in records.values():
            self._validate_record(record)
        encoded = json.dumps({'schema_version': SCHEMA_VERSION, 'records': list(records.values())},
                             allow_nan=False, ensure_ascii=False, indent=2).encode('utf-8')
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise RecoveryCapacityError('Recovery metadata quota was reached.')
        temporary = self.directory / ('recovery.' + uuid.uuid4().hex + '.tmp')
        owned = False
        try:
            _namespace(self.directory)
            if self.manifest_path.exists() or self.manifest_path.is_symlink():
                _regular(self.manifest_path)
            with temporary.open('xb') as destination:
                owned = True
                destination.write(encoded)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.manifest_path)
            if os.name != 'nt':
                fd = os.open(self.directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except OSError as exc:
            raise RecoveryUnavailable('Recovery metadata could not be saved. Existing files were preserved.') from exc
        finally:
            if owned:
                temporary.unlink(missing_ok=True)
        self._records = records

    def open(self):
        with self._lock:
            if self.enabled:
                return self
            try:
                self._acquire()
                records = self._load()
                for record in records.values():
                    if record['state'] in {'queued', 'processing', 'claimed'}:
                        record.update(state='interrupted', claimed_by=None, previous_state=None,
                                      previous_frames_processed=0, updated_at=self._now())
                self._persist(records)
                return self
            except BaseException as exc:
                self.close()
                if isinstance(exc, RecoveryUnavailable) or not isinstance(exc, (OSError, ValueError, TypeError,
                                                                              KeyError, OverflowError, RecursionError)):
                    raise
                raise RecoveryUnavailable('Recovery metadata is unavailable or corrupt; preserve it and repair it explicitly.') from exc

    def close(self):
        with self._lock:
            if self._lease_fd is not None:
                fd, self._lease_fd = self._lease_fd, None
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            self._records = {}

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def _lookup(self, job_id):
        self._require_open()
        _uuid(job_id)
        if job_id not in self._records:
            raise RecoveryConflict('Retained source was not found.')
        return self._records[job_id]

    def _path(self, record):
        _namespace(self.uploads_root)
        path = self.uploads_root / (record['job_id'] + Path(record['filename']).suffix.lower())
        if path.parent.resolve() != self.uploads_root.resolve():
            raise RecoverySourceError('Retained source escaped its uploads directory.')
        return path

    def source_path(self, job_id):
        """Return the derived owned path without reading source bytes."""
        with self._lock:
            return self._path(self._lookup(job_id))

    def _public(self, record):
        result = copy.deepcopy(record)
        try:
            path = self._path(record)
            details = _regular(path)
            present = details.st_size == record['source_size_bytes']
        except (OSError, RecoveryUnavailable):
            present = False
        result.update(source_present=present, resume_mid_video=False, restart_mode='full_source')
        return result

    def list(self):
        with self._lock:
            self._require_open()
            return [self._public(record) for record in sorted(self._records.values(),
                    key=lambda row: (-row['updated_at'], row['job_id']))]

    def admit(self, job_id, filename, sha256, size, settings, template):
        with self._lock:
            self._require_open()
            record = {'job_id': job_id, 'filename': filename, 'source_sha256': sha256,
                      'source_size_bytes': size, 'settings': copy.deepcopy(settings),
                      'output_name_template': template, 'state': 'queued', 'frames_processed': 0,
                      'updated_at': self._now(), 'claimed_by': None, 'previous_state': None,
                      'previous_frames_processed': 0}
            self._validate_record(record)
            if job_id in self._records:
                raise RecoveryConflict('This source already has a recovery record.')
            self._capacity({**self._records, job_id: record})
            try:
                details = _regular(self._path(record))
                if details.st_size != size:
                    raise RecoverySourceError('Retained source size does not match admission.')
            except (OSError, RecoveryUnavailable) as exc:
                raise RecoverySourceError('Retained source is missing, linked, or changed.') from exc
            records = copy.deepcopy(self._records)
            records[job_id] = record
            self._persist(records)
            return self._public(record)

    def checkpoint(self, job_id, state, frames_processed=0, *, claim_id=None):
        with self._lock:
            record = self._lookup(job_id)
            if type(state) is not str or state not in _STATES - {'claimed'}:
                raise ValueError('Choose a supported recovery checkpoint state.')
            _count(frames_processed, 'Checkpoint frame count', 2**63 - 1)
            if record['claimed_by'] is not None and claim_id != record['claimed_by']:
                raise RecoveryConflict('Recovery restart is owned by another job.')
            if claim_id is not None and record['claimed_by'] != claim_id:
                raise RecoveryConflict('Recovery restart claim is no longer active.')
            records = copy.deepcopy(self._records)
            updated = records[job_id]
            updated.update(state=state, frames_processed=frames_processed, updated_at=self._now())
            if state in _TERMINAL:
                updated.update(claimed_by=None, previous_state=None, previous_frames_processed=0)
            self._persist(records)
            return self._public(updated)

    def get(self, job_id, cancel_event=None):
        """Validate source ownership, size, and exact bytes before explicit use."""
        with self._lock:
            record = self._lookup(job_id)
            try:
                path = self._path(record)
                if _regular(path).st_size != record['source_size_bytes']:
                    raise RecoverySourceError('Retained source size changed.')
                if hash_source(path, cancel_event=cancel_event) != record['source_sha256']:
                    raise RecoverySourceError('Retained source bytes changed; relink the exact original source.')
            except InterruptedError:
                raise
            except (OSError, ValueError, RecoveryUnavailable) as exc:
                raise RecoverySourceError('Retained source is missing, linked, or changed; relink the exact original source.') from exc
            return {**self._public(record), 'source_path': path}

    def claim(self, job_id, new_job_id, cancel_event=None):
        with self._lock:
            record = self._lookup(job_id)
            _uuid(new_job_id)
            if new_job_id == job_id or record['state'] not in _TERMINAL or record['claimed_by'] is not None:
                raise RecoveryConflict('Retained source is already active or claimed by a restart.')
            self.get(job_id, cancel_event=cancel_event)
            records = copy.deepcopy(self._records)
            records[job_id].update(state='claimed', claimed_by=new_job_id,
                                  previous_state=record['state'], previous_frames_processed=record['frames_processed'],
                                  updated_at=self._now())
            self._persist(records)
            return self._public(records[job_id])

    def release(self, job_id, new_job_id=None):
        with self._lock:
            record = self._lookup(job_id)
            if (record['claimed_by'] is None or record['state'] != 'claimed'
                    or new_job_id is not None and record['claimed_by'] != new_job_id):
                raise RecoveryConflict('Only the pending restart claim can be released.')
            records = copy.deepcopy(self._records)
            records[job_id].update(state=record['previous_state'], frames_processed=record['previous_frames_processed'],
                                  claimed_by=None, previous_state=None, previous_frames_processed=0, updated_at=self._now())
            self._persist(records)
            return self._public(records[job_id])

    def forget(self, job_id):
        """Remove metadata only; the application decides when source deletion is safe."""
        with self._lock:
            record = self._lookup(job_id)
            if record['claimed_by'] is not None:
                raise RecoveryConflict('An active restart owns this retained source.')
            result = self._public(record)
            records = copy.deepcopy(self._records)
            del records[job_id]
            self._persist(records)
            return result

    def status(self):
        with self._lock:
            return {'enabled': self.enabled, 'retained_jobs': len(self._records),
                    'retained_bytes': sum(row['source_size_bytes'] for row in self._records.values()),
                    'max_records': self.max_records, 'max_bytes': self.max_bytes,
                    'resume_mid_video': False, 'restart_mode': 'full_source'}
