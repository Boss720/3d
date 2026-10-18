"""Opt-in metadata history foundation. Application wiring is intentionally separate.

Only database/lease files are written. Sources are never opened, retained or
deleted; outputs are checked read-only. No executable jobs are reconstructed.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import os
import re
import sqlite3
import stat
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path

from job_settings import validate_settings
from output_naming import render_output_name, validate_output_template


SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 64 * 1024
_UNCHANGED = object()
_TERMINAL = {'completed', 'failed', 'canceled', 'interrupted'}
_STATES = _TERMINAL | {'queued', 'processing'}
_ADMISSION_FIELDS = {'job_id', 'created_at', 'filename', 'media_type', 'source_size_bytes',
                     'source_sha256', 'settings', 'output_name_template', 'build_fingerprint'}
_RECORD_FIELDS = _ADMISSION_FIELDS | {'schema_version', 'boot_id', 'updated_at', 'state', 'last_known_state',
                                    'phase', 'settings_schema_version', 'naming_schema_version',
                                    'planned_output_name', 'output', 'error', 'execution', 'encoder', 'timings'}


def _uuid(value):
    if type(value) is not str:
        raise ValueError('History identity must be a canonical UUID.')
    try:
        if str(uuid.UUID(value)) == value:
            return value
    except (ValueError, AttributeError):
        pass
    raise ValueError('History identity must be a canonical UUID.')


def _number(value):
    try:
        valid = type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 253402300799
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('History timestamps must be finite nonnegative numbers.')
    return value


def _text(value, limit, *, basename=False):
    if (type(value) is not str or not value or len(value) > limit
            or any(unicodedata.category(char) in {'Cc', 'Cf', 'Cs'} for char in value)):
        raise ValueError('History text is invalid or exceeds its bound.')
    if basename and (value in {'.', '..'} or '/' in value or '\\' in value or ':' in value):
        raise ValueError('History filenames must be basenames without path components.')
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('History fingerprints must be lowercase SHA256 values.')


def _finite_json(value, depth=0):
    if depth > 8:
        raise ValueError('History summary nesting exceeds its bound.')
    if value is None or type(value) is bool:
        return
    if type(value) in {int, float}:
        try:
            if math.isfinite(value):
                return
        except OverflowError:
            pass
    elif type(value) is str:
        if len(value) <= 4096 and not any(unicodedata.category(char) == 'Cs' for char in value):
            return
    elif type(value) is list and len(value) <= 64:
        for item in value:
            _finite_json(item, depth + 1)
        return
    elif type(value) is dict and len(value) <= 64:
        for key, item in value.items():
            _text(key, 64)
            _finite_json(item, depth + 1)
        return
    raise ValueError('History summaries require bounded finite JSON values.')


def _validate_admission(metadata):
    if type(metadata) is not dict or set(metadata) != _ADMISSION_FIELDS:
        raise ValueError('Provide exactly the documented history admission fields.')
    _uuid(metadata['job_id'])
    _number(metadata['created_at'])
    _text(metadata['filename'], 255, basename=True)
    if type(metadata['media_type']) is not str or metadata['media_type'] not in {'image', 'video'}:
        raise ValueError('History accepts ordinary image/video exports only.')
    size = metadata['source_size_bytes']
    if type(size) is not int or not 1 <= size <= 8 * 1024 ** 3:
        raise ValueError('Source byte count must be a positive supported integer.')
    _digest(metadata['source_sha256'])
    validate_settings(metadata['settings'])
    validate_output_template(metadata['output_name_template'])
    if metadata['build_fingerprint'] is not None:
        _digest(metadata['build_fingerprint'])


def _planned(record):
    return render_output_name(record['output_name_template'], record['filename'], record['settings']['layout'],
                              record['settings']['depth_mode'], record['created_at'], record['job_id'], record['media_type'])


def _validate_record(record):
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        raise ValueError('History record fields are invalid.')
    _validate_admission({key: record[key] for key in _ADMISSION_FIELDS})
    _uuid(record['boot_id'])
    _number(record['updated_at'])
    if any(type(record[key]) is not int or record[key] != 1 for key in
           ('schema_version', 'settings_schema_version', 'naming_schema_version')):
        raise ValueError('Unsupported history record schema.')
    if type(record['state']) is not str or record['state'] not in _STATES:
        raise ValueError('History state is invalid.')
    if record['state'] == 'interrupted':
        if record['last_known_state'] not in {'queued', 'processing'}:
            raise ValueError('Interrupted history requires its prior state.')
    elif record['last_known_state'] is not None:
        raise ValueError('Only interrupted history has a prior state.')
    _text(record['phase'], 64)
    if record['planned_output_name'] != _planned(record):
        raise ValueError('History planned filename does not match its immutable identity/settings.')
    if record['state'] == 'completed':
        output = record['output']
        if type(output) is not dict or set(output) != {'name', 'size_bytes'}:
            raise ValueError('Completed history requires an exact output reference.')
        _text(output['name'], 255, basename=True)
        if output['name'] != record['planned_output_name']:
            raise ValueError('History output does not match its full job identity.')
        if type(output['size_bytes']) is not int or not 1 <= output['size_bytes'] <= 2 ** 63 - 1:
            raise ValueError('History output size is invalid.')
    elif record['output'] is not None:
        raise ValueError('Uncompleted history cannot expose an output.')
    if record['error'] is not None:
        _text(record['error'], 4096)
    if record['encoder'] is not None:
        _text(record['encoder'], 256)
    for key in ('execution', 'timings'):
        if type(record[key]) is not dict:
            raise ValueError('History execution/timing summaries must be objects.')
        _finite_json(record[key])
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError('History record exceeds 64 KiB.')
    return encoded.decode('utf-8')

class HistoryError(RuntimeError):
    pass


class HistoryLeaseError(HistoryError):
    pass


class HistoryUnavailable(HistoryError):
    pass


class HistoryCapacityError(HistoryError):
    pass


class HistoryConflict(HistoryError):
    pass


class HistoryStore:
    def __init__(self, data_root, instance_name=None, *, max_records=1000,
                 max_age_seconds=90 * 86400, clock=None):
        if instance_name is not None and instance_name != '' and (type(instance_name) is not str or
                re.fullmatch('[a-z0-9][a-z0-9_-]{0,31}', instance_name) is None):
            raise ValueError('History instance must be a lowercase ASCII slug of 1 to 32 characters.')
        if type(max_records) is not int or not 1 <= max_records <= 1000:
            raise ValueError('History record limit must be from 1 to 1000.')
        _number(max_age_seconds)
        if max_age_seconds <= 0:
            raise ValueError('History age limit must be positive.')
        self.data_root = Path(data_root).resolve()
        self.instance_name = instance_name or None
        self.enabled = self.instance_name is not None
        self.directory = (self.data_root / '.runtime' / 'history' / f'instance-{self.instance_name}') if self.enabled else None
        self.database_path = self.directory / 'history.sqlite3' if self.enabled else None
        self.max_records, self.max_age_seconds = max_records, max_age_seconds
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._connection = None
        self._lease_fd = None
        self._degraded = False
        self.invalid_records = 0
        self.boot_id = None

    @classmethod
    def from_environment(cls, data_root, environ=None, **options):
        return cls(data_root, (os.environ if environ is None else environ).get('STEREOLAB_HISTORY_INSTANCE'), **options)

    def _now(self):
        return _number(self._clock())

    def _acquire_lease(self):
        if os.name == 'nt':
            import ctypes
            if str(self.data_root).startswith('\\\\') or ctypes.windll.kernel32.GetDriveTypeW(self.data_root.anchor) == 4:
                raise HistoryUnavailable('History requires a local filesystem.')
        candidate = self.data_root
        for component in ('.runtime', 'history', f'instance-{self.instance_name}'):
            candidate /= component
            if candidate.is_symlink() or candidate.resolve() != candidate.absolute():
                raise HistoryUnavailable('History namespace cannot traverse filesystem links.')
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / 'owner.lock'
        if lock_path.is_symlink():
            raise HistoryUnavailable('History lease must be a regular local file.')
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            details = os.fstat(fd)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise HistoryUnavailable('History lease must be a regular file with one filesystem link.')
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
            raise HistoryLeaseError('History instance is already in use or cannot be locked. Choose another instance.') from exc
        except BaseException:
            os.close(fd)
            raise
        self._lease_fd = fd

    def _check_existing(self):
        if self.database_path.is_symlink():
            raise HistoryUnavailable('History database cannot be a filesystem link.')
        connection = sqlite3.connect(self.database_path.as_uri() + '?mode=ro', uri=True, timeout=1.0)
        try:
            if connection.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
                raise HistoryUnavailable('History database is corrupt; preserve it and repair it explicitly.')
            if connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
                raise HistoryUnavailable('History schema is unsupported; the existing database was preserved.')
            expected = {'sessions': ['boot_id', 'started_at'],
                        'exports': ['job_id', 'boot_id', 'created_at', 'updated_at', 'state', 'record_json']}
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if tables != set(expected):
                raise HistoryUnavailable('History database tables are unsupported; preserve the existing database.')
            for table, columns in expected.items():
                details = [tuple(row)[1:] for row in connection.execute(f'PRAGMA table_info({table})')]
                expected_details = [(name, 'REAL' if name in {'created_at', 'updated_at', 'started_at'} else 'TEXT',
                                     0 if name == ('boot_id' if table == 'sessions' else 'job_id') else 1,
                                     None, 1 if name == ('boot_id' if table == 'sessions' else 'job_id') else 0)
                                    for name in columns]
                if details != expected_details:
                    raise HistoryUnavailable('History database columns are unsupported; preserve the existing database.')
            foreign_keys = [tuple(row) for row in connection.execute('PRAGMA foreign_key_list(exports)')]
            if foreign_keys != [(0, 0, 'sessions', 'boot_id', 'boot_id', 'NO ACTION', 'NO ACTION', 'NONE')]:
                raise HistoryUnavailable('History database ownership constraints are unsupported.')
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' LIMIT 1").fetchone():
                raise HistoryUnavailable('History database triggers are unsupported.')
        finally:
            connection.close()

    def open(self):
        with self._lock:
            if not self.enabled or self._connection is not None:
                return self
            try:
                self._acquire_lease()
                # lstat precedes exists/connect: a dangling link is still an
                # unowned destination, and hardlinks bypass namespace leases.
                if self.database_path.is_symlink():
                    raise HistoryUnavailable('History database cannot be a filesystem link.')
                try:
                    details = self.database_path.lstat()
                except FileNotFoundError:
                    existing = False
                else:
                    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                        raise HistoryUnavailable('History database must be a regular file with one filesystem link.')
                    existing = True
                if existing:
                    self._check_existing()
                self._connection = sqlite3.connect(self.database_path, timeout=1.0, isolation_level=None, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
                self._connection.execute('PRAGMA journal_mode=DELETE')
                self._connection.execute('PRAGMA synchronous=FULL')
                self._connection.execute('PRAGMA foreign_keys=ON')
                self._connection.execute('PRAGMA busy_timeout=1000')
                if not existing:
                    self._connection.executescript('BEGIN; CREATE TABLE sessions(boot_id TEXT PRIMARY KEY, started_at REAL NOT NULL); '
                        'CREATE TABLE exports(job_id TEXT PRIMARY KEY, boot_id TEXT NOT NULL REFERENCES sessions(boot_id), '
                        'created_at REAL NOT NULL, updated_at REAL NOT NULL, state TEXT NOT NULL, record_json TEXT NOT NULL); '
                        'PRAGMA user_version=1; COMMIT;')
                self.boot_id = str(uuid.uuid4())
                self._degraded = False
                with self._transaction() as connection:
                    connection.execute('INSERT INTO sessions VALUES (?,?)', (self.boot_id, self._now()))
                    for record in list(self._records()):
                        if record['state'] not in _TERMINAL:
                            record.update(last_known_state=record['state'], state='interrupted', updated_at=self._now())
                            self._save(connection, record)
                    self._prune(connection)
                return self
            except BaseException as exc:
                try:
                    self.close()
                except Exception:
                    pass  # Cleanup must preserve the triggering startup failure.
                if isinstance(exc, HistoryError) or not isinstance(exc, (OSError, sqlite3.Error)):
                    raise
                raise HistoryUnavailable('History could not open; preserve its files and check configuration/permissions.') from exc

    def close(self):
        with self._lock:
            try:
                if self._connection is not None:
                    self._connection.close()
            finally:
                self._connection = None
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

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def status(self):
        return {'schema_version': 1, 'supported': True, 'enabled': self.enabled, 'instance_name': self.instance_name,
                'state': 'disabled' if not self.enabled else ('degraded' if self._degraded or self._connection is None else 'ready'),
                'records_limit': self.max_records, 'source_retention': False, 'automatic_requeue': False,
                'frame_resume': False, 'invalid_records': self.invalid_records,
                'warning': 'History is unavailable or not fully saved. Account for current outputs before repairing/restarting.'
                           if self.enabled and (self._degraded or self._connection is None) else
                           ('Invalid history records were hidden and preserved for explicit repair.'
                            if self.invalid_records else None)}

    def _require(self, write=False):
        if self._connection is None or (write and self._degraded):
            raise HistoryUnavailable('History is disabled, closed or degraded; no durable change was accepted.')

    def mark_degraded(self):
        """Latch an application persistence failure without affecting live work."""
        with self._lock:
            self._degraded = True

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._require(write=True)
            try:
                self._connection.execute('BEGIN IMMEDIATE')
                yield self._connection
                self._connection.commit()
            except BaseException as exc:
                try:
                    self._connection.rollback()
                except sqlite3.Error:
                    pass
                if isinstance(exc, sqlite3.Error):
                    self._degraded = True
                    raise HistoryUnavailable('History write failed; no durable change was accepted.') from exc
                raise

    def _records(self):
        valid_sessions = set()
        for row in self._connection.execute('SELECT boot_id,started_at FROM sessions'):
            try:
                valid_sessions.add((_uuid(row[0]), _number(row[1]))[0])
            except (ValueError, TypeError):
                pass
        invalid = 0
        for row in self._connection.execute('SELECT * FROM exports ORDER BY created_at DESC,job_id DESC'):
            try:
                if type(row['record_json']) is not str or len(row['record_json'].encode('utf-8')) > MAX_RECORD_BYTES:
                    raise ValueError('Invalid history serialization.')
                record = json.loads(row['record_json'])
                _validate_record(record)
                if (record['boot_id'] not in valid_sessions or any(record[key] != row[key] for key in
                        ('job_id', 'boot_id', 'created_at', 'updated_at', 'state'))):
                    raise ValueError('History row identity differs from its serialized record.')
            except (ValueError, TypeError, OverflowError, KeyError, UnicodeError, RecursionError):
                invalid += 1
                continue
            yield record
        self.invalid_records = invalid

    def _save(self, connection, record):
        text = _validate_record(record)
        connection.execute('UPDATE exports SET boot_id=?,created_at=?,updated_at=?,state=?,record_json=? WHERE job_id=?',
                           (record['boot_id'], record['created_at'], record['updated_at'], record['state'], text, record['job_id']))

    def _prune(self, connection, reserve=0):
        terminal = sorted((record for record in self._records() if record['state'] in _TERMINAL),
                          key=lambda record: (record['updated_at'], record['job_id']))
        count = connection.execute('SELECT COUNT(*) FROM exports').fetchone()[0]
        for record in terminal:
            if record['updated_at'] < self._now() - self.max_age_seconds or count + reserve > self.max_records:
                connection.execute('DELETE FROM exports WHERE job_id=?', (record['job_id'],))
                count -= 1
        connection.execute('DELETE FROM sessions WHERE boot_id<>? AND boot_id NOT IN (SELECT boot_id FROM exports)', (self.boot_id,))
        if count + reserve > self.max_records:
            raise HistoryCapacityError('History has no room; unfinished or invalid records cannot be pruned automatically.')

    def _current(self, job_id, states):
        _uuid(job_id)
        for record in self._records():
            if record['job_id'] == job_id:
                if record['boot_id'] == self.boot_id and record['state'] in states:
                    return record
                break
        raise HistoryConflict('Only this boot\'s eligible export record may change.')

    def admit(self, metadata):
        self._require(write=True)
        _validate_admission(metadata)
        record = copy.deepcopy(metadata)
        record.update(schema_version=1, boot_id=self.boot_id, updated_at=self._now(), state='queued',
                      last_known_state=None, phase='queued', settings_schema_version=1, naming_schema_version=1,
                      planned_output_name=_planned(record), output=None, error=None, execution={}, encoder=None, timings={})
        text = _validate_record(record)
        with self._transaction() as connection:
            if connection.execute('SELECT 1 FROM exports WHERE job_id=?', (record['job_id'],)).fetchone():
                raise HistoryConflict('History already contains this full job identity.')
            self._prune(connection, reserve=1)
            connection.execute('INSERT INTO exports VALUES (?,?,?,?,?,?)',
                (record['job_id'], self.boot_id, record['created_at'], record['updated_at'], 'queued', text))
        return copy.deepcopy(record)

    def check_capacity(self):
        """Reject before upload staging; admission rechecks in its transaction."""
        with self._transaction() as connection:
            self._prune(connection, reserve=1)

    def edit_pending(self, job_id, *, settings=_UNCHANGED, output_name_template=_UNCHANGED):
        with self._transaction() as connection:
            record = self._current(job_id, {'queued'})
            if settings is _UNCHANGED and output_name_template is _UNCHANGED:
                raise ValueError('Provide settings or naming to replace.')
            if settings is not _UNCHANGED:
                record['settings'] = validate_settings(settings)
            if output_name_template is not _UNCHANGED:
                record['output_name_template'] = validate_output_template(output_name_template)
            record.update(updated_at=self._now(), planned_output_name=_planned(record))
            self._save(connection, record)
        return copy.deepcopy(record)

    def mark_processing(self, job_id, phase='initialization'):
        _text(phase, 64)
        with self._transaction() as connection:
            record = self._current(job_id, {'queued', 'processing'})
            record.update(state='processing', phase=phase, updated_at=self._now())
            self._save(connection, record)
        return copy.deepcopy(record)

    def finish(self, job_id, state, *, output=None, error=None, execution=None, encoder=None, timings=None):
        if type(state) is not str or state not in {'completed', 'failed', 'canceled'}:
            raise ValueError('Finish requires completed, failed or canceled state.')
        with self._transaction() as connection:
            record = self._current(job_id, {'processing'} if state == 'completed' else {'queued', 'processing'})
            record.update(state=state, phase='complete' if state == 'completed' else state, updated_at=self._now(),
                          output=copy.deepcopy(output), error=error, execution=copy.deepcopy({} if execution is None else execution),
                          encoder=encoder, timings=copy.deepcopy({} if timings is None else timings))
            _validate_record(record)
            if state == 'completed' and self._output_path(record) is None:
                raise ValueError('Completed history requires its existing regular, nonempty owned output.')
            self._save(connection, record)
            self._prune(connection)
        return copy.deepcopy(record)

    def _output_path(self, record):
        if record['state'] != 'completed':
            return None
        directory = self.data_root / 'outputs'
        path = directory / record['output']['name']
        try:
            if directory.resolve() != directory.absolute():
                return None
            details = path.lstat()
            if (stat.S_ISREG(details.st_mode) and not path.is_symlink()
                    and path.resolve().parent == directory.resolve()
                    and details.st_size == record['output']['size_bytes']):
                with path.open('rb'):
                    return path
        except (OSError, RuntimeError, ValueError):
            pass
        return None

    def _public(self, record):
        result = copy.deepcopy(record)
        result.pop('boot_id')
        output = result.pop('output')
        result.update(readonly=True, source_available=False, requires_source_reselection=True,
                      completed_at=record['updated_at'] if record['state'] == 'completed' else None,
                      output_name=output['name'] if output else None, output_available=self._output_path(record) is not None)
        return result

    def list_prior(self, limit=50, before=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('History page limit must be an integer from 1 to 100.')
        cursor = None
        if before is not None:
            try:
                if type(before) is not str or len(before) > 256 or re.fullmatch('[A-Za-z0-9_-]+', before) is None:
                    raise ValueError('Invalid history cursor.')
                cursor = json.loads(base64.urlsafe_b64decode(before + '=' * (-len(before) % 4)))
                if type(cursor) is not list or len(cursor) != 2:
                    raise ValueError('Invalid history cursor.')
                cursor = (_number(cursor[0]), _uuid(cursor[1]))
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ValueError('Invalid history cursor.') from exc
        with self._lock:
            self._require()
            try:
                records = [record for record in self._records() if record['boot_id'] != self.boot_id
                           and (cursor is None or (record['created_at'], record['job_id']) < cursor)]
                page = records[:limit]
                next_cursor = None
                if len(records) > limit:
                    last = page[-1]
                    next_cursor = base64.urlsafe_b64encode(json.dumps([last['created_at'], last['job_id']]).encode()).decode().rstrip('=')
                return {'schema_version': 1, 'items': [self._public(record) for record in page],
                        'next_cursor': next_cursor, 'invalid_records': self.invalid_records}
            except sqlite3.Error as exc:
                self._degraded = True
                raise HistoryUnavailable('History cannot be read; preserve its files for repair.') from exc

    def lookup_output(self, job_id):
        _uuid(job_id)
        with self._lock:
            self._require()
            try:
                records = list(self._records())
                for record in records:
                    if record['job_id'] == job_id and record['boot_id'] != self.boot_id:
                        return self._output_path(record)
                return None
            except sqlite3.Error as exc:
                self._degraded = True
                raise HistoryUnavailable('History cannot be read; preserve its files for repair.') from exc
