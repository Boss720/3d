"""Metadata-only storage checks. No app, model inference or server is started."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from job_history import HistoryCapacityError, HistoryConflict, HistoryLeaseError, HistoryStore, HistoryUnavailable


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {'layout': 'SBS', 'strength': 5, 'convergence': .5, 'depth_mode': 'balanced', 'input_size': 256,
            'resolution': 'source', 'quality': 'fast', 'invert_depth': False, 'temporal_smoothing': .35,
            'edge_refinement': True, 'execution_mode': 'cpu'}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / '.runtime' / 'job-history-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.quarantine_owned)
        self.now = 1_800_000_000.0

    def quarantine_owned(self):
        target = ROOT / '.runtime' / '_quarantine' / 'job-history-tests' / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(self.root.resolve().is_relative_to((ROOT / '.runtime' / 'job-history-tests').resolve()))
        self.assertTrue(target.resolve().is_relative_to((ROOT / '.runtime' / '_quarantine').resolve()))
        os.replace(self.root, target)

    def store(self, name='desktop', **options):
        store = HistoryStore(self.root, name, clock=lambda: self.now, **options)
        self.addCleanup(store.close)
        return store

    def metadata(self, **changes):
        record = {'job_id': str(uuid.uuid4()), 'created_at': self.now, 'filename': 'source.png',
                  'media_type': 'image', 'source_size_bytes': 12, 'source_sha256': hashlib.sha256(b'source').hexdigest(),
                  'settings': dict(SETTINGS), 'output_name_template': '{name}_{layout}', 'build_fingerprint': None}
        record.update(changes)
        return record

    def test_disabled_creates_no_files_and_cannot_record(self):
        absent = self.root / 'absent'
        with HistoryStore(absent) as store:
            self.assertFalse(store.status()['enabled'])
            with self.assertRaises(HistoryUnavailable):
                store.admit(self.metadata())
        self.assertFalse(absent.exists())

    def test_namespace_validation_rejects_paths_and_aliases_before_files(self):
        for name in ('../outside', 'Upper', '.', '-', 'a' * 33, r'a\b', 'a:b', ' name', 'name ', [], 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store(name).open()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_history_is_prior_boot_only_and_unfinished_is_interrupted_without_file_cleanup(self):
        metadata = self.metadata()
        source = self.root / 'untouched-source.png'
        source.write_bytes(b'owned-by-someone-else')
        first = self.store().open()
        first.admit(metadata)
        self.assertEqual(first.list_prior()['items'], [])
        first.close()
        with self.store() as second:
            items = second.list_prior()['items']
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]['state'], 'interrupted')
            self.assertEqual(items[0]['last_known_state'], 'queued')
            self.assertTrue(items[0]['requires_source_reselection'])
            self.assertFalse(items[0]['source_available'])
            self.assertNotIn('queue_position', items[0])
            self.assertEqual(items[0]['settings'], SETTINGS)
        self.assertEqual(source.read_bytes(), b'owned-by-someone-else')

    def test_invalid_admission_is_rejected_atomically(self):
        store = self.store().open()
        for changes in ({'job_id': 'not-uuid'}, {'created_at': float('nan')}, {'created_at': 1e300}, {'filename': '../secret.png'},
                        {'media_type': []},
                        {'source_size_bytes': True}, {'source_sha256': 'bad'},
                        {'settings': {**SETTINGS, 'extra': 1}}, {'output_name_template': '{name!r}'},
                        {'build_fingerprint': 'bad'}, {'extra': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                store.admit(self.metadata(**changes))
        store.close()
        with self.store() as restored:
            self.assertEqual(restored.list_prior()['items'], [])

    def output(self, record, content=b'finished-output'):
        directory = self.root / 'outputs'
        directory.mkdir(exist_ok=True)
        path = directory / record['planned_output_name']
        path.write_bytes(content)
        return path, {'name': path.name, 'size_bytes': len(content)}

    def test_pending_edits_are_atomic_and_cannot_modify_started_or_previous_boot_records(self):
        store = self.store().open()
        metadata = self.metadata()
        record = store.admit(metadata)
        with self.assertRaises(ValueError):
            store.edit_pending(record['job_id'], settings={**SETTINGS, 'strength': 9}, output_name_template='{name!r}')
        unchanged = store.edit_pending(record['job_id'], output_name_template='{name}_{layout}')
        self.assertEqual(unchanged['settings']['strength'], 5)
        edited = store.edit_pending(record['job_id'], settings={**SETTINGS, 'layout': 'TB'}, output_name_template='{date}_{layout}')
        self.assertEqual(edited['settings']['layout'], 'TB')
        self.assertNotEqual(edited['planned_output_name'], record['planned_output_name'])
        store.mark_processing(record['job_id'])
        with self.assertRaises(HistoryConflict):
            store.edit_pending(record['job_id'], settings=SETTINGS)
        store.close()
        with self.store() as restored:
            self.assertEqual(restored.list_prior()['items'][0]['settings']['layout'], 'TB')
            with self.assertRaises(HistoryConflict):
                restored.mark_processing(record['job_id'])

    def test_completed_output_lookup_is_confined_and_size_checked(self):
        store = self.store().open()
        record = store.admit(self.metadata())
        path, reference = self.output(record)
        with self.assertRaises(HistoryConflict):
            store.finish(record['job_id'], 'completed', output=reference)
        store.mark_processing(record['job_id'])
        store.finish(record['job_id'], 'completed', output=reference, execution={'provider': 'CPU'}, timings={'seconds': .1})
        self.assertIsNone(store.lookup_output(record['job_id']))  # Current boot stays in the ordinary UI.
        store.close()
        with self.store() as restored:
            self.assertEqual(restored.lookup_output(record['job_id']), path)
            self.assertTrue(restored.list_prior()['items'][0]['output_available'])
            path.write_bytes(b'changed-size')
            self.assertIsNone(restored.lookup_output(record['job_id']))
            self.assertEqual(restored.list_prior()['items'][0]['state'], 'completed')

    def test_record_count_and_age_pruning_delete_metadata_only(self):
        marker = self.root / 'never-delete-source'
        marker.write_bytes(b'source')
        store = self.store(max_records=2).open()
        a, b = store.admit(self.metadata()), store.admit(self.metadata())
        with self.assertRaises(HistoryCapacityError):
            store.admit(self.metadata())
        path, output = self.output(a)
        store.mark_processing(a['job_id'])
        store.finish(a['job_id'], 'completed', output=output)
        c = store.admit(self.metadata())
        self.assertTrue(path.is_file())
        store.close()
        with self.store(max_records=2) as restored:
            self.assertEqual({row['job_id'] for row in restored.list_prior()['items']}, {b['job_id'], c['job_id']})
            restored.admit(self.metadata())  # Old interrupted records can be pruned, never their files.
        self.now += 91 * 86400
        with self.store(max_records=2) as restored:
            # The previous boot's queued record has just become interrupted and is retained.
            self.assertLessEqual(len(restored.list_prior()['items']), 1)
        self.assertEqual(marker.read_bytes(), b'source')
        self.assertEqual(path.read_bytes(), b'finished-output')

    def test_runtime_sqlite_write_failure_latches_degraded_and_allows_prior_reads(self):
        first = self.store().open()
        first.admit(self.metadata())
        first.close()
        with self.store() as store:
            store._connection.execute('PRAGMA query_only=ON')
            with self.assertRaises(HistoryUnavailable):
                store.admit(self.metadata())
            self.assertEqual(store.status()['state'], 'degraded')
            self.assertEqual(len(store.list_prior()['items']), 1)
            with self.assertRaises(HistoryUnavailable):
                store.admit(self.metadata())

    def test_corrupt_and_future_schema_are_preserved_and_release_the_lease(self):
        store = self.store().open()
        database = store.database_path
        store.close()
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute('PRAGMA user_version=999')
        before = database.read_bytes()
        with self.assertRaises(HistoryUnavailable):
            self.store().open()
        self.assertEqual(database.read_bytes(), before)
        database.write_bytes(b'not-a-sqlite-database')
        with self.assertRaises(HistoryUnavailable):
            self.store().open()
        self.assertEqual(database.read_bytes(), b'not-a-sqlite-database')
        # A separate namespace never imports or overwrites the failed store.
        with self.store('other') as other:
            self.assertEqual(other.list_prior()['items'], [])

    def test_malicious_records_are_hidden_preserved_and_not_pruned(self):
        store = self.store().open()
        record = store.admit(self.metadata())
        store.mark_processing(record['job_id'])
        path, reference = self.output(record)
        store.finish(record['job_id'], 'completed', output=reference)
        database = store.database_path
        store.close()
        with closing(sqlite3.connect(database)) as connection, connection:
            serialized = json.loads(connection.execute('SELECT record_json FROM exports').fetchone()[0])
            serialized['output']['name'] = '../outside.png'
            connection.execute('UPDATE exports SET record_json=?', (json.dumps(serialized),))
        with self.store() as restored:
            self.assertEqual(restored.list_prior()['items'], [])
            self.assertEqual(restored.status()['invalid_records'], 1)
            self.assertIn('preserved', restored.status()['warning'])
            self.assertIsNone(restored.lookup_output(record['job_id']))
            self.assertEqual(restored._connection.execute('SELECT COUNT(*) FROM exports').fetchone()[0], 1)
        self.assertTrue(path.is_file())

    def test_finite_summary_and_size_validation_roll_back_terminal_update(self):
        store = self.store().open()
        record = store.admit(self.metadata())
        store.mark_processing(record['job_id'])
        for options in ({'execution': {'seconds': float('nan')}}, {'timings': []},
                        {'execution': {'value': '\ud800'}}, {'error': 'x' * 4097},
                        {'execution': {str(i): 'x' * 4096 for i in range(32)}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                store.finish(record['job_id'], 'failed', **options)
        store.close()
        with self.store() as restored:
            row = restored.list_prior()['items'][0]
            self.assertEqual(row['state'], 'interrupted')
            self.assertEqual(row['last_known_state'], 'processing')

    def test_prior_pages_are_stable_and_cursor_validation_is_strict(self):
        store = self.store().open()
        ids = [store.admit(self.metadata())['job_id'] for _ in range(3)]
        store.close()
        with self.store() as restored:
            first = restored.list_prior(limit=2)
            second = restored.list_prior(limit=2, before=first['next_cursor'])
            self.assertEqual([row['job_id'] for row in first['items'] + second['items']], sorted(ids, reverse=True))
            self.assertIsNone(second['next_cursor'])
            for options in ({'limit': True}, {'limit': 101}, {'before': '../path'}, {'before': 'x'}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    restored.list_prior(**options)

    def child(self, code, *arguments, timeout=8):
        return subprocess.run([sys.executable, '-B', '-c', code, str(self.root), *arguments],
                              cwd=ROOT, capture_output=True, text=True, timeout=timeout)

    def test_actual_two_process_lease_contention_different_namespace_and_owner_exit(self):
        store = self.store().open()
        self.assertFalse(os.get_inheritable(store._lease_fd))
        code = "from job_history import HistoryStore,HistoryLeaseError; import sys; " \
               "s=HistoryStore(sys.argv[1],sys.argv[2]);\ntry:\n s.open(); print('OPEN'); s.close()\nexcept HistoryLeaseError:\n print('BUSY')"
        blocked = self.child(code, 'desktop')
        self.assertEqual(blocked.returncode, 0, blocked.stderr)
        self.assertEqual(blocked.stdout.strip(), 'BUSY')
        other = self.child(code, 'second')
        self.assertEqual(other.returncode, 0, other.stderr)
        self.assertEqual(other.stdout.strip(), 'OPEN')
        lock = store.directory / 'owner.lock'
        store.close()
        lock.write_bytes(b'stale pid=123 boot=old; never authoritative')
        reopened = self.child(code, 'desktop')
        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        self.assertEqual(reopened.stdout.strip(), 'OPEN')
        self.assertEqual(lock.read_bytes(), b'stale pid=123 boot=old; never authoritative')

    def test_crashed_child_releases_lease_and_never_promotes_uncommitted_output(self):
        for completed in (False, True):
            name = 'committed' if completed else 'interrupted'
            metadata = self.metadata()
            code = "import json,os,sys; from pathlib import Path; from job_history import HistoryStore; " \
                   "s=HistoryStore(sys.argv[1],sys.argv[2]).open(); r=s.admit(json.loads(sys.argv[3])); " \
                   "s.mark_processing(r['job_id']); p=Path(sys.argv[1])/'outputs'/r['planned_output_name']; " \
                   "p.parent.mkdir(exist_ok=True); p.write_bytes(b'encoded-but-not-necessarily-committed');\n" \
                   "if sys.argv[4]=='yes': s.finish(r['job_id'],'completed',output={'name':p.name,'size_bytes':p.stat().st_size})\n" \
                   "os._exit(23)"
            result = self.child(code, name, json.dumps(metadata), 'yes' if completed else 'no')
            self.assertEqual(result.returncode, 23, result.stderr)
            with self.store(name) as restored:
                row = restored.list_prior()['items'][0]
                self.assertEqual(row['state'], 'completed' if completed else 'interrupted')
                self.assertEqual(restored.lookup_output(metadata['job_id']) is not None, completed)
                self.assertTrue((self.root / 'outputs' / row['planned_output_name']).is_file())

    def test_lease_descriptor_is_not_kept_alive_by_an_exec_child(self):
        store = self.store().open()
        child = subprocess.Popen([sys.executable, '-B', '-c', "import sys; print('READY',flush=True); sys.stdin.readline()"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, close_fds=False)
        ready, lines = threading.Event(), []
        def read_ready():
            lines.append(child.stdout.readline())
            ready.set()
        reader = threading.Thread(target=read_ready, daemon=True)
        reader.start()
        try:
            self.assertTrue(ready.wait(5))
            self.assertEqual(lines, ['READY\n'])
            store.close()
            with self.store() as reopened:
                self.assertEqual(reopened.status()['state'], 'ready')
        finally:
            child.communicate('\n', timeout=5)
            reader.join(timeout=1)

    def test_environment_opt_in_is_explicit_and_does_not_open_on_construction(self):
        off = HistoryStore.from_environment(self.root / 'off', {})
        on = HistoryStore.from_environment(self.root / 'on', {'STEREOLAB_HISTORY_INSTANCE': 'desktop'})
        self.assertFalse(off.status()['enabled'])
        self.assertTrue(on.status()['enabled'])
        self.assertEqual(list(self.root.iterdir()), [])
        with on:
            self.assertEqual(on.status()['instance_name'], 'desktop')

    def test_unknown_structural_schema_preserves_database_without_opening_writer(self):
        store = self.store().open()
        database = store.database_path
        store.close()
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute('ALTER TABLE exports ADD COLUMN surprise TEXT')
        original = database.read_bytes()
        with self.assertRaises(HistoryUnavailable):
            self.store().open()
        self.assertEqual(database.read_bytes(), original)

    def test_untrusted_serialized_records_cannot_break_reads_or_expose_paths(self):
        first = self.store().open()
        record = first.admit(self.metadata())
        first.close()
        with closing(sqlite3.connect(first.database_path)) as connection:
            original = json.loads(connection.execute('SELECT record_json FROM exports').fetchone()[0])
        variants = []
        for key, value in (('state', []), ('media_type', []), ('created_at', float('nan')),
                           ('filename', r'C:\secret\source.png'), ('source_sha256', 'bad'),
                           ('phase', '\ud800'), ('execution', {'seconds': float('inf')}),
                           ('settings', {**SETTINGS, 'edge_refinement': 'false'})):
            changed = dict(original)
            changed[key] = value
            variants.append(json.dumps(changed))
        variants += [b'not-json', '{', 'x' * (64 * 1024 + 1)]
        for serialized in variants:
            with self.subTest(serialized=str(serialized)[:90]):
                with closing(sqlite3.connect(first.database_path)) as connection, connection:
                    connection.execute('UPDATE exports SET record_json=?', (serialized,))
                with self.store() as restored:
                    self.assertEqual(restored.list_prior()['items'], [])
                    self.assertEqual(restored.invalid_records, 1)
                    self.assertIsNone(restored.lookup_output(record['job_id']))
        with closing(sqlite3.connect(first.database_path)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM exports').fetchone()[0], 1)

    def test_link_or_unreadable_output_is_unavailable_without_mutating_it(self):
        first = self.store().open()
        record = first.admit(self.metadata())
        path, reference = self.output(record)
        first.mark_processing(record['job_id'])
        first.finish(record['job_id'], 'completed', output=reference)
        first.close()
        with self.store() as restored:
            is_link = Path.is_symlink
            def linked(candidate):
                return candidate == path or is_link(candidate)
            with patch.object(Path, 'is_symlink', linked):
                self.assertIsNone(restored.lookup_output(record['job_id']))
            original_open = Path.open
            def inaccessible(candidate, *args, **kwargs):
                if candidate == path:
                    raise PermissionError('unreadable')
                return original_open(candidate, *args, **kwargs)
            with patch.object(Path, 'open', inaccessible):
                self.assertIsNone(restored.lookup_output(record['job_id']))
        self.assertEqual(path.read_bytes(), b'finished-output')

    def test_redirected_namespace_is_rejected_before_store_creation(self):
        store = self.store()
        original = Path.resolve
        def redirected(path, *args, **kwargs):
            if path == self.root / '.runtime':
                return self.root.parent / 'some-other-owner'
            return original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', redirected), self.assertRaises(HistoryUnavailable):
            store.open()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_all_startup_failures_release_lease_and_preserve_original_exception(self):
        for failure in (ValueError('invalid clock'), RuntimeError('clock callback failed'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                def bad_clock():
                    raise failure
                broken = HistoryStore(self.root, 'desktop', clock=bad_clock)
                self.addCleanup(broken.close)
                with self.assertRaises(type(failure)):
                    broken.open()
                with self.store() as recovered:
                    self.assertEqual(recovered.status()['state'], 'ready')

    def test_hardlinked_database_or_lease_cannot_cross_instance_ownership(self):
        live = self.store().open()
        record = live.admit(self.metadata())
        for filename in ('history.sqlite3', 'owner.lock'):
            other = self.store('linked-' + filename.replace('.', '-'))
            other.directory.mkdir()
            os.link(live.directory / filename, other.directory / filename)
            with self.subTest(filename=filename), self.assertRaises(HistoryUnavailable):
                other.open()
            saved = json.loads(live._connection.execute('SELECT record_json FROM exports').fetchone()[0])
            self.assertEqual(saved['state'], 'queued')
            self.assertEqual(saved['boot_id'], record['boot_id'])
        self.assertEqual(live._connection.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 1)

    def test_dangling_database_link_rejected_before_any_sqlite_connection(self):
        store = self.store()
        store.directory.mkdir(parents=True)
        original = Path.is_symlink
        def dangling(path):
            return path == store.database_path or original(path)
        with patch.object(Path, 'is_symlink', dangling), patch('job_history.sqlite3.connect') as connect:
            with self.assertRaises(HistoryUnavailable):
                store.open()
            connect.assert_not_called()
        with self.store() as recovered:
            self.assertEqual(recovered.status()['state'], 'ready')

    def test_real_dangling_database_symlink_when_windows_privilege_allows(self):
        store = self.store()
        store.directory.mkdir(parents=True)
        outside = self.root / 'must-not-be-created.sqlite3'
        try:
            os.symlink(outside, store.database_path)
        except OSError as exc:
            self.skipTest(f'Symlink creation unavailable; mocked boundary tested separately: {exc.winerror if os.name == "nt" else exc.errno}')
        with patch('job_history.sqlite3.connect') as connect:
            with self.assertRaises(HistoryUnavailable):
                store.open()
            connect.assert_not_called()
        self.assertFalse(outside.exists())

    def test_database_directory_rejected_before_sqlite_connection(self):
        store = self.store()
        store.database_path.mkdir(parents=True)
        with patch('job_history.sqlite3.connect') as connect:
            with self.assertRaises(HistoryUnavailable):
                store.open()
            connect.assert_not_called()
