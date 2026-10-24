import hashlib
import json
import os
import subprocess
import sys
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from recovery_store import (RecoveryStore, RecoveryUnavailable, RecoveryLeaseError,
                            RecoveryCapacityError, RecoveryConflict, RecoverySourceError)


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {'layout': 'SBS', 'strength': 5, 'convergence': .5, 'depth_mode': 'balanced',
            'input_size': 256, 'resolution': 'source', 'quality': 'fast', 'invert_depth': False,
            'temporal_smoothing': .35, 'edge_refinement': True, 'execution_mode': 'cpu'}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / '.runtime' / 'recovery-tests' / uuid.uuid4().hex
        self.uploads = self.root / 'uploads'
        self.uploads.mkdir(parents=True)
        self.directory = self.root / 'metadata'
        self.now = 1_800_000_000.0
        self.addCleanup(self.quarantine)

    def quarantine(self):
        destination = ROOT / '.runtime' / '_quarantine' / 'recovery-tests' / self.root.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(self.root.resolve().is_relative_to((ROOT / '.runtime' / 'recovery-tests').resolve()))
        os.replace(self.root, destination)

    def store(self, **options):
        store = RecoveryStore(self.directory, self.uploads, clock=lambda: self.now, **options)
        self.addCleanup(store.close)
        return store

    def source(self, contents=b'recover exactly these source bytes'):
        job_id = str(uuid.uuid4())
        path = self.uploads / (job_id + '.mp4')
        path.write_bytes(contents)
        return job_id, path, hashlib.sha256(contents).hexdigest(), len(contents)

    def admit(self, store, source=None, **changes):
        job_id, path, digest, size = source or self.source()
        args = {'job_id': job_id, 'filename': 'original.MP4', 'sha256': digest, 'size': size,
                'settings': dict(SETTINGS), 'template': '{name}_{layout}'}
        args.update(changes)
        return store.admit(**args), path

    def test_only_open_creates_store_and_admit_retains_exact_metadata(self):
        store = self.store()
        self.assertFalse(self.directory.exists())
        store.open()
        record, path = self.admit(store)
        self.assertEqual(record['state'], 'queued')
        self.assertEqual(len(store.list()), 1)
        retained = store.get(record['job_id'])
        self.assertEqual(retained['source_path'], path)
        self.assertEqual(retained['settings']['layout'], 'SBS')
        self.assertEqual(retained['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_restart_marks_active_jobs_interrupted_and_preserves_progress_source(self):
        first = self.store().open()
        record, path = self.admit(first)
        first.checkpoint(record['job_id'], 'processing', frames_processed=73)
        first.close()
        second = self.store().open()
        rows = second.list()
        self.assertEqual(rows[0]['state'], 'interrupted')
        self.assertEqual(rows[0]['frames_processed'], 73)
        self.assertTrue(path.exists())
        self.assertFalse(rows[0]['resume_mid_video'])

    def test_claim_rejects_duplicate_and_keeps_owner_through_processing(self):
        store = self.store().open()
        record, _ = self.admit(store)
        job_id, new_id = record['job_id'], str(uuid.uuid4())
        store.checkpoint(job_id, 'interrupted', frames_processed=8)
        store.claim(job_id, new_id)
        with self.assertRaises(RecoveryConflict):
            store.claim(job_id, str(uuid.uuid4()))
        store.checkpoint(job_id, 'processing', claim_id=new_id)
        self.assertEqual(store.list()[0]['claimed_by'], new_id)
        with self.assertRaises(RecoveryConflict):
            store.checkpoint(job_id, 'completed')
        store.checkpoint(job_id, 'completed', claim_id=new_id)
        self.assertIsNone(store.list()[0]['claimed_by'])

    def test_release_restores_previous_state_and_progress(self):
        store = self.store().open()
        record, _ = self.admit(store)
        job_id, new_id = record['job_id'], str(uuid.uuid4())
        store.checkpoint(job_id, 'interrupted', frames_processed=41)
        store.claim(job_id, new_id)
        store.release(job_id, new_id)
        self.assertEqual(store.list()[0]['state'], 'interrupted')
        self.assertEqual(store.list()[0]['frames_processed'], 41)

    def test_missing_changed_or_same_size_modified_source_rejected(self):
        for replacement in (None, b'wrong source', b'recover exactly these source byteX'):
            with self.subTest(replacement=replacement):
                store = self.store().open()
                record, path = self.admit(store)
                if replacement is None:
                    path.unlink()
                else:
                    path.write_bytes(replacement)
                with self.assertRaises(RecoverySourceError):
                    store.get(record['job_id'])
                self.assertEqual(len(store.list()), 1)
                store.close()
                # Separate metadata namespaces; never discard unavailable source records.
                self.directory = self.root / uuid.uuid4().hex

    def test_hashing_is_cancelable(self):
        store = self.store().open()
        record, _ = self.admit(store)
        event = threading.Event()
        event.set()
        with self.assertRaises(InterruptedError):
            store.get(record['job_id'], cancel_event=event)

    def test_forget_removes_metadata_only_and_releases_quota(self):
        store = self.store(max_records=1).open()
        record, path = self.admit(store)
        with self.assertRaises(RecoveryCapacityError):
            self.admit(store)
        store.forget(record['job_id'])
        self.assertTrue(path.exists())
        self.assertEqual(store.list(), [])
        self.admit(store)

    def test_byte_capacity_and_duplicate_admission_do_not_overwrite(self):
        store = self.store(max_bytes=40).open()
        source = self.source(b'x' * 30)
        self.admit(store, source)
        with self.assertRaises(RecoveryConflict):
            self.admit(store, source)
        with self.assertRaises(RecoveryCapacityError):
            self.admit(store, self.source(b'y' * 11))
        self.assertEqual(len(store.list()), 1)

    def test_corrupt_or_unknown_schema_preserved_and_lease_released(self):
        store = self.store().open()
        self.admit(store)
        store.close()
        manifest = self.directory / 'recovery.json'
        for contents in (b'{broken', b'{"schema_version":999,"records":[]}'):
            manifest.write_bytes(contents)
            with self.assertRaises(RecoveryUnavailable):
                self.store().open()
            self.assertEqual(manifest.read_bytes(), contents)

    def test_actual_cross_process_lease_exclusive_and_released(self):
        first = self.store().open()
        script = ('import sys; from recovery_store import RecoveryStore,RecoveryLeaseError; '
                  's=RecoveryStore(sys.argv[1],sys.argv[2]); '
                  '\ntry: s.open()\nexcept RecoveryLeaseError: sys.exit(23)\nelse: s.close()')
        args = [sys.executable, '-c', script, str(self.directory), str(self.uploads)]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 23, result.stderr)
        first.close()
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hardlinked_source_rejected(self):
        store = self.store().open()
        record, path = self.admit(store)
        os.link(path, self.root / 'alias.mp4')
        with self.assertRaises(RecoverySourceError):
            store.get(record['job_id'])

    def test_metadata_atomic_write_failure_preserves_prior_checkpoint(self):
        store = self.store().open()
        record, _ = self.admit(store)
        before = (self.directory / 'recovery.json').read_bytes()
        with patch('recovery_store.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(RecoveryUnavailable):
                store.checkpoint(record['job_id'], 'processing', frames_processed=2)
        self.assertEqual((self.directory / 'recovery.json').read_bytes(), before)
        self.assertEqual(store.list()[0]['state'], 'queued')

    def test_path_traversal_or_non_uuid_admission_rejected(self):
        store = self.store().open()
        for changes in ({'filename': '../secret.mp4'}, {'job_id': '../outside'},
                        {'filename': 'C:\\outside.mp4'}, {'size': True}, {'sha256': 'bad'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.admit(store, **changes)
        self.assertEqual(store.list(), [])

    def test_pending_claim_crash_preserves_progress_and_allows_new_explicit_claim(self):
        first = self.store().open()
        record, path = self.admit(first)
        job_id = record['job_id']
        first.checkpoint(job_id, 'interrupted', frames_processed=72)
        first.claim(job_id, str(uuid.uuid4()))
        first.close()
        second = self.store().open()
        retained = second.list()[0]
        self.assertEqual(retained['state'], 'interrupted')
        self.assertIsNone(retained['claimed_by'])
        self.assertEqual(retained['frames_processed'], 72)
        self.assertTrue(path.exists())
        second.claim(job_id, str(uuid.uuid4()))

    def test_same_size_edit_fails_digest_check(self):
        store = self.store().open()
        record, path = self.admit(store)
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assertEqual(path.stat().st_size, record['source_size_bytes'])
        with self.assertRaises(RecoverySourceError):
            store.get(record['job_id'])

    def test_symlinked_source_and_upload_root_rejected(self):
        store = self.store().open()
        record, path = self.admit(store)
        target = self.root / 'actual.mp4'
        os.replace(path, target)
        try:
            os.symlink(target, path)
        except OSError as exc:
            self.skipTest('Windows account cannot create symlinks: ' + str(exc))
        with self.assertRaises(RecoverySourceError):
            store.get(record['job_id'])
        alias = self.root / 'upload-alias'
        os.symlink(self.uploads, alias, target_is_directory=True)
        other = RecoveryStore(self.root / 'other-metadata', alias)
        self.addCleanup(other.close)
        with self.assertRaises(RecoveryUnavailable):
            other.open()

    def test_wrong_claim_owner_cannot_release_or_checkpoint(self):
        store = self.store().open()
        record, _ = self.admit(store)
        job_id, owner = record['job_id'], str(uuid.uuid4())
        store.checkpoint(job_id, 'interrupted')
        store.claim(job_id, owner)
        for action in (lambda: store.release(job_id, str(uuid.uuid4())),
                       lambda: store.checkpoint(job_id, 'queued', claim_id=str(uuid.uuid4()))):
            with self.assertRaises(RecoveryConflict):
                action()
        self.assertEqual(store.list()[0]['claimed_by'], owner)

    @unittest.skipUnless(os.name == 'nt', 'Windows junction ownership check')
    def test_windows_junction_upload_root_rejected_without_symlink_privilege(self):
        alias = self.root / 'uploads-junction'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(self.uploads)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        store = RecoveryStore(self.root / 'junction-metadata', alias)
        self.addCleanup(store.close)
        with self.assertRaises(RecoveryUnavailable):
            store.open()

    def test_duplicate_json_keys_and_malformed_record_preserve_manifest(self):
        store = self.store().open()
        self.admit(store)
        store.close()
        manifest = self.directory / 'recovery.json'
        valid = json.loads(manifest.read_text())
        valid['records'][0]['settings']['input_size'] = 123
        for contents in (b'{"schema_version":1,"schema_version":1,"records":[]}',
                         json.dumps(valid).encode()):
            manifest.write_bytes(contents)
            with self.assertRaises(RecoveryUnavailable):
                self.store().open()
            self.assertEqual(manifest.read_bytes(), contents)


if __name__ == '__main__':
    unittest.main()
