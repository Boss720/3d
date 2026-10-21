import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock
import uuid

import numpy as np

from depth_cache import CacheError, DepthCache


KEY = 'a' * 64
OTHER = 'b' * 64
DIGEST = hashlib.sha256(b'frame 1').hexdigest()
NEXT_DIGEST = hashlib.sha256(b'frame 2').hexdigest()


class DepthCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parent / ('cache-test-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.root = self.directory / 'cache'
        self.cache = DepthCache(self.root)

    def cleanup_directory(self):
        # Explicit test-owned files only: no recursive deletion or followed links.
        if self.root.exists():
            for path in self.root.iterdir():
                path.unlink()
            self.root.rmdir()
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def publish(self, key=KEY, values=None):
        values = np.arange(12, dtype=np.float32).reshape(3, 4) if values is None else values
        writer = self.cache.begin(key)
        writer.append(values, DIGEST)
        writer.finish({'execution': {'provider': 'cpu'}, 'stored_stage': 'native-temporal-v1'})
        return values

    def test_float32_bits_reopen_and_owned_arrays(self):
        bits = np.array([0, 0x80000000, 0x3f800001, 0x7fc12345], np.uint32)
        expected_bits = bits.copy()
        values = bits.view(np.float32).reshape(2, 2)
        self.publish(values=values)
        values[:] = 77
        cache = DepthCache(self.root)
        reader = cache.open_reader(KEY)
        self.addCleanup(reader.close)
        self.assertEqual(reader.frame_count, 1)
        self.assertEqual(reader.metadata['execution']['provider'], 'cpu')
        restored = reader.read(DIGEST)
        np.testing.assert_array_equal(restored.view(np.uint32).ravel(), expected_bits)
        self.assertTrue(restored.flags.owndata)
        restored[:] = 9
        reader.close()
        reopened = cache.open_reader(KEY)
        self.addCleanup(reopened.close)
        np.testing.assert_array_equal(reopened.read(DIGEST).view(np.uint32).ravel(), expected_bits)

    def test_sequential_frames_digest_mismatch_does_not_advance(self):
        writer = self.cache.begin(KEY)
        writer.append(np.ones((2, 3), np.float32), DIGEST)
        writer.append(np.full((1, 2), 2, np.float32), NEXT_DIGEST)
        self.assertIsNone(self.cache.open_reader(KEY))
        writer.finish({'fps': 24})
        reader = self.cache.open_reader(KEY)
        self.addCleanup(reader.close)
        with self.assertRaises(CacheError):
            reader.read(NEXT_DIGEST)
        self.assertEqual(reader.read(DIGEST).shape, (2, 3))
        np.testing.assert_array_equal(reader.read(NEXT_DIGEST), [[2, 2]])
        with self.assertRaises(CacheError):
            reader.read(NEXT_DIGEST)
        reader.close()
        reader.close()
        with self.assertRaises(CacheError):
            reader.read(DIGEST)

    def test_delete_refuses_readers_across_instances_and_writer(self):
        self.publish()
        other_cache = DepthCache(self.root)
        reader = other_cache.open_reader(KEY)
        self.addCleanup(reader.close)
        result = self.cache.delete([KEY])
        self.assertEqual(result['deleted'], [])
        self.assertEqual(result['errors'][0]['key'], KEY)
        self.assertTrue(self.cache.inventory()['entries'][0]['active'])
        with self.assertRaises(CacheError):
            self.cache.begin(KEY)
        reader.close()
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        self.assertTrue(self.cache.delete([KEY])['errors'])
        with self.assertRaises(CacheError):
            other_cache.begin(KEY)

    def test_abort_and_crash_partial_are_not_hits_and_can_be_deleted(self):
        writer = self.cache.begin(KEY)
        writer.append(np.ones((2, 2), np.float32), DIGEST)
        self.assertIsNone(self.cache.open_reader(KEY))
        writer.abort()
        writer.abort()
        self.assertEqual([p for p in self.root.iterdir() if p.suffix == '.part'], [])
        partial = self.root / (KEY + '.deadbeef.part')
        partial.write_bytes(b'crash')
        info = self.cache.inventory()
        self.assertEqual(info['bytes'], 5)
        self.assertEqual(info['entries'], [{'key': KEY, 'bytes': 5, 'frames': 0, 'active': False}])
        self.assertTrue(info['errors'])
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])
        self.assertFalse(partial.exists())

    def test_quota_includes_payload_manifest_and_other_writers(self):
        cache = DepthCache(self.root, quota_bytes=1000)
        first = cache.begin(KEY)
        self.addCleanup(first.abort)
        first.append(np.zeros((5, 5), np.float32), DIGEST)
        second = cache.begin(OTHER)
        self.addCleanup(second.abort)
        with self.assertRaises(CacheError):
            second.append(np.zeros((10, 10), np.float32), DIGEST)
        self.assertLessEqual(sum(p.stat().st_size for p in self.root.iterdir()), 1000)
        second.abort()
        first.finish({})
        self.assertLessEqual(cache.inventory()['bytes'], 1000)
        tiny = DepthCache(self.root, quota_bytes=10)
        with self.assertRaises(CacheError):
            tiny.begin(OTHER)

    def test_payload_corruption_and_truncated_entries_are_rejected(self):
        self.publish()
        path = self.root / (KEY + '.depthcache')
        original = path.read_bytes()
        damaged = bytearray(original)
        damaged[20] ^= 1
        path.write_bytes(damaged)
        with self.assertRaises(CacheError):
            self.cache.open_reader(KEY)
        self.assertTrue(self.cache.inventory()['errors'])
        # Recomputing the outer digest cannot hide corruption of an individual frame.
        damaged[-32:] = hashlib.sha256(damaged[:-48]).digest()
        path.write_bytes(damaged)
        with self.assertRaises(CacheError):
            self.cache.open_reader(KEY)
        for cut in (0, 8, len(original) - 1):
            path.write_bytes(original[:cut])
            with self.assertRaises(CacheError):
                self.cache.open_reader(KEY)
        # A failed read must release its lease so a new valid entry can replace it.
        self.publish()
        reader = self.cache.open_reader(KEY)
        self.addCleanup(reader.close)
        self.assertEqual(reader.frame_count, 1)

    def test_manifest_paths_shapes_offsets_and_lengths_are_untrusted(self):
        self.publish()
        path = self.root / (KEY + '.depthcache')
        original = path.read_bytes()
        manifest_length = struct.unpack('<Q', original[-40:-32])[0]
        manifest_start = len(original) - 48 - manifest_length
        manifest = json.loads(original[manifest_start:-48])
        mutations = [
            lambda m: m['frames'][0].update(shape=[999999999, 999999999]),
            lambda m: m['frames'][0].update(offset=-1),
            lambda m: m['frames'][0].update(bytes=0),
            lambda m: m.update(key='../escape'),
            lambda m: m.update(metadata=[]),
        ]
        for change in mutations:
            candidate = json.loads(json.dumps(manifest))
            change(candidate)
            encoded = json.dumps(candidate, separators=(',', ':')).encode()
            body = original[:manifest_start] + encoded
            path.write_bytes(body + original[-48:-40] + struct.pack('<Q', len(encoded)) + hashlib.sha256(body).digest())
            with self.assertRaises(CacheError):
                self.cache.open_reader(KEY)

    def test_invalid_keys_digests_and_arrays_do_not_escape_root(self):
        for key in ('../x', 'A' * 64, '', 'a' * 63, 'a' * 64 + '/x'):
            with self.assertRaises(CacheError):
                self.cache.begin(key)
            with self.assertRaises(CacheError):
                self.cache.open_reader(key)
            self.assertTrue(self.cache.delete([key])['errors'])
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        for array in (np.ones((1,), np.float32), np.ones((0, 2), np.float32), np.ones((2, 2), np.float64)):
            with self.assertRaises(CacheError):
                writer.append(array, DIGEST)
        with self.assertRaises(CacheError):
            writer.append(np.ones((2, 2), np.float32), '../digest')

    def test_publish_and_unlink_failures_release_leases(self):
        writer = self.cache.begin(KEY)
        writer.append(np.ones((2, 2), np.float32), DIGEST)
        with mock.patch('depth_cache.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(CacheError):
                writer.finish({})
        writer.abort()
        self.publish()
        with mock.patch('depth_cache.os.unlink', side_effect=PermissionError('locked')):
            result = self.cache.delete([KEY])
        self.assertEqual(result['deleted'], [])
        self.assertIn('locked', result['errors'][0]['error'])
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
    def test_actual_windows_open_file_blocks_selected_delete(self):
        self.publish()
        with (self.root / (KEY + '.depthcache')).open('rb'):
            result = self.cache.delete([KEY])
            self.assertEqual(result['deleted'], [])
            self.assertEqual(len(result['errors']), 1)
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])

    def test_footer_bounds_json_duplicates_and_oversized_shape_are_rejected(self):
        self.publish()
        path = self.root / (KEY + '.depthcache')
        original = path.read_bytes()
        length = struct.unpack('<Q', original[-40:-32])[0]
        offset = len(original) - 48 - length
        manifest = json.loads(original[offset:-48])
        manifest['frames'][0]['shape'] = [16385, 1]
        encoded = json.dumps(manifest).encode()
        duplicate = original[offset:-48].replace(b'"version":1', b'"version":1,"version":1')
        for encoded in (encoded, duplicate, b'{'):
            body = original[:offset] + encoded
            path.write_bytes(body + original[-48:-40] + struct.pack('<Q', len(encoded)) + hashlib.sha256(body).digest())
            with self.assertRaises(CacheError):
                self.cache.open_reader(KEY)
        path.write_bytes(original[:-40] + struct.pack('<Q', 2**63) + original[-32:])
        with self.assertRaises(CacheError):
            self.cache.open_reader(KEY)

    def test_empty_entry_and_non_json_metadata_abort_without_publication(self):
        writer = self.cache.begin(KEY)
        with self.assertRaises(CacheError):
            writer.finish({})
        self.assertIsNone(self.cache.open_reader(KEY))
        for metadata in ({'value': float('nan')}, {'object': object()}, []):
            writer = self.cache.begin(KEY)
            writer.append(np.zeros((2, 2), np.float32), DIGEST)
            with self.assertRaises(CacheError):
                writer.finish(metadata)
            self.assertIsNone(self.cache.open_reader(KEY))

    def test_hardlinks_are_refused(self):
        outside = self.directory / 'outside'
        outside.write_bytes(b'keep')
        hardlink = self.root / (KEY + '.depthcache')
        try:
            os.link(outside, hardlink)
        except OSError:
            self.skipTest('hardlinks unavailable')
        with self.assertRaises(CacheError):
            self.cache.open_reader(KEY)
        self.assertTrue(self.cache.delete([KEY])['errors'])
        self.assertEqual(outside.read_bytes(), b'keep')
        hardlink.unlink()

    def test_symlink_root_ancestor_and_entries_are_refused(self):
        link = self.directory / 'linked-root'
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest('Windows symlink creation privilege unavailable')
        with self.assertRaises(CacheError):
            DepthCache(link)
        with self.assertRaises(CacheError):
            DepthCache(link / 'child')
        outside = self.directory / 'outside'
        outside.write_bytes(b'keep')
        (self.root / (KEY + '.depthcache')).symlink_to(outside)
        with self.assertRaises(CacheError):
            self.cache.open_reader(KEY)
        self.assertTrue(self.cache.delete([KEY])['errors'])
        self.assertEqual(outside.read_bytes(), b'keep')

    def run_process(self, body):
        source = '\n'.join([
            'from pathlib import Path',
            'import os, sys',
            'import numpy as np',
            'from depth_cache import DepthCache, CacheError',
            'cache = DepthCache(Path(sys.argv[1]))',
            body,
        ])
        result = subprocess.run([sys.executable, '-c', source, str(self.root)],
                                cwd=Path(__file__).parent.parent, capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def test_separate_process_cannot_bypass_writer_or_reader_ownership(self):
        probe = '\n'.join([
            'try:',
            f'    writer = cache.begin({OTHER!r})',
            'except CacheError:',
            '    print("busy")',
            'else:',
            '    writer.abort()',
            '    print("available")',
        ])
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        writer.append(np.zeros((2, 2), np.float32), DIGEST)
        self.assertEqual(self.run_process(probe), 'busy')
        writer.finish({})
        reader = self.cache.open_reader(KEY)
        self.addCleanup(reader.close)
        self.assertEqual(self.run_process(probe), 'busy')
        reader.close()
        self.assertEqual(self.run_process(probe), 'available')

    def test_process_exit_releases_ownership_but_reports_partial(self):
        self.run_process('\n'.join([
            f'writer = cache.begin({KEY!r})',
            f'writer.append(np.zeros((2, 2), np.float32), {DIGEST!r})',
            'os._exit(0)',
        ]))
        info = self.cache.inventory()
        self.assertEqual(info['entries'][0]['key'], KEY)
        self.assertFalse(info['entries'][0]['active'])
        self.assertTrue(info['errors'])
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])
        writer = self.cache.begin(KEY)
        writer.abort()

    def test_metadata_quota_failure_preserves_existing_completed_entry(self):
        expected = self.publish()
        old_size = self.cache.inventory()['bytes']
        cache = DepthCache(self.root, quota_bytes=old_size + 450)
        writer = cache.begin(KEY)
        self.addCleanup(writer.abort)
        writer.append(np.ones((2, 2), np.float32), DIGEST)
        with self.assertRaises(CacheError):
            writer.finish({'too_large': 'x' * 1000})
        reader = self.cache.open_reader(KEY)
        self.addCleanup(reader.close)
        np.testing.assert_array_equal(reader.read(DIGEST), expected)

    def test_inventory_aggregates_same_key_committed_and_stale_partial(self):
        self.publish()
        original_size = self.cache.inventory()['bytes']
        (self.root / (KEY + '.1234.part')).write_bytes(b'stale')
        info = self.cache.inventory()
        self.assertEqual(len(info['entries']), 1)
        self.assertEqual(info['entries'][0]['bytes'], original_size + 5)
        self.assertEqual(info['entries'][0]['frames'], 1)
        self.assertEqual(self.cache.delete([KEY])['deleted'], [KEY])

    def test_abort_failure_does_not_release_a_later_writer_lease(self):
        writer = self.cache.begin(KEY)
        with mock.patch('depth_cache.os.unlink', side_effect=PermissionError('locked')):
            with self.assertRaises(CacheError):
                writer.abort()
        other = self.cache.begin(KEY)
        self.addCleanup(other.abort)
        writer.abort()
        self.assertTrue(self.cache.delete([KEY])['errors'])
        with self.assertRaises(CacheError):
            self.cache.begin(KEY)

    def test_abort_inspection_failure_still_closes_handle_and_releases_ownership(self):
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        handle = writer._handle
        with mock.patch.object(self.cache, '_check_root', side_effect=CacheError('directory unavailable')):
            with self.assertRaisesRegex(CacheError, 'directory unavailable'):
                writer.abort()
        self.assertTrue(handle.closed)
        self.assertEqual(self.cache._state.writers, {})
        self.assertIsNone(self.cache._state.os_lock)
        self.assertFalse(self.cache.inventory()['entries'][0]['active'])
        self.assertEqual(self.run_process(f'writer = cache.begin({KEY!r}); writer.abort(); print("released")'),
                         'released')

    def test_finish_operation_failure_aborts_and_releases_ownership(self):
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        writer.append(np.ones((2, 2), np.float32), DIGEST)
        handle = writer._handle
        with mock.patch.object(self.cache, '_check_root', side_effect=CacheError('directory unavailable')):
            with self.assertRaises(CacheError):
                writer.finish({})
        self.assertTrue(handle.closed)
        self.assertEqual(self.cache._state.writers, {})
        self.assertIsNone(self.cache._state.os_lock)
        self.assertIsNone(self.cache.open_reader(KEY))

    def test_append_failure_then_abort_while_root_unavailable_releases_ownership(self):
        writer = self.cache.begin(KEY)
        self.addCleanup(writer.abort)
        with mock.patch.object(self.cache, '_check_root', side_effect=CacheError('directory unavailable')):
            with self.assertRaises(CacheError):
                writer.append(np.ones((2, 2), np.float32), DIGEST)
            with self.assertRaises(CacheError):
                writer.abort()
        self.assertEqual(self.cache._state.writers, {})
        self.assertIsNone(self.cache._state.os_lock)
        replacement = self.cache.begin(KEY)
        replacement.abort()

    def test_writer_and_reader_close_failures_still_release_ownership(self):
        writer = self.cache.begin(KEY)
        real_handle = writer._handle
        self.addCleanup(real_handle.close)
        self.addCleanup(writer.abort)
        writer._handle = mock.Mock(wraps=real_handle)
        writer._handle.close.side_effect = PermissionError('close failed')
        with self.assertRaises(CacheError):
            writer.abort()
        self.assertEqual(self.cache._state.writers, {})
        self.assertIsNone(self.cache._state.os_lock)
        real_handle.close()
        writer.abort()
        self.publish()
        reader = self.cache.open_reader(KEY)
        real_handle = reader._handle
        self.addCleanup(real_handle.close)
        reader._handle = mock.Mock(wraps=real_handle)
        reader._handle.close.side_effect = PermissionError('close failed')
        with self.assertRaises(OSError):
            reader.close()
        self.assertEqual(self.cache._state.readers, {})
        self.assertIsNone(self.cache._state.os_lock)

    def test_abort_does_not_unlink_a_replaced_temporary_file(self):
        writer = self.cache.begin(KEY)
        writer._handle.close()
        os.replace(writer._path, self.directory / 'original-part')
        writer._path.write_bytes(b'foreign contents')
        with self.assertRaisesRegex(CacheError, 'replaced'):
            writer.abort()
        self.assertEqual(writer._path.read_bytes(), b'foreign contents')
        self.assertEqual(self.cache._state.writers, {})
        self.assertIsNone(self.cache._state.os_lock)


if __name__ == '__main__':
    unittest.main()
