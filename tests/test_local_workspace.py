import os
from pathlib import Path
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import local_workspace as workspace


class LocalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1] / '.runtime' / 'workspace-tests' / uuid.uuid4().hex
        self.roots = {name: self.base / name for name in ('uploads', 'previews', 'models', 'outputs')}
        for root in self.roots.values():
            root.mkdir(parents=True)
        self.addCleanup(self.quarantine)

    def quarantine(self):
        destination = self.base.parents[1] / '_quarantine' / 'workspace-tests' / self.base.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.base, destination)

    def file(self, category, name='fixture.bin', data=b'abc'):
        path = self.roots[category] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_inventory_counts_categories_excludes_active_candidates(self):
        active = self.file('uploads')
        candidate = self.file('previews', 'nested/preview.bin', b'12345')
        self.file('models')
        self.file('outputs')
        result = workspace.storage_inventory(self.roots, [active])
        self.assertEqual(result['categories']['previews']['bytes'], 5)
        self.assertEqual(result['categories']['uploads']['files'], 1)
        self.assertEqual([item['path'] for item in result['candidates']], [str(candidate)])

    def test_cleanup_only_selected_inactive_regular_temporary_files(self):
        selected = self.file('uploads', 'selected')
        retained = self.file('uploads', 'retained')
        active = self.file('previews', 'active')
        output = self.file('outputs')
        model = self.file('models')
        outside = self.base / 'outside'
        outside.write_bytes(b'a')
        result = workspace.cleanup_temporary_files(
            [selected, active, output, model, outside, self.roots['uploads']], self.roots, [active])
        self.assertEqual(result['deleted'], [str(selected)])
        self.assertEqual(len(result['errors']), 5)
        for path in (retained, active, output, model, outside):
            self.assertTrue(path.exists())

    def test_lexical_parent_escape_rejected(self):
        output = self.file('outputs')
        escape = self.roots['uploads'] / '..' / 'outputs' / output.name
        self.assertTrue(workspace.cleanup_temporary_files([escape], self.roots)['errors'])
        self.assertTrue(output.exists())

    def test_overlapping_output_root_is_protected(self):
        nested = self.file('uploads', 'nested/output')
        roots = dict(self.roots, outputs=nested.parent)
        self.assertTrue(workspace.cleanup_temporary_files([nested], roots)['errors'])
        self.assertTrue(nested.exists())
        self.assertEqual(workspace.storage_inventory(roots)['candidates'], [])

    def test_windows_reparse_attribute_rejected_without_link_privilege(self):
        path = self.file('uploads')
        with patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400)):
            with self.assertRaisesRegex(ValueError, 'junctions'):
                workspace._plain_path(path)

    def test_symlink_directory_not_followed_or_deleted(self):
        output = self.file('outputs')
        link = self.roots['uploads'] / 'link'
        try:
            link.symlink_to(self.roots['outputs'], target_is_directory=True)
        except OSError:
            self.skipTest('Creating symbolic links requires Windows privileges.')
        inventory = workspace.storage_inventory(self.roots)
        self.assertEqual(inventory['categories']['uploads']['files'], 0)
        self.assertTrue(inventory['errors'])
        self.assertTrue(workspace.cleanup_temporary_files([link / output.name], self.roots)['errors'])
        self.assertTrue(output.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
    def test_locked_file_error_preserves_file_and_other_cleanup_continues(self):
        locked = self.file('uploads', 'locked')
        free = self.file('uploads', 'free')
        with locked.open('rb'):
            result = workspace.cleanup_temporary_files([locked, free], self.roots)
        self.assertEqual(result['deleted'], [str(free)])
        self.assertEqual(len(result['errors']), 1)
        self.assertTrue(locked.exists())

    def test_output_destination_requires_existing_writable_directory(self):
        folder = self.roots['outputs']
        self.assertEqual(workspace.validate_output_directory(folder), folder)
        self.assertEqual(list(folder.iterdir()), [])
        with self.assertRaises(ValueError):
            workspace.validate_output_directory(folder / 'missing')
        with patch.object(workspace.tempfile, 'TemporaryFile', side_effect=PermissionError('read-only')):
            with self.assertRaises(PermissionError):
                workspace.validate_output_directory(folder)

    @unittest.skipUnless(os.name == 'nt', 'Explorer supported only on Windows')
    def test_desktop_actions_use_argument_lists_and_validate_target(self):
        output = self.file('outputs', 'file with spaces & symbols.mp4')
        with patch.object(workspace.subprocess, 'Popen') as popen:
            workspace.open_folder(output.parent)
            self.assertEqual(popen.call_args.args[0], ['explorer.exe', str(output.parent)])
            workspace.reveal_file(output)
            self.assertEqual(popen.call_args.args[0], ['explorer.exe', '/select,', str(output)])
            self.assertFalse(popen.call_args.kwargs['shell'])
            with self.assertRaises(ValueError):
                workspace.reveal_file(output.parent / 'missing')
            self.assertEqual(popen.call_count, 2)

    def test_linux_folder_opener_uses_xdg_open_without_shell(self):
        folder = self.roots['outputs']
        with patch.object(workspace.sys, 'platform', 'linux'), \
                patch.dict(workspace.os.environ, {'DISPLAY': ':1'}, clear=True), \
                patch.object(workspace.shutil, 'which', return_value='/usr/bin/xdg-open'), \
                patch.object(workspace.subprocess, 'Popen') as popen:
            workspace._open_linux_location(folder)
        popen.assert_called_once_with(['/usr/bin/xdg-open', str(folder)], shell=False)

    def test_linux_folder_opener_supports_gio_fallback(self):
        folder = self.roots['outputs']
        with patch.object(workspace.sys, 'platform', 'linux'), \
                patch.dict(workspace.os.environ, {'WAYLAND_DISPLAY': 'wayland-0'}, clear=True), \
                patch.object(workspace.shutil, 'which', side_effect=lambda name: '/usr/bin/gio' if name == 'gio' else None), \
                patch.object(workspace.subprocess, 'Popen') as popen:
            workspace._open_linux_location(folder)
        popen.assert_called_once_with(['/usr/bin/gio', 'open', str(folder)], shell=False)

    def test_linux_folder_opener_reports_headless_or_missing_helper(self):
        folder = self.roots['outputs']
        with patch.object(workspace.sys, 'platform', 'linux'), \
                patch.dict(workspace.os.environ, {}, clear=True), \
                patch.object(workspace.shutil, 'which') as which:
            with self.assertRaisesRegex(RuntimeError, 'no graphical desktop session'):
                workspace._open_linux_location(folder)
            which.assert_not_called()
        with patch.object(workspace.sys, 'platform', 'linux'), \
                patch.dict(workspace.os.environ, {'DISPLAY': ':1'}, clear=True), \
                patch.object(workspace.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'install xdg-open or gio'):
                workspace._open_linux_location(folder)


if __name__ == '__main__':
    unittest.main()
