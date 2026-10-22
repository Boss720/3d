import unittest
import os
from pathlib import Path, PurePosixPath

from platform_paths import resolve_paths


class PlatformPathTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "real POSIX filesystem paths are unavailable on Windows")
    def test_linux_paths_are_concrete_filesystem_paths(self):
        paths = resolve_paths(frozen=True, root=Path("/opt/stereolab"),
                             platform_name="linux", environ={}, home=Path("/home/tester"))

        self.assertIsInstance(paths.data_root, Path)
        self.assertTrue(callable(paths.data_root.mkdir))

    def test_frozen_linux_uses_xdg_roots_and_separates_durable_state_and_cache(self):
        paths = resolve_paths(
            frozen=True,
            root=Path("/opt/stereolab"),
            platform_name="linux",
            home=Path("/home/tester"),
            environ={
                "XDG_DATA_HOME": "/mnt/data",
                "XDG_STATE_HOME": "/mnt/state",
                "XDG_CACHE_HOME": "/mnt/cache",
            },
        )

        self.assertEqual(paths.data_root, PurePosixPath("/mnt/data/stereolab"))
        self.assertEqual(paths.models, PurePosixPath("/mnt/data/stereolab/models"))
        self.assertEqual(paths.outputs, PurePosixPath("/mnt/data/stereolab/outputs"))
        self.assertEqual(paths.state_root, PurePosixPath("/mnt/state/stereolab"))
        self.assertEqual(paths.history_root, paths.state_root)
        self.assertEqual(paths.calibration_file, paths.state_root / "calibrations.json")
        self.assertEqual(paths.cache_root, PurePosixPath("/mnt/cache/stereolab"))
        self.assertEqual(paths.uploads, paths.cache_root / "uploads")
        self.assertEqual(paths.clip_proxies, paths.cache_root / "clip-proxies")
        self.assertEqual(paths.depth_cache, paths.cache_root / "depth-cache")

    def test_frozen_linux_defaults_and_ignores_relative_xdg_overrides(self):
        paths = resolve_paths(
            frozen=True,
            root=Path("/opt/stereolab"),
            platform_name="linux",
            home=Path("/home/tester"),
            environ={
                "XDG_DATA_HOME": "relative/data",
                "XDG_STATE_HOME": "relative/state",
                "XDG_CACHE_HOME": "relative/cache",
            },
        )

        self.assertEqual(paths.data_root, PurePosixPath("/home/tester/.local/share/stereolab"))
        self.assertEqual(paths.state_root, PurePosixPath("/home/tester/.local/state/stereolab"))
        self.assertEqual(paths.cache_root, PurePosixPath("/home/tester/.cache/stereolab"))

    def test_source_run_keeps_project_layout_even_on_linux(self):
        root = Path("/workspace/stereolab")
        paths = resolve_paths(frozen=False, root=root, platform_name="linux",
                             home=Path("/home/tester"), environ={})

        self.assertEqual(paths.data_root, root)
        self.assertEqual(paths.runtime, root / ".runtime")
        self.assertEqual(paths.uploads, root / ".runtime/uploads")
        self.assertEqual(paths.models, root / ".runtime/models")
        self.assertEqual(paths.outputs, root / "outputs")
        self.assertEqual(paths.calibration_file, root / ".runtime/calibrations.json")
        self.assertEqual(paths.history_root, root)

    def test_frozen_windows_keeps_localappdata_layout(self):
        paths = resolve_paths(frozen=True, root=Path("C:/extract"), platform_name="win32",
                             environ={"LOCALAPPDATA": "C:/Users/test/AppData/Local"})

        data = Path("C:/Users/test/AppData/Local/StereoLab")
        self.assertEqual(paths.data_root, data)
        self.assertEqual(paths.runtime, data / ".runtime")
        self.assertEqual(paths.models, data / ".runtime/models")
        self.assertEqual(paths.outputs, data / "outputs")
        self.assertEqual(paths.history_root, data)


if __name__ == "__main__":
    unittest.main()
