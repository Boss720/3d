import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for frontend lifecycle tests")
class FrontendLifecycleTests(unittest.TestCase):
    def run_lifecycle(self, scenario):
        result = subprocess.run(
            [NODE, str(ROOT / "tests" / "frontend_lifecycle.cjs"), scenario],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_removing_source_discards_pending_preview_success(self):
        self.run_lifecycle("remove-success")

    def test_removing_source_discards_pending_preview_failure(self):
        self.run_lifecycle("remove-failure")

    def test_removed_source_invalidates_calibration_and_allows_other_sources(self):
        self.run_lifecycle("remove-calibration")

    def test_initial_restore_failure_retries_and_recovers_normal_polling(self):
        self.run_lifecycle("restore-recovery")

    def test_restore_retries_use_capped_backoff_and_one_inflight_request(self):
        self.run_lifecycle("restore-backoff")

    def test_manual_status_refresh_restores_queue_and_empty_copy(self):
        self.run_lifecycle("manual-refresh")

    def test_cleanup_warning_is_visible_without_changing_terminal_state(self):
        self.run_lifecycle("cleanup-warning")

    def test_queue_pause_controls_follow_confirmed_server_state(self):
        self.run_lifecycle("queue-controls")

    def test_storage_cleanup_requires_explicit_selection(self):
        self.run_lifecycle("storage-controls")
