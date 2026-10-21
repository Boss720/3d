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

    def test_removing_selected_queued_export_clears_preview_and_calibration(self):
        self.run_lifecycle("remove-selected-queued")

    def test_queue_poll_preserves_disclosure_and_keyboard_focus(self):
        self.run_lifecycle("queue-render-state")

    def test_named_favorite_presets_persist_apply_update_and_delete(self):
        self.run_lifecycle("named-presets")

    def test_queue_pause_controls_follow_confirmed_server_state(self):
        self.run_lifecycle("queue-controls")

    def test_background_terminal_jobs_are_announced_accessibly(self):
        self.run_lifecycle("queue-terminal-announcement")

    def test_storage_cleanup_requires_explicit_selection(self):
        self.run_lifecycle("storage-controls")

    def test_saved_project_round_trip_hash_relink_queue_order_and_preview_time(self):
        self.run_lifecycle("project-roundtrip")

    def test_invalid_project_does_not_partially_mutate_queue(self):
        self.run_lifecycle("project-invalid-atomic")

    def test_source_sha256_vectors_and_bounded_worker_streaming(self):
        result = subprocess.run(
            [NODE, str(ROOT / "tests" / "source_hash_test.cjs")],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
