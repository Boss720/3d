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

    def test_focus_and_visibility_return_refresh_idle_queue(self):
        self.run_lifecycle("queue-return")

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

    def test_linux_runtime_hides_unsupported_hardware_and_normalizes_presets(self):
        runner = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const file = path.resolve(process.argv[1]);
const scenario = process.argv[2];
let source = fs.readFileSync(file, 'utf8');
source = source.replace('ui, generatePreview, removeItem,', 'ui, generatePreview, loadStatus, loadNamedPresets, importPresetFile, removeItem,');
const anchor = '  } else throw new Error(`Unknown scenario: ${scenario}`);';
assert.ok(source.includes(anchor), 'The lifecycle harness scenario hook is available');
const scenarioCode = `  } else if (scenario === 'linux-capabilities') {
    const options = (document, values) => values.map(value => { const option = document.createElement('option'); option.value = value; return option; });
    app.ui.executionMode.options = options(f.document, ['auto', 'hybrid', 'gpu', 'cpu']);
    app.ui.videoDecode.options = options(f.document, ['software', 'amd']);
    const legacy = app.currentSettingsForPreset(); legacy.execution_mode = 'gpu'; legacy.video_decode = 'amd';
    f.storage.set('stereolab.namedPresets.v1', JSON.stringify({schema_version:1,presets:[{name:'Legacy',favorite:false,settings:legacy}]}));
    app.loadNamedPresets();
    f.network(async url => url === '/api/status' ? response({ffmpeg_found:true,runtime_info:{available:true,gpu_available:false,supported_execution_modes:['auto','cpu']}}) : response(url === '/api/models' ? {models:[]} : {}));
    await app.loadStatus();
    assert.equal(app.ui.executionMode.options.find(option => option.value === 'gpu').hidden, true);
    assert.equal(app.ui.executionMode.options.find(option => option.value === 'hybrid').disabled, true);
    assert.equal(app.ui.videoDecode.options.find(option => option.value === 'amd').hidden, true);
    assert.equal(app.ui.deviceDetails.hidden, true, 'Windows GPU help is hidden on a CPU-only Linux runtime');
    assert.match(app.ui.executionNote.textContent, /Linux CPU mode/);
    assert.doesNotMatch(app.ui.executionNote.textContent, /DirectML|PowerShell|setup-gpu|run\\.ps1/i);
    assert.match(app.ui.runtimeDescription.textContent, /CPU execution only/);
    assert.match(app.ui.toolPath.textContent, /ffmpeg available on PATH/);
    assert.doesNotMatch(app.ui.toolPath.textContent, /ffmpeg\\.exe/i);
    assert.match(app.ui.videoDecodeNote.textContent, /software video decoding/);
    const normalized = JSON.parse(f.storage.get('stereolab.namedPresets.v1')).presets[0].settings;
    assert.equal(normalized.execution_mode, 'auto', 'stored unsupported GPU mode normalizes to Auto');
    assert.equal(normalized.video_decode, 'software', 'stored D3D11VA mode normalizes to software decode');
    const imported = app.currentSettingsForPreset(); imported.execution_mode = 'gpu'; imported.video_decode = 'amd';
    await app.importPresetFile({text: async () => JSON.stringify({schema_version:1,settings:imported})});
    assert.equal(app.ui.executionMode.value, 'auto', 'unsupported imported GPU mode normalizes to Auto');
    assert.equal(app.ui.videoDecode.value, 'software', 'unsupported imported D3D11VA mode normalizes to software decode');

    const windows = fixture(), win = windows.app;
    win.ui.executionMode.options = options(windows.document, ['auto', 'hybrid', 'gpu', 'cpu']);
    win.ui.videoDecode.options = options(windows.document, ['software', 'amd']);
    windows.network(async url => url === '/api/status' ? response({ffmpeg_found:true,runtime_info:{available:true,gpu_available:true,supported_execution_modes:['auto','hybrid','gpu','cpu']}}) : response(url === '/api/models' ? {models:[]} : {}));
    await win.loadStatus();
    assert.ok(win.ui.executionMode.options.every(option => !option.hidden && !option.disabled), 'Windows execution choices remain available');
    const winSettings = win.currentSettingsForPreset(); winSettings.execution_mode = 'gpu'; winSettings.video_decode = 'amd';
    await win.importPresetFile({text: async () => JSON.stringify({schema_version:1,settings:winSettings})});
    assert.equal(win.ui.executionMode.value, 'gpu');
    assert.equal(win.ui.videoDecode.value, 'amd');
`;
source = source.replace(anchor, scenarioCode + anchor);
process.argv[2] = scenario;
const mod = new Module(file, module); mod.filename = file; mod.paths = Module._nodeModulePaths(path.dirname(file));
mod._compile(source, file);
'''
        result = subprocess.run(
            [NODE, "-e", runner, str(ROOT / "tests" / "frontend_lifecycle.cjs"), "linux-capabilities"],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
