import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for frontend contract tests")
class FrontendContractTests(unittest.TestCase):
    def run_contract(self, body):
        script = f"""
const assert = require('node:assert/strict');
const contract = require({json.dumps(str(ROOT / 'static' / 'app.js'))});
(async () => {{
{body}
}})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
"""
        result = subprocess.run(
            [NODE, "-e", script], cwd=ROOT, text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_capability_v1_bounds_known_and_unknown_gpu_memory(self):
        self.run_contract("""
const caps = {
  schema_version: 1, named_models_directml: true, gpu_max_input_dimension: 924,
  large_limited_gpu_max_input_dimension: 518, limited_gpu_memory_mb: 4096,
  unknown_memory_policy: 'limited', gpu_failure_cooldown_seconds: 60,
  gpu_retry_supported: true
};
const unknown = contract.gpuPolicy({ capabilities: caps }, { name: 'NVIDIA GeForce GTX 1650 Ti', memory_mb: null });
assert.equal(unknown.limited, true);
assert.equal(unknown.memoryUnknown, true);
assert.equal(unknown.inputCap, 518);
const unavailableMemory = contract.gpuPolicy({ capabilities: caps, gpu_available: true, gpu_name: 'NVIDIA GPU' }, null);
assert.equal(unavailableMemory.limited, true);
assert.equal(unavailableMemory.memoryUnknown, true);
assert.equal(unknown.supported, true);
const known = contract.gpuPolicy({ capabilities: caps }, { vendor_id: 4318, memory_mb: 4096 });
assert.equal(known.limited, true);
assert.equal(known.inputCap, 518);
const small = contract.gpuPolicy({ capabilities: caps }, { vendor_id: 4318, memory_mb: 4096 }, 'depth-anything-v2-small');
assert.equal(small.inputCap, 924);
const otherAdapter = contract.gpuPolicy({ capabilities: caps }, { name: 'Generic DirectML Adapter', vendor_id: 4098, memory_mb: 4096 });
assert.equal(otherAdapter.limited, true);
assert.equal(otherAdapter.inputCap, 518);
const amdRuntime = { capabilities: caps, gpu_available: true, gpu_device: { name: 'AMD Radeon', vendor_id: 4098, memory_mb: 8192 } };
assert.equal(contract.gpuPolicy(amdRuntime, contract.selectedGpuAdapter(amdRuntime)).inputCap, 924);
amdRuntime.gpu_device.memory_mb = 4096;
assert.equal(contract.gpuPolicy(amdRuntime, contract.selectedGpuAdapter(amdRuntime)).inputCap, 518);
const larger = contract.gpuPolicy({ capabilities: caps }, { vendor_id: 4318, memory_mb: 8192 });
assert.equal(larger.limited, false);
assert.equal(larger.inputCap, 924);
""")

    def test_old_backend_requires_upgrade_for_large_without_phrase_matching(self):
        self.run_contract("""
const old = contract.gpuPolicy({ gpu_memory_policy: 'all named models may try DirectML' }, { vendor_id: 4318, memory_mb: 4096 });
assert.equal(old.supported, false);
assert.equal(old.upgradeRequired, true);
assert.match(old.message, /upgrade|newer version/i);
assert.equal(old.inputCap, null);
""")

    def test_gpu_retry_posts_reset_then_refreshes_status(self):
        self.run_contract("""
const calls = [];
let refreshed = 0;
const request = async (url, options) => { calls.push([url, options]); return { reset: true }; };
await contract.retryGpu(request, async () => { refreshed += 1; });
assert.deepEqual(calls, [['/api/runtime/gpu/reset', { method: 'POST' }]]);
assert.equal(refreshed, 1);
await assert.rejects(contract.retryGpu(async () => ({ reset: false }), async () => { throw new Error('must not refresh'); }), /reset/);
""")

    def test_preview_execution_details_show_only_reported_dimensions_cap_and_cache(self):
        self.run_contract("""
const shown = contract.previewExecutionDetails({
  execution: { input_width: 518, input_height: 292, input_size_limit: 518 },
  cache: { hit: true }
});
assert.match(shown, /518 × 292/);
assert.match(shown, /cap 518 px/);
assert.match(shown, /depth cache hit/i);
assert.equal(contract.previewExecutionDetails({ execution: {} }), '');
""")

    def test_job_eta_reports_retry_finalization_and_observed_range(self):
        self.run_contract("""
const fmt = (seconds) => `${seconds}s`;
assert.equal(contract.jobEtaLabel({ state: 'processing', phase: 'retrying', retry_count: 1 }, fmt), 'Restarting export');
assert.equal(contract.jobEtaLabel({ state: 'processing', phase: 'finalization', eta_status: 'finalizing' }, fmt), 'Finalizing output…');
assert.equal(contract.jobEtaLabel({ state: 'processing', eta_status: 'available', eta_range_seconds: [4, 9] }, fmt), 'About 4s–9s remaining (approx.)');
assert.equal(contract.jobEtaLabel({ state: 'queued', eta_range_seconds: [4, 9] }, fmt), 'Waiting in queue');
""")

    def test_preflight_summary_labels_estimate_and_bounds_untrusted_copy(self):
        self.run_contract("""
const bytes = (value) => `${value} B`;
const details = contract.preflightDetails({
  output_width: 1920, output_height: 1080, estimated_output_bytes: 4500000,
  disk_free_bytes: 9000000, estimate_is_guarantee: false,
  estimate_basis: 'estimated from frame dimensions and codec assumptions',
  warnings: ['Free disk space is below the estimated output budget plus reserve. Free more space.', '<b>literal text</b>'.repeat(40), 'extra warning', 'discarded fourth warning']
}, bytes);
assert.match(details.summary, /Output 1920 × 1080 px/);
assert.ok(details.summary.includes('Estimated disk need 4500000 B (not guaranteed)'));
assert.match(details.summary, /Free when checked 9000000 B/);
assert.equal(details.warnings.length, 3);
assert.ok(details.warnings.every((warning) => warning.length <= 180));
assert.match(details.basis, /codec assumptions/);
assert.match(details.visibleWarning, /Free disk space is below/);
assert.ok(!details.summary.includes('Free disk space is below'));
const missing = contract.preflightDetails({ estimated_output_bytes: null, warnings: [] }, bytes);
assert.ok(missing.summary.includes('Disk estimate unavailable (not guaranteed)'));
""")

    def test_performance_presets_are_distinct_and_use_small_with_exact_input_sizes(self):
        self.run_contract("""
const expected = { fast: 256, balanced: 384, quality: 518, conservative4gb: 256 };
for (const [name, inputSize] of Object.entries(expected)) {
  const settings = contract.performancePreset(name);
  assert.equal(settings.depth_mode, 'depth-anything-v2-small');
  assert.equal(settings.input_size, inputSize);
  assert.ok(Number.isFinite(settings.strength));
  assert.ok(Number.isFinite(settings.temporal_smoothing));
  assert.equal(typeof settings.edge_refinement, 'boolean');
}
assert.equal(contract.performancePreset('fast').input_size, contract.performancePreset('conservative4gb').input_size);
assert.notDeepEqual(contract.performancePreset('fast'), contract.performancePreset('conservative4gb'));
assert.notDeepEqual(contract.performancePreset('balanced'), contract.performancePreset('quality'));
assert.equal(contract.performancePreset('missing'), null);
""")

    def test_user_settings_export_import_round_trip_and_strict_schema(self):
        self.run_contract("""
const settings = {
  layout: 'SBS', strength: 5.2, convergence: 0.48, depth_mode: 'depth-anything-v2-small',
  execution_mode: 'auto', input_size: 384, resolution: '1080p', quality: 'balanced',
  temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false
};
const exported = contract.exportUserPreset(settings);
assert.deepEqual(exported, { schema_version: 1, settings });
assert.deepEqual(contract.validateUserPreset(JSON.parse(JSON.stringify(exported))), settings);
assert.throws(() => contract.validateUserPreset({ ...exported, schema_version: 0 }), /schema_version/i);
assert.throws(() => contract.validateUserPreset({ ...exported, obsolete: true }), /unknown|field/i);
assert.throws(() => contract.validateUserPreset({ ...exported, settings: { ...settings, obsolete_key: 1 } }), /unknown|field/i);
assert.throws(() => contract.exportUserPreset({ ...settings, preview_detail: 'full' }), /unknown|field/i);
""")

    def test_user_settings_reject_invalid_bounds_nan_boolean_numeric_and_enum(self):
        self.run_contract("""
const base = contract.performancePreset('balanced');
for (const [key, value, pattern] of [
  ['strength', 10.1, /strength|range|bound/i],
  ['strength', NaN, /strength|finite|number/i],
  ['input_size', true, /input_size|number|type/i],
  ['convergence', false, /convergence|number|type/i],
  ['quality', 'turbo', /quality|enum|value/i],
  ['layout', 'SBS;alert(1)', /layout|enum|value/i],
  ['edge_refinement', 'true', /edge_refinement|boolean|type/i]
]) {
  assert.throws(() => contract.validateUserPreset({ schema_version: 1, settings: { ...base, [key]: value } }), pattern);
}
""")

    def test_dirty_state_is_recomputed_against_selected_baseline(self):
        self.run_contract("""
const baseline = contract.performancePreset('balanced');
assert.equal(contract.hasUnsavedSettings(baseline, { ...baseline }), false);
assert.equal(contract.hasUnsavedSettings(baseline, { ...baseline, strength: baseline.strength + 0.1 }), true);
assert.equal(contract.hasUnsavedSettings(baseline, { ...baseline, edge_refinement: false }), true);
assert.equal(contract.hasUnsavedSettings(baseline, { ...baseline, strength: baseline.strength + 1, strength2: 1 }), true);
""")


    def test_calibration_v1_requires_complete_shape_and_uses_measured_range(self):
        self.run_contract("""
const report = {
  schema_version: 1, budget_seconds: 15, elapsed_seconds: 4.7, total_elapsed_seconds: 17.2, budget_exceeded: true,
  sufficient_evidence: true, fastest_measured_input_size: 384,
  candidates: [256, 384, 518].map((input_size, i) => ({
    input_size, samples_seconds: [[0.4, 0.42], [0.3, 0.32], [0.5, 0.52]][i], warmup_seconds: [0.9],
    median_seconds: [0.41, 0.31, 0.51][i], range_seconds: [[0.4, 0.42], [0.3, 0.32], [0.5, 0.52]][i],
    execution: { active_provider: 'DmlExecutionProvider' }, error: null
  })),
  scope: 'Depth-input inference on one proxy frame.', limitations: ['One source only.'], saved: true
};
const normalized = contract.validateCalibrationReport(report);
assert.equal(contract.calibrationApplicable(normalized), true);
const view = contract.calibrationDetails(normalized);
assert.match(view.summary, /Total elapsed.*17[.]200 s/i);
assert.match(view.summary, /exceeded/i);
assert.match(view.summary, /Fastest measured depth input: 384 px/i);
assert.ok(view.candidates.every((line) => /median/.test(line) && /range/.test(line)));
assert.match(view.candidates[0], /warmup 1/);
assert.match(view.candidates[0], /DmlExecutionProvider/);
assert.match(view.scope, /one proxy frame/);
assert.match(contract.calibrationStatusLabel(normalized), /Measurement complete.*before applying/i);
""")

    def test_calibration_rejects_unsupported_schema_and_missing_candidates_for_apply(self):
        self.run_contract("""
const base = {
  schema_version: 1, budget_seconds: 5, elapsed_seconds: 3, budget_exceeded: false,
  sufficient_evidence: true, fastest_measured_input_size: 256,
  candidates: [256, 384, 518].map((input_size) => ({ input_size, samples_seconds: [1, 1.1], warmup_seconds: [], median_seconds: 1.05, range_seconds: [1, 1.1], execution: 'CPU', error: null })),
  scope: 'one proxy frame', limitations: [], saved: false
};
assert.throws(() => contract.validateCalibrationReport({ ...base, schema_version: 0 }), /schema_version/i);
assert.throws(() => contract.validateCalibrationReport({ ...base, total_elapsed_seconds: NaN }), /total_elapsed_seconds/i);
assert.throws(() => contract.validateCalibrationReport({ ...base, total_elapsed_seconds: true }), /total_elapsed_seconds/i);
const missing = contract.validateCalibrationReport({ ...base, candidates: base.candidates.slice(0, 2) });
assert.equal(contract.calibrationApplicable(missing), false);
assert.match(contract.calibrationDetails(missing).summary, /insufficient evidence/i);
assert.match(contract.calibrationDetails(missing).summary, /persistence time not reported/i);
assert.match(contract.calibrationStatusLabel(missing), /not enough evidence/i);
assert.equal(contract.calibrationApplicable(contract.validateCalibrationReport({ ...base, fastest_measured_input_size: 518 })), false);
""")

    def test_calibration_context_guard_rejects_source_frame_or_settings_changes(self):
        self.run_contract("""
const before = { revision: 4, sourceId: 'a', frameTime: 2.5, settings: 'small:auto:384', budget: 15, modelIdentity: 'sha256-a', previewDetail: 'proxy' };
assert.equal(contract.calibrationContextMatches(before, { ...before }), true);
for (const changed of [
  { ...before, revision: 5 }, { ...before, sourceId: 'b' },
  { ...before, frameTime: 2.6 }, { ...before, settings: 'small:auto:518' },
  { ...before, modelIdentity: 'sha256-b' },
  { ...before, budget: 30 }, { ...before, previewDetail: 'full' }
]) assert.equal(contract.calibrationContextMatches(before, changed), false);
""")

    def test_full_still_eligibility_enforces_decoded_pixel_and_dimension_limits(self):
        self.run_contract("""
assert.deepEqual(contract.fullStillEligibility({ mediaType: 'image', width: 4000, height: 3000 }), { eligible: true, reason: '' });
assert.equal(contract.fullStillEligibility({ mediaType: 'image', width: 4000, height: 3000, fileSizeBytes: 24 * 1024 * 1024 }).eligible, true);
assert.match(contract.fullStillEligibility({ mediaType: 'image', width: 4000, height: 3000, fileSizeBytes: 24 * 1024 * 1024 + 1 }).reason, /24 MiB.*Proxy/i);
assert.equal(contract.fullStillEligibility({ mediaType: 'image', width: 8192, height: 1 }).eligible, true);
assert.match(contract.fullStillEligibility({ mediaType: 'image', width: 8193, height: 1 }).reason, /8192/);
assert.match(contract.fullStillEligibility({ mediaType: 'image', width: 4001, height: 3000 }).reason, /12 MP/);
assert.match(contract.fullStillEligibility({ mediaType: 'video', width: 1920, height: 1080 }).reason, /still image/i);
assert.match(contract.fullStillEligibility({ mediaType: 'image', width: 0, height: 400 }).reason, /decod/i);
""")

    def test_preview_response_mode_validation_rejects_proxy_disguised_as_full(self):
        self.run_contract("""
const source = { image: 'data:image/png;base64,aa', depth: 'data:image/png;base64,bb', source_width: 4000, source_height: 3000, original_width: 4000, original_height: 3000 };
const full = contract.validatePreviewResponse({ ...source, preview_detail: 'full', is_proxy: false }, 'full');
assert.equal(full.preview_detail, 'full');
assert.match(contract.previewDimensionDetails({ ...full, selected_source_width: 4000, selected_source_height: 3000 }), /Full-detail still · processed 4000 × 3000 px · selected source 4000 × 3000 px/);
assert.throws(() => contract.validatePreviewResponse(source, 'full'), /preview_detail|upgrade|proxy/i);
assert.throws(() => contract.validatePreviewResponse({ ...source, preview_detail: 'proxy', is_proxy: true }, 'full'), /proxy|full/i);
assert.throws(() => contract.validatePreviewResponse({ ...source, preview_detail: 'full', is_proxy: true }, 'full'), /proxy|full/i);
assert.throws(() => contract.validatePreviewResponse({ ...source, preview_detail: 'full', is_proxy: false, source_width: 960 }, 'full'), /source|original|downscaled/i);
const proxy = contract.validatePreviewResponse({ ...source, source_width: 960, source_height: 540, preview_detail: 'proxy', is_proxy: true }, 'proxy');
assert.equal(proxy.preview_detail, 'proxy');
assert.match(contract.previewDimensionDetails({ ...proxy, selected_source_width: 4000, selected_source_height: 3000 }), /Proxy · processed 960 × 540 px · selected source 4000 × 3000 px/);
assert.match(contract.previewDimensionDetails(proxy), /Proxy · processed 960 × 540 px · decoded upload 4000 × 3000 px/);
assert.match(contract.previewDimensionDetails({ ...proxy, original_width: 960, original_height: 540 }), /decoded upload 960 × 540 px/);
""")

    def test_processed_clip_capability_is_required_and_bounded(self):
        self.run_contract("""
const supported = { preview_capabilities: { schema_version: 1, processed_clip: { supported: true, max_duration_seconds: 5, max_frames: 120, max_fps: 24, max_width: 960, max_height: 540, temporal_context: 'fresh' } } };
assert.deepEqual(contract.processedClipCapability(supported), supported.preview_capabilities.processed_clip);
assert.equal(contract.processedClipCapability({}), null);
assert.equal(contract.processedClipCapability({ preview_capabilities: { schema_version: 0, processed_clip: supported.preview_capabilities.processed_clip } }), null);
for (const bad of [
  { ...supported.preview_capabilities.processed_clip, max_duration_seconds: 6 },
  { ...supported.preview_capabilities.processed_clip, max_frames: 121 },
  { ...supported.preview_capabilities.processed_clip, max_fps: 25 },
  { ...supported.preview_capabilities.processed_clip, max_width: 961 },
  { ...supported.preview_capabilities.processed_clip, temporal_context: 'continuous' },
  { ...supported.preview_capabilities.processed_clip, max_height: '540' }
]) assert.equal(contract.processedClipCapability({ preview_capabilities: { schema_version: 1, processed_clip: bad } }), null);
assert.equal(contract.validProcessedClipDuration(1, supported), true);
assert.equal(contract.validProcessedClipDuration(5, supported), true);
for (const duration of [0, 1.5, 6, '3']) assert.equal(contract.validProcessedClipDuration(duration, supported), false);
""")

    def test_processed_clip_response_rejects_export_and_validates_interval(self):
        self.run_contract("""
const clip = { id: 'job-clip-a', purpose: 'preview_clip', state: 'processing', clip_start_seconds: 3.25, clip_duration_seconds: 5 };
assert.deepEqual(contract.validatePreviewClipJob(clip), clip);
assert.throws(() => contract.validatePreviewClipJob({ ...clip, purpose: 'export' }), /purpose|preview clip/i);
assert.throws(() => contract.validatePreviewClipJob({ ...clip, purpose: undefined }), /purpose|preview clip/i);
assert.throws(() => contract.validatePreviewClipJob({ ...clip, clip_start_seconds: NaN }), /start|interval|finite/i);
assert.throws(() => contract.validatePreviewClipJob({ ...clip, clip_duration_seconds: 5.1 }), /duration|interval|range/i);
assert.throws(() => contract.validatePreviewClipJob({ ...clip, state: 'completed' }), /proxy details/i);
assert.equal(contract.isOrdinaryExportJob({ id: 'old-job' }), true);
assert.equal(contract.isOrdinaryExportJob({ purpose: 'export' }), true);
assert.equal(contract.isOrdinaryExportJob({ purpose: 'preview_clip' }), false);
""")

    def test_processed_clip_context_and_status_report_stale_inputs_and_actual_proxy(self):
        self.run_contract("""
const before = { sourceId: 'video-a', settings: 'small:384:cpu', startSeconds: 2.5, durationSeconds: 3 };
assert.equal(contract.clipContextMatches(before, { ...before }), true);
for (const after of [{ ...before, sourceId: 'video-b' }, { ...before, settings: 'small:518:cpu' }, { ...before, startSeconds: 3 }, { ...before, durationSeconds: 5 }]) assert.equal(contract.clipContextMatches(before, after), false);
assert.equal(contract.processedClipResponseAction(before, { ...before }), 'display');
assert.equal(contract.processedClipResponseAction(before, { ...before, settings: 'small:518:cpu' }), 'cancel');
assert.equal(contract.processedClipResponseAction(before, { ...before, durationSeconds: 1 }), 'cancel');
assert.equal(contract.processedClipResponseAction(before, { ...before }, true), 'cancel');
const job = { id: 'job-clip-a', purpose: 'preview_clip', state: 'processing', clip_start_seconds: 2.5, clip_duration_seconds: 3, progress: 42, depth_mode: 'depth-anything-v2-small', execution_mode: 'auto', execution: { active_provider: 'CPUExecutionProvider' }, clip: { source_start_seconds: 2.5, requested_duration_seconds: 3, temporal_context: 'fresh', proxy: { width: 960, height: 540, fps: 24, total_frames: 72 } } };
const status = contract.previewClipDetails(job);
assert.match(status, /2[.]50.*5[.]50 s/);
assert.match(status, /960 × 540 px.*24 fps.*72 frames.*3[.]00 s video/);
assert.match(status, /fresh temporal history/i);
assert.match(status, /Converting.*42%/i);
assert.match(status, /model depth-anything-v2-small.*CPUExecutionProvider/i);
assert.equal(contract.isOrdinaryExportJob({ purpose: 'preview_clip' }), false);
assert.throws(() => contract.validatePreviewClipJob({ ...job, clip: { ...job.clip, temporal_context: 'continuous' } }), /fresh|bounded/i);
""")

    def test_processed_clip_capability_load_failure_has_actionable_retry_message(self):
        self.run_contract("""
assert.match(contract.processedClipUnavailableMessage(null, 'connection refused'), /connection refused/i);
assert.match(contract.processedClipUnavailableMessage(null, 'connection refused'), /Refresh status/i);
assert.match(contract.processedClipUnavailableMessage(null, ''), /checking/i);
assert.match(contract.processedClipUnavailableMessage({}, ''), /does not advertise/i);
""")

    def test_processed_clip_success_clears_only_transient_poll_error(self):
        self.run_contract("""
assert.equal(contract.processedClipErrorAfterPoll('Could not refresh processed clip: timeout', 'poll'), '');
assert.equal(contract.processedClipErrorAfterPoll('Could not cancel the processed clip: timeout', 'cancel'), 'Could not cancel the processed clip: timeout');
""")

    def test_processed_clip_quarantine_waits_for_matching_terminal_job(self):
        self.run_contract("""
const pending = { id: 'job-a', state: 'processing', message: 'bad response' };
assert.deepEqual(contract.processedClipQuarantineAfterPoll(pending, { id: 'other', state: 'canceled' }), pending);
assert.deepEqual(contract.processedClipQuarantineAfterPoll(pending, { id: 'job-a', state: 'queued' }), { ...pending, state: 'queued' });
assert.equal(contract.processedClipQuarantineAfterPoll(pending, { id: 'job-a', state: 'canceled' }), null);
assert.equal(contract.processedClipQuarantineAfterPoll(pending, { id: 'job-a', state: 'completed' }), null);
""")

    def test_queue_capability_and_public_snapshot_contract(self):
        self.run_contract("""
const caps = { queue_capabilities: { schema_version: 1, reorder_queued_exports: true, edit_queued_exports: true, remove_queued_exports: true, settings_snapshot: true } };
assert.deepEqual(contract.queueCapabilities(caps), caps.queue_capabilities);
assert.equal(contract.queueCapabilities({}), null);
assert.equal(contract.queueCapabilities({ queue_capabilities: { ...caps.queue_capabilities, settings_snapshot: false } }), null);
const settings = { layout: 'SBS', strength: 5, convergence: 0.5, depth_mode: 'depth-anything-v2-small', execution_mode: 'auto', input_size: 256, resolution: '1080p', quality: 'balanced', temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false };
const job = { settings_schema_version: 1, settings, source_available: true, queue_position: 2 };
assert.deepEqual(contract.validateQueueJobSnapshot(job), { settings, sourceAvailable: true, queuePosition: 2 });
assert.deepEqual(contract.validateQueueJobSnapshot({ ...job, queue_position: null }).queuePosition, null);
for (const bad of [{ ...job, settings_schema_version: 2 }, { ...job, source_available: 'yes' }, { ...job, queue_position: 0 }, { ...job, settings: { ...settings, unexpected: 1 } }]) assert.throws(() => contract.validateQueueJobSnapshot(bad));
""")

    def test_queue_order_helpers_preserve_ready_slots_and_clip_positions(self):
        self.run_contract("""
const local = [
  { uid: 'a', state: 'ready', file: {} },
  { uid: 'remote', state: 'queued', jobId: 'j' },
  { uid: 'b', state: 'ready', file: {} }
];
assert.deepEqual(contract.moveReadyQueueItem(local, 'b', -1).map(x => x.uid), ['b', 'remote', 'a']);
assert.deepEqual(contract.moveReadyQueueItem(local, 'a', -1).map(x => x.uid), ['a', 'remote', 'b']);
const jobs = [
  { id: 'a', purpose: 'export', state: 'queued', queue_position: 1 },
  { id: 'clip', purpose: 'preview_clip', state: 'queued', queue_position: 2 },
  { id: 'b', purpose: 'export', state: 'queued', queue_position: 3 },
  { id: 'active', purpose: 'export', state: 'processing', queue_position: null }
];
assert.deepEqual(contract.moveQueuedExportIds(jobs, 'b', -1), ['b', 'a']);
assert.deepEqual(contract.moveQueuedExportIds(jobs, 'a', -1), ['a', 'b']);
""")

    def test_output_naming_capability_and_template_parser_are_strict(self):
        self.run_contract("""
const status = { naming_capabilities: { schema_version: 1, supported: true, max_template_length: 120, fields: ['name', 'layout', 'model', 'date'], unique_job_id_suffix: true } };
assert.deepEqual(contract.outputNamingCapability(status), status.naming_capabilities);
assert.equal(contract.outputNamingCapability({}), null);
for (const bad of [
  { ...status.naming_capabilities, schema_version: 2 },
  { ...status.naming_capabilities, supported: false },
  { ...status.naming_capabilities, max_template_length: 121 },
  { ...status.naming_capabilities, fields: ['name', 'layout', 'model'] },
  { ...status.naming_capabilities, unique_job_id_suffix: false }
]) assert.equal(contract.outputNamingCapability({ naming_capabilities: bad }), null);
assert.equal(contract.validateOutputNameTemplate('{name}_{layout}-{model}.{date}'), '{name}_{layout}-{model}.{date}');
assert.equal(contract.validateOutputNameTemplate('東京 scene {layout}'), '東京 scene {layout}');
assert.equal(contract.validateOutputNameTemplate('scene-😀_{layout}'), 'scene-😀_{layout}');
assert.equal(contract.validateOutputNameTemplate('😀'.repeat(61) + '{name}'), '😀'.repeat(61) + '{name}');
for (const value of ['', '   ', 'x'.repeat(121), '😀'.repeat(121), '{unknown}', '{name', 'name}', '{name!r}', '{name.attr}', '{name:>20}', 'a/b', 'a\\\\b', 'a:b', 'a*', 'a?', '\\uD800', '\\uDC00', 'a\\u0085b', 'a\\u200Bb', 'a\\u202Eb']) {
  assert.throws(() => contract.validateOutputNameTemplate(value), /template|token|character|120|empty/i, `accepted ${value}`);
}
for (const value of ['a\\u0085b', 'a\\u200Bb', 'a\\u202Eb', 'a\\uD800b']) assert.throws(() => contract.validateOutputBasename(value, 'planned output name'), /unsafe/i);
""")

    def test_output_naming_job_snapshot_and_ready_item_freeze_retry_duplicate(self):
        self.run_contract("""
const settings = { layout: 'SBS', strength: 5, convergence: 0.5, depth_mode: 'depth-anything-v2-small', execution_mode: 'auto', input_size: 256, resolution: '1080p', quality: 'balanced', temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false };
const job = { naming_schema_version: 1, output_name_template: '{name}_{layout}_{date}', planned_output_name: 'scene_sbs_20261018_12345678-1234-1234-1234-123456789abc.png', output_name: null };
assert.deepEqual(contract.validateQueueJobNaming(job), { outputNameTemplate: job.output_name_template, plannedOutputName: job.planned_output_name, outputName: null });
assert.deepEqual(contract.validateQueueJobNaming({}), null);
assert.throws(() => contract.validateQueueJobNaming({ ...job, output_name_template: '{name}/{layout}' }), /template|character/i);
assert.throws(() => contract.validateQueueJobNaming({ ...job, naming_schema_version: 2 }), /schema/i);
assert.throws(() => contract.validateQueueJobNaming({ ...job, planned_output_name: '' }), /planned|name/i);
assert.throws(() => contract.validateQueueJobNaming({ ...job, planned_output_name: 'D:\\\\private\\\\scene.png' }), /unsafe|output name/i);
assert.throws(() => contract.validateQueueJobNaming({ ...job, state: 'completed' }), /actual output name/i);
assert.equal(contract.validateQueueJobNaming({ ...job, state: 'completed', output_name: 'scene-sbs-uuid.png' }).outputName, 'scene-sbs-uuid.png');
const file = { name: 'scene.png' };
const ready = contract.freezeReadyQueueItem({ file, name: file.name, state: 'ready' }, settings, '{name}_{model}');
assert.equal(ready.settingsFrozen, true);
assert.equal(ready.namingFrozen, true);
assert.equal(ready.outputNameTemplateSnapshot, '{name}_{model}');
assert.equal(contract.outputNameTemplateForSubmit(ready, '{name}_{layout}'), '{name}_{model}');
assert.equal(contract.outputNameTemplateForSubmit({ ...ready, namingFrozen: false }, '{name}_{layout}'), '{name}_{layout}');
const capability = contract.outputNamingCapability({ naming_capabilities: { schema_version: 1, supported: true, max_template_length: 120, fields: ['name', 'layout', 'model', 'date'], unique_job_id_suffix: true } });
assert.equal(contract.outputNameTemplateForSubmission(capability, ready, '{name}_{layout}'), '{name}_{model}');
assert.equal(contract.outputNameTemplateForSubmission(null, ready, 'ignored'), null);
assert.equal(contract.canSnapshotOutputNameTemplate(capability, '{name}_{model}'), true);
assert.equal(contract.canSnapshotOutputNameTemplate(capability, '{name}/{layout}'), false);
assert.equal(contract.canSnapshotOutputNameTemplate(null, '{name}/{layout}'), true);
assert.equal(contract.outputNameTemplateForSubmission(capability, ready, '{name}/{layout}'), '{name}_{model}');
assert.throws(() => contract.outputNameTemplateForSubmission(capability, { ...ready, namingFrozen: false }, '{name}/{layout}'), /character|filename/i);
const retry = contract.makeReadyQueueCopy({ ...job, file, name: file.name, size: 10, mediaType: 'image', outputNameTemplateSnapshot: job.output_name_template }, settings, job.output_name_template);
assert.equal(retry.outputNameTemplateSnapshot, job.output_name_template);
assert.equal(contract.outputNameTemplateForQueueCopy({ outputNameTemplateSnapshot: job.output_name_template }, 'retry', '{name}_{layout}'), job.output_name_template);
const duplicate = contract.makeReadyQueueCopy({ file, name: file.name, size: 10, mediaType: 'image' }, settings, '{name}_{date}');
assert.equal(duplicate.outputNameTemplateSnapshot, '{name}_{date}');
assert.equal(contract.outputNameTemplateForQueueCopy({ outputNameTemplateSnapshot: job.output_name_template }, 'duplicate', '{name}_{date}'), '{name}_{date}');
assert.equal(contract.outputNameTemplateForSubmit({ ...duplicate, namingFrozen: false }, '{name}_{layout}'), '{name}_{layout}');
const mutation = contract.queuedNameUpdateRequest('job / 1', '{name}_{model}');
assert.equal(mutation.url, '/api/jobs/job%20%2F%201/naming');
assert.equal(mutation.options.method, 'POST');
assert.deepEqual(JSON.parse(mutation.options.body), { output_name_template: '{name}_{model}' });
assert.throws(() => contract.queuedNameUpdateRequest('', '{name}'), /job ID/i);
""")

    def test_queue_readiness_uses_each_frozen_settings_snapshot(self):
        self.run_contract("""
const small = { ...contract.performancePreset('balanced'), depth_mode: 'depth-anything-v2-small', execution_mode: 'cpu' };
const unsupportedVisible = { ...small, depth_mode: 'depth-anything-v2-large', execution_mode: 'gpu' };
const models = [{ id: 'depth-anything-v2-small', name: 'Small', state: 'ready' }];
const context = { runtime: { available: true, gpu_available: false }, models, customModelAvailable: false };
const frozen = { settingsFrozen: true, settingsSnapshot: small, name: 'frozen.png' };
const unfrozen = { settingsFrozen: false, settingsSnapshot: small, name: 'current.png' };
assert.deepEqual(contract.effectiveQueueSettings(frozen, unsupportedVisible), small);
assert.deepEqual(contract.effectiveQueueSettings(unfrozen, unsupportedVisible), unsupportedVisible);
assert.equal(contract.queueReadiness([frozen], unsupportedVisible, context), '');
assert.match(contract.queueReadiness([frozen, unfrozen], unsupportedVisible, context), /GPU|NVIDIA/i);
assert.equal(contract.queueReadiness([unfrozen], small, context), '');
const missingFrozenModel = { ...frozen, settingsSnapshot: { ...small, depth_mode: 'depth-anything-v2-base' } };
assert.match(contract.queueReadiness([missingFrozenModel], unsupportedVisible, context), /Base|install|model/i);
""")

    def test_completion_notice_policy_is_opt_in_bounded_and_transition_based(self):
        self.run_contract("""
assert.equal(contract.completionNotificationStatus(false, 'default', false).canEnable, false);
assert.equal(contract.completionNotificationStatus(true, 'default', false).enabled, false);
assert.equal(contract.completionNotificationStatus(true, 'default', false).canEnable, true);
assert.match(contract.completionNotificationStatus(true, 'denied', false).message, /browser settings/i);
assert.equal(contract.completionNotificationStatus(true, 'granted', true).enabled, true);
let tracker = null;
let result = contract.observeCompletionNotice(tracker, { id: 'old', state: 'completed', purpose: 'export' });
tracker = result.tracker;
assert.equal(result.notify, false);
result = contract.observeCompletionNotice(tracker, { id: 'clip', state: 'queued', purpose: 'preview_clip' });
tracker = result.tracker;
result = contract.observeCompletionNotice(tracker, { id: 'clip', state: 'completed', purpose: 'preview_clip' }, 'queued');
tracker = result.tracker;
assert.equal(result.notify, false);
result = contract.observeCompletionNotice(tracker, { id: 'new', state: 'queued', purpose: 'export' });
tracker = result.tracker;
assert.equal(result.notify, false);
result = contract.observeCompletionNotice(tracker, { id: 'new', state: 'completed', purpose: 'export' }, 'queued');
tracker = result.tracker;
assert.equal(result.notify, true);
result = contract.observeCompletionNotice(tracker, { id: 'new', state: 'completed', purpose: 'export' }, 'completed');
tracker = result.tracker;
assert.equal(result.notify, false);
for (let index = 0; index < 520; index++) tracker = contract.observeCompletionNotice(tracker, { id: `active-${index}`, state: 'processing', purpose: 'export' }).tracker;
assert.ok(tracker.trackedIds.length <= 512);
assert.ok(tracker.activeIds.length <= 512);
assert.ok(!tracker.trackedIds.includes('active-0'));
""")

    def test_keyboard_shortcut_policy_filters_context_and_readiness(self):
        self.run_contract("""
const base = { altKey: true, code: 'KeyP', previewEnabled: true, submitEnabled: true, target: { tagName: 'DIV' } };
assert.equal(contract.keyboardShortcutAction(base), 'preview');
assert.equal(contract.keyboardShortcutAction({ ...base, code: 'Enter' }), 'submit');
for (const blocked of [
  { ...base, ctrlKey: true }, { ...base, metaKey: true }, { ...base, shiftKey: true },
  { ...base, repeat: true }, { ...base, defaultPrevented: true }, { ...base, isComposing: true },
  { ...base, modalOpen: true }, { ...base, previewEnabled: false }, { ...base, target: { tagName: 'INPUT' } },
  { ...base, target: { tagName: 'TEXTAREA' } }, { ...base, target: { tagName: 'SELECT' } },
  { ...base, target: { tagName: 'DIV', isContentEditable: true } }
]) assert.equal(contract.keyboardShortcutAction(blocked), null);
assert.equal(contract.keyboardShortcutAction({ ...base, altKey: false }), null);
assert.equal(contract.keyboardShortcutAction({ ...base, code: 'KeyP', submitEnabled: false }), 'preview');
assert.equal(contract.keyboardShortcutAction({ ...base, code: 'Enter', submitEnabled: false }), null);
assert.equal(contract.keyboardShortcutAction({ ...base, code: 'KeyP', target: { tagName: 'DIV', isContentEditable: false } }), 'preview');
""")

    def test_retry_or_duplicate_requires_browser_file_and_freezes_snapshot(self):
        self.run_contract("""
const file = { name: 'scene.png' };
const settings = { layout: 'SBS', strength: 5, convergence: 0.5, depth_mode: 'depth-anything-v2-small', execution_mode: 'auto', input_size: 256, resolution: '1080p', quality: 'balanced', temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false };
const copy = contract.makeReadyQueueCopy({ file, name: file.name, size: 10, mediaType: 'image' }, settings);
assert.equal(copy.file, file);
assert.equal(copy.state, 'ready');
assert.equal(copy.settingsFrozen, true);
assert.deepEqual(copy.settingsSnapshot, settings);
assert.throws(() => contract.makeReadyQueueCopy({ name: file.name }, settings), /reselect/i);
""")

    def test_queue_phase_summary_is_readable_for_actual_progress(self):
        self.run_contract("""
assert.equal(contract.queuePhaseLabel('finalization'), 'Phase: finalization');
assert.equal(contract.queuePhaseLabel('retrying'), 'Phase: retrying');
assert.equal(contract.queuePhaseLabel('queued'), '');
""")

    def test_preview_zoom_geometry_clamps_scale_and_offsets_to_content(self):
        self.run_contract("""
const initial = contract.zoomPanGeometry(0.2, 80, -50, 800, 450, 1600, 900);
assert.deepEqual({ zoom: initial.zoom, panX: initial.panX, panY: initial.panY }, { zoom: 1, panX: 0, panY: 0 });
const zoomed = contract.zoomPanGeometry(9, 9999, -9999, 800, 450, 1600, 900);
assert.equal(zoomed.zoom, 4);
assert.equal(zoomed.panX, zoomed.maxPanX);
assert.equal(zoomed.panY, -zoomed.maxPanY);
assert.equal(zoomed.maxPanX, 1200);
assert.equal(zoomed.maxPanY, 675);
const portrait = contract.zoomPanGeometry(2, 900, -900, 800, 450, 600, 900);
assert.ok(Math.abs(portrait.panX) <= portrait.maxPanX);
assert.ok(Math.abs(portrait.panY) <= portrait.maxPanY);
assert.equal(contract.compareSplitPercent(-1), 0);
assert.equal(contract.compareSplitPercent(53.7), 53.7);
assert.equal(contract.compareSplitPercent(101), 100);
""")

    def test_source_depth_compare_requires_backend_source_png(self):
        self.run_contract("""
assert.equal(contract.sourceDepthCompareAvailable({ source: 'data:image/png;base64,abc', depth: 'data:image/png;base64,def' }), true);
assert.equal(contract.sourceDepthCompareAvailable({ depth: 'data:image/png;base64,def' }), false);
assert.equal(contract.sourceDepthCompareAvailable({ source: 'blob:live-video', depth: 'data:image/png;base64,def' }), false);
assert.equal(contract.sourceDepthCompareAvailable({ source: 'data:image/png;base64,abc', depth: '' }), false);
""")

    def test_history_capability_gates_old_disabled_and_degraded_servers(self):
        self.run_contract("""
assert.equal(contract.historyCapability({}), null);
const base = { schema_version: 1, supported: true, enabled: true, state: 'ready', records_limit: 1000, source_retention: false, automatic_requeue: false, frame_resume: false, instance_name: 'desktop', warning: null };
assert.equal(contract.historyCapability({ history_capabilities: base }).enabled, true);
assert.equal(contract.historyCapability({ history_capabilities: { ...base, enabled: false, state: 'disabled', instance_name: null } }).enabled, false);
assert.equal(contract.historyCapability({ history_capabilities: { ...base, state: 'degraded', warning: 'Check history storage.' } }).state, 'degraded');
assert.equal(contract.historyCapability({ history_capabilities: { ...base, schema_version: 2 } }), null);
assert.equal(contract.historyCapability({ history_capabilities: { ...base, enabled: false } }), null);
assert.equal(contract.historyCapability({ history_capabilities: { ...base, automatic_requeue: true } }), null);
""")

    def test_history_page_validates_readonly_records_and_preserves_interrupted_truth(self):
        self.run_contract("""
const item = {
  job_id: '2f17db52-3cf6-4c92-94df-93bf04c0cc17', created_at: 1792300000, updated_at: 1792300100, completed_at: null,
  state: 'interrupted', last_known_state: 'processing', filename: 'scene & take.mp4', media_type: 'video',
  source_size_bytes: 1234, source_sha256: 'a'.repeat(64), settings_schema_version: 1, naming_schema_version: 1,
  settings: contract.performancePreset('balanced'), output_name_template: '{name}_{layout}',
  planned_output_name: 'scene_sbs.mp4', output_name: null, output_available: false,
  readonly: true, source_available: false, requires_source_reselection: true
};
const page = contract.validateHistoryPage({ schema_version: 1, items: [item], next_cursor: 'opaque_cursor', invalid_records: 2 });
assert.equal(page.items[0].filename, 'scene & take.mp4');
assert.equal(contract.historyStateLabel(page.items[0]), 'Interrupted');
assert.match(contract.historyStateLabel({ state: 'completed' }), /Completed/);
assert.throws(() => contract.validateHistoryPage({ schema_version: 2, items: [], next_cursor: null, invalid_records: 0 }), /schema/);
assert.throws(() => contract.validateHistoryPage({ schema_version: 1, items: [ { ...item, source_available: true } ], next_cursor: null, invalid_records: 0 }), /read-only/);
assert.throws(() => contract.validateHistoryPage({ schema_version: 1, items: [ { ...item, settings: { ...item.settings, layout: 'evil' } } ], next_cursor: null, invalid_records: 0 }), /layout/);
assert.throws(() => contract.validateHistoryPage({ schema_version: 1, items: [ { ...item, state: 'interrupted', last_known_state: null } ], next_cursor: null, invalid_records: 0 }), /last-known/);
assert.throws(() => contract.validateHistoryPage({ schema_version: 1, items: [], next_cursor: '../escape', invalid_records: 0 }), /cursor/);
""")

if __name__ == "__main__":
    unittest.main()
