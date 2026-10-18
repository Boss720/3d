const frontendContract = (() => {
  function capabilityV1(runtime) {
    const caps = runtime?.capabilities;
    return Boolean(caps && caps.schema_version === 1 && caps.named_models_directml === true &&
      Number.isFinite(Number(caps.gpu_max_input_dimension)) && Number(caps.gpu_max_input_dimension) > 0 &&
      Number.isFinite(Number(caps.large_limited_gpu_max_input_dimension)) && Number(caps.large_limited_gpu_max_input_dimension) > 0 &&
      Number.isFinite(Number(caps.limited_gpu_memory_mb)) && Number(caps.limited_gpu_memory_mb) > 0 &&
      caps.unknown_memory_policy === "limited" && Number.isFinite(Number(caps.gpu_failure_cooldown_seconds)) &&
      caps.gpu_retry_supported === true);
  }
  function gpuPolicy(runtime, adapter, modelId = "depth-anything-v2-large") {
    const caps = runtime?.capabilities;
    const adapterAvailable = Boolean(adapter || runtime?.gpu_available);
    const reportedMemory = Number(adapter?.memory_mb);
    const memoryUnknown = !Number.isFinite(reportedMemory) || reportedMemory <= 0;
    const supported = capabilityV1(runtime);
    const limited = adapterAvailable && supported && (memoryUnknown
      ? caps.unknown_memory_policy === "limited"
      : reportedMemory <= Number(caps.limited_gpu_memory_mb));
    const inputCap = supported ? Number(limited && modelId === "depth-anything-v2-large" ? caps.large_limited_gpu_max_input_dimension : caps.gpu_max_input_dimension) : null;
    return {
      adapterAvailable, memoryUnknown, limited, supported, inputCap,
      upgradeRequired: adapterAvailable && !supported,
      message: adapterAvailable && !supported ? "This server does not report GPU capability schema v1. Upgrade StereoLab to enable Large GPU inference safely." : ""
    };
  }
  function selectedGpuAdapter(runtime) {
    if (runtime?.gpu_device && typeof runtime.gpu_device === "object") return runtime.gpu_device;
    const adapters = Array.isArray(runtime?.adapters) ? runtime.adapters : [];
    return adapters.find((adapter) => adapter?.selected === true || adapter?.is_selected === true) || (adapters.length === 1 ? adapters[0] : null);
  }
  function retrySupported(runtime) {
    return capabilityV1(runtime) && runtime.capabilities.gpu_retry_supported === true;
  }
  async function retryGpu(request, refreshStatus) {
    const result = await request("/api/runtime/gpu/reset", { method: "POST" });
    if (result?.reset !== true) throw new Error("The server did not confirm the GPU retry reset.");
    await refreshStatus();
    return result;
  }
  const fullStillMaxPixels = 12_000_000, fullStillMaxDimension = 8192, fullStillMaxUploadBytes = 24 * 1024 * 1024;
  function fullStillEligibility({ mediaType, width, height, fileSizeBytes } = {}) {
    if (mediaType !== "image") return { eligible: false, reason: "Full-detail preview is available for still images only. Select Proxy for video frames." };
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) return { eligible: false, reason: "Wait for the still image to finish decoding, then try Full-detail still again." };
    if (width > fullStillMaxDimension || height > fullStillMaxDimension) return { eligible: false, reason: `This still is ${width} × ${height} px. Full-detail still is limited to ${fullStillMaxDimension} px per side; use Proxy to preview it.` };
    if (width * height > fullStillMaxPixels) return { eligible: false, reason: `This still is ${width} × ${height} px (${(width * height / 1_000_000).toFixed(1)} MP). Full-detail still is limited to 12 MP; use Proxy to preview it.` };
    if (Number.isFinite(fileSizeBytes) && fileSizeBytes > fullStillMaxUploadBytes) return { eligible: false, reason: `This still is larger than the 24 MiB full-detail upload limit. Use Proxy to preview it or choose a smaller image.` };
    return { eligible: true, reason: "" };
  }
  function validatePreviewResponse(result, requestedDetail) {
    if (!result || typeof result !== "object" || !result.image || !result.depth) throw new Error("The preview response did not contain processed images.");
    if (!['proxy', 'full'].includes(requestedDetail)) throw new Error("Unknown preview detail request.");
    if (result.preview_detail != null && result.preview_detail !== "proxy" && result.preview_detail !== "full") throw new Error("The preview server returned an invalid preview_detail mode.");
    if (result.is_proxy != null && typeof result.is_proxy !== "boolean") throw new Error("The preview server returned an invalid is_proxy value.");
    if (requestedDetail === "full") {
      if (result.preview_detail !== "full" || result.is_proxy !== false) throw new Error("Full-detail still was requested, but the server did not confirm a full-detail result. Upgrade StereoLab or choose Proxy.");
      const sourceWidth = Number(result.source_width), sourceHeight = Number(result.source_height), originalWidth = Number(result.original_width), originalHeight = Number(result.original_height);
      if (![sourceWidth, sourceHeight, originalWidth, originalHeight].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error("The full-detail response did not report valid source and original dimensions. Upgrade StereoLab or choose Proxy.");
      if (sourceWidth !== originalWidth || sourceHeight !== originalHeight) throw new Error("The server downscaled the requested full-detail still. Choose Proxy or upgrade StereoLab.");
    } else {
      if (result.preview_detail != null && result.preview_detail !== "proxy" || result.is_proxy != null && result.is_proxy !== true) throw new Error("The preview server returned full-detail output for a Proxy request. Refresh or upgrade StereoLab.");
    }
    return result;
  }
  function previewDimensionDetails(result) {
    if (!result) return "";
    const mode = result.preview_detail === "full" && result.is_proxy === false ? "Full-detail still" : "Proxy";
    const sourceWidth = Number(result.source_width), sourceHeight = Number(result.source_height);
    const selectedWidth = Number(result.selected_source_width), selectedHeight = Number(result.selected_source_height);
    const selectedValid = Number.isSafeInteger(selectedWidth) && selectedWidth > 0 && Number.isSafeInteger(selectedHeight) && selectedHeight > 0;
    const uploadWidth = Number(result.original_width), uploadHeight = Number(result.original_height);
    const source = Number.isSafeInteger(sourceWidth) && sourceWidth > 0 && Number.isSafeInteger(sourceHeight) && sourceHeight > 0 ? `${sourceWidth} × ${sourceHeight} px` : "source dimensions unavailable";
    const selected = selectedValid ? ` · selected source ${selectedWidth} × ${selectedHeight} px` : Number.isSafeInteger(uploadWidth) && uploadWidth > 0 && Number.isSafeInteger(uploadHeight) && uploadHeight > 0 ? ` · decoded upload ${uploadWidth} × ${uploadHeight} px` : "";
    return `${mode} · processed ${source}${selected}`;
  }
  function processedClipCapability(status) {
    const caps = status?.preview_capabilities;
    const clip = caps?.processed_clip;
    if (caps?.schema_version !== 1 || !clip || clip.supported !== true || clip.temporal_context !== "fresh") return null;
    if (!Number.isSafeInteger(clip.max_duration_seconds) || clip.max_duration_seconds < 1 || clip.max_duration_seconds > 5 ||
        !Number.isSafeInteger(clip.max_frames) || clip.max_frames < 1 || clip.max_frames > 120 ||
        typeof clip.max_fps !== "number" || !Number.isFinite(clip.max_fps) || clip.max_fps <= 0 || clip.max_fps > 24 ||
        !Number.isSafeInteger(clip.max_width) || clip.max_width < 1 || clip.max_width > 960 ||
        !Number.isSafeInteger(clip.max_height) || clip.max_height < 1 || clip.max_height > 540) return null;
    return clip;
  }
  function validProcessedClipDuration(durationSeconds, status) {
    const capability = processedClipCapability(status);
    return Boolean(capability && Number.isSafeInteger(durationSeconds) && durationSeconds >= 1 && durationSeconds <= capability.max_duration_seconds && durationSeconds * capability.max_fps <= capability.max_frames);
  }
  function processedClipUnavailableMessage(status, loadError = "") {
    if (loadError) return `Could not load processed clip capabilities: ${loadError}. Use Refresh status to retry.`;
    return status ? "This server does not advertise processed clip preview support. Refresh or upgrade StereoLab." : "Checking processed clip preview support…";
  }
  function processedClipErrorAfterPoll(error, errorKind) { return errorKind === "poll" ? "" : error; }
  function validatePreviewClipJob(job) {
    if (!job || typeof job !== "object" || Array.isArray(job) || job.purpose !== "preview_clip") throw new Error("The server did not confirm purpose='preview_clip'. Refresh or upgrade StereoLab before retrying.");
    if (typeof job.id !== "string" || !job.id.trim()) throw new Error("The processed clip response did not include a job ID.");
    if (typeof job.state !== "string" || !["queued", "processing", "completed", "failed", "canceled"].includes(job.state)) throw new Error("The processed clip response contains an invalid job state.");
    if (typeof job.clip_start_seconds !== "number" || !Number.isFinite(job.clip_start_seconds) || job.clip_start_seconds < 0) throw new Error("The processed clip start time is invalid.");
    if (!Number.isSafeInteger(job.clip_duration_seconds) || job.clip_duration_seconds < 1 || job.clip_duration_seconds > 5) throw new Error("The processed clip duration is outside the 1–5 second range.");
    if (job.clip != null) {
      if (typeof job.clip !== "object" || Array.isArray(job.clip)) throw new Error("The processed clip metadata is invalid.");
      const clip = job.clip, proxy = clip.proxy;
      if (typeof clip.source_start_seconds !== "number" || !Number.isFinite(clip.source_start_seconds) || clip.source_start_seconds < 0 ||
          typeof clip.requested_duration_seconds !== "number" || !Number.isFinite(clip.requested_duration_seconds) || clip.requested_duration_seconds < 1 || clip.requested_duration_seconds > 5 ||
          clip.temporal_context !== "fresh" || !proxy || typeof proxy !== "object" || Array.isArray(proxy)) throw new Error("The processed clip metadata does not confirm a fresh, bounded proxy interval.");
      if (Math.abs(clip.source_start_seconds - job.clip_start_seconds) > 0.001 || Math.abs(clip.requested_duration_seconds - job.clip_duration_seconds) > 0.001) throw new Error("The processed clip metadata does not match the requested source interval.");
      if (!Number.isSafeInteger(proxy.width) || proxy.width < 1 || proxy.width > 960 || !Number.isSafeInteger(proxy.height) || proxy.height < 1 || proxy.height > 540 ||
          typeof proxy.fps !== "number" || !Number.isFinite(proxy.fps) || proxy.fps <= 0 || proxy.fps > 24.001 ||
          !Number.isSafeInteger(proxy.total_frames) || proxy.total_frames < 1 || proxy.total_frames > 120) throw new Error("The processed clip proxy dimensions, FPS, or frame count exceed preview limits.");
    }
    if (job.state === "completed" && !job.clip) throw new Error("The completed processed clip did not include proxy details.");
    return job;
  }
  function isOrdinaryExportJob(job) {
    return Boolean(job && (job.purpose == null || job.purpose === "export"));
  }
  function clipContextMatches(before, after) {
    return Boolean(before && after && ["sourceId", "settings", "startSeconds", "durationSeconds"].every((key) => before[key] === after[key]));
  }
  function processedClipResponseAction(requestContext, currentContext, cancelRequested = false) {
    return cancelRequested || !clipContextMatches(requestContext, currentContext) ? "cancel" : "display";
  }
  function processedClipQuarantineAfterPoll(quarantine, observedJob) {
    if (!quarantine) return null;
    if (!observedJob || observedJob.id !== quarantine.id) return quarantine;
    const state = typeof observedJob.state === "string" ? observedJob.state : quarantine.state;
    return ["completed", "failed", "canceled"].includes(state) ? null : { ...quarantine, state };
  }
  function queueCapabilities(status) {
    const caps = status?.queue_capabilities;
    return caps?.schema_version === 1 && caps.reorder_queued_exports === true && caps.edit_queued_exports === true && caps.remove_queued_exports === true && caps.settings_snapshot === true ? caps : null;
  }
  function validateQueueJobSnapshot(job) {
    if (!job || typeof job !== "object" || Array.isArray(job) || job.settings_schema_version !== 1) throw new Error("This server does not provide queue settings schema v1.");
    if (typeof job.source_available !== "boolean") throw new Error("The server did not report source availability.");
    if (job.queue_position !== null && (!Number.isSafeInteger(job.queue_position) || job.queue_position < 1)) throw new Error("The server reported an invalid queue position.");
    return { settings: validateSettings(job.settings), sourceAvailable: job.source_available, queuePosition: job.queue_position };
  }
  function moveReadyQueueItem(items, uid, direction) {
    if (!Array.isArray(items) || ![-1, 1].includes(direction)) return items;
    const readyIndexes = items.flatMap((item, index) => !item.jobId && item.state === "ready" ? [index] : []);
    const readyPosition = readyIndexes.findIndex((index) => items[index].uid === uid);
    const neighborIndex = readyIndexes[readyPosition + direction];
    if (readyPosition < 0 || neighborIndex === undefined) return items;
    const result = [...items], index = readyIndexes[readyPosition];
    [result[index], result[neighborIndex]] = [result[neighborIndex], result[index]];
    return result;
  }
  function moveQueuedExportIds(jobs, jobId, direction) {
    if (!Array.isArray(jobs) || ![-1, 1].includes(direction)) return [];
    const pending = jobs.filter((job) => job && (job.purpose == null || job.purpose === "export") && job.state === "queued" && Number.isSafeInteger(job.queue_position) && job.queue_position > 0)
      .sort((a, b) => a.queue_position - b.queue_position);
    const ids = pending.map((job) => job.id), index = ids.indexOf(jobId), neighbor = index + direction;
    if (index < 0 || neighbor < 0 || neighbor >= ids.length) return ids;
    [ids[index], ids[neighbor]] = [ids[neighbor], ids[index]];
    return ids;
  }
  function makeReadyQueueCopy(item, settings) {
    if (!item || !item.file) throw new Error("The original source file is unavailable. Reselect it before retrying or duplicating.");
    return { file: item.file, name: item.name, size: item.size, mediaType: item.mediaType, state: "ready", progress: 0, error: null, message: "Ready to submit as a new job", settingsSnapshot: validateSettings(settings), settingsFrozen: true };
  }
  function queuePhaseLabel(phase) { return typeof phase === "string" && phase.trim() && phase !== "queued" ? `Phase: ${phase.replace(/[_-]+/g," ")}` : ""; }
  function previewClipDetails(job) {
    const phase = String(job.phase || "").toLowerCase();
    let state = job.state === "queued" ? "Queued" : job.state === "failed" ? "Failed" : job.state === "canceled" ? "Canceled" : job.state === "completed" ? "Complete" : /final/.test(phase) ? "Finalizing clip" : /prepar|extract/.test(phase) ? "Preparing proxy" : "Converting clip";
    const start = Number(job.clip?.source_start_seconds ?? job.clip_start_seconds);
    const duration = Number(job.clip?.requested_duration_seconds ?? job.clip_duration_seconds);
    const interval = Number.isFinite(start) && start >= 0 && Number.isFinite(duration) && duration > 0 ? `requested source interval ${start.toFixed(2)}–${(start + duration).toFixed(2)} s` : "requested source interval unavailable";
    const proxy = job.clip?.proxy || {};
    const width = Number(proxy.width ?? proxy.output_width), height = Number(proxy.height ?? proxy.output_height), fps = Number(proxy.fps), frames = Number(proxy.total_frames);
    const proxyDuration = Number.isSafeInteger(frames) && frames > 0 && Number.isFinite(fps) && fps > 0 ? `${(frames / fps).toFixed(2)} s video` : "proxy duration pending";
    const proxyDetails = [Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0 ? `${width} × ${height} px` : "proxy dimensions pending", Number.isFinite(fps) && fps > 0 ? `${fps} fps` : "proxy FPS pending", Number.isSafeInteger(frames) && frames > 0 ? `${frames} frames` : "", proxyDuration].filter(Boolean).join(" · ");
    const model = typeof job.depth_mode === "string" && job.depth_mode ? `model ${job.depth_mode}` : "model pending";
    const execution = job.execution && typeof job.execution === "object" ? job.execution : {};
    const device = execution.active_provider || execution.device || job.execution_mode || "device pending";
    const progress = Number(job.progress);
    const progressLabel = Number.isFinite(progress) && progress >= 0 ? ` · ${Math.round(Math.min(100, progress))}%` : "";
    const temporal = job.clip?.temporal_context === "fresh" ? "fresh temporal history" : "fresh temporal history pending";
    return `${state}${progressLabel} · ${interval} · ${proxyDetails} · ${model} · ${device} · ${temporal}`;
  }
  function previewExecutionDetails(result) {
    const execution = result?.execution || {};
    const details = [];
    const width = Number(execution.input_width ?? result?.input_width);
    const height = Number(execution.input_height ?? result?.input_height);
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) details.push(`Model input ${width} × ${height} px`);
    const cap = Number(execution.input_cap ?? execution.input_size_limit ?? execution.input_dimension_cap ?? execution.max_input_dimension ?? execution.gpu_max_input_dimension ?? result?.input_cap);
    if (Number.isFinite(cap) && cap > 0) details.push(`cap ${cap} px`);
    if (typeof result?.cache?.hit === "boolean") details.push(result.cache.hit ? "Depth cache hit" : "Depth cache miss");
    return details.join(" · ");
  }
  function preflightDetails(report, formatBytes) {
    if (!report || typeof report !== "object") return { summary: "", basis: "", warnings: [] };
    const byteLabel = (value) => {
      if (value == null || value === "") return "";
      const number = Number(value);
      if (!Number.isFinite(number) || number < 0) return "";
      return number === 0 ? "0 B" : formatBytes(number);
    };
    const estimated = byteLabel(report.estimated_output_bytes);
    const free = byteLabel(report.disk_free_bytes);
    const width = Number(report.output_width), height = Number(report.output_height);
    const summary = [];
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) summary.push(`Output ${width} × ${height} px`);
    summary.push(estimated ? `Estimated disk need ${estimated} (not guaranteed)` : "Disk estimate unavailable (not guaranteed)");
    if (free) summary.push(`Free when checked ${free}`);
    const shorten = (value, limit) => value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
    const basis = typeof report.estimate_basis === "string" ? shorten(report.estimate_basis.trim(), 220) : "";
    const warnings = Array.isArray(report.warnings)
      ? report.warnings.filter((warning) => typeof warning === "string" && warning.trim()).slice(0, 3).map((warning) => shorten(warning.trim(), 180))
      : [];
    const visibleWarning = warnings.find((warning) => /insufficient.*disk space|free disk space is below|low (output )?disk space|disk space.*(low|insufficient)/i.test(warning)) || "";
    return { summary: summary.join(" · "), basis, warnings, visibleWarning };
  }
  function jobEtaLabel(item, formatDuration) {
    if (item.state === "completed" || item.eta_status === "complete") return "Complete";
    if (item.state === "queued") return "Waiting in queue";
    if (item.state !== "processing") return "";
    if (item.phase === "finalization" || item.eta_status === "finalizing") return "Finalizing output…";
    if (item.phase === "retrying") return "Restarting export";
    if (item.mediaType === "image") return "Still image · no time estimate";
    const number = (value) => value == null || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
    const seconds = number(item.eta_seconds);
    const range = Array.isArray(item.eta_range_seconds) ? item.eta_range_seconds.map(number) : [];
    if (item.eta_status === "available" && range.length === 2 && range.every((value) => value !== null && value >= 0) && range[1] >= range[0]) return `About ${formatDuration(range[0])}–${formatDuration(range[1])} remaining (approx.)`;
    if (item.eta_status === "available" && seconds !== null && seconds >= 0) return `About ${formatDuration(seconds)} remaining`;
    if (item.eta_status === "estimating") return "Estimating time left…";
    if (item.eta_status === "unavailable") return "Time estimate unavailable";
    return "Waiting for time estimate";
  }
  const performancePresets = Object.freeze({
    fast: Object.freeze({ layout: "SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 256, resolution: "1080p", quality: "fast", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false }),
    balanced: Object.freeze({ layout: "SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 384, resolution: "1080p", quality: "balanced", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false }),
    quality: Object.freeze({ layout: "SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 518, resolution: "source", quality: "high", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false }),
    conservative4gb: Object.freeze({ layout: "SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 256, resolution: "1080p", quality: "balanced", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false })
  });
  const settingsKeys = Object.freeze(["layout", "strength", "convergence", "depth_mode", "execution_mode", "input_size", "resolution", "quality", "temporal_smoothing", "edge_refinement", "invert_depth"]);
  function performancePreset(name) {
    const preset = performancePresets[name];
    return preset ? { ...preset } : null;
  }
  function validateSettings(settings) {
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) throw new Error("settings must be an object.");
    const keys = Object.keys(settings);
    const unknown = keys.filter((key) => !settingsKeys.includes(key));
    if (unknown.length) throw new Error(`Unknown settings field: ${unknown[0]}.`);
    const missing = settingsKeys.find((key) => !Object.prototype.hasOwnProperty.call(settings, key));
    if (missing) throw new Error(`Missing settings field: ${missing}.`);
    const enums = {
      layout: ["SBS", "HALFSBS", "TB", "HALFTB", "CROSS", "ANAGLYPH", "DEPTH", "RGBD"],
      depth_mode: ["depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large", "dpt-hybrid-midas", "balanced", "precision"],
      execution_mode: ["auto", "hybrid", "gpu", "cpu"], input_size: [256, 384, 518],
      resolution: ["1080p", "source"], quality: ["fast", "balanced", "high"]
    };
    for (const [key, values] of Object.entries(enums)) if (!values.includes(settings[key])) throw new Error(`${key} must be one of the supported values.`);
    for (const [key, min, max] of [["strength", 0, 10], ["convergence", 0, 1], ["temporal_smoothing", 0, 0.9]]) {
      if (typeof settings[key] !== "number" || !Number.isFinite(settings[key]) || settings[key] < min || settings[key] > max) throw new Error(`${key} must be a finite number from ${min} to ${max}.`);
    }
    for (const key of ["edge_refinement", "invert_depth"]) if (typeof settings[key] !== "boolean") throw new Error(`${key} must be a boolean.`);
    return Object.fromEntries(settingsKeys.map((key) => [key, settings[key]]));
  }
  function validateUserPreset(document) {
    if (!document || typeof document !== "object" || Array.isArray(document)) throw new Error("Preset file must contain an object.");
    const unknown = Object.keys(document).filter((key) => !["schema_version", "settings"].includes(key));
    if (unknown.length) throw new Error(`Unknown preset field: ${unknown[0]}.`);
    if (document.schema_version !== 1) throw new Error("Unsupported schema_version; expected 1.");
    return validateSettings(document.settings);
  }
  function exportUserPreset(settings) {
    return { schema_version: 1, settings: validateSettings(settings) };
  }
  function hasUnsavedSettings(baseline, current) {
    if (!baseline || !current) return true;
    const keys = new Set([...Object.keys(baseline), ...Object.keys(current)]);
    return [...keys].some((key) => baseline[key] !== current[key]);
  }
  const calibrationSizes = Object.freeze([256, 384, 518]);
  function calibrationNumber(value, label, nullable = false) {
    if (nullable && value === null) return null;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error(`${label} must be a finite non-negative number.`);
    return value;
  }
  function validateCalibrationReport(report) {
    if (!report || typeof report !== "object" || Array.isArray(report)) throw new Error("Calibration response must be an object.");
    if (report.schema_version !== 1) throw new Error("Unsupported calibration schema_version; expected 1.");
    if (typeof report.budget_seconds !== "number" || !Number.isFinite(report.budget_seconds) || report.budget_seconds < 1 || report.budget_seconds > 30) throw new Error("Calibration budget is invalid.");
    const elapsed = calibrationNumber(report.elapsed_seconds, "elapsed_seconds");
    const totalElapsed = report.total_elapsed_seconds == null ? null : calibrationNumber(report.total_elapsed_seconds, "total_elapsed_seconds");
    if (typeof report.budget_exceeded !== "boolean" || typeof report.sufficient_evidence !== "boolean" || typeof report.saved !== "boolean") throw new Error("Calibration response status fields are invalid.");
    if (report.fastest_measured_input_size !== null && !calibrationSizes.includes(report.fastest_measured_input_size)) throw new Error("fastest_measured_input_size is invalid.");
    if (!Array.isArray(report.candidates)) throw new Error("Calibration candidates are missing.");
    if (typeof report.scope !== "string" || !Array.isArray(report.limitations) || !report.limitations.every((value) => typeof value === "string") || report.storage_warning != null && typeof report.storage_warning !== "string") throw new Error("Calibration scope, limitations, or storage warning are invalid.");
    const seen = new Set();
    const candidates = report.candidates.map((candidate) => {
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate) || !calibrationSizes.includes(candidate.input_size) || seen.has(candidate.input_size)) throw new Error("Calibration candidate input size is invalid or duplicated.");
      seen.add(candidate.input_size);
      if (!Array.isArray(candidate.samples_seconds) || !Array.isArray(candidate.warmup_seconds)) throw new Error("Calibration sample data is invalid.");
      const samples = candidate.samples_seconds.map((value) => calibrationNumber(value, "sample"));
      const warmups = candidate.warmup_seconds.map((value) => calibrationNumber(value, "warmup"));
      const median = calibrationNumber(candidate.median_seconds, "median_seconds", true);
      let range = candidate.range_seconds;
      if (range !== null) {
        if (!Array.isArray(range) || range.length !== 2) throw new Error("Candidate range_seconds is invalid.");
        range = range.map((value) => calibrationNumber(value, "range_seconds"));
        if (range[1] < range[0]) throw new Error("Candidate range_seconds must be ordered.");
      }
      if (candidate.error != null && typeof candidate.error !== "string") throw new Error("Candidate error must be text or null.");
      return { ...candidate, samples_seconds: samples, warmup_seconds: warmups, median_seconds: median, range_seconds: range };
    });
    return { ...report, elapsed_seconds: elapsed, ...(totalElapsed === null ? {} : { total_elapsed_seconds: totalElapsed }), candidates };
  }
  function calibrationEvidenceComplete(report) {
    return Boolean(report?.sufficient_evidence && calibrationSizes.every((size) => {
      const candidate = report.candidates.find((row) => row.input_size === size);
      return candidate && candidate.median_seconds !== null && candidate.samples_seconds.length >= 2 && !candidate.error;
    }));
  }
  function calibrationApplicable(report) {
    if (!calibrationEvidenceComplete(report)) return false;
    const fastest = report.candidates.reduce((best, row) => row.median_seconds < best.median_seconds ? row : best);
    return report.fastest_measured_input_size === fastest.input_size;
  }
  function calibrationSeconds(value) { return `${value.toFixed(3)} s`; }
  function calibrationDetails(report) {
    const complete = calibrationEvidenceComplete(report), applicable = calibrationApplicable(report);
    const elapsed = report.total_elapsed_seconds ?? report.elapsed_seconds;
    const elapsedLabel = report.total_elapsed_seconds == null ? "Elapsed through measurements (persistence time not reported)" : "Total elapsed (including setup, measurement, and persistence)";
    const summary = [`${elapsedLabel}: ${calibrationSeconds(elapsed)} / ${calibrationSeconds(report.budget_seconds)}`];
    if (report.budget_exceeded) summary.push("Budget exceeded; one in-flight inference cannot be interrupted");
    if (applicable) summary.push(`Fastest measured depth input: ${report.fastest_measured_input_size} px`);
    else summary.push("Insufficient evidence to identify a fastest measured input");
    summary.push(report.saved ? "Measurement saved locally" : "Measurement was not saved");
    const candidates = report.candidates.map((row) => {
      const median = row.median_seconds === null ? "median unavailable" : `median ${calibrationSeconds(row.median_seconds)}`;
      const range = row.range_seconds ? `range ${calibrationSeconds(row.range_seconds[0])}–${calibrationSeconds(row.range_seconds[1])}` : "range unavailable";
      const execution = typeof row.execution === "string" ? row.execution : row.execution && typeof row.execution === "object" ? [row.execution.active_provider, row.execution.device, row.execution.requested_mode].filter((value) => typeof value === "string").join(" · ") || "execution details reported" : "execution details unavailable";
      const error = typeof row.error === "string" && row.error.trim() ? `; error: ${row.error.trim().slice(0, 180)}` : "";
      return `${row.input_size} px: ${median} (${range}); ${row.samples_seconds.length} samples; warmup ${row.warmup_seconds.length}; ${execution.slice(0, 120)}${error}`;
    });
    return { summary: summary.join(" · "), candidates, scope: report.scope.slice(0, 300), limitations: report.limitations.slice(0, 8).map((value) => value.slice(0, 220)), applicable: applicable && complete };
  }
  function calibrationStatusLabel(report) { return calibrationApplicable(report) ? "Measurement complete. Review results and the one-frame scope before applying." : "Measurement complete, but there is not enough evidence to apply an input size."; }
  function calibrationContextMatches(before, after) {
    return Boolean(before && after && ["revision", "sourceId", "frameTime", "settings", "budget", "modelIdentity", "previewDetail"].every((key) => before[key] === after[key]));
  }
  function zoomPanGeometry(zoom, panX, panY, viewportWidth, viewportHeight, contentWidth, contentHeight) {
    const finite = (value, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
    const vw = Math.max(0, finite(viewportWidth)), vh = Math.max(0, finite(viewportHeight));
    const cw = Math.max(0, finite(contentWidth, vw)), ch = Math.max(0, finite(contentHeight, vh));
    const scale = Math.max(1, Math.min(4, finite(zoom, 1)));
    const fit = vw > 0 && vh > 0 && cw > 0 && ch > 0 ? Math.min(vw / cw, vh / ch) : 1;
    const maxPanX = Math.max(0, (cw * fit * scale - vw) / 2), maxPanY = Math.max(0, (ch * fit * scale - vh) / 2);
    const clamp = (value, max) => { const clamped=Math.max(-max,Math.min(max,finite(value)));return Object.is(clamped,-0)?0:clamped; };
    return { zoom: scale, panX: clamp(panX, maxPanX), panY: clamp(panY, maxPanY), maxPanX, maxPanY };
  }
  function compareSplitPercent(value) {
    return Math.max(0, Math.min(100, typeof value === "number" && Number.isFinite(value) ? value : 50));
  }
  function sourceDepthCompareAvailable(result) {
    const png = (value) => typeof value === "string" && /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(value);
    return Boolean(result && png(result.source) && png(result.depth));
  }
  return Object.freeze({ capabilityV1, gpuPolicy, selectedGpuAdapter, retrySupported, retryGpu, fullStillEligibility, validatePreviewResponse, previewDimensionDetails, processedClipCapability, validProcessedClipDuration, processedClipUnavailableMessage, processedClipErrorAfterPoll, validatePreviewClipJob, isOrdinaryExportJob, clipContextMatches, processedClipResponseAction, processedClipQuarantineAfterPoll, queueCapabilities, validateQueueJobSnapshot, moveReadyQueueItem, moveQueuedExportIds, makeReadyQueueCopy, queuePhaseLabel, previewClipDetails, previewExecutionDetails, preflightDetails, jobEtaLabel, performancePreset, validateSettings, validateUserPreset, exportUserPreset, hasUnsavedSettings, validateCalibrationReport, calibrationApplicable, calibrationDetails, calibrationStatusLabel, calibrationContextMatches, zoomPanGeometry, compareSplitPercent, sourceDepthCompareAvailable });
})();
if (typeof module !== "undefined" && module.exports) module.exports = frontendContract;
if (typeof window !== "undefined") window.StereoLabFrontendContract = frontendContract;

if (typeof document !== "undefined") (() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ids = ["mediaInput","dropZone","browseButton","sourceVideo","sourceImage","processedImage","previewEmpty","previewStage","sourceName","previewState","generatePreview","layoutSelect","formatHint","previewModeLabel","presetSelect","depthMode","depthNote","installSelected","modelRow","modelInput","modelButton","modelState","modelSizeRow","modelSizeLabel","modelSize","midasSizeNote","strength","strengthValue","convergence","convergenceValue","temporalSmoothing","smoothingValue","edgeRefinement","invertDepth","resolution","quality","convertButton","convertCount","convertHint","queueRows","queueEmpty","queueCount","queueHeadingCount","clearQueueButton","applySettingsToReady","queueWorkflowStatus","enginePill","engineText","toolDescription","toolPath","runtimeDescription","runtimeCommand","modelCards","refreshModels","toast","helpButton","aboutDialog","aboutDone"];
  ids.push("executionMode", "executionNote", "previewExecution", "retryGpu", "performancePreset", "performancePresetStatus", "performancePresetSummary", "resetPerformancePreset", "exportPreset", "importPresetButton", "importPresetFile", "calibrationBudget", "calibrateInput", "calibrationStatus", "calibrationReport", "calibrationSummary", "calibrationCandidates", "calibrationScope", "calibrationLimitations", "calibrationStorageWarning", "applyCalibration", "tabCompare", "previewMedia", "sourceDepthCompare", "compareSource", "compareDepth", "compareDivider", "compareControls", "compareSlider", "compareValue", "compareResolution", "zoomIn", "zoomOut", "zoomLabel", "resetZoom", "previewDetail", "previewDetailHint", "processedClipCapability", "processedClipDuration", "generateProcessedClip", "cancelProcessedClip", "processedClipStatus", "processedClipVideo");
  const ui = Object.fromEntries(ids.map((id) => [id, $(id)]));
  const imageExtensions = new Set(["jpg","jpeg","png","webp"]);
  const videoExtensions = new Set(["mp4","webm","mkv","mov","avi","wmv","flv","ts","mts","m2ts","m4v"]);
  const activeStates = new Set(["uploading","queued","processing"]);
  const layouts = {
    SBS:["Side by side","Two full-resolution views, side by side."],HALFSBS:["Half side by side","Two squeezed views share a source-sized frame."],
    TB:["Top / bottom","Two full-resolution views, stacked vertically."],HALFTB:["Half top / bottom","Two squeezed views share a source-sized frame."],
    CROSS:["Cross-eye","Right view first, for cross-eye viewing."],ANAGLYPH:["Red / cyan","A color-filter stereo image for red/cyan glasses."],
    DEPTH:["Depth map","A grayscale depth estimate: brighter areas are nearer."],RGBD:["RGB + depth","Original color and depth map, side by side."]
  };
  const presets = {cinema:{strength:5,convergence:.5,smoothing:.35,edges:true,invert:false},subtle:{strength:2.5,convergence:.5,smoothing:.5,edges:true,invert:false},action:{strength:8,convergence:.45,smoothing:.25,edges:true,invert:false}};
  let performanceBaseline = frontendContract.performancePreset("balanced"), importedPresetBaseline = false;
  let calibrationBusy = false, calibrationResult = null, calibrationResultContext = null, calibrationNotice = "";
  let items = [], selectedUid = null, previewUrl = null, previewFile = null, previewView = "original", previewResult = null;
  let previewRevision = 0, resultRevision = -1, previewBusy = false, sourceError = "", previewError = "";
  let zoomState = { zoom: 1, panX: 0, panY: 0 }, panPointer = null, compareSplit = 50;
  let models = [], runtime = null, status = null, statusLoadError = "", customModelAvailable = false, catalogError = "";
  let modelTimer = null, jobTimer = null, jobsBusy = false, modelsBusy = false, submitting = false, queueMutationBusy = false, queueWorkflowMessage = "", toastTimer = null;
  let previewClip = { generation: 0, requestPending: false, context: null, job: null, quarantine: null, stale: false, cancelWhenReturned: false, cancelBusy: false, timer: null, error: "", errorKind: "" };
  const hiddenJobs = new Set();
  try { JSON.parse(sessionStorage.getItem("stereolab.hiddenJobs") || "[]").forEach((id) => hiddenJobs.add(id)); } catch { /* Session storage may be disabled. */ }
  function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text != null) node.textContent = text; return node; }
  function toast(message, error = false) { ui.toast.textContent = message; ui.toast.classList.toggle("error", error); ui.toast.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 5500); }
  async function request(url, options = {}) {
    const response = await fetch(url, options); let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) { const error = new Error(typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((x) => x.msg).join("; ") : `Request failed (${response.status}).`); error.status = response.status; throw error; }
    return data;
  }
  function extension(file) { return (file.name.split(".").pop() || "").toLowerCase(); }
  function isImage(file) { return imageExtensions.has(extension(file)); }
  function formatBytes(bytes) { if (!bytes) return ""; const units=["B","KB","MB","GB"]; const n=Math.min(3,Math.floor(Math.log(bytes)/Math.log(1024))); return `${(bytes / 1024 ** n).toFixed(n ? 1 : 0)} ${units[n]}`; }
  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    let remaining = Math.ceil(seconds);
    const hours = Math.floor(remaining / 3600); remaining %= 3600;
    const minutes = Math.floor(remaining / 60); const secs = remaining % 60;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return secs ? `${minutes}m ${secs}s` : `${minutes}m`;
    return `${secs}s`;
  }
  function metricNumber(value) {
    if (value == null || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }
  function etaLabel(item) {
    return frontendContract.jobEtaLabel(item, formatDuration);
  }
  function progressDetails(item) {
    const details = [];
    const eta = etaLabel(item);
    if (eta) details.push(eta);
    const phase = frontendContract.queuePhaseLabel(item.phase);if(phase)details.push(phase);
    const frames = metricNumber(item.frames_processed), total = metricNumber(item.total_frames), fps = metricNumber(item.processing_fps), elapsed = metricNumber(item.elapsed_seconds);
    if (frames !== null && frames >= 0 && total !== null && total > 0) details.push(`Frames ${Math.min(frames, total).toLocaleString()} / ${total.toLocaleString()}`);
    if (fps !== null && fps > 0) details.push(`${fps.toFixed(1)} fps`);
    if (elapsed !== null && elapsed >= 0) details.push(`Elapsed ${formatDuration(elapsed)}`);
    if (typeof item.encoder === "string" && item.encoder.trim()) details.push(`Encoder: ${item.encoder.trim()}`);
    if (item.phase === "retrying" && metricNumber(item.retry_count) !== null) details.push(`Retry ${item.retry_count}`);
    if (typeof item.retry_reason === "string" && item.retry_reason.trim()) details.push(`Retry reason: ${item.retry_reason.trim()}`);
    return details;
  }
  function selectedModel() { return models.find((model) => model.id === ui.depthMode.value); }
  function gpuPolicy() { return frontendContract.gpuPolicy(runtime, frontendContract.selectedGpuAdapter(runtime), ui.depthMode.value); }
  function largeOnLimitedGpu() { return ui.depthMode.value === "depth-anything-v2-large" && gpuPolicy().limited; }
  function largeGpuUpgradeRequired() { return ui.depthMode.value === "depth-anything-v2-large" && Boolean(runtime?.gpu_available) && !frontendContract.capabilityV1(runtime); }
  function gpuBlock() {
    if (ui.executionMode.value !== "gpu") return "";
    if (["balanced", "precision"].includes(ui.depthMode.value)) return "GPU inference requires a named neural model. Choose Auto or CPU only for visual approximation or custom ONNX.";
    if (!runtime) return "Checking NVIDIA GPU support…";
    if (!runtime.gpu_available) return "NVIDIA GPU is unavailable. Run .\\setup-gpu.ps1 to prepare the separate .venv-gpu environment, then start it with .\\run.ps1. Or choose Auto or CPU only.";
    if (ui.depthMode.value === "depth-anything-v2-large" && !frontendContract.capabilityV1(runtime)) return gpuPolicy().message || "This server does not report GPU capability schema v1. Upgrade StereoLab to enable Large GPU inference safely.";
    return "";
  }
  function executionSummary(execution, pipeline) {
    if (!execution) return "";
    const provider = execution.active_provider;
    const label = provider && /CUDA|Dml|Tensorrt/i.test(provider) ? "GPU depth + CPU processing" : provider === "CPUExecutionProvider" || execution.device === "cpu" ? "CPU processing" : provider ? `${provider} depth` : "Device use not reported";
    return [label, execution.fallback_reason ? `CPU fallback: ${execution.fallback_reason}` : "", pipeline === "overlapped" ? "video stages overlapped" : ""].filter(Boolean).join(" · ");
  }
  function renderDeviceNote() {
    const mode = ui.executionMode.value, blocked = gpuBlock();
    ui.retryGpu.hidden = !frontendContract.retrySupported(runtime);
    ui.executionNote.classList.toggle("device-warning", Boolean(blocked) || mode !== "cpu" && (largeOnLimitedGpu() || largeGpuUpgradeRequired() || runtime && !runtime.gpu_available));
    if (blocked) { ui.executionNote.textContent = blocked; return; }
    if (["balanced", "precision"].includes(ui.depthMode.value)) { ui.executionNote.textContent = ui.depthMode.value === "balanced" ? "Visual approximation uses CPU image cues. Choose a named neural depth engine to use GPU depth estimation." : "Custom ONNX models use CPU in this studio. Choose a named neural depth engine to use GPU depth estimation."; return; }
    if (mode === "cpu") { ui.executionNote.textContent = "CPU depth estimation and media processing. GPU is not requested."; return; }
    if (!runtime) { ui.executionNote.textContent = "Checking local GPU support…"; return; }
    if (largeGpuUpgradeRequired()) { ui.executionNote.textContent = gpuPolicy().message; return; }
    if (largeOnLimitedGpu()) {
      const policy = gpuPolicy(), cap = policy.inputCap;
      ui.executionNote.textContent = `Large will try DirectML on this ${policy.memoryUnknown ? "GPU with unknown memory, treated conservatively" : "limited-memory GPU"} with a ${cap} px long-side cap. Start at 256 px detail to reduce memory use. Auto/Hybrid fall back to CPU if GPU inference fails; GPU-only reports the error.`;
      return;
    }
    if (!runtime.gpu_available) { ui.executionNote.textContent = `CPU fallback: ${runtime.error || "no compatible GPU runtime and NVIDIA adapter detected"}. Optional GPU setup: .\\setup-gpu.ps1 (separate .venv-gpu), then .\\run.ps1.`; return; }
    const policy = gpuPolicy(), capNote = policy.supported ? ` Named-model input cap ${policy.inputCap} px.` : "";
    ui.executionNote.textContent = `${runtime.gpu_name || "NVIDIA GPU"} available${runtime.gpu_verified ? " · inference verified" : " · inference not yet verified"}.${capNote} ${mode === "gpu" ? "GPU requested; failures will be reported." : "GPU depth with CPU media processing; CPU fallback if needed."}`;
  }
  function readiness() {
    const blocked = gpuBlock();
    if (blocked) return blocked;
    const mode = ui.depthMode.value;
    if (mode === "balanced") return "";
    if (mode === "precision") return customModelAvailable ? "" : "Choose a compatible ONNX model to use Custom mode.";
    if (runtime && !runtime.available) return "Install ONNX Runtime to use neural depth models. See Local runtime below.";
    const model = selectedModel();
    if (!model) return catalogError ? `Model catalog unavailable: ${catalogError}` : "Checking the selected model…";
    if (model.state === "downloading") return `${model.name} is downloading. Wait for installation to finish.`;
    if (model.state !== "ready") return `Install ${model.name} to use this depth engine.`;
    return "";
  }
  function sourceReady() { return previewFile && !sourceError && (isImage(previewFile) ? ui.sourceImage.complete && ui.sourceImage.naturalWidth > 0 : ui.sourceVideo.readyState >= 2); }
  function previewClipContext(){return {sourceId:selectedUid,settings:JSON.stringify(getSettings()),startSeconds:ui.sourceVideo.currentTime,durationSeconds:Number(ui.processedClipDuration.value)};}
  function previewClipActive(){return previewClip.requestPending||Boolean(previewClip.quarantine)||previewClip.job&&["queued","processing"].includes(previewClip.job.state);}
  function processedClipBlock(){
    const capability=frontendContract.processedClipCapability(status);if(!capability)return status?"This server does not advertise processed clip preview support. Refresh or upgrade StereoLab.":"Checking processed clip preview support…";
    if(!previewFile||isImage(previewFile))return "Select a video to generate a processed clip preview.";
    if(!sourceReady())return sourceError||"Wait for the browser to decode the selected video.";
    if(!status?.ffmpeg_found)return "FFmpeg is required to prepare a bounded clip preview. Install or configure FFmpeg, then refresh StereoLab.";
    const settingsBlock=readiness();if(settingsBlock)return settingsBlock;
    const duration=Number(ui.processedClipDuration.value);if(!frontendContract.validProcessedClipDuration(duration,status))return "Choose a clip length supported by this server.";
    const start=Number(ui.sourceVideo.currentTime);if(!Number.isFinite(start)||start<0)return "The selected video timestamp is unavailable.";
    const remaining=Number(ui.sourceVideo.duration)-start;if(Number.isFinite(remaining)&&remaining+0.001<duration)return `Only ${Math.max(0,remaining).toFixed(1)} seconds remain at this timestamp. Seek earlier or choose a shorter clip.`;
    return "";
  }
  function clearProcessedClipVideo(){ui.processedClipVideo.pause();ui.processedClipVideo.removeAttribute("src");delete ui.processedClipVideo.dataset.jobId;ui.processedClipVideo.load();ui.processedClipVideo.hidden=true;}
  function updateProcessedClipPanel(){
    const capability=frontendContract.processedClipCapability(status),block=processedClipBlock();
    const unavailable=frontendContract.processedClipUnavailableMessage(status,statusLoadError);
    ui.processedClipCapability.textContent=capability?`Available · up to ${capability.max_duration_seconds} seconds, ${capability.max_width} × ${capability.max_height} px, ${capability.max_fps} fps, ${capability.max_frames} frames. Temporal history starts fresh.`:unavailable;
    for(const option of ui.processedClipDuration.options)option.disabled=!frontendContract.validProcessedClipDuration(Number(option.value),status);
    const active=previewClipActive();ui.generateProcessedClip.disabled=Boolean(block)||active;ui.cancelProcessedClip.hidden=!previewClip.requestPending&&!active;ui.cancelProcessedClip.disabled=previewClip.cancelBusy;
    if(previewClip.quarantine){const state=previewClip.quarantine.state&&previewClip.quarantine.state!=="unknown"?` Server state: ${previewClip.quarantine.state}.`:" Waiting for a server state.";ui.processedClipStatus.textContent=`${previewClip.quarantine.message}${state} ${previewClip.quarantine.cancelError||"The job is hidden and will remain blocked until the server reports a terminal state."}`;}
    else if(previewClip.error)ui.processedClipStatus.textContent=previewClip.error;
    else if(previewClip.requestPending)ui.processedClipStatus.textContent=previewClip.stale?"Source, settings, or timestamp changed. Waiting for the submitted request ID so the obsolete job can be canceled…":"Uploading the original video and starting the selected proxy interval…";
    else if(previewClip.stale&&previewClip.job)ui.processedClipStatus.textContent=`Superseded clip · not displayed · ${frontendContract.previewClipDetails(previewClip.job)}`;
    else if(previewClip.job)ui.processedClipStatus.textContent=frontendContract.previewClipDetails(previewClip.job);
    else ui.processedClipStatus.textContent=statusLoadError?unavailable:(block||"Pause or seek the source video, choose a clip length, then generate the processed preview.");
    const playable=Boolean(previewClip.job?.state==="completed"&&!previewClip.stale&&!previewClip.error);ui.processedClipVideo.hidden=!playable;
    if(playable&&ui.processedClipVideo.dataset.jobId!==previewClip.job.id){ui.processedClipVideo.dataset.jobId=previewClip.job.id;ui.processedClipVideo.src=`/api/preview-clips/${encodeURIComponent(previewClip.job.id)}/video`;ui.processedClipVideo.load();}
  }
  function invalidateProcessedClipIfStale(reason="Source, settings, or timestamp changed"){
    if(!previewClip.context||previewClip.stale||frontendContract.clipContextMatches(previewClip.context,previewClipContext()))return;
    previewClip.stale=true;previewClip.cancelWhenReturned=previewClip.requestPending;previewClip.error="";previewClip.errorKind="";clearProcessedClipVideo();
    if(previewClip.job&&["queued","processing"].includes(previewClip.job.state))void cancelProcessedClipJob(previewClip.job.id,previewClip.generation,reason);
    updateProcessedClipPanel();
  }
  async function cancelProcessedClipJob(jobId,generation,reason="Canceled by user"){
    if(!jobId||generation!==previewClip.generation||previewClip.cancelBusy)return;
    previewClip.cancelBusy=true;updateProcessedClipPanel();
    try{const result=await request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`,{method:"POST"});if(generation!==previewClip.generation)return;if(result&&result.id===jobId&&result.purpose==="preview_clip")previewClip.job=frontendContract.validatePreviewClipJob(result);}
    catch(error){if(generation===previewClip.generation){previewClip.error=`Could not cancel the processed clip: ${error.message}`;previewClip.errorKind="cancel";}}
    finally{if(generation===previewClip.generation){previewClip.cancelBusy=false;updateProcessedClipPanel();if(previewClip.job&&["queued","processing"].includes(previewClip.job.state))scheduleProcessedClipPoll(jobId,generation);}}
  }
  function cancelProcessedClip(){
    if(previewClip.quarantine){void retryQuarantinedClipCancellation(previewClip.generation);return;}
    if(!previewClip.requestPending&&!previewClipActive())return;
    previewClip.stale=true;previewClip.cancelWhenReturned=previewClip.requestPending;previewClip.error="";previewClip.errorKind="";clearProcessedClipVideo();
    if(previewClip.job&&["queued","processing"].includes(previewClip.job.state))void cancelProcessedClipJob(previewClip.job.id,previewClip.generation,"Canceled by user");
    updateProcessedClipPanel();
  }
  async function cancelUnexpectedClipResponse(job,generation,message){
    if(generation!==previewClip.generation)return;
    const jobId=typeof job?.id==="string"?job.id:"";
    previewClip.requestPending=false;previewClip.cancelWhenReturned=false;previewClip.error="";previewClip.errorKind="";previewClip.job=null;
    if(!jobId){previewClip.error=message;previewClip.errorKind="response";updateProcessedClipPanel();return;}
    hiddenJobs.add(jobId);saveHidden();
    const rawState=typeof job.state==="string"?job.state:"unknown";
    previewClip.quarantine={id:jobId,state:rawState,message,cancelError:""};
    if(["completed","failed","canceled"].includes(rawState)){
      previewClip.quarantine=frontendContract.processedClipQuarantineAfterPoll(previewClip.quarantine,job);
      previewClip.error=`${message} The server reports terminal state: ${rawState}.`;
      previewClip.errorKind="response";updateProcessedClipPanel();return;
    }
    updateProcessedClipPanel();
    await retryQuarantinedClipCancellation(generation);
  }
  async function retryQuarantinedClipCancellation(generation){
    const quarantine=previewClip.quarantine;if(!quarantine||generation!==previewClip.generation||previewClip.cancelBusy)return;
    previewClip.cancelBusy=true;quarantine.cancelError="";updateProcessedClipPanel();
    try{
      const result=await request(`/api/jobs/${encodeURIComponent(quarantine.id)}/cancel`,{method:"POST"});
      if(generation!==previewClip.generation||previewClip.quarantine?.id!==quarantine.id)return;
      if(result?.id!==quarantine.id)quarantine.cancelError="The cancel endpoint did not confirm this job ID.";
      else{
        previewClip.quarantine=frontendContract.processedClipQuarantineAfterPoll(quarantine,result);
        if(previewClip.quarantine)previewClip.quarantine.cancelError="Cancellation was requested; waiting for the server to report a terminal state.";
        else previewClip.error=`${quarantine.message} The server reports terminal state: ${result.state}.`;
      }
    }catch(error){if(generation===previewClip.generation&&previewClip.quarantine?.id===quarantine.id)quarantine.cancelError=`Cancellation could not be confirmed: ${error.message} Use Cancel clip to retry.`;}
    finally{if(generation===previewClip.generation){previewClip.cancelBusy=false;updateProcessedClipPanel();if(previewClip.quarantine)scheduleProcessedClipPoll(quarantine.id,generation);}}
  }
  async function generateProcessedClip(){
    const block=processedClipBlock();if(block){toast(block,true);updateProcessedClipPanel();return;}if(previewClipActive())return;
    ui.sourceVideo.pause();const duration=Number(ui.processedClipDuration.value),context=previewClipContext(),file=previewFile,settings=getSettings(),generation=previewClip.generation+1;
    clearTimeout(previewClip.timer);clearProcessedClipVideo();previewClip={generation,requestPending:true,context,job:null,stale:false,cancelWhenReturned:false,cancelBusy:false,timer:null,error:""};updateProcessedClipPanel();
    const data=new FormData();data.append("file",file,file.name);data.append("purpose","preview_clip");data.append("clip_start_seconds",String(context.startSeconds));data.append("clip_duration_seconds",String(duration));Object.entries(settings).forEach(([key,value])=>data.append(key,value));
    try{
      const raw=await request("/api/jobs",{method:"POST",body:data});let job;
      try{job=frontendContract.validatePreviewClipJob(raw);}catch(error){await cancelUnexpectedClipResponse(raw,generation,error.message);return;}
      if(generation!==previewClip.generation)return;
      previewClip.requestPending=false;previewClip.job=job;
      if(frontendContract.processedClipResponseAction(context,previewClipContext(),previewClip.cancelWhenReturned)==="cancel"||previewClip.stale){previewClip.stale=true;previewClip.cancelWhenReturned=false;previewClip.error="";clearProcessedClipVideo();void cancelProcessedClipJob(job.id,generation,"Canceled because the preview inputs changed");}
      else{updateProcessedClipPanel();if(["queued","processing"].includes(job.state))scheduleProcessedClipPoll(job.id,generation);}
    }catch(error){if(generation===previewClip.generation){previewClip.requestPending=false;previewClip.cancelWhenReturned=false;previewClip.error=`Could not start processed clip: ${error.message}`;updateProcessedClipPanel();}}
  }
  function scheduleProcessedClipPoll(jobId,generation){clearTimeout(previewClip.timer);previewClip.timer=setTimeout(()=>pollProcessedClip(jobId,generation),1200);}
  async function pollProcessedClip(jobId,generation){
    if(generation!==previewClip.generation)return;
    try{
      const data=await request("/api/jobs"),remote=Array.isArray(data)?data:(data.jobs||[]),raw=remote.find((job)=>job.id===jobId);if(!raw){scheduleProcessedClipPoll(jobId,generation);return;}
      if(previewClip.quarantine?.id===jobId){
        const quarantine=previewClip.quarantine;previewClip.quarantine=frontendContract.processedClipQuarantineAfterPoll(quarantine,raw);
        if(previewClip.quarantine){previewClip.quarantine.state=typeof raw.state==="string"?raw.state:previewClip.quarantine.state;updateProcessedClipPanel();scheduleProcessedClipPoll(jobId,generation);}
        else{previewClip.error=`${quarantine.message} The server reports terminal state: ${raw.state}.`;previewClip.errorKind="response";updateProcessedClipPanel();}
        return;
      }
      let job;try{job=frontendContract.validatePreviewClipJob(raw);}catch(error){await cancelUnexpectedClipResponse(raw,generation,error.message);return;}
      if(generation!==previewClip.generation)return;previewClip.job=job;previewClip.error=frontendContract.processedClipErrorAfterPoll(previewClip.error,previewClip.errorKind);if(previewClip.errorKind==="poll")previewClip.errorKind="";updateProcessedClipPanel();
      if(["queued","processing"].includes(job.state))scheduleProcessedClipPoll(jobId,generation);
    }catch(error){if(generation===previewClip.generation){if(previewClip.quarantine?.id===jobId){previewClip.quarantine.cancelError=`Could not confirm the server state: ${error.message} The job remains blocked; use Cancel clip to retry.`;updateProcessedClipPanel();scheduleProcessedClipPoll(jobId,generation);}else{previewClip.error=`Could not refresh processed clip: ${error.message}`;previewClip.errorKind="poll";updateProcessedClipPanel();scheduleProcessedClipPoll(jobId,generation);}}}
  }
  function calibrationContext(){const model=selectedModel();return {revision:previewRevision,sourceId:selectedUid,frameTime:previewFile&&!isImage(previewFile)?ui.sourceVideo.currentTime:0,settings:JSON.stringify(getSettings()),budget:Number(ui.calibrationBudget.value),modelIdentity:model?.sha256||model?.name||ui.depthMode.value,previewDetail:ui.previewDetail.value};}
  function fullStillBlock(){if(ui.previewDetail.value!=="full")return "";const source=previewFile&&isImage(previewFile)?ui.sourceImage:null;return frontendContract.fullStillEligibility({mediaType:source?"image":previewFile?"video":"none",width:source?.naturalWidth||0,height:source?.naturalHeight||0,fileSizeBytes:previewFile?.size}).reason;}
  function calibrationBlock(){if(calibrationBusy)return "Input-detail measurement is running.";if(previewBusy)return "Wait for the current preview to finish.";if(!previewFile||!sourceReady())return "Choose a decoded source frame to begin.";if(!/^depth-anything-v2-(small|base|large)$/.test(ui.depthMode.value))return "Choose a named Depth Anything model. MiDaS uses fixed input detail; visual approximation and custom models cannot be calibrated.";const model=selectedModel();if(!model||model.state!=="ready")return model?.state==="downloading"?"Wait for the selected model to finish installing.":`Install ${model?.name||"the selected model"} before calibrating.`;if(runtime&&!runtime.available)return "Install ONNX Runtime before calibrating.";if(gpuBlock())return gpuBlock();if(items.some((item)=>item.state==="processing"))return "Wait for active processing jobs to finish.";return "";}
  function invalidateCalibrationIfStale(){if(calibrationResult&&calibrationResultContext&&!frontendContract.calibrationContextMatches(calibrationResultContext,calibrationContext())){calibrationResult=null;calibrationResultContext=null;ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;calibrationNotice="Previous measurement hidden because the source frame, settings, or budget changed. Measure again for the current context.";ui.calibrationStatus.textContent=calibrationNotice;}}
  function refreshCalibrationControls(){invalidateCalibrationIfStale();const reason=calibrationBlock();ui.calibrateInput.disabled=Boolean(reason);ui.calibrationBudget.disabled=calibrationBusy;ui.calibrateInput.textContent=calibrationBusy?"Measuring input detail…":"Measure input detail";if(calibrationBusy)ui.calibrationStatus.textContent="Measuring depth-input inference. One in-flight inference may exceed the budget and cannot be interrupted…";else if(calibrationResult)ui.calibrationStatus.textContent=frontendContract.calibrationStatusLabel(calibrationResult);else if(reason)ui.calibrationStatus.textContent=reason;else ui.calibrationStatus.textContent=calibrationNotice||"Ready. The measurement is opt-in and will not change settings.";ui.applyCalibration.disabled=!calibrationResult||!frontendContract.calibrationApplicable(calibrationResult)||!frontendContract.calibrationContextMatches(calibrationResultContext,calibrationContext());}
  function renderCalibrationReport(report){const view=frontendContract.calibrationDetails(report);calibrationResult=report;calibrationResultContext=calibrationContext();calibrationNotice="";ui.calibrationSummary.textContent=view.summary;ui.calibrationCandidates.replaceChildren(...view.candidates.map((line)=>el("li","",line)));ui.calibrationScope.textContent=`Scope: ${view.scope}`;ui.calibrationLimitations.replaceChildren(...view.limitations.map((line)=>el("li","",line)));ui.calibrationStorageWarning.hidden=!report.storage_warning;ui.calibrationStorageWarning.textContent=typeof report.storage_warning==="string"?report.storage_warning.slice(0,240):"";ui.calibrationReport.hidden=false;ui.applyCalibration.hidden=!view.applicable;ui.applyCalibration.textContent=view.applicable?`Apply ${report.fastest_measured_input_size} px input detail`:"Apply measured input detail";ui.calibrationStatus.textContent=frontendContract.calibrationStatusLabel(report);refreshCalibrationControls();}
  function previewContentDimensions(){const stageW=ui.previewStage.clientWidth||1,stageH=ui.previewStage.clientHeight||1;if(previewView==="compare"&&previewResult)return {width:Number(previewResult.source_width)||stageW,height:Number(previewResult.source_height)||stageH};if(previewView==="original"&&previewFile)return {width:isImage(previewFile)?ui.sourceImage.naturalWidth:ui.sourceVideo.videoWidth,height:isImage(previewFile)?ui.sourceImage.naturalHeight:ui.sourceVideo.videoHeight};if(previewResult)return {width:Number(previewResult.width)||stageW,height:Number(previewResult.height)||stageH};return {width:stageW,height:stageH};}
  function applyZoomPan(zoom=zoomState.zoom,panX=zoomState.panX,panY=zoomState.panY){const viewportW=ui.previewStage.clientWidth||1,viewportH=ui.previewStage.clientHeight||1,content=previewContentDimensions();zoomState=frontendContract.zoomPanGeometry(zoom,panX,panY,viewportW,viewportH,content.width,content.height);ui.previewMedia.style.transform=`translate(${zoomState.panX}px, ${zoomState.panY}px) scale(${zoomState.zoom})`;ui.zoomLabel.textContent=`${Math.round(zoomState.zoom*100)}%`;ui.resetZoom.disabled=zoomState.zoom===1&&zoomState.panX===0&&zoomState.panY===0;ui.zoomOut.disabled=zoomState.zoom<=1;ui.zoomIn.disabled=zoomState.zoom>=4;ui.previewStage.classList.toggle("is-zoomed",zoomState.zoom>1);}
  function resetZoomPan(){applyZoomPan(1,0,0);}
  function updateCompareSplit(){compareSplit=frontendContract.compareSplitPercent(Number(ui.compareSlider.value));ui.compareSlider.value=String(compareSplit);ui.compareSlider.setAttribute("aria-valuetext",`${compareSplit}% depth visible`);ui.compareValue.textContent=`${compareSplit}% depth shown`;ui.compareDepth.style.clipPath=`inset(0 ${100-compareSplit}% 0 0)`;ui.compareDivider.style.left=`${compareSplit}%`;}
  function handlePreviewKeydown(event){if(event.ctrlKey||event.metaKey||event.altKey||event.target.closest("button,input,select,textarea,video"))return;if(event.key==="+"||event.key==="="||event.key==="Add"){event.preventDefault();applyZoomPan(zoomState.zoom+0.25);return;}if(event.key==="-"||event.key==="_"||event.key==="Subtract"){event.preventDefault();applyZoomPan(zoomState.zoom-0.25);return;}if(event.key==="0"){event.preventDefault();resetZoomPan();return;}if(zoomState.zoom>1&&["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key)){event.preventDefault();const step=36,dx=event.key==="ArrowLeft"?-step:event.key==="ArrowRight"?step:0,dy=event.key==="ArrowUp"?-step:event.key==="ArrowDown"?step:0;applyZoomPan(zoomState.zoom,zoomState.panX+dx,zoomState.panY+dy);}}
  function handlePreviewPointerDown(event){if(zoomState.zoom<=1||event.button!==0||event.target.closest("video,button,input,select,textarea"))return;panPointer={id:event.pointerId,x:event.clientX,y:event.clientY,panX:zoomState.panX,panY:zoomState.panY};ui.previewStage.classList.add("is-panning");ui.previewStage.setPointerCapture(event.pointerId);event.preventDefault();}
  function handlePreviewPointerMove(event){if(!panPointer||panPointer.id!==event.pointerId)return;applyZoomPan(zoomState.zoom,panPointer.panX+event.clientX-panPointer.x,panPointer.panY+event.clientY-panPointer.y);event.preventDefault();}
  function handlePreviewPointerEnd(event){if(!panPointer||panPointer.id!==event.pointerId)return;panPointer=null;ui.previewStage.classList.remove("is-panning");if(ui.previewStage.hasPointerCapture(event.pointerId))ui.previewStage.releasePointerCapture(event.pointerId);}
  function previewMessage(message, error = false) { ui.previewState.textContent = message; ui.previewState.classList.toggle("error", error); }
  function markPreviewStale() { previewRevision++; previewError="";invalidateProcessedClipIfStale();updatePreview(); }
  function updatePreview() {
    const original = previewView === "original";
    const comparing = previewView === "compare" && frontendContract.sourceDepthCompareAvailable(previewResult);
    ui.tabCompare.hidden=!frontendContract.sourceDepthCompareAvailable(previewResult);
    ui.sourceImage.hidden = !(original && previewFile && isImage(previewFile) && !sourceError);
    ui.sourceVideo.hidden = !(original && previewFile && !isImage(previewFile) && !sourceError);
    ui.processedImage.hidden = original || comparing || !previewResult;
    ui.sourceDepthCompare.hidden=!comparing;ui.compareControls.hidden=!comparing;
    ui.previewEmpty.hidden = (original ? Boolean(previewFile && !sourceError) : Boolean(previewResult));
    const emptyTitle = ui.previewEmpty.querySelector("strong"), emptyCopy = ui.previewEmpty.querySelector(":scope > span");
    if (sourceError) { emptyTitle.textContent = "Preview unavailable for this source"; emptyCopy.textContent = sourceError; }
    else if (!previewFile) { emptyTitle.textContent = selectedUid ? "Source preview is not retained after reload" : "Your next dimension starts here"; emptyCopy.textContent = selectedUid ? "Your export is still available in the queue. Add the original file again to preview it." : "Add a photo or video to preview a frame."; }
    else { emptyTitle.textContent = "Ready for a new perspective?"; emptyCopy.textContent = "Generate a real processed frame with your current settings."; }
    if (previewResult && !original && !comparing) { ui.processedImage.src = previewView === "depth" ? previewResult.depth : previewResult.image; ui.processedImage.alt = previewView === "depth" ? "Estimated depth map for the selected frame" : "Rendered stereo output for the selected frame"; }
    if(comparing){ui.compareSource.src=previewResult.source;ui.compareDepth.src=previewResult.depth;ui.compareResolution.textContent=`Comparison uses the exact ${frontendContract.previewDimensionDetails(previewResult)} frame sent to inference.`;}
    const detailBlock=fullStillBlock();ui.previewDetailHint.textContent=detailBlock||"Full-detail still sends the original file (up to 12 MP, 8192 px per side, and 24 MiB). Neural input remains subject to the hardware cap shown below.";
    ui.previewDetailHint.classList.toggle("preview-detail-warning",Boolean(detailBlock));
    ui.generatePreview.disabled = previewBusy || calibrationBusy || !sourceReady() || Boolean(detailBlock) || Boolean(readiness());
    ui.previewExecution.hidden = !previewResult;
    ui.previewExecution.textContent = previewResult ? `${resultRevision !== previewRevision ? "Previous preview · " : ""}${[frontendContract.previewDimensionDetails(previewResult),executionSummary(previewResult.execution) || "Device use not reported",frontendContract.previewExecutionDetails(previewResult)].filter(Boolean).join(" · ")}` : "";
    ui.generatePreview.textContent = previewBusy ? "Processing frame…" : previewResult && resultRevision !== previewRevision ? "↻ Update preview" : "✦ Generate preview";
    if (previewBusy) previewMessage("Processing one frame locally…");
    else if (detailBlock) previewMessage(`${detailBlock} Export remains available.`, true);
    else if (sourceError) previewMessage("Export is still available", true);
    else if (previewError) previewMessage(previewError, true);
    else if (previewResult && comparing) previewMessage(resultRevision === previewRevision ? `Source / depth comparison · ${frontendContract.previewDimensionDetails(previewResult)}` : "Previous source / depth result · update preview for the current frame");
    else if (previewResult) previewMessage(resultRevision === previewRevision ? `Processed ${frontendContract.previewDimensionDetails(previewResult)} · output ${previewResult.width} × ${previewResult.height} px` : "Settings or frame changed · update preview");
    else if (previewFile) previewMessage(sourceReady() ? "Ready to generate a frame preview" : "Loading source…");
    else previewMessage(selectedUid ? "Restored local job" : "Select a source to begin");
    ui.previewStage.setAttribute("aria-labelledby",previewView==="original"?"tabOriginal":previewView==="stereo"?"tabStereo":previewView==="depth"?"tabDepth":"tabCompare");
    updateCompareSplit();applyZoomPan();
    refreshCalibrationControls();updateProcessedClipPanel();
  }
  function setView(view) {
    if(view==="compare"&&!frontendContract.sourceDepthCompareAvailable(previewResult))view=previewResult?"stereo":"original";
    if (view !== "original") ui.sourceVideo.pause();
    previewView = view;
    document.querySelectorAll("[data-view]").forEach((button) => { const active=button.dataset.view === view; button.classList.toggle("active",active); button.setAttribute("aria-selected",String(active)); button.tabIndex=active ? 0 : -1; });
    updatePreview();
  }
  function selectItem(item) {
    selectedUid = item.uid;
    invalidateProcessedClipIfStale("Source changed");
    ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; previewFile = item.file || null; previewResult = null; sourceError = ""; previewError=""; previewRevision++;
    resetZoomPan();compareSplit=50;ui.compareSlider.value="50";updateCompareSplit();
    ui.sourceName.textContent = item.name;
    if (previewFile) { previewUrl=URL.createObjectURL(previewFile); (isImage(previewFile) ? ui.sourceImage : ui.sourceVideo).src=previewUrl; }
    setView("original"); updateQueue();
  }
  function addFiles(fileList) {
    let rejected=0; const added=[];
    for (const file of Array.from(fileList || [])) {
      if (!isImage(file) && !videoExtensions.has(extension(file))) { rejected++; continue; }
      const item={uid:crypto.randomUUID(),file,name:file.name,size:file.size,mediaType:isImage(file)?"image":"video",state:"ready",progress:0}; items.push(item); added.push(item);
    }
    if (added.length) { selectItem(added[0]); toast(`${added.length} file${added.length===1?"":"s"} added to your queue.`); }
    if (rejected) toast(`${rejected} unsupported file${rejected===1?"":"s"} skipped.`,true);
    ui.mediaInput.value=""; updateQueue();
  }
  function removeItem(item) {
    if (item.jobId) hiddenJobs.add(item.jobId);
    items=items.filter((x) => x.uid !== item.uid);
    if (selectedUid===item.uid) { selectedUid=null;invalidateProcessedClipIfStale("Source removed"); previewFile=null; previewResult=null; sourceError=""; previewError=""; resetZoomPan();compareSplit=50;ui.compareSlider.value="50";ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src"); if(previewUrl)URL.revokeObjectURL(previewUrl); previewUrl=null; ui.sourceName.textContent="No media selected"; updatePreview(); }
  }
  function action(text, handler) { const button=el("button","row-button",text); button.type="button"; button.addEventListener("click",handler); return button; }
  function queueAction(text, ariaLabel, key, handler, disabled=false) { const button=action(text,handler);button.setAttribute("aria-label",ariaLabel);button.dataset.actionKey=key;button.disabled=disabled||queueMutationBusy;return button; }
  function queueDisplayItems(){
    const rank=(item)=>item.state==="processing"||item.state==="uploading"?0:item.state==="queued"?1:!item.jobId&&item.state==="ready"?2:3;
    return items.map((item,index)=>({item,index})).sort((a,b)=>rank(a.item)-rank(b.item)||(rank(a.item)===1?(a.item.queuePosition??Number.MAX_SAFE_INTEGER)-(b.item.queuePosition??Number.MAX_SAFE_INTEGER):0)||a.index-b.index).map(({item})=>item);
  }
  function focusQueueItem(uid,key){setTimeout(()=>{const row=[...ui.queueRows.querySelectorAll(".queue-row")].find((node)=>node.dataset.itemUid===String(uid));const actions=row?[...row.querySelectorAll("[data-action-key]")]:[];const target=(row&&(actions.find((node)=>node.dataset.actionKey===key&&!node.disabled)||actions.find((node)=>!node.disabled)||row.querySelector(".file-name")))||ui.queueRows.querySelector(".file-name")||(!ui.applySettingsToReady.disabled?ui.applySettingsToReady:ui.clearQueueButton);target?.focus();},0);}
  function freezeReadyItem(item, settings=currentSettingsForPreset()){item.settingsSnapshot=frontendContract.validateSettings(settings);item.settingsFrozen=true;item.message="Ready with frozen settings";}
  function moveLocalReady(item,direction){const key=`move-ready-${item.uid}-${direction}`;items=frontendContract.moveReadyQueueItem(items,item.uid,direction);queueWorkflowMessage=`Moved ${item.name} ${direction<0?"earlier":"later"} in the local ready-file order.`;updateQueue(item.uid,key);}
  function applySettingsToReady(){const ready=items.filter((item)=>!item.jobId&&item.state==="ready");if(!ready.length)return;const snapshot=currentSettingsForPreset();ready.forEach((item)=>freezeReadyItem(item,snapshot));queueWorkflowMessage=`Applied and froze the current settings for ${ready.length} ready file${ready.length===1?"":"s"}.`;updateQueue();}
  function useJobSettings(item){if(!item.settingsSnapshot){toast(item.settingsSnapshotError||"A valid settings snapshot is not available from this server.",true);return;}assignSettings(item.settingsSnapshot);performanceBaseline={...item.settingsSnapshot};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();renderModels();toast(`Loaded settings from ${item.name}. No job was started.`);updateQueue(item.uid,`use-snapshot-${item.uid||item.jobId}`);}
  function addQueueCopy(item,settings,kind){try{const copy=frontendContract.makeReadyQueueCopy(item,settings);copy.uid=crypto.randomUUID();items.splice(items.indexOf(item)+1,0,copy);selectedUid=copy.uid;selectItem(copy);queueWorkflowMessage=`${kind} added as a new ready file. Review it and choose Convert queue when ready.`;updateQueue(copy.uid,`use-current-${copy.uid}`);}catch(error){toast(error.message,true);}}
  function retryQueueItem(item){if(!item.file){toast("The original source is unavailable in this browser tab. Reselect the file to retry.",true);return;}addQueueCopy(item,item.settingsSnapshot||currentSettingsForPreset(),"Retry");}
  function duplicateQueueItem(item){if(!item.file){toast("The original source is unavailable in this browser tab. Reselect the file to duplicate it.",true);return;}addQueueCopy(item,currentSettingsForPreset(),"Duplicate");}
  function eligibleQueuedExports(){return items.filter((item)=>item.jobId&&(item.purpose==null||item.purpose==="export")&&item.state==="queued"&&Number.isSafeInteger(item.queuePosition)&&item.queuePosition>0).map((item)=>({id:item.jobId,purpose:item.purpose,state:item.state,queue_position:item.queuePosition}));}
  async function queueMutationFailed(error,description){
    if(error.status===409){try{await refreshQueueFromServer();queueWorkflowMessage=`${description} was not applied because work started or the queue changed. The queue was refreshed; review it and try again.`;}catch(refreshError){queueWorkflowMessage=`${description} was not applied because work started or the queue changed. Queue refresh failed: ${refreshError.message}`;}}
    else queueWorkflowMessage=`${description} failed: ${error.message}`;
    updateQueue();
  }
  async function reorderQueuedItem(item,direction,focusKey){
    if(!frontendContract.queueCapabilities(status)||queueMutationBusy)return;
    const ordered=frontendContract.moveQueuedExportIds(eligibleQueuedExports(),item.jobId,direction);if(ordered.length<2)return;
    queueMutationBusy=true;queueWorkflowMessage="Updating the server queue order…";updateQueue(item.uid,focusKey);
    try{const response=await request("/api/queue/reorder",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_ids:ordered})});if(!Array.isArray(response.job_ids)||response.job_ids.length!==ordered.length||response.job_ids.some((id,index)=>id!==ordered[index]))throw new Error("The server did not confirm the requested queue order.");const positions=eligibleQueuedExports().map((job)=>job.queue_position).sort((a,b)=>a-b);ordered.forEach((id,index)=>{const queued=items.find((entry)=>entry.jobId===id);if(queued)queued.queuePosition=positions[index];});await pollJobs();queueWorkflowMessage="Server queue order updated.";}
    catch(error){await queueMutationFailed(error,"Queue reorder");}
    finally{queueMutationBusy=false;updateQueue(item.uid,focusKey);}
  }
  async function updateQueuedSettings(item,focusKey){
    if(!frontendContract.queueCapabilities(status)||queueMutationBusy)return;
    const settings=currentSettingsForPreset();queueMutationBusy=true;queueWorkflowMessage=`Updating settings for ${item.name}…`;updateQueue(item.uid,focusKey);
    try{const job=await request(`/api/jobs/${encodeURIComponent(item.jobId)}/settings`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({settings})});mergeJob(item,job);queueWorkflowMessage=`Updated the queued settings for ${item.name}.`;
    }catch(error){await queueMutationFailed(error,"Queued settings update");}
    finally{queueMutationBusy=false;updateQueue(item.uid,focusKey);}
  }
  async function removeQueuedItem(item,focusKey){
    if(!frontendContract.queueCapabilities(status)||queueMutationBusy)return;
    queueMutationBusy=true;queueWorkflowMessage=`Removing ${item.name} from the server queue…`;updateQueue(item.uid,focusKey);
    try{await request(`/api/jobs/${encodeURIComponent(item.jobId)}/remove-queued`,{method:"POST"});hiddenJobs.add(item.jobId);saveHidden();items=items.filter((entry)=>entry!==item);queueWorkflowMessage=`Removed ${item.name} from the server queue.`;}
    catch(error){await queueMutationFailed(error,"Queued removal");}
    finally{queueMutationBusy=false;updateQueue(item.uid,focusKey);}
  }
  function updateQueue(focusUid=null,focusKey=null) {
    if(!focusUid&&ui.queueRows.contains(document.activeElement)){const active=document.activeElement,row=active.closest(".queue-row");if(row){focusUid=row.dataset.itemUid;focusKey=active.dataset.actionKey||null;}}
    const pending=items.filter((x)=>!x.jobId&&x.state==="ready"),block=readiness(),caps=frontendContract.queueCapabilities(status);
    const videoBlocked=status&&!status.ffmpeg_found&&pending.some((x)=>x.mediaType==="video");
    ui.queueCount.textContent=String(items.length);ui.queueHeadingCount.textContent=`(${items.length})`;ui.convertCount.textContent=String(pending.length);ui.queueEmpty.hidden=items.length>0;
    ui.convertButton.disabled=!pending.length||submitting||Boolean(block)||Boolean(videoBlocked);ui.convertButton.firstElementChild.textContent=submitting?"Adding to local queue…":"✦ Convert queue";
    ui.convertHint.textContent=block||(videoBlocked?"FFmpeg is needed for the videos in your queue.":pending.length?`${pending.length} file${pending.length===1?"":"s"} ready. Unfrozen files use the visible settings when submitted.`:"Add media to start a conversion.");
    ui.applySettingsToReady.disabled=!pending.length;ui.queueWorkflowStatus.textContent=queueWorkflowMessage||(caps?"Ready files use the visible settings at submission unless frozen. Server queue edits apply only to pending exports; clip slots stay in place.":"Ready files use the visible settings at submission unless frozen. This server does not advertise queue-edit capability.");
    ui.queueRows.replaceChildren();
    const queuedExports=eligibleQueuedExports(),displayItems=queueDisplayItems();
    for(const item of displayItems){
      const identity=String(item.uid||item.jobId),row=el("article",`queue-row${item.uid===selectedUid?" selected":""}`);row.dataset.itemUid=identity;
      const icon=el("div","file-icon",item.mediaType==="image"?"▧":"▶"),copy=el("div","file-copy"),name=el("button","file-name",item.name);name.type="button";name.dataset.actionKey=`select-${identity}`;name.title=item.name;name.setAttribute("aria-label",`Select ${item.name}`);name.setAttribute("aria-pressed",String(item.uid===selectedUid));name.addEventListener("click",()=>selectItem(item));
      const meta=el("div","file-meta"),state=el("span",`file-status ${item.state}`,item.message||({ready:"Ready to convert",uploading:"Uploading locally",completed:"Complete",failed:"Failed",canceled:"Canceled"}[item.state]||item.state));meta.append(state);
      const info=[formatBytes(item.file?.size||item.size),item.layout].filter(Boolean).join(" · ");if(info)meta.append(el("span","",info));copy.append(name,meta);
      const execution=executionSummary(item.execution,item.pipeline);if(execution)copy.append(el("div","queue-execution",execution));else if(item.executionMode)copy.append(el("div","queue-execution",`Requested: ${{auto:"Auto",hybrid:"GPU + CPU",gpu:"NVIDIA GPU",cpu:"CPU only"}[item.executionMode]||item.executionMode} · awaiting execution`));
      const runtimeDetails=progressDetails(item);if(runtimeDetails.length)copy.append(el("div","queue-runtime",runtimeDetails.join(" · ")));
      const preflight=frontendContract.preflightDetails(item.preflight,formatBytes);if(preflight.summary){const budget=el("div","queue-preflight");budget.append(el("div","queue-preflight-summary",preflight.summary));if(preflight.visibleWarning)budget.append(el("small","queue-preflight-warning",preflight.visibleWarning));const extraWarnings=preflight.warnings.filter((warning)=>warning!==preflight.visibleWarning);if(preflight.basis||extraWarnings.length){const details=el("details","queue-preflight-details");details.append(el("summary","","Estimate details"));if(preflight.basis)details.append(el("small","queue-preflight-basis",`Estimate basis: ${preflight.basis}`));for(const warning of extraWarnings)details.append(el("small","queue-preflight-warning",warning));budget.append(details);}copy.append(budget);}
      if(item.settingsSnapshot)copy.append(el("small","queue-settings-detail",`${item.jobId?"Server snapshot":item.settingsFrozen?"Frozen settings":"Unfrozen · uses current settings at submit"}: ${settingsDescription(item.settingsSnapshot)}`));
      else if(!item.jobId&&item.state==="ready")copy.append(el("small","queue-settings-detail","Unfrozen · uses the visible settings when you submit this file."));
      else if(item.jobId)copy.append(el("small","queue-settings-detail",item.settingsSnapshotError||"Settings snapshot unavailable; use a newer server for queue editing."));
      if(item.jobId&&!item.file)copy.append(el("small","queue-source-detail",item.sourceAvailable===true?"Server has a readable working upload; this browser has no retry source file.":"Source is not retained for retry. Reselect the original file; filenames are not matched automatically."));
      else if(!item.jobId&&item.file)copy.append(el("small","queue-source-detail","Original file is held in this browser tab."));
      const actions=el("div","row-actions");
      if(activeStates.has(item.state)){const progress=el("div","row-progress"),bar=el("span");progress.setAttribute("role","progressbar");progress.setAttribute("aria-label",`Conversion progress for ${item.name}`);progress.setAttribute("aria-valuemin","0");progress.setAttribute("aria-valuemax","100");progress.setAttribute("aria-valuenow",String(item.progress||0));bar.style.width=`${Math.max(2,Math.min(100,item.progress||0))}%`;progress.append(bar);actions.append(progress);}
      if(item.state==="completed"&&item.jobId){const link=el("a","row-button","Download ↓");link.href=`/api/jobs/${encodeURIComponent(item.jobId)}/download`;link.download=item.outputName||"";actions.append(link);}
      if(item.jobId&&["queued","processing"].includes(item.state))actions.append(queueAction("Cancel",`Cancel active job ${item.name}`,`cancel-${identity}`,()=>cancelJob(item)));
      if(!item.jobId&&item.state==="ready"){
        const ready=items.filter((entry)=>!entry.jobId&&entry.state==="ready"),position=ready.indexOf(item);
        actions.append(queueAction("Move up",`Move ${item.name} earlier among ready files`,`move-ready-${item.uid}--1`,()=>moveLocalReady(item,-1),position===0));
        actions.append(queueAction("Move down",`Move ${item.name} later among ready files`,`move-ready-${item.uid}-1`,()=>moveLocalReady(item,1),position===ready.length-1));
        actions.append(queueAction("Use current settings",`Freeze current settings for ${item.name}`,`use-current-${identity}`,()=>{freezeReadyItem(item);queueWorkflowMessage=`Froze the visible settings for ${item.name}.`;updateQueue(item.uid,`use-current-${identity}`);}));
      }
      if(item.jobId&&caps&&item.state==="queued"&&item.queuePosition!==null){
        const remoteIndex=queuedExports.findIndex((job)=>job.id===item.jobId);
        actions.append(queueAction("Move up",`Move queued export ${item.name} earlier`,`move-remote-${identity}--1`,()=>reorderQueuedItem(item,-1,`move-remote-${identity}--1`),remoteIndex<=0));
        actions.append(queueAction("Move down",`Move queued export ${item.name} later`,`move-remote-${identity}-1`,()=>reorderQueuedItem(item,1,`move-remote-${identity}-1`),remoteIndex<0||remoteIndex===queuedExports.length-1));
        actions.append(queueAction("Use current settings",`Apply visible settings to queued export ${item.name}`,`use-current-${identity}`,()=>updateQueuedSettings(item,`use-current-${identity}`)));
        actions.append(queueAction("Remove queued",`Remove ${item.name} from the server queue`,`remove-queued-${identity}`,()=>removeQueuedItem(item,`remove-queued-${identity}`)));
      }
      if(item.settingsSnapshot)actions.append(queueAction("Use settings",`Load settings snapshot from ${item.name} into the controls`,`use-snapshot-${identity}`,()=>useJobSettings(item)));
      if(["failed","canceled"].includes(item.state)&&item.file&&item.settingsSnapshot)actions.append(queueAction("Retry as new job",`Retry ${item.name} as a new job using its original settings`,`retry-${identity}`,()=>retryQueueItem(item)));
      if(item.jobId&&item.file)actions.append(queueAction("Duplicate with current settings",`Create a new ready copy of ${item.name} with current settings`,`duplicate-${identity}`,()=>duplicateQueueItem(item)));
      if(!activeStates.has(item.state))actions.append(queueAction("Remove",`Remove ${item.name} from this browser queue`,`remove-local-${identity}`,()=>{removeItem(item);saveHidden();updateQueue(identity,`remove-local-${identity}`);}));
      row.append(icon,copy,actions);if(item.error)row.append(el("div","job-error",item.error));ui.queueRows.append(row);
    }
    refreshCalibrationControls();if(focusUid)focusQueueItem(focusUid,focusKey);
  }
  function saveHidden(){try{sessionStorage.setItem("stereolab.hiddenJobs",JSON.stringify([...hiddenJobs]));}catch{/* Keep this page usable without storage. */}}
  function getSettings(){const mode=ui.depthMode.value;const inputSize=mode==="dpt-hybrid-midas"?"384":mode==="precision"||/^depth-anything-v2-(small|base|large)$/.test(mode)?ui.modelSize.value:"518";return {layout:ui.layoutSelect.value,strength:ui.strength.value,convergence:ui.convergence.value,depth_mode:mode,execution_mode:ui.executionMode.value,input_size:inputSize,resolution:ui.resolution.value,quality:ui.quality.value,temporal_smoothing:ui.temporalSmoothing.value,edge_refinement:String(ui.edgeRefinement.checked),invert_depth:String(ui.invertDepth.checked)};}
  function currentSettingsForPreset(){const raw=getSettings();return {layout:raw.layout,strength:Number(raw.strength),convergence:Number(raw.convergence),depth_mode:raw.depth_mode,execution_mode:raw.execution_mode,input_size:Number(raw.input_size),resolution:raw.resolution,quality:raw.quality,temporal_smoothing:Number(raw.temporal_smoothing),edge_refinement:ui.edgeRefinement.checked,invert_depth:ui.invertDepth.checked};}
  function settingsDescription(settings){const names={SBS:"Side by side · full",HALFSBS:"Side by side · half",TB:"Top / bottom · full",HALFTB:"Top / bottom · half",CROSS:"Cross-eye",ANAGLYPH:"Anaglyph · red / cyan",DEPTH:"Depth map",RGBD:"RGB + depth"};const models={"depth-anything-v2-small":"Small","depth-anything-v2-base":"Base","depth-anything-v2-large":"Large","dpt-hybrid-midas":"DPT Hybrid · MiDaS",balanced:"Visual approximation",precision:"Custom ONNX"};const devices={auto:"Auto",hybrid:"GPU + CPU · hybrid",gpu:"NVIDIA GPU",cpu:"CPU only"};const sizes=settings.depth_mode==="dpt-hybrid-midas"?"fixed 384 px":`${settings.input_size} px`;return `${names[settings.layout]} · strength ${Number(settings.strength).toFixed(1)} · convergence ${Number(settings.convergence).toFixed(2)} · ${models[settings.depth_mode]} · ${sizes} · ${devices[settings.execution_mode]} · ${settings.resolution==="source"?"Match source":"Up to 1080p"} · ${settings.quality} encoding · smoothing ${Number(settings.temporal_smoothing).toFixed(2)} · edge refinement ${settings.edge_refinement?"on":"off"} · invert ${settings.invert_depth?"on":"off"}`;}
  function refreshPerformancePresetStatus(){const current=currentSettingsForPreset(),dirty=frontendContract.hasUnsavedSettings(performanceBaseline,current);ui.performancePresetStatus.textContent=`${importedPresetBaseline?"Imported settings":ui.performancePreset.options[ui.performancePreset.selectedIndex]?.text || "Custom settings"} · ${dirty?"Unsaved changes":"Saved preset"}`;ui.performancePresetSummary.textContent=`Exact settings: ${settingsDescription(current)}`;ui.resetPerformancePreset.disabled=ui.performancePreset.value==="custom";}
  function assignSettings(settings){ui.layoutSelect.value=settings.layout;ui.strength.value=String(settings.strength);ui.convergence.value=String(settings.convergence);ui.depthMode.value=settings.depth_mode;ui.executionMode.value=settings.execution_mode;ui.modelSize.value=String(settings.input_size);ui.resolution.value=settings.resolution;ui.quality.value=settings.quality;ui.temporalSmoothing.value=String(settings.temporal_smoothing);ui.edgeRefinement.checked=settings.edge_refinement;ui.invertDepth.checked=settings.invert_depth;}
  function applyPerformancePreset(name){const settings=frontendContract.performancePreset(name);if(!settings)return;importedPresetBaseline=false;performanceBaseline=settings;assignSettings(settings);ui.performancePreset.value=name;ui.presetSelect.value="custom";updateSettings();}
  function resetPerformancePreset(){if(ui.performancePreset.value==="custom")return;applyPerformancePreset(ui.performancePreset.value);}
  function exportPresetFile(){try{const preset=frontendContract.exportUserPreset(currentSettingsForPreset());const blob=new Blob([JSON.stringify(preset,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="stereolab-settings-v1.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),0);toast("Settings preset exported.");}catch(error){toast(error.message,true);}}
  async function importPresetFile(file){if(!file)return;try{const parsed=JSON.parse(await file.text()),settings=frontendContract.validateUserPreset(parsed);assignSettings(settings);performanceBaseline={...settings};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();toast("Settings imported and validated. No model was installed or downloaded.");}catch(error){toast(`Preset not applied: ${error.message}`,true);}finally{ui.importPresetFile.value="";}}
  async function measureInputDetail(){const blocked=calibrationBlock();if(blocked){toast(blocked,true);return;}ui.sourceVideo.pause();const context=calibrationContext();calibrationBusy=true;calibrationResult=null;calibrationResultContext=null;calibrationNotice="";ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.calibrationStatus.textContent="Measuring depth-input inference. One in-flight inference may exceed the selected budget…";updatePreview();let discarded=false;
    try{const source=isImage(previewFile)?ui.sourceImage:ui.sourceVideo,w=source.naturalWidth||source.videoWidth,h=source.naturalHeight||source.videoHeight;if(!w||!h)throw new Error("The selected source has no decoded frame.");const scale=Math.min(1,960/w,540/h),canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));canvas.getContext("2d").drawImage(source,0,0,canvas.width,canvas.height);const blob=await new Promise((resolve,reject)=>canvas.toBlob((value)=>value?resolve(value):reject(new Error("Could not capture this frame.")),"image/jpeg",.94));if(!frontendContract.calibrationContextMatches(context,calibrationContext())){discarded=true;calibrationNotice="Measurement discarded because the source frame, settings, or budget changed before it started.";return;}const data=new FormData();data.append("file",blob,"calibration-frame.jpg");data.append("depth_mode",ui.depthMode.value);data.append("execution_mode",ui.executionMode.value);data.append("budget_seconds",String(context.budget));const report=frontendContract.validateCalibrationReport(await request("/api/calibration",{method:"POST",body:data}));if(!frontendContract.calibrationContextMatches(context,calibrationContext())){discarded=true;calibrationNotice="Measurement discarded because the source frame, settings, or budget changed while it was running.";return;}renderCalibrationReport(report);}
    catch(error){calibrationNotice=`Measurement failed: ${String(error.message||error).slice(0,260)}`;ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.calibrationStatus.textContent=calibrationNotice;toast(calibrationNotice,true);}
    finally{calibrationBusy=false;updatePreview();if(discarded)ui.calibrationStatus.textContent=calibrationNotice;}}
  function applyMeasuredInputDetail(){if(!calibrationResult||!frontendContract.calibrationApplicable(calibrationResult)||!frontendContract.calibrationContextMatches(calibrationResultContext,calibrationContext())){invalidateCalibrationIfStale();ui.applyCalibration.hidden=true;return;}const size=calibrationResult.fastest_measured_input_size;calibrationResult=null;calibrationResultContext=null;ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.modelSize.value=String(size);calibrationNotice=`Applied ${size} px from this one-frame depth-input measurement. Lower detail can lose fine edges.`;updateSettings();ui.calibrationStatus.textContent=calibrationNotice;toast(calibrationNotice);}
  function updateSettings(stale=true) {
    renderDeviceNote();
    const layout=layouts[ui.layoutSelect.value];ui.formatHint.textContent=layout[1];ui.previewModeLabel.textContent=layout[0].toUpperCase();
    ui.strengthValue.value=Number(ui.strength.value).toFixed(1);ui.convergenceValue.value=Number(ui.convergence.value).toFixed(2);ui.smoothingValue.value=Number(ui.temporalSmoothing.value).toFixed(2);
    const mode=ui.depthMode.value,model=selectedModel();const selectableInputSize=mode==="precision"||/^depth-anything-v2-(small|base|large)$/.test(mode);ui.modelRow.hidden=mode!=="precision";ui.modelSizeRow.hidden=!selectableInputSize;ui.modelSizeLabel.textContent=mode==="precision"?"Custom model input size":"Neural input detail";ui.midasSizeNote.hidden=mode!=="dpt-hybrid-midas";
    ui.installSelected.hidden=!model || ["ready","downloading"].includes(model.state);ui.installSelected.disabled=false;
    ui.depthNote.replaceChildren();
    if(mode==="balanced")ui.depthNote.textContent="Uses luminance and center cues. This is a visual approximation, not neural AI depth.";
    else if(mode==="precision")ui.depthNote.textContent="Uses your compatible ONNX depth model locally. Input shape and preprocessing must match your model.";
    else if(model){ui.depthNote.textContent=model.state==="ready"?`${model.name} is installed locally. ${runtime?.available?"ONNX Runtime ready.":"Checking runtime."}`:model.state==="downloading"?`Downloading ${model.name} · ${Math.round(model.progress||0)}%`:`${model.name} needs a one-time download (${model.size_mb} MB).`;if(model.license)ui.depthNote.append(document.createTextNode(` License: ${model.license}.`));}
    else ui.depthNote.textContent=catalogError?`Could not load models: ${catalogError}`:"Checking installed models…";
    if(stale)markPreviewStale();else updatePreview();updateQueue();refreshPerformancePresetStatus();
  }
  function applyPreset(){const preset=presets[ui.presetSelect.value];if(!preset)return;ui.strength.value=preset.strength;ui.convergence.value=preset.convergence;ui.temporalSmoothing.value=preset.smoothing;ui.edgeRefinement.checked=preset.edges;ui.invertDepth.checked=preset.invert;updateSettings();}
  async function generatePreview(){
    if(previewBusy || !sourceReady())return;const block=readiness();if(block){toast(block,true);return;}
    ui.sourceVideo.pause();const revision=previewRevision,settings=getSettings();previewError="";previewBusy=true;updatePreview();
    try{
      const source=isImage(previewFile)?ui.sourceImage:ui.sourceVideo;const w=source.naturalWidth||source.videoWidth,h=source.naturalHeight||source.videoHeight,detail=ui.previewDetail.value;
      const eligibility=frontendContract.fullStillEligibility({mediaType:isImage(previewFile)?"image":"video",width:w,height:h,fileSizeBytes:previewFile.size});if(detail==="full"&&!eligibility.eligible)throw new Error(eligibility.reason);
      const data=new FormData();
      if(detail==="full")data.append("file",previewFile,previewFile.name);
      else {const scale=Math.min(1,960/w,540/h),canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));canvas.getContext("2d").drawImage(source,0,0,canvas.width,canvas.height);const blob=await new Promise((resolve,reject)=>canvas.toBlob((value)=>value?resolve(value):reject(new Error("Could not capture this frame.")),"image/jpeg",.94));data.append("file",blob,"preview-frame.jpg");}
      data.append("preview_detail",detail);Object.entries(settings).forEach(([key,value])=>data.append(key,value));
      const result=frontendContract.validatePreviewResponse(await request("/api/preview",{method:"POST",body:data}),detail);
      if(revision!==previewRevision){toast("The source or settings changed. Generate again for the current frame.");return;}
      previewResult={...result,selected_source_width:w,selected_source_height:h};resultRevision=revision;resetZoomPan();compareSplit=50;ui.compareSlider.value="50";updateCompareSplit();setView(previewView==="depth"?"depth":"stereo");
    }catch(error){if(revision===previewRevision){previewError=error.message;previewMessage(error.message,true);toast(error.message,true);}}finally{previewBusy=false;updatePreview();await loadStatus();}
  }
  function renderRuntime(){
    if(!runtime)return;
    const providers=(runtime.providers || []).join(", ") || "none reported";
    const capability=runtime.gpu_available?`${runtime.gpu_name || "NVIDIA GPU"} available · ${runtime.gpu_verified ? "GPU inference verified" : "GPU inference not yet verified"}`:"GPU unavailable in this runtime";
    const actual=runtime.last_execution?.requested_mode ? `Last execution: ${executionSummary(runtime.last_execution)}` : runtime.active_provider ? `Last provider: ${runtime.active_provider}` : "No inference has run in this runtime";
    ui.runtimeDescription.textContent=runtime.available?`ONNX Runtime ${runtime.version || ""} ready. Providers: ${providers}. ${capability}. ${actual}.${runtime.error ? ` ${runtime.error}` : ""}`:`ONNX Runtime is missing. Install the project dependencies, then restart StereoLab.${runtime.error ? ` ${runtime.error}` : ""}`;
    ui.runtimeCommand.hidden=Boolean(runtime.available);
  }
  async function loadStatus(){try{status=await request("/api/status");statusLoadError="";customModelAvailable=Boolean(status.model_available);ui.modelState.textContent=customModelAvailable?(status.model_name||"Custom model loaded"):"No model loaded";runtime=status.runtime_info||status.runtime||runtime;ui.enginePill.classList.toggle("offline",!status.ffmpeg_found);ui.engineText.textContent=status.ffmpeg_found?"Local video tools ready":"FFmpeg needed for video";ui.toolDescription.textContent=status.ffmpeg_found?"FFmpeg ready for local video exports.":"FFmpeg is missing. Photo export remains available.";ui.toolPath.textContent=status.ffmpeg_path||"Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe.";renderRuntime();updateSettings(false);}catch(error){statusLoadError=error.message;ui.enginePill.classList.add("offline");ui.engineText.textContent="Local server unavailable";ui.toolDescription.textContent=error.message;toast("Could not connect to the local server.",true);updateProcessedClipPanel();}}
  async function retryGpuAction(){
    if(!frontendContract.retrySupported(runtime)||ui.retryGpu.disabled)return;
    ui.retryGpu.disabled=true;ui.retryGpu.textContent="Resetting GPU retry…";
    try{await frontendContract.retryGpu(request,loadStatus);toast("GPU retry is ready. Runtime status refreshed.");}
    catch(error){toast(error.message,true);}
    finally{ui.retryGpu.disabled=false;ui.retryGpu.textContent="Retry GPU";renderDeviceNote();}
  }
  function sourceLink(url,label){const link=el("a","",label);try{const parsed=new URL(url);if(["http:","https:"].includes(parsed.protocol)){link.href=parsed.href;link.target="_blank";link.rel="noopener noreferrer";return link;}}catch{/* Skip unsupported URLs. */}return el("span","",label);}
  function renderModels(){
    ui.modelCards.replaceChildren();
    if(!models.length){ui.modelCards.append(el("p","field-hint",catalogError||"No models are available in the catalog."));return;}
    for(const model of models){const card=el("article",`model-card${model.id===ui.depthMode.value?" is-selected":""}`);if(model.recommended)card.append(el("span","model-badge","RECOMMENDED"));card.append(el("h3","",model.name),el("p","",model.description));const license=el("div","model-license",`${model.size_mb} MB · ${model.license || "See model license"} · `);license.append(sourceLink(model.source_url,"Source & license ↗"));card.append(license);
      const text=model.state==="ready"?"● Installed locally":model.state==="downloading"?`Downloading · ${Math.round(model.progress||0)}%`:model.state==="failed"?"Download failed":"Not installed";card.append(el("div",`model-status ${model.state}`,text));if(model.error)card.append(el("p","model-error",model.error));
      const button=el("button","outline-button",model.state==="ready"?(ui.depthMode.value===model.id?"Selected ✓":"Use this model"):model.state==="downloading"?"Installing…":model.state==="failed"?"Retry installation":"Install model ↓");button.type="button";button.disabled=model.state==="downloading"||model.state==="ready"&&ui.depthMode.value===model.id;button.addEventListener("click",()=>{if(model.state==="ready"){ui.depthMode.value=model.id;updateSettings();renderModels();}else installModel(model.id);});card.append(button);ui.modelCards.append(card);
    }
  }
  async function loadModels(){
    if(modelsBusy)return;modelsBusy=true;
    try{const data=await request("/api/models");models=Array.isArray(data.models)?data.models:[];runtime=data.runtime_info||data.runtime||runtime;catalogError="";renderModels();renderRuntime();updateSettings(false);}
    catch(error){catalogError=error.message;renderModels();updateSettings(false);}
    finally{modelsBusy=false;clearTimeout(modelTimer);modelTimer=models.some((model)=>model.state==="downloading")?setTimeout(loadModels,1500):null;}
  }
  async function installModel(id){
    ui.installSelected.disabled=true;
    try{const model=await request(`/api/models/${encodeURIComponent(id)}/install`,{method:"POST"});const index=models.findIndex((x)=>x.id===id);if(index>=0&&model.id)models[index]=model;toast("Model installation started. You can follow progress in the model library.");renderModels();updateSettings(false);await loadModels();}
    catch(error){toast(error.message,true);}finally{ui.installSelected.disabled=false;}
  }
  async function uploadModel(file){if(!file)return;ui.modelButton.disabled=true;ui.modelState.textContent="Checking ONNX model…";const data=new FormData();data.append("model",file);try{const result=await request("/api/model",{method:"POST",body:data});customModelAvailable=Boolean(result.model_available);ui.modelState.textContent=result.model_name||file.name;markPreviewStale();toast("Custom ONNX model loaded locally.");}catch(error){ui.modelState.textContent=customModelAvailable?"Previous model retained":"No model loaded";toast(error.message,true);}finally{ui.modelButton.disabled=false;ui.modelInput.value="";updateSettings(false);}}
  function mergeJob(item,job){
    const metrics=["eta_seconds","eta_status","eta_range_seconds","elapsed_seconds","processing_fps","frames_processed","total_frames","encoder","phase","retry_count","retry_reason","preflight"];
    const reported=Object.fromEntries(metrics.filter((key)=>Object.hasOwn(job,key)).map((key)=>[key,job[key]]));let snapshot=item.settingsSnapshot?{settings:item.settingsSnapshot,sourceAvailable:item.sourceAvailable??Boolean(item.file),queuePosition:item.queuePosition??null}:null,snapshotError=item.settingsSnapshotError||"";
    if(job.settings_schema_version!=null||job.settings!=null){snapshot=null;snapshotError="";try{snapshot=frontendContract.validateQueueJobSnapshot(job);}catch(error){snapshotError=error.message;}}
    Object.assign(item,{jobId:job.id,purpose:job.purpose,name:job.filename||item.name,mediaType:job.media_type||item.mediaType,state:job.state,progress:job.progress,message:job.message,error:job.error,outputName:job.output_name,layout:job.layout,execution:job.execution,executionMode:job.execution_mode,pipeline:job.pipeline,settingsSchemaVersion:job.settings_schema_version??null,settingsSnapshot:snapshot?.settings||null,settingsSnapshotError:snapshotError,settingsFrozen:Boolean(snapshot),sourceAvailable:snapshot?.sourceAvailable??null,queuePosition:snapshot?.queuePosition??null},reported);
  }
  function mergeRemoteJobs(remote){for(const job of remote){if(hiddenJobs.has(job.id)||!frontendContract.isOrdinaryExportJob(job))continue;let item=items.find((x)=>x.jobId===job.id);if(!item){item={uid:job.id};items.push(item);}mergeJob(item,job);}updateQueue();}
  async function refreshQueueFromServer(){const data=await request("/api/jobs"),remote=Array.isArray(data)?data:(data.jobs||[]);mergeRemoteJobs(remote);}
  async function submitItem(item,settings){
    item.settingsSnapshot=frontendContract.validateSettings(settings);item.settingsFrozen=true;item.state="uploading";item.message="Adding to local queue";item.error=null;updateQueue();
    const data=new FormData();data.append("file",item.file,item.name);Object.entries(item.settingsSnapshot).forEach(([key,value])=>data.append(key,value));
    try{const job=await request("/api/jobs",{method:"POST",body:data});mergeJob(item,job);items=items.filter((x)=>x===item||x.jobId!==job.id);}
    catch(error){item.state="failed";item.error=error.message;item.message="Could not start conversion";}
    updateQueue();
  }
  async function convertPending(){if(submitting)return;const block=readiness();if(block){toast(block,true);return;}const pending=items.filter((x)=>!x.jobId&&x.state==="ready");if(!pending.length)return;const settings=currentSettingsForPreset();submitting=true;updateQueue();try{for(const item of pending){if(items.includes(item)&&item.state==="ready")await submitItem(item,item.settingsSnapshot||settings);}}finally{submitting=false;updateQueue();await pollJobs();}}
  async function pollJobs(){
    if(jobsBusy)return;jobsBusy=true;clearTimeout(jobTimer);jobTimer=null;
    try{await refreshQueueFromServer();}
    catch(error){ui.queueEmpty.hidden=items.length>0;if(!items.length){ui.queueEmpty.querySelector("strong").textContent="Could not restore the local queue";ui.queueEmpty.querySelector("small").textContent=error.message;}else toast(`Queue refresh failed: ${error.message}`,true);}
    finally{jobsBusy=false;if(items.some((x)=>["queued","processing"].includes(x.state)))jobTimer=setTimeout(pollJobs,1500);}
  }
  async function cancelJob(item){try{await request(`/api/jobs/${encodeURIComponent(item.jobId)}/cancel`,{method:"POST"});item.message="Cancellation requested";updateQueue();await pollJobs();}catch(error){toast(error.message,true);}}
  ui.browseButton.addEventListener("click",(event)=>{event.stopPropagation();ui.mediaInput.click();});ui.dropZone.addEventListener("click",(event)=>{if(event.target!==ui.browseButton)ui.mediaInput.click();});
  ui.dropZone.addEventListener("keydown",(event)=>{if(event.target===ui.dropZone&&(event.key==="Enter"||event.key===" ")){event.preventDefault();ui.mediaInput.click();}});ui.mediaInput.addEventListener("change",()=>addFiles(ui.mediaInput.files));
  for(const name of ["dragenter","dragover"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.add("drag-over");});for(const name of ["dragleave","drop"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.remove("drag-over");});ui.dropZone.addEventListener("drop",(event)=>addFiles(event.dataTransfer.files));
  document.querySelectorAll("[data-view]").forEach((button)=>{button.addEventListener("click",()=>setView(button.dataset.view));button.addEventListener("keydown",(event)=>{const tabs=[...document.querySelectorAll("[data-view]")].filter((tab)=>!tab.hidden);let index=tabs.indexOf(button);if(event.key==="ArrowRight")index=(index+1)%tabs.length;else if(event.key==="ArrowLeft")index=(index+tabs.length-1)%tabs.length;else if(event.key==="Home")index=0;else if(event.key==="End")index=tabs.length-1;else return;event.preventDefault();setView(tabs[index].dataset.view);tabs[index].focus();});});
  ui.zoomIn.addEventListener("click",()=>applyZoomPan(zoomState.zoom+0.25));ui.zoomOut.addEventListener("click",()=>applyZoomPan(zoomState.zoom-0.25));ui.resetZoom.addEventListener("click",resetZoomPan);ui.previewStage.addEventListener("keydown",handlePreviewKeydown);ui.previewStage.addEventListener("pointerdown",handlePreviewPointerDown);ui.previewStage.addEventListener("pointermove",handlePreviewPointerMove);ui.previewStage.addEventListener("pointerup",handlePreviewPointerEnd);ui.previewStage.addEventListener("pointercancel",handlePreviewPointerEnd);ui.compareSlider.addEventListener("input",updateCompareSplit);window.addEventListener("resize",()=>applyZoomPan());
  ui.sourceImage.addEventListener("load",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("loadeddata",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("seeked",markPreviewStale);ui.sourceVideo.addEventListener("timeupdate",()=>{if(!ui.sourceVideo.paused&&previewResult&&resultRevision===previewRevision)markPreviewStale();invalidateProcessedClipIfStale("Video timestamp changed");updateProcessedClipPanel();refreshCalibrationControls();});
  ui.sourceVideo.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser cannot decode the video container or codec. You can still convert it locally with FFmpeg.";updatePreview();});ui.sourceImage.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser could not open the image. Try another source file.";updatePreview();});
  ui.generatePreview.addEventListener("click",generatePreview);ui.previewDetail.addEventListener("change",markPreviewStale);ui.generateProcessedClip.addEventListener("click",generateProcessedClip);ui.cancelProcessedClip.addEventListener("click",cancelProcessedClip);ui.processedClipDuration.addEventListener("change",()=>{invalidateProcessedClipIfStale("Clip length changed");updateProcessedClipPanel();});ui.retryGpu.addEventListener("click",retryGpuAction);ui.presetSelect.addEventListener("change",()=>{applyPreset();refreshPerformancePresetStatus();});
  ui.performancePreset.addEventListener("change",()=>{if(ui.performancePreset.value==="custom"){performanceBaseline=currentSettingsForPreset();importedPresetBaseline=false;updateSettings(false);}else applyPerformancePreset(ui.performancePreset.value);});ui.resetPerformancePreset.addEventListener("click",resetPerformancePreset);ui.exportPreset.addEventListener("click",exportPresetFile);ui.importPresetButton.addEventListener("click",()=>ui.importPresetFile.click());ui.importPresetFile.addEventListener("change",()=>importPresetFile(ui.importPresetFile.files[0]));
  ui.calibrationBudget.addEventListener("change",refreshCalibrationControls);ui.calibrateInput.addEventListener("click",measureInputDetail);ui.applyCalibration.addEventListener("click",applyMeasuredInputDetail);
  ["strength","convergence","temporalSmoothing"].forEach((id)=>ui[id].addEventListener("input",()=>{ui.presetSelect.value="custom";updateSettings();}));["edgeRefinement","invertDepth"].forEach((id)=>ui[id].addEventListener("change",()=>{ui.presetSelect.value="custom";updateSettings();}));
  ["layoutSelect","modelSize","resolution","quality","executionMode"].forEach((id)=>ui[id].addEventListener("change",()=>updateSettings()));ui.depthMode.addEventListener("change",()=>{updateSettings();renderModels();});
  ui.convertButton.addEventListener("click",convertPending);ui.modelButton.addEventListener("click",()=>ui.modelInput.click());ui.modelInput.addEventListener("change",()=>uploadModel(ui.modelInput.files[0]));ui.installSelected.addEventListener("click",()=>installModel(ui.depthMode.value));ui.refreshModels.addEventListener("click",()=>{loadModels();loadStatus();});
  ui.applySettingsToReady.addEventListener("click",applySettingsToReady);
  ui.clearQueueButton.addEventListener("click",()=>{const finished=items.filter((x)=>["completed","failed","canceled"].includes(x.state));finished.forEach(removeItem);saveHidden();updateQueue();toast(finished.length?`${finished.length} finished item${finished.length===1?"":"s"} removed from this browser’s list.`:"There are no finished items to clear.");});
  ui.helpButton.addEventListener("click",()=>ui.aboutDialog.showModal());ui.aboutDone.addEventListener("click",()=>ui.aboutDialog.close());window.addEventListener("beforeunload",()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);});
  updateSettings(false);Promise.allSettled([loadStatus(),loadModels(),pollJobs()]);
})();

