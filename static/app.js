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
  function nvidiaGpuAvailable(runtime) {
    const adapter = selectedGpuAdapter(runtime);
    if (!runtime?.gpu_available || !adapter) return false;
    const vendorId = Number(adapter.vendor_id);
    return Number.isFinite(vendorId) ? vendorId === 0x10de : /nvidia/i.test(String(adapter.name || runtime.gpu_name || ""));
  }
  function retrySupported(runtime) {
    return capabilityV1(runtime) && runtime.capabilities.gpu_retry_supported === true;
  }
  function effectiveQueueSettings(item, currentSettings) {
    return validateSettings(item?.settingsFrozen ? item.settingsSnapshot : currentSettings);
  }
  function queueReadiness(pending, currentSettings, context = {}) {
    const runtime = context.runtime || null, models = Array.isArray(context.models) ? context.models : [];
    for (const item of pending || []) {
      let settings;
      try { settings = effectiveQueueSettings(item, currentSettings); }
      catch (error) { return `${item?.name || "A ready file"}: its effective settings are invalid (${error.message})`; }
      const mode = settings.depth_mode;
      if (settings.execution_mode === "gpu" || settings.execution_mode === "hybrid") {
        if (mode === "balanced" || mode === "precision") return `${item?.name || "A ready file"}: GPU + CPU and NVIDIA GPU modes require a named neural model.`;
        if (!runtime) return `${item?.name || "A ready file"}: checking NVIDIA GPU support…`;
        if (!runtime.gpu_available) return `${item?.name || "A ready file"}: DirectML is unavailable. Run .\\setup-gpu.ps1 and restart with .\\run.ps1, or choose Auto or CPU.`;
        if (!nvidiaGpuAvailable(runtime)) return `${item?.name || "A ready file"}: NVIDIA GPU mode requires an NVIDIA DirectML adapter, but the selected adapter is ${selectedGpuAdapter(runtime)?.name || "not NVIDIA"}.`;
        if (mode === "depth-anything-v2-large" && !capabilityV1(runtime)) return `${item?.name || "A ready file"}: ${gpuPolicy(runtime, selectedGpuAdapter(runtime), mode).message || "This server does not report GPU capability schema v1. Upgrade StereoLab to enable Large GPU inference safely."}`;
      }
      if (mode === "balanced") continue;
      if (mode === "precision") {
        if (!context.customModelAvailable) return `${item?.name || "A ready file"}: choose a compatible ONNX model to use Custom mode.`;
        continue;
      }
      if (runtime && !runtime.available) return `${item?.name || "A ready file"}: install ONNX Runtime to use neural depth models. See Local runtime below.`;
      const model = models.find((candidate) => candidate.id === mode);
      if (!model) return `${item?.name || "A ready file"}: ${context.catalogError ? `model catalog unavailable: ${context.catalogError}` : "checking the selected model…"}`;
      if (model.state === "downloading") return `${item?.name || "A ready file"}: ${model.name} is downloading. Wait for installation to finish.`;
      if (model.state !== "ready") return `${item?.name || "A ready file"}: install ${model.name} to use this depth engine.`;
    }
    return "";
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
  function completionNotificationStatus(supported, permission, optedIn) {
    if (!supported) return { enabled: false, canEnable: false, message: "Browser notifications are unavailable in this browser." };
    if (permission === "denied") return { enabled: false, canEnable: false, message: "Notifications are blocked for this site. Allow them in browser settings, then return here to enable completion notices." };
    if (permission !== "granted") return { enabled: false, canEnable: true, message: "Completion notifications are off. Enable them here to request browser permission." };
    if (optedIn) return { enabled: true, canEnable: true, message: "Completion notifications are enabled for this tab only." };
    return { enabled: false, canEnable: true, message: "Browser permission is available. Completion notifications are off until enabled here." };
  }
  function observeCompletionNotice(tracker, job, previousState = "") {
    const current = tracker && typeof tracker === "object" ? tracker : { trackedIds: [], activeIds: [], completedIds: [] };
    let trackedIds = Array.isArray(current.trackedIds) ? [...current.trackedIds] : [];
    let activeIds = Array.isArray(current.activeIds) ? [...current.activeIds] : [];
    let completedIds = Array.isArray(current.completedIds) ? [...current.completedIds] : [];
    if (!isOrdinaryExportJob(job) || typeof job.id !== "string" || !job.id.trim()) return { tracker: { trackedIds, activeIds, completedIds }, notify: false };
    const id = job.id, activeStates = ["uploading", "queued", "processing"];
    if (completedIds.includes(id)) return { tracker: { trackedIds, activeIds, completedIds }, notify: false };
    const remember = () => {
      if (!trackedIds.includes(id)) trackedIds.push(id);
      while (trackedIds.length > 512) {
        const removed = trackedIds.shift();
        activeIds = activeIds.filter((value) => value !== removed);
        completedIds = completedIds.filter((value) => value !== removed);
      }
    };
    if (activeStates.includes(job.state)) {
      remember();
      if (!activeIds.includes(id)) activeIds.push(id);
      return { tracker: { trackedIds, activeIds, completedIds }, notify: false };
    }
    if (job.state === "completed") {
      const qualified = activeIds.includes(id) || activeStates.includes(previousState);
      activeIds = activeIds.filter((value) => value !== id);
      if (!qualified) return { tracker: { trackedIds, activeIds, completedIds }, notify: false };
      remember();
      completedIds.push(id);
      return { tracker: { trackedIds, activeIds, completedIds }, notify: true };
    }
    if (["failed", "canceled"].includes(job.state)) activeIds = activeIds.filter((value) => value !== id);
    return { tracker: { trackedIds, activeIds, completedIds }, notify: false };
  }
  function keyboardShortcutAction(event = {}) {
    if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.repeat || event.defaultPrevented || event.isComposing || event.modalOpen) return null;
    const target = event.target || {}, tag = String(target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || target.isContentEditable) return null;
    if (event.code === "KeyP") return event.previewEnabled ? "preview" : null;
    if (event.code === "Enter") return event.submitEnabled ? "submit" : null;
    return null;
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
  function historyCapability(status) {
    const caps = status?.history_capabilities;
    if (!caps || caps.schema_version !== 1 || caps.supported !== true || typeof caps.enabled !== "boolean" ||
        !["disabled", "ready", "degraded"].includes(caps.state) ||
        (caps.enabled && caps.state === "disabled") || (!caps.enabled && caps.state !== "disabled") ||
        caps.records_limit !== 1000 || caps.source_retention !== false || caps.automatic_requeue !== false || caps.frame_resume !== false) return null;
    if (caps.instance_name != null && (typeof caps.instance_name !== "string" || !/^[a-z0-9][a-z0-9_-]{0,31}$/.test(caps.instance_name))) return null;
    if (caps.warning != null && typeof caps.warning !== "string") return null;
    return caps;
  }
  function validateHistoryItem(item) {
    if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("History item must be an object.");
    if (typeof item.job_id !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(item.job_id)) throw new Error("History job identity is invalid.");
    for (const key of ["created_at", "updated_at"]) if (typeof item[key] !== "number" || !Number.isFinite(item[key]) || item[key] < 0) throw new Error(`History ${key} is invalid.`);
    if (!["queued", "processing", "completed", "failed", "canceled", "interrupted"].includes(item.state)) throw new Error("History state is invalid.");
    if (item.state === "completed" ? (typeof item.completed_at !== "number" || !Number.isFinite(item.completed_at) || item.completed_at < 0) : item.completed_at !== null) throw new Error("History completion timestamp is invalid.");
    if (item.state === "interrupted" ? !["queued", "processing", "initializing", "preparing", "finalizing"].includes(item.last_known_state) : item.last_known_state !== null) throw new Error("History last-known state is invalid.");
    validateOutputBasename(item.filename, "source filename");
    if (typeof item.media_type !== "string" || !["image", "video"].includes(item.media_type)) throw new Error("History media type is invalid.");
    if (!Number.isSafeInteger(item.source_size_bytes) || item.source_size_bytes < 0 || typeof item.source_sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(item.source_sha256)) throw new Error("History source details are invalid.");
    if (item.settings_schema_version !== 1 || item.naming_schema_version !== 1) throw new Error("History settings schema is unsupported.");
    const settings = validateSettings(item.settings);
    validateOutputNameTemplate(item.output_name_template);
    validateOutputBasename(item.planned_output_name, "planned output name");
    if (item.output_name != null) validateOutputBasename(item.output_name, "output name");
    if (typeof item.output_available !== "boolean" || typeof item.readonly !== "boolean" || typeof item.source_available !== "boolean" || typeof item.requires_source_reselection !== "boolean") throw new Error("History availability flags are invalid.");
    if (item.readonly !== true || item.source_available !== false || item.requires_source_reselection !== true) throw new Error("History item is not a read-only prior launch record.");
    if (item.output_available && (item.state !== "completed" || typeof item.output_name !== "string")) throw new Error("History output availability is inconsistent.");
    return { ...item, settings };
  }
  function validateHistoryPage(page) {
    if (!page || typeof page !== "object" || Array.isArray(page) || page.schema_version !== 1 || !Array.isArray(page.items)) throw new Error("History response schema is unsupported.");
    if (page.next_cursor !== null && (typeof page.next_cursor !== "string" || !/^[A-Za-z0-9_-]{1,256}$/.test(page.next_cursor))) throw new Error("History page cursor is invalid.");
    if (!Number.isSafeInteger(page.invalid_records) || page.invalid_records < 0) throw new Error("History invalid-record count is invalid.");
    if (page.items.length > 100) throw new Error("History page exceeds its limit.");
    return { ...page, items: page.items.map(validateHistoryItem) };
  }
  function historyStateLabel(item) { return item?.state === "interrupted" ? "Interrupted" : ({ queued: "Queued when stopped", processing: "Processing when stopped", completed: "Completed", failed: "Failed", canceled: "Canceled" }[item?.state] || "Unavailable state"); }
  const defaultOutputNameTemplate = "{name}_{layout}", outputNameTemplateFields = ["name", "layout", "model", "date"], outputNameTemplateMaxLength = 120;
  function outputNamingCapability(status) {
    const caps = status?.naming_capabilities;
    if (!caps || caps.schema_version !== 1 || caps.supported !== true || caps.max_template_length !== outputNameTemplateMaxLength || caps.unique_job_id_suffix !== true ||
        !Array.isArray(caps.fields) || caps.fields.length !== outputNameTemplateFields.length || new Set(caps.fields).size !== outputNameTemplateFields.length ||
        !outputNameTemplateFields.every((field) => caps.fields.includes(field))) return null;
    return caps;
  }
  function validateOutputNameTemplate(value, maxLength=outputNameTemplateMaxLength) {
    if (typeof value !== "string" || !value.trim()) throw new Error("Output name template must not be empty.");
    if (!Number.isSafeInteger(maxLength) || maxLength < 1 || maxLength > outputNameTemplateMaxLength || Array.from(value).length > maxLength) throw new Error(`Output name template must be ${outputNameTemplateMaxLength} characters or fewer.`);
    const invalidUnicode = /[\p{Cc}\p{Cf}\p{Cs}]/u;
    for(const char of value)if(invalidUnicode.test(char))throw new Error("Output name template contains invalid Unicode.");
    const invalidLiteral = '<>:"/\\\\|?*';
    for (let index = 0; index < value.length; index++) {
      const char = value[index], code = value.charCodeAt(index);
      if (code < 32 || code === 127 || invalidLiteral.includes(char)) throw new Error("Output name template contains a character that is not allowed in a filename.");
      if (char === "}") throw new Error("Output name template contains an unmatched closing brace.");
      if (char === "{") {
        const close = value.indexOf("}", index + 1), token = close < 0 ? "" : value.slice(index + 1, close);
        if (!outputNameTemplateFields.includes(token)) throw new Error("Output name template contains an unmatched or unsupported token.");
        index = close;
      }
    }
    return value;
  }
  function validateOutputBasename(value, label) {
    if(typeof value!=="string"||!value.trim())throw new Error(`The server did not report a valid ${label}.`);
    const invalidLiteral='<>:"/\\\\|?*';
    const invalidUnicode=/[\p{Cc}\p{Cf}\p{Cs}]/u;
    for(const char of value)if(invalidUnicode.test(char)||invalidLiteral.includes(char))throw new Error(`The server reported an unsafe ${label}.`);
    return value;
  }
  function validateQueueJobNaming(job) {
    if (!job || typeof job !== "object" || Array.isArray(job)) throw new Error("The queue naming snapshot must be an object.");
    const hasNaming = job.naming_schema_version != null || job.output_name_template != null || job.planned_output_name != null || job.output_name != null;
    if (!hasNaming) return null;
    if (job.naming_schema_version !== 1) throw new Error("This server does not provide output naming schema v1.");
    const outputNameTemplate = validateOutputNameTemplate(job.output_name_template);
    const plannedOutputName=validateOutputBasename(job.planned_output_name,"planned output name");
    const outputName=job.output_name==null?null:validateOutputBasename(job.output_name,"actual output name");
    if (job.state === "completed" && job.output_name == null) throw new Error("The completed job did not report its actual output name.");
    return { outputNameTemplate, plannedOutputName, outputName };
  }
  function validateQueueJobSnapshot(job) {
    if (!job || typeof job !== "object" || Array.isArray(job) || job.settings_schema_version !== 1) throw new Error("This server does not provide queue settings schema v1.");
    if (typeof job.source_available !== "boolean") throw new Error("The server did not report source availability.");
    if (job.queue_position !== null && (!Number.isSafeInteger(job.queue_position) || job.queue_position < 1)) throw new Error("The server reported an invalid queue position.");
    return { settings: validateSettings(job.settings), sourceAvailable: job.source_available, queuePosition: job.queue_position };
  }
  const depthCacheStates = Object.freeze(["disabled", "ineligible", "miss", "writing", "stored", "hit", "unavailable"]);
  function validateDepthCacheStatus(cache) {
    if (cache == null) return null;
    if (!cache || typeof cache !== "object" || Array.isArray(cache) || !depthCacheStates.includes(cache.state) ||
        !Number.isSafeInteger(cache.frames) || cache.frames < 0 || !Number.isSafeInteger(cache.bytes) || cache.bytes < 0 ||
        cache.warning != null && typeof cache.warning !== "string") throw new Error("The server returned invalid depth cache status.");
    if(cache.estimated_bytes!=null&&(!Number.isSafeInteger(cache.estimated_bytes)||cache.estimated_bytes<0))throw new Error("The server returned invalid depth cache estimate.");
    return { state: cache.state, frames: cache.frames, bytes: cache.bytes, warning: cache.warning ?? null,
      ...(cache.estimated_bytes!=null?{estimated_bytes:cache.estimated_bytes}:{}) };
  }
  function depthCacheStatusLabel(cache, formatBytes = (bytes) => `${bytes} B`) {
    const value = validateDepthCacheStatus(cache);
    if (!value) return "";
    const labels = { disabled: "Depth cache off", ineligible: "Depth cache not used", miss: "Depth cache miss", writing: "Depth cache saving", stored: "Depth cache stored", hit: "Depth cache hit", unavailable: "Depth cache unavailable" };
    const detail = value.frames ? `${value.frames} frames` : "";
    const size = value.bytes ? formatBytes(value.bytes) : "";
    const estimate=value.estimated_bytes&&["writing","unavailable"].includes(value.state)?`estimated total ${formatBytes(value.estimated_bytes)}`:"";
    return [labels[value.state], detail, size, estimate].filter(Boolean).join(" · ");
  }
  function validateDepthCacheInventory(report) {
    if (!report || typeof report !== "object" || Array.isArray(report) || !Number.isSafeInteger(report.quota_bytes) || report.quota_bytes < 0 ||
        !Number.isSafeInteger(report.bytes) || report.bytes < 0 || !Array.isArray(report.entries) || !Array.isArray(report.errors)) throw new Error("The server returned invalid depth cache inventory.");
    const entries = report.entries.map((entry) => {
      if (!entry || typeof entry !== "object" || Array.isArray(entry) || typeof entry.key !== "string" || !/^[0-9a-f]{64}$/.test(entry.key) ||
          !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || !Number.isSafeInteger(entry.frames) || entry.frames < 0 || typeof entry.active !== "boolean") throw new Error("The server returned an invalid depth cache entry.");
      return { key: entry.key, bytes: entry.bytes, frames: entry.frames, active: entry.active };
    });
    const errors = report.errors.map((error) => {
      if (typeof error === "string") return error.slice(0, 240);
      if (!error || typeof error !== "object" || Array.isArray(error) || typeof error.key !== "string" || typeof error.error !== "string") throw new Error("The server returned invalid depth cache inventory errors.");
      return `${error.key || "Inventory"}: ${error.error}`.slice(0, 240);
    });
    return { quota_bytes: report.quota_bytes, bytes: report.bytes, entries, errors };
  }
  function depthCacheSelection(entries, keys) {
    const selected = new Set(Array.isArray(keys) ? keys : []);
    return (Array.isArray(entries) ? entries : []).filter((entry) => !entry.active && selected.has(entry.key));
  }
  function depthCacheCleanupRequest(entries, keys) {
    return { keys: depthCacheSelection(entries, keys).map((entry) => entry.key) };
  }
  function depthCacheCleanupSummary(requestedKeys, result, inventoryRefreshed) {
    const deleted = Array.isArray(result?.deleted) ? result.deleted.length : 0;
    const errors = Array.isArray(result?.errors) ? result.errors : [];
    const detail = errors.length ? ` ${errors.map((entry) => `${entry.key || "Cleanup request"}: ${String(entry.error || "Cleanup failed").slice(0, 240)}`).join("; ")}` : " No cleanup errors.";
    return `Removed ${deleted} of ${requestedKeys.length} selected cache entr${requestedKeys.length === 1 ? "y" : "ies"}.${detail}${inventoryRefreshed ? "" : " Cache inventory could not be refreshed."}`;
  }
  function executionSummary(execution, pipeline) {
    if (!execution) return "";
    if (execution.depth_cache_hit === true) return ["Cached depth · CPU rendering", pipeline === "overlapped" ? "video stages overlapped" : ""].filter(Boolean).join(" · ");
    const provider = execution.active_provider;
    const accelerator = typeof execution.device === "string" && !/^cpu$/i.test(execution.device) ? `${execution.device} · ` : "";
    const label = provider && /CUDA|Dml|Tensorrt/i.test(provider) ? `${accelerator}GPU depth + CPU processing` : provider === "CPUExecutionProvider" || execution.device === "cpu" ? "CPU processing" : provider ? `${provider} depth` : "Device use not reported";
    return [label, execution.fallback_reason ? `CPU fallback: ${execution.fallback_reason}` : "", pipeline === "overlapped" ? "video stages overlapped" : "", execution.video_depth_stats ? `${execution.video_depth_stats.neural_frames} neural / ${execution.video_depth_stats.flow_frames} flow frames` : ""].filter(Boolean).join(" · ");
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
  function makeReadyQueueCopy(item, settings, outputNameTemplate=defaultOutputNameTemplate) {
    if (!item || !item.file) throw new Error("The original source file is unavailable. Reselect it before retrying or duplicating.");
    const outputNameTemplateSnapshot=validateOutputNameTemplate(outputNameTemplate);
    return { file: item.file, name: item.name, size: item.size, mediaType: item.mediaType, state: "ready", progress: 0, error: null, message: "Ready to submit as a new job", settingsSnapshot: validateSettings(settings), settingsFrozen: true, outputNameTemplateSnapshot, namingFrozen: true };
  }
  function freezeReadyQueueItem(item, settings, outputNameTemplate=defaultOutputNameTemplate) {
    return { ...item, settingsSnapshot: validateSettings(settings), settingsFrozen: true, outputNameTemplateSnapshot: validateOutputNameTemplate(outputNameTemplate), namingFrozen: true };
  }
  function outputNameTemplateForSubmit(item, currentTemplate=defaultOutputNameTemplate) {
    return validateOutputNameTemplate(item?.namingFrozen && item.outputNameTemplateSnapshot != null ? item.outputNameTemplateSnapshot : currentTemplate);
  }
  function outputNameTemplateForSubmission(capability, item, currentTemplate=defaultOutputNameTemplate) {
    return capability ? validateOutputNameTemplate(outputNameTemplateForSubmit(item,currentTemplate),capability.max_template_length) : null;
  }
  function canSnapshotOutputNameTemplate(capability, currentTemplate=defaultOutputNameTemplate) {
    if(!capability)return true;
    try{validateOutputNameTemplate(currentTemplate,capability.max_template_length);return true;}catch{return false;}
  }
  function outputNameTemplateForQueueCopy(item, kind, currentTemplate=defaultOutputNameTemplate) {
    if (kind === "retry") return validateOutputNameTemplate(item?.outputNameTemplateSnapshot ?? defaultOutputNameTemplate);
    if (kind === "duplicate") return validateOutputNameTemplate(currentTemplate);
    throw new Error("Queue copy kind must be retry or duplicate.");
  }
  function queuedNameUpdateRequest(jobId, outputNameTemplate) {
    if (typeof jobId !== "string" || !jobId.trim()) throw new Error("A queued export job ID is required.");
    return { url: `/api/jobs/${encodeURIComponent(jobId)}/naming`, options: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ output_name_template: validateOutputNameTemplate(outputNameTemplate) }) } };
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
    fast: Object.freeze({ layout: "XREAL_SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 256, resolution: "1080p", quality: "fast", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false, depth_interval: 1, video_decode: "software", use_depth_cache: false }),
    balanced: Object.freeze({ layout: "XREAL_SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 384, resolution: "1080p", quality: "balanced", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false, depth_interval: 1, video_decode: "software", use_depth_cache: false }),
    quality: Object.freeze({ layout: "XREAL_SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 518, resolution: "source", quality: "high", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false, depth_interval: 1, video_decode: "software", use_depth_cache: false }),
    conservative4gb: Object.freeze({ layout: "XREAL_SBS", strength: 5, convergence: 0.5, depth_mode: "depth-anything-v2-small", execution_mode: "auto", input_size: 256, resolution: "1080p", quality: "balanced", temporal_smoothing: 0.35, edge_refinement: true, invert_depth: false, depth_interval: 1, video_decode: "software", use_depth_cache: false })
  });
  const settingsKeys = Object.freeze(["layout", "strength", "convergence", "depth_mode", "execution_mode", "input_size", "resolution", "quality", "temporal_smoothing", "edge_refinement", "invert_depth", "depth_interval", "video_decode", "use_depth_cache"]);
  function performancePreset(name) {
    const preset = performancePresets[name];
    return preset ? { ...preset } : null;
  }
  function validateSettings(settings) {
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) throw new Error("settings must be an object.");
    if (!Object.prototype.hasOwnProperty.call(settings, "depth_interval")) settings = { ...settings, depth_interval: 1 };
    if (!Object.prototype.hasOwnProperty.call(settings, "video_decode")) settings = { ...settings, video_decode: "software" };
    if (!Object.prototype.hasOwnProperty.call(settings, "use_depth_cache")) settings = { ...settings, use_depth_cache: false };
    const keys = Object.keys(settings);
    const unknown = keys.filter((key) => !settingsKeys.includes(key));
    if (unknown.length) throw new Error(`Unknown settings field: ${unknown[0]}.`);
    const missing = settingsKeys.find((key) => !Object.prototype.hasOwnProperty.call(settings, key));
    if (missing) throw new Error(`Missing settings field: ${missing}.`);
    const enums = {
      layout: ["SBS", "XREAL_SBS", "HALFSBS", "TB", "HALFTB", "CROSS", "ANAGLYPH", "DEPTH", "RGBD"],
      depth_mode: ["depth-anything-v2-small", "depth-anything-v2-small-fp16", "depth-anything-v2-base", "depth-anything-v2-large", "dpt-hybrid-midas", "balanced", "precision"],
      execution_mode: ["auto", "hybrid", "gpu", "cpu"], input_size: [256, 384, 518], depth_interval: [1, 2], video_decode: ["software", "amd"],
      resolution: ["1080p", "source"], quality: ["fast", "balanced", "high"]
    };
    for (const [key, values] of Object.entries(enums)) if (!values.includes(settings[key])) throw new Error(`${key} must be one of the supported values.`);
    for (const [key, min, max] of [["strength", 0, 10], ["convergence", 0, 1], ["temporal_smoothing", 0, 0.9]]) {
      if (typeof settings[key] !== "number" || !Number.isFinite(settings[key]) || settings[key] < min || settings[key] > max) throw new Error(`${key} must be a finite number from ${min} to ${max}.`);
    }
    for (const key of ["edge_refinement", "invert_depth", "use_depth_cache"]) if (typeof settings[key] !== "boolean") throw new Error(`${key} must be a boolean.`);
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
  return Object.freeze({ capabilityV1, gpuPolicy, selectedGpuAdapter, nvidiaGpuAvailable, retrySupported, retryGpu, effectiveQueueSettings, queueReadiness, fullStillEligibility, validatePreviewResponse, previewDimensionDetails, processedClipCapability, validProcessedClipDuration, processedClipUnavailableMessage, processedClipErrorAfterPoll, validatePreviewClipJob, isOrdinaryExportJob, completionNotificationStatus, observeCompletionNotice, keyboardShortcutAction, clipContextMatches, processedClipResponseAction, processedClipQuarantineAfterPoll, queueCapabilities, historyCapability, validateHistoryItem, validateHistoryPage, historyStateLabel, outputNamingCapability, validateOutputNameTemplate, validateOutputBasename, validateQueueJobNaming, validateQueueJobSnapshot, validateDepthCacheStatus, depthCacheStatusLabel, validateDepthCacheInventory, depthCacheSelection, depthCacheCleanupRequest, depthCacheCleanupSummary, executionSummary, moveReadyQueueItem, moveQueuedExportIds, makeReadyQueueCopy, freezeReadyQueueItem, outputNameTemplateForSubmit, outputNameTemplateForSubmission, canSnapshotOutputNameTemplate, outputNameTemplateForQueueCopy, queuedNameUpdateRequest, queuePhaseLabel, previewClipDetails, previewExecutionDetails, preflightDetails, jobEtaLabel, performancePreset, validateSettings, validateUserPreset, exportUserPreset, hasUnsavedSettings, validateCalibrationReport, calibrationApplicable, calibrationDetails, calibrationStatusLabel, calibrationContextMatches, zoomPanGeometry, compareSplitPercent, sourceDepthCompareAvailable });
})();
if (typeof module !== "undefined" && module.exports) module.exports = frontendContract;
if (typeof window !== "undefined") window.StereoLabFrontendContract = frontendContract;

if (typeof document !== "undefined") (() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ids = ["mediaInput","dropZone","browseButton","queueLiveStatus","sourceVideo","sourceImage","processedImage","previewEmpty","previewStage","sourceName","previewState","generatePreview","layoutSelect","formatHint","previewModeLabel","presetSelect","depthMode","depthNote","installSelected","modelRow","modelInput","modelButton","modelState","modelSizeRow","modelSizeLabel","modelSize","midasSizeNote","strength","strengthValue","convergence","convergenceValue","temporalSmoothing","depthInterval","videoDecode","useDepthCache","smoothingValue","edgeRefinement","invertDepth","resolution","quality","convertButton","convertCount","convertHint","queueRows","queueEmpty","queueCount","queueHeadingCount","clearQueueButton","applySettingsToReady","queueWorkflowStatus","outputNamingControls","outputNameTemplate","outputNameTemplateHelp","outputNamingStatus","historyPanel","historyStatus","historyRows","historyRefresh","historyMore","enginePill","engineText","toolDescription","toolPath","runtimeDescription","runtimeCommand","modelCards","refreshModels","toast","helpButton","aboutDialog","aboutDone","completionNotifications","completionNotificationStatus","shortcutHelp"];
  ids.push("executionMode", "executionNote", "previewExecution", "retryGpu", "performancePreset", "performancePresetStatus", "performancePresetSummary", "resetPerformancePreset", "exportPreset", "importPresetButton", "importPresetFile", "savedPresetSelect", "savedPresetName", "savedPresetFavorite", "saveNamedPreset", "applyNamedPreset", "deleteNamedPreset", "savedPresetStatus", "calibrationBudget", "calibrateInput", "calibrationStatus", "calibrationReport", "calibrationSummary", "calibrationCandidates", "calibrationScope", "calibrationLimitations", "calibrationStorageWarning", "applyCalibration", "tabCompare", "previewMedia", "sourceDepthCompare", "compareSource", "compareDepth", "compareDivider", "compareControls", "compareSlider", "compareValue", "compareResolution", "zoomIn", "zoomOut", "zoomLabel", "resetZoom", "previewDetail", "previewDetailHint", "glassesPreview", "processedClipCapability", "processedClipDuration", "generateProcessedClip", "cancelProcessedClip", "processedClipStatus", "processedClipVideo");
  ids.push("pauseQueue", "queuePauseStatus", "openOutputFolder", "refreshStorage", "storageStatus", "storageUsage", "storageFiles", "deleteStorageFiles", "copyOutputDirectory", "refreshDepthCache", "depthCacheStatus", "depthCacheUsage", "depthCacheEntries", "cleanupDepthCache");
  ids.push("projectName", "saveProject", "openProject", "projectFileInput", "projectStatus");
  const ui = Object.fromEntries(ids.map((id) => [id, $(id)]));
  const imageExtensions = new Set(["jpg","jpeg","png","webp"]);
  const videoExtensions = new Set(["mp4","webm","mkv","mov","avi","wmv","flv","ts","mts","m2ts","m4v"]);
  const activeStates = new Set(["uploading","queued","processing"]);
  const layouts = {
    SBS:["Side by side","Two full-resolution views, side by side."],XREAL_SBS:["XREAL Full SBS","3840 × 1080 frame; each view is centered without stretching in its 1920 × 1080 eye area."],HALFSBS:["Half side by side","Two squeezed views share a source-sized frame."],
    TB:["Top / bottom","Two full-resolution views, stacked vertically."],HALFTB:["Half top / bottom","Two squeezed views share a source-sized frame."],
    CROSS:["Cross-eye","Right view first, for cross-eye viewing."],ANAGLYPH:["Red / cyan","A color-filter stereo image for red/cyan glasses."],
    DEPTH:["Depth map","A grayscale depth estimate: brighter areas are nearer."],RGBD:["RGB + depth","Original color and depth map, side by side."]
  };
  const presets = {cinema:{strength:5,convergence:.5,smoothing:.35,edges:true,invert:false},subtle:{strength:2.5,convergence:.5,smoothing:.5,edges:true,invert:false},action:{strength:8,convergence:.45,smoothing:.25,edges:true,invert:false}};
  let performanceBaseline = frontendContract.performancePreset("balanced"), importedPresetBaseline = false;
  const namedPresetsKey = "stereolab.namedPresets.v1", namedPresetLimit = 20, namedPresetNameLimit = 40;
  let namedPresets = [];
  const PROJECT_SCHEMA_VERSION=1,PROJECT_MAX_BYTES=2*1024*1024,PROJECT_MAX_SOURCES=200,PROJECT_MAX_QUEUE_ENTRIES=500,PROJECT_MAX_SOURCE_BYTES=8*1024**3,PROJECT_MAX_TOTAL_BYTES=64*1024**3,PROJECT_PREVIEW_TIME_LIMIT=31*86400;
  let projectBusy=false;
  let calibrationBusy = false, calibrationResult = null, calibrationResultContext = null, calibrationNotice = "";
  let items = [], selectedUid = null, previewUrl = null, previewFile = null, previewView = "original", previewResult = null;
  let previewRevision = 0, resultRevision = -1, previewBusy = false, sourceError = "", previewError = "";
  let zoomState = { zoom: 1, panX: 0, panY: 0 }, panPointer = null, compareSplit = 50;
  let models = [], runtime = null, status = null, statusLoadError = "", customModelAvailable = false, catalogError = "";
  let historyCapabilities = null, historyCursor = null, historyLoading = false;
  let namingCapability = null, currentOutputNameTemplate = "{name}_{layout}";
  let modelTimer = null, jobTimer = null, jobsRetryDelay = 0, jobsBusy = false, modelsBusy = false, submitting = false, queueMutationBusy = false, queueWorkflowMessage = "", toastTimer = null;
  let completionNoticesOptedIn = false, completionNotificationBusy = false;
  let completionNoticeTracker = { trackedIds: [], activeIds: [], completedIds: [] };
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
  function updateCompletionNotificationControls() {
    try {
      const supported = typeof window.Notification === "function";
      const permission = supported ? window.Notification.permission : "unsupported";
      if (permission !== "granted") completionNoticesOptedIn = false;
      const state = frontendContract.completionNotificationStatus(supported, permission, completionNoticesOptedIn);
      ui.completionNotifications.disabled = !state.canEnable || completionNotificationBusy;
      ui.completionNotifications.textContent = state.enabled ? "Disable completion notifications" : permission === "denied" ? "Notifications blocked" : "Enable completion notifications";
      ui.completionNotificationStatus.textContent = state.message;
    } catch {
      completionNoticesOptedIn = false;
      ui.completionNotifications.disabled = true;
      ui.completionNotifications.textContent = "Notifications unavailable";
      ui.completionNotificationStatus.textContent = "Browser notification status could not be checked. Queue processing continues normally.";
    }
  }
  async function toggleCompletionNotifications() {
    if (completionNoticesOptedIn) { completionNoticesOptedIn = false; updateCompletionNotificationControls(); return; }
    if (typeof window.Notification !== "function" || window.Notification.permission === "denied") { updateCompletionNotificationControls(); return; }
    completionNotificationBusy = true; updateCompletionNotificationControls();
    let permission = window.Notification.permission, requestError = "";
    try { if (permission !== "granted") permission = await window.Notification.requestPermission(); }
    catch { requestError = "Could not request notification permission. Check browser settings and try again."; }
    completionNoticesOptedIn = permission === "granted";
    completionNotificationBusy = false; updateCompletionNotificationControls();
    if (requestError) ui.completionNotificationStatus.textContent = requestError;
  }
  function showCompletionNoticeBatch(sourceNames) {
    if (!sourceNames.length) return;
    try {
      updateCompletionNotificationControls();
      if (!completionNoticesOptedIn || typeof window.Notification !== "function" || window.Notification.permission !== "granted") return;
      const body = sourceNames.length === 1 ? String(sourceNames[0] || "Export").split(/[\\/]/).pop().slice(0, 120) : `${sourceNames.length} exports completed`;
      new window.Notification("StereoLab export ready", { body });
    } catch {
      completionNoticesOptedIn = false;
      updateCompletionNotificationControls();
      ui.completionNotificationStatus.textContent = "A completion notification could not be shown. Check browser notification settings, then enable notices again.";
    }
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
    if (item.decoder?.backend) details.push(`Decoder: ${item.decoder.backend}${item.decoder.adapter_name?` · ${item.decoder.adapter_name}`:""}${item.decoder.fallback_reason?` · ${item.decoder.fallback_reason}`:""}`);
    if (item.phase === "retrying" && metricNumber(item.retry_count) !== null) details.push(`Retry ${item.retry_count}`);
    if (typeof item.retry_reason === "string" && item.retry_reason.trim()) details.push(`Retry reason: ${item.retry_reason.trim()}`);
    return details;
  }
  function selectedModel() { return models.find((model) => model.id === ui.depthMode.value); }
  function gpuPolicy() { return frontendContract.gpuPolicy(runtime, frontendContract.selectedGpuAdapter(runtime), ui.depthMode.value); }
  function largeOnLimitedGpu() { return ui.depthMode.value === "depth-anything-v2-large" && gpuPolicy().limited; }
  function largeGpuUpgradeRequired() { return ui.depthMode.value === "depth-anything-v2-large" && Boolean(runtime?.gpu_available) && !frontendContract.capabilityV1(runtime); }
  function gpuBlock() {
    const mode = ui.executionMode.value;
    if (mode !== "gpu" && mode !== "hybrid") return "";
    if (["balanced", "precision"].includes(ui.depthMode.value)) return "GPU + CPU and NVIDIA GPU modes require a named neural model. Choose Auto or CPU for visual approximation or custom ONNX.";
    if (!runtime) return "Checking NVIDIA GPU support…";
    if (!runtime.gpu_available) return "DirectML is unavailable. Run .\\setup-gpu.ps1 to prepare the separate .venv-gpu environment, then start it with .\\run.ps1. Or choose Auto or CPU.";
    if (!frontendContract.nvidiaGpuAvailable(runtime)) return `NVIDIA GPU mode requires an NVIDIA DirectML adapter, but the selected adapter is ${frontendContract.selectedGpuAdapter(runtime)?.name || "not NVIDIA"}.`;
    if (ui.depthMode.value === "depth-anything-v2-large" && !frontendContract.capabilityV1(runtime)) return gpuPolicy().message || "This server does not report GPU capability schema v1. Upgrade StereoLab to enable Large GPU inference safely.";
    return "";
  }
  function renderDeviceNote() {
    const mode = ui.executionMode.value, blocked = gpuBlock();
    ui.retryGpu.hidden = !frontendContract.retrySupported(runtime);
    ui.executionNote.classList.toggle("device-warning", Boolean(blocked) || mode !== "cpu" && (largeOnLimitedGpu() || largeGpuUpgradeRequired() || runtime && !runtime.gpu_available));
    if (blocked) { ui.executionNote.textContent = blocked; return; }
    if (["balanced", "precision"].includes(ui.depthMode.value)) { ui.executionNote.textContent = ui.depthMode.value === "balanced" ? "Visual approximation uses CPU image cues. Choose a named neural depth engine to use GPU depth estimation." : "Custom ONNX models use CPU in this studio. Choose a named neural depth engine to use GPU depth estimation."; return; }
    if (mode === "cpu") { ui.executionNote.textContent = "CPU depth estimation and media stages. Video may still use NVIDIA NVENC when available."; return; }
    if (!runtime) { ui.executionNote.textContent = "Checking local GPU support…"; return; }
    if (largeGpuUpgradeRequired()) { ui.executionNote.textContent = gpuPolicy().message; return; }
    if (largeOnLimitedGpu()) {
      const policy = gpuPolicy(), cap = policy.inputCap;
      ui.executionNote.textContent = `Large will try DirectML on this ${policy.memoryUnknown ? "GPU with unknown memory, treated conservatively" : "limited-memory GPU"} with a ${cap} px long-side cap. Start at 256 px detail to reduce memory use. Auto can fall back to CPU if GPU inference fails; GPU + CPU and NVIDIA GPU modes report the error.`;
      return;
    }
    if (!runtime.gpu_available) { ui.executionNote.textContent = `CPU fallback: ${runtime.error || "no compatible GPU runtime and NVIDIA adapter detected"}. Optional GPU setup: .\\setup-gpu.ps1 (separate .venv-gpu), then .\\run.ps1.`; return; }
    const policy = gpuPolicy(), capNote = policy.supported ? ` Named-model input cap ${policy.inputCap} px.` : "";
    const modeNote = mode === "gpu" ? "NVIDIA GPU required for depth; failures will be reported." : mode === "hybrid" ? "NVIDIA GPU required for depth; CPU handles media stages, and GPU failures will be reported." : "Auto prefers the selected GPU for depth and can fall back to CPU.";
    ui.executionNote.textContent = `${runtime.gpu_name || "NVIDIA GPU"} available${runtime.gpu_verified ? " · inference verified" : " · inference not yet verified"}.${capNote} ${modeNote}`;
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
  function queueNamingBlock(pending=items.filter((item)=>!item.jobId&&item.state==="ready")){
    if(!namingCapability||!pending.length)return "";
    try{for(const item of pending)frontendContract.outputNameTemplateForSubmission(namingCapability,item,currentOutputNameTemplate);return "";}
    catch(error){return `Fix the output name template before uploading: ${error.message}`;}
  }
  function sourceReady() { return previewFile && !sourceError && (isImage(previewFile) ? ui.sourceImage.complete && ui.sourceImage.naturalWidth > 0 : ui.sourceVideo.readyState >= 2); }
  function drawSourceFrame(source, canvas) {
    const wasHidden = source === ui.sourceVideo && source.hidden;
    if (wasHidden) source.hidden = false;
    try { canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height); }
    finally { if (wasHidden) source.hidden = true; }
  }
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
  function calibrationBlock(){if(calibrationBusy)return "Input-detail measurement is running.";if(previewBusy)return "Wait for the current preview to finish.";if(!previewFile||!sourceReady())return "Choose a decoded source frame to begin.";if(!/^depth-anything-v2-(small(?:-fp16)?|base|large)$/.test(ui.depthMode.value))return "Choose a named Depth Anything model. MiDaS uses fixed input detail; visual approximation and custom models cannot be calibrated.";const model=selectedModel();if(!model||model.state!=="ready")return model?.state==="downloading"?"Wait for the selected model to finish installing.":`Install ${model?.name||"the selected model"} before calibrating.`;if(runtime&&!runtime.available)return "Install ONNX Runtime before calibrating.";if(gpuBlock())return gpuBlock();if(items.some((item)=>item.state==="processing"))return "Wait for active processing jobs to finish.";return "";}
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
    ui.previewStage.classList.toggle("xreal-sbs-preview", previewView === "stereo" && ui.layoutSelect.value === "XREAL_SBS");
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
    ui.generatePreview.dataset.previewBusy = String(previewBusy);
    ui.glassesPreview.disabled = !previewResult || previewView !== "stereo" || resultRevision !== previewRevision;
    ui.previewExecution.hidden = !previewResult;
    ui.previewExecution.textContent = previewResult ? `${resultRevision !== previewRevision ? "Previous preview · " : ""}${[frontendContract.previewDimensionDetails(previewResult),frontendContract.executionSummary(previewResult.execution) || "Device use not reported",frontendContract.previewExecutionDetails(previewResult)].filter(Boolean).join(" · ")}` : "";
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
  function restoreProjectPreviewTime(item,revision){
    const seconds=Number(item.projectPreviewTimeSeconds);if(!item.file||item.mediaType!=="video"||!Number.isFinite(seconds)||seconds<0)return;
    const restore=()=>{if(selectedUid!==item.uid||previewRevision!==revision||!previewFile)return;const duration=Number(ui.sourceVideo.duration);if(Number.isFinite(duration)&&duration>0&&seconds>=duration){projectStatus(`Saved preview time ${formatDuration(seconds)} is outside this source’s duration.`);return;}try{ui.sourceVideo.currentTime=seconds;updatePreview();}catch{projectStatus("The saved preview time could not be restored for this source.",true);}};
    if(Number(ui.sourceVideo.readyState)>=1)restore();else ui.sourceVideo.addEventListener("loadedmetadata",restore,{once:true});
  }
  function selectItem(item) {
    selectedUid = item.uid;
    invalidateProcessedClipIfStale("Source changed");
    ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; previewFile = item.file || null; previewResult = null; sourceError = ""; previewError=""; previewRevision++;
    resetZoomPan();compareSplit=50;ui.compareSlider.value="50";updateCompareSplit();
    ui.sourceName.textContent = item.name;
    if (previewFile) { previewUrl=URL.createObjectURL(previewFile); (isImage(previewFile) ? ui.sourceImage : ui.sourceVideo).src=previewUrl;restoreProjectPreviewTime(item,previewRevision); }
    setView("original"); updateQueue();
  }
  function addFiles(fileList) {
    if(projectBusy)return;
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
    if(projectBusy)return;
    if (item.jobId) hiddenJobs.add(item.jobId);
    items=items.filter((x) => x.uid !== item.uid);
    if (selectedUid===item.uid) { selectedUid=null;invalidateProcessedClipIfStale("Source removed"); previewFile=null; previewResult=null; sourceError=""; previewError=""; resetZoomPan();compareSplit=50;ui.compareSlider.value="50";ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src"); if(previewUrl)URL.revokeObjectURL(previewUrl); previewUrl=null; ui.sourceName.textContent="No media selected"; markPreviewStale(); }
  }
  function action(text, handler) { const button=el("button","row-button",text); button.type="button"; button.addEventListener("click",handler); return button; }
  function queueAction(text, ariaLabel, key, handler, disabled=false) { const button=action(text,handler);button.setAttribute("aria-label",ariaLabel);button.dataset.actionKey=key;button.disabled=disabled||queueMutationBusy||projectBusy;return button; }
  function queueDisplayItems(){
    const rank=(item)=>item.state==="processing"||item.state==="uploading"?0:item.state==="queued"?1:!item.jobId&&(item.state==="ready"||item.projectSourceSha256)?2:3;
    return items.map((item,index)=>({item,index})).sort((a,b)=>rank(a.item)-rank(b.item)||(rank(a.item)===1?(a.item.queuePosition??Number.MAX_SAFE_INTEGER)-(b.item.queuePosition??Number.MAX_SAFE_INTEGER):0)||a.index-b.index).map(({item})=>item);
  }
  function focusQueueItem(uid,key){setTimeout(()=>{const row=[...ui.queueRows.querySelectorAll(".queue-row")].find((node)=>node.dataset.itemUid===String(uid));const focusables=row?[...row.querySelectorAll("[data-action-key], [data-focus-key]")]:[];const target=(row&&(focusables.find((node)=>(node.dataset.actionKey===key||node.dataset.focusKey===key)&&!node.disabled)||focusables.find((node)=>!node.disabled)||row.querySelector(".file-name")))||ui.queueRows.querySelector(".file-name")||(!ui.applySettingsToReady.disabled?ui.applySettingsToReady:ui.clearQueueButton);target?.focus();},0);}
  function freezeReadyItem(item, settings=currentSettingsForPreset()){Object.assign(item,frontendContract.freezeReadyQueueItem(item,settings,namingCapability?currentOutputNameTemplate:"{name}_{layout}"));item.message=namingCapability?"Ready with frozen settings and name template":"Ready with frozen settings";}
  function moveLocalReady(item,direction){const key=`move-ready-${item.uid}-${direction}`;items=frontendContract.moveReadyQueueItem(items,item.uid,direction);queueWorkflowMessage=`Moved ${item.name} ${direction<0?"earlier":"later"} in the local ready-file order.`;updateQueue(item.uid,key);}
  function applySettingsToReady(){const ready=items.filter((item)=>!item.jobId&&item.state==="ready");if(!ready.length)return;if(!frontendContract.canSnapshotOutputNameTemplate(namingCapability,currentOutputNameTemplate)){toast("Fix the output name template before freezing ready files.",true);ui.outputNameTemplate.focus();return;}const snapshot=currentSettingsForPreset();ready.forEach((item)=>freezeReadyItem(item,snapshot));queueWorkflowMessage=namingCapability?`Applied and froze the current settings and name template for ${ready.length} ready file${ready.length===1?"":"s"}.`:`Applied and froze the current settings for ${ready.length} ready file${ready.length===1?"":"s"}.`;updateQueue();}
  function useJobSettings(item){if(!item.settingsSnapshot){toast(item.settingsSnapshotError||"A valid settings snapshot is not available from this server.",true);return;}let outputTemplate=null;if(namingCapability){if(!item.outputNameTemplateSnapshot){toast(item.namingSnapshotError||"A valid output name template snapshot is not available from this server.",true);return;}try{outputTemplate=frontendContract.validateOutputNameTemplate(item.outputNameTemplateSnapshot,namingCapability.max_template_length);}catch(error){toast(error.message,true);return;}}assignSettings(item.settingsSnapshot);if(outputTemplate){currentOutputNameTemplate=outputTemplate;ui.outputNameTemplate.value=outputTemplate;ui.outputNamingStatus.textContent=`Loaded name template from ${item.name}. No job was started.`;}performanceBaseline={...item.settingsSnapshot};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();renderModels();toast(`Loaded settings from ${item.name}. No job was started.`);updateQueue(item.uid,`use-snapshot-${item.uid||item.jobId}`);}
  function addQueueCopy(item,settings,kind,outputNameTemplate=currentOutputNameTemplate){try{const copy=frontendContract.makeReadyQueueCopy(item,settings,outputNameTemplate);copy.uid=crypto.randomUUID();items.splice(items.indexOf(item)+1,0,copy);selectedUid=copy.uid;selectItem(copy);queueWorkflowMessage=`${kind} added as a new ready file. Review it and choose Convert queue when ready.`;updateQueue(copy.uid,`use-current-${copy.uid}`);}catch(error){toast(error.message,true);}}
  function retryQueueItem(item){if(!item.file){toast("The original source is unavailable in this browser tab. Reselect the file to retry.",true);return;}addQueueCopy(item,item.settingsSnapshot||currentSettingsForPreset(),"Retry",frontendContract.outputNameTemplateForQueueCopy(item,"retry",currentOutputNameTemplate));}
  function duplicateQueueItem(item){if(!item.file){toast("The original source is unavailable in this browser tab. Reselect the file to duplicate it.",true);return;}addQueueCopy(item,currentSettingsForPreset(),"Duplicate",frontendContract.outputNameTemplateForQueueCopy(item,"duplicate",namingCapability?currentOutputNameTemplate:"{name}_{layout}"));}
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
  async function updateQueuedName(item,focusKey){
    if(!namingCapability||queueMutationBusy)return;
    let output_name_template;try{output_name_template=frontendContract.validateOutputNameTemplate(currentOutputNameTemplate,namingCapability.max_template_length);}catch(error){ui.outputNamingStatus.textContent=error.message;ui.outputNameTemplate.setAttribute("aria-invalid","true");ui.outputNameTemplate.focus();toast(error.message,true);return;}
    queueMutationBusy=true;queueWorkflowMessage=`Updating the output name for ${item.name}…`;updateQueue(item.uid,focusKey);
    try{const mutation=frontendContract.queuedNameUpdateRequest(item.jobId,output_name_template),job=await request(mutation.url,mutation.options);mergeJob(item,job);queueWorkflowMessage=`Updated the queued output name for ${item.name}.`;}
    catch(error){await queueMutationFailed(error,"Queued output name update");}
    finally{queueMutationBusy=false;updateQueue(item.uid,focusKey);}
  }
  async function removeQueuedItem(item,focusKey){
    if(!frontendContract.queueCapabilities(status)||queueMutationBusy)return;
    queueMutationBusy=true;queueWorkflowMessage=`Removing ${item.name} from the server queue…`;updateQueue(item.uid,focusKey);
    try{await request(`/api/jobs/${encodeURIComponent(item.jobId)}/remove-queued`,{method:"POST"});removeItem(item);saveHidden();queueWorkflowMessage=`Removed ${item.name} from the server queue.`;}
    catch(error){await queueMutationFailed(error,"Queued removal");}
    finally{queueMutationBusy=false;updateQueue(item.uid,focusKey);}
  }
  function updateQueue(focusUid=null,focusKey=null) {
    const disclosureState=new Map([...ui.queueRows.querySelectorAll("details[data-disclosure-key]")].map((details)=>[details.dataset.disclosureKey,details.open]));
    if(!focusUid&&ui.queueRows.contains(document.activeElement)){const active=document.activeElement,row=active.closest(".queue-row");if(row){focusUid=row.dataset.itemUid;focusKey=active.dataset.actionKey||active.dataset.focusKey||null;}}
    const pending=items.filter((x)=>!x.jobId&&x.state==="ready"),projectSources=items.filter((x)=>!x.jobId&&((x.state==="ready"&&x.file)||(x.projectSourceSha256&&!x.file))),missingProjectSources=items.filter((x)=>!x.jobId&&x.projectSourceSha256&&!x.file),block=frontendContract.queueReadiness(pending,currentSettingsForPreset(),{runtime,models,customModelAvailable,catalogError}),namingBlock=queueNamingBlock(pending),canFreezeName=frontendContract.canSnapshotOutputNameTemplate(namingCapability,currentOutputNameTemplate),caps=frontendContract.queueCapabilities(status);
    const videoBlocked=status&&!status.ffmpeg_found&&pending.some((x)=>x.mediaType==="video");
    ui.outputNamingControls.hidden=!namingCapability;
    if(namingCapability){ui.outputNameTemplate.maxLength=namingCapability.max_template_length*2;try{frontendContract.validateOutputNameTemplate(currentOutputNameTemplate,namingCapability.max_template_length);ui.outputNameTemplate.removeAttribute("aria-invalid");ui.outputNamingStatus.textContent="Ready to name new exports. The unique job ID and file extension are added automatically.";}catch(error){ui.outputNameTemplate.setAttribute("aria-invalid","true");ui.outputNamingStatus.textContent=error.message;}}
    ui.queueCount.textContent=String(items.length);ui.queueHeadingCount.textContent=`(${items.length})`;ui.convertCount.textContent=String(pending.length);ui.queueEmpty.hidden=items.length>0;
    ui.convertButton.disabled=!pending.length||submitting||projectBusy||Boolean(block)||Boolean(namingBlock)||Boolean(videoBlocked);ui.convertButton.firstElementChild.textContent=submitting?"Adding to local queue…":"✦ Convert queue";
    ui.convertHint.textContent=block||namingBlock||(videoBlocked?"FFmpeg is needed for the videos in your queue.":pending.length?`${pending.length} file${pending.length===1?"":"s"} ready. Unfrozen files use visible settings${namingCapability?" and the current name template":""} when submitted.`:"Add media to start a conversion.");
    ui.applySettingsToReady.textContent=namingCapability?"Apply settings + name to ready files":"Apply settings to ready files";
    ui.applySettingsToReady.disabled=!pending.length||!canFreezeName||projectBusy;ui.saveProject.disabled=projectBusy||projectSources.length===0;ui.openProject.disabled=projectBusy;ui.projectFileInput.disabled=projectBusy;ui.projectName.disabled=projectBusy;ui.browseButton.disabled=projectBusy;ui.clearQueueButton.disabled=projectBusy;ui.queueWorkflowStatus.textContent=queueWorkflowMessage||(missingProjectSources.length?`${missingProjectSources.length} saved-project source${missingProjectSources.length===1?" needs":"s need"} verification before conversion.`:(namingCapability?(caps?"Ready files use visible settings and the current name template at submit unless frozen. Server queue edits apply only to pending exports; clip slots stay in place.":"Ready files use visible settings and the current name template at submit unless frozen."):(caps?"Ready files use visible settings at submit unless frozen. Server queue edits apply only to pending exports; clip slots stay in place.":"Ready files use visible settings at submit unless frozen.")));
    ui.queueRows.replaceChildren();
    const queuedExports=eligibleQueuedExports(),displayItems=queueDisplayItems();
    for(const item of displayItems){
      const identity=String(item.uid||item.jobId),row=el("article",`queue-row${item.uid===selectedUid?" selected":""}`);row.dataset.itemUid=identity;
       const icon=el("div","file-icon",item.mediaType==="image"?"▧":"▶"),copy=el("div","file-copy"),name=el("button","file-name",item.name);name.type="button";name.disabled=projectBusy;name.dataset.actionKey=`select-${identity}`;name.title=item.name;name.setAttribute("aria-label",`Select ${item.name}`);name.setAttribute("aria-pressed",String(item.uid===selectedUid));name.addEventListener("click",()=>selectItem(item));
      const meta=el("div","file-meta"),state=el("span",`file-status ${item.state}`,item.message||({ready:"Ready to convert",uploading:"Uploading locally",completed:"Complete",failed:"Failed",canceled:"Canceled"}[item.state]||item.state));meta.append(state);
      if(item.depthCache){const cacheLabel=frontendContract.depthCacheStatusLabel(item.depthCache,formatBytes);if(cacheLabel)meta.append(el("span","",cacheLabel));}
      const cache=item.depthCache; if(cache){const cacheLabel=frontendContract.depthCacheStatusLabel(cache,formatBytes);if(cacheLabel)meta.append(el("span","",cacheLabel));}
      const info=[formatBytes(item.file?.size||item.size),item.layout].filter(Boolean).join(" · ");if(info)meta.append(el("span","",info));copy.append(name,meta);
      const execution=frontendContract.executionSummary(item.execution,item.pipeline);if(execution)copy.append(el("div","queue-execution",execution));else if(item.executionMode)copy.append(el("div","queue-execution",`Requested: ${{auto:"Auto",hybrid:"GPU + CPU",gpu:"NVIDIA GPU",cpu:"CPU only"}[item.executionMode]||item.executionMode} · awaiting execution`));
      if(item.depthCache?.warning)copy.append(el("small","queue-source-detail",`Depth cache warning: ${item.depthCache.warning.slice(0,240)}`));
      if(item.depthCacheError)copy.append(el("small","queue-source-detail",item.depthCacheError));
      if(item.depthCache?.warning)copy.append(el("small","queue-source-detail",`Depth cache warning: ${item.depthCache.warning.slice(0,240)}`));
      const runtimeDetails=progressDetails(item);if(runtimeDetails.length)copy.append(el("div","queue-runtime",runtimeDetails.join(" · ")));
      if(item.historyWarning)copy.append(el("small","queue-source-detail",`History warning: ${String(item.historyWarning).slice(0,240)}`));
      if(item.cleanupWarning)copy.append(el("small","queue-source-detail",`Cleanup warning: ${item.cleanupWarning}`));
      const preflight=frontendContract.preflightDetails(item.preflight,formatBytes);if(preflight.summary){const budget=el("div","queue-preflight");budget.append(el("div","queue-preflight-summary",preflight.summary));if(preflight.visibleWarning)budget.append(el("small","queue-preflight-warning",preflight.visibleWarning));const extraWarnings=preflight.warnings.filter((warning)=>warning!==preflight.visibleWarning);if(preflight.basis||extraWarnings.length){const details=el("details","queue-preflight-details");details.dataset.disclosureKey=`preflight-${identity}`;details.open=disclosureState.get(details.dataset.disclosureKey)??false;const summary=el("summary","","Estimate details");summary.dataset.focusKey=`${details.dataset.disclosureKey}-summary`;details.append(summary);if(preflight.basis)details.append(el("small","queue-preflight-basis",`Estimate basis: ${preflight.basis}`));for(const warning of extraWarnings)details.append(el("small","queue-preflight-warning",warning));budget.append(details);}copy.append(budget);}
      if(item.settingsSnapshot)copy.append(el("small","queue-settings-detail",`${item.jobId?"Server snapshot":item.settingsFrozen?"Frozen settings":"Unfrozen · uses current settings at submit"}: ${settingsDescription(item.settingsSnapshot)}`));
      else if(!item.jobId&&item.state==="ready")copy.append(el("small","queue-settings-detail","Unfrozen · uses the visible settings when you submit this file."));
      else if(item.jobId)copy.append(el("small","queue-settings-detail",item.settingsSnapshotError||"Settings snapshot unavailable; use a newer server for queue editing."));
      if(namingCapability){
        if(item.state==="completed")copy.append(el("small","queue-name-detail",item.outputName?`Output name: ${item.outputName}`:item.namingSnapshotError||"Completed job did not report its actual output name."));
        else if(item.outputName||item.plannedOutputName)copy.append(el("small","queue-name-detail",item.outputName?`Output name: ${item.outputName}`:`Planned output: ${item.plannedOutputName}`));
        if(item.outputNameTemplateSnapshot)copy.append(el("small","queue-name-detail",`${item.jobId?"Server name template":item.namingFrozen?"Frozen name template":"Name template"}: ${item.outputNameTemplateSnapshot}`));
        else if(!item.jobId&&item.state==="ready")copy.append(el("small","queue-name-detail",`Unfrozen · uses current template at submit: ${currentOutputNameTemplate}`));
        else if(item.jobId)copy.append(el("small","queue-name-detail",item.namingSnapshotError||"Name template snapshot unavailable from this server."));
      }
       if(item.projectSourceSha256&&!item.file)copy.append(el("small","queue-source-detail project-source-missing","Saved project source is not linked. Locate the original file; its content hash will be verified."));
       else if(item.jobId&&!item.file)copy.append(el("small","queue-source-detail",item.sourceAvailable===true?"Server has a readable working upload; this browser has no retry source file.":"Source is not retained for retry. Reselect the original file; filenames are not matched automatically."));
      else if(!item.jobId&&item.file)copy.append(el("small","queue-source-detail","Original file is held in this browser tab."));
      const actions=el("div","row-actions");
       if(item.projectSourceSha256&&!item.file)actions.append(queueAction("Locate source",`Locate and verify the saved-project source ${item.name}`,`locate-project-${identity}`,()=>promptProjectSource(item)));
      if(activeStates.has(item.state)){const progress=el("div","row-progress"),bar=el("span");progress.setAttribute("role","progressbar");progress.setAttribute("aria-label",`Conversion progress for ${item.name}`);progress.setAttribute("aria-valuemin","0");progress.setAttribute("aria-valuemax","100");progress.setAttribute("aria-valuenow",String(item.progress||0));bar.style.width=`${Math.max(2,Math.min(100,item.progress||0))}%`;progress.append(bar);actions.append(progress);}
      if(item.state==="completed"&&item.jobId){const link=el("a","row-button","Download ↓");link.dataset.actionKey=`download-${identity}`;link.href=`/api/jobs/${encodeURIComponent(item.jobId)}/download`;link.download=item.outputName||"";actions.append(link);}
      if(item.state==="completed"&&item.jobId){
        actions.append(queueAction("Reveal file",`Reveal output for ${item.name}`,`reveal-${identity}`,()=>workspaceAction(`/api/jobs/${encodeURIComponent(item.jobId)}/reveal`)));
        actions.append(queueAction("Save copy",`Save a copy of ${item.name} to the selected folder`,`copy-${identity}`,()=>saveOutputCopy(item)));
      }
      if(item.cleanupWarning&&item.jobId)actions.append(queueAction("Retry cleanup",`Retry temporary cleanup for ${item.name}`,`cleanup-${identity}`,async()=>{await workspaceAction(`/api/jobs/${encodeURIComponent(item.jobId)}/cleanup`);await pollJobs();}));
      if(item.jobId&&["queued","processing"].includes(item.state))actions.append(queueAction("Cancel",`Cancel active job ${item.name}`,`cancel-${identity}`,()=>cancelJob(item)));
      if(!item.jobId&&item.state==="ready"){
        const ready=items.filter((entry)=>!entry.jobId&&entry.state==="ready"),position=ready.indexOf(item);
        actions.append(queueAction("Move up",`Move ${item.name} earlier among ready files`,`move-ready-${item.uid}--1`,()=>moveLocalReady(item,-1),position===0));
        actions.append(queueAction("Move down",`Move ${item.name} later among ready files`,`move-ready-${item.uid}-1`,()=>moveLocalReady(item,1),position===ready.length-1));
        actions.append(queueAction(namingCapability?"Freeze settings + name":"Freeze settings",namingCapability?`Freeze current settings and output name template for ${item.name}`:`Freeze current settings for ${item.name}`,`use-current-${identity}`,()=>{freezeReadyItem(item);queueWorkflowMessage=namingCapability?`Froze the visible settings and name template for ${item.name}.`:`Froze the visible settings for ${item.name}.`;updateQueue(item.uid,`use-current-${identity}`);},!canFreezeName));
      }
      if(item.jobId&&caps&&item.state==="queued"&&item.queuePosition!==null){
        const remoteIndex=queuedExports.findIndex((job)=>job.id===item.jobId);
        actions.append(queueAction("Move up",`Move queued export ${item.name} earlier`,`move-remote-${identity}--1`,()=>reorderQueuedItem(item,-1,`move-remote-${identity}--1`),remoteIndex<=0));
        actions.append(queueAction("Move down",`Move queued export ${item.name} later`,`move-remote-${identity}-1`,()=>reorderQueuedItem(item,1,`move-remote-${identity}-1`),remoteIndex<0||remoteIndex===queuedExports.length-1));
        actions.append(queueAction("Use current settings",`Apply visible settings to queued export ${item.name}`,`use-current-${identity}`,()=>updateQueuedSettings(item,`use-current-${identity}`)));
        actions.append(queueAction("Remove queued",`Remove ${item.name} from the server queue`,`remove-queued-${identity}`,()=>removeQueuedItem(item,`remove-queued-${identity}`)));
      }
      if(item.jobId&&namingCapability&&item.state==="queued"&&frontendContract.isOrdinaryExportJob(item))actions.append(queueAction("Use current name",`Apply the current output name template to queued export ${item.name}`,`use-current-name-${identity}`,()=>updateQueuedName(item,`use-current-name-${identity}`),!canFreezeName));
      if(item.settingsSnapshot)actions.append(queueAction("Use settings",`Load settings snapshot from ${item.name} into the controls`,`use-snapshot-${identity}`,()=>useJobSettings(item)));
      if(["failed","canceled"].includes(item.state)&&item.file&&item.settingsSnapshot&&(!namingCapability||item.outputNameTemplateSnapshot))actions.append(queueAction("Retry as new job",`Retry ${item.name} as a new job using its original settings and name template`,`retry-${identity}`,()=>retryQueueItem(item)));
      if(item.jobId&&item.file)actions.append(queueAction("Duplicate with current settings",`Create a new ready copy of ${item.name} with current settings`,`duplicate-${identity}`,()=>duplicateQueueItem(item),!frontendContract.canSnapshotOutputNameTemplate(namingCapability,currentOutputNameTemplate)));
      if(!activeStates.has(item.state))actions.append(queueAction("Remove",`Remove ${item.name} from this browser queue`,`remove-local-${identity}`,()=>{removeItem(item);saveHidden();updateQueue(identity,`remove-local-${identity}`);}));
      row.append(icon,copy,actions);if(item.error)row.append(el("div","job-error",item.error));ui.queueRows.append(row);
    }
    refreshCalibrationControls();if(focusUid)focusQueueItem(focusUid,focusKey);
  }
  function saveHidden(){try{sessionStorage.setItem("stereolab.hiddenJobs",JSON.stringify([...hiddenJobs]));}catch{/* Keep this page usable without storage. */}}
  function getSettings(){const mode=ui.depthMode.value;const inputSize=mode==="dpt-hybrid-midas"?"384":mode==="precision"||/^depth-anything-v2-(small(?:-fp16)?|base|large)$/.test(mode)?ui.modelSize.value:"518";return {layout:ui.layoutSelect.value,strength:ui.strength.value,convergence:ui.convergence.value,depth_mode:mode,execution_mode:ui.executionMode.value,input_size:inputSize,resolution:ui.resolution.value,quality:ui.quality.value,temporal_smoothing:ui.temporalSmoothing.value,depth_interval:ui.depthInterval.value,video_decode:ui.videoDecode.value,use_depth_cache:String(ui.useDepthCache.checked),edge_refinement:String(ui.edgeRefinement.checked),invert_depth:String(ui.invertDepth.checked)};}
  function currentSettingsForPreset(){const raw=getSettings();return {layout:raw.layout,strength:Number(raw.strength),convergence:Number(raw.convergence),depth_mode:raw.depth_mode,execution_mode:raw.execution_mode,input_size:Number(raw.input_size),resolution:raw.resolution,quality:raw.quality,temporal_smoothing:Number(raw.temporal_smoothing),depth_interval:Number(raw.depth_interval),video_decode:raw.video_decode,use_depth_cache:ui.useDepthCache.checked,edge_refinement:ui.edgeRefinement.checked,invert_depth:ui.invertDepth.checked};}
  function settingsDescription(settings){const names={SBS:"Side by side · full",XREAL_SBS:"XREAL 1S · Full SBS",HALFSBS:"Side by side · half",TB:"Top / bottom · full",HALFTB:"Top / bottom · half",CROSS:"Cross-eye",ANAGLYPH:"Anaglyph · red / cyan",DEPTH:"Depth map",RGBD:"RGB + depth"};const models={"depth-anything-v2-small":"Small","depth-anything-v2-small-fp16":"Small FP16","depth-anything-v2-base":"Base","depth-anything-v2-large":"Large","dpt-hybrid-midas":"DPT Hybrid · MiDaS",balanced:"Visual approximation",precision:"Custom ONNX"};const devices={auto:"Auto",hybrid:"GPU + CPU · hybrid",gpu:"NVIDIA GPU",cpu:"CPU only"};const sizes=settings.depth_mode==="dpt-hybrid-midas"?"fixed 384 px":`${settings.input_size} px`;return `${names[settings.layout]} · strength ${Number(settings.strength).toFixed(1)} · convergence ${Number(settings.convergence).toFixed(2)} · ${models[settings.depth_mode]} · ${sizes} · ${devices[settings.execution_mode]} · ${settings.resolution==="source"?"Match source":"Up to 1080p"} · ${settings.quality} encoding · ${settings.depth_interval===2?"alternate-frame depth + flow":"depth on every frame"} · ${settings.video_decode==="amd"?"AMD decode requested":"software decode"} · video depth cache ${settings.use_depth_cache?"on":"off"} · smoothing ${Number(settings.temporal_smoothing).toFixed(2)} · edge refinement ${settings.edge_refinement?"on":"off"} · invert ${settings.invert_depth?"on":"off"}`;}
  function refreshPerformancePresetStatus(){const current=currentSettingsForPreset(),dirty=frontendContract.hasUnsavedSettings(performanceBaseline,current);ui.performancePresetStatus.textContent=`${importedPresetBaseline?"Imported settings":ui.performancePreset.options[ui.performancePreset.selectedIndex]?.text || "Custom settings"} · ${dirty?"Unsaved changes":"Saved preset"}`;ui.performancePresetSummary.textContent=`Exact settings: ${settingsDescription(current)}`;ui.resetPerformancePreset.disabled=ui.performancePreset.value==="custom";}
  function assignSettings(settings){ui.layoutSelect.value=settings.layout;ui.strength.value=String(settings.strength);ui.convergence.value=String(settings.convergence);ui.depthMode.value=settings.depth_mode;ui.executionMode.value=settings.execution_mode;ui.modelSize.value=String(settings.input_size);ui.resolution.value=settings.resolution;ui.quality.value=settings.quality;ui.temporalSmoothing.value=String(settings.temporal_smoothing);ui.depthInterval.value=String(settings.depth_interval??1);ui.videoDecode.value=settings.video_decode??"software";ui.useDepthCache.checked=settings.use_depth_cache??false;ui.edgeRefinement.checked=settings.edge_refinement;ui.invertDepth.checked=settings.invert_depth;}
  function writeNamedPresets(){try{window.localStorage.setItem(namedPresetsKey,JSON.stringify({schema_version:1,presets:namedPresets}));return true;}catch{ui.savedPresetStatus.textContent="Browser storage is unavailable; this change was not saved.";return false;}}
  function refreshNamedPresetOptions(selectedName=""){
    const options=[el("option","","Choose a saved preset")];options[0].value="";
    const ordered=[...namedPresets].sort((a,b)=>Number(b.favorite)-Number(a.favorite)||a.name.localeCompare(b.name,undefined,{sensitivity:"base"}));
    for(const preset of ordered){const option=el("option","",`${preset.favorite?"★ ":""}${preset.name}`);option.value=preset.name;options.push(option);}
    ui.savedPresetSelect.replaceChildren(...options);ui.savedPresetSelect.value=selectedName;
    const selected=namedPresets.find((preset)=>preset.name===selectedName);
    ui.savedPresetName.value=selected?.name||"";ui.savedPresetFavorite.checked=Boolean(selected?.favorite);
    ui.deleteNamedPreset.disabled=!selected;ui.applyNamedPreset.disabled=!selected;
  }
  function loadNamedPresets(){
    try{
      const raw=window.localStorage.getItem(namedPresetsKey);if(!raw){namedPresets=[];refreshNamedPresetOptions();return;}
      const parsed=JSON.parse(raw);if(parsed?.schema_version!==1||!Array.isArray(parsed.presets)||parsed.presets.length>namedPresetLimit)throw new Error("Saved presets have an invalid format.");
      const names=new Set();namedPresets=parsed.presets.map((preset)=>{
        if(!preset||typeof preset.name!=="string"||typeof preset.favorite!=="boolean")throw new Error("A saved preset has invalid fields.");
        const name=preset.name.trim();if(!name||name.length>namedPresetNameLimit||/[\x00-\x1f\x7f]/.test(name))throw new Error("A saved preset name is invalid.");
        const identity=name.toLocaleLowerCase();if(names.has(identity))throw new Error("Saved preset names must be unique.");names.add(identity);
        return {name,favorite:preset.favorite,settings:frontendContract.validateSettings(preset.settings)};
      });
      ui.savedPresetStatus.textContent=`${namedPresets.length} of ${namedPresetLimit} presets saved on this device.`;refreshNamedPresetOptions();
    }catch(error){namedPresets=[];refreshNamedPresetOptions();ui.savedPresetStatus.textContent=`Saved presets could not be loaded: ${error.message}`;}
  }
  function selectNamedPreset(){const preset=namedPresets.find((entry)=>entry.name===ui.savedPresetSelect.value);ui.savedPresetName.value=preset?.name||"";ui.savedPresetFavorite.checked=Boolean(preset?.favorite);ui.deleteNamedPreset.disabled=!preset;ui.applyNamedPreset.disabled=!preset;}
  function saveNamedPreset(){
    const name=String(ui.savedPresetName.value||"").trim();
    if(!name||name.length>namedPresetNameLimit||/[\x00-\x1f\x7f]/.test(name)){ui.savedPresetStatus.textContent=`Enter a name of 1–${namedPresetNameLimit} characters.`;ui.savedPresetName.focus();return;}
    const identity=name.toLocaleLowerCase(),index=namedPresets.findIndex((preset)=>preset.name.toLocaleLowerCase()===identity);
    if(index<0&&namedPresets.length>=namedPresetLimit){ui.savedPresetStatus.textContent=`You can save up to ${namedPresetLimit} presets. Delete one to save another.`;return;}
    const preset={name,favorite:ui.savedPresetFavorite.checked,settings:frontendContract.validateSettings(currentSettingsForPreset())};
    const updated=[...namedPresets];if(index<0)updated.push(preset);else updated[index]=preset;
    const previous=namedPresets;namedPresets=updated;if(!writeNamedPresets()){namedPresets=previous;return;}
    refreshNamedPresetOptions(name);performanceBaseline={...preset.settings};importedPresetBaseline=false;ui.performancePreset.value="custom";ui.presetSelect.value="custom";ui.savedPresetStatus.textContent=`Saved “${name}” on this device.`;refreshPerformancePresetStatus();
  }
  function applyNamedPreset(){
    const preset=namedPresets.find((entry)=>entry.name===ui.savedPresetSelect.value);if(!preset)return;
    assignSettings(frontendContract.validateSettings(preset.settings));performanceBaseline={...preset.settings};importedPresetBaseline=false;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();ui.savedPresetStatus.textContent=`Applied “${preset.name}”. No job was started.`;
  }
  function deleteNamedPreset(){
    const name=ui.savedPresetSelect.value;if(!name)return;const previous=namedPresets;namedPresets=namedPresets.filter((preset)=>preset.name!==name);
    if(!writeNamedPresets()){namedPresets=previous;return;}refreshNamedPresetOptions();ui.savedPresetStatus.textContent=`Deleted “${name}”. Current settings were not changed.`;
  }
  function applyPerformancePreset(name){const settings=frontendContract.performancePreset(name);if(!settings)return;importedPresetBaseline=false;performanceBaseline=settings;assignSettings(settings);ui.performancePreset.value=name;ui.presetSelect.value="custom";updateSettings();}
  function resetPerformancePreset(){if(ui.performancePreset.value==="custom")return;applyPerformancePreset(ui.performancePreset.value);}
  function exportPresetFile(){try{const preset=frontendContract.exportUserPreset(currentSettingsForPreset());const blob=new Blob([JSON.stringify(preset,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="stereolab-settings-v1.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),0);toast("Settings preset exported.");}catch(error){toast(error.message,true);}}
  async function importPresetFile(file){if(!file)return;try{const parsed=JSON.parse(await file.text()),settings=frontendContract.validateUserPreset(parsed);assignSettings(settings);performanceBaseline={...settings};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();toast("Settings imported and validated. No model was installed or downloaded.");}catch(error){toast(`Preset not applied: ${error.message}`,true);}finally{ui.importPresetFile.value="";}}
  function projectExactKeys(value,keys){if(!value||typeof value!=="object"||Array.isArray(value))throw new Error("Project data must be an object.");const actual=Object.keys(value).sort(),expected=[...keys].sort();if(actual.length!==expected.length||actual.some((key,index)=>key!==expected[index]))throw new Error("Project data has unsupported or missing fields.");}
  function projectUuid(value){return typeof value==="string"&&/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);}
  function projectMediaType(name){const ext=(String(name).split(".").pop()||"").toLowerCase();if(imageExtensions.has(ext))return "image";if(videoExtensions.has(ext))return "video";return null;}
  function projectSafeBasename(value){return typeof value==="string"&&value.length>0&&value.length<=255&&value===value.trim()&&!/[\\/:\u0000-\u001f\u007f]/.test(value)&&value!=="."&&value!=="..";}
  function validateProjectDocument(value){
    projectExactKeys(value,["schema_version","project_id","project_name","created_at_ms","settings","output_name_template","selected_source_id","preview_time_seconds","sources"]);
    if(value.schema_version!==PROJECT_SCHEMA_VERSION||!projectUuid(value.project_id))throw new Error("This project version or identity is not supported.");
    if(typeof value.project_name!=="string"||!value.project_name.trim()||value.project_name.length>80||/[\\/:\u0000-\u001f\u007f]/.test(value.project_name))throw new Error("Project name must be 1–80 characters without path or control characters.");
    if(!Number.isSafeInteger(value.created_at_ms)||value.created_at_ms<0)throw new Error("Project creation time is invalid.");
    const settings=frontendContract.validateSettings(value.settings),maxTemplate=namingCapability?.max_template_length??120;
    const outputNameTemplate=frontendContract.validateOutputNameTemplate(value.output_name_template,maxTemplate);
    if(!Array.isArray(value.sources)||value.sources.length<1||value.sources.length>PROJECT_MAX_SOURCES)throw new Error(`A project must contain 1–${PROJECT_MAX_SOURCES} sources.`);
    const ids=new Set();let totalBytes=0;
    const sources=value.sources.map(source=>{
      projectExactKeys(source,["source_id","name","media_type","size_bytes","sha256","settings","output_name_template"]);
      if(!projectUuid(source.source_id)||ids.has(source.source_id))throw new Error("Project source identities must be unique UUIDs.");ids.add(source.source_id);
      if(!projectSafeBasename(source.name)||projectMediaType(source.name)!==source.media_type)throw new Error("A project source needs a supported basename and matching media type.");
      if(!Number.isSafeInteger(source.size_bytes)||source.size_bytes<1||source.size_bytes>PROJECT_MAX_SOURCE_BYTES)throw new Error("A project source size is invalid or exceeds the 8 GiB upload limit.");
      totalBytes+=source.size_bytes;if(!Number.isSafeInteger(totalBytes)||totalBytes>PROJECT_MAX_TOTAL_BYTES)throw new Error("Project sources exceed the 64 GiB verification limit.");
      if(typeof source.sha256!=="string"||!/^[a-f0-9]{64}$/.test(source.sha256))throw new Error("A project source fingerprint is invalid.");
      const sourceSettings=frontendContract.validateSettings(source.settings);
      const sourceTemplate=source.output_name_template===null?null:frontendContract.validateOutputNameTemplate(source.output_name_template,maxTemplate);
      return {source_id:source.source_id,name:source.name,media_type:source.media_type,size_bytes:source.size_bytes,sha256:source.sha256,settings:sourceSettings,output_name_template:sourceTemplate};
    });
    if(value.selected_source_id!==null&&(!projectUuid(value.selected_source_id)||!ids.has(value.selected_source_id)))throw new Error("The selected preview source is missing from this project.");
    const selected=value.selected_source_id===null?null:sources.find(source=>source.source_id===value.selected_source_id);
    const previewTime=value.preview_time_seconds;
    if(previewTime!==null&&(!Number.isFinite(previewTime)||previewTime<0||previewTime>PROJECT_PREVIEW_TIME_LIMIT))throw new Error("The saved preview timestamp is outside the supported range.");
    if((!selected||selected.media_type==="image")&&previewTime!==null)throw new Error("Only a selected video can have a preview timestamp.");
    return {schema_version:PROJECT_SCHEMA_VERSION,project_id:value.project_id,project_name:value.project_name.trim(),created_at_ms:value.created_at_ms,settings,output_name_template:outputNameTemplate,selected_source_id:value.selected_source_id,preview_time_seconds:previewTime,sources};
  }
  function hashLocalFile(file,onProgress=()=>{}){
    return new Promise((resolve,reject)=>{
      let worker;
      try{worker=new Worker("/static/source_hash_worker.js?v=stereolab-projects-1");}catch(error){reject(new Error(`Could not start the source fingerprint worker: ${error.message||error}`));return;}
      let settled=false;
      const finish=(error,value)=>{if(settled)return;settled=true;worker.terminate();error?reject(error):resolve(value);};
      worker.onmessage=event=>{const data=event.data||{};if(typeof data.progress==="number"){onProgress(data.progress,data.total);return;}if(typeof data.error==="string"){finish(new Error(data.error));return;}if(typeof data.sha256==="string"&&/^[a-f0-9]{64}$/.test(data.sha256)&&data.size_bytes===file.size){finish(null,data.sha256);return;}finish(new Error("The source fingerprint worker returned an invalid result."));};
      worker.onerror=event=>finish(new Error(String(event.message||"Source fingerprinting failed.")));
      try{worker.postMessage({file});}catch(error){finish(new Error(`Could not read this source: ${error.message||error}`));}
    });
  }
  function projectStatus(message,error=false){ui.projectStatus.textContent=message;ui.projectStatus.classList.toggle("error",error);}
  function projectReadySources(){return items.filter(item=>!item.jobId&&((item.state==="ready"&&item.file)||(item.projectSourceSha256&&!item.file)));}
  async function saveProject(){
    if(projectBusy)return;
    const sources=projectReadySources();if(!sources.length){projectStatus("Add at least one ready source before saving a project.",true);return;}
    if(sources.length>PROJECT_MAX_SOURCES||sources.reduce((sum,item)=>sum+Number(item.file?.size||item.size||0),0)>PROJECT_MAX_TOTAL_BYTES){projectStatus("This project exceeds its 200-source or 64 GiB verification limit.",true);return;}
    let projectName=String(ui.projectName.value||"").trim();if(!projectName)projectName=(sources[0].name||"StereoLab project").replace(/\.[^.]+$/g,"").slice(0,80)||"StereoLab project";
    if(projectName.length>80||/[\\/:\u0000-\u001f\u007f]/.test(projectName)){projectStatus("Project name must be at most 80 characters and cannot contain path characters.",true);ui.projectName.focus();return;}
    const settings=currentSettingsForPreset(),maxTemplate=namingCapability?.max_template_length??120;
    let outputNameTemplate;try{outputNameTemplate=frontendContract.validateOutputNameTemplate(currentOutputNameTemplate,maxTemplate);}catch(error){projectStatus(`Fix the output name template first: ${error.message}`,true);ui.outputNameTemplate.focus();return;}
    projectBusy=true;updateQueue();projectStatus(`Preparing ${sources.length} source${sources.length===1?"":"s"}…`);
    const records=[],idsByUid=new Map(),fingerprints=new Map();
    try{
      for(let index=0;index<sources.length;index++){
        const item=sources[index],mediaType=projectMediaType(item.name),file=item.file||null,size=Number(file?.size??item.size),sourceId=item.projectSourceId||crypto.randomUUID();
        if(!projectSafeBasename(item.name)||!mediaType||mediaType!==item.mediaType||!Number.isSafeInteger(size)||size<1||size>PROJECT_MAX_SOURCE_BYTES)throw new Error(`Source “${item.name}” has invalid identity or size.`);
        if(!projectUuid(sourceId)||records.some(record=>record.source_id===sourceId))throw new Error("Project source identity generation produced a duplicate.");
        let sha256=item.projectSourceSha256||null;
        if(file){
          if(file.size!==size)throw new Error(`Source “${item.name}” changed size before it could be saved.`);
          projectStatus(`Fingerprinting source ${index+1} of ${sources.length}: ${item.name} · ${formatBytes(0)} of ${formatBytes(size)}`);
          const computed=await hashLocalFile(file,(loaded,total)=>projectStatus(`Fingerprinting source ${index+1} of ${sources.length}: ${item.name} · ${formatBytes(loaded)} of ${formatBytes(total)}`));
          if(sha256&&computed!==sha256)throw new Error(`Source “${item.name}” no longer matches its saved project fingerprint.`);
          sha256=computed;fingerprints.set(item.uid,computed);
        }
        if(typeof sha256!=="string"||!/^[a-f0-9]{64}$/.test(sha256))throw new Error(`Source “${item.name}” has no verified fingerprint.`);
        const sourceSettings=frontendContract.validateSettings(item.settingsSnapshot||settings),sourceTemplate=item.outputNameTemplateSnapshot?frontendContract.validateOutputNameTemplate(item.outputNameTemplateSnapshot,maxTemplate):null;
        const record={source_id:sourceId,name:item.name,media_type:mediaType,size_bytes:size,sha256,settings:sourceSettings,output_name_template:sourceTemplate};
        records.push(record);idsByUid.set(item.uid,sourceId);
      }
      const selectedItem=sources.find(item=>item.uid===selectedUid),selectedSourceId=selectedItem?idsByUid.get(selectedItem.uid):null;
      let previewTime=null;
      if(selectedItem&&selectedItem.mediaType==="video"){
        const candidate=selectedItem.file?Number(ui.sourceVideo.currentTime):Number(selectedItem.projectPreviewTimeSeconds??0);
        previewTime=Number.isFinite(candidate)?Math.max(0,Math.min(PROJECT_PREVIEW_TIME_LIMIT,candidate)):0;
      }
      const document=validateProjectDocument({schema_version:PROJECT_SCHEMA_VERSION,project_id:crypto.randomUUID(),project_name:projectName,created_at_ms:Date.now(),settings,output_name_template:outputNameTemplate,selected_source_id:selectedSourceId,preview_time_seconds:previewTime,sources:records});
      const serialized=JSON.stringify(document,null,2);if(new TextEncoder().encode(serialized).length>PROJECT_MAX_BYTES)throw new Error("The project file exceeds the 2 MiB size limit.");
      const blob=new Blob([serialized],{type:"application/json"}),url=URL.createObjectURL(blob),link=documentCreateProjectLink(url,projectName);link.click();setTimeout(()=>URL.revokeObjectURL(url),0);
      for(const item of sources){const sourceId=idsByUid.get(item.uid);item.projectSourceId=sourceId;if(fingerprints.has(item.uid)){item.projectSourceSha256=fingerprints.get(item.uid);item.projectSourceSizeBytes=item.file.size;}if(item.uid===selectedUid)item.projectPreviewTimeSeconds=previewTime;}
      ui.projectName.value=projectName;projectStatus(`Saved “${projectName}”. Media is not copied; reopening will verify each source before adding it to the queue.`);
    }catch(error){projectStatus(`Project was not saved: ${String(error.message||error).slice(0,240)}`,true);}
    finally{projectBusy=false;updateQueue();}
  }
  function documentCreateProjectLink(url,projectName){const link=document.createElement("a"),safeName=projectName.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,"").slice(0,64)||"stereolab-project";link.href=url;link.download=`${safeName}.stereolab-project.json`;return link;}
  async function openProjectFile(file){
    if(!file||projectBusy)return;
    projectBusy=true;updateQueue();projectStatus("Reading and validating project…");
    try{
      if(!Number.isSafeInteger(file.size)||file.size<2||file.size>PROJECT_MAX_BYTES)throw new Error("Project files must be between 2 bytes and 2 MiB.");
      const project=validateProjectDocument(JSON.parse(await file.text()));
      if(items.length+project.sources.length>PROJECT_MAX_QUEUE_ENTRIES)throw new Error(`Opening this project would exceed the ${PROJECT_MAX_QUEUE_ENTRIES}-row local queue limit.`);
      const recovered=project.sources.map(source=>({uid:crypto.randomUUID(),projectSourceId:crypto.randomUUID(),projectSourceSha256:source.sha256,projectSourceSizeBytes:source.size_bytes,projectPreviewTimeSeconds:source.source_id===project.selected_source_id?project.preview_time_seconds:null,name:source.name,size:source.size_bytes,mediaType:source.media_type,file:null,state:"missing",progress:0,message:"Locate the exact source to add it to the ready queue",settingsSnapshot:source.settings,settingsFrozen:true,outputNameTemplateSnapshot:source.output_name_template,namingFrozen:Boolean(source.output_name_template)}));
      if(new Set(recovered.map(item=>item.uid)).size!==recovered.length||new Set(recovered.map(item=>item.projectSourceId)).size!==recovered.length)throw new Error("Could not create unique local identities for project sources.");
      const selectedRecord=project.selected_source_id?project.sources.find(source=>source.source_id===project.selected_source_id):null;
      items.push(...recovered);assignSettings(project.settings);performanceBaseline={...project.settings};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";
      currentOutputNameTemplate=project.output_name_template;ui.outputNameTemplate.value=project.output_name_template;ui.projectName.value=project.project_name;updateSettings(false);
      if(selectedRecord){const selected=recovered[project.sources.indexOf(selectedRecord)];selectItem(selected);}
      const missing=recovered.length;projectStatus(`Opened “${project.project_name}”. ${missing} source${missing===1?" needs":"s need"} verification. No conversion was started.`);updateQueue();
    }catch(error){projectStatus(`Project was not opened: ${String(error.message||error).slice(0,240)}`,true);}
    finally{projectBusy=false;updateQueue();ui.projectFileInput.value="";}
  }
  function promptProjectSource(item){const picker=document.createElement("input");picker.type="file";picker.accept=".mp4,.webm,.mkv,.mov,.avi,.wmv,.flv,.ts,.mts,.m2ts,.m4v,.jpg,.jpeg,.png,.webp";picker.addEventListener("change",()=>{const file=picker.files?.[0];if(file)relinkProjectSource(item,file);});picker.click();}
  async function relinkProjectSource(item,file){
    if(projectBusy||!file||!items.includes(item)||!item.projectSourceSha256)return;
    const mediaType=projectMediaType(file.name);if(!mediaType||mediaType!==item.mediaType){projectStatus(`Choose a supported ${item.mediaType} file for “${item.name}”.`,true);return;}
    if(file.size!==item.projectSourceSizeBytes){projectStatus(`“${file.name}” has a different size. The saved source remains unresolved.`,true);return;}
    projectBusy=true;item.message="Verifying source fingerprint";updateQueue();projectStatus(`Verifying “${item.name}” in bounded chunks…`);
    try{
      const sha256=await hashLocalFile(file,(loaded,total)=>projectStatus(`Verifying “${item.name}” · ${formatBytes(loaded)} of ${formatBytes(total)}`));
      if(sha256!==item.projectSourceSha256)throw new Error("The file content does not match this saved project. The source remains unresolved.");
      if(!items.includes(item))return;
      item.file=file;item.size=file.size;item.state="ready";item.message="Source verified · ready to convert";
      if(selectedUid===item.uid)selectItem(item);
      const remaining=items.filter(entry=>entry.projectSourceSha256&&!entry.file).length;
      projectStatus(remaining?`Verified “${item.name}”. ${remaining} project source${remaining===1?" remains":"s remain"} to locate.`:`Verified “${item.name}”. Every project source is ready; review the queue before converting.`);
    }catch(error){if(items.includes(item)&&!item.file)item.message="Source not verified; locate the original file";projectStatus(`Source was not linked: ${String(error.message||error).slice(0,220)}`,true);}
    finally{projectBusy=false;updateQueue();}
  }
  async function measureInputDetail(){const blocked=calibrationBlock();if(blocked){toast(blocked,true);return;}ui.sourceVideo.pause();const context=calibrationContext();calibrationBusy=true;calibrationResult=null;calibrationResultContext=null;calibrationNotice="";ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.calibrationStatus.textContent="Measuring depth-input inference. One in-flight inference may exceed the selected budget…";updatePreview();let discarded=false;
    try{const source=isImage(previewFile)?ui.sourceImage:ui.sourceVideo,w=source.naturalWidth||source.videoWidth,h=source.naturalHeight||source.videoHeight;if(!w||!h)throw new Error("The selected source has no decoded frame.");const scale=Math.min(1,960/w,540/h),canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));drawSourceFrame(source,canvas);const blob=await new Promise((resolve,reject)=>canvas.toBlob((value)=>value?resolve(value):reject(new Error("Could not capture this frame.")),"image/jpeg",.94));if(!frontendContract.calibrationContextMatches(context,calibrationContext())){discarded=true;calibrationNotice="Measurement discarded because the source frame, settings, or budget changed before it started.";return;}const data=new FormData();data.append("file",blob,"calibration-frame.jpg");data.append("depth_mode",ui.depthMode.value);data.append("execution_mode",ui.executionMode.value);data.append("budget_seconds",String(context.budget));const report=frontendContract.validateCalibrationReport(await request("/api/calibration",{method:"POST",body:data}));if(!frontendContract.calibrationContextMatches(context,calibrationContext())){discarded=true;calibrationNotice="Measurement discarded because the source frame, settings, or budget changed while it was running.";return;}renderCalibrationReport(report);}
    catch(error){calibrationNotice=`Measurement failed: ${String(error.message||error).slice(0,260)}`;ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.calibrationStatus.textContent=calibrationNotice;toast(calibrationNotice,true);}
    finally{calibrationBusy=false;updatePreview();if(discarded)ui.calibrationStatus.textContent=calibrationNotice;}}
  function applyMeasuredInputDetail(){if(!calibrationResult||!frontendContract.calibrationApplicable(calibrationResult)||!frontendContract.calibrationContextMatches(calibrationResultContext,calibrationContext())){invalidateCalibrationIfStale();ui.applyCalibration.hidden=true;return;}const size=calibrationResult.fastest_measured_input_size;calibrationResult=null;calibrationResultContext=null;ui.calibrationReport.hidden=true;ui.applyCalibration.hidden=true;ui.modelSize.value=String(size);calibrationNotice=`Applied ${size} px from this one-frame depth-input measurement. Lower detail can lose fine edges.`;updateSettings();ui.calibrationStatus.textContent=calibrationNotice;toast(calibrationNotice);}
  function updateSettings(stale=true) {
    renderDeviceNote();
    const layout=layouts[ui.layoutSelect.value];ui.formatHint.textContent=layout[1];ui.previewModeLabel.textContent=layout[0].toUpperCase();
    ui.strengthValue.value=Number(ui.strength.value).toFixed(1);ui.convergenceValue.value=Number(ui.convergence.value).toFixed(2);ui.smoothingValue.value=Number(ui.temporalSmoothing.value).toFixed(2);
    const mode=ui.depthMode.value,model=selectedModel();const selectableInputSize=mode==="precision"||/^depth-anything-v2-(small(?:-fp16)?|base|large)$/.test(mode);ui.modelRow.hidden=mode!=="precision";ui.modelSizeRow.hidden=!selectableInputSize;ui.modelSizeLabel.textContent=mode==="precision"?"Custom model input size":"Neural input detail";ui.midasSizeNote.hidden=mode!=="dpt-hybrid-midas";
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
      else {const scale=Math.min(1,960/w,540/h),canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));drawSourceFrame(source,canvas);const blob=await new Promise((resolve,reject)=>canvas.toBlob((value)=>value?resolve(value):reject(new Error("Could not capture this frame.")),"image/jpeg",.94));data.append("file",blob,"preview-frame.jpg");}
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
    const actual=runtime.last_execution?.requested_mode ? `Last execution: ${frontendContract.executionSummary(runtime.last_execution)}` : runtime.active_provider ? `Last provider: ${runtime.active_provider}` : "No inference has run in this runtime";
    ui.runtimeDescription.textContent=runtime.available?`ONNX Runtime ${runtime.version || ""} ready. Providers: ${providers}. ${capability}. ${actual}.${runtime.error ? ` ${runtime.error}` : ""}`:`ONNX Runtime is missing. Install the project dependencies, then restart StereoLab.${runtime.error ? ` ${runtime.error}` : ""}`;
    ui.runtimeCommand.hidden=Boolean(runtime.available);
  }
  function renderHistory(){
    ui.historyPanel.hidden=!historyCapabilities;ui.historyRows.replaceChildren();ui.historyMore.hidden=true;
    if(!historyCapabilities)return;
    const caps=historyCapabilities;
    ui.historyRefresh.disabled=!caps.enabled;ui.historyMore.disabled=!caps.enabled;
    if(!caps.enabled){ui.historyStatus.textContent="History is disabled for this server. Set STEREOLAB_HISTORY_INSTANCE and restart to enable prior-launch records.";return;}
    const warning=typeof caps.warning==="string"&&caps.warning.trim()?` ${caps.warning.trim().slice(0,240)}`:"";
    ui.historyStatus.textContent=`${caps.state==="degraded"?"History storage is degraded; saved records may be incomplete.":"Showing exports from previous launches."}${warning}`;
    for(const item of historyItems){
      const row=el("article","history-row"),copy=el("div","history-copy"),title=el("strong","history-source",item.filename);
      const date=new Date(item.created_at*1000),completed=item.completed_at==null?"":` · Completed ${new Date(item.completed_at*1000).toLocaleString()}`;copy.append(title,el("small","history-meta",`Started ${Number.isNaN(date.getTime())?"at an unknown time":date.toLocaleString()}${completed} · ${frontendContract.historyStateLabel(item)} · ${formatBytes(item.source_size_bytes)||"size unavailable"}`));
      if(item.state==="interrupted")copy.append(el("small","history-note",`Interrupted during ${item.last_known_state}. Select the source to create a new export from the beginning.`));
      copy.append(el("small","history-settings",`Settings: ${settingsDescription(item.settings)}`));
      copy.append(el("small","history-settings",`Name template: ${item.output_name_template} · ${item.output_name||item.planned_output_name}`));
      copy.append(el("small","history-note","Source files are not retained. Reselect the source to create a new export; no file is matched automatically."));
      const actions=el("div","history-actions");
      const use=el("button","row-button","Use settings and reselect source");use.type="button";use.addEventListener("click",()=>useHistorySettings(item));actions.append(use);
      if(item.output_available){const link=el("a","row-button","Download ↓");link.href=`/api/history/${encodeURIComponent(item.job_id)}/download`;link.download=item.output_name;actions.append(link);}
      else if(item.state==="completed")actions.append(el("small","history-unavailable","Output unavailable"));
      row.append(copy,actions);ui.historyRows.append(row);
    }
    if(historyItems.length===0)ui.historyRows.append(el("p","history-empty","No exports from previous launches are available."));
    ui.historyMore.hidden=!historyCursor;
  }
  let historyItems=[];
  async function loadHistory(append=false){
    if(!historyCapabilities?.enabled||historyLoading)return;
    historyLoading=true;ui.historyRefresh.disabled=true;ui.historyMore.disabled=true;
    try{
      const before=append?historyCursor:null,url=`/api/history?limit=50${before?`&before=${encodeURIComponent(before)}`:""}`;
      const page=frontendContract.validateHistoryPage(await request(url));
      historyItems=append?[...historyItems,...page.items]:page.items;historyCursor=page.next_cursor;
      renderHistory();
      const warning=historyCapabilities.state==="degraded"?`${historyCapabilities.warning||"History storage is degraded."} `:"";
      ui.historyStatus.textContent=`${warning}${page.invalid_records?`${page.invalid_records} invalid record${page.invalid_records===1?" was":"s were"} hidden and preserved. `:""}Showing exports from previous launches.`;
    }catch(error){
      ui.historyMore.hidden=!historyCursor;
      ui.historyStatus.textContent=error.status===409?"History is disabled for this server.":`History is unavailable: ${String(error.message||error).slice(0,240)}`;
    }finally{historyLoading=false;ui.historyRefresh.disabled=!historyCapabilities?.enabled;ui.historyMore.disabled=!historyCapabilities?.enabled;}
  }
  function useHistorySettings(item){
    try{
      const settings=frontendContract.validateSettings(item.settings),template=frontendContract.validateOutputNameTemplate(item.output_name_template);
      assignSettings(settings);ui.outputNameTemplate.value=template;currentOutputNameTemplate=template;
      performanceBaseline={...settings};importedPresetBaseline=true;ui.performancePreset.value="custom";ui.presetSelect.value="custom";updateSettings();
      ui.historyStatus.textContent="Validated settings and naming loaded. Source files are not retained: explicitly select a source, then submit a new export.";
      ui.mediaInput.focus();
    }catch(error){ui.historyStatus.textContent=`Saved settings could not be applied: ${String(error.message||error).slice(0,200)}`;}
  }
  async function loadStatus(preserveHistory=false){const previousHistoryCapabilities=historyCapabilities;try{status=await request("/api/status");statusLoadError="";historyCapabilities=frontendContract.historyCapability(status);if(historyCapabilities){if(!preserveHistory)historyItems=[];historyCursor=null;renderHistory();if(historyCapabilities.enabled)await loadHistory();}else{historyItems=[];historyCursor=null;ui.historyPanel.hidden=true;}namingCapability=frontendContract.outputNamingCapability(status);ui.outputNamingControls.hidden=!namingCapability;if(namingCapability)ui.outputNameTemplate.maxLength=namingCapability.max_template_length*2;customModelAvailable=Boolean(status.model_available);ui.modelState.textContent=customModelAvailable?(status.model_name||"Custom model loaded"):"No model loaded";runtime=status.runtime_info||status.runtime||runtime;ui.enginePill.classList.toggle("offline",!status.ffmpeg_found);ui.engineText.textContent=status.ffmpeg_found?"Local video tools ready":"FFmpeg needed for video";ui.toolDescription.textContent=status.ffmpeg_found?"FFmpeg ready for local video exports.":"FFmpeg is missing. Photo export remains available.";ui.toolPath.textContent=status.ffmpeg_path||"Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe.";renderRuntime();updateSettings(false);updateQueue();}catch(error){statusLoadError=error.message;namingCapability=null;if(preserveHistory&&previousHistoryCapabilities){historyCapabilities=previousHistoryCapabilities;renderHistory();ui.historyStatus.textContent=`Could not refresh history status. Showing the last loaded records. ${String(error.message||error).slice(0,180)}`;}else{historyCapabilities=null;ui.historyPanel.hidden=true;}ui.outputNamingControls.hidden=true;ui.enginePill.classList.add("offline");ui.engineText.textContent="Local server unavailable";ui.toolDescription.textContent=error.message;toast("Could not connect to the local server.",true);updateProcessedClipPanel();updateQueue();}}
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
  function announceQueueTransitions(messages){
    const entries=messages.filter(Boolean);if(!entries.length)return;
    const shown=entries.slice(0,4),extra=entries.length-shown.length;
    const announcement=entries.length===1?shown[0]:entries.length+" queue items reached a final status. "+shown.join(" ")+(extra?" "+extra+" more updated.":"");
    ui.queueLiveStatus.textContent="";ui.queueLiveStatus.textContent=announcement;
  }
  function mergeJob(item,job){
    const metrics=["eta_seconds","eta_status","eta_range_seconds","elapsed_seconds","processing_fps","frames_processed","total_frames","encoder","decoder","phase","retry_count","retry_reason","preflight"];
    const reported=Object.fromEntries(metrics.filter((key)=>Object.hasOwn(job,key)).map((key)=>[key,job[key]]));let depthCache=null,depthCacheError="";try{depthCache=frontendContract.validateDepthCacheStatus(job.depth_cache);}catch(error){depthCacheError=`Depth cache status unavailable: ${error.message}`;}let snapshot=item.settingsSnapshot?{settings:item.settingsSnapshot,sourceAvailable:item.sourceAvailable??Boolean(item.file),queuePosition:item.queuePosition??null}:null,snapshotError=item.settingsSnapshotError||"";
    const notice=frontendContract.observeCompletionNotice(completionNoticeTracker,job,item.state);completionNoticeTracker=notice.tracker;
    if(job.settings_schema_version!=null||job.settings!=null){snapshot=null;snapshotError="";try{snapshot=frontendContract.validateQueueJobSnapshot(job);}catch(error){snapshotError=error.message;}}
    let naming=item.outputNameTemplateSnapshot?{outputNameTemplate:item.outputNameTemplateSnapshot,plannedOutputName:item.plannedOutputName||null,outputName:item.outputName||null}:null,namingError=item.namingSnapshotError||"";
    if(job.naming_schema_version!=null||job.output_name_template!=null||job.planned_output_name!=null||job.output_name!=null){naming=null;namingError="";try{naming=frontendContract.validateQueueJobNaming(job);}catch(error){namingError=error.message;}}
    else if(namingCapability){naming=null;namingError="The server did not include an output name template snapshot. Refresh or upgrade StereoLab.";}
    const previousState=item.state;Object.assign(item,{jobId:job.id,purpose:job.purpose,name:job.filename||item.name,mediaType:job.media_type||item.mediaType,state:job.state,progress:job.progress,message:job.message,error:job.error,historyWarning:typeof job.history_warning==="string"?job.history_warning.slice(0,240):item.historyWarning||null,cleanupWarning:typeof job.cleanup_warning==="string"?job.cleanup_warning.slice(0,240):null,depthCache,depthCacheError,outputName:job.output_name??naming?.outputName??item.outputName??null,plannedOutputName:naming?.plannedOutputName??item.plannedOutputName??null,outputNameTemplateSnapshot:naming?.outputNameTemplate??item.outputNameTemplateSnapshot??null,namingFrozen:Boolean(naming||item.namingFrozen),namingSchemaVersion:job.naming_schema_version??item.namingSchemaVersion??null,namingSnapshotError:namingError,layout:job.layout,execution:job.execution,executionMode:job.execution_mode,pipeline:job.pipeline,settingsSchemaVersion:job.settings_schema_version??null,settingsSnapshot:snapshot?.settings||null,settingsSnapshotError:snapshotError,settingsFrozen:Boolean(snapshot),sourceAvailable:snapshot?.sourceAvailable??null,queuePosition:snapshot?.queuePosition??null},reported);
    const terminalLabels={completed:"Conversion complete.",failed:"Conversion failed.",canceled:"Conversion canceled.",interrupted:"Conversion was interrupted."};
    const detail=job.state==="failed"?String(job.error||job.message||"").trim().replace(/\s+/g," ").slice(0,180):"";
    const terminalAnnouncement=activeStates.has(previousState)&&terminalLabels[job.state]?(job.filename||item.name||"A queued file")+": "+terminalLabels[job.state]+(detail?" "+detail:""):"";
    return {notify:notice.notify,terminalAnnouncement};
  }
  function mergeRemoteJobs(remote){const completed=[],terminalAnnouncements=[];for(const job of remote){if(hiddenJobs.has(job.id)||!frontendContract.isOrdinaryExportJob(job))continue;let item=items.find((x)=>x.jobId===job.id);if(!item){item={uid:job.id};items.push(item);}const merged=mergeJob(item,job);if(merged.notify)completed.push(job.filename||item.name);if(merged.terminalAnnouncement)terminalAnnouncements.push(merged.terminalAnnouncement);}updateQueue();announceQueueTransitions(terminalAnnouncements);showCompletionNoticeBatch(completed);}
  async function refreshQueueFromServer(){const data=await request("/api/jobs"),remote=Array.isArray(data)?data:(data.jobs||[]);mergeRemoteJobs(remote);await refreshQueueControl();}
  async function submitItem(item,settings,outputNameTemplate=null){
    const nameTemplate=frontendContract.outputNameTemplateForSubmission(namingCapability,item,outputNameTemplate||currentOutputNameTemplate);
    item.settingsSnapshot=frontendContract.validateSettings(settings);item.settingsFrozen=true;if(nameTemplate){item.outputNameTemplateSnapshot=nameTemplate;item.namingFrozen=true;}item.state="uploading";item.message="Adding to local queue";item.error=null;updateQueue();
    const data=new FormData();data.append("file",item.file,item.name);Object.entries(item.settingsSnapshot).forEach(([key,value])=>data.append(key,value));if(nameTemplate)data.append("output_name_template",nameTemplate);if(item.projectSourceSha256)data.append("expected_source_sha256",item.projectSourceSha256);
    let completed=false;try{const job=await request("/api/jobs",{method:"POST",body:data});const merged=mergeJob(item,job);completed=merged.notify;announceQueueTransitions(merged.terminalAnnouncement?[merged.terminalAnnouncement]:[]);items=items.filter((x)=>x===item||x.jobId!==job.id);}
    catch(error){if(item.projectSourceSha256&&error.status===409&&/(does not match|no longer matches) the saved project/i.test(error.message)){item.file=null;item.state="missing";item.message="Saved-project source changed; locate it again";}else{item.state="failed";item.error=error.message;item.message="Could not start conversion";const detail=String(error.message||error).trim().replace(/\s+/g," ").slice(0,180);announceQueueTransitions([item.name+": Conversion could not start."+(detail?" "+detail:"")]);}}
    updateQueue();
    return completed;
  }
  async function convertPending(){if(submitting||projectBusy)return;const pending=items.filter((x)=>!x.jobId&&x.state==="ready");if(!pending.length)return;const settings=currentSettingsForPreset(),block=frontendContract.queueReadiness(pending,settings,{runtime,models,customModelAvailable,catalogError});if(block){toast(block,true);return;}const namingBlock=queueNamingBlock(pending);if(namingBlock){toast(namingBlock,true);ui.outputNameTemplate.focus();return;}submitting=true;updateQueue();const completed=[];try{for(const item of pending){if(items.includes(item)&&item.state==="ready"&&await submitItem(item,frontendContract.effectiveQueueSettings(item,settings),namingCapability?frontendContract.outputNameTemplateForSubmit(item,currentOutputNameTemplate):null))completed.push(item.name);}showCompletionNoticeBatch(completed);}finally{submitting=false;updateQueue();await pollJobs();}}
  let queuePaused = false, queueControlBusy = false, storageBusy = false, depthCacheBusy = false;
  function showQueueControl(data){
    if(typeof data.paused!=="boolean"||!Number.isInteger(data.pending_count)||data.pending_count<0)throw new Error("Queue controls are unavailable. Refresh status to retry.");
    queuePaused=data.paused;ui.pauseQueue.disabled=queueControlBusy;ui.pauseQueue.textContent=queuePaused?"Resume queue":"Pause after current job";
    ui.pauseQueue.setAttribute("aria-pressed",String(queuePaused));
    ui.queuePauseStatus.textContent=queuePaused?`Queue paused; ${data.pending_count} pending. Current work finishes normally. Restart resumes scheduling.`:`Queue running; ${data.pending_count} pending.`;
  }
  async function refreshQueueControl(){
    try{showQueueControl(await request("/api/queue/control"));}
    catch(error){ui.pauseQueue.disabled=true;ui.queuePauseStatus.textContent=error.message;}
  }
  async function toggleQueuePause(){
    if(queueControlBusy)return;queueControlBusy=true;ui.pauseQueue.disabled=true;
    try{showQueueControl(await request("/api/queue/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({paused:!queuePaused})}));}
    catch(error){toast(error.message,true);}
    finally{queueControlBusy=false;await refreshQueueControl();}
  }
  async function workspaceAction(url,body){
    try{return await request(url,{method:"POST",...(body?{headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{})});}
    catch(error){toast(error.message,true);return null;}
  }
  async function saveOutputCopy(item){
    const directory=ui.copyOutputDirectory.value.trim();
    if(!directory){toast("Enter an existing destination folder in Local storage first.",true);ui.copyOutputDirectory.focus();return;}
    const result=await workspaceAction(`/api/jobs/${encodeURIComponent(item.jobId)}/copy`,{directory});
    if(result)toast(`Saved copy: ${result.path}`);
  }
  function storageSelection(){
    const selected=[...ui.storageFiles.querySelectorAll("input:checked")];
    const bytes=selected.reduce((sum,input)=>sum+Number(input.dataset.bytes||0),0);
    ui.deleteStorageFiles.disabled=storageBusy||!selected.length;
    ui.deleteStorageFiles.textContent=selected.length?`Delete ${selected.length} selected temporary files (${formatBytes(bytes)})`:"Delete selected temporary files";
    return selected.map(input=>input.value);
  }
  async function measureStorage(){
    if(storageBusy)return;storageBusy=true;ui.refreshStorage.disabled=true;ui.deleteStorageFiles.disabled=true;ui.storageStatus.textContent="Measuring managed storage…";
    try{
      const report=await request("/api/storage");
      ui.storageUsage.replaceChildren(...Object.entries(report.categories).map(([name,info])=>el("p","field-hint",`${name}: ${formatBytes(info.bytes)||"0 B"} in ${info.files} files${info.errors.length?" (some entries unavailable)":""}`)));
      ui.storageFiles.replaceChildren(...report.candidates.slice(0,1000).map(candidate=>{
        const label=el("label","storage-file"),input=el("input");input.type="checkbox";input.value=candidate.path;input.dataset.bytes=String(candidate.bytes);input.addEventListener("change",storageSelection);
        label.append(input,el("span","",`${candidate.path} — ${formatBytes(candidate.bytes)}`));return label;
      }));
      ui.storageStatus.textContent=`${report.scope} ${report.candidates.length} inactive temporary files available (showing up to 1000).${report.errors.length?" Some entries could not be measured; protected files are excluded.":""}`;
    }catch(error){ui.storageStatus.textContent=`Storage measurement failed: ${error.message}`;ui.storageFiles.replaceChildren();}
    finally{storageBusy=false;ui.refreshStorage.disabled=false;storageSelection();}
  }
  async function deleteSelectedStorage(){
    if(storageBusy)return;const paths=storageSelection();if(!paths.length)return;
    storageBusy=true;ui.deleteStorageFiles.disabled=true;
    const result=await workspaceAction("/api/storage/cleanup",{paths});storageBusy=false;
    await measureStorage();await pollJobs();
    if(result)ui.storageStatus.textContent=`Removed ${result.deleted.length} selected temporary files. ${result.errors.length?result.errors.map(entry=>`${entry.path}: ${entry.error}`).join("; "):"Cleanup complete."}`;
  }
  function depthCacheSelectedEntries(){
    const keys=[...ui.depthCacheEntries.querySelectorAll("input:checked")].map((input)=>input.value);
    return frontendContract.depthCacheSelection(depthCacheInventory?.entries||[],keys);
  }
  let depthCacheInventory=null;
  function depthCacheSelection(){
    const selected=depthCacheSelectedEntries(),bytes=selected.reduce((sum,entry)=>sum+entry.bytes,0);
    ui.cleanupDepthCache.disabled=depthCacheBusy||!selected.length;
    ui.cleanupDepthCache.textContent=selected.length?`Remove ${selected.length} selected (${formatBytes(bytes)})`:"Remove selected cache entries";
    return selected.map((entry)=>entry.key);
  }
  async function measureDepthCache(){
    if(depthCacheBusy)return false;depthCacheBusy=true;ui.refreshDepthCache.disabled=true;ui.cleanupDepthCache.disabled=true;ui.depthCacheStatus.textContent="Checking saved video depth maps…";
    try{
      const report=frontendContract.validateDepthCacheInventory(await request("/api/depth-cache"));depthCacheInventory=report;
      ui.depthCacheUsage.replaceChildren(el("p","field-hint",`${formatBytes(report.bytes)} used of ${formatBytes(report.quota_bytes)} · ${report.entries.length} entries`));
      ui.depthCacheEntries.replaceChildren(...report.entries.map((entry)=>{
        const label=el("label","storage-file"),input=el("input");input.type="checkbox";input.value=entry.key;input.dataset.bytes=String(entry.bytes);input.disabled=entry.active;input.addEventListener("change",depthCacheSelection);
        label.title=entry.key;label.append(input,el("span","",`Cache ${entry.key.slice(0,12)} · ${formatBytes(entry.bytes)} · ${entry.frames} frame${entry.frames===1?"":"s"}${entry.active?" · active":""}`));return label;
      }));
      const errors=report.errors.length?` ${report.errors.length} inventory warning${report.errors.length===1?"":"s"}: ${report.errors.join("; ")}`:"";
      ui.depthCacheStatus.textContent=`${report.entries.length} saved entries. Active entries are protected.${errors}`;
      return true;
    }catch(error){ui.depthCacheStatus.textContent=`Could not refresh video depth cache: ${error.message}`;ui.depthCacheUsage.replaceChildren();ui.depthCacheEntries.replaceChildren();depthCacheInventory=null;return false;}
    finally{depthCacheBusy=false;ui.refreshDepthCache.disabled=false;depthCacheSelection();}
  }
  async function cleanupDepthCache(){
    if(depthCacheBusy)return;const requested=frontendContract.depthCacheCleanupRequest(depthCacheInventory?.entries||[],depthCacheSelection());const keys=requested.keys;if(!keys.length)return;
    depthCacheBusy=true;ui.cleanupDepthCache.disabled=true;ui.refreshDepthCache.disabled=true;
    let deleted=[],errors=[];
    try{
      const result=await request("/api/depth-cache/cleanup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({keys})});
      if(!result||!Array.isArray(result.deleted)||!Array.isArray(result.errors)||!result.deleted.every((key)=>typeof key==="string"&&/^[0-9a-f]{64}$/.test(key))||!result.errors.every((entry)=>entry&&typeof entry.key==="string"&&/^[0-9a-f]{64}$/.test(entry.key)&&typeof entry.error==="string"))throw new Error("The server returned an invalid cleanup result.");
      deleted=result.deleted;errors=result.errors;
    }catch(error){errors=[{key:"",error:String(error.message||error)}];}
    finally{depthCacheBusy=false;ui.refreshDepthCache.disabled=false;}
    const refreshed=await measureDepthCache();
    ui.depthCacheStatus.textContent=frontendContract.depthCacheCleanupSummary(keys,{deleted,errors},refreshed);
  }
  ui.pauseQueue.addEventListener("click",toggleQueuePause);
  ui.openOutputFolder.addEventListener("click",()=>workspaceAction("/api/output-folder/open"));
  ui.refreshStorage.addEventListener("click",measureStorage);
  ui.deleteStorageFiles.addEventListener("click",deleteSelectedStorage);
  ui.refreshDepthCache.addEventListener("click",measureDepthCache);
  ui.cleanupDepthCache.addEventListener("click",cleanupDepthCache);
  async function pollJobs(){
    updateCompletionNotificationControls();if(jobsBusy)return;jobsBusy=true;clearTimeout(jobTimer);jobTimer=null;
    try{await refreshQueueFromServer();jobsRetryDelay=0;ui.queueEmpty.querySelector("strong").textContent="A little empty, for now.";ui.queueEmpty.querySelector("small").textContent="Add media above. Completed exports are ready to download here.";}
    catch(error){jobsRetryDelay=Math.min(30000,jobsRetryDelay?jobsRetryDelay*2:1500);ui.queueEmpty.hidden=items.length>0;if(!items.length){ui.queueEmpty.querySelector("strong").textContent="Could not restore the local queue";ui.queueEmpty.querySelector("small").textContent=`${error.message} Retrying automatically; Refresh status can retry now.`;}else toast(`Queue refresh failed: ${error.message}`,true);}
    finally{jobsBusy=false;const delay=jobsRetryDelay||(items.some((x)=>["queued","processing"].includes(x.state))?1500:0);if(delay)jobTimer=setTimeout(pollJobs,delay);}
  }
  async function cancelJob(item){try{await request(`/api/jobs/${encodeURIComponent(item.jobId)}/cancel`,{method:"POST"});item.message="Cancellation requested";updateQueue();await pollJobs();}catch(error){toast(error.message,true);}}
  ui.browseButton.addEventListener("click",(event)=>{event.stopPropagation();ui.mediaInput.click();});ui.dropZone.addEventListener("click",(event)=>{if(event.target!==ui.browseButton)ui.mediaInput.click();});
ui.mediaInput.addEventListener("change",()=>addFiles(ui.mediaInput.files));
  for(const name of ["dragenter","dragover"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.add("drag-over");});for(const name of ["dragleave","drop"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.remove("drag-over");});ui.dropZone.addEventListener("drop",(event)=>addFiles(event.dataTransfer.files));
  document.querySelectorAll("[data-view]").forEach((button)=>{button.addEventListener("click",()=>setView(button.dataset.view));button.addEventListener("keydown",(event)=>{const tabs=[...document.querySelectorAll("[data-view]")].filter((tab)=>!tab.hidden);let index=tabs.indexOf(button);if(event.key==="ArrowRight")index=(index+1)%tabs.length;else if(event.key==="ArrowLeft")index=(index+tabs.length-1)%tabs.length;else if(event.key==="Home")index=0;else if(event.key==="End")index=tabs.length-1;else return;event.preventDefault();setView(tabs[index].dataset.view);tabs[index].focus();});});
  ui.zoomIn.addEventListener("click",()=>applyZoomPan(zoomState.zoom+0.25));ui.zoomOut.addEventListener("click",()=>applyZoomPan(zoomState.zoom-0.25));ui.resetZoom.addEventListener("click",resetZoomPan);ui.previewStage.addEventListener("keydown",handlePreviewKeydown);ui.previewStage.addEventListener("pointerdown",handlePreviewPointerDown);ui.previewStage.addEventListener("pointermove",handlePreviewPointerMove);ui.previewStage.addEventListener("pointerup",handlePreviewPointerEnd);ui.previewStage.addEventListener("pointercancel",handlePreviewPointerEnd);ui.compareSlider.addEventListener("input",updateCompareSplit);window.addEventListener("resize",()=>applyZoomPan());
  ui.sourceImage.addEventListener("load",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("loadeddata",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("seeked",markPreviewStale);ui.sourceVideo.addEventListener("timeupdate",()=>{if(!ui.sourceVideo.paused&&previewResult&&resultRevision===previewRevision)markPreviewStale();invalidateProcessedClipIfStale("Video timestamp changed");updateProcessedClipPanel();refreshCalibrationControls();});
  ui.sourceVideo.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser cannot decode the video container or codec. You can still convert it locally with FFmpeg.";updatePreview();});ui.sourceImage.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser could not open the image. Try another source file.";updatePreview();});
  ui.glassesPreview.addEventListener("click",async()=>{try{if(document.fullscreenElement===ui.previewStage){await document.exitFullscreen();}else{await ui.previewStage.requestFullscreen();}}catch{toast("The browser could not open the fullscreen preview on this display.",true);}});document.addEventListener("fullscreenchange",()=>{ui.glassesPreview.textContent=document.fullscreenElement===ui.previewStage?"Exit fullscreen":"View on glasses";});
  ui.generatePreview.addEventListener("click",generatePreview);ui.previewDetail.addEventListener("change",markPreviewStale);ui.generateProcessedClip.addEventListener("click",generateProcessedClip);ui.cancelProcessedClip.addEventListener("click",cancelProcessedClip);ui.processedClipDuration.addEventListener("change",()=>{invalidateProcessedClipIfStale("Clip length changed");updateProcessedClipPanel();});ui.retryGpu.addEventListener("click",retryGpuAction);ui.presetSelect.addEventListener("change",()=>{applyPreset();refreshPerformancePresetStatus();});
  ui.performancePreset.addEventListener("change",()=>{if(ui.performancePreset.value==="custom"){performanceBaseline=currentSettingsForPreset();importedPresetBaseline=false;updateSettings(false);}else applyPerformancePreset(ui.performancePreset.value);});ui.resetPerformancePreset.addEventListener("click",resetPerformancePreset);ui.exportPreset.addEventListener("click",exportPresetFile);ui.importPresetButton.addEventListener("click",()=>ui.importPresetFile.click());ui.importPresetFile.addEventListener("change",()=>importPresetFile(ui.importPresetFile.files[0]));
  ui.savedPresetSelect.addEventListener("change",selectNamedPreset);ui.savedPresetName.addEventListener("input",()=>{if(ui.savedPresetSelect.value&&ui.savedPresetName.value!==ui.savedPresetSelect.value){ui.savedPresetSelect.value="";ui.deleteNamedPreset.disabled=true;ui.applyNamedPreset.disabled=true;}});ui.saveNamedPreset.addEventListener("click",saveNamedPreset);ui.applyNamedPreset.addEventListener("click",applyNamedPreset);ui.deleteNamedPreset.addEventListener("click",deleteNamedPreset);
   ui.saveProject.addEventListener("click",saveProject);ui.openProject.addEventListener("click",()=>ui.projectFileInput.click());ui.projectFileInput.addEventListener("change",()=>openProjectFile(ui.projectFileInput.files?.[0]));
  ui.calibrationBudget.addEventListener("change",refreshCalibrationControls);ui.calibrateInput.addEventListener("click",measureInputDetail);ui.applyCalibration.addEventListener("click",applyMeasuredInputDetail);
  ["strength","convergence","temporalSmoothing"].forEach((id)=>ui[id].addEventListener("input",()=>{ui.presetSelect.value="custom";updateSettings();}));["edgeRefinement","invertDepth","depthInterval","videoDecode","useDepthCache"].forEach((id)=>ui[id].addEventListener("change",()=>{ui.presetSelect.value="custom";updateSettings();}));
  ["layoutSelect","modelSize","resolution","quality","executionMode"].forEach((id)=>ui[id].addEventListener("change",()=>updateSettings()));ui.depthMode.addEventListener("change",()=>{updateSettings();renderModels();});
  ui.convertButton.addEventListener("click",convertPending);ui.modelButton.addEventListener("click",()=>ui.modelInput.click());ui.modelInput.addEventListener("change",()=>uploadModel(ui.modelInput.files[0]));ui.installSelected.addEventListener("click",()=>installModel(ui.depthMode.value));ui.refreshModels.addEventListener("click",()=>{loadModels();loadStatus();pollJobs();});
  ui.completionNotifications.addEventListener("click",toggleCompletionNotifications);
  window.addEventListener("keydown",(event)=>{const action=frontendContract.keyboardShortcutAction({altKey:event.altKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,shiftKey:event.shiftKey,repeat:event.repeat,defaultPrevented:event.defaultPrevented,isComposing:event.isComposing,code:event.code,target:event.target,modalOpen:Boolean(document.querySelector("dialog[open]")),previewEnabled:!ui.generatePreview.disabled,submitEnabled:!ui.convertButton.disabled});if(!action)return;event.preventDefault();(action==="preview"?ui.generatePreview:ui.convertButton).click();});
  window.addEventListener("focus",updateCompletionNotificationControls);document.addEventListener("visibilitychange",()=>{if(!document.hidden)updateCompletionNotificationControls();});
  ui.applySettingsToReady.addEventListener("click",applySettingsToReady);
  ui.historyRefresh.addEventListener("click",()=>loadStatus(true));ui.historyMore.addEventListener("click",()=>loadHistory(true));
  ui.outputNameTemplate.addEventListener("input",()=>{currentOutputNameTemplate=ui.outputNameTemplate.value;try{frontendContract.validateOutputNameTemplate(currentOutputNameTemplate,namingCapability?.max_template_length??120);ui.outputNameTemplate.removeAttribute("aria-invalid");}catch(error){ui.outputNameTemplate.setAttribute("aria-invalid","true");ui.outputNamingStatus.textContent=error.message;}updateQueue();});
  ui.clearQueueButton.addEventListener("click",()=>{const finished=items.filter((x)=>["completed","failed","canceled"].includes(x.state));finished.forEach(removeItem);saveHidden();updateQueue();toast(finished.length?`${finished.length} finished item${finished.length===1?"":"s"} removed from this browser’s list.`:"There are no finished items to clear.");});
  ui.helpButton.addEventListener("click",()=>ui.aboutDialog.showModal());ui.aboutDone.addEventListener("click",()=>ui.aboutDialog.close());window.addEventListener("beforeunload",()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);});
  loadNamedPresets();updateCompletionNotificationControls();updateSettings(false);Promise.allSettled([loadStatus(),loadModels(),pollJobs()]);
})();

