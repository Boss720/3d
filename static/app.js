(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ids = ["mediaInput","dropZone","browseButton","sourceVideo","sourceImage","processedImage","previewEmpty","previewStage","sourceName","previewState","generatePreview","layoutSelect","formatHint","previewModeLabel","presetSelect","depthMode","depthNote","installSelected","modelRow","modelInput","modelButton","modelState","modelSizeRow","modelSizeLabel","modelSize","midasSizeNote","strength","strengthValue","convergence","convergenceValue","temporalSmoothing","smoothingValue","edgeRefinement","invertDepth","resolution","quality","convertButton","convertCount","convertHint","queueRows","queueEmpty","queueCount","queueHeadingCount","clearQueueButton","enginePill","engineText","toolDescription","toolPath","runtimeDescription","runtimeCommand","modelCards","refreshModels","toast","helpButton","aboutDialog","aboutDone"];
  ids.push("executionMode", "executionNote", "previewExecution");
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
  let items = [], selectedUid = null, previewUrl = null, previewFile = null, previewView = "original", previewResult = null;
  let previewRevision = 0, resultRevision = -1, previewBusy = false, sourceError = "", previewError = "";
  let models = [], runtime = null, status = null, customModelAvailable = false, catalogError = "";
  let modelTimer = null, jobTimer = null, jobsBusy = false, modelsBusy = false, submitting = false, toastTimer = null;
  const hiddenJobs = new Set();
  try { JSON.parse(sessionStorage.getItem("stereolab.hiddenJobs") || "[]").forEach((id) => hiddenJobs.add(id)); } catch { /* Session storage may be disabled. */ }
  function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text != null) node.textContent = text; return node; }
  function toast(message, error = false) { ui.toast.textContent = message; ui.toast.classList.toggle("error", error); ui.toast.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 5500); }
  async function request(url, options = {}) {
    const response = await fetch(url, options); let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((x) => x.msg).join("; ") : `Request failed (${response.status}).`);
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
    if (item.state === "completed" || item.eta_status === "complete") return "Complete";
    if (item.state === "queued") return "Waiting in queue";
    if (item.state !== "processing") return "";
    if (item.mediaType === "image") return "Still image · no time estimate";
    const seconds = metricNumber(item.eta_seconds);
    if (item.eta_status === "available" && seconds !== null && seconds >= 0) return `About ${formatDuration(seconds)} remaining`;
    if (item.eta_status === "estimating") return "Estimating time left…";
    if (item.eta_status === "unavailable") return "Time estimate unavailable";
    return "Waiting for time estimate";
  }
  function progressDetails(item) {
    const details = [];
    const eta = etaLabel(item);
    if (eta) details.push(eta);
    const frames = metricNumber(item.frames_processed), total = metricNumber(item.total_frames), fps = metricNumber(item.processing_fps), elapsed = metricNumber(item.elapsed_seconds);
    if (frames !== null && frames >= 0 && total !== null && total > 0) details.push(`Frames ${Math.min(frames, total).toLocaleString()} / ${total.toLocaleString()}`);
    if (fps !== null && fps > 0) details.push(`${fps.toFixed(1)} fps`);
    if (elapsed !== null && elapsed >= 0) details.push(`Elapsed ${formatDuration(elapsed)}`);
    if (typeof item.encoder === "string" && item.encoder.trim()) details.push(`Encoder: ${item.encoder.trim()}`);
    return details;
  }
  function selectedModel() { return models.find((model) => model.id === ui.depthMode.value); }
  function limitedGpu() { return (runtime?.adapters || []).some((adapter) => (/nvidia|1650/i.test(adapter.name || "") || Number(adapter.vendor_id) === 4318) && Number(adapter.memory_mb) > 0 && Number(adapter.memory_mb) <= 4096); }
  function largeOnLimitedGpu() { return ui.depthMode.value === "depth-anything-v2-large" && limitedGpu(); }
  function gpuBlock() {
    if (ui.executionMode.value !== "gpu") return "";
    if (["balanced", "precision"].includes(ui.depthMode.value)) return "GPU inference requires a named neural model. Choose Auto or CPU only for visual approximation or custom ONNX.";
    if (!runtime) return "Checking NVIDIA GPU support…";
    if (!runtime.gpu_available) return "NVIDIA GPU is unavailable. Run .\\setup-gpu.ps1 to prepare the separate .venv-gpu environment, then start it with .\\run.ps1. Or choose Auto or CPU only.";
    if (largeOnLimitedGpu()) return "Large exceeds the 4 GB GPU budget. Choose Small or Base, or choose Auto, hybrid or CPU only to run Large on CPU.";
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
    ui.executionNote.classList.toggle("device-warning", Boolean(blocked) || mode !== "cpu" && (largeOnLimitedGpu() || runtime && !runtime.gpu_available));
    if (blocked) { ui.executionNote.textContent = blocked; return; }
    if (["balanced", "precision"].includes(ui.depthMode.value)) { ui.executionNote.textContent = ui.depthMode.value === "balanced" ? "Visual approximation uses CPU image cues. Choose a named neural depth engine to use GPU depth estimation." : "Custom ONNX models use CPU in this studio. Choose a named neural depth engine to use GPU depth estimation."; return; }
    if (mode === "cpu") { ui.executionNote.textContent = "CPU depth estimation and media processing. GPU is not requested."; return; }
    if (!runtime) { ui.executionNote.textContent = "Checking local GPU support…"; return; }
    if (largeOnLimitedGpu()) { ui.executionNote.textContent = "Large will use CPU on this 4 GB GPU. Select Small for GPU depth estimation."; return; }
    if (!runtime.gpu_available) { ui.executionNote.textContent = `CPU fallback: ${runtime.error || "no compatible GPU runtime and NVIDIA adapter detected"}. Optional GPU setup: .\\setup-gpu.ps1 (separate .venv-gpu), then .\\run.ps1.`; return; }
    ui.executionNote.textContent = `${runtime.gpu_name || "NVIDIA GPU"} available${runtime.gpu_verified ? " · inference verified" : " · inference not yet verified"}. ${mode === "gpu" ? "GPU requested; failures will be reported." : "GPU depth with CPU media processing; CPU fallback if needed."}`;
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
  function previewMessage(message, error = false) { ui.previewState.textContent = message; ui.previewState.classList.toggle("error", error); }
  function markPreviewStale() { previewRevision++; previewError=""; updatePreview(); }
  function updatePreview() {
    const original = previewView === "original";
    ui.sourceImage.hidden = !(original && previewFile && isImage(previewFile) && !sourceError);
    ui.sourceVideo.hidden = !(original && previewFile && !isImage(previewFile) && !sourceError);
    ui.processedImage.hidden = original || !previewResult;
    ui.previewEmpty.hidden = (original ? Boolean(previewFile && !sourceError) : Boolean(previewResult));
    const emptyTitle = ui.previewEmpty.querySelector("strong"), emptyCopy = ui.previewEmpty.querySelector(":scope > span");
    if (sourceError) { emptyTitle.textContent = "Preview unavailable for this source"; emptyCopy.textContent = sourceError; }
    else if (!previewFile) { emptyTitle.textContent = selectedUid ? "Source preview is not retained after reload" : "Your next dimension starts here"; emptyCopy.textContent = selectedUid ? "Your export is still available in the queue. Add the original file again to preview it." : "Add a photo or video to preview a frame."; }
    else { emptyTitle.textContent = "Ready for a new perspective?"; emptyCopy.textContent = "Generate a real processed frame with your current settings."; }
    if (previewResult && !original) { ui.processedImage.src = previewView === "depth" ? previewResult.depth : previewResult.image; ui.processedImage.alt = previewView === "depth" ? "Estimated depth map for the selected frame" : "Rendered stereo output for the selected frame"; }
    ui.generatePreview.disabled = previewBusy || !sourceReady() || Boolean(readiness());
    ui.previewExecution.hidden = !previewResult;
    ui.previewExecution.textContent = previewResult ? `${resultRevision !== previewRevision ? "Previous preview · " : ""}${executionSummary(previewResult.execution) || "Device use not reported"}` : "";
    ui.generatePreview.textContent = previewBusy ? "Processing frame…" : previewResult && resultRevision !== previewRevision ? "↻ Update preview" : "✦ Generate preview";
    if (previewBusy) previewMessage("Processing one frame locally…");
    else if (sourceError) previewMessage("Export is still available", true);
    else if (previewError) previewMessage(previewError, true);
    else if (previewResult) previewMessage(resultRevision === previewRevision ? `Processed frame · ${previewResult.width} × ${previewResult.height}` : "Settings or frame changed · update preview");
    else if (previewFile) previewMessage(sourceReady() ? "Ready to generate a frame preview" : "Loading source…");
    else previewMessage(selectedUid ? "Restored local job" : "Select a source to begin");
  }
  function setView(view) {
    if (view !== "original") ui.sourceVideo.pause();
    previewView = view;
    document.querySelectorAll("[data-view]").forEach((button) => { const active=button.dataset.view === view; button.classList.toggle("active",active); button.setAttribute("aria-selected",String(active)); button.tabIndex=active ? 0 : -1; });
    ui.previewStage.setAttribute("aria-labelledby", view === "original" ? "tabOriginal" : view === "stereo" ? "tabStereo" : "tabDepth");
    updatePreview();
  }
  function selectItem(item) {
    selectedUid = item.uid;
    ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; previewFile = item.file || null; previewResult = null; sourceError = ""; previewError=""; previewRevision++;
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
    if (selectedUid===item.uid) { selectedUid=null; previewFile=null; previewResult=null; sourceError=""; previewError=""; ui.sourceVideo.pause(); ui.sourceVideo.removeAttribute("src"); ui.sourceVideo.load(); ui.sourceImage.removeAttribute("src"); if(previewUrl)URL.revokeObjectURL(previewUrl); previewUrl=null; ui.sourceName.textContent="No media selected"; updatePreview(); }
  }
  function action(text, handler) { const button=el("button","row-button",text); button.type="button"; button.addEventListener("click",handler); return button; }
  function updateQueue() {
    const pending=items.filter((x)=>!x.jobId && x.state==="ready"); const block=readiness();
    const videoBlocked=status && !status.ffmpeg_found && pending.some((x)=>x.mediaType==="video");
    ui.queueCount.textContent=String(items.length); ui.queueHeadingCount.textContent=`(${items.length})`; ui.convertCount.textContent=String(pending.length); ui.queueEmpty.hidden=items.length>0;
    ui.convertButton.disabled=!pending.length || submitting || Boolean(block) || Boolean(videoBlocked);
    ui.convertButton.firstElementChild.textContent=submitting?"Adding to local queue…":"✦ Convert queue";
    ui.convertHint.textContent=block || (videoBlocked?"FFmpeg is needed for the videos in your queue.":pending.length?`${pending.length} file${pending.length===1?"":"s"} ready with the current settings.`:"Add media to start a conversion.");
    ui.queueRows.replaceChildren();
    for(const item of items) {
      const row=el("article",`queue-row${item.uid===selectedUid?" selected":""}`); const icon=el("div","file-icon",item.mediaType==="image"?"▧":"▶"); const copy=el("div","file-copy");
      const name=el("button","file-name",item.name); name.type="button"; name.title=item.name; name.setAttribute("aria-label",`Select ${item.name}`); name.setAttribute("aria-pressed",String(item.uid===selectedUid)); name.addEventListener("click",()=>selectItem(item));
      const meta=el("div","file-meta"); const state=el("span",`file-status ${item.state}`,item.message || ({ready:"Ready to convert",uploading:"Uploading locally",completed:"Complete",failed:"Failed",canceled:"Canceled"}[item.state] || item.state)); meta.append(state);
      const info=[formatBytes(item.file?.size||item.size),item.layout].filter(Boolean).join(" · "); if(info)meta.append(el("span","",info)); copy.append(name,meta);
      const execution=executionSummary(item.execution,item.pipeline);if(execution)copy.append(el("div","queue-execution",execution));else if(item.executionMode)copy.append(el("div","queue-execution",`Requested: ${{auto:"Auto",hybrid:"GPU + CPU",gpu:"NVIDIA GPU",cpu:"CPU only"}[item.executionMode] || item.executionMode} · awaiting execution`));
      const runtimeDetails=progressDetails(item);if(runtimeDetails.length)copy.append(el("div","queue-runtime",runtimeDetails.join(" · ")));
      const actions=el("div","row-actions");
      if(activeStates.has(item.state)) {const progress=el("div","row-progress"),bar=el("span");progress.setAttribute("role","progressbar");progress.setAttribute("aria-label",`Conversion progress for ${item.name}`);progress.setAttribute("aria-valuemin","0");progress.setAttribute("aria-valuemax","100");progress.setAttribute("aria-valuenow",String(item.progress||0));bar.style.width=`${Math.max(2,Math.min(100,item.progress||0))}%`;progress.append(bar);actions.append(progress);}
      if(item.state==="completed"&&item.jobId){const link=el("a","row-button","Download ↓");link.href=`/api/jobs/${encodeURIComponent(item.jobId)}/download`;link.download=item.outputName||"";actions.append(link);}
      if(["queued","processing"].includes(item.state)&&item.jobId)actions.append(action("Cancel",()=>cancelJob(item)));
      if(item.state==="failed"&&item.file)actions.append(action("Retry",()=>{if(item.jobId){hiddenJobs.add(item.jobId);saveHidden();}item.jobId=null;item.state="ready";item.error=null;item.message="";item.execution=null;item.executionMode=null;item.pipeline=null;updateQueue();}));
      if(!activeStates.has(item.state))actions.append(action("Remove",()=>{removeItem(item);saveHidden();updateQueue();}));
      row.append(icon,copy,actions); if(item.error)row.append(el("div","job-error",item.error)); ui.queueRows.append(row);
    }
  }
  function saveHidden(){try{sessionStorage.setItem("stereolab.hiddenJobs",JSON.stringify([...hiddenJobs]));}catch{/* Keep this page usable without storage. */}}
  function getSettings(){const mode=ui.depthMode.value;const inputSize=mode==="dpt-hybrid-midas"?"384":mode==="precision"||/^depth-anything-v2-(small|base|large)$/.test(mode)?ui.modelSize.value:"518";return {layout:ui.layoutSelect.value,strength:ui.strength.value,convergence:ui.convergence.value,depth_mode:mode,execution_mode:ui.executionMode.value,input_size:inputSize,resolution:ui.resolution.value,quality:ui.quality.value,temporal_smoothing:ui.temporalSmoothing.value,edge_refinement:String(ui.edgeRefinement.checked),invert_depth:String(ui.invertDepth.checked)};}
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
    if(stale)markPreviewStale();else updatePreview();updateQueue();
  }
  function applyPreset(){const preset=presets[ui.presetSelect.value];if(!preset)return;ui.strength.value=preset.strength;ui.convergence.value=preset.convergence;ui.temporalSmoothing.value=preset.smoothing;ui.edgeRefinement.checked=preset.edges;ui.invertDepth.checked=preset.invert;updateSettings();}
  async function generatePreview(){
    if(previewBusy || !sourceReady())return;const block=readiness();if(block){toast(block,true);return;}
    ui.sourceVideo.pause();const revision=previewRevision,settings=getSettings();previewError="";previewBusy=true;updatePreview();
    try{
      const source=isImage(previewFile)?ui.sourceImage:ui.sourceVideo;const w=source.naturalWidth||source.videoWidth,h=source.naturalHeight||source.videoHeight;
      const scale=Math.min(1,960/w,540/h),canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));canvas.getContext("2d").drawImage(source,0,0,canvas.width,canvas.height);
      const blob=await new Promise((resolve,reject)=>canvas.toBlob((value)=>value?resolve(value):reject(new Error("Could not capture this frame.")),"image/jpeg",.94));
      const data=new FormData();data.append("file",blob,"preview-frame.jpg");Object.entries(settings).forEach(([key,value])=>data.append(key,value));
      const result=await request("/api/preview",{method:"POST",body:data});
      if(revision!==previewRevision){toast("The source or settings changed. Generate again for the current frame.");return;}
      if(!result.image||!result.depth)throw new Error("The preview response did not contain processed images.");
      previewResult=result;resultRevision=revision;setView(previewView==="depth"?"depth":"stereo");
    }catch(error){previewError=error.message;previewMessage(error.message,true);toast(error.message,true);}finally{previewBusy=false;updatePreview();await loadStatus();}
  }
  function renderRuntime(){
    if(!runtime)return;
    const providers=(runtime.providers || []).join(", ") || "none reported";
    const capability=runtime.gpu_available?`${runtime.gpu_name || "NVIDIA GPU"} available · ${runtime.gpu_verified ? "GPU inference verified" : "GPU inference not yet verified"}`:"GPU unavailable in this runtime";
    const actual=runtime.last_execution?.requested_mode ? `Last execution: ${executionSummary(runtime.last_execution)}` : runtime.active_provider ? `Last provider: ${runtime.active_provider}` : "No inference has run in this runtime";
    ui.runtimeDescription.textContent=runtime.available?`ONNX Runtime ${runtime.version || ""} ready. Providers: ${providers}. ${capability}. ${actual}.${runtime.error ? ` ${runtime.error}` : ""}`:`ONNX Runtime is missing. Install the project dependencies, then restart StereoLab.${runtime.error ? ` ${runtime.error}` : ""}`;
    ui.runtimeCommand.hidden=Boolean(runtime.available);
  }
  async function loadStatus(){try{status=await request("/api/status");customModelAvailable=Boolean(status.model_available);ui.modelState.textContent=customModelAvailable?(status.model_name||"Custom model loaded"):"No model loaded";runtime=status.runtime_info||status.runtime||runtime;ui.enginePill.classList.toggle("offline",!status.ffmpeg_found);ui.engineText.textContent=status.ffmpeg_found?"Local video tools ready":"FFmpeg needed for video";ui.toolDescription.textContent=status.ffmpeg_found?"FFmpeg ready for local video exports.":"FFmpeg is missing. Photo export remains available.";ui.toolPath.textContent=status.ffmpeg_path||"Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe.";renderRuntime();updateSettings(false);}catch(error){ui.enginePill.classList.add("offline");ui.engineText.textContent="Local server unavailable";ui.toolDescription.textContent=error.message;toast("Could not connect to the local server.",true);}}
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
  function mergeJob(item,job){const metrics=["eta_seconds","eta_status","elapsed_seconds","processing_fps","frames_processed","total_frames","encoder"];const reported=Object.fromEntries(metrics.filter((key)=>Object.hasOwn(job,key)).map((key)=>[key,job[key]]));Object.assign(item,{jobId:job.id,name:job.filename||item.name,mediaType:job.media_type||item.mediaType,state:job.state,progress:job.progress,message:job.message,error:job.error,outputName:job.output_name,layout:job.layout,execution:job.execution,executionMode:job.execution_mode,pipeline:job.pipeline},reported);}
  async function submitItem(item,settings){item.state="uploading";item.message="Adding to local queue";updateQueue();const data=new FormData();data.append("file",item.file,item.name);Object.entries(settings).forEach(([key,value])=>data.append(key,value));try{const job=await request("/api/jobs",{method:"POST",body:data});mergeJob(item,job);items=items.filter((x)=>x===item||x.jobId!==job.id);}catch(error){item.state="failed";item.error=error.message;item.message="Could not start conversion";}updateQueue();}
  async function convertPending(){if(submitting)return;const block=readiness();if(block){toast(block,true);return;}const pending=items.filter((x)=>!x.jobId&&x.state==="ready");if(!pending.length)return;const settings=getSettings();submitting=true;updateQueue();try{for(const item of pending){if(items.includes(item)&&item.state==="ready")await submitItem(item,settings);}}finally{submitting=false;updateQueue();await pollJobs();}}
  async function pollJobs(){
    if(jobsBusy)return;jobsBusy=true;clearTimeout(jobTimer);jobTimer=null;
    try{const data=await request("/api/jobs");const remote=Array.isArray(data)?data:(data.jobs||[]);for(const job of remote){if(hiddenJobs.has(job.id))continue;let item=items.find((x)=>x.jobId===job.id);if(!item){item={uid:job.id};items.push(item);}mergeJob(item,job);}updateQueue();}
    catch(error){ui.queueEmpty.hidden=items.length>0;if(!items.length){ui.queueEmpty.querySelector("strong").textContent="Could not restore the local queue";ui.queueEmpty.querySelector("small").textContent=error.message;}else toast(`Queue refresh failed: ${error.message}`,true);}
    finally{jobsBusy=false;if(items.some((x)=>["queued","processing"].includes(x.state)))jobTimer=setTimeout(pollJobs,1500);}
  }
  async function cancelJob(item){try{await request(`/api/jobs/${encodeURIComponent(item.jobId)}/cancel`,{method:"POST"});item.message="Cancellation requested";updateQueue();await pollJobs();}catch(error){toast(error.message,true);}}
  ui.browseButton.addEventListener("click",(event)=>{event.stopPropagation();ui.mediaInput.click();});ui.dropZone.addEventListener("click",(event)=>{if(event.target!==ui.browseButton)ui.mediaInput.click();});
  ui.dropZone.addEventListener("keydown",(event)=>{if(event.target===ui.dropZone&&(event.key==="Enter"||event.key===" ")){event.preventDefault();ui.mediaInput.click();}});ui.mediaInput.addEventListener("change",()=>addFiles(ui.mediaInput.files));
  for(const name of ["dragenter","dragover"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.add("drag-over");});for(const name of ["dragleave","drop"])ui.dropZone.addEventListener(name,(event)=>{event.preventDefault();ui.dropZone.classList.remove("drag-over");});ui.dropZone.addEventListener("drop",(event)=>addFiles(event.dataTransfer.files));
  document.querySelectorAll("[data-view]").forEach((button)=>{button.addEventListener("click",()=>setView(button.dataset.view));button.addEventListener("keydown",(event)=>{const tabs=[...document.querySelectorAll("[data-view]")];let index=tabs.indexOf(button);if(event.key==="ArrowRight")index=(index+1)%tabs.length;else if(event.key==="ArrowLeft")index=(index+tabs.length-1)%tabs.length;else if(event.key==="Home")index=0;else if(event.key==="End")index=tabs.length-1;else return;event.preventDefault();setView(tabs[index].dataset.view);tabs[index].focus();});});
  ui.sourceImage.addEventListener("load",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("loadeddata",()=>{sourceError="";markPreviewStale();});ui.sourceVideo.addEventListener("seeked",markPreviewStale);ui.sourceVideo.addEventListener("timeupdate",()=>{if(!ui.sourceVideo.paused&&previewResult&&resultRevision===previewRevision)markPreviewStale();});
  ui.sourceVideo.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser cannot decode the video container or codec. You can still convert it locally with FFmpeg.";updatePreview();});ui.sourceImage.addEventListener("error",()=>{if(!previewFile)return;sourceError="This browser could not open the image. Try another source file.";updatePreview();});
  ui.generatePreview.addEventListener("click",generatePreview);ui.presetSelect.addEventListener("change",applyPreset);
  ["strength","convergence","temporalSmoothing"].forEach((id)=>ui[id].addEventListener("input",()=>{ui.presetSelect.value="custom";updateSettings();}));["edgeRefinement","invertDepth"].forEach((id)=>ui[id].addEventListener("change",()=>{ui.presetSelect.value="custom";updateSettings();}));
  ["layoutSelect","modelSize","resolution","quality","executionMode"].forEach((id)=>ui[id].addEventListener("change",()=>updateSettings()));ui.depthMode.addEventListener("change",()=>{updateSettings();renderModels();});
  ui.convertButton.addEventListener("click",convertPending);ui.modelButton.addEventListener("click",()=>ui.modelInput.click());ui.modelInput.addEventListener("change",()=>uploadModel(ui.modelInput.files[0]));ui.installSelected.addEventListener("click",()=>installModel(ui.depthMode.value));ui.refreshModels.addEventListener("click",()=>{loadModels();loadStatus();});
  ui.clearQueueButton.addEventListener("click",()=>{const finished=items.filter((x)=>["completed","failed","canceled"].includes(x.state));finished.forEach(removeItem);saveHidden();updateQueue();toast(finished.length?`${finished.length} finished item${finished.length===1?"":"s"} removed from this browser’s list.`:"There are no finished items to clear.");});
  ui.helpButton.addEventListener("click",()=>ui.aboutDialog.showModal());ui.aboutDone.addEventListener("click",()=>ui.aboutDialog.close());window.addEventListener("beforeunload",()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);});
  updateSettings(false);Promise.allSettled([loadStatus(),loadModels(),pollJobs()]);
})();

