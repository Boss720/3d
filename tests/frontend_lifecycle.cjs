// Run the browser closure unchanged apart from replacing boot with a test accessor.
// Network and timers are controlled; no server, model, GPU, or browser is needed.
const assert = require('node:assert/strict');
const nodeCrypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tagName = 'div') {
    this.tagName = tagName.toLowerCase(); this.parentElement = null; this.ownerDocument = null;
    this.value = ''; this.checked = false; this.hidden = false;
    this.dataset = {}; this.style = {}; this.options = []; this.children = [];
    this.listeners = {}; this.queries = new Map(); this.textContent = '';
    this.classList = {add() {}, remove() {}, toggle() {}};
    this.complete = true; this.naturalWidth = 640; this.naturalHeight = 360;
    this.clientWidth = 640; this.clientHeight = 360; this.currentTime = 0;
  }
  querySelector(key) {
    const found = this.querySelectorAll(key)[0];
    if (found) return found;
    if (!this.queries.has(key)) this.queries.set(key, new Element());
    const node = this.queries.get(key); node.ownerDocument = this.ownerDocument; return node;
  }
  querySelectorAll(selector) {
    const selectors = selector.split(',').map(value => value.trim());
    const matches = (node, query) => {
      if (query === 'input:checked') return node.tagName === 'input' && node.checked;
      if (query.startsWith('.')) return String(node.className || '').split(/\s+/).includes(query.slice(1));
      if (query.startsWith('[') && query.endsWith(']')) return Object.hasOwn(node.dataset, query.slice(1, -1).replace(/^data-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase()));
      const tagAndData = query.match(/^([a-z]+)\[data-([a-z-]+)\]$/i);
      if (tagAndData) return node.tagName === tagAndData[1].toLowerCase() && Object.hasOwn(node.dataset, tagAndData[2].replace(/-([a-z])/g, (_, c) => c.toUpperCase()));
      return node.tagName === query.toLowerCase();
    };
    const found = [];
    const visit = node => {
      for (const child of node.children) {
        if (selectors.some(query => matches(child, query))) found.push(child);
        visit(child);
      }
    };
    visit(this); return found;
  }
  get firstElementChild() { return this.children[0] || this.querySelector('first'); }
  append(...nodes) { for (const node of nodes) { if (!node || typeof node !== 'object') continue; node.parentElement = this; node.ownerDocument = this.ownerDocument; this.children.push(node); } }
  replaceChildren(...nodes) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...nodes); }
  addEventListener(name, fn) { (this.listeners[name] ??= []).push(fn); }
  click() { return Promise.all((this.listeners.click ?? []).map(fn => fn({target: this}))); }
  setAttribute() {} removeAttribute(name) { delete this[name]; }
  pause() {} load() {} focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  closest(selector) { for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node; return null; }
  matches(selector) { return selector.startsWith('.') ? String(this.className || '').split(/\s+/).includes(selector.slice(1)) : this.tagName === selector.toLowerCase(); }
  getContext() { return {drawImage() {}}; }
  toBlob(callback) { callback({}); }
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
const response = data => ({ok: true, json: async () => data});

function fixture(storage = new Map()) {
  const nodes = new Map(), timers = new Map(), calls = [], blobs = [];
  let nextTimer = 0, nextUuid = 0, fetchHandler = async url => response(url === '/api/models' ? {models: []} : {});
  const document = {
    activeElement: null,
    getElementById(id) { if (!nodes.has(id)) { const node = new Element(); node.ownerDocument = this; nodes.set(id, node); } return nodes.get(id); },
    createElement(tagName) { const node = new Element(tagName); node.ownerDocument = this; return node; }, createTextNode(text) { return {textContent: text}; },
    querySelectorAll() { return []; }, querySelector() { return null; }, addEventListener() {},
  };
  document.activeElement = new Element('body'); document.activeElement.ownerDocument = document;
  const context = vm.createContext({
    document, window: {addEventListener() {}, localStorage: {getItem(key) { return storage.get(key) ?? null; }, setItem(key, value) { storage.set(key, value); }}}, console,
    sessionStorage: {getItem() { return null; }, setItem() {}},
    URL: {createObjectURL(blob) { blobs.push(blob); return `blob:test-${blobs.length}`; }, revokeObjectURL() {}},
    Blob, TextEncoder,
    Worker: class {
      constructor() { this.terminated = false; }
      postMessage({file}) {
        try {
          const bytes = Buffer.from(file.bytes || file.data || []);
          const sha256 = nodeCrypto.createHash('sha256').update(bytes).digest('hex');
          this.onmessage({data: {progress: file.size, total: file.size}});
          this.onmessage({data: {sha256, size_bytes: file.size}});
        } catch (error) { this.onmessage({data: {error: error.message}}); }
      }
      terminate() { this.terminated = true; }
    },
    FormData: class { constructor() { this.entries = []; } append(...entry) { this.entries.push(entry); } },
    crypto: {randomUUID() { nextUuid++; return `00000000-0000-4000-8000-${String(nextUuid).padStart(12, '0')}`; }},
    fetch: async (url, options) => { calls.push(url); return fetchHandler(url, options); },
    setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  let source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
  const boot = 'loadNamedPresets();updateCompletionNotificationControls();updateSettings(false);Promise.allSettled([loadStatus(),loadModels(),pollJobs()]);';
  assert.equal(source.split(boot).length, 2, 'Expected one application boot statement');
  source = source.replace(boot, `loadNamedPresets();globalThis.lifecycle = {
    ui, generatePreview, removeItem, removeQueuedItem, selectItem, pollJobs, updateQueue, calibrationContext,
    saveProject, openProjectFile, relinkProjectSource, submitItem, currentSettingsForPreset, validateProjectDocument,
    saveNamedPreset, applyNamedPreset, deleteNamedPreset, selectNamedPreset,
    savedPresets() { return namedPresets; },
    seed(next) { items = next; },
    setStatus(next) { status = next; },
    setCalibration() { calibrationResult = {}; calibrationResultContext = calibrationContext(); ui.calibrationReport.hidden = false; ui.applyCalibration.hidden = false; },
    state() { return {items, selectedUid, previewFile, previewResult, previewRevision, previewBusy, previewError, calibrationResult, calibrationResultContext, jobsBusy, projectBusy}; }
  };`);
  vm.runInContext(source, context, {filename: 'static/app.js'});
  const app = context.lifecycle;
  const defaults = {layoutSelect: 'SBS', depthMode: 'balanced', executionMode: 'cpu', modelSize: '384', strength: '5', convergence: '.5', temporalSmoothing: '.35', depthInterval: '1', videoDecode: 'software', resolution: '1080p', quality: 'balanced', previewDetail: 'proxy', processedClipDuration: '3', calibrationBudget: '15', performancePreset: 'balanced'};
  for (const [id, value] of Object.entries(defaults)) app.ui[id].value = value;
  app.ui.edgeRefinement.checked = true;
  app.ui.queueEmpty.querySelector('strong').textContent = 'A little empty, for now.';
  app.ui.queueEmpty.querySelector('small').textContent = 'Add media above. Completed exports are ready to download here.';
  return {
    app, timers, calls, document, storage, blobs,
    network(fn) { fetchHandler = fn; },
    async fire(id) { const timer = timers.get(id); assert.ok(timer, 'Expected scheduled retry'); timers.delete(id); await timer.fn(); },
    async flushImmediate() { for (const [id, timer] of [...timers]) if (timer.delay === 0) { timers.delete(id); await timer.fn(); } },
    jobTimer() { return [...timers].find(([, timer]) => timer.delay !== 5500); },
  };
}

const tick = () => new Promise(resolve => setImmediate(resolve));
const preview = {image: 'stereo-data', depth: 'depth-data', width: 1280, height: 360, source_width: 640, source_height: 360, original_width: 640, original_height: 360, preview_detail: 'proxy', is_proxy: true};
const job = state => ({id: 'restored', filename: 'source.png', media_type: 'image', purpose: 'export', state, progress: 0});
function selectSource(app, uid = 'source') {
  const item = {uid, file: {name: 'source.png', size: 100}, name: 'source.png', state: 'ready', mediaType: 'image'};
  app.seed([item]); app.selectItem(item); return item;
}

async function run(scenario) {
  const f = fixture(), {app} = f;
  if (scenario.startsWith('remove-')) {
    const item = selectSource(app), before = app.state().previewRevision;
    if (scenario === 'remove-calibration') {
      app.setCalibration();
      const context = app.calibrationContext();
      app.removeItem(item);
      assert.ok(app.state().previewRevision > before, 'Removing selected source must advance preview revision');
      assert.equal(app.state().calibrationResult, null);
      assert.equal(app.state().calibrationResultContext, null);
      assert.equal(app.ui.calibrationReport.hidden, true);
      assert.notEqual(app.calibrationContext().sourceId, context.sourceId);
      selectSource(app, 'next');
      f.network(async url => response(url === '/api/preview' ? preview : {}));
      await app.generatePreview();
      assert.ok(app.state().previewResult, 'A later source must still generate previews');
      return;
    }
    const pending = deferred();
    f.network(url => url === '/api/preview' ? pending.promise : Promise.resolve(response({})));
    const generating = app.generatePreview(); await tick();
    assert.ok(f.calls.includes('/api/preview'), 'Must reach a real pending preview request');
    app.removeItem(item);
    if (scenario === 'remove-success') pending.resolve(response(preview));
    else pending.reject(new Error('obsolete preview failed'));
    await generating;
    assert.equal(app.state().previewResult, null, 'A removed source must not regain a processed result');
    assert.equal(app.state().previewError, '', 'An obsolete request must not show an error for a removed source');
    assert.equal(app.state().previewFile, null);
    assert.equal(app.state().previewBusy, false);
    assert.equal(app.ui.processedImage.hidden, true);
    assert.equal(app.ui.glassesPreview.disabled, true);
    assert.equal(app.ui.previewState.textContent, 'Select a source to begin');
  } else if (scenario === 'restore-recovery') {
    let count = 0;
    f.network(async () => { if (++count === 1) throw new Error('temporarily offline'); return response([job(count === 2 ? 'processing' : 'completed')]); });
    await app.pollJobs();
    assert.match(app.ui.queueEmpty.querySelector('strong').textContent, /Could not restore/);
    assert.ok(f.jobTimer(), 'A failed empty restore must schedule a retry');
    await f.fire(f.jobTimer()[0]);
    assert.equal(app.state().items[0].state, 'processing');
    assert.equal(f.jobTimer()[1].delay, 1500, 'Recovery resumes ordinary active polling');
    await f.fire(f.jobTimer()[0]);
    assert.equal(app.state().items[0].state, 'completed');
    assert.equal(f.jobTimer(), undefined, 'Terminal jobs stop ordinary polling');
    assert.equal(app.ui.queueEmpty.querySelector('strong').textContent, 'A little empty, for now.');
  } else if (scenario === 'restore-backoff') {
    const pending = deferred();
    f.network(() => pending.promise);
    const first = app.pollJobs();
    await app.pollJobs(); await app.ui.refreshModels.click();
    assert.equal(f.calls.filter(url => url === '/api/jobs').length, 1, 'Only one jobs request may be in flight');
    pending.reject(new Error('offline')); await first; await tick();
    f.network(async () => { throw new Error('still offline'); });
    const delays = [];
    for (let attempt = 0; attempt < 9; attempt++) {
      const scheduled = f.jobTimer(); assert.ok(scheduled, 'Failures must continue retrying even without local jobs');
      delays.push(scheduled[1].delay); await f.fire(scheduled[0]);
    }
    assert.equal(delays[0], 1500);
    assert.ok(delays[1] > delays[0], 'Repeated failures back off');
    assert.ok(delays.every(delay => delay > 0 && delay <= 30000), 'Retry delay is capped at 30 seconds');
    assert.equal(delays.at(-1), delays.at(-2), 'Backoff eventually reaches its cap');
    assert.equal([...f.timers.values()].filter(timer => timer.delay !== 5500).length, 1, 'Only one retry timer is pending');
  } else if (scenario === 'manual-refresh') {
    f.network(async () => { throw new Error('offline'); });
    await app.pollJobs();
    const before = f.calls.filter(url => url === '/api/jobs').length;
    f.network(async url => response(url === '/api/jobs' ? [] : url === '/api/models' ? {models: []} : {}));
    await app.ui.refreshModels.click(); await tick();
    assert.equal(f.calls.filter(url => url === '/api/jobs').length, before + 1, 'Refresh status must also refresh queue');
    assert.equal(app.ui.queueEmpty.hidden, false);
    assert.equal(app.ui.queueEmpty.querySelector('strong').textContent, 'A little empty, for now.');
    assert.match(app.ui.queueEmpty.querySelector('small').textContent, /Add media above/);
    assert.equal(f.jobTimer(), undefined, 'A successful empty refresh cancels retries');
  } else if (scenario === 'cleanup-warning') {
    for (const state of ['completed', 'failed', 'canceled']) {
      f.network(async () => response([{...job(state), cleanup_warning: 'Temporary upload could not be removed.'}]));
      await app.pollJobs();
      assert.equal(app.state().items[0].state, state);
      assert.equal(app.state().items[0].cleanupWarning, 'Temporary upload could not be removed.');
      const text = node => [node.textContent, ...node.children.map(text)].join(' ');
      assert.match(text(app.ui.queueRows), /Cleanup warning: Temporary upload could not be removed[.]/);
    }
    f.network(async () => response([{...job('completed'), cleanup_warning: null}]));
    await app.pollJobs();
    assert.equal(app.state().items[0].cleanupWarning, null, 'A later successful cleanup clears its warning');
  } else if (scenario === 'remove-selected-queued') {
    app.setStatus({queue_capabilities:{schema_version:1,reorder_queued_exports:true,edit_queued_exports:true,remove_queued_exports:true,settings_snapshot:true}});
    const item={uid:'queued-1',jobId:'queued-1',purpose:'export',queuePosition:1,name:'source.png',file:{name:'source.png',size:100},mediaType:'image',state:'queued'};
    app.seed([item]);app.selectItem(item);app.setCalibration();
    const before=app.state().previewRevision;
    f.network(async url => response(url==='/api/jobs/queued-1/remove-queued'?{}:{}));
    await app.removeQueuedItem(item,'remove-queued-queued-1');
    assert.ok(f.calls.includes('/api/jobs/queued-1/remove-queued'),'The server queued-export removal endpoint must succeed first');
    assert.equal(app.state().items.length,0);
    assert.equal(app.state().selectedUid,null,'Removing the selected queued export clears selection');
    assert.equal(app.state().previewFile,null,'Removing the selected queued export clears its source preview');
    assert.ok(app.state().previewRevision>before,'Removing the selected queued export invalidates pending preview work');
    assert.equal(app.state().calibrationResult,null,'Removing the selected queued export invalidates calibration');
    assert.equal(app.state().calibrationResultContext,null);
    assert.equal(app.ui.calibrationReport.hidden,true);
  } else if (scenario === 'queue-render-state') {
    const item={uid:'local-1',name:'source.png',file:{name:'source.png',size:100},mediaType:'image',state:'ready',preflight:{estimate_basis:'Representative fixture basis',warnings:['One fixture warning.']}};
    app.seed([item]);app.updateQueue();
    let row=app.ui.queueRows.querySelector('.queue-row'),name=row.querySelector('.file-name');
    name.focus();
    f.network(async url => response(url==='/api/jobs'?[]:{paused:false,pending_count:0}));
    await app.pollJobs();await f.flushImmediate();
    assert.equal(f.document.activeElement.dataset.actionKey,'select-local-1','Polling must restore focus to the selected row action');
    row=app.ui.queueRows.querySelector('.queue-row');let details=row.querySelector('details[data-disclosure-key]');
    assert.ok(details,'The preflight fixture must render disclosure details');
    details.open=true;const disclosureKey=details.dataset.disclosureKey,summary=details.querySelector('summary');summary.focus();
    await app.pollJobs();await f.flushImmediate();
    row=app.ui.queueRows.querySelector('.queue-row');details=row.querySelector('details[data-disclosure-key]');
    assert.equal(details.open,true,'Polling must preserve expanded estimate details');
    assert.equal(f.document.activeElement.dataset.focusKey,`${disclosureKey}-summary`,'Polling must restore focus to the disclosure summary');
  } else if (scenario === 'named-presets') {
    app.ui.depthMode.value='depth-anything-v2-small-fp16';
    app.ui.depthInterval.value='2';app.ui.videoDecode.value='amd';
    app.ui.savedPresetName.value='XREAL Films';app.ui.savedPresetFavorite.checked=true;app.saveNamedPreset();
    app.ui.savedPresetName.value='Desktop';app.ui.savedPresetFavorite.checked=false;app.saveNamedPreset();
    assert.equal(app.savedPresets().length,2);assert.equal(app.savedPresets()[0].name,'XREAL Films');
    assert.equal(app.ui.savedPresetSelect.children[1].textContent,'★ XREAL Films');
    assert.equal(app.ui.savedPresetSelect.children[2].textContent,'Desktop','Favorites sort before other saved presets');
    const saved=JSON.parse(f.storage.get('stereolab.namedPresets.v1'));
    assert.equal(saved.schema_version,1);assert.equal(saved.presets[0].settings.layout,app.ui.layoutSelect.value);
    app.ui.strength.value='1';app.ui.depthInterval.value='1';app.ui.videoDecode.value='software';
    app.ui.savedPresetSelect.value='XREAL Films';app.selectNamedPreset();app.applyNamedPreset();
    assert.equal(app.ui.strength.value,'5','Applying a saved preset restores its settings');
    assert.equal(app.ui.depthInterval.value,'2');assert.equal(app.ui.videoDecode.value,'amd');
    assert.equal(app.currentSettingsForPreset().input_size,384,'FP16 preserves selectable neural detail');
    assert.match(app.ui.savedPresetStatus.textContent,/No job was started/);
    const reopened=fixture(f.storage);assert.equal(reopened.app.savedPresets().length,2,'Saved presets persist across page sessions');
    app.ui.strength.value='6';app.ui.savedPresetName.value='xreal films';app.ui.savedPresetFavorite.checked=false;app.saveNamedPreset();
    assert.equal(app.savedPresets().length,2,'Names are updated case-insensitively');assert.equal(app.savedPresets().find(preset=>preset.name==='xreal films').settings.strength,6);
    app.ui.savedPresetName.value='Bad\u0001Name';app.saveNamedPreset();assert.equal(app.savedPresets().length,2,'Invalid names are rejected');
    app.ui.savedPresetSelect.value='xreal films';app.deleteNamedPreset();assert.equal(app.savedPresets().length,1);
    app.ui.savedPresetSelect.value='Desktop';app.deleteNamedPreset();assert.equal(app.savedPresets().length,0);
    assert.equal(JSON.parse(f.storage.get('stereolab.namedPresets.v1')).presets.length,0);
  } else if (scenario === 'queue-controls') {
    let paused = false, posts = 0;
    f.network(async (url, options) => {
      if(url === '/api/jobs')return response([]);
      if(url === '/api/queue/control') {
        if(options?.method === 'POST'){paused=JSON.parse(options.body).paused;posts++;}
        return response({paused,pending_count:2,restart_behavior:'running'});
      }
      return response({});
    });
    await app.pollJobs();
    assert.equal(app.ui.pauseQueue.disabled,false);
    await app.ui.pauseQueue.click();
    assert.equal(posts,1);assert.equal(app.ui.pauseQueue.textContent,'Resume queue');
    assert.match(app.ui.queuePauseStatus.textContent,/2 pending/);
    await app.ui.pauseQueue.click();assert.equal(posts,2);
    assert.equal(app.ui.pauseQueue.textContent,'Pause after current job');
    f.network(async url => {if(url === '/api/jobs')return response([]);throw new Error('control unavailable');});
    await app.pollJobs();assert.equal(app.ui.pauseQueue.disabled,true);
    assert.match(app.ui.queuePauseStatus.textContent,/control unavailable/);
  } else if (scenario === 'storage-controls') {
    let cleanups=0;
    const candidate={path:'C:\\data\\uploads\\orphan.png',bytes:123,category:'uploads'};
    f.network(async (url,options) => {
      if(url === '/api/storage')return response({categories:{uploads:{bytes:123,files:1,errors:[]}},candidates:cleanups?[]:[candidate],errors:[],scope:'Managed files.'});
      if(url === '/api/storage/cleanup') {assert.deepEqual(JSON.parse(options.body),{paths:[candidate.path]});cleanups++;return response({deleted:[candidate.path],errors:[]});}
      return response(url === '/api/jobs'?[]:{paused:false,pending_count:0});
    });
    await app.ui.refreshStorage.click();
    assert.equal(app.ui.deleteStorageFiles.disabled,true,'Deletion requires explicit file selection');
    const checkbox=app.ui.storageFiles.children[0].children[0];checkbox.checked=true;
    checkbox.listeners.change[0]();assert.equal(app.ui.deleteStorageFiles.disabled,false);
    assert.match(app.ui.deleteStorageFiles.textContent,/123 B/);
    await app.ui.deleteStorageFiles.click();assert.equal(cleanups,1);
    assert.equal(app.ui.deleteStorageFiles.disabled,true);
    assert.match(app.ui.storageStatus.textContent,/Removed 1/);
  } else if (scenario === 'project-roundtrip') {
    const makeFile = (name, bytes) => ({name,size:bytes.length,bytes:Buffer.from(bytes)});
    const firstBytes=Buffer.from('first project source'),secondBytes=Buffer.from('second project video source');
    const firstFile=makeFile('still.png',firstBytes),secondFile=makeFile('clip.mp4',secondBytes);
    const frozen={...app.currentSettingsForPreset(),strength:2.5};
    app.ui.projectName.value='Winter trip';app.ui.strength.value='6';
    const first={uid:'first',file:firstFile,name:firstFile.name,size:firstFile.size,mediaType:'image',state:'ready'};
    const second={uid:'second',file:secondFile,name:secondFile.name,size:secondFile.size,mediaType:'video',state:'ready',settingsSnapshot:frozen,settingsFrozen:true,outputNameTemplateSnapshot:'{name}_{layout}',namingFrozen:true};
    app.seed([first,second]);app.selectItem(second);app.ui.sourceVideo.currentTime=12.5;app.ui.sourceVideo.readyState=1;app.ui.sourceVideo.duration=120;
    await app.saveProject();
    const projectText=await f.blobs.at(-1).text(),project=JSON.parse(projectText);
    assert.equal(project.project_name,'Winter trip');assert.deepEqual(project.sources.map(source=>source.name),['still.png','clip.mp4']);
    assert.equal(project.settings.strength,6);assert.equal(project.sources[1].settings.strength,2.5);
    assert.equal(project.preview_time_seconds,12.5);assert.equal(project.selected_source_id,project.sources[1].source_id);
    assert.equal(project.sources[0].sha256,nodeCrypto.createHash('sha256').update(firstBytes).digest('hex'));
    assert.equal(project.sources[1].sha256,nodeCrypto.createHash('sha256').update(secondBytes).digest('hex'));
    assert.equal(projectText.includes('first project source'),false,'Project files contain identities, never source bytes');

    const reopened=fixture(),reopenedApp=reopened.app;
    await reopenedApp.openProjectFile({size:Buffer.byteLength(projectText),text:async()=>projectText});
    let items=reopenedApp.state().items;
    assert.equal(Array.from(items,item=>item.name).join('|'),'still.png|clip.mp4');assert.ok(items.every(item=>item.state==='missing'));
    assert.equal(items[1].settingsSnapshot.strength,2.5);assert.equal(reopenedApp.ui.strength.value,'6');
    assert.equal(reopenedApp.state().selectedUid,items[1].uid);assert.equal(reopened.calls.some(url=>url==='/api/jobs'),false,'Opening a project never submits work');
    reopenedApp.ui.sourceVideo.readyState=1;reopenedApp.ui.sourceVideo.duration=120;
    const wrong=Buffer.from(secondBytes);wrong[0]^=0xff;
    await reopenedApp.relinkProjectSource(items[1],makeFile('renamed.mp4',wrong));
    assert.equal(items[1].state,'missing');assert.equal(items[1].file,null,'Same-size wrong content stays unresolved');
    await reopenedApp.relinkProjectSource(items[1],makeFile('renamed.mp4',secondBytes));
    assert.equal(items[1].state,'ready');assert.equal(items[1].name,'clip.mp4','The saved source name remains stable after relinking');
    assert.equal(reopenedApp.ui.sourceVideo.currentTime,12.5,'Selected video preview time is restored after relinking');
    await reopenedApp.relinkProjectSource(items[0],makeFile('renamed.png',firstBytes));
    assert.equal(Array.from(reopenedApp.state().items,item=>item.name).join('|'),'still.png|clip.mp4','Relinking in a different order preserves project queue order');
    assert.equal(reopened.calls.some(url=>url==='/api/jobs'),false);
    let postedFields=[];
    reopened.network(async (url,options)=>{if(url==='/api/jobs'){postedFields=options.body.entries;return {ok:false,status:409,json:async()=>({detail:'This source no longer matches the saved project. Locate the verified source again.'})};}return response({});});
    await reopenedApp.submitItem(items[1],reopenedApp.currentSettingsForPreset());
    assert.equal(postedFields.find(entry=>entry[0]==='expected_source_sha256')?.[1],project.sources[1].sha256,'Submission rechecks the relinked fingerprint on the server');
    assert.equal(items[1].state,'missing');assert.equal(items[1].file,null,'A changed source must be relinked again after server rejection');
  } else if (scenario === 'project-invalid-atomic') {
    const settings=app.currentSettingsForPreset(),source={source_id:'00000000-0000-4000-8000-000000000001',name:'source.png',media_type:'image',size_bytes:1,sha256:'a'.repeat(64),settings,output_name_template:null};
    const project={schema_version:1,project_id:'00000000-0000-4000-8000-000000000002',project_name:'Invalid duplicate',created_at_ms:0,settings,output_name_template:'{name}_{layout}',selected_source_id:null,preview_time_seconds:null,sources:[source,{...source}]};
    const old={uid:'existing',file:{name:'existing.png',size:1},name:'existing.png',mediaType:'image',state:'ready'};app.seed([old]);
    await app.openProjectFile({size:Buffer.byteLength(JSON.stringify(project)),text:async()=>JSON.stringify(project)});
    assert.equal(Array.from(app.state().items,item=>item.uid).join('|'),'existing','Invalid projects do not partially append queue entries');
    assert.match(app.ui.projectStatus.textContent,/not opened/);
  } else throw new Error(`Unknown scenario: ${scenario}`);
}

run(process.argv[2]).then(() => console.log(`PASS ${process.argv[2]}`)).catch(error => { console.error(error); process.exitCode = 1; });
