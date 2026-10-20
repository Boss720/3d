// Run the browser closure unchanged apart from replacing boot with a test accessor.
// Network and timers are controlled; no server, model, GPU, or browser is needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor() {
    this.value = ''; this.checked = false; this.hidden = false;
    this.dataset = {}; this.style = {}; this.options = []; this.children = [];
    this.listeners = {}; this.queries = new Map(); this.textContent = '';
    this.classList = {add() {}, remove() {}, toggle() {}};
    this.complete = true; this.naturalWidth = 640; this.naturalHeight = 360;
    this.clientWidth = 640; this.clientHeight = 360; this.currentTime = 0;
  }
  querySelector(key) {
    if (!this.queries.has(key)) this.queries.set(key, new Element());
    return this.queries.get(key);
  }
  querySelectorAll(selector) {
    if(selector === 'input:checked') {
      const collect = node => [...(node.type === 'checkbox' && node.checked ? [node] : []), ...node.children.flatMap(collect)];
      return this.children.flatMap(collect);
    }
    return [];
  }
  get firstElementChild() { return this.querySelector('first'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, fn) { (this.listeners[name] ??= []).push(fn); }
  click() { return Promise.all((this.listeners.click ?? []).map(fn => fn({target: this}))); }
  setAttribute() {} removeAttribute(name) { delete this[name]; }
  pause() {} load() {} focus() {} contains() { return false; }
  getContext() { return {drawImage() {}}; }
  toBlob(callback) { callback({}); }
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
const response = data => ({ok: true, json: async () => data});

function fixture() {
  const nodes = new Map(), timers = new Map(), calls = [];
  let nextTimer = 0, fetchHandler = async url => response(url === '/api/models' ? {models: []} : {});
  const document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    createElement() { return new Element(); }, createTextNode(text) { return {textContent: text}; },
    querySelectorAll() { return []; }, querySelector() { return null; }, addEventListener() {},
  };
  const context = vm.createContext({
    document, window: {addEventListener() {}}, console,
    sessionStorage: {getItem() { return null; }, setItem() {}},
    URL: {createObjectURL() { return 'blob:source'; }, revokeObjectURL() {}},
    FormData: class { append() {} }, crypto: {randomUUID() { return 'uuid'; }},
    fetch: async (url, options) => { calls.push(url); return fetchHandler(url, options); },
    setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  let source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
  const boot = 'updateCompletionNotificationControls();updateSettings(false);Promise.allSettled([loadStatus(),loadModels(),pollJobs()]);';
  assert.equal(source.split(boot).length, 2, 'Expected one application boot statement');
  source = source.replace(boot, `globalThis.lifecycle = {
    ui, generatePreview, removeItem, selectItem, pollJobs, calibrationContext,
    seed(next) { items = next; },
    setCalibration() { calibrationResult = {}; calibrationResultContext = calibrationContext(); ui.calibrationReport.hidden = false; ui.applyCalibration.hidden = false; },
    state() { return {items, selectedUid, previewFile, previewResult, previewRevision, previewBusy, previewError, calibrationResult, calibrationResultContext, jobsBusy}; }
  };`);
  vm.runInContext(source, context, {filename: 'static/app.js'});
  const app = context.lifecycle;
  const defaults = {layoutSelect: 'SBS', depthMode: 'balanced', executionMode: 'cpu', modelSize: '384', strength: '5', convergence: '.5', temporalSmoothing: '.35', resolution: '1080p', quality: 'balanced', previewDetail: 'proxy', processedClipDuration: '3', calibrationBudget: '15', performancePreset: 'balanced'};
  for (const [id, value] of Object.entries(defaults)) app.ui[id].value = value;
  app.ui.edgeRefinement.checked = true;
  app.ui.queueEmpty.querySelector('strong').textContent = 'A little empty, for now.';
  app.ui.queueEmpty.querySelector('small').textContent = 'Add media above. Completed exports are ready to download here.';
  return {
    app, timers, calls,
    network(fn) { fetchHandler = fn; },
    async fire(id) { const timer = timers.get(id); assert.ok(timer, 'Expected scheduled retry'); timers.delete(id); await timer.fn(); },
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
  } else throw new Error(`Unknown scenario: ${scenario}`);
}

run(process.argv[2]).then(() => console.log(`PASS ${process.argv[2]}`)).catch(error => { console.error(error); process.exitCode = 1; });
