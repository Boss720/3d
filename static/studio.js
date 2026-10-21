/* Cinematic workspace controls. Existing app.js owns media/settings/exports. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video = $('sourceVideo'), image = $('sourceImage'), strip = $('frameThumbnails');
  let generation = 0, decoder = null, disposeWait = null, activeTimes = [], thumbnailByUid = new Map();
  let lastSource = '';
  const clock = value => {
    const seconds = Math.max(0, Math.floor(Number(value) || 0));
    return seconds >= 3600 ? `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}` : `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  };
  function captureThumbnail(source, width, height) {
    const canvas=document.createElement('canvas');canvas.width=180;canvas.height=102;
    const scale=Math.min(canvas.width/width,canvas.height/height),w=width*scale,h=height*scale;
    const context=canvas.getContext('2d');context.fillStyle='#111116';context.fillRect(0,0,180,102);
    context.drawImage(source,(180-w)/2,(102-h)/2,w,h);
    return canvas.toDataURL('image/jpeg',.76);
  }
  function stopDecoder() {
    generation++;
    if (disposeWait) disposeWait();
    disposeWait = null;
    if (decoder) { decoder.pause(); decoder.removeAttribute('src'); decoder.load(); decoder = null; }
  }
  function seek(seconds) {
    if (!Number.isFinite(video.duration) || !video.getAttribute('src')) return;
    video.pause(); $('tabOriginal').click(); video.currentTime = Math.max(0, Math.min(video.duration, seconds));
    syncTransport();
  }
  function syncTransport() {
    const available = Boolean(video.getAttribute('src')) && Number.isFinite(video.duration) && video.duration > 0;
    for (const id of ['studioPlay', 'studioSeek', 'studioMute', 'studioRate']) $(id).disabled = !available;
    $('studioPlay').textContent = video.paused ? '▶' : 'Ⅱ';
    $('studioPlay').setAttribute('aria-label', video.paused ? 'Play source video' : 'Pause source video');
    $('studioMute').textContent = video.muted ? '♩' : '♫';
    $('studioMute').setAttribute('aria-label', video.muted ? 'Unmute source video' : 'Mute source video');
    $('studioTime').textContent = available ? `${clock(video.currentTime)} / ${clock(video.duration)}` : image.getAttribute('src') ? 'STILL IMAGE' : '00:00 / 00:00';
    $('studioSeek').value = available ? String(video.currentTime / video.duration * 100) : '0';
    $('studioRate').value = String(video.playbackRate);
    let nearest = -1;
    if (activeTimes.length) nearest = activeTimes.reduce((best, time, i) => Math.abs(time-video.currentTime) < Math.abs(activeTimes[best]-video.currentTime) ? i : best, 0);
    [...strip.querySelectorAll('button')].forEach((button, i) => button.setAttribute('aria-pressed', String(i === nearest)));
    $('framePrevious').disabled = !available || !activeTimes.length;
    $('frameNext').disabled = !available || !activeTimes.length;
  }
  function metadata() {
    const isVideo = Boolean(video.getAttribute('src'));
    const width = isVideo ? video.videoWidth : image.naturalWidth;
    const height = isVideo ? video.videoHeight : image.naturalHeight;
    $('sourceMetadata').textContent = width && height ? `${width} × ${height}${isVideo && Number.isFinite(video.duration) ? `   |   ${clock(video.duration)}   |   VIDEO` : '   |   PHOTO'}` : 'Import a photo or video to begin';
  }
  function rememberThumbnail(url) {
    const selected = $('queueRows').querySelector('.file-name[aria-pressed="true"]');
    const uid = selected?.closest('.queue-row')?.dataset.itemUid;
    if (uid) { thumbnailByUid.set(uid, url); compactQueue(); }
  }
  function frameButton(url, seconds, label) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'frame-thumb';
    button.setAttribute('aria-label', label); button.setAttribute('aria-pressed', 'false');
    const img = document.createElement('img'); img.src = url; img.alt = ''; img.draggable = false;
    const text = document.createElement('span'); text.textContent = seconds === null ? 'SOURCE IMAGE' : clock(seconds);
    button.append(img, text); if (seconds !== null) button.addEventListener('click', () => seek(seconds));
    else button.addEventListener('click', () => $('tabOriginal').click());
    return button;
  }
  function waitForMedia(element, event, token, action) {
    return new Promise((resolve, reject) => {
      let timeout;
      const clean = () => { clearTimeout(timeout); element.removeEventListener(event, done); element.removeEventListener('error', failed); if (disposeWait === cancel) disposeWait = null; };
      const done = () => { clean(); token === generation ? resolve() : reject(new Error('superseded')); };
      const failed = () => { clean(); reject(new Error('decode')); };
      const cancel = () => { clean(); reject(new Error('superseded')); };
      disposeWait = cancel; element.addEventListener(event, done, {once:true}); element.addEventListener('error', failed, {once:true});
      timeout = setTimeout(failed, 5000);
      try { action(); } catch (error) { clean(); reject(error); }
    });
  }
  async function buildVideoStrip() {
    const url = video.getAttribute('src');
    if (!url || !Number.isFinite(video.duration) || video.duration <= 0 || lastSource === url) return;
    stopDecoder(); lastSource = url; const token = generation;
    strip.replaceChildren(); activeTimes = [];
    $('filmstripStatus').textContent = 'Loading source frame thumbnails…';
    const media = document.createElement('video'); decoder = media; media.muted = true; media.preload = 'auto'; media.playsInline = true;
    try {
      await waitForMedia(media, 'loadeddata', token, () => { media.src = url; media.load(); });
      const count = 9;
      for (let i = 0; i < count; i++) {
        if (token !== generation) return;
        const seconds = Math.max(0, (video.duration - Math.min(.1,video.duration / 10)) * i / (count - 1));
        if (Math.abs(media.currentTime - seconds) > .001) await waitForMedia(media, 'seeked', token, () => { media.currentTime = seconds; });
        if (token !== generation) return;
        const thumb = captureThumbnail(media,media.videoWidth,media.videoHeight);
        strip.append(frameButton(thumb, seconds, `Seek to ${clock(seconds)}`)); activeTimes.push(seconds);
        if (i === 0) rememberThumbnail(thumb);
        syncTransport();
      }
      $('filmstripStatus').textContent = 'Nine source frames ready. Choose a frame to seek.';
    } catch (error) {
      if (token === generation) {
        $('filmstripStatus').textContent = 'Some frame thumbnails could not be decoded. Playback seeking remains available.';
        if (!strip.children.length) { const note = document.createElement('p'); note.className='filmstrip-empty'; note.textContent='Frame thumbnails unavailable for this video'; strip.append(note); }
      }
    } finally {
      media.pause(); media.removeAttribute('src'); media.load(); if (decoder === media) decoder = null;
    }
  }
  function resetSource() {
    stopDecoder(); lastSource = ''; activeTimes = []; strip.replaceChildren();
    const note = document.createElement('p'); note.className = 'filmstrip-empty'; note.textContent = 'Import a video to browse its frames'; strip.append(note);
    $('filmstripStatus').textContent = ''; syncTransport(); metadata();
  }
  function compactQueue() {
    for (const row of $('queueRows').children) {
      const knownThumb = thumbnailByUid.get(row.dataset.itemUid), knownIcon = row.querySelector('.file-icon');
      if (knownThumb && knownIcon && !knownIcon.querySelector('img')) { const img=document.createElement('img');img.src=knownThumb;img.alt='';knownIcon.replaceChildren(img); }
      if (row.dataset.studioCompact) continue; row.dataset.studioCompact = 'true';
      const uid = row.dataset.itemUid, copy = row.querySelector('.file-copy'), actions = row.querySelector('.row-actions');
      if (!copy || !actions) continue;
      const details = document.createElement('details'); details.className='queue-row-details'; details.dataset.disclosureKey=`studio-${uid}`;
      const summary = document.createElement('summary'); summary.textContent='Details & actions'; summary.dataset.focusKey=`studio-${uid}-summary`; details.append(summary);
      for (const node of [...copy.children]) if (!node.classList.contains('file-name') && !node.classList.contains('file-meta')) details.append(node);
      const status = copy.querySelector('.file-status');
      if (status) {
        const labels={ready:'Ready',uploading:'Uploading',queued:'Queued',processing:'Processing',completed:'Complete',failed:'Failed',canceled:'Canceled'};
        for(const [state,label] of Object.entries(labels))if(status.classList.contains(state)){
          const message=status.textContent;status.title=message;status.textContent=label;
          if(message!==label){const note=document.createElement('p');note.className='queue-source-detail';note.textContent=message;details.append(note);}
          break;
        }
      }
      const statusColumn = document.createElement('div'); statusColumn.className='queue-status-column'; if (status) statusColumn.append(status);
      const format = document.createElement('span'); format.className='queue-format';
      const meta = copy.querySelector('.file-meta'); const info = meta?.textContent || '';
      format.textContent = info.includes('·') ? info.split('·').at(-1).trim() : '—';
      const more = document.createElement('div'); more.className='queue-extra-actions';
      for (const action of [...actions.children]) if (!action.classList.contains('row-progress') && !['Download ↓','Cancel'].includes(action.textContent)) more.append(action);
      if (more.children.length) details.append(more);
      // Keep existing stable focus keys so app.js polling can restore focus/open state.
      copy.append(details); row.insertBefore(format, actions); row.insertBefore(statusColumn, actions);
      const thumb = thumbnailByUid.get(uid); const icon = row.querySelector('.file-icon');
      if (thumb && icon) { const img=document.createElement('img');img.src=thumb;img.alt='';icon.replaceChildren(img); }
    }
  }
  // app.js restores disclosures synchronously during its render, before this observer.
  // Save our open state separately because inserted details do not exist at that point.
  const disclosureOpen = new Set();
  $('queueRows').addEventListener('toggle', event => { const key=event.target.dataset.disclosureKey;if(key?.startsWith('studio-'))event.target.open?disclosureOpen.add(key):disclosureOpen.delete(key); }, true);
  new MutationObserver(() => { compactQueue(); for(const d of $('queueRows').querySelectorAll('.queue-row-details')) d.open=disclosureOpen.has(d.dataset.disclosureKey); }).observe($('queueRows'), {childList:true});
  $('studioPlay').addEventListener('click', async () => { if (!video.paused) video.pause(); else { $('tabOriginal').click(); try { await video.play(); } catch { $('filmstripStatus').textContent='Playback unavailable. Try another source container or codec.'; } } syncTransport(); });
  $('studioSeek').addEventListener('input', () => seek(Number($('studioSeek').value) / 100 * video.duration));
  $('studioMute').addEventListener('click', () => {video.muted=!video.muted;syncTransport();});
  $('studioRate').addEventListener('change', () => {video.playbackRate=Number($('studioRate').value);});
  $('studioFullscreen').addEventListener('click', async () => {try {if(document.fullscreenElement)await document.exitFullscreen();else await $('previewStage').requestFullscreen();} catch { $('filmstripStatus').textContent='Fullscreen is unavailable in this browser.'; }});
  $('framePrevious').addEventListener('click',()=>seek([...activeTimes].reverse().find(time=>time<video.currentTime-.1)??0));
  $('frameNext').addEventListener('click',()=>seek(activeTimes.find(time=>time>video.currentTime+.1)??video.duration));
  for (const event of ['timeupdate','play','pause','ended','volumechange','ratechange','durationchange','emptied']) video.addEventListener(event,syncTransport);
  video.addEventListener('loadeddata',()=>{metadata();syncTransport();void buildVideoStrip();});
  image.addEventListener('load',()=>{
    if(!image.getAttribute('src'))return;
    stopDecoder();
    // app.js revokes its source URL on selection changes. Keep an independent,
    // bounded thumbnail so a queue image survives selection of another file.
    const thumb=captureThumbnail(image,image.naturalWidth,image.naturalHeight);
    strip.replaceChildren(frameButton(thumb,null,'Show source image'));activeTimes=[];rememberThumbnail(thumb);syncTransport();metadata();
  });
  new MutationObserver(resetSource).observe(video,{attributes:true,attributeFilter:['src']});
  new MutationObserver(()=>{if(!image.getAttribute('src'))resetSource();}).observe(image,{attributes:true,attributeFilter:['src']});
  new MutationObserver(()=>{$('previewTitle').textContent=$('sourceName').textContent==='No media selected'?'Untitled source':$('sourceName').textContent;}).observe($('sourceName'),{childList:true,subtree:true,characterData:true});
  function syncFormats() {
    let quickFormat=false;
    for(const button of document.querySelectorAll('[data-layout]')){const selected=button.dataset.layout===$('layoutSelect').value;button.setAttribute('aria-pressed',String(selected));quickFormat ||= selected;}
    $('formatMoreLabel').textContent=quickFormat?'More formats & encoding':($('layoutSelect').selectedOptions[0]?.textContent||'More formats & encoding');
  }
  for(const button of document.querySelectorAll('[data-layout]'))button.addEventListener('click',()=>{$('layoutSelect').value=button.dataset.layout;$('layoutSelect').dispatchEvent(new Event('change',{bubbles:true}));syncFormats();});
  $('layoutSelect').addEventListener('change',syncFormats);
  // Preset application may change select.value without a native change event.
  new MutationObserver(syncFormats).observe($('previewModeLabel'),{childList:true,characterData:true,subtree:true});
  video.controls=false;syncTransport();syncFormats();compactQueue();
  window.addEventListener('beforeunload',()=>{stopDecoder();thumbnailByUid.clear();});
})();
