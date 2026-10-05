// Content script: captura keydown/keyup. Solo actúa si la grabación está activa
// y el sitio está en la allowlist. No calcula features: solo eventos brutos.
(() => {
  if (window.__tdCollectorLoaded) return;
  window.__tdCollectorLoaded = true;

  const MIN_PRESSES_PER_SAMPLE = 3;   // evita muestras vacías (Enter en campo vacío)
  const TEXT_INPUT_TYPES = new Set(['text', 'search', 'email', 'url', 'tel', 'number', '']);

  let settings = null;
  let sampleId = 1;
  let active = false;
  let pool = { next: 0, end: 0 };
  let requestingPool = false;
  const pressing = new Map();          // código de tecla -> {id, sample, participant, session, key, keyCode}
  let buffer = [];
  let flushTimer = null;
  let pressesInSample = 0;
  let pendingEnter = null;             // press_id de un Enter que podría cerrar la muestra
  const flaggedLocal = new Set();
  let warn = '';

  // ---------- utilidades ----------
  const alive = () => { try { return !!(chrome.runtime && chrome.runtime.id); } catch (_) { return false; } };

  function send(msg) {
    if (!alive()) { handleDead(); return Promise.resolve(null); }
    try { return chrome.runtime.sendMessage(msg).catch(() => null); }
    catch (_) { handleDead(); return Promise.resolve(null); }
  }

  function handleDead() {   // la extensión se recargó: esta página ya no puede registrar
    if (!active) return;
    active = false;
    warn = 'Extensión recargada: refresca la página';
    updateIndicator(true);
  }

  const target = (e) => (e.composedPath && e.composedPath()[0]) || e.target;
  const keyId = (e) => e.code || ('k' + e.keyCode);

  function isRecordableTarget(el) {
    if (!el || el.nodeType !== 1) return false;
    if (el.tagName === 'INPUT') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'password') return false;                       // nunca contraseñas
      const ac = (el.getAttribute('autocomplete') || '').toLowerCase();
      if (/password|cc-/.test(ac)) return false;
      return TEXT_INPUT_TYPES.has(t);
    }
    if (el.tagName === 'TEXTAREA') return true;
    return el.isContentEditable === true;
  }

  // ---------- indicador visible ----------
  let host = null, shadow = null;

  function updateIndicator(force) {
    if (!active && !force) { if (host) { host.remove(); host = null; shadow = null; } return; }
    if (!host) {
      host = document.createElement('div');
      host.style.cssText = 'all:initial;position:fixed;top:10px;right:10px;z-index:2147483647;pointer-events:none;';
      shadow = host.attachShadow({ mode: 'closed' });
      shadow.innerHTML = `
        <style>
          .b{font:600 12px/1 system-ui,sans-serif;color:#fff;background:#d93025;padding:7px 10px;border-radius:999px;
             display:flex;align-items:center;gap:7px;box-shadow:0 2px 8px rgba(0,0,0,.3);opacity:.92}
          .b.warn{background:#b45309}
          .d{width:9px;height:9px;border-radius:50%;background:#fff;animation:p 1.2s infinite}
          @keyframes p{50%{opacity:.25}}
        </style>
        <div class="b"><span class="d"></span><span class="t"></span></div>`;
      (document.documentElement || document).appendChild(host);
    }
    const b = shadow.querySelector('.b');
    b.classList.toggle('warn', !!warn);
    shadow.querySelector('.t').textContent = warn
      ? '⚠ ' + warn
      : `REC · ${settings.participant_id}/${settings.session_id} · muestra ${sampleId}`;
  }

  // ---------- estado / ajustes ----------
  function hostAllowed(list) {
    const h = location.hostname.toLowerCase();
    return (list || []).some(d => {
      d = String(d).trim().toLowerCase();
      return d && (h === d || h.endsWith('.' + d));
    });
  }

  function apply(s) {
    settings = s;
    const should = !!(s && s.enabled && hostAllowed(s.allowlist));
    if (should !== active) {
      active = should;
      if (active) { ensurePool(); }
      else { flush(); pressing.clear(); pendingEnter = null; warn = ''; }
      send({ type: 'status', recording: active });
    }
    updateIndicator();
  }

  async function ensurePool() {
    if (requestingPool) return;
    requestingPool = true;
    try {
      const r = await send({ type: 'allocPressIds' });
      if (r && r.ok) pool = { next: r.start, end: r.end };
    } finally { requestingPool = false; }
  }

  // ---------- registro ----------
  function record(type, e, p) {
    const ts = Number((performance.timeOrigin + e.timeStamp).toFixed(3));   // ms epoch, sub-ms
    buffer.push([p.participant, p.session, p.sample, type, p.keyCode, ts, p.id, p.key]);
    if (buffer.length >= 200) flush();
    else if (!flushTimer) flushTimer = setTimeout(flush, 1000);
  }

  function flush() {
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
    if (!buffer.length) return;
    const events = buffer;
    buffer = [];
    send({ type: 'events', events });
  }

  function onKeyDown(e) {
    if (!active || !e.isTrusted) return;
    if (e.repeat) return;                         // auto-repeat del SO
    const k = keyId(e);
    if (pressing.has(k)) return;                  // repeat sin flag .repeat
    if (!isRecordableTarget(target(e))) return;
    if (pool.next >= pool.end) { ensurePool(); return; }
    const id = pool.next++;
    if (pool.end - pool.next < 2000) ensurePool();
    const p = { id, sample: sampleId, participant: settings.participant_id,
                session: settings.session_id, key: e.key, keyCode: e.keyCode };
    pressing.set(k, p);
    record('down', e, p);
    pressesInSample++;
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) pendingEnter = id;
  }

  function onKeyUp(e) {
    if (!active || !e.isTrusted) return;
    const k = keyId(e);
    const p = pressing.get(k);
    if (!p) return;                               // up sin down registrado: se ignora
    pressing.delete(k);                           // OJO: sin comprobar el target (el foco pudo cambiar)
    record('up', e, p);
    if (pendingEnter === p.id) {
      pendingEnter = null;
      if (pressesInSample >= MIN_PRESSES_PER_SAMPLE) endSample();
    }
  }

  function endSample() {
    flush();
    const closing = sampleId;
    pressesInSample = 0;
    sampleId = closing + 1;
    flaggedLocal.clear();
    warn = '';
    send({ type: 'endSample', sample_id: closing });
    updateIndicator();
  }

  // ---------- detección de contaminación (pegar / soltar / autocompletar) ----------
  function flag(reason) {
    if (!active) return;
    const key = sampleId + ':' + reason;
    if (flaggedLocal.has(key)) return;
    flaggedLocal.add(key);
    send({ type: 'flag', participant_id: settings.participant_id, session_id: settings.session_id,
           sample_id: sampleId, reason });
    warn = 'Pegado/autocompletado detectado: muestra marcada';
    updateIndicator();
  }

  window.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('keyup', onKeyUp, true);
  window.addEventListener('blur', () => { flush(); }, true);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') flush(); });
  window.addEventListener('pagehide', flush);

  document.addEventListener('paste', (e) => { if (isRecordableTarget(target(e))) flag('paste'); }, true);
  document.addEventListener('drop', (e) => { if (isRecordableTarget(target(e))) flag('drop'); }, true);
  document.addEventListener('beforeinput', (e) => {
    if (!isRecordableTarget(target(e))) return;
    const t = e.inputType;
    if (t === 'insertFromPaste' || t === 'insertFromPasteAsQuotation' || t === 'insertFromYank') flag('paste');
    else if (t === 'insertFromDrop') flag('drop');
    else if (t === 'insertReplacementText') flag('autocomplete');
  }, true);

  // Cierre de muestra al pulsar el botón de enviar del chat
  document.addEventListener('click', (e) => {
    if (!active || !e.isTrusted) return;
    const el = target(e);
    const b = el && el.closest && el.closest('button,[role="button"]');
    if (!b) return;
    const s = [b.getAttribute('aria-label'), b.getAttribute('data-testid'), b.getAttribute('title'), b.id].join(' ');
    if (/send|envoy|enviar|submit/i.test(s) && pressesInSample >= MIN_PRESSES_PER_SAMPLE) endSample();
  }, true);

  // ---------- arranque y sincronización ----------
  chrome.storage.onChanged.addListener((ch, area) => {
    if (area !== 'local') return;
    if (ch.sampleState && ch.sampleState.newValue) {
      const n = ch.sampleState.newValue.sample_id;
      if (n !== sampleId) { sampleId = n; pressesInSample = 0; flaggedLocal.clear(); warn = ''; }
      updateIndicator();
    }
    if (ch.settings) apply(ch.settings.newValue);
  });

  (async () => {
    try {
      const d = await chrome.storage.local.get(['settings', 'sampleState']);
      sampleId = (d.sampleState && d.sampleState.sample_id) || 1;
      apply(d.settings);
    } catch (_) {}
  })();
})();
