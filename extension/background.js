// Service worker: almacenamiento serializado, contadores globales y badge.
const CHUNK_SIZE = 5000;     // eventos por chunk en chrome.storage.local
const PRESS_BLOCK = 10000;   // bloque de press_id reservado por pestaña

const DEFAULT_SETTINGS = {
  enabled: false,
  participant_id: 'P01',
  session_id: 'S01',
  allowlist: [
    'chatgpt.com', 'chat.openai.com', 'claude.ai', 'gemini.google.com',
    'chat.mistral.ai', 'copilot.microsoft.com', 'perplexity.ai', 'poe.com'
  ]
};

// Cola para que las escrituras read-modify-write nunca se pisen.
let queue = Promise.resolve();
const enqueue = (fn) => {
  const run = queue.then(fn);
  queue = run.catch(() => {});
  return run;
};

const store = chrome.storage.local;

async function getSettings() {
  const { settings } = await store.get('settings');
  return { ...DEFAULT_SETTINGS, ...(settings || {}) };
}

chrome.runtime.onInstalled.addListener(() => enqueue(async () => {
  const d = await store.get(['settings', 'meta', 'sampleState', 'pressCounter', 'flagged']);
  const init = {};
  if (!d.settings) init.settings = DEFAULT_SETTINGS;
  if (!d.meta) init.meta = { chunks: 1, count: 0 };
  if (!d.sampleState) init.sampleState = { sample_id: 1 };
  if (!d.pressCounter) init.pressCounter = { next: 1 };
  if (!d.flagged) init.flagged = [];
  if (Object.keys(init).length) await store.set(init);
}));

async function appendEvents(events) {
  if (!events.length) return { ok: true };
  const { meta } = await store.get('meta');
  const m = meta || { chunks: 1, count: 0 };
  let idx = m.chunks - 1;
  const curKey = 'chunk:' + idx;
  let chunk = (await store.get(curKey))[curKey] || [];
  const out = {};
  let i = 0;
  while (i < events.length) {
    const room = CHUNK_SIZE - chunk.length;
    if (room <= 0) {
      out['chunk:' + idx] = chunk;
      idx++;
      chunk = [];
      continue;
    }
    const take = events.slice(i, i + room);
    chunk = chunk.concat(take);
    i += take.length;
  }
  out['chunk:' + idx] = chunk;
  out.meta = { chunks: idx + 1, count: m.count + events.length };
  await store.set(out);
  return { ok: true };
}

async function handle(msg, sender) {
  switch (msg.type) {
    case 'events':
      return enqueue(() => appendEvents(msg.events || []));

    case 'allocPressIds':
      return enqueue(async () => {
        const { pressCounter } = await store.get('pressCounter');
        const start = (pressCounter && pressCounter.next) || 1;
        await store.set({ pressCounter: { next: start + PRESS_BLOCK } });
        return { ok: true, start, end: start + PRESS_BLOCK };
      });

    case 'endSample':
      return enqueue(async () => {
        const { sampleState } = await store.get('sampleState');
        const cur = (sampleState && sampleState.sample_id) || 1;
        // Solo avanza si quien pide cerrar conoce el sample actual (evita dobles saltos).
        if (cur === msg.sample_id) await store.set({ sampleState: { sample_id: cur + 1 } });
        return { ok: true };
      });

    case 'flag':
      return enqueue(async () => {
        const { flagged } = await store.get('flagged');
        const list = flagged || [];
        const f = { participant_id: msg.participant_id, session_id: msg.session_id, sample_id: msg.sample_id, reason: msg.reason };
        if (!list.some(x => x.participant_id === f.participant_id && x.session_id === f.session_id &&
                            x.sample_id === f.sample_id && x.reason === f.reason)) {
          list.push(f);
          await store.set({ flagged: list });
        }
        return { ok: true };
      });

    case 'saveSettings':
      return enqueue(async () => {
        const old = await getSettings();
        const next = { ...old, ...msg.settings };
        if (next.participant_id !== old.participant_id || next.session_id !== old.session_id) {
          await store.set({ sampleState: { sample_id: 1 } });
        }
        await store.set({ settings: next });
        return { ok: true };
      });

    case 'clearAll':
      return enqueue(async () => {
        const { meta } = await store.get('meta');
        const n = (meta && meta.chunks) || 1;
        await store.remove(Array.from({ length: n }, (_, i) => 'chunk:' + i));
        // pressCounter NO se reinicia: las pestañas abiertas conservan sus bloques de ids.
        await store.set({ meta: { chunks: 1, count: 0 }, flagged: [], sampleState: { sample_id: 1 } });
        return { ok: true };
      });

    case 'status':
      if (sender && sender.tab && sender.tab.id != null) {
        const tabId = sender.tab.id;
        await chrome.action.setBadgeText({ tabId, text: msg.recording ? 'REC' : '' });
        if (msg.recording) await chrome.action.setBadgeBackgroundColor({ tabId, color: '#d93025' });
      }
      return { ok: true };

    default:
      return { ok: false, error: 'unknown message' };
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  handle(msg, sender)
    .then(r => sendResponse(r || { ok: true }))
    .catch(e => sendResponse({ ok: false, error: String(e) }));
  return true; // respuesta asíncrona
});
