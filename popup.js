const $ = (id) => document.getElementById(id);
const store = chrome.storage.local;
let settings = { enabled: false, participant_id: 'P01', session_id: 'S01', allowlist: [] };
let sampleId = 1, meta = { chunks: 1, count: 0 }, flagged = [];

const parseAllow = (txt) => [...new Set(txt.split(/\n/).map(s => s.trim().toLowerCase()
  .replace(/^https?:\/\//, '').replace(/^www\./, '').split('/')[0]).filter(Boolean))];

function render() {
  const on = settings.enabled;
  $('pill').textContent = on ? 'grabando' : 'inactivo';
  $('pill').classList.toggle('on', on);
  $('toggle').textContent = on ? 'Detener grabación' : 'Iniciar grabación';
  $('toggle').classList.toggle('rec', on);
  $('pid').disabled = $('sid').disabled = on;
  if (document.activeElement !== $('pid')) $('pid').value = settings.participant_id;
  if (document.activeElement !== $('sid')) $('sid').value = settings.session_id;
  if (document.activeElement !== $('allow')) $('allow').value = settings.allowlist.join('\n');
  $('stats').textContent = `Eventos: ${meta.count} · Muestra actual: ${sampleId} · Muestras marcadas: ${flagged.length}`;
  $('hint').textContent = `Hoy es ${new Date().toISOString().slice(0, 10)}. Usa un session_id distinto por día. ` +
    `Escribe sin pegar, dictar ni autocompletar. No se graban campos de contraseña.`;
}

async function load() {
  const d = await store.get(['settings', 'sampleState', 'meta', 'flagged']);
  if (d.settings) settings = d.settings;
  sampleId = (d.sampleState && d.sampleState.sample_id) || 1;
  if (d.meta) meta = d.meta;
  flagged = d.flagged || [];
  render();
}

const save = (partial) => chrome.runtime.sendMessage({ type: 'saveSettings', settings: partial });

$('toggle').onclick = async () => {
  if (!settings.enabled) {
    const pid = $('pid').value.trim(), sid = $('sid').value.trim();
    const allow = parseAllow($('allow').value);
    if (!pid || !sid) return alert('Indica participant_id y session_id.');
    if (!allow.length) return alert('Añade al menos un sitio donde grabar.');
    await save({ enabled: true, participant_id: pid, session_id: sid, allowlist: allow });
  } else {
    await save({ enabled: false });
  }
};

$('allow').onchange = () => save({ allowlist: parseAllow($('allow').value) });
$('pid').onchange = () => { if (!settings.enabled) save({ participant_id: $('pid').value.trim() || 'P01' }); };
$('sid').onchange = () => { if (!settings.enabled) save({ session_id: $('sid').value.trim() || 'S01' }); };

$('addSite').onclick = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    const h = new URL(tab.url).hostname.replace(/^www\./, '');
    if (!h) return;
    const list = parseAllow($('allow').value);
    if (!list.includes(h)) list.push(h);
    $('allow').value = list.join('\n');
    save({ allowlist: list });
  } catch (_) { alert('No se pudo leer la URL de esta pestaña.'); }
};

$('endSample').onclick = () => chrome.runtime.sendMessage({ type: 'endSample', sample_id: sampleId });

const esc = (v) => { const s = String(v); return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

$('export').onclick = async () => {
  const { meta: m } = await store.get('meta');
  const n = (m && m.chunks) || 1;
  const keys = Array.from({ length: n }, (_, i) => 'chunk:' + i);
  const data = await store.get(keys);
  const lines = ['participant_id,session_id,sample_id,event,key_code,timestamp_ms,press_id,key'];
  for (const k of keys) for (const row of (data[k] || [])) lines.push(row.map(esc).join(','));
  if (lines.length === 1) return alert('No hay datos que exportar.');
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  download(`keystrokes_${stamp}.csv`, lines.join('\n') + '\n');
  const { flagged: f } = await store.get('flagged');
  if (f && f.length) {
    const fl = ['participant_id,session_id,sample_id,reason', ...f.map(x => [x.participant_id, x.session_id, x.sample_id, x.reason].map(esc).join(','))];
    download(`flagged_samples_${stamp}.csv`, fl.join('\n') + '\n');
  }
};

$('clear').onclick = async () => {
  if (!confirm('¿Eliminar TODOS los datos registrados? Esta acción no se puede deshacer.')) return;
  await chrome.runtime.sendMessage({ type: 'clearAll' });
};

chrome.storage.onChanged.addListener((ch, area) => {
  if (area !== 'local') return;
  if (ch.settings) settings = ch.settings.newValue;
  if (ch.sampleState) sampleId = ch.sampleState.newValue.sample_id;
  if (ch.meta) meta = ch.meta.newValue;
  if (ch.flagged) flagged = ch.flagged.newValue || [];
  render();
});

load();
