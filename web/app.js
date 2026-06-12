// Vendored locally (web/vendor/) so the editor is fully offline — nothing is
// fetched from an external CDN. wavesurfer.js v7.12.7 (each file is a
// self-contained ESM bundle). app.js is served at /static/app.js, so these
// './vendor/…' specifiers resolve to /static/vendor/….
import WaveSurfer from './vendor/wavesurfer.esm.js'
import RegionsPlugin from './vendor/plugins/regions.esm.js'
import TimelinePlugin from './vendor/plugins/timeline.esm.js'
import HoverPlugin from './vendor/plugins/hover.esm.js'

const $ = (s) => document.querySelector(s)
const COLORS = { pause: '79,141,249', filler: '245,158,11', profanity: '239,68,68', bad_take: '167,139,250', hesitation: '45,212,191', manual: '52,211,153' }
const TYPE_RU = { pause: 'пауза', filler: 'паразит', profanity: 'мат', bad_take: 'дубль', hesitation: 'заминка', manual: 'ручной' }
const colorOf = (seg) => `rgba(${COLORS[seg.type] || '120,120,120'},${seg.enabled ? 0.22 : 0.08})`
const round = (x) => Math.round(x * 1000) / 1000
const escapeHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
// SVG-иконка из спрайта index.html (<symbol id="i-...">) — замена эмодзи-глифов.
const icon = (name) => `<svg class="ic" aria-hidden="true"><use href="#i-${name}"/></svg>`
const fmt = (t) => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` }
const fmtcs = (t) => { t = Math.max(0, t || 0); return `${fmt(t)}.${String(Math.floor((t % 1) * 100)).padStart(2, '0')}` }

const st = {
  media: null, duration: 0, words: [], paras: [], segs: [],
  selected: null, selRange: null, preview: false, showCuts: true,
  manualN: 0, addingRegion: false, activeWord: -1, splitMark: null,
  inP: null, outP: null, fitPx: 0,
  hasSession: false, curDir: null, outDir: '', rdefaults: {}, pickFolderCb: null, cb: '',
  syncOffset: parseFloat(localStorage.getItem('fve_sync') || '0') || 0,
  spans: null,            // cached NodeList of .w spans (perf)
  _merged: null,          // memoized mergedRemoves() result
  // «После» — финальный таймлайн (after-strip)
  afterMode: false,       // показываем ли производную «после»-полосу
  _kept: null,            // memoized keptSegments() (инвалидируется вместе с _merged)
  _newDuration: 0,        // длина финального видео (сумма kept-сегментов)
  _pkRaw: null,           // ссылка на сырой массив peaks из /api/peaks (для рисования after-strip)
  task: null,             // current running task name from SSE (null if idle)
  // F2 — предпросмотр субтитров и глав (ПЕРЕИСПОЛЬЗУЕТ origToFinal/finalToOrig из F1)
  subsMode: false,        // показываем ли оверлей-капшн поверх <video>
  subsCues: [],           // реплики [{start,end,text}] в КООРДИНАТАХ ФИНАЛА (пострезных)
  subsActive: -1,         // индекс показанной сейчас реплики (-1 = ничего)
  chaptersData: [],       // главы [{time,title}] в координатах финала
  metadataData: null,     // B — метаданные {title,description,tags:[],hook} или null
  // F3 — очередь нескольких роликов (отдельно от одиночного редактора)
  queueJobs: [],          // последний снимок [{id,name,status,percent,stage,result,error,...}]
  queueRunning: false,    // крутится ли воркер очереди
  queuePollTimer: 0,      // setInterval id опроса GET /api/queue
  // F4 — Clip Maker: кандидаты Shorts (план §4) в ОРИГИНАЛЬНЫХ координатах
  clipsData: [],          // кандидаты [{id,start,end,score,hook_phrase,...}], сорт. по score desc
  clipsSel: new Set(),    // id карточек, выбранных чекбоксами под рендер
  clipsActive: null,      // id карточки с янтарной подсветкой диапазона на волне (null = нет)
  clipsResults: new Map(),// id → результат рендера {ok, mp4|error} (переживает пере-рендер карточек)
  _clipsRenderIds: null,  // порядок id в последнем POST /api/clips/render (мапа результатов)
  _clipPreviewEnd: null,  // ▶-предпросмотр: авто-пауза на этом orig-времени (null = выключен)
  _clipPrevPreview: false,// каким был «пропускать вырезы» до ▶ (восстановить после авто-паузы)
  // save serialization
  dirty: false, saving: false, savingPromise: null, saveError: false, _navigating: false,
  // A6 — онбординг: тост о тихом CPU-фоллбэке показываем ОДИН раз на файл
  // (смена файла = location.reload() → st пересоздаётся, флаг сбрасывается сам)
  cpuWarned: false,
}
const UNDO_CAP = 50
const undoStack = [], redoStack = []
let ws, regions, video, wrapper = null
const regionById = new Map()
// F4 — Clip Maker: подсветка диапазона кандидата на волне (янтарная, как .clipBadge)
// + DOM-карточки по id (живые ссылки для обновления eff-длительности и результатов).
const CLIP_HL_COLOR = 'rgba(245,158,11,0.20)'
let clipHlRegion = null
const clipEls = new Map()
// Таб-бар правой колонки: декларации живут ДО вызова init() (:122) — bindTabs
// зовётся из init синхронно, const/let ниже по файлу были бы в TDZ.
const TABS = [
  { id: 'cuts', tab: 'tab-cuts', panel: 'panel-cuts' },
  { id: 'chapters', tab: 'tab-chapters', panel: 'panel-chapters' },
  { id: 'meta', tab: 'tab-meta', panel: 'panel-meta' },
  { id: 'clips', tab: 'tab-clips', panel: 'panel-clips' },
]
const TAB_KEY = 'fve_tab'
// Вкладка-адресат фоновой задачи (busy-спиннер вместо бейджа, точка при ошибке).
const TASK_TAB = { transcribe: 'cuts', detect: 'cuts', preview_chapters: 'chapters', preview_metadata: 'meta', preview_clips: 'clips', render_clips: 'clips' }
let activeTab = 'cuts'
const tabDescr = (id) => TABS.find((t) => t.id === id)
let saveTimer = null, es = null, raf = 0
let phMetrics = { total: 0, client: 0 }   // cached scroll/client widths for playhead
let phScrollRaf = 0
const AFTER_H = 44                         // CSS-px высота after-strip (см. .afterCanvas)
let afterRedrawTimer = 0                    // debounce таймер для drawAfterStrip
let afterScrubbing = false                 // активен ли скраб по after-strip

/* ---------- toast layer ---------- */
function toast(msg, kind = 'info', opts = {}) {
  const layer = $('#toasts'); if (!layer) { console.log(kind, msg); return null }
  const el = document.createElement('div')
  el.className = 'toast ' + kind
  el.setAttribute('role', 'status')
  const ic = document.createElement('span'); ic.className = 'ticon'
  ic.innerHTML = icon(kind === 'success' ? 'check' : kind === 'error' ? 'x' : 'info')
  const m = document.createElement('div'); m.className = 'tmsg'; m.textContent = msg
  const x = document.createElement('button'); x.className = 'tclose'; x.innerHTML = icon('x'); x.title = 'Закрыть'; x.setAttribute('aria-label', 'Закрыть')
  el.appendChild(ic); el.appendChild(m); el.appendChild(x)
  const remove = () => { if (el.parentNode) el.remove() }
  x.onclick = remove
  layer.appendChild(el)
  const ttl = opts.sticky ? 0 : (opts.ttl || (kind === 'error' ? 6000 : 3000))
  if (ttl) setTimeout(remove, ttl)
  return el
}

/* ---------- progress bar (visual + ARIA) ---------- */
// Держим role="progressbar" в синхроне: width + aria-valuenow + aria-valuetext.
function setProgress(pct, text) {
  const bar = $('#progressBar')
  if (bar) {
    const n = Math.max(0, Math.min(100, Math.round(pct)))
    bar.style.width = pct + '%'
    bar.setAttribute('aria-valuenow', String(n))
  }
  if (text != null) {
    const t = $('#progressText'); if (t) t.textContent = text
    if (bar) bar.setAttribute('aria-valuetext', text)
  }
}

let errorToastEl = null
function showSaveError(msg) {
  if (errorToastEl && errorToastEl.parentNode) errorToastEl.remove()
  errorToastEl = toast(msg, 'error', { sticky: true })
}
function clearSaveError() { if (errorToastEl && errorToastEl.parentNode) errorToastEl.remove(); errorToastEl = null }

// Warn before leaving with unsaved or in-flight changes.
window.addEventListener('beforeunload', (e) => {
  if (st.dirty || st.saving) { e.preventDefault(); e.returnValue = ''; return '' }
})

init()

async function init() {
  bindFiles()
  bindQueue()
  bindPrivacy()
  bindModels()
  bindTabs()      // ДО ранних return'ов: таб-бар жив и кликабелен в пустой сессии
  checkHealth()   // A7: карточка «ffmpeg не найден» (не блокирует загрузку — fire-and-forget)
  const s = await (await fetch('/api/state')).json()
  if (s.network) renderNetBadge(s.network)   // P2-#4: zero-upload badge from bootstrap
  if (s.no_session) {
    st.hasSession = false; st.curDir = s.start_dir
    $('#filename').textContent = 'Ролик не выбран'
    renderEmptyStepper()   // A6: степпер «с чего начать» вместо плейсхолдера транскрипта
    if (s.queue_running) startQueuePoll()   // очередь может крутиться без активного клипа
    else if (s.queue_pending > 0) loadQueueList()   // P2-#6: засветить бейдж восстановленной очереди (воркер стоит)
    openFiles(false)
    return
  }
  if (s.queue_running) startQueuePoll()      // показать бейдж занятой очереди при загрузке
  else if (s.queue_pending > 0) loadQueueList()   // P2-#6: бейдж восстановленной очереди при старте без автозапуска воркера
  st.hasSession = true
  st.outDir = s.out_dir || ''
  st._inpPath = s.path || ''   // F3: абсолютный путь текущего клипа (для постановки в очередь)
  st.rdefaults = s.defaults || {}
  // LLM-off badge
  if (s.llm_ready === false) $('#llmBadge').classList.remove('hidden')
  else $('#llmBadge').classList.add('hidden')
  // Cache-bust by session token so switching clips can never reuse a cached
  // /api/video (the bug where the new clip loaded but the OLD video played).
  const cb = s.v ? ('?v=' + encodeURIComponent(s.v)) : ''
  video = $('#video'); video.src = '/api/video' + cb
  st.cb = cb
  st.media = s.media; st.duration = s.media.duration
  $('#filename').textContent = s.filename
  $('#censor').textContent = 'цензура мата: ' + s.censor_method
  $('#tcTot').textContent = fmt(st.duration)
  $('#btnTranscribe').classList.toggle('hidden', s.has_transcript)
  if (!s.has_transcript) { $('#btnTranscribe').classList.add('pulse'); flash('Не транскрибировано — нажми «Транскрибировать»') }

  $('#progress').classList.remove('hidden'); $('#progress').classList.add('indeterminate')
  setProgress(35, 'Готовлю аудиоволну…')
  let pk
  try { const r = await fetch('/api/peaks' + (st.cb || '')); if (!r.ok) throw new Error('HTTP ' + r.status); pk = await r.json() }
  catch (e) { pk = { peaks: [], duration: st.duration }; toast('Не удалось построить аудиоволну: ' + e.message, 'error') }
  $('#progress').classList.add('hidden'); $('#progress').classList.remove('indeterminate')
  st._pkRaw = pk.peaks   // ссылка (не копия) на сырые peaks для рисования «После»-полосы
  setupWave(pk)
  bindUI(); bindKeys()
  if (s.has_transcript) await loadData()
  loadClipsFromCache()   // F4: панель клипов из out/<stem>.clips.json — без LLM
  if (s.task && s.task.running) followTask(s.task.name)

  video.addEventListener('timeupdate', () => { if (video.paused) onFrame() })
  video.addEventListener('seeked', onFrame)
  video.addEventListener('play', () => { $('#btnPlay').innerHTML = icon('pause'); cancelAnimationFrame(raf); raf = requestAnimationFrame(loop) })
  video.addEventListener('pause', () => { $('#btnPlay').innerHTML = icon('play'); cancelAnimationFrame(raf); onFrame() })
}

const flash = (t) => { $('#cutSummary').textContent = t }
const segOf = (id) => st.segs.find((s) => s.id === id)

/* ---------- waveform ---------- */
function setupWave(pk) {
  ws = WaveSurfer.create({
    container: '#waveform', media: video, height: 82,
    waveColor: ['#7b84b8', '#525c8f'], progressColor: ['#aeb6ea', '#7e88c8'],
    cursorColor: '#ffffff', cursorWidth: 2,
    peaks: [pk.peaks], duration: pk.duration || st.duration,
    fillParent: true, autoScroll: true,
  })
  try {
    ws.registerPlugin(TimelinePlugin.create({
      container: $('#ruler'), height: 18, primaryLabelInterval: 5,
      style: { color: '#8b93a7', fontSize: '10px' }, formatTimeCallback: fmt,
    }))
    ws.registerPlugin(HoverPlugin.create({
      lineColor: '#6c8cff', lineWidth: 1, labelBackground: '#222b3b',
      labelColor: '#e8ebf2', formatTimeCallback: fmtcs,
    }))
  } catch (e) { console.warn('plugin', e) }
  regions = ws.registerPlugin(RegionsPlugin.create())
  regions.enableDragSelection({ color: 'rgba(40,180,110,0.30)' })
  regions.on('region-created', onRegionCreated)
  // In wavesurfer v7 'region-updated' fires continuously during drag/resize. Update seg bounds
  // live (cheap), but coalesce the heavy renderCutlist+refreshCuts to after movement stops.
  regions.on('region-updated', (r) => onRegionDragging(r))
  regions.on('region-clicked', (r, e) => { e.stopPropagation(); const s = segOf(r.id); if (s) { select(s.id); seek(s.start) } })
  ws.on('ready', () => {
    wrapper = ws.getWrapper ? ws.getWrapper() : null
    if (wrapper) wrapper.addEventListener('scroll', onWaveScroll)
    st.fitPx = $('#waveform').clientWidth / Math.max(1, st.duration)
    refreshPhMetrics(); setupAfterStrip(); onFrame()
  })
  ws.on('redraw', refreshPhMetrics)
  window.addEventListener('resize', () => { refreshPhMetrics(); st.fitPx = $('#waveform').clientWidth / Math.max(1, st.duration); if (st.afterMode) scheduleAfterRedraw() })
}

let altDown = false
window.addEventListener('keydown', (e) => { if (e.key === 'Alt') altDown = true })
window.addEventListener('keyup', (e) => { if (e.key === 'Alt') altDown = false })
window.addEventListener('blur', () => { altDown = false })

// Snap a time to the nearest word boundary array via binary search; null if out of tolerance.
function snapTo(t, key, tol = 0.18) {
  const ws_ = st.words; if (!ws_.length) return t
  let lo = 0, hi = ws_.length - 1, best = t, bestD = Infinity
  // binary search for closest, then check neighbors
  while (lo <= hi) { const m = (lo + hi) >> 1; if (ws_[m][key] < t) lo = m + 1; else hi = m - 1 }
  for (let i = Math.max(0, lo - 1); i <= Math.min(ws_.length - 1, lo + 1); i++) {
    const d = Math.abs(ws_[i][key] - t); if (d < bestD) { bestD = d; best = ws_[i][key] }
  }
  return bestD <= tol ? best : t
}

let dragTimer = 0, dragUndoTaken = false, dragRegionId = null, dragAlt = false
function onRegionDragging(r) {
  if (st.addingRegion) return
  const s = segOf(r.id); if (!s) return
  // Take a single undo snapshot at the start of a drag gesture.
  if (!dragUndoTaken || dragRegionId !== r.id) { pushUndo(); dragUndoTaken = true; dragRegionId = r.id }
  dragAlt = altDown   // capture live so a late release doesn't change snapping
  // Live (cheap) bound update so cut math / playhead stay in sync mid-drag.
  s.start = round(r.start); s.end = round(r.end); st._merged = null; st._kept = null
  clearTimeout(dragTimer)
  dragTimer = setTimeout(() => finishRegionDrag(r), 140)   // fires once movement settles = drag END
}
function finishRegionDrag(r) {
  dragUndoTaken = false; dragRegionId = null
  const s = segOf(r.id); if (!s) return
  let a = r.start, b = r.end
  // Word-snapping (Alt bypasses); guard min width.
  if (!dragAlt) {
    const sa = snapTo(a, 'start'), sb = snapTo(b, 'end')
    if (sb - sa >= 0.02) { a = sa; b = sb }
    if (a !== r.start || b !== r.end) { st.addingRegion = true; r.setOptions({ start: a, end: b }); st.addingRegion = false }
  }
  s.start = round(a); s.end = round(b); st._merged = null; st._kept = null
  renderCutlist(); refreshCuts(); markDirty()
}

function onRegionCreated(r) {
  if (st.addingRegion) return
  // Min-width guard mirroring addManual(): drop too-tiny drag selections.
  if (r.end - r.start < 0.02) { st.addingRegion = true; r.remove(); st.addingRegion = false; return }
  pushUndo()
  const seg = { id: r.id, start: round(r.start), end: round(r.end), type: 'manual', action: 'remove', enabled: true, text: 'ручной вырез', reason: '', word: '' }
  st.manualN++; st.segs.push(seg); regionById.set(seg.id, r); r.setOptions({ color: colorOf(seg) })
  renderCutlist(); refreshCuts(); select(seg.id); markDirty()
}

function renderRegions() {
  st.addingRegion = true
  regions.clearRegions(); regionById.clear()
  if (st.showCuts) for (const seg of st.segs) {
    const r = regions.addRegion({ id: seg.id, start: seg.start, end: seg.end, color: colorOf(seg), drag: true, resize: true })
    regionById.set(seg.id, r)
  }
  // F4: clearRegions() снёс и янтарную подсветку выбранного клипа — вернуть её.
  clipHlRegion = null
  const ac = (st.clipsData || []).find((c) => String(c.id) === st.clipsActive)
  if (ac) clipHlRegion = regions.addRegion({ id: 'clip-hl', start: ac.start, end: ac.end, color: CLIP_HL_COLOR, drag: false, resize: false })
  st.addingRegion = false
}
const refreshRegionColor = (id) => { const r = regionById.get(id), s = segOf(id); if (r && s) r.setOptions({ color: colorOf(s) }) }

/* ---------- data ---------- */
async function loadData() {
  const tr = await (await fetch('/api/transcript')).json()
  // A6: тихий CPU-фоллбэк — бейдж «CPU» + один тост (device_used пишется в кэш
  // транскрипта; null у старых кэшей или до A6-бэкенда → ничего не показываем).
  updateCpuBadge(tr.device_used, tr.device_configured)
  st.words = []; st.paras = []; let gi = 0
  tr.segments.forEach((seg, si) => {
    const para = { words: [] }
    ;(seg.words || []).forEach((w, wi) => {
      // si/wi — адрес слова для PUT /api/transcript/word; edited — флаг «изменено»
      // (хранится в st.words, чтобы пережить пере-рендер renderTranscript).
      const o = { i: gi++, word: w.w, start: w.s, end: w.e, si, wi, edited: false }
      st.words.push(o); para.words.push(o)
    })
    if (para.words.length) st.paras.push(para)
  })
  let cl; try { const r = await fetch('/api/cutlist'); cl = r.ok ? await r.json() : { segments: [] } } catch { cl = { segments: [] } }
  st.segs = cl.segments || []; st.manualN = st.segs.filter((s) => s.type === 'manual').length
  renderTranscript(); renderRegions(); renderCutlist(); refreshCuts()
  flash('Готово к монтажу')
}

function renderTranscript() {
  const box = $('#transcript'); box.replaceChildren()
  if (!st.paras.length) {
    const ph = document.createElement('div'); ph.className = 'empty placeholder'
    ph.innerHTML = icon('edit') + '<div>Транскрипт появится здесь — нажми «Транскрибировать»</div>'
    box.appendChild(ph); st.spans = null; return
  }
  st.paras.forEach((para, p) => {
    const el = document.createElement('p'); el.className = 'para'
    const g = document.createElement('span'); g.className = 'gutter'; g.innerHTML = icon('scissors'); g.title = 'Вырезать абзац'; g.dataset.p = p
    el.appendChild(g)
    for (const w of para.words) {
      const sp = document.createElement('span'); sp.className = 'w'; sp.dataset.i = w.i
      if (w.edited) { sp.classList.add('edited'); sp.title = 'изменено' }   // восстановить пометку после пере-рендера
      sp.textContent = w.word.trim() + ' '; el.appendChild(sp)
    }
    box.appendChild(el)
  })
  st.spans = box.querySelectorAll('.w')   // cache for highlight/refreshCuts/doSearch
}

/* ---------- A6/A7 — онбординг: степпер, CPU-фоллбэк, карточка ffmpeg ---------- */
// A6: 3 шага «с чего начать». inFiles=true — вариант ВНУТРИ пикера файлов:
// при пустой сессии пикер открыт модально поверх всего и не закрывается,
// так что подсказки первого запуска обязаны жить в нём, а не под скримом.
function stepperHTML(inFiles) {
  const step1 = inFiles
    ? 'Выберите ролик ниже или перетащите файл'
    : 'Выберите ролик — <a href="#" id="stepperFiles">Файлы</a>'
  return '<div class="stepperTitle">С чего начать</div>' +
    '<ol class="steps">' +
      `<li><span class="stepNum">1</span><span>${step1}</span></li>` +
      '<li><span class="stepNum">2</span><span>Нажмите «Транскрибировать» — речь превратится в текст (в первый раз скачается модель распознавания)</span></li>' +
      '<li><span class="stepNum">3</span><span>Проверьте вырезы и нажмите «Рендер»</span></li>' +
    '</ol>'
}

// A6: когда сессии нет — вместо плейсхолдера транскрипта компактный степпер
// из 3 шагов (ссылка «Файлы» открывает тот же пикер, что и кнопка в хедере).
function renderEmptyStepper() {
  const box = $('#transcript'); if (!box) return
  const el = document.createElement('div')
  el.className = 'stepper'
  el.innerHTML = stepperHTML(false)
  box.replaceChildren(el)
  el.querySelector('#stepperFiles').onclick = (e) => { e.preventDefault(); openFiles(false) }
}

// A6: транскрипция просилась на CUDA, но реально шла на CPU (нет драйвера/cuDNN).
// Бейдж «CPU» висит, пока открыт этот транскрипт; тост — один раз (st.cpuWarned).
const CPU_FALLBACK_MSG = 'Транскрипция шла на CPU — CUDA недоступна (медленнее в разы). Проверьте драйвер NVIDIA.'
function updateCpuBadge(used, configured) {
  const fellBack = used === 'cpu' && configured === 'cuda'
  const b = $('#cpuBadge'); if (b) b.classList.toggle('hidden', !fellBack)
  if (fellBack && !st.cpuWarned) { st.cpuWarned = true; toast(CPU_FALLBACK_MSG, 'warn', { ttl: 8000 }) }
}

// A7: /api/health при старте — нет ffmpeg → несдвигаемая карточка-предупреждение.
// Обычно живёт поверх левой панели (НЕ оверлей: остальной UI кликабелен), но при
// пустой сессии — ВНУТРИ открытого пикера файлов: тот модален и неотключаем
// (скрим z-50 выше карточки), и без переноса юзер первого запуска без ffmpeg
// застрял бы в пикере, так и не увидев команду установки. Старый сервер без
// /api/health или сетевая ошибка → молча пропускаем.
async function checkHealth() {
  let h
  try { const r = await fetch('/api/health'); if (!r.ok) return; h = await r.json() }
  catch { return }
  if (h && h.ffmpeg && h.ffmpeg.found === false) showFfmpegCard()
}

function showFfmpegCard() {
  if ($('#ffmpegCard')) return
  const el = document.createElement('div')
  el.id = 'ffmpegCard'; el.className = 'ffmpegCard'
  el.innerHTML =
    `<div class="ffTitle">${icon('info')}ffmpeg не найден</div>` +
    '<div class="ffText">Открытие видео и рендер не работают. Установите: <code>winget install Gyan.FFmpeg</code> — затем откройте новый терминал и перезапустите редактор.</div>' +
    '<div class="ffActs"><button id="btnFfRecheck" class="btn small">Проверить снова</button></div>'
  const head = $('#files .filesHead')
  if (!st.hasSession && head && !$('#files').classList.contains('hidden')) {
    el.classList.add('inFiles'); head.after(el)   // онбординг: пикер уже открыт поверх всего
  } else {
    const pane = $('#scriptPane'); if (!pane) return
    pane.appendChild(el)
  }
  el.querySelector('#btnFfRecheck').onclick = recheckFfmpeg
}

async function recheckFfmpeg() {
  const btn = $('#btnFfRecheck'); if (btn) btn.disabled = true
  let h = null
  try { const r = await fetch('/api/health'); if (r.ok) h = await r.json() } catch {}
  if (btn) btn.disabled = false
  if (h && h.ffmpeg && h.ffmpeg.found) {
    const card = $('#ffmpegCard'); if (card) card.remove()
    toast('ffmpeg найден', 'success')
  } else {
    toast(h ? 'ffmpeg всё ещё не найден — откройте новый терминал и перезапустите редактор' : 'Не удалось проверить — сервер недоступен', 'warn')
  }
}

/* ---------- cut math ---------- */
function mergedRemoves() {
  if (st._merged) return st._merged
  const iv = st.segs.filter((s) => s.enabled && s.action === 'remove').map((s) => [s.start, s.end]).filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0])
  const out = []
  for (const [a, b] of iv) { if (out.length && a <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], b); else out.push([a, b]) }
  st._merged = out
  return out
}
// Binary search over the cached sorted, non-overlapping intervals (called per preview frame).
function insideRemoved(t) {
  const m = mergedRemoves()
  let lo = 0, hi = m.length - 1
  while (lo <= hi) { const k = (lo + hi) >> 1, iv = m[k]; if (t < iv[0]) hi = k - 1; else if (t >= iv[1]) lo = k + 1; else return iv }
  return null
}

/* ---------- финальный таймлайн: карта координат (зеркало vpipe/timeline.py) ---------- */
// Дополнение merged-вырезов к [0, duration] → kept-сегменты с префикс-суммой финальной длины.
// Кэшируется в st._kept / st._newDuration; инвалидируется вместе с st._merged.
function keptSegments() {
  if (st._kept) return st._kept
  const rem = mergedRemoves()
  const kept = []
  let cursor = 0, finalStart = 0
  for (const [a, b] of rem) {
    if (a > cursor) { kept.push({ start: cursor, end: a, finalStart }); finalStart += (a - cursor) }
    cursor = Math.max(cursor, b)
  }
  if (cursor < st.duration) { kept.push({ start: cursor, end: st.duration, finalStart }); finalStart += (st.duration - cursor) }
  st._kept = kept; st._newDuration = finalStart
  return kept
}
// orig-время → финальное время. Внутри вырезанного → стык (seam = конец предыдущего kept).
function origToFinal(t) {
  const kept = keptSegments()
  if (!kept.length) return 0
  if (t <= kept[0].start) return 0
  if (t >= st.duration) return st._newDuration
  // бинарный поиск: последний kept с start <= t
  let lo = 0, hi = kept.length - 1, i = 0
  while (lo <= hi) { const m = (lo + hi) >> 1; if (kept[m].start <= t) { i = m; lo = m + 1 } else hi = m - 1 }
  const seg = kept[i]
  if (t < seg.end) return seg.finalStart + (t - seg.start)   // внутри kept
  return seg.finalStart + (seg.end - seg.start)               // в вырезе после seg → стык
}
// финальное время → orig-время (для скраба). Бинарный поиск по finalStart.
function finalToOrig(ft) {
  const kept = keptSegments()
  if (!kept.length) return 0
  if (ft <= 0) return kept[0].start
  if (ft >= st._newDuration) return kept[kept.length - 1].end
  let lo = 0, hi = kept.length - 1, i = 0
  while (lo <= hi) { const m = (lo + hi) >> 1; if (kept[m].finalStart <= ft) { i = m; lo = m + 1 } else hi = m - 1 }
  return kept[i].start + (ft - kept[i].finalStart)
}

/* ---------- финальный таймлайн: рисование, плейхед, скраб ---------- */
function scheduleAfterRedraw() {
  clearTimeout(afterRedrawTimer)
  afterRedrawTimer = setTimeout(drawAfterStrip, 80)
}
// Рисует свёрнутую (пострезную) волну, переиспользуя сырые peaks из /api/peaks. Один проход canvas.
function drawAfterStrip() {
  const cv = $('#afterCanvas'); if (!cv) return
  const dpr = window.devicePixelRatio || 1
  const cssW = cv.clientWidth || $('#waveform').clientWidth || 1
  cv.width = Math.max(1, Math.round(cssW * dpr))
  cv.height = Math.round(AFTER_H * dpr)
  const ctx = cv.getContext('2d'); if (!ctx) return
  const W = cv.width, H = cv.height, mid = H / 2
  ctx.clearRect(0, 0, W, H)
  // тёмный фон
  ctx.fillStyle = 'rgba(20,26,38,1)'; ctx.fillRect(0, 0, W, H)
  const kept = keptSegments()
  const peaks = st._pkRaw
  const newDur = st._newDuration
  if (!kept.length || !newDur || !peaks || !peaks.length || !st.duration) return
  const peaksLen = peaks.length
  // волна
  ctx.fillStyle = 'rgba(126,136,200,0.85)'
  for (let x = 0; x < W; x++) {
    const ft = (x / W) * newDur
    const tOrig = finalToOrig(ft)   // переиспользуем карту координат (kept кэширован в st._kept)
    let pi = Math.floor(tOrig / st.duration * peaksLen)
    if (pi < 0) pi = 0; else if (pi >= peaksLen) pi = peaksLen - 1
    const amp = Math.abs(peaks[pi] || 0)
    const h = Math.max(1, amp * (mid - 1))
    ctx.fillRect(x, mid - h, 1, h * 2)
  }
  // риски на стыках (все kept кроме последнего)
  const accent = (getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#6c8cff').trim()
  ctx.fillStyle = accent
  for (let i = 0; i < kept.length - 1; i++) {
    const ft = kept[i].finalStart + (kept[i].end - kept[i].start)
    const x = Math.round(ft / newDur * W)
    ctx.fillRect(x, 0, Math.max(1, Math.round(dpr)), H)
  }
}
// Второй плейхед на after-strip (в CSS-пикселях, абс. внутри #afterWrap).
function updateAfterPlayhead() {
  const ph = $('#afterPh'), cv = $('#afterCanvas'); if (!ph || !cv) return
  const newDur = st._newDuration; if (!newDur) { ph.style.left = '0px'; return }
  const ft = origToFinal(video.currentTime)
  const x = ft / newDur * cv.clientWidth
  ph.style.left = Math.max(0, Math.min(cv.clientWidth, x)) + 'px'
}
// Скраб по after-strip: финальное время под курсором → seek в соответствующий original-момент.
function afterScrubTo(e, cv) {
  const newDur = st._newDuration; if (!newDur) return
  const rect = cv.getBoundingClientRect()
  const px = (e.clientX != null ? e.clientX : 0) - rect.left
  const ft = Math.max(0, Math.min(1, px / Math.max(1, cv.clientWidth))) * newDur
  seek(finalToOrig(ft))
}
function setupAfterStrip() {
  const cv = $('#afterCanvas'); if (!cv) return
  cv.onpointerdown = (e) => {
    if (!st.afterMode) return
    afterScrubbing = true
    try { cv.setPointerCapture(e.pointerId) } catch {}
    afterScrubTo(e, cv)
  }
  cv.onpointermove = (e) => { if (afterScrubbing) afterScrubTo(e, cv) }
  const end = (e) => { if (!afterScrubbing) return; afterScrubbing = false; try { cv.releasePointerCapture(e.pointerId) } catch {} }
  cv.onpointerup = end
  cv.onpointercancel = end
}
// Переключатель «До / После». Переиспользует существующую логику preview/skipCuts.
function setAfterMode(on) {
  st.afterMode = !!on
  $('#waveform').classList.toggle('source-dim', st.afterMode)
  $('#afterWrap').classList.toggle('hidden', !st.afterMode)
  $('#tcFinal').classList.toggle('hidden', !st.afterMode)
  const btn = $('#btnAfterToggle')
  if (btn) { btn.setAttribute('aria-pressed', String(st.afterMode)); btn.innerHTML = icon('split') + (st.afterMode ? 'До' : 'После') }
  // «После» включает пропуск вырезов; при возврате в «До» восстанавливаем
  // прежний выбор пользователя (P / чекбокс), а не затираем его в false.
  const skip = $('#skipCuts')
  if (st.afterMode) {
    st._previewBeforeAfter = st.preview
    st.preview = true
    if (skip) skip.checked = true
  } else {
    st.preview = !!st._previewBeforeAfter
    if (skip) skip.checked = st.preview
    st._previewBeforeAfter = false
  }
  if (st.afterMode) { scheduleAfterRedraw(); updateAfterPlayhead() }
}

/* ---------- F2: субтитры (оверлей поверх плеера) ---------- */
// Реплики приходят в КООРДИНАТАХ ФИНАЛА — показываем по origToFinal(currentTime).
let _subsFetch = 0
async function loadSubtitles() {
  if (!st.words.length) { st.subsCues = []; st.subsActive = -1; updateSubCaption(); return }   // нет транскрипта — нечего показывать
  // Generation token: autosave (700ms) can fire loadSubtitles repeatedly; only
  // the latest fetch may apply, so out-of-order resolves can't clobber newer cues.
  const gen = ++_subsFetch
  let data
  try {
    const r = await fetch('/api/preview/subtitles', { method: 'POST' })
    if (gen !== _subsFetch) return
    if (!r.ok) { let d = 'HTTP ' + r.status; try { const j = await r.json(); if (j && j.detail) d = j.detail } catch {} ; throw new Error(d) }
    data = await r.json()
    if (gen !== _subsFetch) return
  } catch (e) {
    if (gen !== _subsFetch) return
    st.subsCues = []; st.subsActive = -1; updateSubCaption()
    toast('Не удалось загрузить субтитры: ' + e.message, 'error'); return
  }
  st.subsCues = data.cues || []
  st.subsActive = -1
  updateSubCaption()
}
// Вызывается каждый кадр из onFrame(), когда subsMode включён. O(log n) на кадр.
function updateSubCaption() {
  const el = $('#subOverlay'); if (!el) return
  const cues = st.subsCues
  if (!st.subsMode || !cues.length) { if (st.subsActive !== -1) { el.classList.add('hidden'); st.subsActive = -1 } return }
  const ft = origToFinal(video.currentTime)
  // бинарный поиск: последняя реплика со start <= ft
  let lo = 0, hi = cues.length - 1, i = -1
  while (lo <= hi) { const m = (lo + hi) >> 1; if (cues[m].start <= ft) { i = m; lo = m + 1 } else hi = m - 1 }
  const hit = (i >= 0 && ft < cues[i].end) ? i : -1
  if (hit !== -1) {
    if (hit !== st.subsActive) { el.textContent = cues[hit].text; el.classList.remove('hidden'); st.subsActive = hit }
  } else if (st.subsActive !== -1) { el.classList.add('hidden'); st.subsActive = -1 }
}
function setSubsMode(on) {
  st.subsMode = !!on
  const btn = $('#btnSubsToggle')
  if (btn) btn.setAttribute('aria-pressed', String(st.subsMode))
  if (st.subsMode) { loadSubtitles() }
  else { const el = $('#subOverlay'); if (el) el.classList.add('hidden'); st.subsActive = -1 }
}

/* ---------- F2: главы (фоновая задача — LLM) ---------- */
async function loadChapters() {
  if (st.task) { toast('Дождись завершения задачи перед предпросмотром глав', 'info'); return }
  let res
  try { res = await fetch('/api/preview/chapters', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось запросить главы (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить предпросмотр глав'); return }
  let j; try { j = await res.json() } catch { j = {} }
  if (j && j.ok === false && j.reason === 'llm_off') {
    toast('LLM выключена — главы недоступны. Включи Ollama.', 'info'); return
  }
  // Задача запущена — следим по SSE (renderChapters вызовется в followTask).
  followTask('preview_chapters')
}
// ---- Таб-бар правой колонки (Вырезы/Главы/Мета/Клипы) -----------------------
// Панели смонтированы ВСЕГДА (contenteditable-правки меты и clipEls живут в
// DOM); переключение — только атрибут hidden. Приход данных НИКОГДА не
// переключает вкладку — только бейдж + конечный пульс + точка «непрочитано».
function setActiveTab(id) {
  const target = tabDescr(id) || TABS[0]
  // Фокус внутри скрываемой панели не должен «умереть» в display:none —
  // переносим на активный таб (единственное исключение из «хоткей не трогает фокус»).
  const focusLost = TABS.some((t) => {
    if (t.id === target.id) return false
    const p = document.getElementById(t.panel)
    return p && !p.hidden && p.contains(document.activeElement)
  })
  for (const t of TABS) {
    const btn = document.getElementById(t.tab)
    const panel = document.getElementById(t.panel)
    const active = t.id === target.id
    if (panel) panel.hidden = !active
    if (!btn) continue
    btn.setAttribute('aria-selected', String(active))
    btn.tabIndex = active ? 0 : -1
    if (active) {
      // Первая активация гасит точку «непрочитано» и останавливает пульс.
      const dot = btn.querySelector('.tDot'); if (dot) dot.classList.add('hidden')
      for (const el of btn.querySelectorAll('.fresh')) el.classList.remove('fresh')
    }
  }
  activeTab = target.id
  if (focusLost) { const tb = document.getElementById(target.tab); if (tb) tb.focus() }
  try { localStorage.setItem(TAB_KEY, target.id) } catch {}
}

// Бейдж вкладки: chapters/clips — счётчик (пуст при 0 → скрыт через :empty),
// meta — точка наличия. Бейдж «Вырезов» (#cutCount) пишет renderCutlist.
function updateTabBadge(id, n) {
  const d = tabDescr(id); if (!d) return
  const btn = document.getElementById(d.tab); if (!btn) return
  if (id === 'meta') {
    const dot = btn.querySelector('.tHas'); if (dot) dot.classList.toggle('hidden', !n)
    return
  }
  const c = btn.querySelector('.tCount'); if (c) c.textContent = n ? String(n) : ''
}

// Данные приехали в НЕактивную вкладку → точка «непрочитано» + конечный пульс
// (1.3s × 2, рестарт через reflow). Активной вкладке сигнал не нужен.
function notifyTab(id) {
  if (id === activeTab) return
  const d = tabDescr(id); if (!d) return
  const btn = document.getElementById(d.tab); if (!btn) return
  const dot = btn.querySelector('.tDot'); if (dot) dot.classList.remove('hidden')
  for (const el of btn.querySelectorAll('.tCount, .tHas, .tDot')) {
    el.classList.remove('fresh'); void el.offsetWidth; el.classList.add('fresh')
  }
}

// Busy-спиннер на вкладке-адресате текущей задачи (вызов из setRunning).
function setTabBusy(name) {
  const busyTab = name ? TASK_TAB[name] : null
  for (const t of TABS) {
    const btn = document.getElementById(t.tab)
    if (btn) btn.classList.toggle('busy', t.id === busyTab)
  }
}

// Вызывается из init() ДО ранних return'ов: таб-бар жив и в пустой сессии.
function bindTabs() {
  // Восстановление fve_tab; однократная миграция со старых ключей аккордеона:
  // развёрнутая панель ('1') становится вкладкой (порядок ACC_PANELS — последняя
  // выигрывает), затем все fve_acc_* удаляются (и '0'-значения тоже).
  let initial = null
  try {
    const valid = new Set(TABS.map((t) => t.id))
    const saved = localStorage.getItem(TAB_KEY)
    if (saved && valid.has(saved)) initial = saved
    else {
      const old = { chaptersPanel: 'chapters', metadataPanel: 'meta', clipsPanel: 'clips' }
      for (const [panel, tab] of Object.entries(old)) {
        if (localStorage.getItem('fve_acc_' + panel) === '1') initial = tab
      }
      for (const panel of Object.keys(old)) localStorage.removeItem('fve_acc_' + panel)
    }
  } catch {}
  setActiveTab(initial || 'cuts')
  updateMetaEmpty()

  for (const t of TABS) {
    const btn = document.getElementById(t.tab)
    if (btn) btn.addEventListener('click', () => setActiveTab(t.id))
  }
  const list = document.querySelector('.paneTabs')
  if (!list) return
  // WAI-ARIA Tabs: roving tabindex; ←/→ — automatic activation (данные уже в
  // st.*, переключение дёшево); Home/End — крайние; 1–4 — прямой переход.
  // stopPropagation обязателен: глобальный keydown иначе словит seek ±5с / play.
  const TAB_HOTKEYS = { 1: 'cuts', 2: 'chapters', 3: 'meta', 4: 'clips' }
  list.addEventListener('keydown', (e) => {
    const k = e.key
    let next = null
    const idx = TABS.findIndex((t) => t.id === activeTab)
    if (k === 'ArrowLeft') next = TABS[(idx + TABS.length - 1) % TABS.length].id
    else if (k === 'ArrowRight') next = TABS[(idx + 1) % TABS.length].id
    else if (k === 'Home') next = TABS[0].id
    else if (k === 'End') next = TABS[TABS.length - 1].id
    else if (TAB_HOTKEYS[k]) next = TAB_HOTKEYS[k]
    else if (k === ' ' || k === 'Enter') {
      // Активировать фокусный таб; глушим, чтобы Space не дошёл до play/pause.
      e.preventDefault(); e.stopPropagation()
      const t = TABS.find((x) => x.tab === (document.activeElement && document.activeElement.id))
      if (t) setActiveTab(t.id)
      return
    } else return
    e.preventDefault(); e.stopPropagation()
    setActiveTab(next)
    const b = document.getElementById(tabDescr(next).tab); if (b) b.focus()
  })
}

// Главы приходят в КООРДИНАТАХ ФИНАЛА; клик → seek(finalToOrig(time)).
function renderChapters(chapters) {
  st.chaptersData = chapters || []
  updateTabBadge('chapters', st.chaptersData.length)
  notifyTab('chapters')   // вкладку НЕ переключаем — пульс + точка, если неактивна
  const box = $('#chaptersList'); if (!box) return
  box.replaceChildren()
  if (!st.chaptersData.length) {
    const ph = document.createElement('div'); ph.className = 'empty placeholder'
    ph.innerHTML = icon('clock') + '<div>Глав пока нет — нажми «Сгенерировать главы»</div>'
    box.appendChild(ph); return
  }
  for (const ch of st.chaptersData) {
    const row = document.createElement('div'); row.className = 'chapter-row'
    const badge = document.createElement('span'); badge.className = 'chTime'; badge.textContent = fmt(ch.time)
    const title = document.createElement('span'); title.className = 'chTitle'; title.textContent = ch.title || 'Глава'
    row.appendChild(badge); row.appendChild(title)
    row.onclick = () => seek(finalToOrig(ch.time))
    box.appendChild(row)
  }
}

/* ---------- B: метаданные YouTube (фоновая задача — LLM) ---------- */
async function loadMetadata() {
  if (st.task) { toast('Дождись завершения задачи перед генерацией метаданных', 'info'); return }
  let res
  try { res = await fetch('/api/preview/metadata', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось запросить метаданные (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить генерацию метаданных'); return }
  let j; try { j = await res.json() } catch { j = {} }
  if (j && j.ok === false && j.reason === 'llm_off') {
    toast('LLM выключена — метаданные недоступны. Включи Ollama.', 'info'); return
  }
  // Задача запущена — следим по SSE (renderMetadata вызовется в followTask).
  followTask('preview_metadata')
}
function renderMetadata(meta) {
  meta = meta || {}
  st.metadataData = meta
  const title = meta.title || ''
  const desc = meta.description || ''
  const tags = (meta.tags || []).join(', ')
  const hook = meta.hook || ''
  const set = (sel, val) => { const el = $(sel); if (el) el.textContent = val }
  set('#metaTitle', title)
  set('#metaDesc', desc)
  set('#metaTags', tags)
  set('#metaHook', hook)
  const len = $('#metaTitleLen')
  if (len) len.textContent = title ? `(${title.length}/100)` : ''
  updateTabBadge('meta', !!(title || desc || tags || hook))
  notifyTab('meta')   // вкладку НЕ переключаем — пульс + точка, если неактивна
  updateMetaEmpty()
}
// Текущее содержимое поля берём из DOM (пользователь мог отредактировать вручную).
function metaFieldText(field) {
  const map = { title: '#metaTitle', desc: '#metaDesc', tags: '#metaTags', hook: '#metaHook' }
  const el = $(map[field]); return el ? (el.textContent || '').trim() : ''
}
// Пустое состояние «Меты»: placeholder, пока генерации не было И поля пусты.
function updateMetaEmpty() {
  const ph = $('#metaEmpty'), fields = $('#metaFields')
  if (!ph || !fields) return
  const empty = st.metadataData == null &&
    !metaFieldText('title') && !metaFieldText('desc') && !metaFieldText('tags') && !metaFieldText('hook')
  ph.classList.toggle('hidden', !empty)
  fields.classList.toggle('hidden', empty)
}
async function copyToClipboard(text) {
  if (!text) { toast('Пусто — нечего копировать', 'info'); return false }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text) }
    else {
      const ta = document.createElement('textarea'); ta.value = text
      ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta)
      ta.select(); document.execCommand('copy'); ta.remove()
    }
    return true
  } catch (e) { toast('Не удалось скопировать (' + e.message + ')', 'error'); return false }
}
async function copyField(field) {
  if (await copyToClipboard(metaFieldText(field))) toast('Скопировано', 'success')
}
async function copyAllMeta() {
  const all =
    'ЗАГОЛОВОК:\n' + metaFieldText('title') +
    '\n\nОПИСАНИЕ:\n' + metaFieldText('desc') +
    '\n\nТЕГИ:\n' + metaFieldText('tags') +
    '\n\nХУК:\n' + metaFieldText('hook')
  if (await copyToClipboard(all)) toast('Скопировано всё', 'success')
}

/* ---------- F4: Clip Maker — кандидаты Shorts (план §4) ---------- */
// Кандидаты живут в ОРИГИНАЛЬНЫХ координатах; eff-длительность считается НА
// ФРОНТЕ через карту финального таймлайна (origToFinal/keptSegments) — сервер
// для этого не нужен, и число живо реагирует на правки вырезов.
function clipEffDur(c) { return Math.max(0, origToFinal(c.end) - origToFinal(c.start)) }

// Пресет рендера клипов (модалка ⚙) — живёт в localStorage (план §4.4).
const CLIPS_OPTS_KEY = 'fve_clips_opts'
function clipsOptsLoad() {
  let o = {}
  try { o = JSON.parse(localStorage.getItem(CLIPS_OPTS_KEY) || '{}') || {} } catch { o = {} }
  return {
    vertical: o.vertical !== false,            // дефолт: вкл — канон Shorts (§1.2)
    center: o.center === 'manual' ? 'manual' : 'auto',
    center_pos: (typeof o.center_pos === 'number' && o.center_pos >= 0 && o.center_pos <= 1) ? o.center_pos : 0.5,
    burn: o.burn !== false,                    // дефолт: вкл — караоке-капшены
    out_dir: typeof o.out_dir === 'string' ? o.out_dir : '',
  }
}
function clipsOptsStore(o) { try { localStorage.setItem(CLIPS_OPTS_KEY, JSON.stringify(o)) } catch {} }

function openClipsModal() {
  const o = clipsOptsLoad()
  $('#cVertical').checked = o.vertical
  $('#cCenterMode').value = o.center
  $('#cCenterRow').classList.toggle('hidden', o.center !== 'manual')
  $('#cCenterPos').value = Math.round(o.center_pos * 100)
  $('#cCenterVal').textContent = Math.round(o.center_pos * 100) + '%'
  $('#cBurn').checked = o.burn
  $('#cOutDir').value = o.out_dir || st.outDir || ''
  openOverlay('#clipsModal')
}
function saveClipsOpts() {
  clipsOptsStore({
    vertical: $('#cVertical').checked,
    center: $('#cCenterMode').value === 'manual' ? 'manual' : 'auto',
    center_pos: Math.max(0, Math.min(100, parseInt($('#cCenterPos').value, 10) || 0)) / 100,
    burn: $('#cBurn').checked,
    out_dir: $('#cOutDir').value.trim(),
  })
  closeOverlay('#clipsModal')
  toast('Настройки рендера клипов сохранены', 'success')
}

// render_opts для POST /api/clips/render (план §2.4): пресет модалки + текущие
// кодек/качество/громкость редактора. chapters/metadata сервер прибивает в
// false и сам, но шлём честно — контракт виден в запросе.
function clipsRenderOpts() {
  const o = clipsOptsLoad()
  const d = st.rdefaults || {}
  const opts = {
    ...currentRenderOpts(),
    subtitles: false, chapters: false, metadata: false,
    vertical: !!o.vertical,
    denoise_loudnorm: !!d.denoise_loudnorm,
    loudnorm_mode: d.loudnorm_mode === '2pass' ? '2pass' : 'dynamic',
    out_dir: o.out_dir || st.outDir || '',
  }
  if (d.cut_fade != null) opts.cut_fade = d.cut_fade
  if (o.vertical) {
    opts.vertical_target = '1080x1920'
    opts.vertical_center = o.center === 'manual' ? o.center_pos : 'auto'
  }
  if (o.burn) { opts.burn_subtitles = true; opts.burn_style = { karaoke: true } }
  else opts.burn_subtitles = false
  return opts
}

// «Предложить клипы» — фоновая задача preview_clips (паттерн loadChapters).
async function loadClips() {
  if (st.task) { toast('Дождись завершения задачи перед подбором клипов', 'info'); return }
  let res
  try { res = await fetch('/api/clips/suggest', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось запросить клипы (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить подбор клипов'); return }
  let j; try { j = await res.json() } catch { j = {} }
  if (j && j.ok === false && j.reason === 'llm_off') {
    toast('LLM выключена — клипы недоступны. Включи Ollama (модель qwen3:8b).', 'info'); return
  }
  followTask('preview_clips')
}

// Восстановление панели при открытии файла: GET /api/clips (кэш clips.json).
// stale (хэш от другого входа) и пустота — молча оставляем placeholder.
async function loadClipsFromCache() {
  let j
  try { const r = await fetch('/api/clips'); if (!r.ok) return; j = await r.json() } catch { return }
  // silent: кэш — не новость; тело + тихий счётчик, БЕЗ пульса/точки/переключения.
  if (j && Array.isArray(j.clips) && j.clips.length && !j.stale) renderClips(j.clips, { silent: true })
}

function renderClips(clips, opts = {}) {
  st.clipsData = (Array.isArray(clips) ? clips.slice() : [])
    .filter((c) => c && typeof c === 'object')
    .sort((a, b) => (b.score || 0) - (a.score || 0))
  updateTabBadge('clips', st.clipsData.length)
  if (!opts.silent) notifyTab('clips')   // вкладку НЕ переключаем
  // Выбор/подсветка/результаты переживают пере-рендер, но не смену набора.
  const ids = new Set(st.clipsData.map((c) => String(c.id)))
  st.clipsSel = new Set([...st.clipsSel].filter((id) => ids.has(id)))
  for (const id of [...st.clipsResults.keys()]) if (!ids.has(id)) st.clipsResults.delete(id)
  if (st.clipsActive && !ids.has(st.clipsActive)) { st.clipsActive = null; clipsHighlight(null) }
  const box = $('#clipsList'); if (!box) return
  box.replaceChildren(); clipEls.clear()
  if (!st.clipsData.length) {
    const ph = document.createElement('div'); ph.className = 'empty placeholder'
    ph.innerHTML = icon('film') + '<div>Достойных кандидатов не нашлось — попробуйте после правки вырезов</div>'
    box.appendChild(ph); updateClipsFoot(); return
  }
  st.clipsData.forEach((c, i) => box.appendChild(clipCard(c, i)))
  updateClipsFoot()
}

// Карточка кандидата (план §4.2, решение №1: ранг #N + скор-полоса, число —
// только в title-tooltip). Все юзер-данные — ТОЛЬКО через textContent/title.
function clipCard(c, i) {
  const id = String(c.id || 'c' + (i + 1))
  const row = document.createElement('div')
  row.className = 'clip-row' + (st.clipsActive === id ? ' sel' : '')
  if (c.reason) row.title = String(c.reason)            // причина от LLM — tooltip
  const score = Math.max(0, Math.min(100, Math.round(c.score || 0)))

  const cb = document.createElement('input')
  cb.type = 'checkbox'; cb.className = 'csel'; cb.checked = st.clipsSel.has(id)
  cb.setAttribute('aria-label', 'Выбрать клип для рендера')
  cb.onclick = (e) => e.stopPropagation()
  cb.onchange = () => { if (cb.checked) st.clipsSel.add(id); else st.clipsSel.delete(id); updateClipsFoot() }

  const rank = document.createElement('div'); rank.className = 'clipRank'
  rank.title = `Скор: ${score}/100`
  const rn = document.createElement('span'); rn.className = 'rankN'; rn.textContent = '#' + (i + 1)
  const bar = document.createElement('div'); bar.className = 'scoreBar'
  const fill = document.createElement('i'); fill.style.width = score + '%'
  bar.appendChild(fill); rank.appendChild(rn); rank.appendChild(bar)

  const meta = document.createElement('div'); meta.className = 'cmeta'
  const hook = document.createElement('div'); hook.className = 'clipHook'
  hook.textContent = c.hook_phrase || 'Без названия'
  const line = document.createElement('div'); line.className = 'tline muted'
  const tc = document.createElement('span'); tc.textContent = `${fmt(c.start)}–${fmt(c.end)}`
  const eff = document.createElement('span'); eff.className = 'clipEff'
  eff.textContent = `~${Math.round(clipEffDur(c))}с`
  eff.title = 'Эффективная длительность — за вычетом внутренних вырезов'
  line.appendChild(tc); line.appendChild(eff)
  if (c.fuzzy_boundary) {
    const b = document.createElement('span'); b.className = 'clipBadge'
    b.textContent = '~граница'; b.title = 'Граница определена неточно — проверьте начало и конец'
    line.appendChild(b)
  }
  if (c.short) {
    const b = document.createElement('span'); b.className = 'clipBadge'
    b.textContent = 'коротковат'; b.title = '15–20 секунд — короче целевых 20–40'
    line.appendChild(b)
  }
  meta.appendChild(hook); meta.appendChild(line)

  const acts = document.createElement('div'); acts.className = 'acts'
  const play = document.createElement('button')
  play.innerHTML = icon('play'); play.title = 'Предпросмотр диапазона'
  play.setAttribute('aria-label', 'Предпросмотр клипа')
  play.onclick = (e) => { e.stopPropagation(); clipsPreview(c, id) }
  acts.appendChild(play)

  row.appendChild(cb); row.appendChild(rank); row.appendChild(meta); row.appendChild(acts)
  const resBox = document.createElement('div'); resBox.className = 'clipResult hidden'
  row.appendChild(resBox)
  clipEls.set(id, { row, eff, res: resBox })
  applyClipResult(id, st.clipsResults.get(id))

  row.onclick = () => clipsSetActive(st.clipsActive === id ? null : id)
  return row
}

// Выбор карточки: рамка + seek на старт + янтарный регион на волне; повторный
// клик/Escape снимает выбор и убирает регион (план §4.2).
function clipsSetActive(id) {
  st.clipsActive = id || null
  st._clipPreviewEnd = null            // смена выбора отменяет ждущую авто-паузу
  for (const [cid, els] of clipEls) els.row.classList.toggle('sel', cid === st.clipsActive)
  const c = st.clipsData.find((x) => String(x.id) === st.clipsActive)
  clipsHighlight(c || null)
  if (c) seek(c.start)
}
function clipsHighlight(c) {
  if (!regions) return
  st.addingRegion = true
  if (clipHlRegion) { try { clipHlRegion.remove() } catch {} clipHlRegion = null }
  if (c) clipHlRegion = regions.addRegion({ id: 'clip-hl', start: c.start, end: c.end, color: CLIP_HL_COLOR, drag: false, resize: false })
  st.addingRegion = false
}

// ▶ — предпросмотр диапазона: seek(start), play с пропуском внутренних вырезов,
// авто-пауза на end (срабатывает в onFrame). Прежний выбор «пропускать вырезы»
// восстанавливается после авто-паузы.
function clipsPreview(c, id) {
  clipsSetActive(id)
  st._clipPrevPreview = st.preview
  st.preview = true
  const skip = $('#skipCuts'); if (skip) skip.checked = true
  st._clipPreviewEnd = c.end
  seek(c.start)
  video.play()
}

function updateClipsFoot() {
  const n = st.clipsSel.size
  const cnt = $('#clipsSelCount'); if (cnt) cnt.textContent = String(n)
  const foot = $('#clipsFoot'); if (foot) foot.classList.toggle('hidden', n === 0)
}
// Вырезы изменились → eff-длительности карточек пересчитать (вызов из refreshCuts).
function updateClipsEff() {
  for (const c of st.clipsData) {
    const els = clipEls.get(String(c.id)); if (!els) continue
    els.eff.textContent = `~${Math.round(clipEffDur(c))}с`
  }
}

// «Рендерить выбранные (N)» → одна фоновая задача render_clips (план §2.4).
async function clipsRender() {
  if (st.task) { toast('Дождись завершения текущей задачи', 'info'); return }
  const chosen = st.clipsData.filter((c) => st.clipsSel.has(String(c.id)))
  if (!chosen.length) { toast('Выбери хотя бы один клип — чекбокс на карточке', 'info'); return }
  await save()   // сервер режет по своему live-катлисту — зафиксировать правки
  const stem = ($('#filename').textContent || 'clip').replace(/\.[^.]+$/, '')
  const body = {
    clips: chosen.map((c, i) => ({
      start: c.start, end: c.end,
      filename: `${stem}_clip${String(i + 1).padStart(2, '0')}`,
    })),
    render_opts: clipsRenderOpts(),
  }
  st._clipsRenderIds = chosen.map((c) => String(c.id))
  let res
  try { res = await fetch('/api/clips/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) }
  catch (e) { toast('Сеть: не удалось запустить рендер клипов (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить рендер клипов'); return }
  followTask('render_clips')
}

// Результаты render_clips → на карточки: ссылка на mp4 или текст ошибки.
// Порядок results совпадает с порядком clips в POST (st._clipsRenderIds).
function applyClipRenderResults(results) {
  const ids = st._clipsRenderIds || []
  ;(results || []).forEach((r, i) => {
    const id = ids[i]; if (!id) return
    st.clipsResults.set(id, r)
    applyClipResult(id, r)
  })
}
function applyClipResult(id, r) {
  const els = clipEls.get(id); if (!els || !r) return
  const box = els.res
  box.classList.remove('hidden', 'error')
  box.replaceChildren()
  if (r.ok) {
    const base = (typeof r.mp4 === 'string' && r.mp4) ? r.mp4.split(/[\\/]/).pop()
      : ((r.filename || 'clip') + '.mp4')
    const a = document.createElement('a')
    a.href = '/api/output/' + encodeURIComponent(base)
    a.target = '_blank'
    a.textContent = 'Открыть ' + base
    a.onclick = (e) => e.stopPropagation()
    box.appendChild(a)
  } else {
    box.classList.add('error')
    box.textContent = 'Ошибка: ' + (r.error || 'неизвестная')
  }
}

function refreshCuts() {
  // Always recompute from current segs: callers mutate st.segs then call us
  // BEFORE markDirty() invalidates the memo, so a stale st._merged would make
  // the «Итог» total and struck-through words lag one edit behind. refreshCuts
  // is never called per playback frame (preview uses insideRemoved directly),
  // so clearing here costs nothing and keeps the memo correct.
  st._merged = null; st._kept = null   // финальная карта зависит от вырезов — тоже сбрасываем
  const rem = mergedRemoves()
  const spans = st.spans || (st.spans = $('#transcript').querySelectorAll('.w'))
  for (const sp of spans) {
    const w = st.words[+sp.dataset.i]; const mid = (w.start + w.end) / 2
    // intervals sorted -> linear ok, but reuse binary insideRemoved for the midpoint
    sp.classList.toggle('cut', st.showCuts && insideRemoved(mid) != null)
  }
  updateKept()
  updateClipsEff()        // F4: eff-длительность карточек клипов зависит от вырезов
  scheduleAfterRedraw()   // вырезы изменились → перерисовать «После»-полосу (debounced)
}
function updateKept() {
  const removed = mergedRemoves().reduce((a, [s, e]) => a + (e - s), 0)
  const pct = st.duration ? Math.round(100 * removed / st.duration) : 0
  $('#kept').textContent = `Итог: ${fmt(st.duration - removed)} (−${pct}%)`
}

/* ---------- cut list (AI suggestions) ---------- */
function renderCutlist() {
  const box = $('#cutlist')
  const segs = [...st.segs].sort((a, b) => a.start - b.start)
  // Бейдж в табе «Вырезы»: «N/M» (+ data-short для узкой колонки, см. CSS).
  const en = segs.filter((s) => s.enabled).length
  const cc = $('#cutCount')
  cc.textContent = `${en}/${segs.length}`
  cc.dataset.short = String(en)
  cc.title = `включено ${en} из ${segs.length}`
  const tabBtn = document.getElementById('tab-cuts')
  if (tabBtn) tabBtn.setAttribute('aria-label', `Вырезы — включено ${en} из ${segs.length}`)
  box.replaceChildren()
  if (!segs.length) {
    const ph = document.createElement('div'); ph.className = 'empty placeholder'
    ph.innerHTML = icon('scissors') + '<div>Вырезов пока нет — выдели текст или нажми «Передетектировать»</div>'
    box.appendChild(ph); return
  }
  for (const seg of segs) {
    const rgb = COLORS[seg.type] || '120,120,120'
    const row = document.createElement('div')
    row.className = 'cut-row' + (seg.enabled ? '' : ' off') + (seg.id === st.selected ? ' sel' : '')
    row.style.borderLeftColor = `rgb(${rgb})`
    row.innerHTML = `
      <input type="checkbox" class="en" ${seg.enabled ? 'checked' : ''}>
      <div class="meta">
        <div class="tline"><span class="badge" style="background:rgba(${rgb},.15);color:rgb(${rgb})">${TYPE_RU[seg.type] || seg.type}/${seg.action === 'censor' ? 'цензура' : 'вырезать'}</span>
          <span class="muted">${fmt(seg.start)}–${fmt(seg.end)} (${(seg.end - seg.start).toFixed(2)}s)</span></div>
        <div class="ttext">${(seg.reason || seg.text || '').replace(/</g, '&lt;')}</div>
      </div>
      <div class="acts">
        <button class="jump" title="перейти" aria-label="Перейти к вырезу">${icon('arrow-right')}</button>
        <button class="act" title="вырезать/цензура" aria-label="Вкл/выкл вырез">${icon('swap')}</button>
        <button class="del" title="удалить" aria-label="Удалить вырез">${icon('backspace')}</button>
      </div>`
    const cb = row.querySelector('.en')
    cb.onclick = (e) => e.stopPropagation()   // don't let the click bubble to row (would re-render before 'change')
    cb.onchange = () => toggleEnabled(seg.id)
    row.querySelector('.jump').onclick = (e) => { e.stopPropagation(); select(seg.id); seek(seg.start) }
    row.querySelector('.act').onclick = (e) => { e.stopPropagation(); cycleAction(seg.id) }
    row.querySelector('.del').onclick = (e) => { e.stopPropagation(); deleteSeg(seg.id) }
    row.onclick = () => { select(seg.id); seek(seg.start) }
    box.appendChild(row)
  }
}

/* ---------- undo / redo ---------- */
function snapshot() { return JSON.parse(JSON.stringify(st.segs)) }
function pushUndo() {
  undoStack.push(snapshot())
  if (undoStack.length > UNDO_CAP) undoStack.shift()
  redoStack.length = 0   // new action invalidates redo
}
function restoreSegs(segs) {
  st.segs = segs
  st.manualN = st.segs.filter((s) => s.type === 'manual').length
  if (st.selected && !segOf(st.selected)) st.selected = null
  renderRegions(); renderCutlist(); refreshCuts(); markDirty()
}
function undo() {
  if (!undoStack.length) { flash('Нечего отменять'); return }
  redoStack.push(snapshot())
  restoreSegs(undoStack.pop())
  flash('Отменено')
}
function redo() {
  if (!redoStack.length) { flash('Нечего вернуть'); return }
  undoStack.push(snapshot())
  restoreSegs(redoStack.pop())
  flash('Возвращено')
}

/* ---------- mutations ---------- */
function toggleEnabled(id) { const s = segOf(id); if (!s) return; pushUndo(); s.enabled = !s.enabled; refreshRegionColor(id); renderCutlist(); refreshCuts(); markDirty() }
function cycleAction(id) { const s = segOf(id); if (!s) return; pushUndo(); s.action = s.action === 'remove' ? 'censor' : 'remove'; renderCutlist(); refreshCuts(); markDirty() }
function deleteSeg(id) {
  if (!segOf(id)) return
  pushUndo()
  st.segs = st.segs.filter((s) => s.id !== id)
  const r = regionById.get(id); if (r) { st.addingRegion = true; r.remove(); st.addingRegion = false; regionById.delete(id) }
  if (st.selected === id) st.selected = null
  renderCutlist(); refreshCuts(); markDirty()
}
function select(id) { st.selected = id; renderCutlist() }
function seek(t) { video.currentTime = Math.max(0, Math.min(t, st.duration - 0.04)); onFrame() }

function addManual(start, end) {
  if (end - start < 0.02) return
  pushUndo()
  const id = 'man' + Date.now().toString(36) + (st.manualN++)
  const seg = { id, start: round(start), end: round(end), type: 'manual', action: 'remove', enabled: true, text: 'ручной вырез', reason: '', word: '' }
  st.segs.push(seg)
  if (st.showCuts) { st.addingRegion = true; const r = regions.addRegion({ id, start: seg.start, end: seg.end, color: colorOf(seg), drag: true, resize: true }); st.addingRegion = false; regionById.set(id, r) }
  renderCutlist(); refreshCuts(); select(id); markDirty()
}

/* ---------- selection -> cut ---------- */
function onSelectionChange() {
  const float = $('#cutFloat'); const sel = window.getSelection()
  // Во время инлайн-правки слова выделение внутри спана — это правка текста,
  // а не «выделил → вырезать»: плашку ✂ не показываем, selRange не трогаем.
  if (editingSpan) { float.classList.add('hidden'); st.selRange = null; return }
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) { float.classList.add('hidden'); st.selRange = null; return }
  const range = sel.getRangeAt(0)
  if (!$('#transcript').contains(range.commonAncestorContainer)) { float.classList.add('hidden'); st.selRange = null; return }
  let lo = Infinity, hi = -1
  const spans = st.spans || (st.spans = $('#transcript').querySelectorAll('.w'))
  for (const sp of spans) if (range.intersectsNode(sp)) { const i = +sp.dataset.i; if (i < lo) lo = i; if (i > hi) hi = i }
  if (hi < 0) { float.classList.add('hidden'); st.selRange = null; return }
  st.selRange = [lo, hi]
  const r = range.getBoundingClientRect()
  float.style.left = (r.left + r.width / 2 - 45) + 'px'; float.style.top = (r.top - 42) + 'px'
  float.classList.remove('hidden')
}
function cutSelection() {
  if (!st.selRange) return
  addManual(st.words[st.selRange[0]].start, st.words[st.selRange[1]].end)
  window.getSelection().removeAllRanges(); $('#cutFloat').classList.add('hidden'); st.selRange = null
}
function cutFromMarks() { if (st.inP == null || st.outP == null) { flash('Поставь метки I и O'); return } addManual(Math.min(st.inP, st.outP), Math.max(st.inP, st.outP)); st.inP = st.outP = null }

/* ---------- редактируемый транскрипт: двойной клик по слову ---------- */
// contenteditable включается ТОЛЬКО на время правки одного спана .w.
// Enter / blur = сохранить (PUT /api/transcript/word, откат + тост при ошибке),
// Esc = отмена. Правка меняет только текст — тайминги и вырезы не трогаются.
let editingSpan = null   // спан .w в режиме правки (null, если правки нет)

function startWordEdit(sp) {
  if (editingSpan === sp) return
  if (editingSpan) cancelWordEdit(editingSpan)        // одна правка за раз
  const w = st.words[+sp.dataset.i]; if (!w) return
  editingSpan = sp
  sp.dataset.orig = (w.word || '').trim()
  sp.textContent = sp.dataset.orig                    // без хвостового пробела на время правки
  sp.setAttribute('contenteditable', 'true')
  sp.setAttribute('spellcheck', 'false')
  sp.classList.add('editing')
  sp.addEventListener('keydown', onWordEditKey)
  sp.addEventListener('blur', onWordEditBlur)
  sp.focus()
  const r = document.createRange(); r.selectNodeContents(sp)
  const sel = window.getSelection(); if (sel) { sel.removeAllRanges(); sel.addRange(r) }
}

// Снять режим правки с DOM (слушатели/атрибуты/класс) — текст НЕ трогаем.
function endWordEdit(sp) {
  sp.removeEventListener('keydown', onWordEditKey)
  sp.removeEventListener('blur', onWordEditBlur)
  sp.removeAttribute('contenteditable')
  sp.removeAttribute('spellcheck')
  sp.classList.remove('editing')
  delete sp.dataset.orig
  editingSpan = null
  const sel = window.getSelection(); if (sel) sel.removeAllRanges()
  try { sp.blur() } catch {}
}

async function commitWordEdit(sp) {
  const w = st.words[+sp.dataset.i]
  const orig = sp.dataset.orig != null ? sp.dataset.orig : ''
  const next = (sp.textContent || '').replace(/\s+/g, ' ').trim()
  endWordEdit(sp)
  const restore = () => { sp.textContent = orig + ' ' }
  if (!w) { restore(); return }
  if (!next || next === orig) { restore(); return }   // пусто или без изменений — тихий откат
  if (next.length > 200) { restore(); toast('Слишком длинный текст слова — максимум 200 символов', 'error'); return }
  sp.textContent = next + ' '                          // оптимистично; при ошибке откатим
  let res
  try {
    res = await fetch('/api/transcript/word', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ si: w.si, wi: w.wi, text: next }),
    })
  } catch (e) { restore(); toast('Сеть: не удалось сохранить слово (' + e.message + ')', 'error'); return }
  if (!res.ok) {
    restore()
    let detail = 'HTTP ' + res.status
    try { const j = await res.json(); if (j && j.detail) detail = j.detail } catch {}
    toast('Не удалось сохранить слово: ' + detail, 'error')
    return
  }
  let j = {}; try { j = await res.json() } catch {}
  // Сервер вернул слово в конвенции Whisper (с ведущим пробелом, если был).
  w.word = (j && typeof j.text === 'string') ? j.text : (' ' + next)
  w.edited = true
  sp.classList.add('edited'); sp.title = 'изменено'
  flash('Слово сохранено')
}

function cancelWordEdit(sp) {
  const orig = sp.dataset.orig != null ? sp.dataset.orig : ''
  endWordEdit(sp)
  sp.textContent = orig + ' '
}

function onWordEditKey(e) {
  e.stopPropagation()   // Enter/Del/Space и пр. не должны дойти до глобальных хоткеев
  if (e.key === 'Enter') { e.preventDefault(); commitWordEdit(e.currentTarget) }
  else if (e.key === 'Escape') { e.preventDefault(); cancelWordEdit(e.currentTarget) }
}
function onWordEditBlur(e) { commitWordEdit(e.currentTarget) }

/* ---------- playback ---------- */
function loop() { onFrame(); raf = requestAnimationFrame(loop) }
function onFrame() {
  const t = video.currentTime
  if (st.preview && !video.paused) { const iv = insideRemoved(t); if (iv) { video.currentTime = iv[1] + 0.02; return } }
  // F4: ▶-предпросмотр клипа — авто-пауза на конце диапазона (план §4.2);
  // вернуть прежний выбор «пропускать вырезы» (кроме режима «После» — он сам
  // держит пропуск включённым).
  if (st._clipPreviewEnd != null && t >= st._clipPreviewEnd - 0.03) {
    st._clipPreviewEnd = null
    video.pause()
    if (!st.afterMode) {
      st.preview = !!st._clipPrevPreview
      const sk = $('#skipCuts'); if (sk) sk.checked = st.preview
    }
  }
  $('#tcCur').textContent = fmtcs(t)
  updatePlayhead(); highlight(t + st.syncOffset)
  if (st.afterMode) {
    updateAfterPlayhead()
    $('#tcFinal').textContent = fmt(origToFinal(t)) + ' / ' + fmt(st._newDuration)
  }
  if (st.subsMode) updateSubCaption()
}
function saveSync() {
  localStorage.setItem('fve_sync', String(st.syncOffset))
  flash(`сдвиг подсветки: ${st.syncOffset > 0 ? '+' : ''}${Math.round(st.syncOffset * 1000)} мс  (клавиши ; и ')`)
  st.activeWord = -1; highlight(video.currentTime + st.syncOffset)
}
function refreshPhMetrics() {
  const wf = $('#waveform'); if (!wf) return
  phMetrics.client = wf.clientWidth
  phMetrics.total = wrapper ? wrapper.scrollWidth : wf.clientWidth
}
function onWaveScroll() {
  if (phScrollRaf) return
  phScrollRaf = requestAnimationFrame(() => { phScrollRaf = 0; updatePlayhead() })
}
function updatePlayhead() {
  const t = video.currentTime, lbl = $('#phLabel')
  if (!lbl) return
  if (!phMetrics.client) refreshPhMetrics()
  const total = phMetrics.total || phMetrics.client
  const sl = wrapper ? wrapper.scrollLeft : 0
  const x = (st.duration ? t / st.duration : 0) * total - sl
  lbl.style.left = Math.max(8, Math.min(phMetrics.client - 8, x)) + 'px'
  lbl.textContent = fmtcs(t)
}
function highlight(t) {
  let lo = 0, hi = st.words.length - 1, idx = -1
  while (lo <= hi) { const m = (lo + hi) >> 1, w = st.words[m]; if (t < w.start) hi = m - 1; else if (t >= w.end) lo = m + 1; else { idx = m; break } }
  if (idx === st.activeWord) return
  const spans = st.spans || (st.spans = $('#transcript').querySelectorAll('.w'))
  if (st.activeWord >= 0 && spans[st.activeWord]) spans[st.activeWord].classList.remove('active')
  st.activeWord = idx
  if (idx >= 0 && spans[idx]) { spans[idx].classList.add('active'); spans[idx].scrollIntoView({ block: 'nearest' }) }
}

/* ---------- nav between cuts ---------- */
const boundaries = () => { const b = new Set(); for (const s of st.segs) if (s.enabled) { b.add(round(s.start)); b.add(round(s.end)) } return [...b].sort((x, y) => x - y) }
function prevCut() { const b = boundaries().filter((x) => x < video.currentTime - 0.06); if (b.length) seek(b[b.length - 1]) }
function nextCut() { const b = boundaries().filter((x) => x > video.currentTime + 0.06); if (b.length) seek(b[0]) }

/* ---------- zoom ---------- */
function zoomTo(val) {
  if (!st.fitPx) st.fitPx = $('#waveform').clientWidth / Math.max(1, st.duration)
  try { ws.zoom(st.fitPx * (1 + (val / 100) * 5)) } catch (e) {}
  // zoom changes scrollWidth; refresh metrics + reposition after layout settles
  requestAnimationFrame(() => { refreshPhMetrics(); updatePlayhead() })
}

/* ---------- save ---------- */
function setSavePill(state, text) {
  const p = $('#savePill'); if (!p) return
  p.classList.remove('hidden', 'dirty', 'saving', 'saved', 'error')
  p.classList.add(state); p.textContent = text
}
function markDirty() {
  // Set dirty SYNCHRONOUSLY so beforeunload sees it even mid-mutation.
  st.dirty = true; st._merged = null; st._kept = null  // invalidate memoized merged intervals + kept map
  if (!st.saving && !st.saveError) setSavePill('dirty', 'Изменено')
  clearTimeout(saveTimer); saveTimer = setTimeout(save, 700)
}

// Serialized, DRAINING save: one PUT at a time, and the returned promise resolves
// only when the cutlist is FULLY flushed. An edit made while a PUT is in flight
// re-dirties the state and is sent in the SAME chain — so `await save()` before
// render / detect / export / queue can never miss it (it used to: the old code
// resolved after the first PUT and flushed the pending edit on a detached timer).
function save() {
  clearTimeout(saveTimer)
  if (st.saving) return st.savingPromise   // the running doSave loops until clean
  st.savingPromise = doSave()
  return st.savingPromise
}
async function doSave() {
  st.saving = true; setSavePill('saving', 'Сохранение…')
  try {
    while (st.dirty && !st._navigating) {
      // Claim the current state up front: a concurrent edit during the await
      // re-sets st.dirty (and st.segs), so the loop sends the newest state next.
      st.dirty = false
      const body = JSON.stringify({ version: 1, source: '', duration: st.duration, segments: st.segs })
      let res
      try {
        res = await fetch('/api/cutlist', {
          method: 'PUT', headers: { 'Content-Type': 'application/json' }, body })
      } catch (e) { st.dirty = true; throw e }            // network: restore dirty
      if (!res.ok) {
        st.dirty = true                                   // unsaved -> keep dirty
        let detail = 'HTTP ' + res.status
        try { const j = await res.json(); if (j && j.detail) detail = j.detail } catch {}
        throw new Error(detail)
      }
    }
    st.saveError = false; clearSaveError()
    if (!st._navigating) {
      flash('Сохранено ' + new Date().toLocaleTimeString())
      setSavePill('saved', 'Сохранено')
      if (st.subsMode) loadSubtitles()   // правки сохранены → обновить реплики
    }
  } catch (e) {
    st.saveError = true   // keep dirty
    setSavePill('error', 'Не сохранено')
    showSaveError('Не удалось сохранить правки: ' + e.message + '. Изменения не потеряны — повтор автоматически.')
    clearTimeout(saveTimer); saveTimer = setTimeout(save, 2000)   // auto-retry
  } finally {
    st.saving = false; st.savingPromise = null
  }
}

/* ---------- tasks ---------- */
function setRunning(name) {
  st.task = name || null
  const busy = !!name
  for (const id of ['#btnTranscribe', '#btnRedetect', '#btnRender', '#btnChaptersToggle', '#btnMetaGenerate', '#btnClipsSuggest', '#btnClipsRender']) { const b = $(id); if (b) b.disabled = busy }
  setTabBusy(st.task)   // busy-спиннер на табе-адресате (вкладки НЕ дизейблим)
  const cancel = $('#btnCancelTask'); if (cancel) cancel.classList.toggle('hidden', !busy)
}
async function cancelTask() {
  if (!st.task) return
  toast('Отменяю…')
  try { await fetch('/api/cancel', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось отменить (' + e.message + ')', 'error') }
}
let taskStart = 0
function followTask(name) {
  setRunning(name); taskStart = Date.now()
  $('#progress').classList.remove('hidden'); $('#progress').classList.add('indeterminate')
  setProgress(35, `${name}…`)
  if (es) es.close()
  es = new EventSource('/api/events')
  es.onerror = () => { /* let EventSource auto-retry; final state arrives via message */ }
  es.onmessage = (ev) => {
    let t; try { t = JSON.parse(ev.data) } catch { return }
    const pct = Math.max(0, Math.min(100, t.percent || 0))
    if (pct > 0) {
      $('#progress').classList.remove('indeterminate')
      setProgress(pct)
    }
    // Simple ETA from percent over wall-clock.
    let eta = ''
    if (pct >= 3 && pct < 100) {
      const elapsed = (Date.now() - taskStart) / 1000
      const remain = elapsed * (100 - pct) / pct
      // секунды при остатке <60с, иначе минуты — не «~1м» для 5 секунд
      if (remain >= 60) eta = `  · осталось ~${Math.round(remain / 60)}м`
      else if (remain > 1) eta = `  · осталось ~${Math.max(1, Math.round(remain))}с`
    }
    setProgress(pct, `${t.stage || t.name || name || ''} ${Math.round(pct)}%${eta}`)
    if (!t.running) {
      es.close(); es = null; $('#progress').classList.add('hidden'); $('#progress').classList.remove('indeterminate')
      setRunning(null)
      if (t.error) {
        // A cancelled LLM chapters task may have finished anyway — keep results.
        if (t.name === 'preview_chapters' && t.results && (t.results.chapters || []).length) renderChapters(t.results.chapters)
        if (t.name === 'preview_metadata' && t.results && t.results.metadata) renderMetadata(t.results.metadata)
        // F4: отменённый render_clips сохраняет частичные результаты — показать
        // готовые клипы/ошибки на карточках (план §2.4: «остальные не теряются»).
        if (t.name === 'render_clips' && t.results && t.results.clips) applyClipRenderResults(t.results.clips)
        // Ошибка = существующий тост + точка на вкладке-адресате задачи.
        const errTab = TASK_TAB[t.name]; if (errTab) notifyTab(errTab)
        if (t.cancelled || t.error === 'cancelled') toast('Задача отменена', 'info')
        else toast('Ошибка задачи: ' + t.error, 'error')
        return
      }
      if (t.name === 'transcribe') { $('#btnTranscribe').classList.add('hidden'); toast('Транскрипция готова', 'success'); loadData(); notifyTab('cuts') }
      if (t.name === 'detect') { reloadCutlist(); notifyTab('cuts') }
      if (t.name === 'render' && t.results) { toast('Рендер завершён', 'success'); showResults(t.results) }
      if (t.name === 'preview_chapters' && t.results) { renderChapters(t.results.chapters || []); toast('Главы готовы', 'success') }
      if (t.name === 'preview_metadata' && t.results && t.results.metadata) { renderMetadata(t.results.metadata); toast('Метаданные готовы', 'success') }
      if (t.name === 'preview_clips' && t.results) {
        const cl = t.results.clips || []
        renderClips(cl)
        toast(cl.length ? `Найдено кандидатов: ${cl.length}` : 'Достойных кандидатов не нашлось', cl.length ? 'success' : 'info')
      }
      if (t.name === 'render_clips' && t.results && t.results.clips) {
        const rs = t.results.clips
        applyClipRenderResults(rs)
        const ok = rs.filter((r) => r && r.ok).length
        toast(`Готово: ${ok}/${rs.length} клипов`, ok === rs.length ? 'success' : 'error')
      }
    }
  }
}

// Parse {detail} from a non-ok response and toast it; return false so callers can bail.
async function failToast(res, prefix) {
  let detail = 'HTTP ' + res.status
  try { const j = await res.json(); if (j && j.detail) detail = (typeof j.detail === 'string') ? j.detail : JSON.stringify(j.detail) } catch {}
  toast(prefix + ': ' + detail, 'error')
  return false
}

async function transcribe() {
  if (st.task) return
  // A6: первая транскрипция тянет модель Whisper из сети (до ~3 ГБ) — честно
  // предупредить ДО старта. Ошибка/недоступность /api/models НЕ блокирует:
  // ведём себя как раньше (модель докачается молча в ходе задачи).
  try {
    const r = await fetch('/api/models')
    if (r.ok) {
      const j = await r.json()
      const w = (j && j.whisper) || {}
      const p = (w.presets || []).find((x) => x.model === w.current)
      if (p && !p.cached) {
        const size = (p.download_gb != null) ? ` (~${String(p.download_gb).replace('.', ',')} ГБ)` : ''
        if (!confirm(`Модель Whisper «${p.label || p.model}» ещё не скачана${size}. Скачать сейчас? Это однократно — дальше работает офлайн.`)) return
      }
    }
  } catch {}
  // Старт действия — единственный разрешённый auto-switch (§3.3): прогресс и
  // адрес результата видны сразу.
  setActiveTab('cuts')
  let res
  try { res = await fetch('/api/transcribe', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось запустить транскрипцию (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить транскрипцию'); return }
  followTask('transcribe')
}

async function redetect() {
  if (st.task) return
  if (!confirm('Перестроить вырезы из транскрипта? Ручные вырезы сохранятся, остальные правки сбросятся.')) return
  setActiveTab('cuts')   // старт действия — единственный разрешённый auto-switch (§3.3)
  // Flush pending edits first: the server rebuilds from its OWN cutlist and
  // re-appends manual cuts from it, so an unsaved manual cut (added within the
  // 700ms autosave window) would otherwise be lost. Mirrors startRender().
  await save()
  let res
  try { res = await fetch('/api/detect', { method: 'POST' }) }
  catch (e) { toast('Сеть: не удалось запустить детекцию (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить детекцию'); return }
  // /api/detect is now a background task; follow it over SSE and reload cutlist on done.
  followTask('detect')
}

async function reloadCutlist() {
  let cl
  try { const r = await fetch('/api/cutlist'); if (!r.ok) throw new Error('HTTP ' + r.status); cl = await r.json() }
  catch (e) { toast('Не удалось загрузить вырезы: ' + e.message, 'error'); return }
  pushUndo()
  st.segs = cl.segments || []; st.manualN = st.segs.filter((s) => s.type === 'manual').length
  st._merged = null; st._kept = null
  renderRegions(); renderCutlist(); refreshCuts()
  if (st.subsMode) loadSubtitles()   // вырезы изменились → финальные координаты реплик тоже
  toast('Детекция обновлена', 'success')
}
// Encoder-appropriate default quality + label (NVENC uses QP, x264 uses CRF).
const QUAL_DEFAULT = { nvenc: 19, x264: 17 }
function qualLabel(encoder) { return encoder === 'x264' ? 'CRF' : 'QP' }
function seedQuality(encoder, q) {
  const val = (q != null) ? q : (QUAL_DEFAULT[encoder] != null ? QUAL_DEFAULT[encoder] : 19)
  $('#rQuality').value = val
  $('#rQualVal').textContent = `${qualLabel(encoder)} ${val}`
}

function openRenderModal() {
  if (!st.hasSession) return
  const d = st.rdefaults || {}, m = st.media || {}
  const enc = d.encoder || 'nvenc'
  $('#rEncoder').value = enc
  seedQuality(enc, (d.quality != null) ? d.quality : null)
  $('#rAudio').value = d.audio_bitrate || '320k'
  $('#rCensor').value = d.censor_method || 'partial'
  $('#rScale').options[0].textContent = `Как у источника (${m.width || '?'}×${m.height || '?'})`
  $('#rFps').options[0].textContent = `Как у источника (${m.fps ? Math.round(m.fps) : '?'})`
  $('#rScale').value = ''; $('#rFps').value = ''
  $('#rSubs').checked = true; $('#rChapters').checked = true
  if ($('#rBurn')) {
    $('#rBurn').checked = false
    $('#rBurnOpts').classList.add('hidden')
    $('#rBurnSizeVal').textContent = `${$('#rBurnSize').value}px`
  }
  if ($('#rVertical')) $('#rVertical').checked = false
  if ($('#rDenoise')) {
    $('#rDenoise').checked = !!d.denoise
    $('#rDenoiseOpts').classList.toggle('hidden', !d.denoise)
    const nf = (d.denoise_strength != null) ? Math.round(d.denoise_strength) : -25
    $('#rDenoiseStrength').value = nf
    $('#rDenoiseStrengthVal').textContent = denoiseStrengthLabel(nf)
    $('#rDenoiseNorm').checked = !!d.denoise_normalize
    // Движок шумоподавления: whitelist на сервере — сюда попадает только
    // "afftdn" | "deepfilter"; всё прочее показываем как стандартный.
    if ($('#rDenoiseEngine')) {
      $('#rDenoiseEngine').value = (d.denoise_engine === 'deepfilter') ? 'deepfilter' : 'afftdn'
    }
  }
  // Мастеринг звука: независимы от шумоподавления (работают и без него).
  if ($('#rDeess')) $('#rDeess').checked = !!d.denoise_deess
  if ($('#rLoudnorm')) $('#rLoudnorm').checked = !!d.denoise_loudnorm
  // Точный (двухпроходный) loudnorm — под-опция «Громкости под YouTube»:
  // активна только вместе с родительским чекбоксом (сервер всё равно
  // игнорирует режим при выключенном loudnorm — это чисто UX-подсказка).
  if ($('#rLoudnorm2p')) {
    $('#rLoudnorm2p').checked = (d.loudnorm_mode === '2pass')
    $('#rLoudnorm2p').disabled = !($('#rLoudnorm') && $('#rLoudnorm').checked)
  }
  if ($('#rCutFade')) {
    const ms = (d.cut_fade != null) ? Math.round(d.cut_fade * 1000) : 15
    $('#rCutFade').value = ms
    setCutFadeLabel(ms)
  }
  // Сбросить прошлый результат экспорта в NLE — иначе показывает устаревшую
  // ссылку на файл от прежнего набора вырезов (риск отдать не тот таймлайн).
  const nle = $('#nleResult'); if (nle) { nle.classList.add('hidden'); nle.classList.remove('error'); nle.textContent = '' }
  // Экспорт в монтажку требует хотя бы один вырез — иначе будет 409.
  const exBtn = $('#btnExportNle')
  if (exBtn) {
    const hasCuts = (st.segs || []).some((s) => s.enabled && s.action === 'remove')
    exBtn.disabled = !hasCuts
    exBtn.title = hasCuts ? '' : 'Сначала сделай хотя бы один вырез'
  }
  $('#rOutDir').value = st.outDir || ''
  $('#rFilename').value = ($('#filename').textContent || 'output').replace(/\.[^.]+$/, '')
  openOverlay('#renderModal')
}

// ASS PrimaryColour/etc. are &HAABBGGRR (alpha-blue-green-red, AA=00 opaque).
// The <input type=color> gives us #RRGGBB — flip the byte order to BBGGRR.
function hexToAss(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim())
  if (!m) return null
  const rr = m[1].slice(0, 2), gg = m[1].slice(2, 4), bb = m[1].slice(4, 6)
  return ('&H00' + bb + gg + rr).toUpperCase()
}

// Human label for the afftdn noise-floor slider (-12 soft … -40 strong).
function denoiseStrengthLabel(nf) {
  const v = parseInt(nf, 10)
  const tag = v >= -18 ? 'щадящее' : (v <= -32 ? 'сильное' : 'среднее')
  return `${v} dB (${tag})`
}

function setCutFadeLabel(ms) {
  const v = parseInt(ms, 10)
  const el = $('#rCutFadeVal'); if (!el) return
  el.textContent = v === 0 ? '0 мс (жёстко)' : `${v} мс` + (v === 15 ? ' (реком.)' : '')
}

// Collect the burn-in subtitle opts from the render modal (or null if off).
function burnOptsFromUI() {
  if (!$('#rBurn') || !$('#rBurn').checked) return null
  const style = {
    font: $('#rBurnFont').value,
    size: parseInt($('#rBurnSize').value, 10),
    position: $('#rBurnPos').value,
    karaoke: $('#rBurnKaraoke').checked,
  }
  const pc = hexToAss($('#rBurnColor').value); if (pc) style.primary_color = pc
  const kc = hexToAss($('#rBurnKColor').value); if (kc) style.karaoke_color = kc
  return { burn_subtitles: true, burn_style: style }
}

function submitRender() {
  const opts = {
    encoder: $('#rEncoder').value,
    quality: parseInt($('#rQuality').value, 10),
    scale_h: $('#rScale').value ? parseInt($('#rScale').value, 10) : null,
    fps: $('#rFps').value ? parseFloat($('#rFps').value) : null,
    audio_bitrate: $('#rAudio').value,
    censor_method: $('#rCensor').value,
    subtitles: $('#rSubs').checked,
    chapters: $('#rChapters').checked,
    vertical: !!($('#rVertical') && $('#rVertical').checked),
    denoise: !!($('#rDenoise') && $('#rDenoise').checked),
    // Мастеринг — отдельные флаги, шлются всегда (работают и без шумоподавления).
    denoise_deess: !!($('#rDeess') && $('#rDeess').checked),
    denoise_loudnorm: !!($('#rLoudnorm') && $('#rLoudnorm').checked),
    // Режим loudnorm: "2pass" = точный (измерительный пасс + linear);
    // "dynamic" = прежний однопроходный. Сервер принимает только эти два.
    loudnorm_mode: ($('#rLoudnorm2p') && $('#rLoudnorm2p').checked) ? '2pass' : 'dynamic',
    cut_fade: $('#rCutFade') ? parseInt($('#rCutFade').value, 10) / 1000 : undefined,  // ms -> s
    out_dir: $('#rOutDir').value.trim(),
    filename: $('#rFilename').value.trim(),
  }
  if (opts.denoise) {
    opts.denoise_strength = parseInt($('#rDenoiseStrength').value, 10)
    opts.denoise_normalize = !!($('#rDenoiseNorm') && $('#rDenoiseNorm').checked)
    // Движок: "deepfilter" = нейро (DeepFilterNet CLI, с авто-фолбэком на
    // afftdn на сервере), "afftdn" = прежний ffmpeg-путь.
    opts.denoise_engine = $('#rDenoiseEngine') ? $('#rDenoiseEngine').value : 'afftdn'
  }
  Object.assign(opts, burnOptsFromUI() || { burn_subtitles: false })
  closeOverlay('#renderModal')
  startRender(opts)
}

// Export the cut decisions as an NLE timeline project (FCPXML / EDL). No render:
// the server writes the file instantly, we surface a download link in the modal.
async function exportNle() {
  const fmt = ($('#rNleFormat') && $('#rNleFormat').value) || 'fcpxml'
  const out = $('#nleResult')
  const btn = $('#btnExportNle')
  if (btn) btn.disabled = true
  if (out) { out.classList.remove('hidden'); out.classList.remove('error'); out.textContent = 'Готовлю таймлайн…' }
  try {
    await save()
    const res = await fetch('/api/export/nle', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: fmt }),
    })
    if (!res.ok) {
      let detail = 'HTTP ' + res.status
      try { const j = await res.json(); if (j && j.detail) detail = (typeof j.detail === 'string') ? j.detail : JSON.stringify(j.detail) } catch {}
      if (out) { out.classList.add('error'); out.textContent = 'Не удалось экспортировать: ' + detail }
      toast('Экспорт в монтажку: ' + detail, 'error')
      return
    }
    const r = await res.json()
    const label = (r.format === 'edl') ? 'EDL' : 'FCPXML'
    if (out) {
      out.classList.remove('error')
      out.innerHTML = `Готово ✓ <a href="/api/output/${encodeURIComponent(r.name)}" target="_blank">скачать ${escapeHtml(r.name)}</a> · ${label}, ${r.segments} клип(ов)`
    }
    toast('Таймлайн готов: ' + r.name, 'success')
  } catch (e) {
    if (out) { out.classList.add('error'); out.textContent = 'Сеть: ' + e.message }
    toast('Экспорт в монтажку: ' + e.message, 'error')
  } finally {
    if (btn) btn.disabled = false
  }
}

async function startRender(opts) {
  if (st.task) return
  await save()
  let res
  try { res = await fetch('/api/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(opts || {}) }) }
  catch (e) { toast('Сеть: не удалось запустить рендер (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'Не удалось запустить рендер'); return }
  followTask('render')
}
function showResults(r) {
  // Open each artifact via /api/output/<basename> only when its path is present; otherwise show '—'.
  const link = (p, l) => {
    if (!p || typeof p !== 'string') return '—'
    const base = p.split(/[\\/]/).pop()
    if (!base) return '—'
    return `<a href="/api/output/${encodeURIComponent(base)}" target="_blank">${l}</a>`
  }
  $('#resultsCard').innerHTML = `
    <h2>Готово ✓</h2>
    <p>Длительность: <b>${fmt(r.old_duration)} &rarr; ${fmt(r.new_duration)}</b> · кодек ${r.encoder}</p>
    <p>Видео: ${link(r.mp4, 'открыть .mp4')}${r.vertical ? ' · вертикальный 9:16 ✓' : ''}</p>
    <p>Субтитры: ${link(r.srt, '.srt')} · ${link(r.vtt, '.vtt')} (${r.cues} реплик)${r.burned_subtitles ? ' · вшиты в видео ✓' : ''}</p>
    <p>Главы: ${link(r.chapters, 'chapters.txt')} (${r.n_chapters})</p>
    <p>Метаданные: ${link(r.metadata_path, 'metadata.txt')}${r.metadata ? ' (' + escapeHtml(r.metadata.substring(0, 40)) + '…)' : ''}</p>
    <button class="btn" id="btnCloseResults">Закрыть</button>`
  openOverlay('#results')
  $('#btnCloseResults').onclick = () => closeOverlay('#results')
}

/* ---------- modal / overlay management (a11y) ---------- */
let lastFocus = null
const OVERLAY_IDS = ['#help', '#results', '#files', '#renderModal', '#queueModal', '#privacyModal', '#modelsModal', '#clipsModal']
function openOverlay(id) {
  const ov = $(id); if (!ov) return
  lastFocus = document.activeElement
  ov.classList.remove('hidden')
  const first = focusables(ov)[0]
  if (first) first.focus()
}
function closeOverlay(id) {
  if (id === '#files') return closeFiles(null)   // files has its own exit path (picker cb + focus)
  const ov = $(id); if (!ov || ov.classList.contains('hidden')) return
  ov.classList.add('hidden')
  if (lastFocus && document.contains(lastFocus)) { try { lastFocus.focus() } catch {} }
  lastFocus = null
}
function openOverlayEl() { return OVERLAY_IDS.map($).find((el) => el && !el.classList.contains('hidden')) || null }
function focusables(root) {
  return [...root.querySelectorAll('a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea,[tabindex]:not([tabindex="-1"])')]
    .filter((el) => el.offsetParent !== null)
}
function trapTab(e, ov) {
  if (e.key !== 'Tab') return
  const f = focusables(ov); if (!f.length) { e.preventDefault(); return }
  const first = f[0], last = f[f.length - 1]
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
}

/* ---------- search ---------- */
function doSearch(q) {
  q = q.trim().toLowerCase()
  let first = null
  const spans = st.spans || (st.spans = $('#transcript').querySelectorAll('.w'))
  for (const sp of spans) {
    const hit = q && sp.textContent.toLowerCase().includes(q)
    sp.classList.toggle('match', hit); if (hit && !first) first = sp
  }
  if (first) first.scrollIntoView({ block: 'center' })
}

/* ---------- UI ---------- */
function bindUI() {
  $('#btnTranscribe').onclick = transcribe
  $('#btnRedetect').onclick = redetect
  $('#btnRender').onclick = openRenderModal
  $('#btnCancelTask').onclick = cancelTask
  $('#btnCloseRender').onclick = () => closeOverlay('#renderModal')
  $('#btnCancelRender').onclick = () => closeOverlay('#renderModal')
  $('#btnDoRender').onclick = submitRender
  if ($('#btnExportNle')) $('#btnExportNle').onclick = exportNle
  $('#rEncoder').onchange = (e) => seedQuality(e.target.value, null)
  $('#rQuality').oninput = (e) => $('#rQualVal').textContent = `${qualLabel($('#rEncoder').value)} ${e.target.value}`
  if ($('#rBurn')) {
    $('#rBurn').onchange = (e) => $('#rBurnOpts').classList.toggle('hidden', !e.target.checked)
    $('#rBurnSize').oninput = (e) => $('#rBurnSizeVal').textContent = `${e.target.value}px`
  }
  if ($('#rDenoise')) {
    $('#rDenoise').onchange = (e) => $('#rDenoiseOpts').classList.toggle('hidden', !e.target.checked)
    $('#rDenoiseStrength').oninput = (e) => $('#rDenoiseStrengthVal').textContent = denoiseStrengthLabel(e.target.value)
  }
  // «точный (2 прохода)» доступен только при включённой «Громкости под YouTube».
  if ($('#rLoudnorm') && $('#rLoudnorm2p')) {
    $('#rLoudnorm').onchange = (e) => { $('#rLoudnorm2p').disabled = !e.target.checked }
  }
  if ($('#rCutFade')) $('#rCutFade').oninput = (e) => setCutFadeLabel(e.target.value)
  $('#rBrowseOut').onclick = () => {
    $('#renderModal').classList.add('hidden')
    openFiles(true, (dir) => { if (dir) $('#rOutDir').value = dir; openOverlay('#renderModal') })
  }
  $('#btnHelp').onclick = () => { const h = $('#help'); h.classList.contains('hidden') ? openOverlay('#help') : closeOverlay('#help') }
  $('#btnCloseHelp').onclick = () => closeOverlay('#help')
  $('#btnPlay').onclick = () => video.paused ? video.play() : video.pause()
  $('#btnStop').onclick = () => { video.pause(); seek(0) }
  $('#speed').onchange = (e) => { video.playbackRate = +e.target.value }
  $('#showCuts').onchange = (e) => { st.showCuts = e.target.checked; renderRegions(); refreshCuts() }
  $('#skipCuts').onchange = (e) => { if (st.afterMode) { e.target.checked = true; return } st.preview = e.target.checked }
  $('#btnAfterToggle').onclick = () => setAfterMode(!st.afterMode)
  $('#btnSubsToggle').onclick = () => setSubsMode(!st.subsMode)
  // Кнопки генерации живут внутри своих вкладок — раскрывать нечего.
  $('#btnChaptersToggle').onclick = loadChapters
  $('#btnMetaGenerate').onclick = loadMetadata
  // F4 — Clip Maker: вкладка + модалка пресета рендера клипов
  $('#btnClipsSuggest').onclick = loadClips
  $('#btnClipsRender').onclick = clipsRender
  $('#btnClipsSettings').onclick = openClipsModal
  $('#btnCloseClips').onclick = () => closeOverlay('#clipsModal')
  $('#btnClipsOptsCancel').onclick = () => closeOverlay('#clipsModal')
  $('#btnClipsOptsSave').onclick = saveClipsOpts
  $('#cCenterMode').onchange = (e) => $('#cCenterRow').classList.toggle('hidden', e.target.value !== 'manual')
  $('#cCenterPos').oninput = (e) => { const el = $('#cCenterVal'); if (el) el.textContent = e.target.value + '%' }
  $('#cBrowseOut').onclick = () => {
    $('#clipsModal').classList.add('hidden')
    openFiles(true, (dir) => { if (dir) $('#cOutDir').value = dir; openOverlay('#clipsModal') })
  }
  for (const el of document.querySelectorAll('.metaCopyBtn')) { el.onclick = () => copyField(el.dataset.field) }
  $('#btnMetaCopyAll').onclick = copyAllMeta
  $('#metaTitle').addEventListener('input', () => { const l = $('#metaTitleLen'); const t = ($('#metaTitle').textContent || '').trim(); if (l) l.textContent = t ? `(${t.length}/100)` : '' })
  $('#btnPrevCut').onclick = prevCut
  $('#btnNextCut').onclick = nextCut
  $('#btnSplit').onclick = () => {
    if (st.splitMark == null) { st.splitMark = video.currentTime; $('#btnSplit').innerHTML = icon('scissors') + `до: ${fmt(st.splitMark)}` }
    else { addManual(Math.min(st.splitMark, video.currentTime), Math.max(st.splitMark, video.currentTime)); st.splitMark = null; $('#btnSplit').innerHTML = icon('scissors') + 'Разрез' }
  }
  $('#zoom').oninput = (e) => zoomTo(+e.target.value)
  $('#zoomIn').onclick = () => { $('#zoom').value = Math.min(100, +$('#zoom').value + 12); zoomTo(+$('#zoom').value) }
  $('#zoomOut').onclick = () => { $('#zoom').value = Math.max(0, +$('#zoom').value - 12); zoomTo(+$('#zoom').value) }
  $('#search').oninput = (e) => doSearch(e.target.value)

  const cf = $('#cutFloat')
  cf.addEventListener('mousedown', (e) => e.preventDefault())  // keep selection
  cf.onclick = cutSelection
  document.addEventListener('selectionchange', () => setTimeout(onSelectionChange, 0))

  $('#transcript').addEventListener('click', (e) => {
    const g = e.target.closest('.gutter')
    if (g) { const ws_ = st.paras[+g.dataset.p].words; addManual(ws_[0].start, ws_[ws_.length - 1].end); return }
    const sp = e.target.closest('.w'); if (!sp) return
    if (sp.isContentEditable) return  // идёт правка слова — клик ставит каретку, не перематывает
    const sel = window.getSelection(); if (sel && !sel.isCollapsed) return  // drag = select, not seek
    const w = st.words[+sp.dataset.i]; seek(w.start)
    const mid = (w.start + w.end) / 2
    const seg = st.segs.find((s) => s.enabled && s.action === 'remove' && mid >= s.start && mid < s.end)
    if (seg) select(seg.id)
  })

  // Двойной клик по слову — инлайн-правка текста (одиночный клик = seek, как был).
  $('#transcript').addEventListener('dblclick', (e) => {
    const sp = e.target.closest('.w'); if (!sp || sp.isContentEditable) return
    e.preventDefault()
    startWordEdit(sp)
  })
}

// '+'/'-' live on the digit row; in the Russian layout those keys still emit '+'/'=' and '-'/'_'.
const ZOOM_KEYS_IN = new Set(['+', '='])
const ZOOM_KEYS_OUT = new Set(['-', '_'])
// '['/']' physical keys emit Cyrillic 'х'/'ъ' under the Russian layout.
const BRACKET_PREV = new Set(['[', 'х', 'Х'])
const BRACKET_NEXT = new Set([']', 'ъ', 'Ъ'])
function bindKeys() {
  document.addEventListener('keydown', (e) => {
    const k = e.key

    // Undo / Redo — work regardless of focus context (except text inputs handled below).
    // contenteditable (правка слова, поля метаданных) — тоже текстовый контекст:
    // глобальные хоткеи не должны срабатывать поверх набора текста.
    const tagEarly = (e.target.tagName || '').toLowerCase()
    const inField = tagEarly === 'input' || tagEarly === 'textarea' || tagEarly === 'select' || e.target.isContentEditable
    if ((e.ctrlKey || e.metaKey) && !inField && (k === 'z' || k === 'Z' || k === 'я' || k === 'Я')) {
      e.preventDefault(); e.shiftKey ? redo() : undo(); return
    }
    if ((e.ctrlKey || e.metaKey) && !inField && (k === 'y' || k === 'Y' || k === 'н' || k === 'Н')) {
      e.preventDefault(); redo(); return
    }

    // EARLY-RETURN when any overlay is open: only handle that overlay's own keys (Esc, Tab-trap).
    const ov = openOverlayEl()
    if (ov) {
      if (k === 'Tab') { trapTab(e, ov); return }
      if (k === 'Escape') {
        e.preventDefault()
        if (ov.id === 'files') { if (st.hasSession || st.pickFolderCb) closeFiles(null) }   // no-session opener: keep it modal
        else closeOverlay('#' + ov.id)
        return
      }
      return  // swallow all other shortcuts while a modal is up
    }

    if (inField) {
      if (k === 'Enter' && e.target.id === 'search') doSearch(e.target.value)
      return
    }
    // Фокус в таб-баре: стрелки/Space/1–4 обрабатывает сам tablist (bindTabs),
    // глобальные seek ±5с / play-pause не должны конкурировать (зеркально
    // inField-гарду). Escape пропускаем — он безопасен и работает отовсюду.
    const ae = document.activeElement
    if (k !== 'Escape' && ae && ae.closest && ae.closest('[role="tablist"]')) return
    if (e.ctrlKey || e.metaKey || e.altKey) return  // leave OS/browser combos alone

    const fps = (st.media && st.media.fps) ? st.media.fps : 25
    if (k === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause() }
    else if (k === 'ArrowLeft') { e.preventDefault(); seek(video.currentTime - 5) }
    else if (k === 'ArrowRight') { e.preventDefault(); seek(video.currentTime + 5) }
    else if (k === ',') seek(video.currentTime - 1 / fps)
    else if (k === '.') seek(video.currentTime + 1 / fps)
    else if (k === 'i' || k === 'ш') { st.inP = video.currentTime; flash(`метка I: ${fmt(st.inP)}`) }
    else if (k === 'o' || k === 'щ') { st.outP = video.currentTime; flash(`метка O: ${fmt(st.outP)}`) }
    else if (k === 'm' || k === 'ь') cutFromMarks()
    else if (k === 'x' || k === 'ч') cutSelection()
    else if (k === 'Enter') { if (st.selected) toggleEnabled(st.selected) }
    else if (k === 'c' || k === 'с') { if (st.selected) cycleAction(st.selected) }
    else if (k === 'Delete' || k === 'Backspace') {
      // Выделенный текст транскрипта → вырез (как ✂/X); гарды те же, что у X:
      // не в инпуте/contenteditable (см. inField выше) и есть валидное выделение.
      if (st.selRange) { e.preventDefault(); cutSelection() }
      else if (k === 'Delete' && st.selected) deleteSeg(st.selected)   // Delete only — never Backspace
    }
    else if (BRACKET_PREV.has(k)) prevCut()
    else if (BRACKET_NEXT.has(k)) nextCut()
    else if (k === 'p' || k === 'з') { if (st.afterMode) return; $('#skipCuts').checked = !$('#skipCuts').checked; st.preview = $('#skipCuts').checked }
    else if (k === 'a' || k === 'ф') setAfterMode(!st.afterMode)
    else if (ZOOM_KEYS_IN.has(k)) { $('#zoom').value = Math.min(100, +$('#zoom').value + 12); zoomTo(+$('#zoom').value) }
    else if (ZOOM_KEYS_OUT.has(k)) { $('#zoom').value = Math.max(0, +$('#zoom').value - 12); zoomTo(+$('#zoom').value) }
    else if (k === 's' || k === 'ы') { e.preventDefault(); save() }
    else if (k === 'r' || k === 'к') openRenderModal()
    else if (k === ';' || k === 'ж') { st.syncOffset = Math.round((st.syncOffset - 0.05) * 100) / 100; saveSync() }
    else if (k === "'" || k === 'э') { st.syncOffset = Math.round((st.syncOffset + 0.05) * 100) / 100; saveSync() }
    // Вкладки правой колонки: 1–4 без модификаторов (раскладко-независимы RU/EN).
    // Фокус не трогаем (исключение внутри setActiveTab — фокус из скрытой панели).
    else if (k === '1') setActiveTab('cuts')
    else if (k === '2') setActiveTab('chapters')
    else if (k === '3') setActiveTab('meta')
    else if (k === '4') setActiveTab('clips')
    else if (k === '?' || k === 'h') openOverlay('#help')
    else if (k === 'Escape') { window.getSelection().removeAllRanges(); $('#cutFloat').classList.add('hidden'); st.selected = null; clipsSetActive(null); renderCutlist() }
  })
}

/* ---------- file browser / open new clip ---------- */
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]))
const mb = (b) => b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' ГБ' : (b / 1048576).toFixed(1) + ' МБ'
const pathJoin = (dir, name) => { const s = dir.includes('\\') ? '\\' : '/'; return dir.endsWith(s) ? dir + name : dir + s + name }

function openFiles(closable, pickCb) {
  st.pickFolderCb = pickCb || null
  $('#btnPickDir').classList.toggle('hidden', !pickCb)
  $('#btnCloseFiles').style.display = (closable || pickCb) ? '' : 'none'
  lastFocus = document.activeElement
  $('#files').classList.remove('hidden')
  // A6/A7 — онбординг первого запуска: no-session пикер модален и неотключаем,
  // его скрим (z-50) накрывает степпер в #transcript и карточку ffmpeg (z-35).
  // Поэтому шаги «С чего начать» дублируем ВНУТРЬ модалки, а уже показанную
  // карточку «ffmpeg не найден» переносим сюда же (health мог ответить раньше).
  if (!st.hasSession && !pickCb) {
    const head = $('#files .filesHead')
    if (head && !$('#filesStepper')) {
      const stp = document.createElement('div')
      stp.id = 'filesStepper'; stp.className = 'stepper inFiles'
      stp.innerHTML = stepperHTML(true)
      head.after(stp)
    }
    const card = $('#ffmpegCard')
    if (head && card && !card.closest('#files')) { card.classList.add('inFiles'); head.after(card) }
  }
  const fi = $('#pathInput'); if (fi) fi.focus()
  browseDir(st.curDir)
}
// Single exit path for #files: always clears pickFolderCb (fixes Escape/close leak), restores focus.
function closeFiles(pickedDir) {
  $('#files').classList.add('hidden')
  const cb = st.pickFolderCb; st.pickFolderCb = null
  if (lastFocus && document.contains(lastFocus)) { try { lastFocus.focus() } catch {} }
  lastFocus = null
  if (cb) cb(pickedDir != null ? pickedDir : null)
}

async function browseDir(dir) {
  const url = '/api/browse' + (dir ? ('?dir=' + encodeURIComponent(dir)) : '')
  const res = await fetch(url)
  if (!res.ok) { $('#browser').innerHTML = '<div class="empty">Не удалось открыть папку</div>'; return }
  const j = await res.json()
  st.curDir = j.dir
  $('#curDir').textContent = j.dir
  $('#btnUp').disabled = !j.parent
  $('#btnUp').dataset.parent = j.parent || ''
  const box = $('#browser'); box.replaceChildren()
  if (!j.folders.length && !j.files.length) { box.innerHTML = '<div class="empty">Нет видео в этой папке</div>'; return }
  for (const name of j.folders) {
    const el = document.createElement('div'); el.className = 'fitem folder'
    el.innerHTML = `${icon('folder')} <span>${esc(name)}</span>`
    el.onclick = () => browseDir(pathJoin(j.dir, name))
    box.appendChild(el)
  }
  for (const f of j.files) {
    const el = document.createElement('div'); el.className = 'fitem video'
    el.innerHTML = `${icon('film')} <span>${esc(f.name)}</span> <span class="fsize">${mb(f.size)}</span>`
    el.onclick = () => openPath(pathJoin(j.dir, f.name))
    box.appendChild(el)
  }
}

const VIDEO_RE = /\.(mp4|mov|mkv|webm|avi|m4v|ts|flv|wmv|mpg|mpeg)$/i
const DROPZONE_HTML = '<input type="file" id="fileInput" accept="video/*" hidden>Перетащи видео сюда или <span class="link">выбери файл</span>'

function resetDropzone() {
  const dz = $('#dropzone'); dz.innerHTML = DROPZONE_HTML
  $('#fileInput').onchange = (e) => uploadFile(e.target.files[0])
}

async function openPath(path) {
  if (!path) return
  // Switching clips is deliberate navigation: stop the pending autosave timer and
  // flag navigation BEFORE awaiting /api/open, so a 700 ms-delayed PUT can't fire
  // during the swap and write THIS clip's cuts into the NEW clip's cutlist.
  clearTimeout(saveTimer); st._navigating = true; st.dirty = false
  flash('Открываю…')
  let res
  try { res = await fetch('/api/open', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }) }
  catch (e) { st._navigating = false; toast('Сеть: не удалось открыть (' + e + ')', 'error'); resetDropzone(); return }
  if (!res.ok) { st._navigating = false; const e = await res.json().catch(() => ({})); toast('Не удалось открыть: ' + (e.detail || res.status), 'error'); resetDropzone(); return }
  location.reload()
}

// Upload via XHR so we can show progress and never get stuck silently.
function uploadFile(file) {
  if (!file) return
  if (!VIDEO_RE.test(file.name)) { toast('Это не похоже на видеофайл: ' + file.name, 'error'); return }
  const dz = $('#dropzone')
  const xhr = new XMLHttpRequest()
  xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name))
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) dz.textContent = `Загрузка ${file.name} — ${Math.round(e.loaded / e.total * 100)}%`
  }
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      try { const j = JSON.parse(xhr.responseText); dz.textContent = 'Открываю ' + file.name + ' …'; openPath(j.path) }
      catch { dz.textContent = 'Ошибка ответа сервера'; setTimeout(resetDropzone, 2500) }
    } else { dz.textContent = 'Ошибка загрузки (' + xhr.status + ')'; setTimeout(resetDropzone, 2500) }
  }
  xhr.onerror = () => { dz.textContent = 'Не удалось загрузить (соединение разорвано)'; setTimeout(resetDropzone, 2500) }
  xhr.timeout = 0
  dz.textContent = `Загрузка ${file.name} … 0%`
  xhr.send(file)
}

function bindFiles() {
  $('#btnFiles').onclick = () => openFiles(st.hasSession)
  $('#btnCloseFiles').onclick = () => closeFiles(null)                 // cancel -> reopen caller via cb(null)
  $('#btnPickDir').onclick = () => { if (st.pickFolderCb) closeFiles(st.curDir) }
  $('#btnUp').onclick = (e) => { const p = e.currentTarget.dataset.parent; if (p) browseDir(p) }
  $('#btnOpenPath').onclick = () => openPath($('#pathInput').value.trim())
  $('#pathInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') openPath(e.target.value.trim()) })
  $('#fileInput').onchange = (e) => uploadFile(e.target.files[0])

  const dz = $('#dropzone')
  ;['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag') }))
  ;['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove('drag')))
  document.addEventListener('dragover', (e) => e.preventDefault())
  document.addEventListener('drop', (e) => {
    e.preventDefault()
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) { if ($('#files').classList.contains('hidden')) openFiles(st.hasSession); uploadFile(f) }
  })
}

/* ---------- F3: очередь нескольких роликов ---------- */
const Q_STATUS_RU = { pending: 'в очереди', running: 'обработка…', done: 'готово', error: 'ошибка' }

function bindQueue() {
  const on = (sel, fn) => { const el = $(sel); if (el) el.onclick = fn }
  on('#btnQueue', openQueueModal)
  on('#btnCloseQueue', () => closeOverlay('#queueModal'))
  on('#btnAddToQueue', addCurrentToQueue)
  on('#btnAddFileToQueue', addFileToQueue)
  on('#btnQueueStart', startQueue)
  on('#btnQueueStop', stopQueue)
  on('#btnQueueClear', clearQueue)
  const pi = $('#queuePathInput'); if (pi) pi.addEventListener('keydown', (e) => { if (e.key === 'Enter') addFileToQueue() })
}

async function openQueueModal() {
  await loadQueueList()
  openOverlay('#queueModal')
}

// Текущие настройки рендера для постановки клипа в очередь: дефолты + текущая папка вывода.
function currentRenderOpts() {
  const d = st.rdefaults || {}
  return {
    encoder: d.encoder || 'nvenc',
    quality: (d.quality != null) ? d.quality : null,
    audio_bitrate: d.audio_bitrate || '320k',
    censor_method: d.censor_method || 'partial',
    subtitles: true, chapters: true,
    out_dir: st.outDir || '',
  }
}

async function queuePost(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  if (!res.ok) { await failToast(res, 'Очередь'); return null }
  try { return await res.json() } catch { return {} }
}

async function addCurrentToQueue() {
  if (!st.hasSession) { toast('Сначала открой ролик', 'info'); return }
  // st._inpPath — абсолютный путь текущего клипа (из /api/state.path, см. init).
  const p = st._inpPath
  if (!p) { toast('Не удалось определить путь текущего клипа — добавь файл через «+ Добавить файл…»', 'error'); return }
  await save()   // зафиксировать правки текущего клипа на диск перед постановкой в очередь
  const r = await queuePost('/api/queue/add', { path: p, render_opts: currentRenderOpts() })
  if (r && r.ok) { toast('Клип добавлен в очередь', 'success'); loadQueueList() }
}

async function addFileToQueue() {
  const inp = $('#queuePathInput')
  const p = (inp && inp.value || '').trim()
  if (!p) { toast('Вставь путь к видеофайлу в поле слева', 'info'); if (inp) inp.focus(); return }
  const r = await queuePost('/api/queue/add', { path: p, render_opts: currentRenderOpts() })
  if (r && r.ok) { toast('Файл добавлен в очередь', 'success'); if (inp) inp.value = ''; loadQueueList() }
}

async function startQueue() {
  const r = await queuePost('/api/queue/start', {})
  if (r && r.ok) { toast('Очередь запущена', 'success'); startQueuePoll() }   // startQueuePoll() уже делает первый loadQueueList()
}

async function stopQueue() {
  const r = await queuePost('/api/queue/stop', {})
  if (r && r.ok) { toast('Очередь останавливается…', 'info'); loadQueueList() }
}

async function clearQueue() {
  const r = await queuePost('/api/queue/clear', {})
  if (r && r.ok) { if (r.removed) toast(`Удалено заданий: ${r.removed}`, 'info'); loadQueueList() }
}

async function removeQueueJob(id) {
  const r = await queuePost('/api/queue/remove', { id })
  if (r && r.ok) loadQueueList()
}

async function loadQueueList() {
  let j
  try { const res = await fetch('/api/queue'); if (!res.ok) throw new Error('HTTP ' + res.status); j = await res.json() }
  catch (e) { toast('Не удалось загрузить очередь: ' + e.message, 'error'); return }
  st.queueJobs = j.jobs || []
  st.queueRunning = !!j.running
  renderQueueList()
  updateQueueBadge()
  // Автостоп опроса, когда воркер встал и нет «бегущих» заданий.
  if (!st.queueRunning && !st.queueJobs.some((x) => x.status === 'running')) stopQueuePoll()
}

function updateQueueBadge() {
  const badge = $('#queueBadge'); if (!badge) return
  const active = st.queueJobs.filter((x) => x.status === 'pending' || x.status === 'running').length
  badge.textContent = String(active)
  badge.classList.toggle('hidden', active === 0)
  // Старт ↔ Стоп по состоянию воркера
  const running = st.queueRunning
  const start = $('#btnQueueStart'), stop = $('#btnQueueStop')
  if (start) start.classList.toggle('hidden', running)
  if (stop) stop.classList.toggle('hidden', !running)
}

// Ссылка на готовый результат через /api/output/<basename> (как в showResults).
function queueResultLink(job) {
  const r = job.result || {}
  const mp4 = r.mp4
  if (!mp4 || typeof mp4 !== 'string') return ''
  const base = mp4.split(/[\\/]/).pop()
  if (!base) return ''
  const dur = (r.old_duration != null && r.new_duration != null)
    ? ` · ${fmt(r.old_duration)} &rarr; ${fmt(r.new_duration)}` : ''
  return `<a href="/api/output/${encodeURIComponent(base)}" target="_blank">открыть .mp4</a>${dur}`
}

function renderQueueList() {
  const box = $('#queueList'); if (!box) return
  box.replaceChildren()
  const jobs = st.queueJobs
  if (!jobs.length) {
    const ph = document.createElement('div'); ph.className = 'empty placeholder'
    ph.innerHTML = icon('queue') + '<div>Очередь пуста</div>'
    box.appendChild(ph); return
  }
  for (const job of jobs) {
    const row = document.createElement('div'); row.className = 'qJob'
    const name = document.createElement('div'); name.className = 'qJob-name'
    name.textContent = job.name || job.path || job.id; name.title = job.path || ''
    const status = document.createElement('span')
    status.className = 'qJob-status status-' + job.status
    status.textContent = Q_STATUS_RU[job.status] || job.status
    const del = document.createElement('button')
    del.className = 'qJob-del'; del.innerHTML = icon('x'); del.title = 'Убрать из очереди'; del.setAttribute('aria-label', 'Убрать из очереди')
    del.disabled = job.status === 'running'
    del.onclick = () => removeQueueJob(job.id)
    row.appendChild(name); row.appendChild(status); row.appendChild(del)
    if (job.status === 'running') {
      const bar = document.createElement('div'); bar.className = 'qJob-bar'
      const fill = document.createElement('div'); fill.className = 'qJob-fill'
      fill.style.width = Math.max(0, Math.min(100, job.percent || 0)) + '%'
      bar.appendChild(fill); row.appendChild(bar)
      if (job.stage) { const st_ = document.createElement('div'); st_.className = 'qJob-stage'; st_.textContent = job.stage + ' · ' + Math.round(job.percent || 0) + '%'; row.appendChild(st_) }
    }
    if (job.status === 'error' && job.error) {
      const er = document.createElement('div'); er.className = 'qJob-stage'; er.textContent = 'Ошибка: ' + job.error
      row.appendChild(er)
    }
    if (job.status === 'done') {
      const html = queueResultLink(job)
      if (html) { const res = document.createElement('div'); res.className = 'qJob-result'; res.innerHTML = html; row.appendChild(res) }
    }
    box.appendChild(row)
  }
}

function startQueuePoll() {
  if (st.queuePollTimer) return
  loadQueueList()
  st.queuePollTimer = setInterval(loadQueueList, 1500)
}
function stopQueuePoll() {
  if (st.queuePollTimer) { clearInterval(st.queuePollTimer); st.queuePollTimer = 0 }
}

/* ---------- P2-#4: бейдж «zero-upload» + модалка приватности ---------- */
function bindPrivacy() {
  const b = $('#netBadge'); if (b) b.onclick = openPrivacyModal
  const c = $('#btnClosePrivacy'); if (c) c.onclick = () => closeOverlay('#privacyModal')
  const t = $('#pOfflineToggle'); if (t) t.onchange = () => setOffline(t.checked)
}

// Перерисовать компактный бейдж в топбаре по {offline, external_allowed, blocked}.
function renderNetBadge(net) {
  const b = $('#netBadge'); if (!b || !net) return
  st.network = net
  b.classList.remove('net-local', 'net-offline', 'net-warn')
  if (net.offline) {
    b.classList.add('net-offline'); b.innerHTML = icon('shield') + '<span>Оффлайн</span>'
    b.title = 'Оффлайн-режим — исходящие соединения в интернет заблокированы'
  } else if ((net.external_allowed || 0) > 0) {
    b.classList.add('net-warn'); b.innerHTML = icon('globe-warn') + `<span>${escapeHtml(String(net.external_allowed))} внешн.</span>`
    b.title = `Внешних соединений: ${net.external_allowed} (вероятно, загрузка модели). Клик — подробности.`
  } else {
    b.classList.add('net-local'); b.innerHTML = icon('shield') + '<span>Локально · 0 внешних</span>'
    b.title = 'Всё локально — ни одного внешнего соединения. Клик — подробности.'
  }
}

async function openPrivacyModal() {
  openOverlay('#privacyModal')
  await refreshNetwork()
}

// Подтянуть полную сводку /api/network и обновить бейдж + счётчики в модалке.
async function refreshNetwork() {
  let j
  try { const res = await fetch('/api/network'); if (!res.ok) throw new Error('HTTP ' + res.status); j = await res.json() }
  catch (e) { toast('Не удалось получить сетевую сводку: ' + e.message, 'error'); return }
  renderNetBadge({ offline: j.offline, external_allowed: j.stats.external_allowed, blocked: j.stats.blocked })
  const s = j.stats || {}
  const setNum = (id, v) => { const el = $(id); if (el) el.textContent = String(v || 0) }
  setNum('#pStatExternal', s.external_allowed)
  setNum('#pStatLocal', s.local)
  setNum('#pStatBlocked', s.blocked)
  const tog = $('#pOfflineToggle'); if (tog) tog.checked = !!j.offline
  const sum = $('#pSummary'); if (sum) sum.textContent = j.summary || ''
  // Прозрачно перечислить внешние хосты (если были) — это НЕ ваши данные.
  const hostsEl = $('#pHosts')
  const ext = s.external_hosts || {}, blk = s.blocked_hosts || {}
  const names = [...new Set([...Object.keys(ext), ...Object.keys(blk)])]
  if (hostsEl) {
    if (!names.length) { hostsEl.classList.add('hidden'); hostsEl.innerHTML = '' }
    else {
      const rows = names.map((h) => {
        const a = ext[h] || 0, d = blk[h] || 0
        const tag = d && !a ? 'заблокировано' : (a ? 'разовая загрузка модели' : 'заблокировано')
        return `<div class="pHostRow"><span class="pHostName">${escapeHtml(h)}</span><span class="pHostTag muted">${tag}${a ? ' · ' + a : ''}${d ? ' · ×' + d : ''}</span></div>`
      }).join('')
      hostsEl.innerHTML = `<div class="pHostsHead muted">Внешние адреса (не ваши данные — служебная загрузка моделей):</div>${rows}`
      hostsEl.classList.remove('hidden')
    }
  }
}

// Тумблер оффлайна: оптимистично перерисовываем, при ошибке — откатываем.
async function setOffline(enabled) {
  const tog = $('#pOfflineToggle')
  const prev = st.network || { offline: !enabled }
  // оптимистичная отрисовка бейджа
  renderNetBadge({ offline: enabled, external_allowed: (prev.external_allowed || 0), blocked: (prev.blocked || 0) })
  let res
  try {
    res = await fetch('/api/network/offline', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !!enabled }),
    })
  } catch (e) {
    if (tog) tog.checked = !!prev.offline
    renderNetBadge(prev)
    toast('Не удалось переключить оффлайн-режим: ' + e.message, 'error')
    return
  }
  if (!res.ok) {
    if (tog) tog.checked = !!prev.offline
    renderNetBadge(prev)
    await failToast(res, 'Оффлайн-режим')
    return
  }
  toast(enabled ? 'Оффлайн-режим включён — всё заблокировано' : 'Оффлайн-режим выключен', enabled ? 'success' : 'info')
  await refreshNetwork()
}

/* ---------- P2-#5: модалка «⚙ Модели» (сменные Whisper + LLM) ---------- */
// Снимок последнего GET /api/models + текущий выбор пользователя в модалке.
const mState = { whisperCurrent: null, whisperSel: null, llmCurrent: null, llmSel: null, installed: [], available: false }

function bindModels() {
  const on = (sel, fn) => { const el = $(sel); if (el) el.onclick = fn }
  on('#btnModels', openModelsModal)
  on('#btnCloseModels', () => closeOverlay('#modelsModal'))
  on('#btnModelsCancel', () => closeOverlay('#modelsModal'))
  on('#btnModelsSave', saveModels)
}

async function openModelsModal() {
  // Открываем сразу (мгновенный отклик), затем подтягиваем актуальное состояние.
  openOverlay('#modelsModal')
  await loadModels()
}

// GET /api/models -> наполнить блоки Whisper и LLM, сбросить выбор к серверным значениям.
async function loadModels() {
  let j
  try { const res = await fetch('/api/models'); if (!res.ok) throw new Error('HTTP ' + res.status); j = await res.json() }
  catch (e) { toast('Не удалось загрузить список моделей: ' + e.message, 'error'); return }
  renderModels(j)
}

function renderModels(j) {
  const w = (j && j.whisper) || {}, l = (j && j.llm) || {}
  mState.whisperCurrent = w.current || null
  mState.whisperSel = w.current || null
  mState.llmCurrent = l.current || null
  mState.llmSel = l.current || null
  mState.installed = Array.isArray(l.installed) ? l.installed : []
  mState.available = !!l.available

  // --- Whisper: радио-пресеты с подсказками VRAM/скорость ---
  const list = $('#mWhisperList'); if (list) list.innerHTML = ''
  const presets = Array.isArray(w.presets) ? w.presets : []
  for (const p of presets) {
    const isCur = p.model === w.current
    const lbl = document.createElement('label')
    lbl.className = 'mPreset' + (isCur ? ' sel' : '')
    const tag = isCur ? '<span class="mTag">текущая</span>' : ''
    lbl.innerHTML =
      `<input type="radio" name="mWhisper" value="${escapeHtml(p.model)}"${isCur ? ' checked' : ''}>` +
      `<span class="mPresetText">` +
        `<span class="mPresetLabel">${escapeHtml(p.label || p.model)}${tag}</span>` +
        `<span class="mPresetHint">${escapeHtml(p.hint || '')}</span>` +
      `</span>`
    const input = lbl.querySelector('input')
    input.onchange = () => {
      mState.whisperSel = p.model
      for (const el of list.querySelectorAll('.mPreset')) el.classList.remove('sel')
      lbl.classList.add('sel')
    }
    if (list) list.appendChild(lbl)
  }
  if (list && !presets.length) list.innerHTML = '<div class="empty placeholder">Пресеты недоступны</div>'

  // Модель текущего транскрипта (если ролик уже распознан) — чтобы было видно, что применится позже.
  const note = $('#mWhisperTranscript')
  if (note) note.textContent = w.transcript ? `Текущий транскрипт: ${w.transcript}` : ''

  // --- LLM: дропдаун из установленных моделей + статус Ollama ---
  renderLlmBlock()
}

function renderLlmBlock() {
  const sel = $('#mLlmSelect'); const status = $('#mLlmStatus')
  if (sel) {
    sel.innerHTML = ''
    const names = [...mState.installed]
    // Если текущая модель не установлена — всё равно показать её в списке (отмеченной), чтобы выбор не «прыгал».
    if (mState.llmCurrent && !names.includes(mState.llmCurrent)) names.unshift(mState.llmCurrent)
    if (!names.length) {
      const opt = document.createElement('option')
      opt.value = ''; opt.textContent = mState.available ? 'Нет установленных моделей' : 'Ollama не запущена'
      opt.disabled = true; opt.selected = true
      sel.appendChild(opt)
      sel.disabled = true
    } else {
      sel.disabled = false
      for (const n of names) {
        const opt = document.createElement('option')
        opt.value = n; opt.textContent = n
        if (n === mState.llmSel) opt.selected = true
        sel.appendChild(opt)
      }
    }
    sel.onchange = () => { mState.llmSel = sel.value; renderLlmHint() }
  }
  if (status) {
    status.textContent = mState.available ? 'Ollama запущена' : 'Ollama не запущена'
    status.classList.toggle('warnNote', !mState.available)
  }
  renderLlmHint()
}

function renderLlmHint() {
  const hint = $('#mLlmHint'); if (!hint) return
  if (!mState.available) {
    hint.textContent = 'Ollama не запущена — ИИ-функции (дубли, главы, метаданные) выключены. Запустите Ollama и откройте окно снова.'
    return
  }
  const chosen = mState.llmSel || mState.llmCurrent
  if (chosen && !mState.installed.includes(chosen)) {
    hint.textContent = `Модель «${chosen}» не установлена. Выполните в терминале:  ollama pull ${chosen}`
    return
  }
  hint.textContent = 'Используется для поиска неудачных дублей, генерации глав и метаданных YouTube.'
}

// POST /api/models — отправить только изменившиеся поля, применить, показать тост.
async function saveModels() {
  const body = {}
  if (mState.whisperSel && mState.whisperSel !== mState.whisperCurrent) body.whisper = mState.whisperSel
  if (mState.llmSel && mState.llmSel !== mState.llmCurrent) body.llm = mState.llmSel
  if (!Object.keys(body).length) { toast('Изменений нет', 'info'); closeOverlay('#modelsModal'); return }

  const btn = $('#btnModelsSave'); if (btn) btn.disabled = true
  let res
  try {
    res = await fetch('/api/models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (e) {
    if (btn) btn.disabled = false
    toast('Не удалось сохранить выбор моделей: ' + e.message, 'error'); return
  }
  if (btn) btn.disabled = false
  if (!res.ok) { await failToast(res, 'Модели'); return }
  const j = await res.json().catch(() => ({}))

  // Зафиксировать новые «текущие» значения в снимке.
  if (body.whisper) mState.whisperCurrent = j.whisper || mState.whisperSel
  if (body.llm) { mState.llmCurrent = j.llm || mState.llmSel; mState.available = !!j.llm_available; mState.installed = Array.isArray(j.llm_installed) ? j.llm_installed : mState.installed }

  // Сообщить пользователю результат, особенно по LLM (Ollama могла не подхватить модель).
  const msgs = []
  if (body.whisper) msgs.push(`Распознавание: ${mState.whisperCurrent} — применится при следующей транскрипции`)
  if (body.llm) {
    if (j.llm_ready) msgs.push(`ИИ-модель: ${mState.llmCurrent}`)
    else if (j.llm_reason === 'model_missing') msgs.push(`ИИ-модель «${mState.llmCurrent}» не установлена — выполните: ollama pull ${mState.llmCurrent}`)
    else if (j.llm_reason === 'ollama_off') msgs.push('ИИ-модель сохранена, но Ollama не запущена — ИИ-функции выключены')
    else if (j.llm_reason === 'disabled') msgs.push('ИИ-модель сохранена (ИИ отключён ключом запуска)')
    else msgs.push(`ИИ-модель: ${mState.llmCurrent}`)
  }
  const kind = (body.llm && !j.llm_ready) ? 'info' : 'success'
  toast(msgs.join(' · ') || 'Сохранено', kind)

  // Обновить бейдж «ИИ выключен» в правой панели, если LLM сменилась и сессия открыта.
  if (body.llm) {
    const badge = $('#llmBadge')
    if (badge) badge.classList.toggle('hidden', !!j.llm_ready)
  }
  closeOverlay('#modelsModal')
}
