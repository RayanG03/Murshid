// واجهة «مُرشِد»: منطق نسخة Angular (app.ts) منقول إلى JavaScript عادي ومربوط بـ app.py.
// كل سؤال يمر بمرحلتين: /api/search (المصادر) ثم /api/answer (الإجابة) مع آخر 3 أدوار.

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const UNIVERSITIES = [
  { id: 'ksu', api: 'KSU', name: 'جامعة الملك سعود', short: 'الملك سعود', en: 'King Saud University', logo: '/static/ksu_emblem.png' },
  { id: 'kau', api: 'KAU', name: 'جامعة الملك عبدالعزيز', short: 'الملك عبدالعزيز', en: 'King Abdulaziz University', logo: '/static/kau_logo.png' },
];

const SUGGESTIONS = {
  ksu: [
    'كم مقرر يحق لي الاعتذار عنه في الفصل؟',
    'غبت أكثر من 25% من المحاضرات، وش يصير؟',
    'ما شروط مرتبة الشرف الأولى؟',
    'متى تبدأ اختبارات نهاية الفصل الأول؟',
  ],
  kau: [
    'كم إنذار أكاديمي قبل الفصل؟',
    'متى تنقطع مكافأتي الشهرية؟',
    'ما عقوبة الغش في الاختبار؟',
    'متى يبدأ الفصل الدراسي الأول؟',
  ],
};

// أسئلة متابعة تظهر بعد كل رد؛ تُرسل مع سجل المحادثة فيفهمها النظام
const FOLLOWUPS = ['اشرح لي أكثر', 'لخّص الإجابة بنقاط', 'هل هناك مواعيد محددة؟', 'أعطني مثالاً'];

const STORAGE_KEY = 'murshid_history';
const MAX_CONVERSATIONS = 3;
const MAX_AGE_MS = 3 * 24 * 60 * 60 * 1000;
const HISTORY_TURNS = 3;
const FS_MIN = 0.85;
const FS_MAX = 1.4;
const FS_STEP = 0.15;
const CONNECTION_ERROR = 'تعذّر الاتصال بالخادم، تحقّق من تشغيله ثم أعد المحاولة.';

const ICONS = {
  bot: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.6" /><path d="M15.6 8.4 13 13l-4.6 2.6L11 11l4.6-2.6Z" fill="currentColor" /></svg>',
  user: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none"><circle cx="12" cy="8.5" r="3.2" stroke="currentColor" stroke-width="1.6" /><path d="M5.5 19c.7-3.3 3.3-5 6.5-5s5.8 1.7 6.5 5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>',
  check: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><path d="M5 12.5 10 17.5 19 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  copy: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2.5" stroke="currentColor" stroke-width="1.6" /><path d="M5 15V6.5A2.5 2.5 0 0 1 7.5 4H15" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>',
  stop: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" /></svg>',
  speak: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><path d="M4 9.5v5h3.5L12 18V6L7.5 9.5H4Z" fill="currentColor" /><path d="M15.5 9a3.5 3.5 0 0 1 0 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>',
  regen: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><path d="M19 12a7 7 0 1 1-2-4.9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /><path d="M17 3v4.5h-4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  thumb: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><path d="M7 11v8H5a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /><path d="M7 11l3.4-6.4A1.7 1.7 0 0 1 13.8 5.4V9H18a2 2 0 0 1 2 2.3l-1 5.4A2 2 0 0 1 17 19H7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  ask: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M9.2 9.3a2.8 2.8 0 1 1 4 2.5c-.9.5-1.2 1-1.2 1.9M12 17h.01" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  go: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true"><path d="M14 7l-5 5 5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  plus: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /></svg>',
  close: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" /></svg>',
  tick: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none"><path d="M5 12.5 10 17.5 19 7" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" /></svg>',
  sun: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4" stroke="currentColor" stroke-width="1.6" /><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>',
  moon: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4 7 7 0 1 0 20 14.5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" /></svg>',
  error: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" aria-hidden="true"><path d="M19 12a7 7 0 1 1-2-4.9" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" /><path d="M17 3v4.5h-4.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>',
};

// ===== الحالة =====
const state = {
  selected: UNIVERSITIES[0],
  messages: [],
  loading: false,
  phase: '',
  history: [],
  currentId: null,
  corpus: {},
  model: '',
  online: null,
  speakingId: null,
  listening: false,
  dark: false,
  autoRead: false,
  fontScale: 1,
  copiedId: null,
};

let reqId = 0;
let lastScrollKey = '';
let recognition = null;
const ttsSupported = 'speechSynthesis' in window;
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

const store = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* تجاهل */ }
  },
};

// ===== أدوات صغيرة =====
const uid = () => Date.now() + Math.floor(Math.random() * 100000);
const time = () => new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' });
const uniById = (id) => UNIVERSITIES.find((u) => u.id === id) || UNIVERSITIES[0];

function relativeTime(ts) {
  const min = Math.floor((Date.now() - ts) / 60000);
  if (min < 1) return 'الآن';
  if (min < 60) return 'قبل ' + min + ' د';
  const hr = Math.floor(min / 60);
  if (hr < 24) return 'قبل ' + hr + ' س';
  return 'قبل ' + Math.floor(hr / 24) + ' ي';
}

// النص المقروء أو المنسوخ: بلا ** ولا أرقام الاستشهاد ولا خطوط الجداول
function plainText(text) {
  return String(text || '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\s*\[\d+\]/g, '')
    .replace(/^\s*\|?[\s\-:|]+\|?\s*$/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ===== تنسيق الإجابة: خط عريض، أرقام الاستشهاد، جداول Markdown =====
function inline(s, n) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, (m, i) =>
      +i >= 1 && +i <= n ? `<span class="cite" data-n="${i}" title="المصدر ${i}">${i}</span>` : m,
    );
}

function formatAnswer(text, n) {
  const out = [];
  const lines = String(text || '').trim().split('\n');
  const cells = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim().startsWith('|')) {
      out.push(inline(lines[i], n));
      continue;
    }
    const rows = [];
    while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++]);
    i--;
    const body = rows.filter((r) => !/^\s*\|?[\s\-:|]+\|?\s*$/.test(r)).map(cells);
    const [head, ...rest] = body;
    if (!head) continue;
    out.push(
      `<div class="tbl"><table><thead><tr>${head.map((c) => `<th>${inline(c, n)}</th>`).join('')}</tr></thead>` +
        `<tbody>${rest.map((r) => `<tr>${r.map((c) => `<td>${inline(c, n)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`,
    );
  }
  return out.join('\n').replace(/\n?(<div class="tbl">)/g, '$1').replace(/(<\/div>)\n/g, '$1');
}

// ===== الاتصال بالخادم =====
async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch { /* رد غير JSON */ }
  if (!data) throw new Error('bad response');
  return data;
}

// آخر الأدوار المكتملة قبل الموضع المعطى، بالشكل الذي يقبله app.py
function historyBefore(list, index) {
  const turns = [];
  for (let i = 0; i < index; i++) {
    const m = list[i];
    if (m.role !== 'bot' || m.isError || m.greeting || m.info) continue;
    const q = list.slice(0, i).reverse().find((x) => x.role === 'user');
    if (q) turns.push({ question: q.text, answer: m.text });
  }
  return turns.slice(-HISTORY_TURNS);
}

function payload(question, history) {
  return {
    question,
    university: state.selected.api,
    history,
  };
}

// يرجع رسالة البوت لسؤال؛ يمر بالبحث ثم الإجابة
async function ask(question, history, id) {
  const p = payload(question, history);
  setPhase('يبحث في اللوائح…');
  const s = await post('/api/search', p);
  if (id !== reqId) return null;
  if (s.error) return { text: String(s.error), info: true };
  if (!s.sources || !s.sources.length)
    return { text: 'ما لقيت مقاطع مطابقة في لوائح الجامعة.', info: true };

  setPhase('يكتب الإجابة…');
  const d = await post('/api/answer', p);
  if (id !== reqId) return null;
  if (d.error) return { text: d.answer || CONNECTION_ERROR, isError: true };

  const notes = [];
  if (s.rerank_failed) notes.push('تعذّر الترتيب الدقيق للمصادر، فهي مرتبة بدرجة البحث.');
  if (d.model_used && state.model && d.model_used !== state.model)
    notes.push(`أجاب نموذج احتياطي (${d.model_used}) لأن رصيد النموذج الأساسي لا يكفي.`);
  return {
    text: d.answer,
    sources: d.sources && d.sources.length ? d.sources : s.sources,
    cited: d.cited || [],
    grounded: !!d.grounded,
    refused: !!d.refused,
    greeting: !!d.greeting,
    notes,
  };
}

function botMessage(result) {
  return { id: uid(), role: 'bot', time: time(), feedback: null, ...result };
}

async function send(text) {
  const question = (text ?? $('q').value).trim();
  if (!question || state.loading) return;
  stopSpeak();
  const history = historyBefore(state.messages, state.messages.length);
  state.messages.push({ id: uid(), role: 'user', text: question, time: time() });
  $('q').value = '';
  syncConversation();
  await runAsk(question, history, (msg) => state.messages.push(msg));
}

// يشغّل السؤال ويضع الرد بالدالة place؛ عند الخطأ رسالة خطأ مع زر إعادة المحاولة
async function runAsk(question, history, place, keepOnError = false) {
  const id = ++reqId;
  state.loading = true;
  render();
  let msg = null;
  try {
    const result = await ask(question, history, id);
    if (result) msg = botMessage(result);
  } catch {
    if (id === reqId && !keepOnError) msg = botMessage({ text: CONNECTION_ERROR, isError: true });
  }
  if (id !== reqId) return;
  state.loading = false;
  state.phase = '';
  if (msg) {
    place(msg);
    if (state.autoRead && ttsSupported && !msg.isError) speak(msg);
  }
  syncConversation();
  render();
}

function lastUserIndex(before) {
  for (let i = before - 1; i >= 0; i--) if (state.messages[i].role === 'user') return i;
  return -1;
}

function retryLast() {
  if (state.loading) return;
  const list = state.messages;
  if (list.length && list[list.length - 1].role === 'bot') list.pop();
  const qi = lastUserIndex(list.length);
  if (qi < 0) return render();
  runAsk(list[qi].text, historyBefore(list, qi), (msg) => state.messages.push(msg));
}

function regenerate(id) {
  if (state.loading) return;
  const idx = state.messages.findIndex((x) => x.id === id);
  const qi = lastUserIndex(idx);
  if (idx < 0 || qi < 0) return;
  stopSpeak();
  runAsk(
    state.messages[qi].text,
    historyBefore(state.messages, qi),
    (msg) => {
      const at = state.messages.findIndex((x) => x.id === id);
      if (at >= 0) state.messages[at] = { ...msg, id };
    },
    true,
  );
}

function setPhase(text) {
  state.phase = text;
  const el = document.querySelector('.bubble.phase .phase-text');
  if (el) el.textContent = text;
}

// ===== المحادثات المحفوظة =====
function makeTitle(msgs) {
  const firstUser = msgs.find((m) => m.role === 'user');
  const t = (firstUser ? firstUser.text : 'محادثة جديدة').trim();
  return t.length > 34 ? t.slice(0, 34) + '…' : t;
}

function syncConversation() {
  const msgs = state.messages;
  if (!msgs.length) return;
  const now = Date.now();
  const title = makeTitle(msgs);
  if (!state.currentId) {
    state.currentId = 'c_' + now + '_' + Math.random().toString(36).slice(2, 7);
    state.history = [
      { id: state.currentId, uni: state.selected.id, title, messages: [...msgs], createdAt: now, updatedAt: now },
      ...state.history,
    ].slice(0, MAX_CONVERSATIONS);
  } else {
    state.history = state.history.map((c) =>
      c.id === state.currentId ? { ...c, title, messages: [...msgs], updatedAt: now } : c,
    );
  }
  saveHistory();
}

function loadHistory() {
  try {
    const parsed = JSON.parse(store.get(STORAGE_KEY) || '[]');
    const now = Date.now();
    state.history = parsed.filter((c) => now - c.updatedAt < MAX_AGE_MS).slice(0, MAX_CONVERSATIONS);
  } catch {
    state.history = [];
  }
}

function saveHistory() {
  store.set(STORAGE_KEY, JSON.stringify(state.history));
}

function openConversation(id) {
  const c = state.history.find((x) => x.id === id);
  if (!c) return;
  cancelPending();
  setUniversity(uniById(c.uni), false);
  state.messages = [...c.messages];
  state.currentId = c.id;
  $('q').value = '';
  render();
}

function deleteConversation(id) {
  state.history = state.history.filter((c) => c.id !== id);
  saveHistory();
  if (state.currentId === id) {
    cancelPending();
    state.messages = [];
    state.currentId = null;
  }
  render();
}

function clearChat() {
  cancelPending();
  state.messages = [];
  state.currentId = null;
  $('q').value = '';
  render();
  $('q').focus();
}

// إلغاء أي طلب جارٍ حتى لا يصل رده لمحادثة أخرى
function cancelPending() {
  stopSpeak();
  reqId++;
  state.loading = false;
  state.phase = '';
}

// ===== الجامعة =====
function setUniversity(u, reset = true) {
  state.selected = u;
  store.set('uni', u.id);
  if (reset) {
    cancelPending();
    state.messages = [];
    state.currentId = null;
    $('q').value = '';
  }
  renderStatic();
}

function selectUniversity(id) {
  if (id === state.selected.id) return;
  setUniversity(uniById(id));
  render();
}

function updateStatus() {
  const c = state.corpus[state.selected.api];
  const box = $('ctxStatus');
  box.classList.toggle('down', state.online === false);
  if (state.online === false) $('statText').textContent = 'تعذّر الاتصال بالخادم';
  else if (c) $('statText').textContent = `متصل · ${c.docs.length} وثائق · ${c.chunks} مقطع`;
  else $('statText').textContent = 'جارِ الاتصال…';
}

// ===== الرسم =====
// الأجزاء التي لا تتغير إلا بتغيير الجامعة أو التفضيلات
function renderStatic() {
  const u = state.selected;
  const shell = $('shell');
  shell.dataset.theme = u.id;
  shell.classList.toggle('dark', state.dark);
  shell.style.setProperty('--fs', state.fontScale);

  $('uniList').innerHTML = UNIVERSITIES.map(
    (x) => `
    <button class="uni-card${x.id === u.id ? ' on' : ''}" data-uni="${x.id}" aria-pressed="${x.id === u.id}">
      <span class="uni-mark" data-uni="${x.id}" aria-hidden="true"><img src="${x.logo}" alt="" /></span>
      <span class="uni-meta">
        <span class="uni-name">${esc(x.name)}</span>
        <span class="uni-en">${esc(x.en)}</span>
      </span>
      <span class="uni-check" aria-hidden="true">${ICONS.tick}</span>
    </button>`,
  ).join('');

  $('topics').innerHTML = SUGGESTIONS[u.id]
    .map(
      (s) => `
    <button class="topic" data-q="${esc(s)}">
      <span class="topic-dot" aria-hidden="true"></span>
      <span class="topic-text">${esc(s)}</span>
      <span class="topic-go" aria-hidden="true">${ICONS.go}</span>
    </button>`,
    )
    .join('');

  $('ctxLogo').src = u.logo;
  $('ctxLogo').parentElement.dataset.uni = u.id;
  $('ctxName').textContent = u.name;
  updateStatus();

  $('fsSmall').disabled = state.fontScale <= FS_MIN;
  $('fsBig').disabled = state.fontScale >= FS_MAX;
  $('autoRead').disabled = !ttsSupported;
  $('autoRead').classList.toggle('on', state.autoRead);
  $('autoRead').setAttribute('aria-pressed', state.autoRead);
  $('darkBtn').classList.toggle('on', state.dark);
  $('darkBtn').setAttribute('aria-pressed', state.dark);
  $('darkBtn').setAttribute('aria-label', state.dark ? 'الوضع النهاري' : 'الوضع الليلي');
  $('darkBtn').innerHTML = state.dark ? ICONS.sun : ICONS.moon;
}

function renderHistory() {
  $('histCount').textContent = `${state.history.length}/${MAX_CONVERSATIONS}`;
  if (!state.history.length) {
    $('histBox').innerHTML = '<p class="hist-empty">ما عندك محادثات محفوظة بعد</p>';
    return;
  }
  $('histBox').innerHTML =
    '<div class="hist-list">' +
    state.history
      .map(
        (c) => `
      <button class="hist${state.currentId === c.id ? ' on' : ''}" data-conv="${esc(c.id)}" aria-label="${esc('فتح محادثة: ' + c.title)}">
        <span class="hist-dot" data-uni="${esc(c.uni)}" aria-hidden="true"></span>
        <span class="hist-meta">
          <span class="hist-title">${esc(c.title)}</span>
          <span class="hist-sub">${esc(uniById(c.uni).short)} · ${relativeTime(c.updatedAt)}</span>
        </span>
        <span class="hist-del" role="button" tabindex="0" data-del="${esc(c.id)}" aria-label="حذف المحادثة">${ICONS.close}</span>
      </button>`,
      )
      .join('') +
    '</div>';
}

function emptyView() {
  const u = state.selected;
  return `
    <div class="empty">
      <div class="empty-mark" aria-hidden="true">${ICONS.bot.replace(/17/g, '30')}</div>
      <h2>كيف أقدر أساعدك اليوم؟</h2>
      <p>اسألني عن لوائح <strong>${esc(u.name)}</strong> وبجاوبك من نصوصها الرسمية مع المصادر.</p>
      <div class="prompts">
        ${SUGGESTIONS[u.id]
          .map(
            (s, i) => `
          <button class="prompt" data-q="${esc(s)}" style="animation-delay:${i * 60}ms">
            <span class="prompt-ic" aria-hidden="true">${ICONS.ask}</span>
            <span class="prompt-text">${esc(s)}</span>
            <span class="prompt-go" aria-hidden="true">${ICONS.go}</span>
          </button>`,
          )
          .join('')}
      </div>
    </div>`;
}

function sourcesView(m) {
  if (!m.sources || !m.sources.length || m.greeting) return '';
  const cited = new Set(m.cited || []);
  const items = m.sources
    .map(
      (s) => `
      <details class="src${cited.has(s.n) ? ' cited' : ''}" data-src="${m.id}-${s.n}">
        <summary>
          <span class="num">${s.n}</span>
          <span>
            <div class="t">${esc(s.doc_title)}${s.label ? ' — ' + esc(s.label) : ''}</div>
            <div class="meta"><span class="id">${esc(s.chunk_id)}</span>${
              s.source_url ? `<a href="${esc(s.source_url)}" target="_blank" rel="noopener">الصفحة الرسمية</a>` : ''
            }</div>
          </span>
        </summary>
        <div class="body">${esc(s.body)}</div>
      </details>`,
    )
    .join('');
  const label = cited.size ? `المصادر (${m.sources.length}) · استُشهد بـ ${cited.size}` : `المصادر (${m.sources.length})`;
  return `<details class="srcs" data-srcs="${m.id}"><summary>${label}</summary><div class="src-list">${items}</div></details>`;
}

function botExtras(m) {
  let html = '';
  if (!m.isError && !m.info && !m.greeting && !m.refused && m.grounded === false)
    html += '<div class="flag">لم يستشهد النموذج بأي مقطع — تحقّق من الإجابة بنفسك قبل الاعتماد عليها.</div>';
  for (const note of m.notes || []) html += `<div class="note">${esc(note)}</div>`;
  return html + sourcesView(m);
}

function actionsView(m) {
  const copied = state.copiedId === m.id;
  const speaking = state.speakingId === m.id;
  return `
    <div class="actions" role="group" aria-label="خيارات الإجابة">
      <button class="act${copied ? ' ok' : ''}" data-act="copy" data-id="${m.id}" aria-label="${copied ? 'تم النسخ' : 'نسخ الإجابة'}">${copied ? ICONS.check : ICONS.copy}</button>
      <button class="act${speaking ? ' on' : ''}" data-act="speak" data-id="${m.id}" ${ttsSupported ? '' : 'disabled'} aria-pressed="${speaking}" aria-label="${speaking ? 'إيقاف القراءة' : 'استماع للإجابة'}">${speaking ? ICONS.stop : ICONS.speak}</button>
      <button class="act" data-act="regen" data-id="${m.id}" ${state.loading ? 'disabled' : ''} aria-label="إعادة توليد الإجابة">${ICONS.regen}</button>
      <button class="act${m.feedback === 'up' ? ' up' : ''}" data-act="up" data-id="${m.id}" aria-pressed="${m.feedback === 'up'}" aria-label="الإجابة ساعدتني">${ICONS.thumb}</button>
      <button class="act dis${m.feedback === 'down' ? ' down' : ''}" data-act="down" data-id="${m.id}" aria-pressed="${m.feedback === 'down'}" aria-label="الإجابة ما نفعتني">${ICONS.thumb}</button>
    </div>`;
}

function messageView(m) {
  const bot = m.role === 'bot';
  let text;
  if (bot && !m.isError && !m.info) text = formatAnswer(m.text, (m.sources || []).length);
  else text = esc(m.text);
  const cls = ['bubble', m.isError ? 'error' : '', m.refused ? 'refusal' : ''].filter(Boolean).join(' ');
  return `
    <div class="row${bot ? '' : ' me'}" data-msg="${m.id}">
      <div class="avatar${bot ? ' bot' : ''}" aria-hidden="true">${bot ? ICONS.bot : ICONS.user}</div>
      <div class="msg">
        <span class="who">${bot ? 'مُرشِد' : 'أنت'}</span>
        <div class="${cls}"><div class="text">${text}</div></div>
        ${m.isError ? `<button class="retry-btn" data-act="retry">${ICONS.error} أعد المحاولة</button>` : ''}
        ${bot ? botExtras(m) : ''}
        ${bot && !m.isError && !m.info ? actionsView(m) : ''}
        <span class="time">${esc(m.time)}</span>
      </div>
    </div>`;
}

function loadingView() {
  return `
    <div class="row">
      <div class="avatar bot" aria-hidden="true">${ICONS.bot}</div>
      <div class="msg">
        <span class="who">مُرشِد</span>
        <div class="bubble phase" aria-label="يكتب الآن">
          <span class="dots"><span></span><span></span><span></span></span>
          <span class="phase-text">${esc(state.phase || 'يبحث في اللوائح…')}</span>
        </div>
      </div>
    </div>`;
}

function render() {
  const thread = $('thread');
  // نحفظ المصادر المفتوحة حتى لا تنطوي عند إعادة الرسم
  const open = [...thread.querySelectorAll('details[open]')].map((d) => d.dataset.src || 'list-' + d.dataset.srcs);
  const msgs = state.messages;
  let html = msgs.length ? msgs.map(messageView).join('') : emptyView();
  if (state.loading) html += loadingView();
  const last = msgs[msgs.length - 1];
  if (!state.loading && last && last.role === 'bot' && !last.isError && !last.info && !last.greeting) {
    html += `<div class="followups" aria-label="أسئلة مقترحة">${FOLLOWUPS.map(
      (f) => `<button class="followup" data-q="${esc(f)}">${ICONS.plus} ${esc(f)}</button>`,
    ).join('')}</div>`;
  }
  thread.innerHTML = html;
  for (const d of thread.querySelectorAll('details')) {
    const key = d.dataset.src || 'list-' + d.dataset.srcs;
    if (open.includes(key)) d.open = true;
  }
  renderHistory();
  $('newChat').disabled = !msgs.length;
  $('q').disabled = state.loading;
  updateSend();
  // ننزل لآخر المحادثة فقط إذا وصلت رسالة جديدة أو بدأ التحميل، لا عند النسخ أو التقييم
  const key = `${state.currentId}:${msgs.length}:${state.loading}`;
  if (key !== lastScrollKey) thread.scrollTop = thread.scrollHeight;
  lastScrollKey = key;
}

function updateSend() {
  $('send').disabled = state.loading || !$('q').value.trim();
  $('mic').disabled = !SpeechRec;
  $('mic').classList.toggle('on', state.listening);
  $('mic').setAttribute('aria-pressed', state.listening);
  $('mic').title = SpeechRec ? 'سؤال بالصوت' : 'الإدخال الصوتي غير مدعوم في هذا المتصفّح';
  const hint = $('hint');
  hint.classList.toggle('listening', state.listening);
  hint.innerHTML = state.listening
    ? '<span class="listen-dot" aria-hidden="true"></span> يتم الاستماع، تكلّم الآن...'
    : 'اضغط Enter للإرسال · الإجابات مولّدة آلياً من نصوص اللوائح — راجع المصدر قبل الاعتماد عليها';
}

// ===== النسخ =====
function copy(m) {
  const text = plainText(m.text);
  const done = () => {
    state.copiedId = m.id;
    render();
    setTimeout(() => {
      if (state.copiedId === m.id) {
        state.copiedId = null;
        render();
      }
    }, 1500);
  };
  const fallback = () => {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      done();
    } catch { /* تجاهل */ }
  };
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(done).catch(fallback);
  else fallback();
}

// ===== الصوت =====
function speak(m) {
  if (!ttsSupported) return;
  if (state.speakingId === m.id) return stopSpeak();
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(plainText(m.text));
  u.lang = 'ar-SA';
  const arVoice = window.speechSynthesis.getVoices().find((v) => (v.lang || '').toLowerCase().startsWith('ar'));
  if (arVoice) u.voice = arVoice;
  const end = () => {
    if (state.speakingId === m.id) {
      state.speakingId = null;
      render();
    }
  };
  u.onend = end;
  u.onerror = end;
  state.speakingId = m.id;
  window.speechSynthesis.speak(u);
  render();
}

function stopSpeak() {
  if (ttsSupported) window.speechSynthesis.cancel();
  if (state.speakingId !== null) {
    state.speakingId = null;
    if (!state.loading) render();
  }
}

function toggleMic() {
  if (state.listening) {
    try { recognition && recognition.stop(); } catch { /* تجاهل */ }
    state.listening = false;
    return updateSend();
  }
  if (!SpeechRec) return;
  const rec = new SpeechRec();
  rec.lang = 'ar-SA';
  rec.interimResults = true;
  rec.continuous = false;
  rec.onresult = (e) => {
    let txt = '';
    for (let i = 0; i < e.results.length; i++) txt += e.results[i][0].transcript;
    $('q').value = txt;
    updateSend();
  };
  rec.onend = rec.onerror = () => {
    state.listening = false;
    updateSend();
  };
  recognition = rec;
  state.listening = true;
  rec.start();
  updateSend();
}

// ===== التقييم =====
function rate(m, value) {
  m.feedback = m.feedback === value ? null : value;
  syncConversation();
  render();
  const idx = state.messages.indexOf(m);
  const qi = lastUserIndex(idx);
  fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      university: state.selected.api,
      question: qi >= 0 ? state.messages[qi].text : '',
      answer: m.text,
      feedback: m.feedback,
    }),
  }).catch(() => {});
}

// ===== التفضيلات =====
function loadPrefs() {
  state.dark = store.get('murshid_dark') === '1';
  state.autoRead = store.get('murshid_autoread') === '1';
  const fs = parseFloat(store.get('murshid_fs') || '1');
  if (!isNaN(fs)) state.fontScale = Math.min(FS_MAX, Math.max(FS_MIN, fs));
  const saved = store.get('uni');
  if (saved) state.selected = uniById(saved.toLowerCase());
}

function savePrefs() {
  store.set('murshid_dark', state.dark ? '1' : '0');
  store.set('murshid_autoread', state.autoRead ? '1' : '0');
  store.set('murshid_fs', String(state.fontScale));
  renderStatic();
}

// ===== الأحداث =====
function findMessage(id) {
  return state.messages.find((x) => String(x.id) === String(id));
}

function openSource(row, n) {
  const list = row.querySelector('details.srcs');
  const src = row.querySelector(`details.src[data-src$="-${n}"]`);
  if (!list || !src) return;
  list.open = true;
  src.open = true;
  src.scrollIntoView({ behavior: 'smooth', block: 'center' });
  src.classList.add('flash');
  setTimeout(() => src.classList.remove('flash'), 1400);
}

$('thread').addEventListener('click', (e) => {
  const cite = e.target.closest('.cite');
  if (cite) return openSource(cite.closest('.row'), cite.dataset.n);
  const q = e.target.closest('[data-q]');
  if (q) return send(q.dataset.q);
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const act = btn.dataset.act;
  if (act === 'retry') return retryLast();
  const m = findMessage(btn.dataset.id);
  if (!m) return;
  if (act === 'copy') copy(m);
  else if (act === 'speak') speak(m);
  else if (act === 'regen') regenerate(m.id);
  else if (act === 'up' || act === 'down') rate(m, act);
});

$('uniList').addEventListener('click', (e) => {
  const b = e.target.closest('[data-uni]');
  if (b) selectUniversity(b.dataset.uni);
});

$('topics').addEventListener('click', (e) => {
  const b = e.target.closest('[data-q]');
  if (b) send(b.dataset.q);
});

$('histBox').addEventListener('click', (e) => {
  const del = e.target.closest('[data-del]');
  if (del) {
    e.stopPropagation();
    return deleteConversation(del.dataset.del);
  }
  const c = e.target.closest('[data-conv]');
  if (c) openConversation(c.dataset.conv);
});

$('histBox').addEventListener('keydown', (e) => {
  const del = e.target.closest('[data-del]');
  if (del && e.key === 'Enter') {
    e.preventDefault();
    e.stopPropagation();
    deleteConversation(del.dataset.del);
  }
});

$('q').addEventListener('input', updateSend);
$('q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
$('send').onclick = () => send();
$('mic').onclick = toggleMic;
$('newChat').onclick = clearChat;
$('fsSmall').onclick = () => {
  state.fontScale = Math.max(FS_MIN, +(state.fontScale - FS_STEP).toFixed(2));
  savePrefs();
};
$('fsBig').onclick = () => {
  state.fontScale = Math.min(FS_MAX, +(state.fontScale + FS_STEP).toFixed(2));
  savePrefs();
};
$('autoRead').onclick = () => {
  state.autoRead = !state.autoRead;
  if (!state.autoRead) stopSpeak();
  savePrefs();
};
$('darkBtn').onclick = () => {
  state.dark = !state.dark;
  savePrefs();
};

// ===== البدء =====
loadPrefs();
loadHistory();
renderStatic();
render();

fetch('/api/health')
  .then((r) => r.json())
  .then((d) => {
    state.model = d.model;
    state.online = true;
    updateStatus();
  })
  .catch(() => {
    state.online = false;
    updateStatus();
  });

fetch('/api/corpus')
  .then((r) => r.json())
  .then((d) => {
    state.corpus = d;
    updateStatus();
  })
  .catch(() => {});

// تحديث «قبل كذا دقيقة» في قائمة المحادثات
setInterval(renderHistory, 60000);
