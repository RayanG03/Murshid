import { Component, signal, computed, effect, ElementRef, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ChatMessage {
  id: number;
  role: 'user' | 'bot';
  text: string;
  time: string;
  feedback?: 'up' | 'down' | null;
  isError?: boolean;
}

interface University {
  id: 'ksu' | 'kau';
  name: string;
  short: string;
  logo: string;
}

interface Conversation {
  id: string;
  uni: 'ksu' | 'kau';
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'murshid_history';
const MAX_CONVERSATIONS = 3;
const MAX_AGE_MS = 3 * 24 * 60 * 60 * 1000;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly universities: University[] = [
    { id: 'ksu', name: 'جامعة الملك سعود', short: 'الملك سعود', logo: '🟦' },
    { id: 'kau', name: 'جامعة الملك عبدالعزيز', short: 'الملك عبدالعزيز', logo: '🟩' },
  ];

  readonly selected = signal<University>(this.universities[0]);
  readonly messages = signal<ChatMessage[]>([]);
  readonly draft = signal('');
  readonly loading = signal(false);

  readonly history = signal<Conversation[]>([]);
  readonly currentId = signal<string | null>(null);

  // الصوت
  readonly speakingId = signal<number | null>(null);
  readonly listening = signal(false);
  readonly ttsSupported = signal(typeof window !== 'undefined' && 'speechSynthesis' in window);
  readonly micSupported = signal(
    typeof window !== 'undefined' &&
      !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition),
  );

  // التفضيلات
  readonly dark = signal(false);
  readonly autoRead = signal(false);
  readonly fontScale = signal(1);
  readonly copiedId = signal<number | null>(null);

  private readonly FS_MIN = 0.85;
  private readonly FS_MAX = 1.4;
  private readonly FS_STEP = 0.15;

  private recognition: any = null;

  // أسئلة متابعة ثابتة تظهر بعد كل رد
  readonly followups = ['اشرح لي أكثر', 'لخّص الإجابة بنقاط', 'هل هناك مواعيد محددة؟', 'أعطني مثالاً'];

  readonly suggestions = computed(() =>
    this.selected().id === 'ksu'
      ? [
          'ما هي شروط القبول في كلية علوم الحاسب؟',
          'متى يبدأ التسجيل للفصل الدراسي الأول؟',
          'كيف أحسب معدلي التراكمي؟',
          'ما هي مواعيد الاختبارات النهائية؟',
        ]
      : [
          'ما هي التخصصات المتاحة في كلية الحاسبات؟',
          'كيف أقدّم على المنح الدراسية؟',
          'ما هي إجراءات تأجيل الفصل الدراسي؟',
          'أين أجد الجدول الدراسي؟',
        ],
  );

  readonly lastIsBot = computed(() => {
    const m = this.messages();
    return m.length > 0 && m[m.length - 1].role === 'bot' && !m[m.length - 1].isError;
  });

  private readonly scrollBox = viewChild<ElementRef<HTMLDivElement>>('scrollBox');

  constructor() {
    this.loadPrefs();
    this.loadHistory();

    effect(() => {
      this.messages();
      this.loading();
      queueMicrotask(() => {
        const el = this.scrollBox()?.nativeElement;
        if (el) el.scrollTop = el.scrollHeight;
      });
    });

    effect(() => {
      this.saveHistory(this.history());
    });
  }

  selectUniversity(u: University): void {
    if (u.id === this.selected().id) return;
    this.stopSpeak();
    this.selected.set(u);
    this.messages.set([]);
    this.draft.set('');
    this.currentId.set(null);
  }

  useSuggestion(text: string): void {
    this.draft.set(text);
    this.send();
  }

  openConversation(c: Conversation): void {
    this.stopSpeak();
    const uni = this.universities.find((u) => u.id === c.uni);
    if (uni) this.selected.set(uni);
    this.messages.set([...c.messages]);
    this.currentId.set(c.id);
    this.draft.set('');
  }

  deleteConversation(id: string, ev: Event): void {
    ev.stopPropagation();
    this.history.update((list) => list.filter((c) => c.id !== id));
    if (this.currentId() === id) {
      this.messages.set([]);
      this.currentId.set(null);
    }
  }

  clearChat(): void {
    this.stopSpeak();
    this.messages.set([]);
    this.currentId.set(null);
    this.draft.set('');
  }

  async send(): Promise<void> {
    const question = this.draft().trim();
    if (!question || this.loading()) return;

    this.stopSpeak();
    this.messages.update((m) => [
      ...m,
      { id: this.uid(), role: 'user', text: question, time: this.time(), feedback: null },
    ]);
    this.draft.set('');
    this.loading.set(true);
    this.syncConversation();

    try {
      const answer = await this.ask(question);
      const botMsg: ChatMessage = {
        id: this.uid(),
        role: 'bot',
        text: answer,
        time: this.time(),
        feedback: null,
      };
      this.messages.update((m) => [...m, botMsg]);
      if (this.autoRead() && this.ttsSupported()) this.speak(botMsg);
    } catch {
      this.messages.update((m) => [
        ...m,
        {
          id: this.uid(),
          role: 'bot',
          text: 'تعذّر الاتصال بالخادم، تحقّق من تشغيله ثم أعد المحاولة.',
          time: this.time(),
          feedback: null,
          isError: true,
        },
      ]);
    } finally {
      this.loading.set(false);
      this.syncConversation();
    }
  }

  // إعادة المحاولة بعد خطأ
  async retryLast(): Promise<void> {
    if (this.loading()) return;
    const list = this.messages();
    let q = '';
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === 'user') {
        q = list[i].text;
        break;
      }
    }
    if (!q) return;

    this.messages.update((l) =>
      l.length && l[l.length - 1].role === 'bot' ? l.slice(0, -1) : l,
    );
    this.loading.set(true);
    try {
      const answer = await this.ask(q);
      const botMsg: ChatMessage = {
        id: this.uid(),
        role: 'bot',
        text: answer,
        time: this.time(),
        feedback: null,
      };
      this.messages.update((m) => [...m, botMsg]);
      if (this.autoRead() && this.ttsSupported()) this.speak(botMsg);
    } catch {
      this.messages.update((m) => [
        ...m,
        {
          id: this.uid(),
          role: 'bot',
          text: 'تعذّر الاتصال بالخادم، تحقّق من تشغيله ثم أعد المحاولة.',
          time: this.time(),
          feedback: null,
          isError: true,
        },
      ]);
    } finally {
      this.loading.set(false);
      this.syncConversation();
    }
  }

  // إعادة توليد إجابة معيّنة
  async regenerate(m: ChatMessage): Promise<void> {
    if (this.loading()) return;
    const list = this.messages();
    const idx = list.findIndex((x) => x.id === m.id);
    if (idx < 0) return;

    let q = '';
    for (let i = idx - 1; i >= 0; i--) {
      if (list[i].role === 'user') {
        q = list[i].text;
        break;
      }
    }
    if (!q) return;

    this.stopSpeak();
    this.loading.set(true);
    try {
      const answer = await this.ask(q);
      this.messages.update((l) =>
        l.map((x) =>
          x.id === m.id ? { ...x, text: answer, time: this.time(), feedback: null } : x,
        ),
      );
    } catch {
      // نُبقي الإجابة القديمة عند الفشل
    } finally {
      this.loading.set(false);
      this.syncConversation();
    }
  }

  // نسخ الإجابة
  copy(m: ChatMessage): void {
    const done = () => {
      this.copiedId.set(m.id);
      setTimeout(() => {
        if (this.copiedId() === m.id) this.copiedId.set(null);
      }, 1500);
    };
    try {
      navigator.clipboard.writeText(m.text).then(done).catch(() => this.fallbackCopy(m.text, done));
    } catch {
      this.fallbackCopy(m.text, done);
    }
  }

  private fallbackCopy(text: string, done: () => void): void {
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
    } catch {
      // تجاهل
    }
  }

  // ===== الصوت =====
  speak(m: ChatMessage): void {
    if (!this.ttsSupported()) return;
    if (this.speakingId() === m.id) {
      this.stopSpeak();
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(m.text);
    u.lang = 'ar-SA';
    u.rate = 1;
    const arVoice = window.speechSynthesis
      .getVoices()
      .find((v) => (v.lang || '').toLowerCase().startsWith('ar'));
    if (arVoice) u.voice = arVoice;
    u.onend = () => this.speakingId.set(null);
    u.onerror = () => this.speakingId.set(null);
    this.speakingId.set(m.id);
    window.speechSynthesis.speak(u);
  }

  stopSpeak(): void {
    if (this.ttsSupported()) window.speechSynthesis.cancel();
    this.speakingId.set(null);
  }

  toggleMic(): void {
    if (this.listening()) {
      this.stopMic();
      return;
    }
    this.startMic();
  }

  private startMic(): void {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) {
      this.micSupported.set(false);
      return;
    }
    const rec = new SR();
    rec.lang = 'ar-SA';
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = (e: any) => {
      let txt = '';
      for (let i = 0; i < e.results.length; i++) {
        txt += e.results[i][0].transcript;
      }
      this.draft.set(txt);
    };
    rec.onend = () => this.listening.set(false);
    rec.onerror = () => this.listening.set(false);
    this.recognition = rec;
    this.listening.set(true);
    rec.start();
  }

  private stopMic(): void {
    try {
      this.recognition?.stop();
    } catch {
      // تجاهل
    }
    this.listening.set(false);
  }

  // ===== التقييم =====
  rate(m: ChatMessage, value: 'up' | 'down'): void {
    const next = m.feedback === value ? null : value;
    this.messages.update((list) =>
      list.map((x) => (x.id === m.id ? { ...x, feedback: next } : x)),
    );
    this.syncConversation();
    this.postFeedback(m, next);
  }

  private postFeedback(m: ChatMessage, value: 'up' | 'down' | null): void {
    const url = 'http://localhost:8000/feedback';
    try {
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ university: this.selected().id, answer: m.text, feedback: value }),
      }).catch(() => {});
    } catch {
      // تجاهل
    }
  }

  // ===== التفضيلات =====
  toggleDark(): void {
    this.dark.set(!this.dark());
    this.savePref();
  }

  toggleAutoRead(): void {
    const next = !this.autoRead();
    this.autoRead.set(next);
    if (!next) this.stopSpeak();
    this.savePref();
  }

  biggerFont(): void {
    this.fontScale.set(Math.min(this.FS_MAX, +(this.fontScale() + this.FS_STEP).toFixed(2)));
    this.savePref();
  }

  smallerFont(): void {
    this.fontScale.set(Math.max(this.FS_MIN, +(this.fontScale() - this.FS_STEP).toFixed(2)));
    this.savePref();
  }

  private loadPrefs(): void {
    if (typeof localStorage === 'undefined') return;
    try {
      this.dark.set(localStorage.getItem('murshid_dark') === '1');
      this.autoRead.set(localStorage.getItem('murshid_autoread') === '1');
      const fs = parseFloat(localStorage.getItem('murshid_fs') || '1');
      if (!isNaN(fs)) this.fontScale.set(Math.min(this.FS_MAX, Math.max(this.FS_MIN, fs)));
    } catch {
      // تجاهل
    }
  }

  private savePref(): void {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem('murshid_dark', this.dark() ? '1' : '0');
      localStorage.setItem('murshid_autoread', this.autoRead() ? '1' : '0');
      localStorage.setItem('murshid_fs', String(this.fontScale()));
    } catch {
      // تجاهل
    }
  }

  uniShort(id: 'ksu' | 'kau'): string {
    return this.universities.find((u) => u.id === id)?.short ?? '';
  }

  relativeTime(ts: number): string {
    const min = Math.floor((Date.now() - ts) / 60000);
    if (min < 1) return 'الآن';
    if (min < 60) return 'قبل ' + min + ' د';
    const hr = Math.floor(min / 60);
    if (hr < 24) return 'قبل ' + hr + ' س';
    return 'قبل ' + Math.floor(hr / 24) + ' ي';
  }

  private uid(): number {
    return Date.now() + Math.floor(Math.random() * 100000);
  }

  private syncConversation(): void {
    const msgs = this.messages();
    if (msgs.length === 0) return;

    const now = Date.now();
    const title = this.makeTitle(msgs);
    const id = this.currentId();

    if (!id) {
      const newId = 'c_' + now + '_' + Math.random().toString(36).slice(2, 7);
      this.currentId.set(newId);
      const conv: Conversation = {
        id: newId,
        uni: this.selected().id,
        title,
        messages: [...msgs],
        createdAt: now,
        updatedAt: now,
      };
      this.history.update((list) => [conv, ...list].slice(0, MAX_CONVERSATIONS));
    } else {
      this.history.update((list) =>
        list.map((c) => (c.id === id ? { ...c, title, messages: [...msgs], updatedAt: now } : c)),
      );
    }
  }

  private makeTitle(msgs: ChatMessage[]): string {
    const firstUser = msgs.find((m) => m.role === 'user');
    const t = (firstUser?.text ?? 'محادثة جديدة').trim();
    return t.length > 34 ? t.slice(0, 34) + '…' : t;
  }

  private loadHistory(): void {
    if (typeof localStorage === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed: Conversation[] = JSON.parse(raw);
      const now = Date.now();
      const fresh = parsed
        .filter((c) => now - c.updatedAt < MAX_AGE_MS)
        .slice(0, MAX_CONVERSATIONS)
        .map((c) => ({
          ...c,
          messages: c.messages.map((m) => ({
            id: m.id ?? this.uid(),
            role: m.role,
            text: m.text,
            time: m.time,
            feedback: m.feedback ?? null,
            isError: m.isError ?? false,
          })),
        }));
      this.history.set(fresh);
    } catch {
      // تجاهل
    }
  }

  private saveHistory(list: Conversation[]): void {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
    } catch {
      // تجاهل
    }
  }

  private async ask(question: string): Promise<string> {
    const apiUrl = 'http://localhost:8000/ask';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          university: this.selected().id,
          history: this.recentTurns(),
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (!res.ok) throw new Error('bad response');
      const data = await res.json();
      return data.answer ?? data.response ?? 'لم يصل رد من النظام.';
    } catch (e) {
      clearTimeout(timer);
      // رد تجريبي محلي. عند ربط الخادم احذف السطرين التاليين ليظهر الخطأ الحقيقي وزر إعادة المحاولة:
      await this.wait(900);
      return this.demoAnswer(question);
    }
  }

  // Last up-to-2 (question, answer) pairs from the current conversation,
  // sent so the backend can rewrite conversational follow-ups into
  // standalone search queries. Error bubbles never enter the history.
  private recentTurns(): { question: string; answer: string }[] {
    const turns: { question: string; answer: string }[] = [];
    const m = this.messages();
    for (let i = m.length - 2; i >= 0 && turns.length < 2; i--) {
      if (m[i].role !== 'user') continue;
      const bot = m[i + 1];
      if (bot && bot.role === 'bot' && !bot.isError) {
        turns.unshift({ question: m[i].text, answer: bot.text });
      }
    }
    return turns;
  }

  private demoAnswer(question: string): string {
    return (
      'هذا رد تجريبي من الواجهة (لم يُربط الخادم بعد).\n' +
      'الجامعة المختارة: ' + this.selected().name + '\n' +
      'سؤالك: ' + question + '\n' +
      'بعد ربط النظام سيظهر هنا الرد الحقيقي المبني على مصادر الجامعة.'
    );
  }

  private wait(ms: number): Promise<void> {
    return new Promise((r) => setTimeout(r, ms));
  }

  private time(): string {
    return new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' });
  }
}