import { Component, DestroyRef, computed, inject, output, signal } from '@angular/core';

/** One provider as reported by GET /usage (backend src/status_page.py). */
interface ProviderUsage {
  id: string;
  role: string;
  configured: boolean;
  model: string;
  last_ok_at: number | null;
  last_error: { status: number; at: number; detail: string; retry_after_s: number | null } | null;
  /** Requests sent from THIS machine today (other team members not included). */
  device_today: { requests?: number; prompt_tokens?: number; completion_tokens?: number; credits?: number; errors?: number };
  note: string;
  headers_at: number | null;
  check_source?: string | null;
  groq?: {
    requests_limit_day: number | null;
    requests_remaining_day: number | null;
    tokens_limit_minute: number | null;
    tokens_remaining_minute: number | null;
    requests_reset_at: number | null;
    tokens_reset_at: number | null;
  };
  headers?: Record<string, string>;
  account?: {
    error?: string | number;
    fetched_at?: number;
    key?: { usage?: number; limit?: number | null };
    account?: { current_plan?: string; plan_usage?: number; plan_limit?: number };
  } | null;
}

interface UsageReport {
  now: number;
  live: boolean;
  fallback_mode: string;
  providers: ProviderUsage[];
  local_models: string[];
}

/** A labelled "remaining / limit" bar with an optional reset countdown. */
interface Meter {
  label: string;
  remaining: number | null;
  limit: number | null;
  resetAt: number | null;
}

interface ProviderCard {
  id: string;
  name: string;
  role: string;
  model: string;
  state: string;
  tone: 'ok' | 'bad' | 'muted';
  meters: Meter[];
  extraLines: string[];
  todayMain: string;
  todayDetail: string;
  error: { text: string; current: boolean } | null;
  note: string;
  lastOk: number | null;
}

const API = 'http://localhost:8000';
const NAMES: Record<string, string> = { groq: 'Groq', mistral: 'Mistral', gemini: 'Gemini', tavily: 'Tavily' };
const MODES: Record<string, string> = {
  off: 'مقفل',
  web: 'مفعّل',
  llm: 'جواب عام',
  web_then_llm: 'مفعّل + جواب عام',
};

@Component({
  selector: 'app-usage-status',
  templateUrl: './usage-status.html',
  styleUrl: './usage-status.css',
})
export class UsageStatus {
  /** Emitted when the user wants to go back to the chat. */
  readonly closed = output<void>();

  readonly report = signal<UsageReport | null>(null);
  readonly error = signal<string | null>(null);
  readonly refreshing = signal(false);
  /** Seconds since epoch, ticking every second for the countdowns. */
  readonly now = signal(Date.now() / 1000);

  readonly cards = computed<ProviderCard[]>(() => (this.report()?.providers ?? []).map((p) => this.toCard(p)));
  readonly subtitle = computed(() => {
    const r = this.report();
    if (!r) return 'جارٍ التحميل...';
    const kind = r.live ? 'قراءة مباشرة من المزوّدين' : 'آخر أرقام وصلت من المزوّدين';
    return `${kind} · ${this.clock(r.now)} · البحث في موقع الجامعة: ${MODES[r.fallback_mode] ?? r.fallback_mode}`;
  });

  private readonly fmt = new Intl.NumberFormat('ar-SA-u-nu-latn');

  constructor() {
    const tick = setInterval(() => this.now.set(Date.now() / 1000), 1000);
    const poll = setInterval(() => this.load(false), 30000);
    inject(DestroyRef).onDestroy(() => {
      clearInterval(tick);
      clearInterval(poll);
    });
    void this.load(true);
  }

  /** live=true asks every provider now; false shows the latest numbers they sent. */
  async load(live = false): Promise<void> {
    try {
      const res = await fetch(`${API}/usage${live ? '?live=true' : ''}`);
      if (!res.ok) throw new Error(String(res.status));
      this.report.set((await res.json()) as UsageReport);
      this.error.set(null);
    } catch {
      this.error.set('تعذّر الاتصال بالخادم. تأكد أن الباكاند يعمل.');
    }
  }

  async refreshLive(): Promise<void> {
    this.refreshing.set(true);
    try {
      await this.load(true);
    } finally {
      this.refreshing.set(false);
    }
  }

  num(n: number | null | undefined): string {
    return n == null ? '—' : this.fmt.format(n);
  }

  percent(m: Meter): number {
    if (m.remaining == null || !m.limit) return 0;
    return Math.max(0, Math.min(100, (m.remaining / m.limit) * 100));
  }

  level(m: Meter): 'ok' | 'warn' | 'bad' {
    const p = this.percent(m);
    return p > 40 ? 'ok' : p > 15 ? 'warn' : 'bad';
  }

  countdown(at: number | null | undefined): string {
    if (!at) return '—';
    let s = Math.max(0, Math.round(at - this.now()));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    s %= 60;
    if (h) return `${h} ساعة ${m} دقيقة`;
    if (m) return `${m} دقيقة ${s} ثانية`;
    return `${s} ثانية`;
  }

  ago(t: number | null | undefined): string {
    if (!t) return 'لم يُستخدم بعد';
    const s = Math.round(this.now() - t);
    if (s < 60) return `قبل ${Math.max(s, 0)} ثانية`;
    if (s < 3600) return `قبل ${Math.round(s / 60)} دقيقة`;
    return `قبل ${Math.round(s / 3600)} ساعة`;
  }

  clock(t: number): string {
    return new Date(t * 1000).toLocaleString('ar-SA-u-nu-latn', {
      hour: '2-digit',
      minute: '2-digit',
      day: 'numeric',
      month: 'short',
    });
  }

  private toCard(p: ProviderUsage): ProviderCard {
    const meters: Meter[] = [];
    const extraLines: string[] = [];

    if (p.groq) {
      const g = p.groq;
      meters.push(
        { label: 'الطلبات المتبقية اليوم', remaining: g.requests_remaining_day, limit: g.requests_limit_day, resetAt: g.requests_reset_at },
        { label: 'التوكنز المتبقية هذه الدقيقة', remaining: g.tokens_remaining_minute, limit: g.tokens_limit_minute, resetAt: g.tokens_reset_at },
      );
      extraLines.push(`كما أرسلها Groq ${this.ago(p.headers_at)}`);
    }
    if (p.id === 'tavily' && p.account) {
      const a = p.account;
      if (a.error != null) {
        extraLines.push(`تعذّر جلب الرصيد (${a.error})`);
      } else {
        const limit = a.key?.limit ?? a.account?.plan_limit ?? null;
        const used = a.key?.usage ?? a.account?.plan_usage ?? null;
        meters.push({
          label: 'الرصيد المتبقي هذا الشهر',
          remaining: limit != null && used != null ? limit - used : null,
          limit,
          resetAt: null,
        });
        extraLines.push(
          `المستخدم حسب Tavily: ${this.num(used)} · الخطة: ${a.account?.current_plan ?? '—'}` +
            (a.fetched_at ? ` · قُرئ ${this.ago(a.fetched_at)}` : ''),
        );
      }
    }
    if (p.headers && !p.groq) {
      extraLines.push(...Object.entries(p.headers).map(([k, v]) => `${k}: ${v}`));
    }

    let state = 'لم يُختبر';
    let tone: ProviderCard['tone'] = 'muted';
    const err = p.last_error;
    const errIsCurrent = !!err && (!p.last_ok_at || err.at > p.last_ok_at);
    if (!p.configured) {
      state = 'غير مفعّل';
    } else if (errIsCurrent && err) {
      state = err.status === 429 ? 'خلص الرصيد مؤقتاً' : 'خطأ';
      tone = 'bad';
    } else if (p.last_ok_at) {
      state = 'شغّال';
      tone = 'ok';
    }

    const t = p.device_today ?? {};
    const todayDetail =
      p.id === 'tavily'
        ? `الرصيد المستخدم: ${this.num(t.credits ?? 0)}`
        : `توكنز الإدخال: ${this.num(t.prompt_tokens ?? 0)} · الإخراج: ${this.num(t.completion_tokens ?? 0)}`;

    return {
      id: p.id,
      name: NAMES[p.id] ?? p.id,
      role: p.role,
      model: p.model,
      state,
      tone,
      meters,
      extraLines,
      todayMain: `${this.num(t.requests ?? 0)} طلب`,
      todayDetail: t.errors ? `${todayDetail} · أخطاء: ${this.num(t.errors)}` : todayDetail,
      error: err
        ? {
            text: `آخر خطأ (${err.status}) ${this.ago(err.at)}${err.retry_after_s ? ` · أعد المحاولة بعد ${Math.round(err.retry_after_s)} ثانية` : ''}`,
            current: errIsCurrent,
          }
        : null,
      note: p.note,
      lastOk: p.last_ok_at,
    };
  }
}
