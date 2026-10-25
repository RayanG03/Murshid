import { TestBed } from '@angular/core/testing';
import { UsageStatus } from './usage-status';

const NOW = Date.now() / 1000;
const REPORT = {
  now: NOW,
  live: true,
  fallback_mode: 'web',
  local_models: ['intfloat/multilingual-e5-large'],
  providers: [
    {
      id: 'groq', role: 'المولّد الأساسي', configured: true, model: 'openai/gpt-oss-120b',
      last_ok_at: NOW - 30, last_error: null, device_today: { requests: 3, prompt_tokens: 900, completion_tokens: 120 },
      note: '', headers_at: NOW - 30,
      groq: {
        requests_limit_day: 1000, requests_remaining_day: 990, tokens_limit_minute: 8000,
        tokens_remaining_minute: 500, requests_reset_at: NOW + 120, tokens_reset_at: NOW + 9,
      },
    },
    {
      id: 'gemini', role: 'احتياطي ثانٍ', configured: true, model: 'gemini-flash-latest',
      last_ok_at: null, last_error: { status: 429, at: NOW - 60, detail: 'quota', retry_after_s: null },
      device_today: { errors: 1 }, note: 'لا يتيح رقم الرصيد', headers_at: null,
    },
  ],
};

describe('UsageStatus', () => {
  beforeEach(async () => {
    globalThis.fetch = (async () => new Response(JSON.stringify(REPORT), { status: 200 })) as typeof fetch;
    await TestBed.configureTestingModule({ imports: [UsageStatus] }).compileComponents();
  });

  it('builds provider cards from /usage', async () => {
    const fixture = TestBed.createComponent(UsageStatus);
    const cmp = fixture.componentInstance;
    await cmp.load();
    const cards = cmp.cards();
    expect(cards.map((c) => c.id)).toEqual(['groq', 'gemini']);
    expect(cards[0].state).toBe('شغّال');
    expect(cards[0].meters[0].remaining).toBe(990);
    expect(cmp.level(cards[0].meters[1])).toBe('bad'); // 500 / 8000 tokens left
    expect(cards[1].state).toBe('خلص الرصيد مؤقتاً');
    expect(cards[1].note).toContain('لا يتيح');
    expect(cmp.subtitle()).toContain('قراءة مباشرة');
  });

  it('shows an error when the backend is down', async () => {
    globalThis.fetch = (async () => {
      throw new Error('offline');
    }) as typeof fetch;
    const cmp = TestBed.createComponent(UsageStatus).componentInstance;
    await cmp.load();
    expect(cmp.error()).toContain('تعذّر');
  });
});
