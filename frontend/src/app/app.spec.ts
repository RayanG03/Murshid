import { vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should default to King Saud University', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.selected().id).toBe('ksu');
  });

  it('should toggle the usage page and return to chat on university switch', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.view()).toBe('chat');
    app.toggleUsage();
    expect(app.view()).toBe('usage');
    app.selectUniversity(app.universities[1]);
    expect(app.view()).toBe('chat');
  });

  it('should switch university and clear messages', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.selectUniversity(app.universities[1]);
    expect(app.selected().id).toBe('kau');
    expect(app.messages().length).toBe(0);
  });
  it('should show a retryable error when the backend fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    try {
      const app = TestBed.createComponent(App).componentInstance;
      app.draft.set('شروط التحويل');
      await app.send();
      const last = app.messages().at(-1);
      expect(last?.isError).toBe(true);
      expect(last?.text).not.toContain('رد تجريبي');
    } finally { vi.unstubAllGlobals(); }
  });

  it('should send the request ID with feedback', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ answer: 'جواب', request_id: 'request-1' }) });
    vi.stubGlobal('fetch', fetchMock);
    try {
      const app = TestBed.createComponent(App).componentInstance;
      app.draft.set('شروط التحويل');
      await app.send();
      const last = app.messages().at(-1)!;
      expect(last.requestId).toBe('request-1');
      app.rate(last, 'up');
      expect(JSON.parse(fetchMock.mock.calls.at(-1)![1].body).request_id).toBe('request-1');
    } finally { vi.unstubAllGlobals(); }
  });
});