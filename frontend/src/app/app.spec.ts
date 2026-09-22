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

  it('should switch university and clear messages', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.selectUniversity(app.universities[1]);
    expect(app.selected().id).toBe('kau');
    expect(app.messages().length).toBe(0);
  });
});