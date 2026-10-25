"""Provider usage tracking for the /status page.

Records, per provider, what each API response tells us:
- Groq sends x-ratelimit-* headers on every response (remaining requests
  per day, remaining tokens per minute, and time until each resets).
- Mistral / DeepInfra: any rate-limit headers they send are kept as-is.
- Gemini sends no quota headers; daily quotas reset at midnight Pacific
  time (documented), and 429 errors are recorded.
- Tavily: credits per search; the live balance comes from GET /usage.
Plus our own per-day counters (requests, prompt/completion tokens),
persisted to data/logs/usage.json so they survive restarts. API keys and
request bodies are never stored.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(value: str | None) -> float | None:
    """'2m59.56s' / '7.66s' / '1h2m' / '250ms' / '12' -> seconds."""
    if not value:
        return None
    v = str(value).strip()
    try:
        return float(v)
    except ValueError:
        pass
    m = _DURATION_RE.match(v)
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


class UsageTracker:
    """Thread-safe in-memory usage registry with optional JSON persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._providers: dict[str, dict[str, Any]] = {}
        self._daily: dict[str, dict[str, dict[str, int]]] = {}
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._daily = data.get("daily", {})
                self._providers = data.get("providers", {})
            except Exception:  # noqa: BLE001 - a corrupt file never breaks /ask
                pass

    # ------------------------------------------------------------ writes

    def record_response(
        self, provider: str, model: str = "", headers: Any = None,
        prompt_tokens: int = 0, completion_tokens: int = 0, credits: int = 0,
        count: bool = True,
    ) -> None:
        """Store the provider's own rate-limit headers; ``count=False`` for
        live status checks (they must not inflate this device's counters)."""
        now = time.time()
        rate = _rate_headers(headers)
        with self._lock:
            p = self._providers.setdefault(provider, {})
            p.update({"model": model or p.get("model", ""), "last_ok_at": now})
            if rate:
                p["rate_headers"] = rate
                p["rate_headers_at"] = now
            if not count:
                self._save_locked()
                return
            day = self._daily.setdefault(_today(), {}).setdefault(
                provider, {"requests": 0, "prompt_tokens": 0,
                           "completion_tokens": 0, "credits": 0, "errors": 0})
            day["requests"] += 1
            day["prompt_tokens"] += int(prompt_tokens or 0)
            day["completion_tokens"] += int(completion_tokens or 0)
            day["credits"] += int(credits or 0)
            self._save_locked()

    def record_error(self, provider: str, status: int, detail: str = "",
                     headers: Any = None) -> None:
        now = time.time()
        rate = _rate_headers(headers)
        retry_after = parse_duration((rate or {}).get("retry-after"))
        with self._lock:
            p = self._providers.setdefault(provider, {})
            p["last_error"] = {"status": int(status), "at": now,
                               "detail": " ".join(str(detail).split())[:200],
                               "retry_after_s": retry_after}
            if rate:
                p["rate_headers"] = rate
                p["rate_headers_at"] = now
            day = self._daily.setdefault(_today(), {}).setdefault(
                provider, {"requests": 0, "prompt_tokens": 0,
                           "completion_tokens": 0, "credits": 0, "errors": 0})
            day["errors"] = day.get("errors", 0) + 1
            self._save_locked()

    # ------------------------------------------------------------- reads

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            today = self._daily.get(_today(), {})
            return {
                "now": time.time(),
                "providers": json.loads(json.dumps(self._providers)),
                "today": json.loads(json.dumps(today)),
            }

    def _save_locked(self) -> None:
        if self._path is None:
            return
        try:
            keep = dict(sorted(self._daily.items())[-14:])  # two weeks
            self._daily = keep
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(
                {"daily": keep, "providers": self._providers},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _rate_headers(headers: Any) -> dict[str, str]:
    """Only rate-limit related headers (never auth or anything else)."""
    if not headers:
        return {}
    try:
        items = dict(headers).items()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, v in items:
        name = str(k).lower()
        if "ratelimit" in name or name == "retry-after":
            out[name] = str(v)
    return out


def groq_view(rate: dict[str, str], seen_at: float) -> dict[str, Any]:
    """Structured Groq numbers from its documented headers."""
    def num(key: str) -> int | None:
        try:
            return int(float(rate[key]))
        except (KeyError, ValueError):
            return None

    view: dict[str, Any] = {
        "requests_limit_day": num("x-ratelimit-limit-requests"),
        "requests_remaining_day": num("x-ratelimit-remaining-requests"),
        "tokens_limit_minute": num("x-ratelimit-limit-tokens"),
        "tokens_remaining_minute": num("x-ratelimit-remaining-tokens"),
    }
    for key, out in (("x-ratelimit-reset-requests", "requests_reset_at"),
                     ("x-ratelimit-reset-tokens", "tokens_reset_at")):
        secs = parse_duration(rate.get(key))
        view[out] = seen_at + secs if secs is not None else None
    return view


def next_pacific_midnight(now: float | None = None) -> float:
    """Epoch seconds of the next midnight in America/Los_Angeles (Gemini
    daily quota reset). Falls back to UTC-8 without tz data."""
    now = time.time() if now is None else now
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Los_Angeles")
    except Exception:  # noqa: BLE001
        from datetime import timedelta

        tz = timezone(timedelta(hours=-8))
    local = datetime.fromtimestamp(now, tz)
    from datetime import timedelta

    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


_TRACKER: UsageTracker | None = None
_TRACKER_LOCK = threading.Lock()


def get_tracker() -> UsageTracker:
    """Process-wide tracker (persisted unless settings disable it)."""
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is None:
            from src.config import get_settings

            s = get_settings()
            path = (s.data_dir / "logs" / "usage.json") if s.usage_tracking_persist else None
            _TRACKER = UsageTracker(path)
        return _TRACKER
