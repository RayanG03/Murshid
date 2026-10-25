"""Small thread-safe TTL + LRU cache for /ask answers without history.

Repeated demo questions then cost no provider quota and return instantly.
Only successful, non-follow-up answers are cached; the key is the
university plus the whitespace-normalized question.
"""

import re
import threading
import time
from collections import OrderedDict
from typing import Any

_WS_RE = re.compile(r"\s+")


def cache_key(university: str, question: str) -> tuple[str, str]:
    """Normalized (university, question) key."""
    return university, _WS_RE.sub(" ", question).strip()


class AnswerCache:
    """LRU with per-entry expiry; size 0 disables it."""

    def __init__(self, size: int, ttl_s: float) -> None:
        self.size, self.ttl_s = size, ttl_s
        self._data: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str]) -> Any | None:
        if self.size <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if time.monotonic() - stored_at > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: tuple[str, str], value: Any) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
