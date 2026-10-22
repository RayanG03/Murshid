"""Provider-agnostic generation: DeepSeek/Qwen/Llama via DeepInfra, Gemini.

Every call goes through retry with exponential backoff and emits a
per-request progress log line, per AGENTS.md's hard rules.
"""

import abc
import logging
import time
from typing import Any

import httpx

from src.config import Settings, get_settings

logger = logging.getLogger("capstone.generation")

# httpx/httpcore log request URLs (which once carried the Gemini key as a
# query param) at INFO; pin them to WARNING so no URL ever reaches the logs.
for _http_logger_name in ("httpx", "httpcore"):
    logging.getLogger(_http_logger_name).setLevel(logging.WARNING)

PROVIDERS = ("deepseek", "qwen", "llama", "gemini")

# Disables Gemini thinking: thinkingBudget=0 is allowed on the 2.5-series
# Flash (which gemini-flash-latest resolves to) and prevents reasoning
# tokens from consuming the shared maxOutputTokens budget. Do NOT also send
# thinkingLevel (Gemini 3 only) — mixing the two is a 400 error.
GEMINI_THINKING_BUDGET_DISABLED = 0


class GenerationError(RuntimeError):
    """Raised when a generator exhausts its retries."""


class BaseGenerator(abc.ABC):
    """Common interface: generate(prompt, context) -> Arabic answer."""

    def __init__(
        self,
        name: str,
        settings: Settings | None = None,
        backoff_base: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.name = name
        self._settings = settings or get_settings()
        self._backoff_base = (
            backoff_base
            if backoff_base is not None
            else self._settings.llm_retry_backoff_base_seconds
        )
        self._max_attempts = (
            max_attempts
            if max_attempts is not None
            else max(2, self._settings.llm_max_retries)
        )
        # Set by generators whose finish reason is observable (Gemini);
        # reset at the start of every generate() call.
        self.last_truncated: bool = False

    def generate(self, system: str, prompt: str, label: str = "") -> str:
        """Invoke the backend with retry, backoff and progress logging."""
        label = label or "request"
        attempts_cap = self._max_attempts
        attempt = 0
        last_err = ""
        self.last_truncated = False
        while True:
            attempt += 1
            try:
                text = self._invoke(system, prompt)
                logger.info(
                    "[%s] %s attempt %d/%d: ok (%d chars)",
                    self.name, label, attempt, attempts_cap, len(text),
                )
                return text
            except Exception as exc:  # noqa: BLE001 - retried below
                last_err = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, _RETRYABLE) or _retryable_status(exc):
                    attempts_cap = max(attempts_cap, 6)
            backoff = min(self._backoff_base * (2 ** (attempt - 1)), 90.0)
            logger.warning(
                "[%s] %s attempt %d/%d FAILED (%s) - backoff %.0fs",
                self.name, label, attempt, attempts_cap, last_err[:160], backoff,
            )
            if attempt >= attempts_cap:
                break
            time.sleep(backoff)
        raise GenerationError(
            f"{self.name} failed after {attempt} retries on '{label}': {last_err}"
        )

    @abc.abstractmethod
    def _invoke(self, system: str, prompt: str) -> str:
        """One backend request; raise on any failure."""


class _HttpStatusError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status


_RETRYABLE = (httpx.HTTPError, TimeoutError)


def _retryable_status(exc: Exception) -> bool:
    return getattr(exc, "status", 200) in (429, 503)


class DeepInfraGenerator(BaseGenerator):
    """DeepSeek / Qwen / Llama through DeepInfra's OpenAI-compatible API."""

    def __init__(self, name: str, model_id: str, settings: Settings | None = None,
                 **kwargs: Any) -> None:
        super().__init__(name, settings, **kwargs)
        s = self._settings
        self._model_id = model_id
        self._url = f"{s.deepinfra_base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {s.deepinfra_api_key}",
            "Content-Type": "application/json",
        }

    def _invoke(self, system: str, prompt: str) -> str:
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        resp = httpx.post(
            self._url, headers=self._headers, json=payload,
            timeout=self._settings.llm_timeout_seconds,
        )
        if resp.status_code != 200:
            raise _HttpStatusError(resp.status_code, resp.text)
        return resp.json()["choices"][0]["message"]["content"]


class GeminiGenerator(BaseGenerator):
    """Gemini via the v1beta generateContent REST endpoint.

    Auth is the x-goog-api-key HEADER (never a ?key= query parameter, which
    could leak into URL logs; the httpx/httpcore loggers are pinned to
    WARNING for the same reason).
    """

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        super().__init__("gemini", settings, **kwargs)
        s = self._settings
        self._url = (
            f"{s.gemini_base_url.rstrip('/')}/models/"
            f"{s.gemini_model_id}:generateContent"
        )
        self._headers = {
            "x-goog-api-key": s.gemini_api_key,
            "Content-Type": "application/json",
        }

    def _invoke(self, system: str, prompt: str) -> str:
        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self._settings.llm_temperature,
                "maxOutputTokens": self._settings.gemini_max_output_tokens,
                "thinkingConfig": {
                    "thinkingBudget": GEMINI_THINKING_BUDGET_DISABLED,
                },
            },
        }
        resp = httpx.post(
            self._url,
            headers=self._headers,
            json=payload,
            timeout=self._settings.llm_timeout_seconds,
        )
        if resp.status_code != 200:
            raise _HttpStatusError(resp.status_code, resp.text)
        data = resp.json()
        finish_reason = data["candidates"][0].get("finishReason", "")
        self.last_truncated = finish_reason == "MAX_TOKENS"
        if self.last_truncated:
            logger.warning(
                "gemini: answer truncated by maxOutputTokens "
                "(finishReason=MAX_TOKENS)"
            )
        return data["candidates"][0]["content"]["parts"][0]["text"]


def get_generator(provider: str, settings: Settings | None = None) -> BaseGenerator:
    """Factory: all four models swappable behind the same interface."""
    s = settings or get_settings()
    providers = {
        "deepseek": lambda: DeepInfraGenerator(
            "deepseek", s.deepseek_model_id, s),
        "qwen": lambda: DeepInfraGenerator("qwen", s.qwen_model_id, s),
        "llama": lambda: DeepInfraGenerator("llama", s.llama_model_id, s),
        "gemini": lambda: GeminiGenerator(s),
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Use one of {PROVIDERS}.")
    return providers[provider]()
