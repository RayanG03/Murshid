"""Provider-agnostic generation: DeepSeek/Qwen/Llama via DeepInfra, Gemini,
Groq, Mistral.

Every call goes through retry with exponential backoff and emits a
per-request progress log line, per AGENTS.md's hard rules. Failed-over
requests run through generate_with_meta(), which walks a provider chain
under ONE absolute request deadline (anchored at /ask start, passed in
by the caller) and returns a GenerationResult carrying the
truncation flag (never instance state — generators are shared across
concurrent requests).
"""

import abc
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import Settings, get_settings

logger = logging.getLogger("capstone.generation")

# httpx/httpcore log request URLs (which once carried the Gemini key as a
# query param) at INFO; pin them to WARNING so no URL ever reaches the logs.
for _http_logger_name in ("httpx", "httpcore"):
    logging.getLogger(_http_logger_name).setLevel(logging.WARNING)

PROVIDERS = ("deepseek", "qwen", "llama", "gemini", "groq", "mistral")

# Disables Gemini thinking: thinkingBudget=0 is allowed on the 2.5-series
# Flash (which gemini-flash-latest resolves to) and prevents reasoning
# tokens from consuming the shared maxOutputTokens budget. Do NOT also send
# thinkingLevel (Gemini 3 only) — mixing the two is a 400 error.
GEMINI_THINKING_BUDGET_DISABLED = 0


class GenerationError(RuntimeError):
    """Raised when a generator or the failover chain exhausts its retries."""


@dataclass(frozen=True)
class AttemptOutcome:
    """One attempted provider call inside a failover chain run."""

    provider: str
    outcome: str
    seconds: float
    detail: str = ""

    def summary(self) -> str:
        """Compact ASCII description for logs and error messages."""
        head = (
            f"{self.provider}={self.outcome}({self.seconds:.1f}s)"
            if self.seconds > 0
            else f"{self.provider}={self.outcome}"
        )
        return f"{head} {self.detail[:120]}" if self.detail else head


@dataclass
class GenerationResult:
    """Full metadata for one generation across the failover chain.

    ``truncated`` travels HERE (per request), never on the generator
    instance, because generators are shared across concurrent requests.
    """

    text: str
    provider: str
    model: str
    truncated: bool
    attempts: list[AttemptOutcome] = field(default_factory=list)
    total_seconds: float = 0.0


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
        self.model: str = ""
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
        # Set by generate(); for a chain use GenerationResult.truncated
        # (instance state is unreliable when instances are shared).
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

    def _invoke(self, system: str, prompt: str) -> str:
        """One backend call at the default timeout; keeps last_truncated fresh."""
        text, truncated = self._invoke_once(
            system, prompt, self._settings.llm_timeout_seconds
        )
        self.last_truncated = truncated
        return text

    @abc.abstractmethod
    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        """One backend request; raise on any failure.

        Returns (text, truncated). ``truncated`` must come from the
        backend's finish reason, never from instance state.
        """


class _HttpStatusError(Exception):
    def __init__(
        self, status: int, body: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        # Response headers (lower-cased keys by httpx) — read for 429
        # zero-rate-limit detection; None when a caller does not have them.
        self.headers = headers or {}


_RETRYABLE = (httpx.HTTPError, TimeoutError)


def _retryable_status(exc: Exception) -> bool:
    return getattr(exc, "status", 200) in (429, 503)


def _classify_attempt_exception(
    exc: Exception,
) -> tuple[str, bool, bool]:
    """Map one failed attempt to (outcome, retryable, auth_error).

    Timeout / network errors / 429 / 5xx are retryable; 401/403 skip the
    provider immediately; anything else is a provider failure that fails
    over to the next provider without retrying.
    """
    if isinstance(exc, _HttpStatusError):
        status = exc.status
        if status in (401, 403):
            return f"http_{status}", False, True
        if status == 429:
            return "http_429", True, False
        if status >= 500:
            return "http_5xx", True, False
        return f"http_{status}", False, False
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", True, False
    if isinstance(exc, _RETRYABLE):
        return "network_error", True, False
    return "error", False, False


# Providers skipped for the rest of the process after a 401/403 (key is
# invalid/wrong scope): retrying them every request is waste. Loud log ONCE.
_AUTH_SKIPPED: set[str] = set()

# Providers skipped for the rest of the process after a 429 whose response
# headers show a request rate limit of 0 (e.g. x-ratelimit-limit-req-minute:
# "0"): the provider cannot serve ANY request this minute, so retrying it —
# this request or on later requests — is guaranteed waste. Same policy as
# the 401/403 skip: marked once, loud log ONCE.
_RATE_LIMIT_SKIPPED: set[str] = set()

# Header-name prefixes whose VALUE is a request-count limit. A value of 0
# means the provider is configured to allow zero requests.
_ZERO_LIMIT_HEADER_PREFIXES = (
    "x-ratelimit-limit-req",
    "x-ratelimit-limit-requests",
    "ratelimit-limit-requests",
)


def _zero_request_limit_header(headers: dict[str, str] | None) -> str | None:
    """Return the first request-limit header whose value is exactly 0."""
    if not headers:
        return None
    for name, value in headers.items():
        normalized = name.lower().strip()
        if not normalized.startswith(_ZERO_LIMIT_HEADER_PREFIXES):
            continue
        if value.strip() in {"0", "0.0"}:
            return normalized
    return None


class DeepInfraGenerator(BaseGenerator):
    """DeepSeek / Qwen / Llama through DeepInfra's OpenAI-compatible API."""

    def __init__(self, name: str, model_id: str, settings: Settings | None = None,
                 **kwargs: Any) -> None:
        super().__init__(name, settings, **kwargs)
        s = self._settings
        self.model = model_id
        self._model_id = model_id
        self._url = f"{s.deepinfra_base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {s.deepinfra_api_key}",
            "Content-Type": "application/json",
        }

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
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
            timeout=timeout_seconds,
        )
        if resp.status_code != 200:
            raise _HttpStatusError(
                resp.status_code, resp.text, dict(resp.headers)
            )
        choice = resp.json()["choices"][0]
        finish_reason = str(choice.get("finish_reason", ""))
        truncated = finish_reason == "length"
        return choice["message"]["content"], truncated


class OpenAICompatGenerator(BaseGenerator):
    """One class for any OpenAI-compatible /chat/completions endpoint.

    Used for Groq and Mistral: Bearer-key auth, JSON payload, finish reason
    read from choices[0].finish_reason ("length" -> truncated). Optional
    ``extra_payload`` carries provider-specific controls (Groq reasoning
    parameters — verified against Groq docs, see config.py).
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str,
        api_key: str,
        settings: Settings | None = None,
        extra_payload: dict[str, Any] | None = None,
        use_max_completion_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, settings, **kwargs)
        self.model = model_id
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._extra_payload = dict(extra_payload or {})
        self._use_max_completion_tokens = use_max_completion_tokens

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._settings.llm_temperature,
            # Mistral takes max_tokens; Groq deprecates it in favor of
            # max_completion_tokens (reasoning tokens count against it).
        }
        if self._use_max_completion_tokens:
            payload["max_completion_tokens"] = self._settings.llm_max_tokens
        else:
            payload["max_tokens"] = self._settings.llm_max_tokens
        payload.update(self._extra_payload)
        resp = httpx.post(
            self._url, headers=self._headers, json=payload,
            timeout=timeout_seconds,
        )
        if resp.status_code != 200:
            raise _HttpStatusError(
                resp.status_code, resp.text, dict(resp.headers)
            )
        choice = resp.json()["choices"][0]
        finish_reason = str(choice.get("finish_reason", ""))
        truncated = finish_reason == "length"
        if truncated:
            logger.warning(
                "%s: answer truncated (finish_reason=length, "
                "max_tokens cap from config)",
                self.name,
            )
        # NEVER expose reasoning text: include_reasoning=false keeps Groq
        # reasoning out of the response; message.content is all we read.
        content = choice.get("message", {}).get("content")
        return (content or ""), truncated


class GroqGenerator(OpenAICompatGenerator):
    """Groq /chat/completions for openai/gpt-oss-120b (reasoning model).

    Reasoning tokens count against max_completion_tokens, so:
    - reasoning_effort="low" (docs: gpt-oss supports low|medium|high),
    - include_reasoning=False so the response has no reasoning field,
    - reasoning_format is NOT supported on gpt-oss (docs) — not sent.
    Parameters verified against Groq docs on 2026-10-23 (console.groq.com).
    """

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        s = settings or get_settings()
        super().__init__(
            "groq",
            s.groq_model_id,
            s.groq_base_url,
            s.groq_api_key,
            settings=s,
            extra_payload={
                "reasoning_effort": s.groq_reasoning_effort,
                "include_reasoning": s.groq_include_reasoning,
            },
            use_max_completion_tokens=True,
            **kwargs,
        )


class MistralGenerator(OpenAICompatGenerator):
    """Mistral /chat/completions (standard OpenAI-compatible payload)."""

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        s = settings or get_settings()
        super().__init__(
            "mistral",
            s.mistral_model_id,
            s.mistral_base_url,
            s.mistral_api_key,
            settings=s,
            **kwargs,
        )


class GeminiGenerator(BaseGenerator):
    """Gemini via the v1beta generateContent REST endpoint.

    Auth is the x-goog-api-key HEADER (never a ?key= query parameter, which
    could leak into URL logs; the httpx/httpcore loggers are pinned to
    WARNING for the same reason).
    """

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        super().__init__("gemini", settings, **kwargs)
        s = self._settings
        self.model = s.gemini_model_id
        self._url = (
            f"{s.gemini_base_url.rstrip('/')}/models/"
            f"{s.gemini_model_id}:generateContent"
        )
        self._headers = {
            "x-goog-api-key": s.gemini_api_key,
            "Content-Type": "application/json",
        }

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
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
            timeout=timeout_seconds,
        )
        if resp.status_code != 200:
            raise _HttpStatusError(
                resp.status_code, resp.text, dict(resp.headers)
            )
        data = resp.json()
        finish_reason = data["candidates"][0].get("finishReason", "")
        truncated = finish_reason == "MAX_TOKENS"
        if truncated:
            logger.warning(
                "gemini: answer truncated by maxOutputTokens "
                "(finishReason=MAX_TOKENS)"
            )
        return data["candidates"][0]["content"]["parts"][0]["text"], truncated


def get_generator(provider: str, settings: Settings | None = None) -> BaseGenerator:
    """Factory: all models swappable behind the same interface."""
    s = settings or get_settings()
    providers = {
        "deepseek": lambda: DeepInfraGenerator(
            "deepseek", s.deepseek_model_id, s),
        "qwen": lambda: DeepInfraGenerator("qwen", s.qwen_model_id, s),
        "llama": lambda: DeepInfraGenerator("llama", s.llama_model_id, s),
        "gemini": lambda: GeminiGenerator(s),
        "groq": lambda: GroqGenerator(settings=s),
        "mistral": lambda: MistralGenerator(settings=s),
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Use one of {PROVIDERS}.")
    return providers[provider]()


def get_generation_chain(
    providers: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> list[BaseGenerator]:
    """Build the failover chain: default order comes from config only."""
    s = settings or get_settings()
    names = tuple(providers) if providers is not None else s.default_generation_providers
    if not names:
        raise ValueError("Failover chain is empty (default_generation_providers).")
    return [get_generator(name, s) for name in names]


def _chain_fail_message(label: str, attempts: list[AttemptOutcome]) -> str:
    """One clear message listing every provider's outcome (and last error)."""
    by_provider: dict[str, list[AttemptOutcome]] = {}
    for a in attempts:
        by_provider.setdefault(a.provider, []).append(a)
    summary = "; ".join(
        (
            f"{provider}[{','.join(o.outcome for o in outcomes)}]"
            f"{f' last: {outcomes[-1].detail[:120]}' if outcomes[-1].detail else ''}"
        )
        for provider, outcomes in by_provider.items()
    )
    return (
        f"failover chain exhausted on '{label}': {summary or 'no providers tried'}"
    )


def generate_with_meta(
    chain: Sequence[BaseGenerator],
    system: str,
    prompt: str,
    label: str = "",
    settings: Settings | None = None,
    deadline: float | None = None,
) -> GenerationResult:
    """Run the failover chain under ONE absolute request deadline.

    ``deadline`` is a time.monotonic() timestamp anchored at the START of
    /ask handling (before retrieval/rerank) and passed down through the
    pipeline; when None, it is anchored here (offline callers). No attempt
    may start unless it can finish within the remaining deadline.
    Timeout / network error / 5xx fail over to the next provider (after
    in-provider retries if the deadline allows); a 429 fails over
    IMMEDIATELY (no same-provider retry) whenever another provider
    remains in the chain, and a 429 whose headers show a zero request
    rate limit skips that provider for the whole process (like 401/403);
    401/403 skip the provider immediately for the whole process.
    Every provider failing raises ONE GenerationError listing outcomes.
    """
    s = settings or get_settings()
    label = label or "request"
    started = time.monotonic()
    if deadline is None:
        deadline = started + s.request_deadline_s
    attempts: list[AttemptOutcome] = []

    for index, gen in enumerate(chain):
        provider = gen.name
        has_next = index + 1 < len(chain)
        if provider in _AUTH_SKIPPED:
            attempts.append(AttemptOutcome(
                provider, "auth_skipped", 0.0, "skipped (earlier 401/403)"
            ))
            continue
        if provider in _RATE_LIMIT_SKIPPED:
            attempts.append(AttemptOutcome(
                provider, "rate_limit_skipped", 0.0,
                "skipped (earlier 429 with zero request rate limit)",
            ))
            continue
        result = _run_provider_with_budget(
            gen, system, prompt, label, deadline, s, attempts,
            has_next=has_next,
        )
        if result is not None:
            return GenerationResult(
                text=result[0],
                provider=provider,
                model=gen.model or provider,
                truncated=result[1],
                attempts=attempts,
                total_seconds=time.monotonic() - started,
            )
    raise GenerationError(_chain_fail_message(label, attempts))


def _run_provider_with_budget(
    gen: BaseGenerator,
    system: str,
    prompt: str,
    label: str,
    deadline: float,
    settings: Settings,
    attempts: list[AttemptOutcome],
    has_next: bool = False,
) -> tuple[str, bool] | None:
    """Retry loop for one provider, bounded by the shared request deadline.

    Returns (text, truncated) on success, None when the provider failed
    (each outcome appended to ``attempts`` with seconds spent). A 429
    fails over immediately when ``has_next`` is True (another provider
    remains in the chain); on the last provider the legacy in-provider
    retry within the remaining deadline still applies.
    """
    provider = gen.name
    attempt = 0
    attempts_cap = gen._max_attempts
    started_any = False
    while True:
        remaining = deadline - time.monotonic()
        # Never start an attempt that cannot finish within the budget.
        if remaining < settings.per_attempt_timeout_s:
            if not started_any:
                attempts.append(AttemptOutcome(
                    provider, "skipped_budget", 0.0,
                    f"remaining {remaining:.1f}s < "
                    f"per_attempt {settings.per_attempt_timeout_s:.1f}s",
                ))
            return None
        attempt += 1
        started_any = True
        t0 = time.monotonic()
        try:
            text, truncated = gen._invoke_once(
                system, prompt, settings.per_attempt_timeout_s
            )
            seconds = time.monotonic() - t0
            attempts.append(AttemptOutcome(provider, "ok", seconds))
            logger.info(
                "[failover] %s %s attempt %d: ok (%d chars, %.2fs)",
                provider, label, attempt, len(text), seconds,
            )
            return text, truncated
        except Exception as exc:  # noqa: BLE001 - classified + logged below
            seconds = time.monotonic() - t0
            outcome, retryable, auth_error = _classify_attempt_exception(exc)
            if outcome == "http_429":
                zero_header = _zero_request_limit_header(
                    getattr(exc, "headers", None)
                )
                if zero_header is not None:
                    detail = (
                        f"{zero_header}: 0 - provider serves no requests; "
                        f"skipping it for the rest of the process"
                    )
                    attempts.append(AttemptOutcome(
                        provider, "http_429", seconds, detail
                    ))
                    logger.warning(
                        "[failover] %s %s attempt %d FAILED (http_429, %.2fs)",
                        provider, label, attempt, seconds,
                    )
                    if provider not in _RATE_LIMIT_SKIPPED:
                        _RATE_LIMIT_SKIPPED.add(provider)
                        logger.error(
                            "[failover] %s RATE LIMIT 0 (%s): the provider "
                            "cannot serve any request; skipping %s for every "
                            "future request in this process.",
                            provider, zero_header, provider,
                        )
                    return None
                if has_next:
                    attempts.append(AttemptOutcome(
                        provider, outcome, seconds, f"{type(exc).__name__}: {exc}"[:160],
                    ))
                    logger.warning(
                        "[failover] %s %s attempt %d FAILED (http_429, %.2fs) "
                        "- failing over immediately, no same-provider retry",
                        provider, label, attempt, seconds,
                    )
                    return None
            attempts.append(AttemptOutcome(
                provider, outcome, seconds, f"{type(exc).__name__}: {exc}"[:160]
            ))
            logger.warning(
                "[failover] %s %s attempt %d FAILED (%s, %.2fs)%s",
                provider, label, attempt, outcome, seconds,
                " - provider skipped for the process" if auth_error else "",
            )
            if auth_error:
                if provider not in _AUTH_SKIPPED:
                    _AUTH_SKIPPED.add(provider)
                    logger.error(
                        "[failover] %s REJECTED AUTH (401/403): check its "
                        "API key in backend/.env; skipping %s for every "
                        "future request in this process.",
                        provider, provider,
                    )
                return None
            if retryable:
                attempts_cap = max(attempts_cap, 6)
            if attempt >= attempts_cap:
                return None
            backoff = min(gen._backoff_base * (2 ** (attempt - 1)), 90.0)
            remaining = deadline - time.monotonic()
            if backoff >= remaining:
                logger.warning(
                    "[failover] %s: backoff %.1fs exceeds remaining "
                    "budget %.1fs - failing over now",
                    provider, backoff, remaining,
                )
                return None
            logger.warning(
                "[failover] %s %s backoff %.1fs (remaining %.1fs)",
                provider, label, backoff, remaining,
            )
            time.sleep(min(backoff, remaining))
