"""Failover chain (generate_with_meta) unit tests — fake HTTP only.

Covers: 429 immediate failover, 429 zero-limit process skip, 429 retry on
the last provider, 500 failover, timeout failover, 401 skip, all providers
failing, request-deadline exhaustion, truncation flag.
"""

import logging
from typing import Any, Callable

import httpx
import pytest

import src.generators as gens
from src.config import Settings
from src.generators import GenerationError, generate_with_meta


class _FakeResponse:
    def __init__(
        self, payload: dict[str, Any], status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload) if status_code != 200 else ""
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok(content: str = "الإجابة النهائية.", finish: str = "stop") -> dict[str, Any]:
    """200 chat-completion payload (not a response object)."""
    return {
        "choices": [
            {"finish_reason": finish, "message": {"content": content}}
        ]
    }


def _chain_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        groq_api_key="GROQ_TEST_KEY",
        mistral_api_key="MISTRAL_TEST_KEY",
        gemini_api_key="GEMINI_TEST_KEY",
        groq_base_url="https://api.groq.invalid/openai/v1",
        mistral_base_url="https://api.mistral.invalid/v1",
        gemini_base_url="https://generativelanguage.invalid/v1beta",
        llm_retry_backoff_base_seconds=0.01,
        request_deadline_s=30.0,
        per_attempt_timeout_s=8.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _clean_auth_skips():
    """The process-global skip sets must not leak between tests."""
    gens._AUTH_SKIPPED.clear()
    gens._RATE_LIMIT_SKIPPED.clear()
    yield
    gens._AUTH_SKIPPED.clear()
    gens._RATE_LIMIT_SKIPPED.clear()


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(gens.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _install_post(
    monkeypatch: pytest.MonkeyPatch,
    responder: Callable[[str, int], Any],
) -> list[dict[str, Any]]:
    """Fake httpx.post.

    ``responder(provider, attempt_index_for_provider)`` returns either a
    (status, payload) tuple, a (status, payload, headers) tuple, or an
    Exception instance to raise. Calls are recorded per provider.
    """

    def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        provider = next(
            p for p in ("groq", "mistral", "gemini") if p in url
        )
        seen = _call_counts.setdefault(provider, 0)
        _call_counts[provider] = seen + 1
        _recorded.append({"provider": provider, "url": url, "json": kwargs})
        event = responder(provider, seen)
        if isinstance(event, Exception):
            raise event
        if len(event) == 3:
            status, payload, headers = event
            return _FakeResponse(payload, status, headers)
        status, payload = event
        return _FakeResponse(payload, status)

    monkeypatch.setattr(gens.httpx, "post", _fake_post)
    return _recorded


_recorded: list[dict[str, Any]] = []
_call_counts: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _reset_records():
    _recorded.clear()
    _call_counts.clear()
    yield
    _recorded.clear()
    _call_counts.clear()


def _chain(names: list[str], settings: Settings):
    return gens.get_generation_chain(names, settings)


def _outcomes(result) -> list[str]:
    return [f"{a.provider}={a.outcome}" for a in result.attempts]


# ------------------------------------------------------------- tests


def test_429_fails_over_immediately_when_next_provider_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (429, {"error": "rate limited"})
        return (200, _ok("الإجابة بعد الحد."))

    calls = _install_post(monkeypatch, _responder)
    result = generate_with_meta(
        _chain(["groq", "mistral"], settings),
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.provider == "mistral"
    assert result.model == "mistral-small-latest"
    assert result.text == "الإجابة بعد الحد."
    # No same-provider retry: one groq call, then straight to mistral.
    assert _outcomes(result) == ["groq=http_429", "mistral=ok"]
    groq_calls = [c for c in calls if "groq" in c["url"]]
    assert len(groq_calls) == 1
    assert result.truncated is False


def test_429_on_last_provider_retries_same_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy retry stays for the LAST provider: there is nowhere to go."""
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq" and seen == 0:
            return (429, {"error": "rate limited"})
        return (200, _ok("الإجابة بعد الحد."))

    _install_post(monkeypatch, _responder)
    result = generate_with_meta(
        _chain(["groq"], settings),
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.provider == "groq"
    assert _outcomes(result) == ["groq=http_429", "groq=ok"]


def test_429_zero_limit_skips_provider_for_the_process(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (
                429,
                {"error": "rate limited"},
                {"x-ratelimit-limit-req-minute": "0"},
            )
        return (200, _ok("من ميسترال بعد الصفر."))

    calls = _install_post(monkeypatch, _responder)
    chain = _chain(["groq", "mistral"], settings)
    with caplog.at_level(logging.INFO, logger="capstone.generation"):
        result = generate_with_meta(
            chain, "sys", "prompt", label="t", settings=settings
        )
    assert result.provider == "mistral"
    assert _outcomes(result) == ["groq=http_429", "mistral=ok"]
    assert "groq" in gens._RATE_LIMIT_SKIPPED
    assert "groq" not in gens._AUTH_SKIPPED
    skips = [r for r in caplog.records if "RATE LIMIT 0" in r.getMessage()]
    assert len(skips) == 1  # loud log fires exactly ONCE
    groq_calls = [c for c in calls if "groq" in c["url"]]

    # Second request in this process skips groq silently, no extra call.
    result2 = generate_with_meta(
        chain, "sys", "prompt", label="t2", settings=settings
    )
    assert result2.provider == "mistral"
    assert _outcomes(result2) == ["groq=rate_limit_skipped", "mistral=ok"]
    assert [c for c in calls if "groq" in c["url"]] == groq_calls


def test_429_nonzero_limit_header_still_fails_over_not_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal 429 (limit > 0) fails over but does NOT mark the provider."""
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (
                429,
                {"error": "rate limited"},
                {"x-ratelimit-limit-req-minute": "30"},
            )
        return (200, _ok("من ميسترال."))

    _install_post(monkeypatch, _responder)
    result = generate_with_meta(
        _chain(["groq", "mistral"], settings),
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.provider == "mistral"
    assert gens._RATE_LIMIT_SKIPPED == set()


def test_500_fails_over_to_next_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (500, {"error": "boom"})
        return (200, _ok("من ميسترال."))

    _install_post(monkeypatch, _responder)
    result = generate_with_meta(
        _chain(["groq", "mistral"], settings),
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.provider == "mistral"
    assert result.model == "mistral-small-latest"
    assert result.attempts and _outcomes(result)[0] == "groq=http_5xx"


def test_timeout_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            raise httpx.ReadTimeout("read timed out")
        return (200, _ok("جبتو بعد التايم آوت."))

    _install_post(monkeypatch, _responder)
    result = generate_with_meta(
        _chain(["groq", "mistral"], settings),
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.provider == "mistral"
    assert result.attempts[0].outcome == "timeout"


def test_401_skips_provider_immediately_and_marks_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (401, {"error": "bad key"})
        return (200, _ok("بنص عن ما بعد."))

    calls = _install_post(monkeypatch, _responder)
    chain = _chain(["groq", "mistral"], settings)
    with caplog.at_level(logging.INFO, logger="capstone.generation"):
        result = generate_with_meta(
            chain, "sys", "prompt", label="t", settings=settings
        )
    assert result.provider == "mistral"
    # No retries after the 401: failover was immediate.
    assert _outcomes(result) == ["groq=http_401", "mistral=ok"]
    assert "groq" in gens._AUTH_SKIPPED
    rejects = [r for r in caplog.records if "REJECTED AUTH" in r.getMessage()]
    assert len(rejects) == 1  # loud log fires exactly ONCE

    groq_calls = [c for c in calls if "groq" in c["url"]]
    # Second request in this process skips groq silently, no extra call.
    result2 = generate_with_meta(
        chain, "sys", "prompt", label="t2", settings=settings
    )
    assert result2.provider == "mistral"
    assert _outcomes(result2) == ["groq=auth_skipped", "mistral=ok"]
    assert [c for c in calls if "groq" in c["url"]] == groq_calls


def test_all_providers_fail_raises_one_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            return (429, {"error": "rate limited"})
        if provider == "mistral":
            return (500, {"error": "boom"})
        raise httpx.ReadTimeout("gemini timed out")

    _install_post(monkeypatch, _responder)
    with pytest.raises(GenerationError) as excinfo:
        generate_with_meta(
            _chain(["groq", "mistral", "gemini"], settings),
            "sys", "prompt", label="t", settings=settings,
        )
    message = str(excinfo.value)
    assert "failover chain exhausted on 't'" in message
    for provider in ("groq", "mistral", "gemini"):
        assert f"{provider}[" in message
    assert "http_429" in message and "http_5xx" in message and "timeout" in message


def test_shared_deadline_exhaustion_before_any_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _chain_settings(request_deadline_s=0.5,
                               per_attempt_timeout_s=8.0)

    def _responder(provider: str, seen: int) -> Any:  # pragma: no cover
        return (200, _ok("مميب وممنوع النجاز."))

    _install_post(monkeypatch, _responder)
    with pytest.raises(GenerationError) as excinfo:
        generate_with_meta(
            _chain(["groq", "mistral"], settings),
            "sys", "prompt", label="t", settings=settings,
        )
    message = str(excinfo.value)
    assert message.count("skipped_budget") == 2
    assert not _recorded  # no HTTP call may be attempted at all


def test_attempt_never_started_without_enough_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1.0s < min_attempt_seconds (1.5s since 2026-10-25): no attempt may
    # start. (With >= 1.5s left an attempt now starts, timeout clipped.)
    settings = _chain_settings(request_deadline_s=1.0,
                               per_attempt_timeout_s=8.0)
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        if provider == "groq":
            raise httpx.ConnectTimeout("g connection timed out")
        return (200, _ok("ممنوع الوصول."))  # never reached

    _install_post(monkeypatch, _responder)
    before = time_monotonic()
    with pytest.raises(GenerationError) as excinfo:
        generate_with_meta(
            _chain(["groq", "mistral"], settings),
            "sys", "prompt", label="t", settings=settings,
        )
    message = str(excinfo.value)
    assert "groq" in message and "mistral[skipped_budget]" in message
    assert not any("mistral" in c["url"] for c in _recorded)
    assert time_monotonic() - before < 1.5  # no 8s attempt ever started


def time_monotonic() -> float:
    import time as _time

    return _time.monotonic()


def test_truncation_flag_travels_in_result_not_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _chain_settings()
    _no_sleep(monkeypatch)

    def _responder(provider: str, seen: int) -> Any:
        return (200, _ok("جواب مقطوع...", "length"))

    _install_post(monkeypatch, _responder)
    # A shared instance passed into the chain must stay untouched:
    shared_gen = gens.GroqGenerator(settings)
    result = generate_with_meta(
        [shared_gen, gens.MistralGenerator(settings)],
        "sys", "prompt", label="t", settings=settings,
    )
    assert result.truncated is True  # finish_reason == "length" carried here
    assert shared_gen.last_truncated is False  # never instance state


def test_default_chain_order_configured_groq_first() -> None:
    settings = _chain_settings()
    assert settings.default_generation_providers == ("groq", "mistral", "gemini")
    chain = gens.get_generation_chain(None, settings)
    assert [g.name for g in chain] == ["groq", "mistral", "gemini"]
    # DeepInfra models stay out of the default chain:
    assert "deepseek" not in settings.default_generation_providers


def test_attempt_starts_with_short_budget_and_clipped_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5s left < per_attempt 8s: the attempt still starts (min 3s), with
    its timeout clipped to the remaining budget (2026-10-25 change)."""
    settings = _chain_settings(request_deadline_s=5.0,
                               per_attempt_timeout_s=8.0)
    seen_timeouts: list[float] = []
    real_invoke = gens.GroqGenerator._invoke_once

    def _spy(self, system, prompt, timeout_seconds):
        seen_timeouts.append(timeout_seconds)
        return real_invoke(self, system, prompt, timeout_seconds)

    monkeypatch.setattr(gens.GroqGenerator, "_invoke_once", _spy)
    _install_post(monkeypatch, lambda provider, seen: (200, _ok("جواب.")))
    result = generate_with_meta(
        _chain(["groq"], settings), "sys", "prompt", label="t",
        settings=settings,
    )
    assert result.provider == "groq"
    assert seen_timeouts and seen_timeouts[0] <= 5.0
