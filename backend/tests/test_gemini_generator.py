"""Gemini generator unit tests: header auth, thinkingConfig, MAX_TOKENS.

Fake transport only (monkeypatched httpx.post): no live API calls.
"""

from typing import Any

import logging

import pytest

import src.generators as gens
from src.config import Settings
from src.generators import GeminiGenerator


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


def _settings() -> Settings:
    return Settings(
        gemini_api_key="TEST_KEY",
        gemini_base_url="https://generativelanguage.invalid/v1beta",
        gemini_model_id="gemini-flash-latest",
    )


def _payload(finish_reason: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {"parts": [{"text": "الجواب"}]},
            }
        ]
    }


def _captured_post(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None = None):
    """Replace httpx.post with a recorder; return (calls, captured_payloads)."""
    calls: list[dict[str, Any]] = []

    def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return _FakeResponse(_payload() if payload is None else payload)

    monkeypatch.setattr(gens.httpx, "post", _fake_post)
    return calls


def _generate_once(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None = None):
    gen = GeminiGenerator(_settings())
    calls = _captured_post(monkeypatch, payload)
    answer = gen.generate("sys", "prompt", label="test")
    return gen, calls, answer


def test_key_sent_in_header_never_in_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="capstone.generation"):
        _gen, calls, answer = _generate_once(monkeypatch)
    assert answer == "الجواب"
    call = calls[0]
    assert "TEST_KEY" in call["headers"]["x-goog-api-key"]
    assert "params" not in call  # no query string is only key-carrying field
    assert "key=" not in call["url"]
    # No transport-level INFO log can print the URL with the key either.
    with caplog.at_level(logging.DEBUG, logger="capstone.generation"):
        pass
    assert all("TEST_KEY" not in rec.getMessage() for rec in caplog.records)


def test_thinking_config_disables_thinking_named_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gen, calls, _answer = _generate_once(monkeypatch)
    generation_cfg = calls[0]["json"]["generationConfig"]
    # thinkingBudget=0 disables thinking on the 2.5-series Flash alias so
    # reasoning tokens can never consume the answer budget.
    assert generation_cfg["thinkingConfig"]["thinkingBudget"] == 0
    assert "thinkingLevel" not in generation_cfg["thinkingConfig"]
    assert generation_cfg["maxOutputTokens"] == Settings().gemini_max_output_tokens


def test_max_tokens_finish_reason_sets_flag_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="capstone.generation"):
        gen, _calls, answer = _generate_once(monkeypatch, _payload("MAX_TOKENS"))
    assert answer == "الجواب"  # truncated text is still returned, shape kept
    assert gen.last_truncated is True
    assert any("MAX_TOKENS" in rec.getMessage() for rec in caplog.records)


def test_stop_finish_reason_flag_false_and_reset_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen, _calls, _answer = _generate_once(monkeypatch, _payload("STOP"))
    assert gen.last_truncated is False
    gen.generate("sys", "prompt", label="second")
    assert gen.last_truncated is False


def test_http_loggers_pinned_to_warning() -> None:
    import src.generators  # noqa: F401 - re-import keeps side effect applied

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
