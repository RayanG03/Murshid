"""Groq / Mistral OpenAI-compatible generator unit tests.

Fake transport only (monkeypatched httpx.post): no live API calls.
Groq reasoning-parameter expectations verified against Groq docs
(console.groq.com) on 2026-10-23.
"""

import logging
from typing import Any

import pytest

import src.generators as gens
from src.config import Settings
from src.generators import GroqGenerator, MistralGenerator


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


def _chat_payload(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ]
    }


def _completions_settings() -> Settings:
    return Settings(
        groq_api_key="GROQ_TEST_KEY",
        mistral_api_key="MISTRAL_TEST_KEY",
        groq_base_url="https://api.groq.invalid/openai/v1",
        mistral_base_url="https://api.mistral.invalid/v1",
    )


def _captured_post(
    monkeypatch: pytest.MonkeyPatch,
    responder,
) -> list[dict[str, Any]]:
    """Replace httpx.post with a recorder; return the calls list."""
    calls: list[dict[str, Any]] = []

    def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return responder(calls)

    monkeypatch.setattr(gens.httpx, "post", _fake_post)
    return calls


# ---------------------------------------------------------------- Groq


def test_groq_payload_verified_against_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = GroqGenerator(_completions_settings())
    calls = _captured_post(
        monkeypatch, lambda _c: _FakeResponse(_chat_payload("الجواب"))
    )
    assert gen.generate("sys", "prompt", label="g") == "الجواب"
    call = calls[0]
    assert call["url"] == "https://api.groq.invalid/openai/v1/chat/completions"
    payload = call["json"]
    assert payload["model"] == "openai/gpt-oss-120b"
    # Reasoning tokens count against max_completion_tokens: low effort,
    # no reasoning exposure, correct token cap (docs-verified names).
    assert payload["reasoning_effort"] == "low"
    assert payload["include_reasoning"] is False
    assert payload["max_completion_tokens"] == Settings().llm_max_tokens
    assert "max_tokens" not in payload
    assert "reasoning_format" not in payload  # unsupported on gpt-oss
    assert "reasoning" not in payload  # never a bare reasoning field
    assert call["headers"]["Authorization"] == "Bearer GROQ_TEST_KEY"


def test_groq_reasoning_text_never_in_output(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If reasoning text arrives anyway, ONLY message.content is returned."""
    gen = GroqGenerator(_completions_settings())
    payload = _chat_payload("الجواب النهائي.")
    payload["choices"][0]["message"]["reasoning"] = "الكوليبر السري"
    _captured_post(monkeypatch, lambda _c: _FakeResponse(payload))
    with caplog.at_level(logging.WARNING, logger="capstone.generation"):
        answer = gen.generate("sys", "prompt")
    assert answer == "الجواب النهائي."
    assert "الكوليبر" not in answer
    assert all("الكوليبر" not in rec.getMessage() for rec in caplog.records)


def test_groq_truncation_flag_from_finish_reason_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = GroqGenerator(_completions_settings())
    _captured_post(
        monkeypatch,
        lambda _c: _FakeResponse(_chat_payload("بداية مقطوعة", "length")),
    )
    assert gen.generate("sys", "prompt") == "بداية مقطوعة"
    assert gen.last_truncated is True


# ---------------------------------------------------------------- Mistral


def test_mistral_payload_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = MistralGenerator(_completions_settings())
    calls = _captured_post(
        monkeypatch, lambda _c: _FakeResponse(_chat_payload("إجابة"))
    )
    answer = gen.generate("sys", "prompt")
    assert answer == "إجابة"
    assert gen.last_truncated is False
    payload = calls[0]["json"]
    assert payload["model"] == "mistral-small-latest"
    assert payload["max_tokens"] == Settings().llm_max_tokens
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert calls[0]["headers"]["Authorization"] == "Bearer MISTRAL_TEST_KEY"


def test_truncation_flag_resets_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = MistralGenerator(_completions_settings())
    state = {"length": True}

    def _responder(_c: list[dict[str, Any]]) -> _FakeResponse:
        reason = "length" if state["length"] else "stop"
        return _FakeResponse(_chat_payload("نص", reason))

    _captured_post(monkeypatch, _responder)
    gen.generate("sys", "prompt")
    assert gen.last_truncated is True
    state["length"] = False
    gen.generate("sys", "prompt")
    assert gen.last_truncated is False


def test_factory_exposes_completions_generators() -> None:
    assert gens.PROVIDERS == (
        "deepseek", "qwen", "llama", "gemini", "groq", "mistral"
    )
    built = [
        gens.get_generator(p, _completions_settings())
        for p in ("groq", "mistral")
    ]
    assert [g.name for g in built] == ["groq", "mistral"]
    assert built[0].model == "openai/gpt-oss-120b"
    assert built[1].model == "mistral-small-latest"
