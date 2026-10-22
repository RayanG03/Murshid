"""Stage 4 unit tests: floor logic, prompt labels, retry/backoff behavior."""

import logging

import pytest

from src.context_assembly import assemble_context, render_context
from src.generators import DeepInfraGenerator, GenerationError
from src.prompts import build_user_prompt
from src.retrieval import Retrieved


def _hit(source_type: str, score: float, doc: str = "نص", md: dict | None = None) -> Retrieved:
    return Retrieved(source_type=source_type, score=score, document=doc,
                     metadata=md or {})


def test_floors_filter_independently() -> None:
    qa = [_hit("kau_qa_verified", 0.90), _hit("kau_qa_verified", 0.70)]
    docs = [_hit("kau_web", 0.95), _hit("kau_pdf", 0.60)]
    ctx = assemble_context(qa, docs, floor_qa=0.83, floor_docs=0.79,
                           max_qa=3, max_docs=4)
    assert [r.score for r in ctx.qa_items] == [0.90]
    assert ctx.qa_gate == "strong-leader"
    assert [r.score for r in ctx.doc_items] == [0.95]
    assert ctx.doc_gate == "strong-leader"
    assert not ctx.is_empty


def test_empty_context_when_all_below_floor() -> None:
    ctx = assemble_context([_hit("kau_qa_verified", 0.5)], [_hit("kau_web", 0.5)],
                           floor_qa=0.83, floor_docs=0.79, max_qa=3, max_docs=4)
    assert ctx.is_empty
    assert ctx.qa_gate == "below-floor"
    assert ctx.doc_gate == "below-floor"
    assert render_context(ctx) == ""


def test_gate_rejects_flat_domain_adjacent_top() -> None:
    # Case (d) shape: docs 0.811/0.801/0.800 -> above floor, flat, no leader.
    docs = [_hit("kau_web", s) for s in (0.811, 0.801, 0.800)]
    ctx = assemble_context([], docs, floor_qa=0.83, floor_docs=0.79,
                           max_qa=3, max_docs=4)
    assert ctx.doc_gate == "ambiguous-flat"
    assert ctx.is_empty


def test_gate_rejects_bus_probe_shape() -> None:
    # Bus query shape: docs 0.867/0.829/0.829 -> below strong leader and
    # top-to-anchor gap (0.038) below the 0.04 threshold.
    docs = [_hit("kau_web", s) for s in (0.867, 0.829, 0.829)]
    ctx = assemble_context([], docs, floor_qa=0.83, floor_docs=0.79,
                           max_qa=3, max_docs=4)
    assert ctx.doc_gate == "ambiguous-flat"
    assert ctx.is_empty


def test_gate_accepts_clear_gap_leader() -> None:
    docs = [_hit("kau_web", s) for s in (0.85, 0.79, 0.79)]
    ctx = assemble_context([], docs, floor_qa=0.83, floor_docs=0.79,
                           max_qa=3, max_docs=4)
    assert ctx.doc_gate == "clear-gap"
    assert not ctx.is_empty


def test_gate_single_weak_survivor_rejected() -> None:
    docs = [_hit("kau_web", 0.80)]
    ctx = assemble_context([], docs, floor_qa=0.83, floor_docs=0.79,
                           max_qa=3, max_docs=4)
    assert ctx.doc_gate == "ambiguous-flat"
    assert ctx.is_empty


def test_prompt_has_labeled_blocks() -> None:
    prompt = build_user_prompt("س?", "qa block", "doc block")
    assert "أسئلة وإجابات موثّقة" in prompt
    assert "مقتطفات من اللوائح والأدلة" in prompt
    assert "س?" in prompt


def test_retry_backoff_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    from src.config import Settings

    offline = Settings(
        deepinfra_base_url="http://127.0.0.1:9", deepinfra_api_key="unused"
    )
    gen = DeepInfraGenerator(
        "deepseek", "test-model", settings=offline,
        backoff_base=0.01, max_attempts=3,
    )
    with caplog.at_level(logging.WARNING, logger="capstone.generation"):
        with pytest.raises(GenerationError):
            gen.generate("sys", "prompt", label="simulated-failure")
    attempts = [r for r in caplog.records if "FAILED" in r.getMessage()]
    # Network-layer errors (connection refused) extend the cap to 6 attempts.
    assert len(attempts) == 6
    assert any("attempt 1/6" in r.getMessage() for r in attempts)
    assert any("attempt 6/6" in r.getMessage() for r in attempts)
