"""اختبارات مسار الاسترجاع بلا شبكة: كاش الاسترجاع، والتحويل إلى ملف المتجهات إن توقف Qdrant.

التشغيل:
    python -m pytest tests/test_retrieval_flow.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

import openrouter
from answer import Answerer
from retrieve import Retriever


def fake_hit(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "content_hash": chunk_id,
        "score": 0.5,
        "text": chunk_id,
        "body": chunk_id,
        "doc_title": "وثيقة",
        "article_label": None,
        "university_ar": "جامعة الملك سعود",
    }


def fake_search(
    query: str,
    k: int,
    university: str,
    query_vector: np.ndarray | None = None,
) -> list[dict]:
    return [fake_hit(f"{query}-{number}") for number in range(6)]


def fake_encode(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 4), dtype=np.float32)


def no_chat(model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    raise AssertionError("الاسترجاع لا يستدعي نموذج محادثة")


def fake_retriever(search=fake_search) -> SimpleNamespace:
    return SimpleNamespace(chunks=[], search=search, encode=fake_encode)


def test_search_then_answer_runs_the_reranker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rerank_calls = []

    def fake_rerank(model: str, query: str, documents: list[str]) -> dict:
        rerank_calls.append(query)
        return {"order": list(range(len(documents))), "cost": 0.0}

    monkeypatch.setattr(openrouter, "chat", no_chat)
    monkeypatch.setattr(openrouter, "rerank", fake_rerank)
    retriever = fake_retriever()
    answerer = Answerer(retriever)
    first = answerer.retrieve("سؤال", "KSU")
    second = answerer.retrieve("سؤال", "KSU")
    assert first is second
    assert len(rerank_calls) == 1
    assert first["reranked"] and not first["rerank_failed"]


def test_failed_reranker_keeps_the_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_rerank(model: str, query: str, documents: list[str]) -> None:
        return None

    monkeypatch.setattr(openrouter, "chat", no_chat)
    monkeypatch.setattr(openrouter, "rerank", failing_rerank)
    result = Answerer(fake_retriever()).retrieve("سؤال", "KSU")
    assert result["rerank_failed"] and not result["reranked"]
    assert [hit["chunk_id"] for hit in result["hits"]] == [f"سؤال-{n}" for n in range(4)]


def test_qdrant_failure_falls_back_to_the_vectors_file() -> None:
    def failing_query(*args: object, **kwargs: object) -> None:
        raise ResponseHandlingException(ConnectionError("Qdrant stopped"))

    retriever = Retriever(use_qdrant=False)
    retriever.qdrant = SimpleNamespace(query_points=failing_query)
    rows = retriever.university_rows["KSU"]
    scores = retriever.dense_scores(retriever.vectors[rows[0]], "KSU", rows, True)
    assert retriever.qdrant is None
    assert scores.shape == (len(rows),)
    assert abs(float(scores[0]) - 1.0) < 1e-5


def test_follow_up_searches_with_the_earlier_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openrouter, "chat", no_chat)
    history = [{"question": "كم مرة أقدر أأجل الدراسة؟", "answer": "فصلين متتاليين [1]."}]
    answerer = Answerer(fake_retriever(), use_reranker=False)
    result = answerer.retrieve("وفي الماجستير؟", "KSU", history=history)
    assert result["hits"][0]["chunk_id"] == "كم مرة أقدر أأجل الدراسة؟ وفي الماجستير؟-0"
