"""اختبار نقاط API مع المحادثة السابقة، بلا شبكة: النماذج مستبدلة بدوال ثابتة.

التشغيل:
    python -m pytest tests/test_api_history.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import openrouter
from answer import Answerer
from app import app

HISTORY = [
    {"question": "وش شروط الاختبار البديل؟", "answer": "يُشترط أن يتقدم الطالب بعذر مقبول [1]."}
]


def fake_search(
    query: str,
    k: int,
    university: str,
    query_vector: np.ndarray | None = None,
) -> list[dict]:
    hits = []
    for number in range(3):
        chunk_id = f"{query}-{number}"
        hits.append(
            {
                "chunk_id": chunk_id,
                "content_hash": chunk_id,
                "score": 0.5,
                "text": chunk_id,
                "body": chunk_id,
                "doc_title": "لائحة",
                "doc_type": "لائحة",
                "part": "1/1",
                "article_label": None,
                "university_ar": "جامعة الملك سعود",
            }
        )
    return hits


def fake_encode(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 4), dtype=np.float32)


def test_history_reaches_search_and_answer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    chat_calls = []
    rerank_calls = []

    def fake_chat(model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        chat_calls.append(messages)
        return {"content": "جواب [1]", "model": model, "cost": 0.0}

    def fake_rerank(model: str, query: str, documents: list[str]) -> dict:
        rerank_calls.append(query)
        return {"order": list(range(len(documents))), "cost": 0.0}

    monkeypatch.setattr(openrouter, "chat", fake_chat)
    monkeypatch.setattr(openrouter, "rerank", fake_rerank)
    retriever = SimpleNamespace(chunks=[], search=fake_search, encode=fake_encode)
    app.state.answerer = Answerer(retriever)
    client = TestClient(app)
    payload = {"question": "ومتى آخر موعد اقدم عليه؟", "university": "KSU", "history": HISTORY}
    search = client.post("/api/search", json=payload).json()
    answer = client.post("/api/answer", json=payload).json()

    assert len(chat_calls) == 1
    assert rerank_calls == ["وش شروط الاختبار البديل؟ ومتى آخر موعد اقدم عليه؟"]
    answer_messages = chat_calls[0]
    assert [message["role"] for message in answer_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert answer_messages[1]["content"] == HISTORY[0]["question"]
    assert [source["chunk_id"] for source in search["sources"]] == [
        source["chunk_id"] for source in answer["sources"]
    ]


def test_empty_question_gets_an_arabic_message() -> None:
    retriever = SimpleNamespace(chunks=[], search=fake_search, encode=fake_encode)
    app.state.answerer = Answerer(retriever)
    reply = TestClient(app).post("/api/answer", json={"question": "  ", "university": "KSU"})
    assert reply.json()["answer"] == "اكتب سؤالك أولاً."
