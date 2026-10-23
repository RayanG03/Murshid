"""API layer tests: fake retriever/reranker/generator, no models, no network."""

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.main as main_module
from src.api_logging import JsonlLogger
from src.generators import BaseGenerator
from src.main import QUESTION_MAX_CHARS, create_app
from src.retrieval import Retrieved


def _ret(rtype: str, score: float, pos: int) -> Retrieved:
    return Retrieved(
        source_type=rtype,
        score=score,
        document=f"doc-{rtype}-{pos}",
        metadata={"q_id": pos, "answer": "ج وهمية."} if rtype.endswith("_qa") else {},
    )


class _FakeRetriever:
    def __init__(
        self,
        qa: list[float] | None = None,
        docs: list[float] | None = None,
        qa_type: str = "kau_qa_verified",
        doc_type: str = "kau_web",
    ) -> None:
        self.qa_hits = [
            _ret(qa_type, s, i) for i, s in enumerate(qa or [])
        ]
        self.doc_hits = [
            _ret(doc_type, s, 100 + i) for i, s in enumerate(docs or [])
        ]

    def retrieve_split(
        self, query: str, k: int | None = None, university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        del query, k, university
        time.sleep(0.01)  # measurable search_seconds in the JSONL record
        return self.qa_hits, self.doc_hits


class _FakeReranker:
    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)

    def predict(self, pairs: Any, batch_size: int = 8,
                show_progress_bar: bool = False) -> list[float]:
        return self.probs


class _FakeChainGenerator(BaseGenerator):
    """One 'provider' in the fake chain; returns a fixed Arabic answer."""

    def __init__(self, name: str = "fake") -> None:
        super().__init__(name, backoff_base=0.01, max_attempts=1)
        self.model = f"{name}-model"
        self.calls = 0

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        self.calls += 1
        time.sleep(0.01)  # measurable generation_seconds in the JSONL record
        return f"إجابة وهمية من {self.name}.", False


class _FailingChainGenerator(_FakeChainGenerator):
    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        self.calls += 1
        raise RuntimeError("fake provider backend down")


def _configure(
    state: Any,
    retriever: _FakeRetriever | None = None,
    reranker: _FakeReranker | None = None,
    chain: list[BaseGenerator] | None = None,
) -> None:
    state.retriever = retriever or _FakeRetriever()
    state.reranker = reranker
    state.generation_chain = chain or [_FakeChainGenerator("fake")]


@pytest.fixture()
def env(tmp_path):
    """App + TestClient with fakes; JSONL writers pointed at tmp_path."""
    app = create_app(load_resources=False)
    with TestClient(app) as client:
        app.state.ask_logger = JsonlLogger(tmp_path, "ask")
        app.state.feedback_logger = JsonlLogger(tmp_path, "feedback")
        _configure(app.state)
        yield client, app, tmp_path


def _read_jsonl(tmp_path, name: str) -> list[dict]:
    lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------- /ask


def test_ask_happy_path_kau(env) -> None:
    client, app, tmp_path = env
    _configure(app.state, retriever=_FakeRetriever(qa=[0.90]))
    res = client.post(
        "/ask",
        json={"question": "ما وسائل التواصل مع عمادة شؤون الطلاب؟",
              "university": "kau"},
    )
    assert res.status_code == 200
    assert res.json()["answer"] == "إجابة وهمية من fake."
    records = _read_jsonl(tmp_path, "ask")
    assert len(records) == 1
    rec = records[0]
    assert rec["university"] == "kau"
    assert rec["refused"] is False
    assert rec["provider"] == "fake"
    assert rec["model"] == "fake-model"
    assert rec["truncated"] is False
    assert rec["seconds"]["search"] > 0
    assert rec["seconds"]["rerank"] == 0  # KAU has no rerank stage
    assert rec["seconds"]["generation"] > 0
    assert rec["seconds"]["total"] > 0
    # no missing fields
    assert set(rec) == {
        "timestamp", "university", "question", "answer", "refused",
        "provider", "model", "truncated", "seconds",
    }


def test_ask_refusal_returns_cannot_confirm_200(env) -> None:
    client, app, tmp_path = env
    _configure(app.state, retriever=_FakeRetriever())
    res = client.post("/ask", json={"question": "سؤال موجود", "university": "kau"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"].startswith("لا يمكنني تأكيد")
    rec = _read_jsonl(tmp_path, "ask")[0]
    assert rec["refused"] is True
    assert rec["provider"] == "none"
    assert rec["seconds"]["generation"] == 0


def test_ask_ksu__happy_path_and_refusal(env) -> None:
    client, app, tmp_path = env
    _configure(
        app.state,
        retriever=_FakeRetriever(qa=[0.90], qa_type="ksu_qa",
                                 doc_type="ksu_pdf"),
        reranker=_FakeReranker([0.90]),
    )
    res = client.post("/ask",
                      json={"question": "وش شروط الاعتذار عن الاختبار؟",
                            "university": "ksu"})
    assert res.status_code == 200
    assert res.json()["answer"] == "إجابة وهمية من fake."
    rec = _read_jsonl(tmp_path, "ask")[0]
    assert rec["provider"] == "fake" and rec["refused"] is False
    assert rec["seconds"]["rerank"] == 0.0  # fake reranker predict is untimed

    # refusal below threshold
    _configure(
        app.state,
        retriever=_FakeRetriever(qa=[0.90], qa_type="ksu_qa",
                                 doc_type="ksu_pdf"),
        reranker=_FakeReranker([0.10]),
    )
    (tmp_path / "ask.jsonl").unlink()
    res2 = client.post("/ask", json={"question": "سؤال", "university": "ksu"})
    assert res2.status_code == 200
    rec2 = _read_jsonl(tmp_path, "ask")[0]
    assert rec2["refused"] is True


def test_ask_validation_errors(env) -> None:
    client, _app, _tmp = env
    cases: list[tuple[dict[str, Any], int]] = [
        ({"question": "   ", "university": "kau"}, 400),
        ({"question": "x" * (QUESTION_MAX_CHARS + 1), "university": "kau"}, 400),
        ({"question": "سؤال", "university": "nau"}, 400),
        ({"question": "سؤال", "university": None}, 400),
        ({"question": 5, "university": "kau"}, 400),
    ]
    for payload, expected in cases:
        res = client.post("/ask", json=payload)
        assert res.status_code == expected, payload
        assert "error" in res.json()
    # boundary: exactly the cap is accepted (refusal path with empty gates)
    res = client.post(
        "/ask", json={"question": "x" * QUESTION_MAX_CHARS, "university": "kau"}
    )
    assert res.status_code == 200


def test_ask_unavailable_university_returns_503(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _app, _tmp = env
    monkeypatch.setattr(
        main_module, "is_university_available", lambda uni: False
    )
    res = client.post("/ask", json={"question": "سؤال", "university": "kau"})
    assert res.status_code == 503
    assert "not available" in res.json()["error"]


def test_ask_chain_exhaustion_returns_502(env) -> None:
    client, app, _tmp = env
    _configure(app.state, retriever=_FakeRetriever(qa=[0.90]),
               chain=[_FailingChainGenerator("fake")])
    res = client.post("/ask", json={"question": "سؤال", "university": "kau"})
    assert res.status_code == 502
    body = res.json()
    assert body["error"].endswith("fake provider backend down")
    assert "fake[error]" in body["error"]


# ------------------------------------------------------------ /feedback


def test_feedback_up_down_null_written_with_matched_question(env) -> None:
    client, app, tmp_path = env
    _configure(app.state, retriever=_FakeRetriever(qa=[0.90]))
    ask_res = client.post(
        "/ask", json={"question": "سؤال المطابقة", "university": "kau"}
    )
    answer = ask_res.json()["answer"]
    for value in ("up", "down", None):
        res = client.post(
            "/feedback",
            json={"university": "kau", "answer": answer, "feedback": value},
        )
        assert res.status_code == 200
        assert res.json() == {}
        assert res.content in (b"{}", b"null")
    records = _read_jsonl(tmp_path, "feedback")
    assert [r["feedback"] for r in records] == ["up", "down", None]
    assert all(r["matched_question"] == "سؤال المطابقة" for r in records)
    assert all(r["answer"] == answer for r in records)


def test_feedback_unmatched_answer_records_none(env) -> None:
    client, _app, tmp_path = env
    res = client.post(
        "/feedback",
        json={"university": "ksu", "answer": "نص غير موجود", "feedback": "up"},
    )
    assert res.status_code == 200
    rec = _read_jsonl(tmp_path, "feedback")[0]
    assert rec["matched_question"] is None
    assert rec["university"] == "ksu"


def test_feedback_validation_errors(env) -> None:
    client, _app, _tmp = env
    cases = [
        ({"university": "xyz", "answer": "نص", "feedback": "up"}, 400),
        ({"university": "kau", "answer": "", "feedback": "up"}, 400),
        ({"university": "kau", "answer": "نص", "feedback": "sideways"}, 400),
        ({"university": "kau", "answer": None, "feedback": None}, 400),
    ]
    for payload, expected in cases:
        res = client.post("/feedback", json=payload)
        assert res.status_code == expected, payload
        assert "error" in res.json()


# ------------------------------------------------------- /universities


def test_universities_registry_driven(env) -> None:
    client, _app, _tmp = env
    res = client.get("/universities")
    assert res.status_code == 200
    rows = res.json()
    assert [r["id"] for r in rows] == list(main_module.UNIVERSITY_COLLECTIONS)
    assert all(r["available"] for r in rows)
    assert all(isinstance(r["name"], str) and r["name"] for r in rows)


def test_health(env) -> None:
    client, _app, _tmp = env
    assert client.get("/health").json() == {"status": "ok"}
