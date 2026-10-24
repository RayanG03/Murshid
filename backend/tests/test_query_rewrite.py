"""History-aware query rewriting (src/query_rewrite.py) — fake HTTP only.

Standalone on purpose (self-contained fakes, no imports from other test
modules) so it applies cleanly regardless of local edits to sibling test
files. Covers:
  - history present -> follow-up rewritten, RETRIEVAL sees the rewritten
    query, answer prompt keeps the user's original wording;
  - no history -> no rewrite call, raw question goes to retrieval;
  - rewrite chain failure -> silent fallback to the raw question;
  - exhausted request deadline -> rewrite skipped, answer raises 502-level
    GenerationError, nothing calls an LLM;
  - sanitize/normalize unit behavior;
  - API level: /ask with history logs a seconds.rewrite field and answers
    from the rewritten query; malformed history never 400s.
"""

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api_logging import JsonlLogger
from src.generators import BaseGenerator, GenerationError
from src.main import create_app
from src.pipeline import answer_question
from src.query_rewrite import normalize_history, sanitize_rewrite
from src.retrieval import Retrieved

REWRITE_MARKER = "المحادثة السابقة"  # only the rewrite prompt contains it
REWRITTEN = "ما مواعيد القبول في كلية علوم الحاسب؟"


def _ret(rtype: str, score: float, pos: int) -> Retrieved:
    return Retrieved(
        source_type=rtype,
        score=score,
        document=f"doc-{rtype}-{pos}",
        metadata={"q_id": pos, "answer": "ج وهمية."} if rtype.endswith("_qa") else {},
    )


class _RecordingRetriever:
    """Fake KAU retriever: records every query, always returns one strong
    qa hit (0.90 clears the KAU floor) so the gate never refuses."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve_split(
        self, query: str, k: int | None = None, university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        self.queries.append(query)
        return [_ret("kau_qa_verified", 0.90, 1)], []


class _ScriptedGenerator(BaseGenerator):
    """Fake chain provider: pops queued outputs in call order; an queued
    Exception instance is raised (one attempt per call, max_attempts=1)."""

    def __init__(self, name: str = "scripted") -> None:
        super().__init__(name, backoff_base=0.01, max_attempts=1)
        self.model = f"{name}-model"
        self.calls: list[str] = []
        self.outputs: list[Any] = []

    def queue(self, *outputs: Any) -> "_ScriptedGenerator":
        self.outputs = list(outputs)
        return self

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        self.calls.append(prompt)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out, False


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.generators.time.sleep", lambda s: None)


# ----------------------------------------------------- pipeline level


def test_history_rewrites_the_search_query() -> None:
    gen = _ScriptedGenerator().queue(REWRITTEN, "جواب مصدر.")
    retriever = _RecordingRetriever()
    result = answer_question(
        "هل هناك مواعيد محددة؟", retriever, gen, university="kau",
        history=[("ما شروط القبول في كلية علوم الحاسب؟", "الشروط تشمل الجنسية.")],
    )
    assert result.refused is False
    assert result.answer == "جواب مصدر."
    # Retrieval ran against the REWRITTEN query, not the follow-up.
    assert retriever.queries == [REWRITTEN]
    # Two LLM calls: rewrite first, then the answer.
    assert len(gen.calls) == 2
    assert REWRITE_MARKER in gen.calls[0]
    assert "هل هناك مواعيد محددة؟" in gen.calls[0]
    # The previous turn's content reached the rewrite prompt.
    assert "الشروط تشمل الجنسية." in gen.calls[0]
    # The answer prompt keeps the user's original wording (not the rewrite).
    assert REWRITE_MARKER not in gen.calls[1]
    assert "هل هناك مواعيد محددة؟" in gen.calls[1]
    assert result.rewrite_seconds > 0


def test_no_history_skips_rewrite_entirely() -> None:
    gen = _ScriptedGenerator().queue("جواب واحد.")
    retriever = _RecordingRetriever()
    result = answer_question(
        "سؤال مستقل تماماً", retriever, gen, university="kau",
    )
    assert result.refused is False
    assert retriever.queries == ["سؤال مستقل تماماً"]
    assert len(gen.calls) == 1  # only the answer call
    assert result.rewrite_seconds == 0.0


def test_rewrite_failure_falls_back_to_raw_question() -> None:
    """The whole chain fails during the rewrite: the request must still
    answer, with retrieval falling back to the raw follow-up text.
    RuntimeError classifies as a terminal (non-retryable) provider error,
    so the rewrite makes exactly ONE call before GenerationError."""
    gen = _ScriptedGenerator().queue(
        RuntimeError("rewrite provider down"),
        "جواب بعد فشل إعادة الصياغة.",
    )
    retriever = _RecordingRetriever()
    result = answer_question(
        "لخّص الإجابة بنقاط", retriever, gen, university="kau",
        history=[("سؤال سابق", "جواب سابق")],
    )
    assert result.refused is False
    assert result.answer == "جواب بعد فشل إعادة الصياغة."
    assert retriever.queries == ["لخّص الإجابة بنقاط"]
    assert result.rewrite_seconds > 0
    # rewrite attempt (1 call, failed) + answer call = 2 prompt records.
    assert len(gen.calls) == 2


def test_exhausted_deadline_skips_rewrite_and_answer_raises() -> None:
    """A deadline already in the past (slow rerank case) must skip the
    rewrite AND the answer: GenerationError surfaces, zero LLM calls —
    the rewrite can never resurrect a request past its deadline."""
    gen = _ScriptedGenerator().queue("سؤال معدل", "جواب.")
    retriever = _RecordingRetriever()
    with pytest.raises(GenerationError):
        answer_question(
            "هل هناك مواعيد محددة؟", retriever, gen, university="kau",
            deadline=time.monotonic() - 1.0,
            history=[("سؤال سابق", "جواب سابق")],
        )
    assert gen.calls == []
    # The rewrite failed first and fell back; the answer then raised.
    assert retriever.queries == [] or retriever.queries == [
        "هل هناك مواعيد محددة؟"
    ]


# --------------------------------------------------------- unit level


def test_sanitize_rewrite_variants() -> None:
    assert sanitize_rewrite('  "ما مواعيد القبول؟"  ', "fb") == "ما مواعيد القبول؟"
    assert sanitize_rewrite("السؤال: ما الموعد؟", "fb") == "ما الموعد؟"
    assert sanitize_rewrite("الاستعلام: مواعيد التسجيل", "fb") == "مواعيد التسجيل"
    assert sanitize_rewrite("س\nؤ  ال\tتاء", "fb") == "س ؤ ال تاء"
    assert sanitize_rewrite("لا يمكنني الإجابة عن ذلك.", "fb") == "fb"
    assert sanitize_rewrite("", "fb") == "fb"
    assert sanitize_rewrite(None, "fb") == "fb"  # type: ignore[arg-type]
    assert sanitize_rewrite("س" * 400, "fb", max_chars=300) == "fb"
    assert sanitize_rewrite("سؤال جيد", "fb") == "سؤال جيد"


def test_normalize_history_shapes() -> None:
    assert normalize_history(None) == []
    assert normalize_history("junk") == []
    assert normalize_history([("س", "ج")]) == []  # tuples are not dicts
    assert normalize_history([{"question": "س1", "answer": "ج1"}]) == [
        ("س1", "ج1")
    ]
    assert normalize_history([{"q": "x"}, {"question": "س", "answer": "ج"}]) == [
        ("س", "ج")
    ]
    assert normalize_history([{"question": "س", "answer": ""}]) == []
    # Keeps the LAST two turns (most recent context wins).
    three = [
        {"question": f"س{i}", "answer": "ج"} for i in (1, 2, 3)
    ]
    assert normalize_history(three) == [("س2", "ج"), ("س3", "ج")]
    # Clips oversized turn text.
    long_turn = [{"question": "س", "answer": "ا" * 5000}]
    assert len(normalize_history(long_turn, max_chars=1200)[0][1]) == 1200


# ------------------------------------------------------------ API level


def test_ask_with_history_rewrites_and_logs(tmp_path) -> None:
    gen = _ScriptedGenerator().queue("سؤال معدل", "جواب وهمي من المزود.")
    app = create_app(load_resources=False)
    with TestClient(app) as client:
        retriever = _RecordingRetriever()
        app.state.ask_logger = JsonlLogger(tmp_path, "ask")
        app.state.feedback_logger = JsonlLogger(tmp_path, "feedback")
        app.state.retriever = retriever
        app.state.generation_chain = [gen]
        app.state.reranker = None

        res = client.post(
            "/ask",
            json={
                "question": "هل هناك مواعيد محددة؟",
                "university": "kau",
                "history": [
                    {"question": "ما شروط القبول؟", "answer": "الشروط تشمل."}
                ],
            },
        )
    assert res.status_code == 200
    assert res.json()["answer"] == "جواب وهمي من المزود."
    assert retriever.queries == ["سؤال معدل"]
    record = json.loads((tmp_path / "ask.jsonl").read_text("utf-8").splitlines()[-1])
    assert "rewrite" in record["seconds"]
    assert record["seconds"]["rewrite"] >= 0
    # The scripted generator answers instantly, so generation can round to
    # 0.0 — the >0 timing guarantee lives in test_api_layer's fakes, which
    # sleep. Here we assert schema, not magnitude.
    assert record["seconds"]["generation"] >= 0


def test_ask_malformed_history_never_400s(tmp_path) -> None:
    gen = _ScriptedGenerator().queue("جواب مباشر.")
    app = create_app(load_resources=False)
    with TestClient(app) as client:
        retriever = _RecordingRetriever()
        app.state.ask_logger = JsonlLogger(tmp_path, "ask")
        app.state.feedback_logger = JsonlLogger(tmp_path, "feedback")
        app.state.retriever = retriever
        app.state.generation_chain = [gen]
        app.state.reranker = None

        res = client.post(
            "/ask",
            json={
                "question": "سؤال جديد",
                "university": "kau",
                "history": "not-a-list",
            },
        )
    assert res.status_code == 200
    assert res.json()["answer"] == "جواب مباشر."
    # Malformed history degraded to no history: raw question, one LLM call.
    assert retriever.queries == ["سؤال جديد"]
    assert len(gen.calls) == 1
