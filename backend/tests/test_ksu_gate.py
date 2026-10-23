"""KSU rerank-gate and pipeline-branch tests (fakes only, no models)."""

from __future__ import annotations

import pytest

from src.generators import BaseGenerator, GenerationError
from src.ksu_gate import KSU_QA_SOURCE, dense_pool, rerank_and_gate
from src.pipeline import CANNOT_CONFIRM, AnswerResult, answer_question
from src.retrieval import Retrieved


_THRESHOLD = 0.40


def _ret(rtype: str, score: float, pos: int) -> Retrieved:
    return Retrieved(
        source_type=rtype,
        score=score,
        document=f"doc-{rtype}-{pos}",
        metadata={"q_id": pos, "answer": "ج خ"} if rtype == KSU_QA_SOURCE else {},
    )


class _FakeRetriever:
    def retrieve_split(
        self, query: str, k: int | None = None, university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        del query, k, university
        return self.qa_hits, self.doc_hits


class _FakeReranker:
    """Returns configured probabilities in pair order."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.pairs: list[tuple[str, str]] = []

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> list[float]:
        self.pairs = list(pairs)
        return self.probs


class _CountingGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__("fake", backoff_base=0.01, max_attempts=1)
        self.calls = 0

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        self.calls += 1
        return "fake answer", False

    def generate(self, system: str, prompt: str, label: str = "") -> str:
        return super().generate(system, prompt, label)


def _retriever(
    qa: list[float], docs: list[float] | None = None,
    doc_type: str = "ksu_pdf",
) -> _FakeRetriever:
    r = _FakeRetriever()
    r.qa_hits = [_ret(KSU_QA_SOURCE, s, i) for i, s in enumerate(qa)]
    r.doc_hits = [
        _ret(doc_type, s, 100 + i) for i, s in enumerate(docs or [])
    ]
    return r


class TestDensePool:
    def test_top_n_of_merged_sides(self) -> None:
        qa = [0.90, 0.87]
        docs = [0.93, 0.84, 0.83]
        pool = dense_pool(
            [_ret(KSU_QA_SOURCE, s, i) for i, s in enumerate(qa)],
            [_ret("ksu_pdf", s, i) for i, s in enumerate(docs)],
            pool_size=4,
        )
        assert [r.score for r in pool] == [0.93, 0.90, 0.87, 0.84]

    def test_empty_pool(self) -> None:
        assert dense_pool([], [], pool_size=6) == []


class TestRerankAndGate:
    def test_pass_when_top_above_threshold(self) -> None:
        r = _retriever([0.90], [0.88])
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([0.55, 0.70]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.passed is True
        assert res.top_score == pytest.approx(0.70)
        assert len(res.ranked) == 2

    def test_boundary_top_equals_threshold_passes(self) -> None:
        r = _retriever([0.90], [])
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([_THRESHOLD]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.passed is True

    def test_refuse_just_below_threshold(self) -> None:
        r = _retriever([0.90], [])
        top = _THRESHOLD - 1e-9
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([top]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.passed is False

    def test_probabilities_used_without_resquashing(self) -> None:
        r = _retriever([0.9], [])
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([0.5]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.top_score == pytest.approx(0.5)

    def test_reranked_order_desc(self) -> None:
        r = _retriever([0.90, 0.85], [0.87])
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([0.1, 0.9, 0.5]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.scores == pytest.approx([0.9, 0.5, 0.1])
        assert res.ranked[0].document == "doc-ksu_pdf-100"

    def test_pool_capped_at_pool_size(self) -> None:
        r = _retriever([0.9, 0.89, 0.88, 0.87, 0.86], [0.85, 0.84, 0.83])
        fake = _FakeReranker([0.9] * 6)
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits, fake, pool_size=6,
            threshold=_THRESHOLD,
        )
        assert res.pool_size == 6
        assert len(fake.pairs) == 6

    def test_empty_pool_refuses(self) -> None:
        r = _retriever([], [])
        res = rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits,
            _FakeReranker([]), pool_size=6, threshold=_THRESHOLD,
        )
        assert res.passed is False
        assert res.top_score == 0.0

    def test_predict_pairs_are_question_document(self) -> None:
        r = _retriever([0.9], [])
        fake = _FakeReranker([0.9])
        rerank_and_gate(
            "سؤال", r.qa_hits, r.doc_hits, fake, pool_size=6,
            threshold=_THRESHOLD,
        )
        assert fake.pairs == [("سؤال", "doc-ksu_qa-0")]


class TestPipelineKsu:
    def _fake_generator(self) -> _CountingGenerator:
        return _CountingGenerator()

    def test_refusal_returns_cannot_confirm_without_llm(self) -> None:
        r = _retriever([0.9], [])
        gen = self._fake_generator()
        res = answer_question(
            "سؤال", r, gen, university="ksu",
            reranker=_FakeReranker([0.1]),
        )
        assert res.refused is True
        assert res.answer == CANNOT_CONFIRM
        assert res.qa_items == [] and res.doc_items == []
        assert gen.calls == 0

    def test_missing_reranker_raises(self) -> None:
        r = _retriever([0.9], [])
        with pytest.raises(ValueError):
            answer_question("سؤال", r, self._fake_generator(), university="ksu")

    def test_pass_builds_qa_first_context(self) -> None:
        r = _retriever([0.90], [0.88])
        gen = self._fake_generator()
        res = answer_question(
            "سؤال", r, gen, university="ksu",
            reranker=_FakeReranker([0.60, 0.90]),
        )
        assert res.refused is False
        assert isinstance(res, AnswerResult)
        assert gen.calls == 1
        assert res.qa_items and res.qa_items[0].source_type == KSU_QA_SOURCE
        assert res.doc_items and res.doc_items[0].source_type == "ksu_pdf"
        assert res.context_text.index("س: doc-ksu_qa-0") < res.context_text.index(
            "doc-ksu_pdf-100"
        )

    def test_rerank_order_kept_within_each_side(self) -> None:
        r = _retriever([0.90, 0.89], [0.88, 0.87])
        res = answer_question(
            "سؤال", r, self._fake_generator(), university="ksu",
            reranker=_FakeReranker([0.2, 0.8, 0.9, 0.4]),
        )
        assert res.refused is False
        assert [i.document for i in res.qa_items] == [
            "doc-ksu_qa-1", "doc-ksu_qa-0",
        ]
        assert [i.document for i in res.doc_items][:1] == ["doc-ksu_pdf-100"]

    def test_caps_respected(self) -> None:
        r = _retriever(
            [0.90, 0.91, 0.92],
            [0.93, 0.94, 0.95, 0.96, 0.97],
        )
        res = answer_question(
            "سؤال", r, self._fake_generator(), university="ksu",
            reranker=_FakeReranker([0.99] * 6),
        )
        assert res.refused is False
        assert len(res.qa_items) == 1
        assert len(res.doc_items) == 4
