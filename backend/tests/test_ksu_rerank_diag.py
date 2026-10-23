"""Tests for the KSU reranking diagnostic (pure helpers + fakes only).

No model downloads, no Chroma, no live calls; real components are only
built inside ksu_rerank_diag.main().
"""

import math

import pytest

from src.ksu_rerank_diag import (
    best_threshold,
    collect_rows,
    roc_auc,
    sigmoid,
    separation,
    summarize_scores,
    tune_queries,
)


class _FakeRetriever:
    """Records every query passed; returns fakes or an empty pool."""

    def __init__(self, pool_source=None) -> None:
        self.seen: list[str] = []
        self._pool_source = pool_source

    def retrieve_split(self, query: str, k: int, university: str):
        self.seen.append(query)
        if self._pool_source is None:
            return [], []
        return self._pool_source(query)


class _FakeHit:
    def __init__(self, source_type: str, score: float, document: str) -> None:
        self.source_type = source_type
        self.score = score
        self.document = document


class _FakeReranker:
    """Returns fixed outputs regardless of input pairs.

    The outputs mimic the real CrossEncoder.predict() contract for
    num_labels == 1: already-activated probabilities in [0, 1] (verified
    against sentence-transformers 6.0.1 — no extra sigmoid is applied).
    """

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self.pair_count = 0

    def predict(self, pairs, batch_size: int = 8, show_progress_bar: bool = False):
        self.pair_count = len(pairs)
        return list(self._probabilities[: len(pairs)])


def test_sigmoid_basic_and_extremes() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert sigmoid(-50.0) == pytest.approx(1.928749847e-22, rel=1e-6)
    assert sigmoid(50.0) == pytest.approx(1.0, abs=1e-9)
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)


def test_summarize_scores_full_pool() -> None:
    stats = summarize_scores([0.70, 0.90, 0.80, 0.65, 0.80])
    assert stats["top1"] == pytest.approx(0.90)
    assert stats["gap"] == pytest.approx(0.10)
    assert stats["spread"] == pytest.approx(0.90 - 0.65)


def test_summarize_scores_top10_window() -> None:
    scores = [1.0 - i * 0.01 for i in range(15)]  # 1.00 ... 0.86
    stats = summarize_scores(scores)
    assert stats["top1"] == pytest.approx(1.00)
    assert stats["gap"] == pytest.approx(0.01)
    assert stats["spread"] == pytest.approx(1.00 - 0.91)  # 10th candidate


def test_summarize_scores_single_candidate() -> None:
    stats = summarize_scores([0.62])
    assert stats["top1"] == pytest.approx(0.62)
    assert stats["gap"] == 0.0
    assert stats["spread"] == 0.0


def test_summarize_scores_empty_pool_does_not_crash() -> None:
    stats = summarize_scores([])
    assert stats == {"top1": 0.0, "gap": 0.0, "spread": 0.0}


def test_separation_clean() -> None:
    margin, overlap = separation([0.90, 0.85], [0.70, 0.60])
    assert margin == pytest.approx(0.85 - 0.70)
    assert overlap == 0


def test_separation_exact_overlap_counts_boundary() -> None:
    margin, overlap = separation([0.85, 0.90], [0.85, 0.60])
    assert margin == pytest.approx(0.0)
    assert overlap == 1  # >= lowest relevant (0.85) counts


def test_separation_full_overlap() -> None:
    margin, overlap = separation([0.80], [0.95, 0.88])
    assert margin == pytest.approx(-0.15)
    assert overlap == 2


def test_separation_empty_lists_do_not_crash() -> None:
    assert separation([], [0.5]) == (0.0, 0)
    assert separation([0.5], []) == (0.0, 0)


def test_roc_auc_perfect_separation_is_one() -> None:
    assert roc_auc([0.9, 0.8, 0.7], [0.6, 0.5]) == pytest.approx(1.0)


def test_roc_auc_full_overlap_equal_means_half() -> None:
    # identical multisets: half the pairs have rel > ref, half rel < ref
    assert roc_auc([0.9, 0.7], [0.9, 0.7]) == pytest.approx(0.5)


def test_roc_auc_inverted_is_zero() -> None:
    assert roc_auc([0.4, 0.3], [0.9, 0.8]) == pytest.approx(0.0)


def test_roc_auc_ties_count_half() -> None:
    # rel > ref for (0.8>0.5) only; ties (0.8,0.8) and (0.5,0.5) count 0.5 each
    assert roc_auc([0.8, 0.5], [0.8, 0.5]) == pytest.approx(
        (1.0 + 0.5 + 0.5 + 0.0) / 4.0
    )


def test_roc_auc_empty_side_is_chance() -> None:
    assert roc_auc([], [0.5]) == pytest.approx(0.5)
    assert roc_auc([0.5], []) == pytest.approx(0.5)
    assert roc_auc([], []) == pytest.approx(0.5)


def test_best_threshold_perfect_separation_full_recall() -> None:
    threshold, recall = best_threshold([0.9, 0.8], [0.6, 0.5])
    assert threshold == pytest.approx(math.nextafter(0.6, math.inf))
    assert recall == pytest.approx(1.0)


def test_best_threshold_tie_with_max_refusal_is_refused() -> None:
    # relevant 0.7 ties the max refusal 0.7 -> strictly below the 1-ULP
    # higher threshold -> refused; 0.9 passes -> recall 1/2
    threshold, recall = best_threshold([0.9, 0.7], [0.7, 0.5])
    assert threshold == pytest.approx(math.nextafter(0.7, math.inf))
    assert recall == pytest.approx(0.5)


def test_best_threshold_full_overlap_zero_recall() -> None:
    _threshold, recall = best_threshold([0.9, 0.7], [0.9, 0.7])
    assert recall == pytest.approx(0.0)


def test_best_threshold_empty_refusal_refuses_nothing() -> None:
    threshold, recall = best_threshold([0.8, 0.6], [])
    assert threshold == pytest.approx(math.nextafter(0.0, math.inf))
    assert recall == pytest.approx(1.0)


def test_best_threshold_empty_inputs_do_not_crash() -> None:
    threshold, recall = best_threshold([], [0.5])
    assert recall == 0.0
    threshold, recall = best_threshold([], [])
    assert recall == 0.0


def test_refusal_tie_at_threshold_refuses_all_refusal_queries() -> None:
    # threshold 1 ULP above max refusal refuses even exact-tie refusals
    threshold, _recall = best_threshold([0.9], [0.9, 0.7])
    assert 0.9 < threshold


TUNE_QUERIES = tune_queries()
RELEVANT_TUNE = TUNE_QUERIES["RELEVANT"]


def test_collect_rows_passes_only_tune_queries_to_retriever() -> None:
    retriever = _FakeRetriever()
    reranker = _FakeReranker([3.0, 1.0, -2.0])
    collect_rows(retriever, reranker, TUNE_QUERIES)
    assert len(retriever.seen) == sum(len(q) for q in TUNE_QUERIES.values())
    # Every retriever query is a TUNE slice member...
    all_tune = {q for qs in TUNE_QUERIES.values() for q in qs}
    assert set(retriever.seen) == all_tune
    # ...and no TEST-query-like content appears (test slice differs from tune).
    from src.ksu_probes import (
        DOMAIN_ADJACENT_WRONG_KSU,
        OFF_TOPIC_KSU,
        RELEVANT_KSU,
        split_tune_test,
    )

    test_pool: set[str] = set()
    for full in (RELEVANT_KSU, OFF_TOPIC_KSU, DOMAIN_ADJACENT_WRONG_KSU):
        tune, test = split_tune_test(full)
        test_pool |= set(test)
    assert not set(retriever.seen) & test_pool


def test_collect_rows_row_shape_and_rerank_order() -> None:
    def _pool(_query: str):
        return (
            [_FakeHit("ksu_qa", 0.88, "qa document")],
            [
                _FakeHit("ksu_pdf", 0.83, "regulation chunk A"),
                _FakeHit("ksu_txt", 0.81, "regulation chunk B"),
            ],
        )

    retriever = _FakeRetriever(_pool)
    reranker = _FakeReranker([0.0511, 0.9610, 0.9204])
    rows = collect_rows(retriever, reranker, {"RELEVANT": [RELEVANT_TUNE[0]]})
    assert len(rows) == 1
    row = rows[0]
    assert row["dense"]["top1"] == pytest.approx(0.88)
    # predict() probabilities are used AS RECEIVED — no extra sigmoid
    # (double activation previously compressed scores into [0.5, 0.731]).
    assert row["rerank"]["top1"] == pytest.approx(0.9610)
    assert row["rerank"]["top1"] != pytest.approx(sigmoid(0.9610))
    assert row["rerank"]["source_type"] == "ksu_pdf"
    assert row["rerank"]["preview"] == "regulation chunk A"
    assert reranker.pair_count == 3
    assert row["dense"]["spread"] == pytest.approx(0.88 - 0.81)
    # per-pool rows exist for every size and the 20-row aliases "rerank"
    assert row["rerank_20"]["top1"] == pytest.approx(0.9610)
    assert row["rerank_10"]["seconds"] >= 0.0
    assert "seconds" in row["rerank_6"]


def test_collect_rows_empty_pool_does_not_crash() -> None:
    retriever = _FakeRetriever()  # returns ([], [])
    reranker = _FakeReranker([])
    rows = collect_rows(retriever, reranker, {"RELEVANT": [RELEVANT_TUNE[0]]})
    assert rows[0]["rerank"]["top1"] == 0.0
    assert rows[0]["rerank"]["source_type"] == ""
    assert rows[0]["rerank"]["preview"] == ""
    assert reranker.pair_count == 0
