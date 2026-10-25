"""KSU rerank gate: merged dense pool -> cross-encoder scores -> threshold.

The gate is the KSU counterpart of the KAU floor/leader/gate quartet: the
top `pool_size` candidates by dense score (from the merged the qa+txt+pdf
retrieval sides) are reranked with a cross-encoder whose predict() returns
PROBABILITIES (num_labels==1; used as received, never re-squashed). The
request passes only if the top probability reaches the configured
threshold; otherwise the pipeline refuses without calling any LLM.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.retrieval import Retrieved

KSU_QA_SOURCE = "ksu_qa"


class RerankerLike(Protocol):
    """Minimal cross-encoder surface (predict returns probabilities)."""

    def predict(
        self, pairs: Sequence[Any], batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> Any: ...


@dataclass
class KsuGateResult:
    """Outcome of the KSU rerank gate for one request."""

    passed: bool
    top_score: float
    pool_size: int
    ranked: list[Retrieved]
    scores: list[float]
    rerank_seconds: float


def dense_pool(
    qa_results: Sequence[Retrieved], doc_results: Sequence[Retrieved],
    pool_size: int,
) -> list[Retrieved]:
    """Top-N candidates by dense cosine from the merged retrieval sides."""
    merged = sorted(
        list(qa_results) + list(doc_results),
        key=lambda r: -r.score,
    )
    return merged[:pool_size]


def _probability_list(raw: Any) -> list[float]:
    """Normalized float list from a predict() call (list/tuple/ndarray)."""
    values = list(raw) if hasattr(raw, "__len__") else [raw]
    return [float(v) for v in values]


def rerank_and_gate(
    question: str,
    qa_results: Sequence[Retrieved],
    doc_results: Sequence[Retrieved],
    reranker: RerankerLike,
    pool_size: int,
    threshold: float,
    extra_candidates: Sequence[Retrieved] = (),
) -> KsuGateResult:
    """Rerank the dense pool and apply the refusal threshold.

    Passing requires the TOP reranked probability >= threshold (ties with
    the threshold pass). An empty pool refuses with top 0.0. The ranked
    list is ordered by reranking (descending), stable for ties.
    """
    pool = dense_pool(qa_results, doc_results, pool_size)
    # Hybrid retrieval: keyword candidates the dense pool missed are added
    # (deduplicated by source + text) so the reranker can judge them too.
    seen = {(r.source_type, r.document) for r in pool}
    for cand in extra_candidates:
        key = (cand.source_type, cand.document)
        if key not in seen:
            seen.add(key)
            pool.append(cand)
    if not pool:
        return KsuGateResult(
            passed=False, top_score=0.0, pool_size=0,
            ranked=[], scores=[], rerank_seconds=0.0,
        )
    started = time.perf_counter()
    probs = _probability_list(
        reranker.predict([(question, c.document) for c in pool])
    )
    seconds = time.perf_counter() - started
    paired = sorted(
        zip(probs, pool), key=lambda pair: -pair[0],
    )
    scores = [p for p, _r in paired]
    top_score = scores[0] if scores else 0.0
    return KsuGateResult(
        passed=top_score >= threshold,
        top_score=top_score,
        pool_size=len(pool),
        ranked=[r for _p, r in paired],
        scores=scores,
        rerank_seconds=seconds,
    )


def filter_ranked(
    gate: KsuGateResult, min_score: float, relative: float,
) -> list[Retrieved]:
    """Ranked items worth sending to the LLM after a passing gate.

    Keeps items scoring >= max(min_score, relative * top_score); the top
    item always survives (it passed the gate threshold).
    """
    if not gate.ranked:
        return []
    floor = max(min_score, relative * gate.top_score)
    kept = [r for r, s in zip(gate.ranked, gate.scores) if s >= floor]
    return kept or gate.ranked[:1]
