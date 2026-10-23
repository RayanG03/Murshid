"""KSU read-only diagnostic: does cross-encoder reranking separate?

KSU gate calibration found no clean threshold on TUNE because the dense
score distribution over the KSU regulation is flat (top scores 0.81-0.84
with no leader/gap structure). This script re-scores the TUNE probe pools
with a cross-encoder (BAAI/bge-reranker-v2-m3, CPU) and reports the same
statistics on both orderings: min RELEVANT top1, max REFUSAL top1, their
margin, overlap, ROC AUC, and the best single threshold.

SCALE NOTE (verified against sentence-transformers 6.0.1): CrossEncoder
predict() with activation_fn=None applies nn.Sigmoid() itself when
num_labels == 1 (BAAI/bge-reranker-v2-m3 is 1-label), so predict() returns
PROBABILITIES in [0, 1], not raw logits. These probabilities are used as
received; do NOT apply a second sigmoid (double activation compresses the
whole range into [0.5, 0.731]). The sigmoid() helper below is kept only for
predict calls made with activation_fn=nn.Identity (raw logits).

Experiment only: no existing behavior changes; nothing from retrieval,
config, prompts, or the TEST split is modified.

run:  uv run python -X utf8 -m src.ksu_rerank_diag
"""

import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from sentence_transformers import CrossEncoder

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    UNIVERSITY,
    split_tune_test,
)
from src.retrieval import Retrieved, Retriever

# Kept as a module-local constant on purpose: this is an experiment, not a
# serving component, so no Settings field is added for it.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_N = 10  # retrieval k per side (pools of up to 20) / spread window
POOL_SIZES: tuple[int, ...] = (20, 10, 6)  # reranked candidate-pool variants

RELEVANT_LABEL = "RELEVANT"
OFF_TOPIC_VIEW = "OFF-TOPIC"
DAW_VIEW = "DOMAIN-ADJACENT-WRONG"
VIEWS: tuple[tuple[str, str], ...] = (
    ("dense", "DENSE"),
    ("rerank_20", "RERANKED pool-20"),
    ("rerank_10", "RERANKED pool-10"),
    ("rerank_6", "RERANKED pool-6"),
)


class _RetrieverProtocol(Protocol):
    def retrieve_split(
        self, query: str, k: int, university: str
    ) -> tuple[list, list]: ...


class _RerankerProtocol(Protocol):
    def predict(
        self, pairs: Sequence[tuple[str, str]], batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> Any: ...


def sigmoid(x: float) -> float:
    """Logistic squashing of RAW logits to 0-1 (numerically safe).

    Only needed for predict() calls made with activation_fn=nn.Identity;
    the default predict() output is already a probability (see module doc).
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def summarize_scores(scores: Sequence[float]) -> dict[str, float]:
    """(top1, top1-top2 gap, top1 minus 10th) for one candidate pool.

    Sensible behavior on degenerate pools (must not crash):
    - empty pool: all zeros;
    - single candidate: top1 = that score, gap/spread = 0.0;
    - spread window truncates at the last available candidate.
    """
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return {"top1": 0.0, "gap": 0.0, "spread": 0.0}
    top1 = ordered[0]
    gap = top1 - ordered[1] if len(ordered) > 1 else 0.0
    spread = top1 - ordered[min(TOP_N, len(ordered)) - 1]
    return {"top1": top1, "gap": gap, "spread": spread}


def separation(
    relevant_top1s: Sequence[float], refusal_top1s: Sequence[float]
) -> tuple[float, int]:
    """(margin, overlap_count) between the two top-score distributions.

    margin = min(RELEVANT top1) - max(REFUSAL top1); positive margin means a
    clean score threshold separates the categories. overlap_count counts
    REFUSAL queries whose top1 reaches at least the lowest RELEVANT top1.
    Empty inputs yield (0.0, 0) and must not crash.
    """
    if not relevant_top1s or not refusal_top1s:
        return 0.0, 0
    lowest_relevant = min(relevant_top1s)
    margin = lowest_relevant - max(refusal_top1s)
    overlap = sum(1 for s in refusal_top1s if s >= lowest_relevant)
    return margin, overlap


def roc_auc(relevant_top1s: Sequence[float], refusal_top1s: Sequence[float]) -> float:
    """ROC AUC of RELEVANT-vs-REFUSAL on the top1 statistic (pure Python).

    Mann-Whitney style: for every (relevant, refusal) pair the relevant
    score counts 1 if larger, 0.5 on an exact tie; normalized by the pair
    count. Undefined for an empty side: returns 0.5 (chance level).
    """
    if not relevant_top1s or not refusal_top1s:
        return 0.5
    wins = 0.0
    for rel in relevant_top1s:
        for ref in refusal_top1s:
            if rel > ref:
                wins += 1.0
            elif rel == ref:
                wins += 0.5
    return wins / (len(relevant_top1s) * len(refusal_top1s))


def best_threshold(
    relevant_top1s: Sequence[float], refusal_top1s: Sequence[float]
) -> tuple[float, float]:
    """(threshold, RELEVANT recall) where a query passes iff top1 >= threshold.

    The threshold is one float ULP above the highest REFUSAL top1, so it
    refuses every REFUSAL query (ties included). RELEVANT queries tied
    exactly with the max REFUSAL score are (correctly) counted as refused.
    With no REFUSAL queries the threshold sits just above 0.0 and nothing
    is refused; with no RELEVANT queries recall is 0.0. Never crashes on
    empty inputs.
    """
    if refusal_top1s:
        threshold = math.nextafter(max(refusal_top1s), math.inf)
    else:
        threshold = math.nextafter(0.0, math.inf)
    if not relevant_top1s:
        return threshold, 0.0
    kept = sum(1 for s in relevant_top1s if s >= threshold)
    return threshold, kept / len(relevant_top1s)


def tune_queries() -> dict[str, list[str]]:
    """TUNE slice only (never TEST) per the ksu_probes seed-42 split."""
    relevant = split_tune_test(RELEVANT_KSU)[0]
    off = split_tune_test(OFF_TOPIC_KSU)[0]
    daw = split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)[0]
    return {
        RELEVANT_LABEL: relevant,
        OFF_TOPIC_VIEW: off,
        DAW_VIEW: daw,
    }


def _is_refusal_category(category: str) -> bool:
    return category in (OFF_TOPIC_VIEW, DAW_VIEW)


def _ramp_pool_by_dense(
    qa: list[Retrieved], docs: list[Retrieved]
) -> list[Retrieved]:
    """Merged candidate pool, dense-score-descending (top-20 first)."""
    return sorted(list(qa) + list(docs), key=lambda c: -c.score)


def _rerank_pool(
    reranker: _RerankerProtocol, query: str, pool: list[Retrieved]
) -> tuple[dict[str, Any], float]:
    """Score one pool with the cross-encoder; return (stats, seconds).

    predict() returns probabilities for this model (see module doc); they
    are used as received — no extra activation is applied.
    """
    t0 = time.perf_counter()
    pairs = [(query, c.document) for c in pool]
    probabilities = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    seconds = time.perf_counter() - t0
    scored = sorted(
        zip((float(p) for p in probabilities), pool),
        key=lambda pair: pair[0],
        reverse=True,
    )
    stats = summarize_scores([score for score, _ in scored])
    top_score, top_hit = scored[0]
    stats["source_type"] = top_hit.source_type
    stats["preview"] = top_hit.document[:100]
    return stats, seconds


def collect_rows(
    retriever: _RetrieverProtocol,
    reranker: _RerankerProtocol,
    categories: dict[str, list[str]],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """One diagnostic row per query: dense stats + per-pool rerank stats.

    The candidate pool is the merged qa+docs top-10/10 dense retrieval (up
    to 20); pools of 20/10/6 are the top-N of it by dense score, each
    reranked independently. Timings are recorded separately for the
    retrieval and the rerank step. progress (if provided) receives a line
    per query; real model objects are only constructed in main().
    """
    rows: list[dict[str, Any]] = []
    total = sum(len(qs) for qs in categories.values())
    done = 0
    for category, queries in categories.items():
        for query in queries:
            done += 1
            t0 = time.perf_counter()
            qa, docs = retriever.retrieve_split(
                query, k=TOP_N, university=UNIVERSITY
            )
            retrieval_seconds = time.perf_counter() - t0
            pool = _ramp_pool_by_dense(qa, docs)
            dense_stats = summarize_scores([c.score for c in pool])
            row: dict[str, Any] = {
                "category": category,
                "query": query,
                "retrieval_seconds": retrieval_seconds,
                "dense": dense_stats,
            }
            rerank_seconds = ""
            for size in POOL_SIZES:
                pool_n = pool[:size]
                if pool_n:
                    stats, seconds = _rerank_pool(reranker, query, pool_n)
                    rerank_seconds += f"{seconds:.1f}/"
                else:
                    stats, seconds = summarize_scores([]), 0.0
                    stats["source_type"] = ""
                    stats["preview"] = ""
                    rerank_seconds += "0.0/"
                stats["seconds"] = seconds
                row[f"rerank_{size}"] = stats
            row["rerank"] = row["rerank_20"]  # alias for pool-20
            rows.append(row)
            if progress is not None:
                progress(
                    f"[{done}/{total}] ret/re {retrieval_seconds:.1f}s "
                    f"rerank {rerank_seconds[:-1]}s (20/10/6) | {query[:44]}"
                )
    return rows


def _view_facts(rows: list[dict[str, Any]], view: str) -> dict[str, Any]:
    """Category-level metrics for one view ("dense" | "rerank_<size>")."""
    relevant = [r[view]["top1"] for r in rows if r["category"] == RELEVANT_LABEL]
    refusal = [r[view]["top1"] for r in rows if _is_refusal_category(r["category"])]
    rel_gaps = [r[view]["gap"] for r in rows if r["category"] == RELEVANT_LABEL]
    ref_gaps = [r[view]["gap"] for r in rows if _is_refusal_category(r["category"])]
    margin, overlap = separation(relevant, refusal)
    gap_margin, gap_overlap = separation(rel_gaps, ref_gaps)
    threshold, recall = best_threshold(relevant, refusal)
    wrongly_refused = [
        f"{r['query'][:52]} ({r[view]['top1']:.4f})"
        for r in rows
        if r["category"] == RELEVANT_LABEL and r[view]["top1"] < threshold
    ]
    return {
        "min_relevant_top1": min(relevant, default=0.0),
        "max_refusal_top1": max(refusal, default=0.0),
        "margin": margin,
        "overlap_count": overlap,
        "min_relevant_gap": min(rel_gaps, default=0.0),
        "max_refusal_gap": max(ref_gaps, default=0.0),
        "gap_margin": gap_margin,
        "gap_overlap_count": gap_overlap,
        "roc_auc_top1": roc_auc(relevant, refusal),
        "best_threshold_top1": threshold,
        "best_threshold_recall": recall,
        "wrongly_refused": wrongly_refused,
    }


def format_rows(rows: list[dict[str, Any]]) -> str:
    """Per-query table grouped by category (report section A)."""
    lines: list[str] = []
    for category in (RELEVANT_LABEL, OFF_TOPIC_VIEW, DAW_VIEW):
        lines.append(f"\n[{category}]")
        lines.append(
            "dense top1 | dense gap | rer20 top1 | rer sm | rer20-top1 source | "
            "preview of reranked top1 document"
        )
        for r in rows:
            if r["category"] != category:
                continue
            lines.append(
                f"{r['dense']['top1']:9.4f} | {r['dense']['gap']:9.4f} | "
                f"{r['rerank_20']['top1']:10.4f} | {r['rerank_20']['seconds']:6.1f} | "
                f"{r['rerank_20']['source_type']:<15} | {r['rerank_20']['preview']}"
            )
    lines.append(
        "\n(rer10/rer6 per-pool seconds: see the progress lines and the "
        "timing summary; their top1/gap/source/preview facts are in the "
        "per-view summaries and the JSON payload.)\n"
    )
    return "\n".join(lines)


def _timing_summary(rows: list[dict[str, Any]]) -> str:
    """Rerank-step seconds per query per pool size (retrieval excluded)."""
    lines = ["\n=== RERANK-STEP TIMINGS (seconds/query, retrieval excluded) ==="]
    total_retrieval = sum(r["retrieval_seconds"] for r in rows)
    lines.append(f"retrieval+merge mean/query : {total_retrieval / len(rows):.1f}s")
    for size in POOL_SIZES:
        view = f"rerank_{size}"
        secs = [r[view]["seconds"] for r in rows]
        pairs = min(size, 20)
        lines.append(
            f"{view} mean {statistics_mean(secs):.1f}s query — total "
            f"{sum(secs):.1f}s over {len(rows)} queries x up to {pairs} pairs"
        )
    return "\n".join(lines)


def statistics_mean(values: Sequence[float]) -> float:
    """Arithmetic mean (pure helper; empty -> 0.0)."""
    return sum(values) / len(values) if values else 0.0


def format_summary(rows: list[dict[str, Any]]) -> str:
    """View-level report (sections B and C) for every view."""
    lines: list[str] = []
    for view, title in VIEWS:
        facts = _view_facts(rows, view)
        lines.append(f"\n=== {title} (TUNE) ===")
        lines.append(f"min RELEVANT top1 : {facts['min_relevant_top1']:.4f}")
        lines.append(f"max REFUSAL top1 : {facts['max_refusal_top1']:.4f}")
        lines.append(f"margin top1      : {facts['margin']:+.4f}")
        lines.append(f"overlap top1     : {facts['overlap_count']}")
        lines.append(f"min RELEVANT gap : {facts['min_relevant_gap']:.4f}")
        lines.append(f"max REFUSAL gap  : {facts['max_refusal_gap']:.4f}")
        lines.append(f"margin gap       : {facts['gap_margin']:+.4f}")
        lines.append(f"overlap gap      : {facts['gap_overlap_count']}")
        lines.append(f"ROC AUC top1     : {facts['roc_auc_top1']:.4f}")
        lines.append(
            f"best threshold top1 : {facts['best_threshold_top1']:.17g} "
            f"-> RELEVANT recall {facts['best_threshold_recall']:.3f} "
            f"({len(facts['wrongly_refused'])} wrongly refused)"
        )
        if facts["wrongly_refused"]:
            for entry in facts["wrongly_refused"]:
                lines.append(f"  wrongly refused: {entry}")
        if facts["margin"] > 0.0:
            lines.append(f"{title} verdict: clean threshold exists")
        else:
            lines.append(f"{title} verdict: overlap of {facts['overlap_count']}")
    lines.append(_timing_summary(rows))
    return "\n".join(lines)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON payload: rows + per-view facts for data/ksu_rerank_diag.json."""
    facts = {
        key: _view_facts(rows, key) for key, _title in VIEWS
    }
    return {
        "reranker_model": RERANKER_MODEL,
        "top_n": TOP_N,
        "pool_sizes": list(POOL_SIZES),
        "n_queries": len(rows),
        "per_query": rows,
        "facts": facts,
    }


def main() -> None:
    """Build the real components once, then print + write the report."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    retriever = Retriever(get_client(settings.chroma_dir), embedder)
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    queries = tune_queries()
    rows = collect_rows(retriever, reranker, queries, progress=print)
    report = build_report(rows)
    out_path = settings.data_dir / "ksu_rerank_diag.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(format_rows(rows))
    print(format_summary(rows))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
