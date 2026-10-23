"""Read-only experiment: does cleaning ksu_pdf text restore score separation?

Damage report on the stored ksu_pdf search texts (tatweel U+0640, Arabic
diacritics U+064B-U+0652), then an in-memory re-embedding of the same
chunks under three variants:

  A: the stored text exactly as it lives in Chroma                   (baseline)
  B: A after clean_search_text (removes ONLY U+0640 and U+064B-U+0652)
  C: B with the repeated long heading shortened to the bare unit label
     (e.g. "المادة الرابعة عشرة", taken from metadata["unit"])

ksu_qa and ksu_txt texts are embedded once and reused unchanged in every
variant. Retrieval is a numpy cosine over freshly embedded vectors with
the exact ingestion pipeline (E5, "passage: " prefix, normalized) —
nothing is written to Chroma and no existing file is modified.

TUNE queries only, with two TEMPORARY overrides defined below:
- RELEVANT_EXTRA moves to RELEVANT
- EXCLUDED_FROM_ALL is excluded from both groups

Rerank uses BAAI/bge-reranker-v2-m3 whose predict() returns probabilities
for num_labels == 1; they are used as received (no extra sigmoid).

Console output is numbers-only (queries are labeled q1..q16 etc.); the
index-to-query mapping, the damage report and the 3 before/after sample
chunks are written to data/clean_diag_samples.md (UTF-8, Arabic allowed
only there).

run:  uv run python -X utf8 -m src.ksu_text_clean_diag
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sentence_transformers import CrossEncoder

from src.chroma_store import COL_KSU_PDF, COL_KSU_QA, COL_KSU_TXT
from src.chroma_store import get_client, get_existing_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    split_tune_test,
)
from src.ksu_rerank_diag import RERANKER_MODEL, roc_auc, separation

TATWEEL = "\u0640"
DIACRITICS = "".join(chr(cp) for cp in range(0x064B, 0x0653))
VARIANTS: tuple[str, ...] = ("A", "B", "C")

# TEMPORARY overrides for this experiment only (never written back).
RELEVANT_EXTRA = "متى تنزل نتائج الاختبارات النهائية؟"
EXCLUDED_FROM_ALL = "ما مواعيد التسجيل للفصل الصيفي في التقويم الجامعي؟"

TARGET_CHUNK_ID = "ksu_pdf_14_040"
TARGET_QUERY = "وش الفرق بين الدرجة المزدوجة والدرجة المشتركة؟"
SAMPLE_CHUNK_COUNT = 3
OUTPUT_NAME = "clean_diag_samples.md"
_RERANK_POOL = 6

_CLEAN_TABLE: dict[int, str | None] = {ord(TATWEEL): None}
_CLEAN_TABLE.update({cp: None for cp in range(0x064B, 0x0653)})


class _Reranker(Protocol):
    """Minimal CrossEncoder surface (predict returns probabilities)."""

    def predict(
        self, pairs: Sequence[Any], batch_size: int = 8,
        show_progress_bar: bool = False,
    ) -> Any: ...


def clean_search_text(text: str) -> str:
    """Remove ONLY U+0640 and U+064B-U+0652; nothing else is touched."""
    return text.translate(_CLEAN_TABLE)


def chunk_damage(doc: str) -> dict[str, int]:
    """Per-chunk tatweel and diacritic occurrence counts."""
    return {
        "tatweel": doc.count(TATWEEL),
        "diacritics": sum(doc.count(ch) for ch in DIACRITICS),
    }


def damage_stats(documents: Sequence[str]) -> dict[str, int]:
    """Chunk-level and total damage counts over a collection."""
    per_chunk = [chunk_damage(doc) for doc in documents]
    return {
        "n_chunks": len(documents),
        "chunks_with_tatweel": sum(1 for c in per_chunk if c["tatweel"] > 0),
        "chunks_with_diacritics": sum(1 for c in per_chunk if c["diacritics"] > 0),
        "total_tatweel": sum(c["tatweel"] for c in per_chunk),
        "total_diacritics": sum(c["diacritics"] for c in per_chunk),
    }


def variant_text(md: Mapping[str, Any], document: str, variant: str) -> str:
    """Search text of one chunk under the requested variant."""
    if variant == "A":
        return document
    cleaned = clean_search_text(document)
    if variant == "B":
        return cleaned
    unit = str(md.get("unit", ""))
    body = document.split("\n", 1)[1] if "\n" in document else document
    body = clean_search_text(body)
    return clean_search_text(unit) + "\n" + body if unit else cleaned


def tune_query_sets() -> tuple[list[str], list[str]]:
    """(RELEVANT, REFUSAL) TUNE queries with this file's TEMPORARY overrides."""
    relevant = list(split_tune_test(RELEVANT_KSU)[0])
    refusal = list(split_tune_test(OFF_TOPIC_KSU)[0])
    refusal += list(split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)[0])
    relevant = [q for q in relevant if q != EXCLUDED_FROM_ALL]
    refusal = [q for q in refusal if q not in (EXCLUDED_FROM_ALL, RELEVANT_EXTRA)]
    if RELEVANT_EXTRA not in relevant:
        relevant.append(RELEVANT_EXTRA)
    return relevant, refusal


def separation_summary(
    relevant_top1s: Sequence[float], refusal_top1s: Sequence[float]
) -> dict[str, Any]:
    """Numbers-only summary reusing roc_auc and separation."""
    margin, overlap = separation(relevant_top1s, refusal_top1s)
    return {
        "auc": roc_auc(relevant_top1s, refusal_top1s),
        "min_relevant_top1": min(relevant_top1s) if relevant_top1s else None,
        "max_refusal_top1": max(refusal_top1s) if refusal_top1s else None,
        "margin": margin,
        "overlap": overlap,
    }


def _load_collection(
    client: Any, name: str
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """(ids, documents, metadatas) of an existing collection."""
    col = get_existing_collection(client, name)
    if col is None:
        raise SystemExit(f"collection missing: {name}")
    res = col.get(include=["documents", "metadatas"])
    return (
        [str(i) for i in res["ids"]],
        [str(d) for d in res["documents"]],
        [dict(m) for m in res["metadatas"]],
    )


def _matrix(embedder: E5Embedder, texts: Sequence[str]) -> np.ndarray:
    """Unit-normalized E5 passage matrix (rows) for a corpus."""
    return np.asarray(embedder.embed_passages(texts), dtype=np.float32)


def _top1(query_vec: np.ndarray, mat: np.ndarray) -> float:
    """Highest cosine score of a query against a passage matrix."""
    return float(np.max(mat @ query_vec))


def _pdf_spread(query_vec: np.ndarray, pdf_mat: np.ndarray) -> float:
    """Top1 minus 10th-highest cosine across ksu_pdf chunks only."""
    scores = np.sort(pdf_mat @ query_vec)[::-1]
    return float(scores[0] - scores[min(9, len(scores) - 1)])


def _top_ids(
    query_vec: np.ndarray,
    parts: Sequence[tuple[np.ndarray, Sequence[str]]],
    k: int,
) -> list[tuple[float, str]]:
    """Top-k (score, id) over the merged parts (dense candidates)."""
    scored: list[tuple[float, str]] = []
    for mat, ids in parts:
        scores = mat @ query_vec
        scored.extend(zip(scores, ids))
    scored.sort(key=lambda pair: -float(pair[0]))
    return [(float(s), tid) for s, tid in scored[:k]]


def _rerank_probs(
    reranker: _Reranker, query: str, texts: Sequence[str],
) -> list[float]:
    """CrossEncoder probabilities for (query, text) pairs, as received."""
    probs = reranker.predict([(query, t) for t in texts])
    return [float(p) for p in np.asarray(probs).ravel()]


def main() -> None:
    """Run the full read-only experiment and print numbers only."""
    t_embed_fixed = 0.0
    t_embed_pdf = 0.0
    t_embed_q = 0.0
    t_rerank: dict[str, float] = {v: 0.0 for v in VARIANTS}
    settings = get_settings()
    client = get_client(settings.chroma_dir)

    qa_ids, qa_docs, _qa_meta = _load_collection(client, COL_KSU_QA)
    txt_ids, txt_docs, _txt_meta = _load_collection(client, COL_KSU_TXT)
    pdf_ids, pdf_docs, pdf_meta = _load_collection(client, COL_KSU_PDF)

    dmg = damage_stats(pdf_docs)

    relevant, refusal = tune_query_sets()
    queries = relevant + refusal
    q_labels = [f"q{i + 1}" for i in range(len(relevant))] + [
        f"f{i + 1}" for i in range(len(refusal))
    ]

    embedder = E5Embedder(settings.embedding_model_name)
    t0 = time.perf_counter()
    fixed: list[tuple[np.ndarray, list[str]]] = []
    for ids, docs in ((qa_ids, qa_docs), (txt_ids, txt_docs)):
        fixed.append((_matrix(embedder, docs), ids))
    t_embed_fixed = time.perf_counter() - t0

    variant_data: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}
    for v in VARIANTS:
        texts = [variant_text(m, d, v) for m, d in zip(pdf_meta, pdf_docs)]
        t0 = time.perf_counter()
        mat = _matrix(embedder, texts)
        t_embed_pdf += time.perf_counter() - t0
        variant_data[v] = (mat, pdf_ids, texts)

    t0 = time.perf_counter()
    q_vecs = np.asarray(embedder.embed_queries(queries), dtype=np.float32)
    t_embed_q = time.perf_counter() - t0

    dense_rel: dict[str, list[float]] = {v: [] for v in VARIANTS}
    dense_ref: dict[str, list[float]] = {v: [] for v in VARIANTS}
    spreads: dict[str, list[float]] = {v: [] for v in VARIANTS}
    rer_rel: dict[str, list[float]] = {v: [] for v in VARIANTS}
    rer_ref: dict[str, list[float]] = {v: [] for v in VARIANTS}
    rer_pair_counts = 0

    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    for qi, query in enumerate(queries):
        is_relevant = qi < len(relevant)
        for v in VARIANTS:
            pdf_mat, _pdf_ids, pdf_texts = variant_data[v]
            parts: list[tuple[np.ndarray, Sequence[str]]] = [
                (mat, ids) for mat, ids in fixed
            ]
            parts.append((pdf_mat, pdf_ids))
            dense_top1 = _top1(q_vecs[qi], pdf_mat)
            merged_top1 = max(
                [_top1(q_vecs[qi], pdf_mat)]
                + [_top1(q_vecs[qi], mat) for mat, _ids in fixed]
            )
            if is_relevant:
                spreads[v].append(_pdf_spread(q_vecs[qi], pdf_mat))
                dense_rel[v].append(merged_top1)
            else:
                dense_ref[v].append(merged_top1)
            top = _top_ids(q_vecs[qi], parts, _RERANK_POOL)
            id_to_text: dict[str, str] = dict(zip(pdf_ids, pdf_texts))
            id_to_text.update(zip(qa_ids, qa_docs))
            id_to_text.update(zip(txt_ids, txt_docs))
            cand_texts = []
            for _score, tid in top:
                if tid not in id_to_text:
                    raise SystemExit(f"unknown candidate id: {tid}")
                cand_texts.append(id_to_text[tid])
            t0 = time.perf_counter()
            probs = _rerank_probs(reranker, query, cand_texts)
            t_rerank[v] += time.perf_counter() - t0
            rer_pair_counts += len(cand_texts)
            top_rer = max(probs) if probs else 0.0
            if is_relevant:
                rer_rel[v].append(top_rer)
            else:
                rer_ref[v].append(top_rer)

    # Target-chunk probe: rerank probability under each variant.
    target_probs: dict[str, float] = {}
    if TARGET_CHUNK_ID in pdf_ids:
        id_pos = pdf_ids.index(TARGET_CHUNK_ID)
        for v in VARIANTS:
            _mat, _ids, texts = variant_data[v]
            t0 = time.perf_counter()
            target_probs[v] = _rerank_probs(
                reranker, TARGET_QUERY, [texts[id_pos]]
            )[0]
            t_rerank[v] += time.perf_counter() - t0

    out_path = settings.data_dir / OUTPUT_NAME
    _write_report(
        out_path, dmg, pdf_ids, pdf_docs, pdf_meta,
        relevant, refusal, q_labels,
        dense_rel, dense_ref, spreads, rer_rel, rer_ref,
        target_probs, t_rerank, rer_pair_counts,
    )

    print("=== DAMAGE REPORT (ksu_pdf) ===")
    print(
        f"chunks={dmg['n_chunks']} with_tatweel={dmg['chunks_with_tatweel']} "
        f"with_diacritics={dmg['chunks_with_diacritics']}"
    )
    print(f"total_tatweel={dmg['total_tatweel']} "
          f"total_diacritics={dmg['total_diacritics']}")
    print(f"samples(3) -> {out_path}")
    print()
    print("=== TUNE QUERIES ===")
    print(f"relevant={len(relevant)} refusal={len(refusal)} "
          f"(overrides applied; mapping in md)")
    print()
    print("=== DENSE VIEW top1 per query [q | A | B | C] ===")
    for i, label in enumerate(q_labels):
        cells = []
        for v in VARIANTS:
            source = dense_rel[v] if i < len(relevant) else dense_ref[v]
            idx = i if i < len(relevant) else i - len(relevant)
            cells.append(f"{source[idx]:.4f}")
        print(f"{label} " + " | ".join(cells))
    print("--- pdf-only spread (RELEVANT rows) [q | A | B | C] ---")
    for i in range(len(relevant)):
        cells = [f"{spreads[v][i]:.4f}" for v in VARIANTS]
        print(f"q{i + 1} " + " | ".join(cells))
    for v in VARIANTS:
        s = separation_summary(dense_rel[v], dense_ref[v])
        print(
            f"dense[{v}] auc={_fmt(s['auc'])} "
            f"min_rel={_fmt(s['min_relevant_top1'])} "
            f"max_ref={_fmt(s['max_refusal_top1'])} "
            f"margin={_fmt(s['margin'])} overlap={s['overlap']}"
        )
    print()
    print("=== RERANK pool-6 top1 per query [q | A | B | C] ===")
    for i, label in enumerate(q_labels):
        cells = []
        for v in VARIANTS:
            source = rer_rel[v] if i < len(relevant) else rer_ref[v]
            idx = i if i < len(relevant) else i - len(relevant)
            cells.append(f"{source[idx]:.4f}")
        print(f"{label} " + " | ".join(cells))
    for v in VARIANTS:
        s = separation_summary(rer_rel[v], rer_ref[v])
        print(
            f"rerank[{v}] auc={_fmt(s['auc'])} "
            f"min_rel={_fmt(s['min_relevant_top1'])} "
            f"max_ref={_fmt(s['max_refusal_top1'])} "
            f"margin={_fmt(s['margin'])} overlap={s['overlap']}"
        )
    print()
    print("=== TARGET CHUNK CHECK ===")
    if target_probs:
        row = " ".join(f"{target_probs[v]:.4f}" for v in VARIANTS)
        print(f"{TARGET_CHUNK_ID} A/B/C -> {row}")
    else:
        print(f"{TARGET_CHUNK_ID} absent from ksu_pdf")
    print()
    print("=== TIMINGS (seconds) ===")
    print(
        f"embed_fixed(qa+txt)={t_embed_fixed:.1f} "
        f"embed_pdf(A+B+C)={t_embed_pdf:.1f} embed_queries={t_embed_q:.1f}"
    )
    print("rerank queries=" + str(len(queries)) + " "
          + " ".join(f"{v}={t_rerank[v]:.1f}s" for v in VARIANTS))


def _fmt(x: float | None) -> str:
    """4-decimal number or n/a for None."""
    return "n/a" if x is None else f"{x:.4f}"


def _write_report(
    out_path: Path,
    dmg: Mapping[str, int],
    pdf_ids: Sequence[str],
    pdf_docs: Sequence[str],
    pdf_meta: Sequence[Mapping[str, Any]],
    relevant: Sequence[str],
    refusal: Sequence[str],
    q_labels: Sequence[str],
    dense_rel: Mapping[str, Sequence[float]],
    dense_ref: Mapping[str, Sequence[float]],
    spreads: Mapping[str, Sequence[float]],
    rer_rel: Mapping[str, Sequence[float]],
    rer_ref: Mapping[str, Sequence[float]],
    target_probs: Mapping[str, float],
    t_rerank: Mapping[str, float],
    rer_pair_counts: int,
) -> None:
    """Write the UTF-8 markdown report (Arabic allowed here, not console)."""
    n_rel = len(relevant)
    lines = ["# ksu_pdf text-clean diagnostic", "", "## Damage report", ""]
    for key in sorted(dmg):
        lines.append(f"- {key}: {dmg[key]}")
    lines += ["", "## Sample chunks (before / after clean_search_text)", ""]
    order = sorted(range(len(pdf_ids)), key=lambda i: pdf_ids[i])
    damaged = [
        i for i in order
        if chunk_damage(pdf_docs[i])["tatweel"]
        or chunk_damage(pdf_docs[i])["diacritics"]
    ]
    for pos, idx in enumerate(damaged[:SAMPLE_CHUNK_COUNT]):
        before = pdf_docs[idx]
        after = clean_search_text(before)
        lines += [
            f"### {pos + 1}. {pdf_ids[idx]}",
            "",
            "metadata: " + json.dumps(
                pdf_meta[idx], ensure_ascii=False, sort_keys=True
            ),
            "",
            "BEFORE:",
            "```",
            before,
            "```",
            "",
            "AFTER:",
            "```",
            after,
            "```",
            "",
        ]
    lines += ["", "## TUNE query mapping (overrides applied)", "", "### RELEVANT", ""]
    for i, q in enumerate(relevant):
        lines.append(f"- {q_labels[i]} | {q}")
    lines += ["", "### REFUSAL", ""]
    for j, q in enumerate(refusal):
        lines.append(f"- {q_labels[n_rel + j]} | {q}")
    lines += [
        "",
        "## Per-query top1 (merged pool dense; pool-6 rerank)",
        "",
        "| q | dense A | dense B | dense C | rerank A | rerank B | rerank C |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, label in enumerate(q_labels):
        is_rel = i < n_rel
        cells = []
        for v in VARIANTS:
            dense_src = dense_rel[v] if is_rel else dense_ref[v]
            rer_src = rer_rel[v] if is_rel else rer_ref[v]
            idx = i if is_rel else i - n_rel
            cells.append(f"{dense_src[idx]:.4f}")
            cells.append(f"{rer_src[idx]:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## ksu_pdf-only spread (top1 minus 10th), RELEVANT rows",
        "",
        "| q | A | B | C |",
        "|---|---|---|---|",
    ]
    for i in range(n_rel):
        cells = [f"{spreads[v][i]:.4f}" for v in VARIANTS]
        lines.append(f"| {q_labels[i]} | " + " | ".join(cells) + " |")
    lines += ["", "## Target chunk rerank probability (target query)", ""]
    if target_probs:
        for v in VARIANTS:
            lines.append(f"- {v}: {target_probs[v]:.4f}")
    else:
        lines.append(f"- {TARGET_CHUNK_ID} absent from ksu_pdf")
    lines += ["", "## Rerank timings", "", f"- pairs: {rer_pair_counts}"]
    for v in VARIANTS:
        lines.append(f"- {v}: {t_rerank[v]:.1f}s")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
