"""Read-only evidence dump for the TEST probes across the KSU collections.

For every query in the TEST slices of RELEVANT_KSU, DOMAIN_ADJACENT_WRONG_KSU
and OFF_TOPIC_KSU (split_tune_test, untouched), retrieve top-5 from
ksu_qa / ksu_txt / ksu_pdf via Retriever.retrieve_split, merge and sort by
score (scores are used for ordering ONLY and are never printed), then write
data/test_probe_evidence.md (UTF-8) with per query each hit's record id,
source type, the ksu_qa answer when present, and the first 400 characters
of the document. No gate metrics, no relevance judgments, no writes to
Chroma; the console shows counts only (no Arabic).

run:  uv run python -X utf8 ../data/test_probe_evidence.py
"""

from __future__ import annotations

from pathlib import Path

from src.chroma_store import get_client, get_existing_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    split_tune_test,
)
from src.retrieval import Retrieved, Retriever

OUT_NAME = "test_probe_evidence.md"
K = 5
DOC_PREVIEW_CHARS = 400

GROUPS: tuple[tuple[str, list[str]], ...] = (
    ("RELEVANT_TEST", split_tune_test(RELEVANT_KSU)[1]),
    ("DOMAIN_ADJACENT_WRONG_TEST", split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)[1]),
    ("OFF_TOPIC_TEST", split_tune_test(OFF_TOPIC_KSU)[1]),
)


def _id_lookup(client, source_type: str, document: str) -> str:
    """Chroma record id of the hit, found by exact full-document match."""
    col = get_existing_collection(client, source_type)
    if col is None:
        return "<collection missing>"
    res = col.get(where_document={"$contains": document}, include=["documents"])
    ids = [str(i) for i in res["ids"]]
    if not ids:
        return "<id not found>"
    return ids[0]


def _hit_block(
    client, pos: int, hit: Retrieved
) -> list[str]:
    """Markdown lines for one retrieved hit (no scores)."""
    rid = _id_lookup(client, hit.source_type, hit.document)
    lines = [
        f"- hit {pos}: id `{rid}` | source `{hit.source_type}`",
    ]
    answer = hit.metadata.get("answer")
    if answer is not None:
        lines += ["  - answer (ksu_qa):", "  ```", f"  {answer}", "  ```"]
    lines += [
        f"  - document[:{DOC_PREVIEW_CHARS}]:",
        "  ```",
        hit.document[:DOC_PREVIEW_CHARS],
        "```",
        "",
    ]
    return lines


def main() -> None:
    settings = get_settings()
    client = get_client(settings.chroma_dir)
    embedder = E5Embedder(settings.embedding_model_name)
    retriever = Retriever(client, embedder)

    lines = [
        "# TEST probe evidence (raw retrieval, top-5 merged)",
        "",
        f"k_per_collection={K}; per-query rows: merged hits, scores not shown.",
        "",
    ]
    total_queries = 0
    total_hits = 0
    for group_name, queries in GROUPS:
        for qi, query in enumerate(queries):
            qa_hits, doc_hits = retriever.retrieve_split(
                query, K, university="ksu"
            )
            merged = sorted(
                qa_hits + doc_hits, key=lambda h: -h.score
            )[:K]
            total_queries += 1
            total_hits += len(merged)
            lines += [
                f"## {group_name} {qi + 1}/{len(queries)}",
                "",
                f"query: {query}",
                "",
                f"hits: {len(merged)}",
                "",
            ]
            for pos, hit in enumerate(merged, start=1):
                lines += _hit_block(client, pos, hit)
            lines.append("")
    out_path = Path(__file__).parent / OUT_NAME
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"queries={total_queries} hits_total={total_hits}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
