"""KSU gate verification: every probe (all categories, TUNE and TEST) through
the production rerank gate. Numbers only.

run:  uv run python -X utf8 -m src.ksu_gate_eval
"""

from __future__ import annotations

import logging
import time

from sentence_transformers import CrossEncoder

from src.chroma_store import COL_KSU_QA, get_client, get_existing_collection
from src.config import get_settings, get_university_gates
from src.embeddings import E5Embedder
from src.ksu_gate import rerank_and_gate
from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    split_tune_test,
)
from src.retrieval import Retriever

logging.basicConfig(level=logging.WARNING)

CATEGORIES: tuple[tuple[str, list[str]], ...] = (
    ("RELEVANT", RELEVANT_KSU),
    ("DOMAIN-ADJACENT-WRONG", DOMAIN_ADJACENT_WRONG_KSU),
    ("OFF-TOPIC", OFF_TOPIC_KSU),
)


def main() -> None:
    """Run each probe through the new KSU gate and print the verdict."""
    settings = get_settings()
    gates = get_university_gates("ksu")
    embedder = E5Embedder(settings.embedding_model_name)
    reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
    retriever = Retriever(get_client(settings.chroma_dir), embedder)
    col = get_existing_collection(get_client(settings.chroma_dir), COL_KSU_QA)
    if col is None:
        raise SystemExit("ksu_qa collection missing")
    print(
        f"threshold={gates['rerank_threshold']} "
        f"pool_size={int(gates['rerank_pool_size'])} "
        f"collections={col.count()} ksu_qa"
    )
    t0 = time.perf_counter()
    wrong: dict[str, int] = {}
    for category, queries in CATEGORIES:
        tune, test = split_tune_test(queries)
        wrong[category] = 0
        for slice_name, qs in (("TUNE", tune), ("TEST", test)):
            print(f"--- {category} {slice_name} ---")
            for qi, query in enumerate(qs):
                qa, docs = retriever.retrieve_split(query, university="ksu")
                gate = rerank_and_gate(
                    query, qa, docs, reranker,
                    pool_size=settings.ksu_rerank_pool_size,
                    threshold=gates["rerank_threshold"],
                )
                verdict = "PASS" if gate.passed else "REFUSE"
                is_refusal_category = category != "RELEVANT"
                if gate.passed == is_refusal_category:
                    wrong[category] += 1
                print(
                    f"{verdict} | top={gate.top_score:.4f} | "
                    f"pool={gate.pool_size} | {query}"
                )
    print(f"\n== wrong verdicts: "
          + " ".join(f"{k}={v}" for k, v in wrong.items())
          + f" | total_seconds={time.perf_counter() - t0:.1f} ==")


if __name__ == "__main__":
    main()
