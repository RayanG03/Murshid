"""Ingest backend/data/kau_qa.csv (371 verified pairs) into kau_qa_verified."""

import logging
from pathlib import Path

import pandas as pd

from src.chroma_store import COL_KAU_QA, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.qa")


def ingest_kau_qa(client, embedder: E5Embedder) -> int:
    """One record per verified pair; embeds the question (passage prefix).

    The answer, Q_ID and source URL ride along as metadata for citation.
    """
    settings = get_settings()
    csv_path = settings.data_dir / "kau_qa.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"[ingest] kau_qa_verified: {csv_path} not found - the existing "
            "collection was left untouched. backend/data/ is not in git; "
            "copy kau_qa.csv from the team drive."
        )
    df = pd.read_csv(csv_path)
    if len(df) != 371:
        raise SystemExit(f"Expected 371 verified KAU pairs, found {len(df)}")

    questions = df["question_original"].astype(str).tolist()
    logger.info("Embedding %d questions ...", len(questions))
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(q) for q in questions]
    )

    collection = rebuild_collection(client, COL_KAU_QA)
    added = add_batched(
        collection,
        ids=[f"kau_qa_{int(q)}" for q in df["Q_ID"]],
        documents=questions,
        metadatas=[
            {
                "type": "qa",
                "q_id": int(row.Q_ID),
                "answer": str(row.answer),
                "category": str(row.category),
                "match_status": str(row.match_status),
                "source_url": str(row.source_url),
            }
            for row in df.itertuples(index=False)
        ],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("kau_qa_verified: %d records.", added)
    return added
