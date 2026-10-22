"""Ingest backend/data/ksu_qa.csv (all 574 KSU pairs) into ksu_qa."""

import logging

import pandas as pd

from src.chroma_store import COL_KSU_QA, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.ksu_qa")

EXPECTED_ROWS = 574


def ingest_ksu_qa(client, embedder: E5Embedder) -> int:
    """One record per row; embeds question_original (passage prefix).

    The answer, Q_ID and category ride along as metadata. No KAU audit
    columns and no source URL exist in ksu_qa.csv by construction.
    """
    settings = get_settings()
    csv_path = settings.data_dir / "ksu_qa.csv"
    df = pd.read_csv(csv_path)
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} KSU pairs, found {len(df)}")

    questions = df["question_original"].astype(str).tolist()
    logger.info("Embedding %d questions ...", len(questions))
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(q) for q in questions]
    )

    collection = rebuild_collection(client, COL_KSU_QA)
    added = add_batched(
        collection,
        ids=[f"ksu_qa_{int(q)}" for q in df["Q_ID"]],
        documents=questions,
        metadatas=[
            {
                "type": "qa",
                "q_id": int(row.Q_ID),
                "answer": str(row.answer),
                "category": str(row.category),
            }
            for row in df.itertuples(index=False)
        ],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("ksu_qa: %d records.", added)
    return added
