"""Ingest backend/data/ksu_qa.csv (validated KSU pairs) into ksu_qa."""

import logging

import pandas as pd

from src.ingest.qa_validation import validate_qa, qa_record_id

from src.chroma_store import COL_KSU_QA, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.ksu_qa")



def ingest_ksu_qa(client, embedder: E5Embedder) -> int:
    """One record per row; embeds question_original (passage prefix).

    The answer, Q_ID and category ride along as metadata. No KAU audit
    columns and no source URL exist in ksu_qa.csv by construction.
    """
    settings = get_settings()
    csv_path = settings.data_dir / "ksu_qa.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"[ingest] ksu_qa: {csv_path} not found - the existing collection "
            "was left untouched. backend/data/ is not in git; copy ksu_qa.csv "
            "from the team drive."
        )
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    validate_qa(df, "ksu", settings.qa_expected_ksu_rows)

    questions = df["question_original"].astype(str).tolist()
    logger.info("Embedding %d questions ...", len(questions))
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(q) for q in questions]
    )

    collection = rebuild_collection(client, COL_KSU_QA)
    added = add_batched(
        collection,
        ids=[qa_record_id("ksu", q) for q in df["Q_ID"]],
        documents=questions,
        metadatas=[
            {"type": "qa", "q_id": row["Q_ID"], "answer": row["answer"],
             **{key: row.get(key, "") for key in (
                 "category", "match_status", "source_url", "evidence_status",
                 "source_locator", "source_codes", "verification_date", "university")}}
            for row in df.to_dict("records")
        ],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("ksu_qa: %d records.", added)
    return added
