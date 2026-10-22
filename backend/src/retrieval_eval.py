"""Hit@3 evaluation for KAU retrieval.

Methodology: self-retrieval over the 371 verified KAU Q&A questions. Each
question is used as a query against the full pipeline (Q&A collection plus
web/pdf chunks, Q&A priority); a hit means the query's own Q_ID appears in
the top-3 results. This validates the embedding + index + priority wiring.
A no-self variant (own Q_ID excluded) is also reported to show how often a
near-duplicate question covers the query.
"""

import logging

import pandas as pd

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.retrieval import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("capstone.retrieval_eval")

SAMPLE_CATEGORIES = 4


def main() -> None:
    """Run the Hit@3 evaluation and print example query/result pairs."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)

    df = pd.read_csv(settings.data_dir / "kau_qa.csv")
    questions = df["question_original"].astype(str).tolist()
    q_ids = df["Q_ID"].astype(int).tolist()

    hits1 = hits3 = 0
    hits3_no_self = 0
    for query, q_id in zip(questions, q_ids):
        results = retriever.retrieve(query, k=3)
        ids = [r.metadata.get("q_id") for r in results]
        if ids and ids[0] == q_id:
            hits1 += 1
        if q_id in ids:
            hits3 += 1
        if any(i == q_id for i in ids[1:]):
            hits3_no_self += 1

    n = len(questions)
    print(f"\n=== Hit@3 (self-retrieval, {n} queries) ===")
    print(f"Hit@1  : {hits1}/{n} = {hits1 / n:.3f}")
    print(f"Hit@3  : {hits3}/{n} = {hits3 / n:.3f}")
    print(f"Hit@3 excluding rank-1 self : {hits3_no_self}/{n} "
          f"(near-duplicate coverage, informational)")

    print("\n=== Example queries (top-3, Q&A priority) ===")
    step = max(1, n // 8)
    shown = 0
    for i in range(0, n, step):
        query = questions[i]
        results = retriever.retrieve(query, k=3)
        print(f"\nQ: {query}")
        for r in results:
            label = r.metadata.get("type", r.source_type)
            extra = (
                f"q_id={r.metadata.get('q_id')}"
                if label == "qa"
                else f"section={r.metadata.get('section') or r.metadata.get('page')}"
            )
            print(f"  [{label} {r.score:.3f} {extra}] {r.document[:90]}")
        shown += 1
        if shown >= 4:
            break

    print("\n=== Free-text queries against document chunks ===")
    for query in [
        "ما شروط التحويل الداخلي إلى كلية الحاسبات وتقنية المعلومات؟",
        "كيف أتواصل مع عمادة القبول والتسجيل؟",
    ]:
        results = retriever.retrieve(query, k=3)
        print(f"\nQ: {query}")
        for r in results:
            label = r.metadata.get("type", r.source_type)
            extra = (
                f"q_id={r.metadata.get('q_id')}"
                if label == "qa"
                else f"section={r.metadata.get('section') or r.metadata.get('page')}"
            )
            print(f"  [{label} {r.score:.3f} {extra}] {r.document[:90]}")


if __name__ == "__main__":
    main()
