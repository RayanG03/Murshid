"""Stage runner: ingest the KSU sources into ChromaDB.

Run from backend/:
  uv run python -X utf8 -m src.ingest.run_ksu [--only qa|txt|pdf]
"""

import logging
import sys

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ingest.ksu_pdf import ingest_ksu_pdf
from src.ingest.ksu_qa import ingest_ksu_qa
from src.ingest.ksu_txt import ingest_ksu_txt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("capstone.ingest.run_ksu")


def main() -> None:
    """Ingest the requested (default: all) KSU collections and report sizes."""
    settings = get_settings()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)

    counts: dict[str, int] = {}
    if only in (None, "qa"):
        counts["ksu_qa"] = ingest_ksu_qa(client, embedder)
    if only in (None, "txt"):
        counts["ksu_txt"] = ingest_ksu_txt(client, embedder)
    if only in (None, "pdf"):
        counts["ksu_pdf"] = ingest_ksu_pdf(client, embedder)

    print("\n=== KSU ingestion report ===")
    for name, added in counts.items():
        live = client.get_collection(name).count()
        print(f"{name}: ingested={added} stored={live}")

    # Token-length check of the stored display texts (embed budget safety).
    for name in counts:
        docs = client.get_collection(name).get(include=["documents"])["documents"]
        lens = sorted(embedder.token_lengths(docs))
        n = len(lens)
        p95 = lens[int(0.95 * (n - 1))] if n else 0
        over_512 = sum(1 for t in lens if t > 512)
        print(
            f"{name}: tokens max={lens[-1] if lens else 0} p95={p95} "
            f"over_512={over_512}"
        )


if __name__ == "__main__":
    main()
