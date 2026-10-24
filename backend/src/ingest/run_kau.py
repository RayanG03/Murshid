"""Stage 3 runner: ingest all KAU sources into ChromaDB.

Run from backend/:
  uv run python -X utf8 -m src.ingest.run_kau [--only qa|web|pdf]
"""

import logging
import sys

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ingest.kau_pdf import ingest_kau_pdf
from src.ingest.kau_qa import ingest_kau_qa
from src.ingest.kau_web import ingest_kau_web

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("capstone.ingest.run")


def main() -> None:
    """Ingest the requested (default: all) KAU collections and report sizes."""
    settings = get_settings()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)

    counts: dict[str, int] = {}
    if only in (None, "qa"):
        counts["kau_qa_verified"] = ingest_kau_qa(client, embedder)
    if only in (None, "web"):
        counts["kau_web"] = ingest_kau_web(client, embedder)
    if only in (None, "pdf"):
        counts["kau_pdf"] = ingest_kau_pdf(client, embedder)

    print("\n=== Ingestion report ===")
    for name, added in counts.items():
        live = client.get_collection(name).count()
        print(f"{name}: ingested={added} stored={live}")
    print("Tip: KSU collections are built separately: "
          "uv run python -X utf8 -m src.ingest.run_ksu")


if __name__ == "__main__":
    main()