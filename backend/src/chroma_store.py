"""ChromaDB persistent store: collections per university and source type."""

import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb

logger = logging.getLogger("capstone.chroma")

COL_KAU_QA = "kau_qa_verified"
COL_KAU_WEB = "kau_web"
COL_KAU_PDF = "kau_pdf"

ALL_KAU_COLLECTIONS = (COL_KAU_QA, COL_KAU_WEB, COL_KAU_PDF)

COL_KSU_QA = "ksu_qa"
COL_KSU_TXT = "ksu_txt"
COL_KSU_PDF = "ksu_pdf"

ALL_KSU_COLLECTIONS = (COL_KSU_QA, COL_KSU_TXT, COL_KSU_PDF)


def get_client(chroma_dir: Path) -> chromadb.api.ClientAPI:
    """Open (or create) the persistent ChromaDB client."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_existing_collection(
    client: chromadb.api.ClientAPI, name: str
) -> chromadb.api.Collection | None:
    """Return an already-existing collection, or None; never creates one.

    Resolution for serving paths must not have create side effects: a
    missing collection simply reads as unavailable to the caller.
    """
    try:
        return client.get_collection(name)
    except Exception:
        logger.info("Collection '%s' not found -> unavailable.", name)
        return None


def rebuild_collection(
    client: chromadb.api.ClientAPI, name: str
) -> chromadb.api.Collection:
    """Delete and recreate a collection (cosine space) for a clean rebuild."""
    try:
        client.delete_collection(name)
        logger.info("Deleted existing collection '%s'.", name)
    except Exception:
        pass
    collection = client.create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    logger.info("Created collection '%s'.", name)
    return collection


def add_batched(
    collection: chromadb.api.Collection,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    batch_size: int,
) -> int:
    """Add records in batches; returns the number added."""
    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=list(ids[start:end]),
            documents=list(documents[start:end]),
            metadatas=list(metadatas[start:end]),
            embeddings=[list(e) for e in embeddings[start:end]],
        )
    return total
