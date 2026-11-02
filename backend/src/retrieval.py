"""Document-only retrieval by default; optional legacy QA comparison mode.

Collections are resolved per university from src.config.UNIVERSITY_
COLLECTIONS and are only ever opened if they already exist (missing or
empty = unavailable; no query path may create a collection).
"""

from dataclasses import dataclass
import logging
import threading

import numpy as np

from src.chroma_store import get_existing_collection
from src.config import UNIVERSITY_COLLECTIONS, get_settings
from src.embeddings import E5Embedder
from src.lexical import LexicalIndexes

logger = logging.getLogger("capstone.retrieval")


@dataclass
class Retrieved:
    """One retrieved item, normalized across source types."""

    source_type: str
    score: float
    document: str
    metadata: dict


def _university_collections(university: str) -> tuple[str, tuple[str, ...]]:
    """Validate the university and return (qa_collection, doc_collections)."""
    if university not in UNIVERSITY_COLLECTIONS:
        raise ValueError(
            f"Unknown university {university!r}; "
            f"expected one of {sorted(UNIVERSITY_COLLECTIONS)}"
        )
    names = UNIVERSITY_COLLECTIONS[university]
    return names[0], names[1:]


class Retriever:
    """Search only document collections unless legacy QA mode is selected."""

    def __init__(self, client, embedder: E5Embedder) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = get_settings()
        self._lexical = LexicalIndexes(client)
        # A broken HNSW reader must not discard the collection's evidence.
        # Serving data is immutable: restart after every external ingest/update.
        self._exact_collections: set[str] = set()
        self._exact_cache: dict[str, tuple[int, np.ndarray, list[Retrieved]]] = {}
        self._exact_lock = threading.RLock()

    def _query_collection(self, collection, embedding, k: int) -> list[Retrieved]:
        if collection is None or k <= 0:
            return []
        count = collection.count()
        if count == 0:
            return []
        with self._exact_lock:
            use_exact = collection.name in self._exact_collections
        if use_exact:
            return self._query_exact(collection, embedding, k, count)
        try:
            res = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            # Only this known index-reader failure permits the alternate path.
            # Authentication, permissions, dimensions and other errors propagate.
            message = str(exc).lower()
            if type(exc).__name__ != "InternalError" or not (
                "hnsw" in message and (
                    "nothing found on disk" in message
                    or "error loading hnsw index" in message
                )
            ):
                raise
            with self._exact_lock:
                if collection.name not in self._exact_collections:
                    logger.warning(
                        "[retrieval] HNSW unavailable for '%s'; using exact "
                        "cosine search over stored vectors (%d records). "
                        "No collection is removed or rebuilt.", collection.name, count,
                    )
                    self._exact_collections.add(collection.name)
            return self._query_exact(collection, embedding, k, count)
        return [
            Retrieved(
                source_type=collection.name,
                score=1.0 - float(d),
                document=str(doc),
                metadata={**dict(md or {}), "source_id": str(rid)},
            )
            for rid, doc, md, d in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _query_exact(
        self, collection, embedding, k: int, count: int,
    ) -> list[Retrieved]:
        """Dense cosine search without the HNSW reader, for this small corpus.

        Reads the persisted embeddings through get(), never computes a new
        embedding or calls a generator. Cache loads are serialized, then the
        immutable matrix is scored without a lock. Any incomplete/corrupt
        stored data fails visibly rather than silently reducing the evidence.
        """
        with self._exact_lock:
            cached = self._exact_cache.get(collection.name)
            if cached is None or cached[0] != count:
                vectors = []
                items: list[Retrieved] = []
                for offset in range(0, count, 512):
                    batch = collection.get(
                        limit=min(512, count - offset), offset=offset,
                        include=["documents", "metadatas", "embeddings"],
                    )
                    ids = batch["ids"]
                    documents = batch["documents"]
                    metadata = batch["metadatas"]
                    embeddings = batch["embeddings"]
                    if embeddings is None or not (
                        len(ids) == len(documents) == len(metadata) == len(embeddings)
                    ):
                        raise ValueError(f"Incomplete stored vectors in {collection.name}")
                    for rid, document, md, vector in zip(ids, documents, metadata, embeddings):
                        vectors.append(vector)
                        items.append(Retrieved(
                            source_type=collection.name, score=0.0,
                            document=str(document),
                            metadata={**dict(md or {}), "source_id": str(rid)},
                        ))
                if len(items) != count or len({i.metadata["source_id"] for i in items}) != count:
                    raise ValueError(f"Collection changed during read: {collection.name}; restart after ingest")
                matrix = np.asarray(vectors, dtype=np.float32)
                if matrix.ndim != 2 or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid stored vectors in {collection.name}")
                norms = np.linalg.norm(matrix, axis=1)
                if not np.isfinite(norms).all() or np.any(norms <= 0):
                    raise ValueError(f"Zero or invalid stored vectors in {collection.name}")
                matrix = matrix / norms[:, None]
                matrix.setflags(write=False)
                cached = (count, matrix, items)
                self._exact_cache[collection.name] = cached
        _, matrix, items = cached
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != matrix.shape[1] or not np.isfinite(query).all():
            raise ValueError("Query embedding dimension or values are invalid")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Query embedding must have a finite, positive norm")
        scores = np.clip(matrix @ (query / norm), -1.0, 1.0)
        order = np.argsort(-scores, kind="stable")[:min(k, len(items))]
        return [Retrieved(
            source_type=items[int(i)].source_type, score=float(scores[i]),
            document=items[int(i)].document, metadata=dict(items[int(i)].metadata),
        ) for i in order]

    def retrieve(
        self, query: str, k: int | None = None, university: str = "kau"
    ) -> list[Retrieved]:
        """Return top-k document results (QA-first only in explicit legacy mode)."""
        qa, docs = self.retrieve_split(query, k, university=university)
        return (qa + docs)[:k if k is not None else self._settings.retrieval_top_k]

    def retrieve_split(
        self,
        query: str,
        k: int | None = None,
        university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        """Return (qa_hits, doc_hits) independently, each side top-k.

        Used by context assembly so Q&A and document evidence are filtered
        and capped separately rather than merged into one ranked list.
        The query embedding is computed once for all collections.
        """
        k = k or self._settings.retrieval_top_k
        qa_name, doc_names = _university_collections(university)
        embedding = self._embedder.embed_queries([query])[0]
        qa = []
        if self._settings.evidence_mode == "qa_and_documents":
            qa = self._query_collection(
                get_existing_collection(self._client, qa_name), embedding, k
            )
        doc_hits: list[Retrieved] = []
        for name in doc_names:
            doc_hits.extend(
                self._query_collection(
                    get_existing_collection(self._client, name), embedding, k
                )
            )
        docs = sorted(doc_hits, key=lambda r: -r.score)
        return qa, docs[:k]

    def warm_lexical(self, university: str) -> None:
        """Build the BM25 indexes of a university up front (startup)."""
        qa_name, doc_names = _university_collections(university)
        names = doc_names if self._settings.evidence_mode == "documents_only" else (qa_name, *doc_names)
        for name in names:
            self._lexical.get(name)

    def lexical_split(
        self, query: str, k: int, university: str = "kau",
    ) -> tuple[list[Retrieved], list[Retrieved]]:
        """Keyword (BM25) hits as (qa_hits, doc_hits), each side top-k.

        ``score`` carries the raw BM25 score (NOT comparable to cosine);
        these candidates are only ever judged by the reranker.
        """
        qa_name, doc_names = _university_collections(university)

        def side(names: tuple[str, ...]) -> list[Retrieved]:
            hits: list[Retrieved] = []
            for name in names:
                index = self._lexical.get(name)
                if index is None:
                    continue
                for i, score in index.search(query, k):
                    hits.append(Retrieved(
                        source_type=name, score=score,
                        document=index.documents[i],
                        metadata=dict(index.metadatas[i]),
                    ))
            return sorted(hits, key=lambda r: -r.score)[:k]

        qa = side((qa_name,)) if self._settings.evidence_mode == "qa_and_documents" else []
        return qa, side(doc_names)
