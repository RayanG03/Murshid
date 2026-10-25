"""Hybrid retrieval: verified Q&A pairs first, then document chunks.

Collections are resolved per university from src.config.UNIVERSITY_
COLLECTIONS and are only ever opened if they already exist (missing or
empty = unavailable; no query path may create a collection).
"""

from dataclasses import dataclass

from src.chroma_store import get_existing_collection
from src.config import UNIVERSITY_COLLECTIONS, get_settings
from src.embeddings import E5Embedder
from src.lexical import LexicalIndexes


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
    """Searches a university's collections with Q&A priority per AGENTS.md."""

    def __init__(self, client, embedder: E5Embedder) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = get_settings()
        self._lexical = LexicalIndexes(client)

    def _query_collection(self, collection, embedding, k: int) -> list[Retrieved]:
        if collection is None or collection.count() == 0:
            return []
        res = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Retrieved(
                source_type=collection.name,
                score=1.0 - float(d),
                document=str(doc),
                metadata=dict(md),
            )
            for doc, md, d in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def retrieve(
        self, query: str, k: int | None = None, university: str = "kau"
    ) -> list[Retrieved]:
        """Return top-k results: Q&A hits ranked ahead of document chunks."""
        qa, docs = self.retrieve_split(query, k, university=university)
        return (qa + docs)[:k]

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
        for name in (qa_name, *doc_names):
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

        return side((qa_name,)), side(doc_names)
