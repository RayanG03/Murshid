"""Keyword (BM25) search over the Chroma collections — no new dependency.

Dense e5 retrieval is good at meaning but can miss exact tokens that
matter in regulations: article numbers, GPA values, course codes, named
forms. This module builds a small in-memory BM25 index per collection
(from the documents already stored in Chroma) so the rerank gates can
consider keyword candidates the dense pool missed
(settings.hybrid_lexical_extra).
"""

import logging
import math
import re
import threading
from collections import Counter

from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.lexical")

_TOKEN_RE = re.compile(r"[ء-ي0-9A-Za-z]+")
_DIACRITICS_RE = re.compile(r"[ً-ْـ]")

STOPWORDS = frozenset(
    "في من على إلى الى عن مع ما ماذا هل كيف متى أين اين هو هي هذا هذه ذلك "
    "التي الذي الذين و او أو ثم لا لم لن قد كل بعد قبل عند بين أن ان إن "
    "كان كانت يكون تكون له لها لهم به بها فيه فيها وش ايش ابي ابغى".split()
)


def normalize_token(token: str) -> str:
    """Light Arabic normalization + prefix stripping for keyword matching."""
    t = _DIACRITICS_RE.sub("", token)
    t = re.sub("[إأآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    for prefix in ("وال", "بال", "فال", "كال", "لل", "ال"):
        if t.startswith(prefix) and len(t) - len(prefix) >= 3:
            t = t[len(prefix):]
            break
    return t


def tokenize(text: str) -> list[str]:
    """Normalized, stopword-free tokens."""
    raw = _TOKEN_RE.findall(normalize_digits_for_search(text))
    tokens = [normalize_token(t) for t in raw]
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


class Bm25Index:
    """Plain BM25 (k1=1.5, b=0.75) over one collection's documents."""

    def __init__(self, documents: list[str], metadatas: list[dict],
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.metadatas = metadatas
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(d)) for d in documents]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(documents)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (doc_index, score) with score > 0."""
        q_terms = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self._lens[i] / (self._avg or 1.0)
                )
                score += self._idf[term] * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda p: -p[1])
        return scored[:k]


class LexicalIndexes:
    """Lazily built BM25 index per collection name (thread-safe)."""

    def __init__(self, client) -> None:
        self._client = client
        self._indexes: dict[str, Bm25Index | None] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Bm25Index | None:
        with self._lock:
            if name not in self._indexes:
                self._indexes[name] = self._build(name)
            return self._indexes[name]

    def _build(self, name: str) -> Bm25Index | None:
        from src.chroma_store import get_existing_collection

        collection = get_existing_collection(self._client, name)
        if collection is None or collection.count() == 0:
            return None
        data = collection.get(include=["documents", "metadatas"])
        index = Bm25Index(
            [str(d) for d in data["documents"]],
            [dict(m or {}) for m in data["metadatas"]],
        )
        logger.info("BM25 index for '%s': %d documents.", name, len(index.documents))
        return index
