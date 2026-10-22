"""E5 embedding wrapper: loads the model once, applies E5 prefixes."""

import logging
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from src.config import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX

logger = logging.getLogger("capstone.embeddings")


class E5Embedder:
    """Wraps the multilingual-e5 model with query/passage prefixes.

    The model must be loaded exactly once per process (startup or script
    entry), never lazily per request.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model '%s' ...", model_name)
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_embedding_dimension())
        logger.info("Embedding model loaded (dim=%d).", self.dimension)

    def _encode(self, prefixed: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(prefixed),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Embed user questions with the E5 'query: ' prefix."""
        return self._encode([E5_QUERY_PREFIX + q for q in queries])

    def embed_passages(self, passages: Sequence[str]) -> list[list[float]]:
        """Embed documents/chunks with the E5 'passage: ' prefix."""
        return self._encode([E5_PASSAGE_PREFIX + p for p in passages])

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Raw token counts (no truncation, no special tokens).

        Uses the tokenizer of the already-loaded model; no download and no
        second model instance. SentenceTransformer additionally truncates at
        its max_seq_length (512) during encoding, so chunks must stay well
        below 512 including the passage/query prefix and CLS/SEP tokens.
        """
        encoded = self._model.tokenizer(list(texts), add_special_tokens=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def token_length(self, text: str) -> int:
        """Token count for one text (see token_lengths)."""
        return self.token_lengths([text])[0]
