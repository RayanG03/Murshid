"""Bounded evidence traces for local opt-in retrieval investigations."""
from collections.abc import Sequence
from src.retrieval import Retrieved

_METADATA_KEYS = (
    'source_id', 'type', 'university', 'doc', 'filename', 'page_title',
    'section', 'main_section', 'service', 'unit', 'page', 'page_offset',
    'source_url',
)


def evidence_trace(
    items: Sequence[Retrieved], scores: Sequence[float] | None = None,
    max_chars: int = 6000,
) -> list[dict]:
    """Copy source evidence only; never serialize settings or provider keys.

    These are retrieved candidate passages, not assertions that an answer
    is supported. A rerank score is not a calibrated probability of truth.
    """
    result = []
    for index, item in enumerate(items):
        row = {
            'rank': index + 1,
            'source_type': item.source_type,
            'retrieval_score': float(item.score),
            'metadata': {k: item.metadata[k] for k in _METADATA_KEYS if k in item.metadata},
            'document': item.document[:max_chars],
            'document_truncated_in_log': len(item.document) > max_chars,
        }
        if scores is not None:
            row['rerank_score'] = float(scores[index])
        result.append(row)
    return result
