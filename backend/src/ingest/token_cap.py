"""Token-budget splitting shared by the KSU chunkers (step 0).

Text is never edited: parts are slices joined exactly at the original
boundaries (block/line/sentence/word); no character is added, dropped or
moved (slices carry their own trailing separators). The heading (page/service
prefix, article number, or FAQ question line) is repeated verbatim on every
part. Table rows are line boundaries, so tables are only ever split BETWEEN
rows, never mid-row.

No model is imported here: callers pass a ``token_len`` callable (wired to
the embedder's tokenizer at ingest time) so tests can use a cheap fake
counter.
"""

import re
from collections.abc import Callable, Sequence

# Boundaries are cut positions (offsets) of a text, exclusive ends.
_SENTENCE_END_RE = re.compile(r"[.!؟?؛][ \t]+")
_WORD_RE = re.compile(r"\S+\s*")

# Granularity ladder per mode: pdf body splits at line (row) boundaries,
# txt paragraphs fall back to sentences; both fall back to words.
GRANULARITY_LADDERS: dict[str, tuple[str, ...]] = {
    "sentences": ("sentences", "lines", "words"),
    "lines": ("lines", "words"),
}

Measure = Callable[[str], int]


def _boundary_indices(text: str, granularity: str) -> list[int]:
    """Cut positions of ``text`` at the requested boundaries, ascending."""
    cuts: list[int] = []
    if granularity == "lines":
        cuts.extend(i + 1 for i, ch in enumerate(text) if ch == "\n")
    elif granularity == "sentences":
        cuts.extend(m.end() for m in _SENTENCE_END_RE.finditer(text))
        cuts.extend(i + 1 for i, ch in enumerate(text) if ch == "\n")
    elif granularity == "words":
        cuts.extend(m.end() for m in _WORD_RE.finditer(text))
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")
    return sorted(c for c in set(cuts) if c < len(text))


def _pack_sliced(
    text: str, cuts: list[int], fits: Callable[[str], bool]
) -> list[str]:
    """Greedy concatenation of [start:cut) intervals; concat == text."""
    points = [0, *cuts, len(text)]
    out: list[str] = []
    current = ""
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        candidate = current + text[start:end]
        if current and not fits(candidate):
            out.append(current)
            current = text[start:end]
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _slice_piece(
    text: str,
    fits: Callable[[str], bool],
    ladder: tuple[str, ...],
) -> list[str]:
    """Slice text at safe boundaries until every slice fits; concat == text."""
    if fits(text):
        return [text]
    for index, level in enumerate(ladder):
        cuts = _boundary_indices(text, level)
        packed = _pack_sliced(text, cuts, fits)
        if len(packed) == 1:
            continue  # no useful boundary at this level; try finer
        if all(fits(piece) for piece in packed):
            return packed
        rest = ladder[index + 1:]
        return [
            sub for piece in packed for sub in _slice_piece(piece, fits, rest)
        ]
    # No boundary anywhere helped (pathological): keep whole, never edit.
    return [text]


def split_to_fit(
    heading: str,
    blocks: Sequence[str],
    joiner: str,
    max_chars: int,
    token_len: Callable[[str], int] | None,
    max_tokens: int,
    granularity: str = "sentences",
    raw_body: str | None = None,
) -> list[str]:
    """Pack ``blocks`` under the char and (optional) token budgets.

    Returns ``[heading + "\n" + body]`` parts. Blocks that alone exceed the
    budget are sliced at safe boundaries (concatenation of a block's slices
    equals the block; each slice carries its original trailing separator).
    When ``token_len`` is None the pure char budget is used and an oversized
    block stays whole (the historical behavior). ``raw_body`` (optional) is
    the historically-shaped body returned verbatim when it already fits, so
    the char-only path stays byte-identical. The heading is measured
    together with the body so no part can overflow through the heading alone.
    """
    if not blocks:
        return [heading] if heading else []

    def body_fits(body: str) -> bool:
        candidate = heading + "\n" + body
        if len(candidate) > max_chars:
            return False
        if token_len is not None:
            return token_len(candidate) <= max_tokens
        return True

    if raw_body is not None and body_fits(raw_body):
        return [heading + "\n" + raw_body]

    whole_body = joiner.join(blocks)
    if body_fits(whole_body):
        return [heading + "\n" + whole_body]

    # Atomic units as (separator, text): continuation slices of one sliced
    # block rejoin with "" because they carry their original separator.
    units: list[tuple[str, str]] = []
    for block in blocks:
        if body_fits(block) or token_len is None:
            units.append((joiner, block))
        else:
            for i, slice_ in enumerate(
                _slice_piece(block, body_fits, GRANULARITY_LADDERS[granularity])
            ):
                units.append(("" if i else joiner, slice_))
    first_sep, first_text = units[0]
    units[0] = ("", first_text)

    parts: list[str] = []
    current: list[tuple[str, str]] = []

    def flush() -> None:
        parts.append(
            heading + "\n" + "".join(sep + text for sep, text in current)
        )

    for sep, text in units:
        unit = (sep, text)
        candidate = "".join(s + t for s, t in [*current, unit])
        if current and not body_fits(candidate):
            flush()
            current = [("", text)]  # part body starts fresh after the heading
        else:
            current.append(unit)
    if current:
        flush()
    return parts
