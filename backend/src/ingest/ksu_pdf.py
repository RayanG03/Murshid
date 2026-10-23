"""Chunk and ingest the current KSU academic-affairs regulation PDF.

Source: sources/KSU/اللائحة التنظيمية للشؤون الدراسية24.pdf — the 2024
consolidated edition of the 73-article regulation (board decision (3)A/4/1445,
session of 25/02/2024) plus its executive rules (updated through 1446H).

Extraction notes (Stage diagnostics):
- The text layer uses Arabic presentation forms; NFKC folding restores
  standard letters losslessly. Lam-alef is NOT scrambled.
- Multi-token digit runs (dates, decision numbers) come out BIDI-reversed;
  decimals use ','. These stay as extracted (never "corrected").
- Pages 1-2 are empty (cover) and pages 3-6 are the TOC; front matter is
  skipped. The last line of each content page is a footer page number
  (dropped).

Chunking: one chunk per heading unit (المادة / الباب / القسم / القاعدة
التنفيذية / الملحق). The PDF extracts without blank lines, so oversized
units split at sub-item boundaries (bullet '•' lines, BIDI list markers
'.N' / '1.'): sub-items and single lines are never cut mid-line, and table
units (الملحق) split only BETWEEN rows. Every part fits the model token
budget (src.config.ksu_chunk_max_tokens, default 450) with the heading
repeated on every part. The PDF page of the unit's first line is kept in
metadata (printed page = pdf page - 6).
"""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from src.chroma_store import COL_KSU_PDF, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ingest.token_cap import split_to_fit
from src.text import clean_search_text, normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.ksu_pdf")

PDF_FILE_NAME = "اللائحة التنظيمية للشؤون الدراسية24.pdf"
DOC_TITLE = "اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية"

# Cover (empty pages 1-2) + table of contents (pages 3-6).
FRONT_MATTER_PAGES = 6

TOC_DOTS_RE = re.compile(r"\.{4,}")
FOOTER_NUM_RE = re.compile(r"^[0-9\u0660-\u0669]{1,2}$")
MARKER_RE = re.compile(
    r"^[•\-]$|^\.[0-9\u0660-\u0669]{1,2}$|^[0-9\u0660-\u0669]{1,2}[.)]$"
)

_ORDINALS = (
    "العام|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|"
    "العاشر|الحادي عشر|الثاني عشر|الثالث عشر"
)
UNIT_RE = re.compile(
    rf"^(?:المادة\s+\S+(?:\s+\S+)?|(?:القسم|الباب)\s+(?:{_ORDINALS})\b[^\n]*|"
    r"القاعدة التنفيذية للمادة\s*\S+|القواعد التنفيذية\b[^\n]*|"
    r"\(?\d+\)?\s*الملحق|الملحق(?:\s+\S+)?)\s*[:：]?\s*$"
)


@dataclass
class PdfChunk:
    """One ingestable regulation chunk with unit and page provenance."""

    display: str
    unit: str
    page: int
    metadata: dict = field(default_factory=dict)


def extract_pages(pdf_path: Path) -> list[list[tuple[int, str]]]:
    """Extract per-page lines as (page_number, line), NFKC-folded.

    Front-matter pages are skipped; TOC dot-leader lines and one trailing
    footer page number per page are dropped.
    """
    doc = fitz.open(pdf_path)
    try:
        pages: list[list[tuple[int, str]]] = []
        for i, page in enumerate(doc):
            page_no = i + 1
            if page_no <= FRONT_MATTER_PAGES:
                pages.append([])
                continue
            lines = [
                (page_no, unicodedata.normalize("NFKC", ln.strip()))
                for ln in page.get_text().splitlines()
                if ln.strip() and not TOC_DOTS_RE.search(ln)
            ]
            if lines and FOOTER_NUM_RE.match(lines[-1][1]):
                lines.pop()
            pages.append(lines)
        return pages
    finally:
        doc.close()


def build_units(
    pages: list[list[tuple[int, str]]],
) -> list[tuple[str, int, list[str]]]:
    """Group lines into (unit_heading, start_page, body_lines) units.

    A unit starts at a standalone heading line; heading-only units (e.g.
    الباب/القسم immediately followed by an article) are dropped. Heading
    matching strips tatweel so stretched headings still match.
    """
    units: list[list] = []
    for page_lines in pages:
        if not page_lines:
            continue
        for page_no, line in page_lines:
            compact = re.sub(r"\s+", " ", line.replace("\u0640", "")).strip()
            if UNIT_RE.match(compact):
                units.append([re.sub(r"\s+", " ", line).strip(), page_no, []])
            elif units:
                units[-1][2].append(line)
    return [
        (str(u[0]), int(u[1]), u[2])
        for u in units
        if "\n".join(u[2]).strip()
    ]


def split_subitems(body_lines: list[str]) -> list[str]:
    """Group lines into sub-items at bullet/marker boundaries.

    Marker lines (•, '-', '.N', 'N.') start a new sub-item; lines and
    sub-items are never cut mid-line.
    """
    subitems: list[str] = []
    current: list[str] = []
    for line in body_lines:
        if MARKER_RE.match(line.strip()):
            if current:
                subitems.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        subitems.append("\n".join(current).strip())
    return [s for s in subitems if s]


def _pack_lines_char(line_list: list[str], max_chars: int) -> list[str]:
    """Split lines into packs of at most max_chars (historical behavior)."""
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in line_list:
        extra = len(line) + 1
        if current and current_len + extra > max_chars:
            parts.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += extra
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_unit(
    unit: str,
    body_lines: list[str],
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[str]:
    """One chunk per unit, or marker/line-delimited parts with the heading.

    Char-only mode is the historical stage-1 behavior byte-for-byte. Token
    mode fits both budgets and slices oversized sub-items at line (row)
    boundaries only — a table row is never cut mid-line.
    """
    heading = f"{DOC_TITLE} — {unit}"
    body = "\n".join(body_lines).strip()
    if token_len is None:
        # Historical char-only path (byte-identical to the stage-1 code).
        if len(heading) + len(body) + 1 <= max_chars:
            return [f"{heading}\n{body}"]
        parts: list[str] = []
        current: list[str] = []
        current_len = len(heading)
        for subitem in split_subitems(body_lines):
            pieces = [subitem]
            if len(subitem) > max_chars:
                pieces = _pack_lines_char(subitem.splitlines(), max_chars)
            for piece in pieces:
                extra = len(piece) + 2
                if current and current_len + extra > max_chars:
                    parts.append(heading + "\n" + "\n".join(current))
                    current, current_len = [], len(heading)
                current.append(piece)
                current_len += extra
        if current:
            parts.append(heading + "\n" + "\n".join(current))
        return parts

    # Token mode: char budget AND token budget, split at line (row) bounds.
    subitems = split_subitems(body_lines)
    if not subitems:
        return []
    return split_to_fit(
        heading,
        subitems,
        "\n",
        max_chars,
        token_len,
        max_tokens,
        granularity="lines",
        raw_body=body,
    )


def build_pdf_chunks(
    pdf_path: Path,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[PdfChunk]:
    """Extract, group by unit, and chunk the regulation PDF."""
    pages = extract_pages(pdf_path)
    units = build_units(pages)
    chunks: list[PdfChunk] = []
    for unit, page, body_lines in units:
        for part in chunk_unit(unit, body_lines, max_chars, token_len, max_tokens):
            chunks.append(
                PdfChunk(
                    display=part,
                    unit=unit,
                    page=page,
                    metadata={
                        "type": "pdf",
                        "doc": DOC_TITLE,
                        "unit": unit,
                        "page": page,
                        "source_url": "",
                    },
                )
            )
    return chunks


def final_pdf_text(unit: str, chunk_text: str) -> str:
    """Variant-C stored text: bare unit label + cleaned body.

    The repeated 'DOC_TITLE — unit' heading is shortened to just the
    article label (unit, e.g. "المادة الرابعة عشرة") and the tatweel /
    diacritic extraction artifacts are removed. Applied to BOTH the
    embedded text and the stored display text. The full title and unit
    stay in the chunk metadata for citations.
    """
    body = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else ""
    cleaned_unit = clean_search_text(unit)
    cleaned_body = clean_search_text(body)
    return cleaned_unit + "\n" + cleaned_body if cleaned_unit else cleaned_body


def ingest_ksu_pdf(client, embedder: E5Embedder) -> int:
    """Chunk and ingest the regulation into ksu_pdf."""
    settings = get_settings()
    pdf_path = settings.sources_dir / "KSU" / PDF_FILE_NAME
    chunks = build_pdf_chunks(
        pdf_path,
        settings.chunk_max_chars,
        token_len=embedder.token_length,
        max_tokens=settings.ksu_chunk_max_tokens,
    )
    logger.info(
        "ksu_pdf: %d chunks from %d units.", len(chunks), len({c.unit for c in chunks})
    )
    texts = [final_pdf_text(c.unit, c.display) for c in chunks]
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(t) for t in texts]
    )
    collection = rebuild_collection(client, COL_KSU_PDF)
    added = add_batched(
        collection,
        ids=[f"ksu_pdf_{c.page:02d}_{i:03d}" for i, c in enumerate(chunks)],
        documents=texts,
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("ksu_pdf: %d records.", added)
    return added
