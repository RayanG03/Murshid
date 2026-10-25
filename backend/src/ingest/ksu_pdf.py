"""Chunk and ingest the KSU PDF documents into ksu_pdf.

Primary document: sources/KSU/اللائحة التنظيمية للشؤون الدراسية24.pdf — the
2024 consolidated edition of the 73-article regulation (board decision
(3)A/4/1445, session of 25/02/2024) plus its executive rules (updated
through 1446H). Every other PDF under sources/KSU/ is ingested with the
same machinery (unit chunking when regulation headings are found, page
chunking otherwise).

Extraction notes (stage diagnostics, 2024 regulation):
- The text layer uses Arabic presentation forms; NFKC folding restores
  standard letters losslessly. Lam-alef is NOT scrambled.
- Multi-token digit runs (dates, decision numbers) come out BIDI-reversed;
  decimals use ','. These stay as extracted (never "corrected").
- Pages 1-2 are empty (cover) and pages 3-6 are the TOC; front matter is
  skipped. The last line of each content page is a footer page number
  (dropped).

Text source per document (committed wins — fresh-clone rebuildable):
1. Committed per-page transcriptions: sources/KSU/transcribed/<stem>/
   page_NN.txt (produced by src.data_prep.extract_ksu_pages or OCR).
2. Fallback: the PDF text layer read directly at ingest time (historical
   behavior — byte-identical for the 2024 regulation).

Chunking: one chunk per heading unit (المادة / الباب / القسم / القاعدة
التنفيذية / الملحق) when the document yields heading units; otherwise one
chunk per page group ('<title> — صفحة <page>' heading). Oversized units
split at sub-item boundaries (bullet '•' lines, BIDI list markers '.N' /
'1.'): sub-items and single lines are never cut mid-line, and table units
(الملحق) split only BETWEEN rows. Every part fits both budgets
(src.config.chunk_max_chars and ksu_chunk_max_tokens, default 450) with
the heading repeated on every part.
"""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from src.chroma_store import (
    COL_KSU_PDF,
    add_batched,
    ensure_chunks,
    rebuild_collection,
)
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ingest.token_cap import split_to_fit
from src.text import clean_search_text, normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.ksu_pdf")

PDF_FILE_NAME = "اللائحة التنظيمية للشؤون الدراسية24.pdf"
DOC_TITLE = "اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية"

# PDF file stem -> document title used in chunk headings and citations.
# Unknown stems fall back to the stem itself; add nicer titles here as
# transcriptions are added.
KSU_DOC_TITLES: dict[str, str] = {
    "اللائحة التنظيمية للشؤون الدراسية24": DOC_TITLE,
}

# Per-document front-matter (cover + TOC) page count, skipped at ingest and
# reported as metadata.page_offset so citations print the printed page.
FRONT_MATTER_PAGES = 6
FRONT_MATTER_BY_STEM: dict[str, int] = {
    "اللائحة التنظيمية للشؤون الدراسية24": FRONT_MATTER_PAGES,
}

# Cover (empty pages 1-2) + table of contents (pages 3-6).
TOC_DOTS_RE = re.compile(r"\.{4,}")
FOOTER_NUM_RE = re.compile(r"^[0-9\u0660-\u0669]{1,2}$")
MARKER_RE = re.compile(
    r"^[•\-]$|^\.[0-9\u0660-\u0669]{1,2}$|^[0-9\u0660-\u0669]{1,2}[.)]$"
)
PAGE_NUM_RE = re.compile(r"page_(\d+)")

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


def extract_pages(
    pdf_path: Path, front_matter: int = FRONT_MATTER_PAGES
) -> list[list[tuple[int, str]]]:
    """Extract per-page lines as (page_number, line), NFKC-folded.

    The first ``front_matter`` pages are skipped; TOC dot-leader lines and
    one trailing footer page number per page are dropped.
    """
    doc = fitz.open(pdf_path)
    try:
        pages: list[list[tuple[int, str]]] = []
        for i, page in enumerate(doc):
            page_no = i + 1
            if page_no <= front_matter:
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


def load_pages_from_dir(
    pages_dir: Path, front_matter: int = 0
) -> list[list[tuple[int, str]]]:
    """Load committed page_NN.txt files into the extract_pages shape.

    Files are sorted by their embedded page number. Defensive re-cleaning
    (empty/TOC/footer lines) is idempotent for extractor output and also
    normalizes hand-made OCR pages. The first ``front_matter`` pages are
    skipped to mirror the PDF-layer path.
    """

    def page_no(path: Path) -> int:
        match = PAGE_NUM_RE.search(path.name)
        return int(match.group(1)) if match else 10**6

    pages: list[list[tuple[int, str]]] = []
    for page_file in sorted(pages_dir.glob("page_*.txt"), key=page_no):
        number = page_no(page_file)
        page_lines: list[tuple[int, str]] = []
        if number < 10**6 and number > front_matter:
            text = page_file.read_text(encoding="utf-8")
            lines = [
                ln.strip()
                for ln in text.splitlines()
                if ln.strip() and not TOC_DOTS_RE.search(ln)
            ]
            if lines and FOOTER_NUM_RE.match(lines[-1]):
                lines.pop()
            page_lines = [(number, ln) for ln in lines]
        pages.append(page_lines)
    return pages


def doc_title_for(stem: str) -> str:
    """Citation title for a PDF stem (explicit map, else the stem)."""
    return KSU_DOC_TITLES.get(stem, stem)


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
    title: str = DOC_TITLE,
) -> list[str]:
    """One chunk per unit, or marker/line-delimited parts with the heading.

    Char-only mode is the historical stage-1 behavior byte-for-byte. Token
    mode fits both budgets and slices oversized sub-items at line (row)
    boundaries only — a table row is never cut mid-line.
    """
    heading = f"{title} — {unit}"
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


def chunk_page(
    title: str,
    page_lines: list[tuple[int, str]],
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[str]:
    """Page-mode chunks for documents without regulation heading units.

    Heading '<title> — صفحة <page>' repeated on every part; parts fit both
    budgets and split only between lines.
    """
    if not page_lines:
        return []
    page = page_lines[0][0]
    heading = f"{title} — صفحة {page}"
    lines = [ln for _, ln in page_lines]
    return split_to_fit(
        heading,
        lines,
        "\n",
        max_chars,
        token_len,
        max_tokens,
        granularity="lines",
        raw_body="\n".join(lines),
    )


def build_doc_chunks(
    title: str,
    pages: list[list[tuple[int, str]]],
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
    page_offset: int = 0,
) -> tuple[list[PdfChunk], str]:
    """Build chunks for one document; returns (chunks, mode).

    mode is 'units' when regulation heading units were found (one chunk per
    unit) and 'pages' otherwise (one chunk per page group).
    """
    chunks: list[PdfChunk] = []
    units = build_units(pages)
    if units:
        for unit, page, body_lines in units:
            for part in chunk_unit(
                unit, body_lines, max_chars, token_len, max_tokens, title
            ):
                chunks.append(
                    PdfChunk(
                        display=part,
                        unit=unit,
                        page=page,
                        metadata={
                            "type": "pdf",
                            "doc": title,
                            "unit": unit,
                            "page": page,
                            "page_offset": page_offset,
                            "source_url": "",
                        },
                    )
                )
        return chunks, "units"

    for page_lines in pages:
        for part in chunk_page(title, page_lines, max_chars, token_len, max_tokens):
            page = page_lines[0][0]
            chunks.append(
                PdfChunk(
                    display=part,
                    unit="",
                    page=page,
                    metadata={
                        "type": "pdf",
                        "doc": title,
                        "unit": "",
                        "page": page,
                        "page_offset": page_offset,
                        "source_url": "",
                    },
                )
            )
    return chunks, "pages"


def build_pdf_chunks(
    pdf_path: Path,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[PdfChunk]:
    """Extract, group by unit, and chunk the 2024 regulation PDF (PDF layer)."""
    chunks, _mode = build_doc_chunks(
        DOC_TITLE,
        extract_pages(pdf_path, FRONT_MATTER_PAGES),
        max_chars,
        token_len,
        max_tokens,
        page_offset=FRONT_MATTER_PAGES,
    )
    return chunks


def discover_ksu_docs(sources_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Resolve (stem, title, pdf_path, pages_dir) for every KSU PDF.

    Documents are returned in sorted-stem order; ``pages_dir`` is the
    committed transcription folder (may not exist yet).
    """
    ksu_dir = sources_dir / "KSU"
    docs: list[tuple[str, str, Path, Path]] = []
    for pdf_path in sorted(ksu_dir.glob("*.pdf")):
        stem = pdf_path.stem
        docs.append(
            (stem, doc_title_for(stem), pdf_path, ksu_dir / "transcribed" / stem)
        )
    return docs


def resolve_doc_pages(
    stem: str, pdf_path: Path, pages_dir: Path
) -> tuple[list[list[tuple[int, str]]], str]:
    """Resolve a document's pages: committed transcriptions or the PDF layer.

    Returns (pages, source) with source in {'committed', 'pdf-layer',
    'none'}. Committed wins; the PDF layer is the byte-identical fallback.
    """
    front_matter = FRONT_MATTER_BY_STEM.get(stem, 0)
    if pages_dir.is_dir() and any(pages_dir.glob("page_*.txt")):
        return load_pages_from_dir(pages_dir, front_matter), "committed"
    if pdf_path.exists():
        try:
            return extract_pages(pdf_path, front_matter), "pdf-layer"
        except Exception as exc:  # corrupt/unreadable PDF must not kill ingest
            logger.error(
                "ksu_pdf: unreadable PDF for '%s' — treating as missing (%s).",
                stem, exc,
            )
    return [], "none"


def build_ksu_pdf_chunks(
    sources_dir: Path,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> tuple[list[PdfChunk], list[dict[str, object]]]:
    """Build chunks for every KSU document; returns (chunks, per-doc infos).

    Each info dict reports stem, title, chunk mode, page source, page and
    chunk counts — the ingestor logs one line per document so nothing is
    skipped silently.
    """
    all_chunks: list[PdfChunk] = []
    infos: list[dict[str, object]] = []
    for ordinal, (stem, title, pdf_path, pages_dir) in enumerate(
        discover_ksu_docs(sources_dir)
    ):
        pages, source = resolve_doc_pages(stem, pdf_path, pages_dir)
        if source == "none":
            infos.append(
                {
                    "stem": stem,
                    "title": title,
                    "mode": "-",
                    "source": "none",
                    "pages": 0,
                    "chunks": 0,
                }
            )
            continue
        page_count = sum(1 for page_lines in pages if page_lines)
        chunks, mode = build_doc_chunks(
            title,
            pages,
            max_chars,
            token_len,
            max_tokens,
            page_offset=FRONT_MATTER_BY_STEM.get(stem, 0),
        )
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"ksu_pdf_{ordinal:02d}_{chunk.page:02d}_{chunk_index:03d}"
            )
        all_chunks.extend(chunks)
        infos.append(
            {
                "stem": stem,
                "title": title,
                "mode": mode,
                "source": source,
                "pages": page_count,
                "chunks": len(chunks),
            }
        )
    return all_chunks, infos


def final_pdf_text(unit: str, chunk_text: str) -> str:
    """Variant-C stored text: bare unit label + cleaned body.

    The repeated 'title — unit' heading is shortened to just the article
    label (unit, e.g. "المادة الرابعة عشرة") and the tatweel / diacritic
    extraction artifacts are removed. Applied to BOTH the embedded text and
    the stored display text. The full title and unit stay in the chunk
    metadata for citations.
    """
    body = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else ""
    cleaned_unit = clean_search_text(unit)
    cleaned_body = clean_search_text(body)
    return cleaned_unit + "\n" + cleaned_body if cleaned_unit else cleaned_body


def ingest_ksu_pdf(client, embedder: E5Embedder) -> int:
    """Chunk and ingest every KSU document into ksu_pdf."""
    settings = get_settings()
    chunks, infos = build_ksu_pdf_chunks(
        settings.sources_dir,
        settings.chunk_max_chars,
        token_len=embedder.token_length,
        max_tokens=settings.ksu_chunk_max_tokens,
    )
    for info in infos:
        if info["source"] == "none":
            logger.warning(
                "ksu_pdf: no text for '%s' — commit transcriptions under "
                "sources/KSU/transcribed/%s/ (src.data_prep.extract_ksu_pages) "
                "or add the PDF first.",
                info["stem"], info["stem"],
            )
        else:
            logger.info(
                "ksu_pdf: %s mode=%s source=%s pages=%d chunks=%d",
                info["stem"], info["mode"], info["source"],
                info["pages"], info["chunks"],
            )
    # Unit chunks carry only the bare unit label as stored text (variant C);
    # page chunks keep their '<title> — صفحة <page>' heading (KAU style).
    texts = [
        final_pdf_text(c.unit, c.display) if c.unit else c.display for c in chunks
    ]
    logger.info(
        "ksu_pdf: %d chunks from %d documents.",
        len(chunks), sum(1 for i in infos if i["source"] != "none"),
    )
    ensure_chunks(
        COL_KSU_PDF, len(chunks),
        "expected PDFs or page_NN.txt files under sources/KSU/",
    )
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(t) for t in texts]
    )
    collection = rebuild_collection(client, COL_KSU_PDF)
    added = add_batched(
        collection,
        ids=[c.metadata["chunk_id"] for c in chunks],
        documents=texts,
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("ksu_pdf: %d records.", added)
    return added
