"""Chunk and ingest all KAU PDF documents into kau_pdf.

Every PDF under sources/KAU/ is ingested from per-page transcription files
(page_NN.txt) committed under sources/KAU/transcribed/<pdf-stem>/ — so a
fresh clone rebuilds kau_pdf without any machine-local files.

Producing the transcriptions:
- Text-layer PDFs:  uv run python -X utf8 -m src.data_prep.extract_kau_pages
- Scanned PDFs:     OCR each page (Surya / Mistral OCR / Gemini) and save the
                    text as page_NN.txt in the same folder — see
                    sources/KAU/transcribed/README.md for the exact layout.

Legacy fallback: the human-verified Gemini transcription of
دليل التحويل الداخلي (backend/data/extracted/_diagnostics/دليل التحويل
الداخلي/page_*.gemini.txt) is still honored for that document when the
committed transcribed/ folder has no pages for it. A committed folder always
wins over the legacy path.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.chroma_store import COL_KAU_PDF, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.pdf")

# PDF file stem -> document title used in chunk headers and citations.
# Unknown stems fall back to the stem itself; add nicer titles here as
# transcriptions are added.
DOC_TITLES: dict[str, str] = {
    "دليل التحويل الداخلي": "دليل التحويل الداخلي بين الكليات",
}

TRANSFER_GUIDE_TITLE = DOC_TITLES["دليل التحويل الداخلي"]

# The one document with human-verified legacy transcriptions outside the repo.
LEGACY_DIAGNOSTICS_STEM = "دليل التحويل الداخلي"

PAGE_NUM_RE = re.compile(r"page_(\d+)")
FOOTER_NUM_RE = re.compile(r"^[0-9\u0660-\u0669]{1,3}$")


@dataclass
class PdfChunk:
    """One ingestable PDF chunk with page provenance."""

    display: str
    page: int
    metadata: dict = field(default_factory=dict)


def doc_page_chunks(
    page_text: str, title: str, page: int, target: int, max_chars: int
) -> list[str]:
    """Group blank-line-delimited blocks into chunks within size bounds.

    Table rows arrive as consecutive non-blank lines, so blocks stay whole
    (tables are never split mid-block). An oversized single block is kept
    intact rather than cut. Every chunk is prefixed with the per-document
    citation header '<title> — صفحة <page>'.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", page_text) if b.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        extra = len(block) + 2
        if current and current_len + extra > target:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += extra
        if current_len >= max_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
    if current:
        chunks.append("\n".join(current))
    header = f"{title} — صفحة {page}"
    return [f"{header}\n{c}" for c in chunks]


def chunk_page_blocks(
    page_text: str, page: int, target: int, max_chars: int
) -> list[str]:
    """Historical single-document wrapper around doc_page_chunks.

    Kept (byte-identical output) for the stage-3 unit tests and for callers
    that only ever handled the internal transfer guide.
    """
    return doc_page_chunks(
        page_text, TRANSFER_GUIDE_TITLE, page, target, max_chars
    )


def strip_page_furniture(text: str, title: str) -> str:
    """Drop repeated title headers and a trailing bare page-number line."""
    header_re = re.compile(rf"^{re.escape(title)}\s*[\d/]+\s*هـ\s*$")
    lines = [ln for ln in text.splitlines() if not header_re.match(ln.strip())]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and FOOTER_NUM_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def _page_files(folder: Path, pattern: str) -> list[Path]:
    """Page files sorted by their embedded page number (not lexicographically)."""

    def page_no(path: Path) -> int:
        match = PAGE_NUM_RE.search(path.name)
        return int(match.group(1)) if match else 10**6

    return sorted(folder.glob(pattern), key=lambda p: (page_no(p), p.name))


def discover_doc_pages(
    sources_dir: Path, data_dir: Path
) -> list[tuple[str, str, list[Path]]]:
    """Resolve (stem, title, page_files) for every KAU PDF, in sorted order.

    Resolution per PDF: committed sources/KAU/transcribed/<stem>/page_*.txt
    first; for the transfer guide, the legacy human-verified diagnostics
    transcription as fallback. Documents with no transcription resolve to an
    empty page list (reported and skipped by the ingestor).
    """
    kau_dir = sources_dir / "KAU"
    docs: list[tuple[str, str, list[Path]]] = []
    for pdf_path in sorted(kau_dir.glob("*.pdf")):
        stem = pdf_path.stem
        title = DOC_TITLES.get(stem, stem)
        pages: list[Path] = []
        transcribed = kau_dir / "transcribed" / stem
        if transcribed.is_dir():
            pages = _page_files(transcribed, "*.txt")
        if not pages and stem == LEGACY_DIAGNOSTICS_STEM:
            legacy = data_dir / "extracted" / "_diagnostics" / stem
            if legacy.is_dir():
                pages = _page_files(legacy, "*.gemini.txt")
        docs.append((stem, title, pages))
    return docs


def build_doc_chunks(
    title: str,
    page_files: list[Path],
    target: int,
    max_chars: int,
    doc_ordinal: int = 0,
) -> list[PdfChunk]:
    """Build chunks for one document from its per-page files.

    Chunk ids embed the document ordinal, page and chunk index so they stay
    unique across all documents in the collection.
    """
    chunks: list[PdfChunk] = []
    for page_file in page_files:
        match = PAGE_NUM_RE.search(page_file.name)
        if not match:
            logger.warning("Unreadable page number in %s — skipped.", page_file.name)
            continue
        page = int(match.group(1))
        text = page_file.read_text(encoding="utf-8")
        cleaned = strip_page_furniture(text, title)
        parts = doc_page_chunks(cleaned, title, page, target, max_chars)
        for chunk_index, part in enumerate(parts):
            chunks.append(
                PdfChunk(
                    display=part,
                    page=page,
                    metadata={
                        "type": "pdf",
                        "doc": title,
                        "page": page,
                        "source_url": "",
                        "chunk_id": f"kau_pdf_{doc_ordinal:02d}_{page:02d}_{chunk_index:03d}",
                    },
                )
            )
    return chunks


def build_pdf_chunks(
    sources_dir: Path, data_dir: Path, target: int, max_chars: int
) -> tuple[list[PdfChunk], dict[str, int], list[str]]:
    """Build chunks for every transcribed KAU document.

    Returns (chunks, pages_per_stem, skipped_stems) — skipped stems are KAU
    PDFs with no transcription available yet.
    """
    chunks: list[PdfChunk] = []
    pages_per_stem: dict[str, int] = {}
    skipped: list[str] = []
    for ordinal, (stem, title, page_files) in enumerate(
        discover_doc_pages(sources_dir, data_dir)
    ):
        if not page_files:
            skipped.append(stem)
            continue
        pages_per_stem[stem] = len(page_files)
        chunks.extend(
            build_doc_chunks(title, page_files, target, max_chars, doc_ordinal=ordinal)
        )
    return chunks, pages_per_stem, skipped


def ingest_kau_pdf(client, embedder: E5Embedder) -> int:
    """Chunk and ingest every transcribed KAU document into kau_pdf."""
    settings = get_settings()
    chunks, pages_per_stem, skipped = build_pdf_chunks(
        settings.sources_dir,
        settings.data_dir,
        settings.chunk_target_chars,
        settings.chunk_max_chars,
    )
    total_pages = sum(pages_per_stem.values())
    logger.info(
        "kau_pdf: %d chunks from %d documents (%d pages).",
        len(chunks), len(pages_per_stem), total_pages,
    )
    for stem in skipped:
        logger.warning(
            "kau_pdf: no transcription for '%s' — run "
            "src.data_prep.extract_kau_pages (text layer) or OCR its pages "
            "into sources/KAU/transcribed/%s/ first.",
            stem, stem,
        )
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(c.display) for c in chunks]
    )
    collection = rebuild_collection(client, COL_KAU_PDF)
    added = add_batched(
        collection,
        ids=[c.metadata["chunk_id"] for c in chunks],
        documents=[c.display for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("kau_pdf: %d records.", added)
    return added
