"""Chunk and ingest the verified KAU PDF (دليل التحويل الداخلي) transcription."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.chroma_store import COL_KAU_PDF, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.pdf")

DOC_TITLE = "دليل التحويل الداخلي بين الكليات"
PAGE_HEADER_RE = re.compile(r"دليل التحويل الداخلي بين الكليات\s*[\d/]+\s*هـ")


@dataclass
class PdfChunk:
    """One ingestable PDF chunk with page provenance."""

    display: str
    page: int
    metadata: dict = field(default_factory=dict)


def chunk_page_blocks(
    page_text: str, page: int, target: int, max_chars: int
) -> list[str]:
    """Group blank-line-delimited blocks into chunks within size bounds.

    Table rows arrive as consecutive non-blank lines, so blocks stay whole
    (tables are never split mid-block). An oversized single block is kept
    intact rather than cut.
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
    header = f"{DOC_TITLE} — صفحة {page}"
    return [f"{header}\n{c}" for c in chunks]


def build_pdf_chunks(
    pages_dir: Path, target: int, max_chars: int
) -> list[PdfChunk]:
    """Build chunks from the corrected per-page transcription files."""
    chunks: list[PdfChunk] = []
    for page_file in sorted(pages_dir.glob("page_*.gemini.txt")):
        page_match = re.search(r"page_(\d+)", page_file.name)
        if not page_match:
            continue
        page = int(page_match.group(1))
        text = page_file.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines()
                 if not PAGE_HEADER_RE.search(ln)]
        cleaned = "\n".join(lines).strip()
        for part in chunk_page_blocks(cleaned, page, target, max_chars):
            chunks.append(
                PdfChunk(
                    display=part,
                    page=page,
                    metadata={
                        "type": "pdf",
                        "doc": DOC_TITLE,
                        "page": page,
                        "source_url": "",
                    },
                )
            )
    return chunks


def ingest_kau_pdf(client, embedder: E5Embedder) -> int:
    """Chunk and ingest the internal transfer guide into kau_pdf."""
    settings = get_settings()
    pages_dir = (
        settings.data_dir / "extracted" / "_diagnostics" / "دليل التحويل الداخلي"
    )
    chunks = build_pdf_chunks(
        pages_dir, settings.chunk_target_chars, settings.chunk_max_chars
    )
    logger.info("kau_pdf: %d chunks from 12 pages.", len(chunks))
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(c.display) for c in chunks]
    )
    collection = rebuild_collection(client, COL_KAU_PDF)
    added = add_batched(
        collection,
        ids=[f"kau_pdf_{c.page:02d}_{i:03d}" for i, c in enumerate(chunks)],
        documents=[c.display for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("kau_pdf: %d records.", added)
    return added
