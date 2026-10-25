"""Chunk and ingest the KAU web markdown pages (backend/sources/KAU/web/)."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.chroma_store import (
    COL_KAU_WEB,
    add_batched,
    ensure_chunks,
    rebuild_collection,
)
from src.config import get_settings
from src.embeddings import E5Embedder
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.web")

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class WebChunk:
    """One ingestable chunk: display text plus citation metadata."""

    display: str
    metadata: dict = field(default_factory=dict)


def split_markdown(md_text: str) -> tuple[dict, list[tuple[str, str]]]:
    """Return (frontmatter dict, [(section_heading, section_body), ...]).

    Splits on '## ' headings. Content before the first '##' (H1 etc.) is
    attached to the first section if one exists, otherwise skipped: page H1s
    duplicate page_title.
    """
    fm_match = FRONTMATTER_RE.match(md_text)
    frontmatter: dict = {}
    body = md_text
    if fm_match:
        frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        body = md_text[fm_match.end():]

    matches = list(SECTION_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((m.group(1).strip(), body[start:end].strip()))
    return frontmatter, sections


def chunk_section(
    heading: str, body: str, page_title: str, max_chars: int
) -> list[str]:
    """Split one section into parts, heading attached to every part.

    Sections within the size limit stay whole; long sections split on blank
    lines (paragraphs). Repeated contact blocks stay inside their section.
    """
    prefix = f"{page_title} > {heading}"
    whole = f"{prefix}\n{body}"
    if len(whole) <= max_chars:
        return [whole]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = len(prefix)
    for para in paragraphs:
        extra = len(para) + 2
        if current and current_len + extra > max_chars:
            parts.append(prefix + "\n" + "\n\n".join(current))
            current, current_len = [], len(prefix)
        current.append(para)
        current_len += extra
    if current:
        parts.append(prefix + "\n" + "\n\n".join(current))
    return parts


def build_web_chunks(web_dir: Path, max_chars: int) -> list[WebChunk]:
    """Build chunks for every markdown page in web_dir."""
    chunks: list[WebChunk] = []
    for md_path in sorted(web_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        frontmatter, sections = split_markdown(text)
        page_title = str(frontmatter.get("page_title", md_path.stem))
        source_url = str(frontmatter.get("source_url", ""))
        if not sections:
            logger.warning("%s: no '##' sections found; skipping.", md_path.name)
            continue
        for heading, body in sections:
            for part in chunk_section(heading, body, page_title, max_chars):
                chunks.append(
                    WebChunk(
                        display=part,
                        metadata={
                            "type": "web",
                            "page_title": page_title,
                            "section": heading,
                            "source_url": source_url,
                        },
                    )
                )
    return chunks


def ingest_kau_web(client, embedder: E5Embedder) -> int:
    """Chunk and ingest all KAU web pages into kau_web."""
    settings = get_settings()
    web_dir = settings.sources_dir / "KAU" / "web"
    chunks = build_web_chunks(web_dir, settings.chunk_max_chars)
    logger.info("kau_web: %d chunks from %d pages.",
                len(chunks), len(list(web_dir.glob('*.md'))))
    ensure_chunks(COL_KAU_WEB, len(chunks), f"expected *.md files in {web_dir}")
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(c.display) for c in chunks]
    )
    collection = rebuild_collection(client, COL_KAU_WEB)
    added = add_batched(
        collection,
        ids=[f"kau_web_{i:04d}" for i in range(len(chunks))],
        documents=[c.display for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("kau_web: %d records.", added)
    return added
