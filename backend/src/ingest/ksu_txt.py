"""Chunk and ingest the KSU registrar text (sources/KSU/university documents.txt).

Layout of the source (a dump of KSU admission/registration web pages):
- "القسم الرئيسي :<name>" main sections (2: القبول, الطلاب).
- "الخدمة المطلوبة:(<name>)" service sections (23; the first occurrence is
  misspelled "الخدمة الطلوبة" in the source).
- Two FAQ blocks announced by parenthesized headings containing "شائع":
  "(الاساله الشائعة حول القبول)" (18 questions) and
  "(الأسئلة الشائعة حول المكافآت)" (7 questions). Each numbered question
  plus its answer becomes one chunk; a nested restart-at-1 list inside an
  answer stays attached to that answer. An oversized item splits on blank
  lines with the prefix and question line kept on every part.
- Any other section: whole when it fits, otherwise split on blank lines with
  the heading kept on every part (blocks are never cut mid-block).

Display text always stays as-is in the source; only the embedded copy gets
digit normalization (src.text.normalize_digits_for_search).
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.chroma_store import COL_KSU_TXT, add_batched, rebuild_collection
from src.config import get_settings
from src.embeddings import E5Embedder
from src.ingest.token_cap import split_to_fit
from src.text import normalize_digits_for_search

logger = logging.getLogger("capstone.ingest.ksu_txt")

MAIN_SECTION_RE = re.compile(r"^\s*القسم الرئيسي\s*[:：]\s*-?\s*(.+?)\s*$", re.M)
SERVICE_RE = re.compile(r"^\s*الخدمة\s*\S{0,12}?\s*[:：]\s*\(([^)]+)\)\s*$", re.M)
FAQ_HEADING_RE = re.compile(r"^\s*\(([^)]*شائع[^)]*)\)\s*$", re.M)
NUMBERED_RE = re.compile(r"^\s*(\d{1,2}|[\u0660-\u0669]{1,2})\s*[.)\-–:،]\s*(.*)$")

ARABIC_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

SOURCE_FILE_NAME = "university documents.txt"


@dataclass
class TxtChunk:
    """One ingestable chunk: display text plus provenance metadata."""

    display: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    """One service section: heading names plus its raw body text."""

    main_section: str
    service: str
    body: str


def parse_sections(text: str) -> list[Section]:
    """Split the dump into sections, one per service header.

    Header lines themselves are not part of any body. Content of a main
    section before its first service header (e.g. the القسم الفرعي line)
    becomes a section with an empty service name so nothing is lost.
    """
    sections: list[Section] = []
    main_matches = list(MAIN_SECTION_RE.finditer(text))
    for i, main in enumerate(main_matches):
        main_name = main.group(1).strip()
        main_end = (
            main_matches[i + 1].start() if i + 1 < len(main_matches) else len(text)
        )
        span = text[main.end():main_end]
        service_matches = list(SERVICE_RE.finditer(span))
        if not service_matches:
            body = span.strip()
            if body:
                sections.append(Section(main_name, "", body))
            continue
        preamble = span[: service_matches[0].start()].strip()
        if preamble:
            sections.append(Section(main_name, "", preamble))
        for j, svc in enumerate(service_matches):
            svc_end = (
                service_matches[j + 1].start()
                if j + 1 < len(service_matches)
                else len(span)
            )
            body = span[svc.end():svc_end].strip()
            sections.append(Section(main_name, svc.group(1).strip(), body))
    return sections


def split_numbered_items(body: str) -> list[tuple[str, str]]:
    """Split an FAQ block into (question_line, answer_text) items.

    A numbered line starts a new item only when it ends with the Arabic
    question mark '؟': every question line in the two FAQ blocks does, and
    nested restart-at-1 lists inside an answer never do, so they stay
    attached to their question.
    """
    lines = body.splitlines()
    items: list[tuple[str, str]] = []
    current_q: str | None = None
    answer: list[str] = []

    def flush() -> None:
        nonlocal current_q, answer
        if current_q is not None:
            items.append((current_q, "\n".join(answer).strip()))
        current_q, answer = None, []

    for line in lines:
        stripped = line.strip()
        if NUMBERED_RE.match(stripped) and stripped.endswith("؟"):
            flush()
            current_q = stripped
            continue
        if current_q is not None:
            answer.append(line)
    flush()
    return items


def _split_with_heading(
    heading: str,
    body: str,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[str]:
    """Group blank-line-delimited blocks into parts, heading on every part.

    Char-only mode is the historical stage-1 behavior byte-for-byte. Token
    mode packs blocks under both budgets; an oversized block is sliced at
    sentence/line boundaries with the text unchanged.
    """
    whole = f"{heading}\n{body}" if body else heading
    if token_len is None:
        # Historical char-only path (byte-identical to the stage-1 code).
        if len(whole) <= max_chars:
            return [whole]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        parts: list[str] = []
        current: list[str] = []
        current_len = len(heading)
        for para in paragraphs:
            extra = len(para) + 2
            if current and current_len + extra > max_chars:
                parts.append(heading + "\n" + "\n\n".join(current))
                current, current_len = [], len(heading)
            current.append(para)
            current_len += extra
        if current:
            parts.append(heading + "\n" + "\n\n".join(current))
        return parts

    # Token mode: char budget AND token budget.
    paragraphs = (
        [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if body
        else []
    )
    return split_to_fit(
        heading,
        paragraphs,
        "\n\n",
        max_chars,
        token_len,
        max_tokens,
        granularity="sentences",
        raw_body=body,
    )


def chunk_section(
    section: Section,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[TxtChunk]:
    """Build chunks for one section: FAQ items first, then plain paragraphs."""
    base_prefix = (
        f"{section.main_section} > {section.service}"
        if section.service
        else section.main_section
    )
    displays: list[str] = []
    faq_match = FAQ_HEADING_RE.search(section.body)
    if faq_match:
        before = section.body[: faq_match.start()].strip()
        if before:
            displays.extend(
                _split_with_heading(base_prefix, before, max_chars, token_len, max_tokens)
            )
        prefix = f"{base_prefix} > {faq_match.group(1).strip()}"
        faq_body = section.body[faq_match.end():].strip()
        items = split_numbered_items(faq_body)
        if items:
            for question, answer in items:
                heading = f"{prefix}\n{question}"
                displays.extend(
                    _split_with_heading(heading, answer, max_chars, token_len, max_tokens)
                )
        else:
            displays.extend(
                _split_with_heading(prefix, faq_body, max_chars, token_len, max_tokens)
            )
    elif section.body:
        displays.extend(
            _split_with_heading(base_prefix, section.body, max_chars, token_len, max_tokens)
        )

    return [
        TxtChunk(
            display=display,
            metadata={
                "type": "txt",
                "main_section": section.main_section,
                "service": section.service,
                "source_file": SOURCE_FILE_NAME,
                "source_url": "",
            },
        )
        for display in displays
    ]


def build_txt_chunks(
    txt_path: Path,
    max_chars: int,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 450,
) -> list[TxtChunk]:
    """Build chunks for the whole university documents dump."""
    text = txt_path.read_text(encoding="utf-8-sig")
    sections = parse_sections(text)
    chunks: list[TxtChunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, max_chars, token_len, max_tokens))
    return chunks


def ingest_ksu_txt(client, embedder: E5Embedder) -> int:
    """Chunk and ingest the KSU registrar text dump into ksu_txt."""
    settings = get_settings()
    txt_path = settings.sources_dir / "KSU" / SOURCE_FILE_NAME
    chunks = build_txt_chunks(
        txt_path,
        settings.chunk_max_chars,
        token_len=embedder.token_length,
        max_tokens=settings.ksu_chunk_max_tokens,
    )
    logger.info("ksu_txt: %d chunks.", len(chunks))
    embeddings = embedder.embed_passages(
        [normalize_digits_for_search(c.display) for c in chunks]
    )
    collection = rebuild_collection(client, COL_KSU_TXT)
    added = add_batched(
        collection,
        ids=[f"ksu_txt_{i:04d}" for i in range(len(chunks))],
        documents=[c.display for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("ksu_txt: %d records.", added)
    return added
