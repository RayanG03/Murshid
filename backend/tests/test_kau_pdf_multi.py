"""Unit tests for the multi-document KAU PDF ingestor (no model)."""

from pathlib import Path

from src.ingest.kau_pdf import (
    DOC_TITLES,
    LEGACY_DIAGNOSTICS_STEM,
    TRANSFER_GUIDE_TITLE,
    build_doc_chunks,
    build_pdf_chunks,
    chunk_page_blocks,
    discover_doc_pages,
    strip_page_furniture,
)
from src.data_prep.extract_kau_pages import looks_visual_order


def _make_layout(root: Path) -> tuple[Path, Path]:
    """Create a fake sources/ + data/ tree with two KAU PDFs transcribed."""
    sources = root / "sources"
    data = root / "data"
    kau = sources / "KAU"
    kau.mkdir(parents=True)

    # Two PDFs (stems only matter — the ingestor never opens the PDFs).
    (kau / "doc-a.pdf").write_bytes(b"%PDF-1.4 fake")
    (kau / "دليل التحويل الداخلي.pdf").write_bytes(b"%PDF-1.4 fake")

    # doc-a: committed transcription, 2 pages.
    doc_a = kau / "transcribed" / "doc-a"
    doc_a.mkdir(parents=True)
    (doc_a / "page_1.txt").write_text("السطر الأول\n\nالسطر الثاني", encoding="utf-8")
    (doc_a / "page_2.txt").write_text("محمد الصفحة الثانية", encoding="utf-8")

    # Transfer guide: NO committed folder -> legacy gemini fallback.
    legacy = data / "extracted" / "_diagnostics" / LEGACY_DIAGNOSTICS_STEM
    legacy.mkdir(parents=True)
    (legacy / "page_1.gemini.txt").write_text("نص موثّق", encoding="utf-8")

    # A third PDF with no transcription at all.
    (kau / "doc-missing.pdf").write_bytes(b"%PDF-1.4 fake")
    return sources, data


def test_discover_prefers_committed_over_legacy(tmp_path: Path) -> None:
    sources, data = _make_layout(tmp_path)
    docs = {stem: pages for stem, _, pages in discover_doc_pages(sources, data)}

    assert [p.name for p in docs["doc-a"]] == ["page_1.txt", "page_2.txt"]

    # Transfer guide has no committed folder -> legacy gemini files.
    assert [p.name for p in docs[LEGACY_DIAGNOSTICS_STEM]] == ["page_1.gemini.txt"]


def test_discover_marks_missing_docs(tmp_path: Path) -> None:
    sources, data = _make_layout(tmp_path)
    docs = {stem: pages for stem, _, pages in discover_doc_pages(sources, data)}
    assert docs["doc-missing"] == []


def test_discover_titles(tmp_path: Path) -> None:
    sources, data = _make_layout(tmp_path)
    titles = {stem: title for stem, title, _ in discover_doc_pages(sources, data)}
    assert titles[LEGACY_DIAGNOSTICS_STEM] == TRANSFER_GUIDE_TITLE
    assert titles["doc-a"] == "doc-a"  # unknown stems fall back to the stem


def test_committed_wins_when_both_exist(tmp_path: Path) -> None:
    sources, data = _make_layout(tmp_path)
    committed = sources / "KAU" / "transcribed" / LEGACY_DIAGNOSTICS_STEM
    committed.mkdir(parents=True)
    (committed / "page_1.txt").write_text("نسخة ملتزمة", encoding="utf-8")

    docs = {stem: pages for stem, _, pages in discover_doc_pages(sources, data)}
    assert [p.name for p in docs[LEGACY_DIAGNOSTICS_STEM]] == ["page_1.txt"]


def test_strip_page_furniture_header_and_footer() -> None:
    title = TRANSFER_GUIDE_TITLE
    text = (
        f"{title} 12/3 هـ\n"
        "فقرة أولى\n"
        "فقرة ثانية\n"
        "12\n"
    )
    cleaned = strip_page_furniture(text, title)
    assert cleaned == "فقرة أولى\nفقرة ثانية"
    # Numbers inside content lines are never touched.
    assert "12" in strip_page_furniture("المادة 12\n", title)


def test_build_doc_chunks_ids_and_metadata(tmp_path: Path) -> None:
    pages = []
    for no, body in ((1, "أ\n\nب\n\nج"), (2, "د")):
        page_file = tmp_path / f"page_{no}.txt"
        page_file.write_text(body, encoding="utf-8")
        pages.append(page_file)

    chunks = build_doc_chunks("وثيقة", pages, target=10_000, max_chars=20_000)

    # Blocks pack up to target, so each page yields exactly one chunk.
    assert len(chunks) == 2
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.metadata["doc"] == "وثيقة" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[0].display.startswith("وثيقة — صفحة 1\n")


def test_build_pdf_chunks_skips_and_reports(tmp_path: Path) -> None:
    sources, data = _make_layout(tmp_path)
    chunks, pages_per_stem, skipped = build_pdf_chunks(
        sources, data, target=10_000, max_chars=20_000
    )
    assert set(pages_per_stem) == {"doc-a", LEGACY_DIAGNOSTICS_STEM}
    assert pages_per_stem["doc-a"] == 2
    assert skipped == ["doc-missing"]
    assert all(c.metadata["chunk_id"] for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    # The transfer-guide chunk header uses the real document title.
    legacy_chunks = [c for c in chunks if c.metadata["doc"] == TRANSFER_GUIDE_TITLE]
    assert legacy_chunks and legacy_chunks[0].display.startswith(
        f"{TRANSFER_GUIDE_TITLE} — صفحة 1\n"
    )


def test_chunk_page_blocks_legacy_header_unchanged() -> None:
    """Historical stage-3 wrapper keeps its byte-identical header."""
    chunks = chunk_page_blocks("نص\n\nالمزيد", page=3, target=500, max_chars=900)
    assert chunks[0].startswith(f"{DOC_TITLES[LEGACY_DIAGNOSTICS_STEM]} — صفحة 3\n")


def test_looks_visual_order_detection() -> None:
    """Logical-order Arabic has word-initial 'ال'; reversed text does not."""
    logical = " ".join(
        [
            "المادة الأولى: يجوز التحويل بين الكليات وفق القواعد التنظيمية للطلاب.",
            "المادة الثانية: يتقدم الطالب بطلب إلى اللجنة المختصة في الجامعة.",
            "المادة الثالثة: تحفظ جميع الوثائق في الملف الأكاديمي للطالب الدائم.",
        ]
    )
    assert looks_visual_order(logical) is False
    assert looks_visual_order(logical[::-1]) is True
    # Too few Arabic words to judge -> never flags.
    assert looks_visual_order("كلمة واحدة فقط") is False
