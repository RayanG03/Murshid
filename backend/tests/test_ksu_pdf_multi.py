"""Unit tests for the multi-document KSU PDF ingestor (no model)."""

from pathlib import Path

from src.ingest.ksu_pdf import (
    DOC_TITLE,
    FRONT_MATTER_PAGES,
    KSU_DOC_TITLES,
    build_doc_chunks,
    build_ksu_pdf_chunks,
    chunk_page,
    discover_ksu_docs,
    doc_title_for,
    final_pdf_text,
    load_pages_from_dir,
    resolve_doc_pages,
)
from src.ingest.token_cap import split_to_fit

REG_STEM = "اللائحة التنظيمية للشؤون الدراسية24"
FAKE_TOKEN_LEN = lambda text: max(1, len(text) // 3)  # noqa: E731


def _make_layout(root: Path) -> Path:
    """Create a fake sources/ tree with three KSU PDFs, two transcribed."""
    sources = root / "sources"
    ksu = sources / "KSU"
    ksu.mkdir(parents=True)

    # Regulation: committed transcription with unit headings.
    (ksu / f"{REG_STEM}.pdf").write_bytes(b"%PDF-1.4 fake")
    reg = ksu / "transcribed" / REG_STEM
    reg.mkdir(parents=True)
    (reg / "page_01.txt").write_text("", encoding="utf-8")  # cover
    (reg / "page_07.txt").write_text(
        "المادة الأولى :\nنص المادة الأولى\n",
        encoding="utf-8",
    )
    (reg / "page_10.txt").write_text(
        "المادة الثانية :\nنص المادة الثانية\n",
        encoding="utf-8",
    )

    # doc-b: committed transcription WITHOUT regulation headings.
    (ksu / "doc-b.pdf").write_bytes(b"%PDF-1.4 fake")
    doc_b = ksu / "transcribed" / "doc-b"
    doc_b.mkdir(parents=True)
    (doc_b / "page_1.txt").write_text("سطر عربي أول\nسطر عربي ثانٍ\n", encoding="utf-8")
    (doc_b / "page_2.txt").write_text("سطر الصفحة الثانية\n", encoding="utf-8")

    # doc-missing: PDF only, no transcription (PDF-layer fallback path).
    (ksu / "doc-missing.pdf").write_bytes(b"%PDF-1.4 fake")
    return sources


def test_doc_title_for_map_and_fallback() -> None:
    assert doc_title_for(REG_STEM) == DOC_TITLE
    assert KSU_DOC_TITLES[REG_STEM] == DOC_TITLE
    assert doc_title_for("وثيقة غير معروفة") == "وثيقة غير معروفة"


def test_discover_lists_all_pdfs_sorted(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    docs = discover_ksu_docs(sources)
    assert [stem for stem, _t, _p, _d in docs] == [
        "doc-b",
        "doc-missing",
        REG_STEM,
    ] or [stem for stem, _t, _p, _d in docs] == sorted(
        ["doc-b", "doc-missing", REG_STEM]
    )
    assert all(title == doc_title_for(stem) for stem, title, _p, _d in docs)


def test_load_pages_numeric_sort_and_front_matter(tmp_path: Path) -> None:
    _sources = _make_layout(tmp_path)
    reg = tmp_path / "sources" / "KSU" / "transcribed" / REG_STEM
    pages = load_pages_from_dir(reg, front_matter=FRONT_MATTER_PAGES)
    # page_01 (cover, <= front matter) skipped; numeric order kept (7 < 10).
    assert [lines[0][0] for lines in pages if lines] == [7, 10]
    assert pages[1][0][1] == "المادة الأولى :"


def test_resolve_prefers_committed(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    ksu = sources / "KSU"
    pages, source = resolve_doc_pages(
        REG_STEM, ksu / f"{REG_STEM}.pdf", ksu / "transcribed" / REG_STEM
    )
    assert source == "committed"
    assert pages and pages[1][0][0] == 7


def test_resolve_falls_back_to_pdf_layer(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    ksu = sources / "KSU"
    # No transcribed/ folder for doc-missing: pdf-layer path (no fitz parse
    # of the fake bytes here — assert the branch without reading the PDF by
    # checking a nonexistent pdf resolves to 'none').
    pages, source = resolve_doc_pages(
        "absent", ksu / "absent.pdf", ksu / "transcribed" / "absent"
    )
    assert source == "none" and pages == []


def test_unit_mode_chunks_from_committed(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    chunks, infos = build_ksu_pdf_chunks(
        sources, max_chars=10_000, token_len=FAKE_TOKEN_LEN, max_tokens=450
    )
    reg_info = next(i for i in infos if i["stem"] == REG_STEM)
    assert reg_info["mode"] == "units"
    assert reg_info["source"] == "committed"
    reg_chunks = [c for c in chunks if c.metadata["doc"] == DOC_TITLE]
    assert len(reg_chunks) == 2
    assert {c.unit for c in reg_chunks} == {"المادة الأولى :", "المادة الثانية :"}
    # Variant-C stored text: bare unit label, no repeated doc title.
    stored = final_pdf_text(reg_chunks[0].unit, reg_chunks[0].display)
    assert stored.startswith("المادة الأولى :\n")
    assert DOC_TITLE not in stored
    # Front-matter offset is metadata now (printed page = page - offset).
    assert all(c.metadata["page_offset"] == FRONT_MATTER_PAGES for c in reg_chunks)


def test_page_mode_fallback_without_units(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    chunks, infos = build_ksu_pdf_chunks(
        sources, max_chars=10_000, token_len=FAKE_TOKEN_LEN, max_tokens=450
    )
    info = next(i for i in infos if i["stem"] == "doc-b")
    assert info["mode"] == "pages"
    page_chunks = [c for c in chunks if c.metadata["doc"] == "doc-b"]
    assert len(page_chunks) == 2
    assert page_chunks[0].display.startswith("doc-b — صفحة 1\n")
    assert page_chunks[0].metadata["unit"] == ""
    assert page_chunks[0].metadata["page_offset"] == 0


def test_missing_document_is_reported_not_silent(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    _chunks, infos = build_ksu_pdf_chunks(
        sources, max_chars=10_000, token_len=FAKE_TOKEN_LEN, max_tokens=450
    )
    missing = next(i for i in infos if i["stem"] == "doc-missing")
    assert missing["source"] == "none" and missing["chunks"] == 0


def test_chunk_ids_unique_across_docs(tmp_path: Path) -> None:
    sources = _make_layout(tmp_path)
    chunks, _infos = build_ksu_pdf_chunks(
        sources, max_chars=10_000, token_len=FAKE_TOKEN_LEN, max_tokens=450
    )
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids and len(ids) == len(set(ids))
    assert all(i.startswith("ksu_pdf_") for i in ids)


def test_chunk_page_splits_within_budget() -> None:
    lines = [f"سطر رقم {i} مع بعض النص العربي الإضافي" for i in range(40)]
    page_lines = [(3, ln) for ln in lines]
    parts = chunk_page(
        "وثيقة", page_lines, max_chars=200, token_len=FAKE_TOKEN_LEN, max_tokens=30
    )
    assert len(parts) > 1
    assert all(p.startswith("وثيقة — صفحة 3\n") for p in parts)
    # No line ever cut mid-line: every source line appears whole somewhere.
    joined = "\n".join(parts)
    for ln in lines:
        assert ln in joined


def test_build_doc_chunks_pages_mode_concatenates_source(tmp_path: Path) -> None:
    lines = ["أول", "ثانٍ", "ثالث"]
    chunks, mode = build_doc_chunks(
        "وثيقة",
        [[(2, ln) for ln in lines]],
        max_chars=10_000,
        token_len=None,
        max_tokens=450,
    )
    assert mode == "pages" and len(chunks) == 1
    body = chunks[0].display.split("\n", 1)[1]
    assert body == "\n".join(lines)


def test_split_to_fit_lines_granularity_no_mid_line_cut() -> None:
    blocks = ["س" * 50, "ص" * 50]
    parts = split_to_fit(
        "عنوان", blocks, "\n", max_chars=60, token_len=None, max_tokens=10**6,
        granularity="lines",
    )
    assert len(parts) >= 2
    assert all(p.startswith("عنوان\n") for p in parts)
