"""Unit tests for Stage 3 pure functions (no model, no ChromaDB)."""

from pathlib import Path

from src.ingest.kau_pdf import chunk_page_blocks
from src.ingest.kau_web import chunk_section, split_markdown
from src.text import normalize_digits_for_search


def test_normalize_digits() -> None:
    assert normalize_digits_for_search("٤.٥ من ٥") == "4.5 من 5"
    assert normalize_digits_for_search("١٤٤٨/١٤٤٧") == "1448/1447"


def test_split_markdown_sections() -> None:
    md = (
        "---\n"
        "source_url: https://example.com/page\n"
        "page_title: الصفحة الرئيسية\n"
        "---\n"
        "# الصفحة الرئيسية\n"
        "## القسم الأول\n"
        "نص القسم الأول.\n"
        "## القسم الثاني\n"
        "نص القسم الثاني.\n"
    )
    fm, sections = split_markdown(md)
    assert fm["page_title"] == "الصفحة الرئيسية"
    assert [h for h, _ in sections] == ["القسم الأول", "القسم الثاني"]
    assert "نص القسم الأول." in sections[0][1]


def test_chunk_section_keeps_heading_on_each_part() -> None:
    body = "\n\n".join(f"فقرة رقم {i} " + "كلمة " * 40 for i in range(10))
    parts = chunk_section("العنوان", body, "الصفحة", max_chars=400)
    assert len(parts) > 1
    assert all(p.startswith("الصفحة > العنوان") for p in parts)


def test_chunk_page_blocks_never_explodes() -> None:
    text = "\n\n".join("سطر داخل كتلة " * 20 for _ in range(20))
    chunks = chunk_page_blocks(text, page=3, target=500, max_chars=900)
    assert chunks
    assert all(c.startswith("دليل التحويل الداخلي بين الكليات — صفحة 3") for c in chunks)
    assert all(len(c) < 2000 for c in chunks)
