"""Tests for the ksu_pdf text cleaning in the ingest path (no models)."""

from __future__ import annotations

from src.ingest.ksu_pdf import DOC_TITLE, final_pdf_text
from src.text import TATWEEL, clean_search_text

FATHA = "\u064E"
SUKUN = "\u0652"


class TestCleanSearchText:
    def test_removes_only_tatweel_and_diacritics(self) -> None:
        src = f"{TATWEEL}a{FATHA}b{SUKUN}c"
        assert clean_search_text(src) == "abc"

    def test_keeps_everything_else(self) -> None:
        src = "المادة 4,50 نصا 1448/1447 • End"
        assert clean_search_text(src) == src

    def test_empty_string(self) -> None:
        assert clean_search_text("") == ""


class TestFinalPdfText:
    def test_replaces_full_heading_with_unit_label(self) -> None:
        part = f"{DOC_TITLE} — المادة الرابعة عشرة\n{FATHA}نص{TATWEEL} الجسم"
        out = final_pdf_text("المادة الرابعة عشرة", part)
        assert out == "المادة الرابعة عشرة\nنص الجسم"
        assert DOC_TITLE not in out

    def test_unit_cleaned_too(self) -> None:
        part = f"{DOC_TITLE} — المادة{TATWEEL} (1)\nbody"
        out = final_pdf_text(f"المادة{TATWEEL} (1)", part)
        assert out == "المادة (1)\nbody"

    def test_single_line_chunk(self) -> None:
        out = final_pdf_text("الملحق (1)", f"{DOC_TITLE} — الملحق (1)\nrow")
        assert out == "الملحق (1)\nrow"

    def test_bodies_of_all_parts_cleaned(self) -> None:
        out = final_pdf_text(
            "المادة الأولى", f"{DOC_TITLE} — المادة الأولى\na{TATWEEL}b"
        )
        assert out == "المادة الأولى\nab"
