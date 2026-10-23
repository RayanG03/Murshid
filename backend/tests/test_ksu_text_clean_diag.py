"""Unit tests for the pure helpers of the ksu_pdf text-clean experiment."""

from __future__ import annotations

import pytest

from src.ksu_probes import (
    DOMAIN_ADJACENT_WRONG_KSU,
    OFF_TOPIC_KSU,
    RELEVANT_KSU,
    split_tune_test,
)
from src.ksu_text_clean_diag import (
    EXCLUDED_FROM_ALL,
    RELEVANT_EXTRA,
    chunk_damage,
    clean_search_text,
    damage_stats,
    separation_summary,
    tune_query_sets,
    variant_text,
)

TATWEEL = "\u0640"
FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SUKUN = "\u0652"
ALEF = "\u0623"
YA = "\u064A"
TA_MARBUTA = "\u0629"


class TestCleanSearchText:
    def test_removes_tatweel_and_diacritics_only(self) -> None:
        src = f"{ALEF}{TATWEEL}{FATHA}l{DAMMA}w{KASRA}r{SUKUN}d{TA_MARBUTA}"
        assert clean_search_text(src) == ALEF + "lwrd" + TA_MARBUTA

    def test_keeps_letters_and_digits(self) -> None:
        src = "المادة 4,50 نصا 1448/1447End"
        assert clean_search_text(src) == src

    def test_empty_string(self) -> None:
        assert clean_search_text("") == ""

    def test_only_removable_chars_becomes_empty(self) -> None:
        src = TATWEEL + FATHA + DAMMA + KASRA + SUKUN + TATWEEL
        assert clean_search_text(src) == ""


class TestChunkDamage:
    def test_counts_both_kinds(self) -> None:
        doc = f"a{TATWEEL}{TATWEEL}b{FATHA}c{SUKUN}"
        assert chunk_damage(doc) == {"tatweel": 2, "diacritics": 2}

    def test_clean_doc_zeroed(self) -> None:
        assert chunk_damage("plain عربي 123") == {"tatweel": 0, "diacritics": 0}


class TestDamageStats:
    def test_aggregates(self) -> None:
        docs = [
            f"x{TATWEEL}{TATWEEL}{FATHA}",
            "clean",
            f"y{TATWEEL}z{DAMMA}{KASRA}",
        ]
        stats = damage_stats(docs)
        assert stats == {
            "n_chunks": 3,
            "chunks_with_tatweel": 2,
            "chunks_with_diacritics": 2,
            "total_tatweel": 3,
            "total_diacritics": 3,
        }

    def test_empty_collection(self) -> None:
        assert damage_stats([])["n_chunks"] == 0


class TestVariantText:
    DOC = (
        "اللائحة التنظيمية — المادة الرابعة عشرة\n"
        f"body{TATWEEL} text{FATHA} here"
    )

    def test_a_is_identity(self) -> None:
        md = {"unit": "المادة الرابعة عشرة"}
        assert variant_text(md, self.DOC, "A") == self.DOC

    def test_b_cleans_whole_document(self) -> None:
        md = {"unit": "المادة الرابعة عشرة"}
        out = variant_text(md, self.DOC, "B")
        assert TATWEEL not in out and FATHA not in out
        assert out.startswith("اللائحة التنظيمية")

    def test_c_shortens_heading_to_unit(self) -> None:
        md = {"unit": "المادة الرابعة عشرة"}
        out = variant_text(md, self.DOC, "C")
        assert out == "المادة الرابعة عشرة\nbody text here"

    def test_c_without_newline_uses_whole_doc_as_body(self) -> None:
        md = {"unit": "U"}
        out = variant_text(md, f"a{TATWEEL}b", "C")
        assert out == "U\nab"

    def test_c_with_missing_unit_falls_back_to_b(self) -> None:
        doc = f"a{TATWEEL}b\nc{FATHA}d"
        assert variant_text({}, doc, "C") == clean_search_text(doc)


class TestTuneQuerySets:
    def test_overrides_applied(self) -> None:
        relevant, refusal = tune_query_sets()
        assert RELEVANT_EXTRA in relevant
        assert relevant.count(RELEVANT_EXTRA) == 1
        assert EXCLUDED_FROM_ALL not in relevant
        assert EXCLUDED_FROM_ALL not in refusal
        assert RELEVANT_EXTRA not in refusal

    def test_tune_only(self) -> None:
        relevant, refusal = tune_query_sets()
        forbidden: set[str] = set(
            split_tune_test(RELEVANT_KSU)[1]
            + split_tune_test(OFF_TOPIC_KSU)[1]
            + split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)[1]
        )
        assert not (set(relevant) | set(refusal)) & forbidden
        rel_tuned, off_tuned, daw_tuned = (
            split_tune_test(RELEVANT_KSU)[0],
            split_tune_test(OFF_TOPIC_KSU)[0],
            split_tune_test(DOMAIN_ADJACENT_WRONG_KSU)[0],
        )
        assert set(relevant) | set(refusal) <= (
            set(rel_tuned) | set(off_tuned) | set(daw_tuned)
        )

    def test_counts(self) -> None:
        relevant, refusal = tune_query_sets()
        assert len(relevant) == 9
        assert len(refusal) == 7


class TestSeparationSummary:
    def test_basic_numbers(self) -> None:
        s = separation_summary([0.9, 0.8], [0.7, 0.6])
        assert s["auc"] == 1.0
        assert s["min_relevant_top1"] == pytest.approx(0.8)
        assert s["max_refusal_top1"] == pytest.approx(0.7)
        assert s["margin"] == pytest.approx(0.1)
        assert s["overlap"] == 0

    def test_tie_half_credit(self) -> None:
        s = separation_summary([0.5], [0.5])
        assert s["auc"] == 0.5
        assert s["overlap"] == 1

    def test_empty_sides(self) -> None:
        s = separation_summary([], [])
        assert s["auc"] == 0.5
        assert s["min_relevant_top1"] is None
        assert s["max_refusal_top1"] is None
        assert s["margin"] == 0.0
        assert s["overlap"] == 0
