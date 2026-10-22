"""Unit tests for the KSU data prep and ingestion chunkers (no model)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_prep.build_ksu_qa import EXPECTED_ROWS, build_ksu_dataset, print_report
from src.ingest.ksu_txt import (
    Section,
    _split_with_heading,
    chunk_section,
    parse_sections,
    split_numbered_items,
)

TXT_FIXTURE = (
    "القسم الرئيسي :القبول\n"
    "الخدمة الطلوبة:(دليل القبول)\n"
    "مقدمة الخدمة.\n"
    "\n"
    "(الاساله الشائعة حول القبول)\n"
    "1-  ما هي شروط القبول؟\n"
    "\n"
    "تشمل شروط القبول ما يلي.\n"
    "\n"
    "2-  ماذا عن التحويل؟\n"
    "\n"
    "يحتاج إلى ستين وحدة.\n"
    "1.    سبب داخلي أول.\n"
    "2.    سبب داخلي ثانٍ.\n"
    "\n"
    "3-  هل يوجد اختبار؟\n"
    "\n"
    "نعم هناك اختبار.\n"
    "\n"
    "القسم الرئيسي:-الطلاب\n"
    "القسم الفرعي:الطلاب المستجدون\n"
    "\n"
    "الخدمة المطلوبة:(البطاقات الجامعية)\n"
    "تُصدر البطاقة من العمادة.\n"
)


def test_parse_sections_service_headers() -> None:
    sections = parse_sections(TXT_FIXTURE)
    services = [(s.main_section, s.service) for s in sections]
    assert ("القبول", "دليل القبول") in services
    assert ("الطلاب", "البطاقات الجامعية") in services
    assert ("الطلاب", "") in services  # القسم الفرعي preamble kept
    assert len(sections) == 3


def test_split_numbered_items_keeps_nested_lists() -> None:
    body = (
        "1-  ماذا عن التحويل؟\n"
        "يحتاج إلى ستين وحدة.\n"
        "1. سبب داخلي أول.\n"
        "2. سبب داخلي ثانٍ.\n"
        "2-  هل يوجد اختبار؟\n"
        "نعم هناك اختبار.\n"
    )
    items = split_numbered_items(body)
    assert len(items) == 2
    assert items[0][0].startswith("1-")
    assert "سبب داخلي ثانٍ" in items[0][1]  # nested list stays attached
    assert items[1][0].startswith("2-")


def test_chunk_section_faq_items() -> None:
    section = parse_sections(TXT_FIXTURE)[0]
    chunks = chunk_section(section, max_chars=1200)
    faq = [c for c in chunks if "شائع" in c.display.splitlines()[0]]
    assert len(faq) == 3  # one chunk per numbered question
    assert all(c.metadata["service"] == "دليل القبول" for c in faq)
    assert all(c.metadata["type"] == "txt" for c in faq)
    assert all(c.metadata["source_url"] == "" for c in faq)
    assert faq[0].display.startswith("القبول > دليل القبول > الاساله الشائعة حول القبول")
    plain = [c for c in chunks if c not in faq]
    assert len(plain) == 1  # section preamble before the FAQ heading


def test_split_with_heading_keeps_heading_on_every_part() -> None:
    body = "\n\n".join(f"فقرة رقم {i} " + "كلمة " * 60 for i in range(6))
    parts = _split_with_heading("الرئيسي > الخدمة", body, max_chars=300)
    assert len(parts) > 1
    assert all(p.startswith("الرئيسي > الخدمة") for p in parts)


def test_chunk_section_real_source() -> None:
    from src.config import get_settings

    settings = get_settings()
    path = settings.sources_dir / "KSU" / "university documents.txt"
    if not path.exists():
        pytest.skip("KSU source file not present")
    chunks = chunk_section(
        Section("القبول", "المكافات", path.read_text(encoding="utf-8-sig").split(
            "الخدمة المطلوبة:(المكافات)", 1)[1].strip()), max_chars=1200
    )
    faq = [c for c in chunks if "شائع" in c.display.splitlines()[0]]
    assert len(faq) == 7  # the المكافات FAQ block has 7 questions
    assert any("لماذا يحذف طلبي" in c.display for c in faq)


def _fake_audit_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Q_ID": range(1, n_rows + 1),
            "Question_Original": [f"سؤال {i}؟" for i in range(1, n_rows + 1)],
            "Question_Normalized": [f"سؤال {i}" for i in range(1, n_rows + 1)],
            "Answer": [f"جواب {i} جامعة الملك سعود" for i in range(1, n_rows + 1)],
            "Category": ["القبول"] * n_rows,
            "حالة المطابقة مع جامعة الملك عبدالعزيز": ["مطابق بالمعنى"] * n_rows,
            "إجابة جامعة الملك عبدالعزيز": [
                f"جواب {i} جامعة الملك عبدالعزيز" for i in range(1, n_rows + 1)
            ],
            "المصدر الرسمي": ["https://kau.edu.sa/x"] * n_rows,
        }
    )


def test_build_ksu_dataset_574_rows_and_no_kau_leakage() -> None:
    df = _fake_audit_frame(EXPECTED_ROWS)
    ksu = build_ksu_dataset(df)
    assert len(ksu) == EXPECTED_ROWS
    assert list(ksu.columns) == [
        "Q_ID",
        "question_original",
        "question_normalized",
        "answer",
        "category",
    ]
    dumped = ksu.to_csv(index=False)
    assert "عبدالعزيز" not in dumped  # no KAU answers
    assert "kau.edu.sa" not in dumped  # no KAU source URLs
    assert "مطابق" not in dumped  # no KAU audit columns
    print_report(ksu)  # must not raise for exactly 574 rows


def test_print_report_stops_on_wrong_count() -> None:
    ksu = build_ksu_dataset(_fake_audit_frame(10))
    with pytest.raises(SystemExit):
        print_report(ksu)
