"""اختبارات التقطيع (chunk.py): تحويل الجداول إلى جمل، وسطر «تابع:» في الأجزاء المقسومة.

التشغيل:
    python -m pytest tests/test_chunk.py
"""

from chunk import linearize_table, move_dangling_headings, split_items, units_from_blocks


def test_table_row_carries_its_column_names() -> None:
    block = {
        "t": "table",
        "header": ["الكلية", "المعدل"],
        "rows": [["الطب", "4.5"], ["الهندسة", "4.0"]],
    }
    assert linearize_table(block) == ["الكلية: الطب؛ المعدل: 4.5.", "الكلية: الهندسة؛ المعدل: 4.0."]


def test_rows_of_the_same_entry_are_numbered_in_one_sentence() -> None:
    block = {
        "t": "table",
        "header": ["الكلية", "المقرر", "الدرجة"],
        "rows": [["الطب", "أحياء 101", "90"], ["الطب", "كيمياء 101", "85"]],
    }
    assert linearize_table(block) == [
        "الكلية: الطب؛ (1) المقرر: أحياء 101، الدرجة: 90؛ (2) المقرر: كيمياء 101، الدرجة: 85."
    ]


def test_empty_cells_are_skipped() -> None:
    block = {"t": "table", "header": ["البند", "المبلغ"], "rows": [["رسوم", "-"]]}
    assert linearize_table(block) == ["البند: رسوم."]


def test_table_without_header_joins_cells() -> None:
    block = {"t": "table", "header": None, "rows": [["أ", "", "ب"]]}
    assert linearize_table(block) == ["أ | ب"]


def test_single_cell_row_is_a_note_line() -> None:
    block = {"t": "table", "header": ["أ", "ب", "ج"], "rows": [["حاشية الجدول"]]}
    assert linearize_table(block) == ["حاشية الجدول"]


def long_item(number: int) -> dict:
    return {"kind": "text", "text": f"{number}. شرط رقم {number} " + "نص " * 30}


def test_second_part_starts_with_continuation_line() -> None:
    items = [{"kind": "sub", "text": "شروط القبول"}, {"kind": "text", "text": "يشترط ما يلي:"}]
    items += [long_item(number) for number in range(1, 6)]
    parts = split_items(items, max_chars=300)
    assert len(parts) > 1
    assert parts[1][0]["kind"] == "ctx"
    assert parts[1][0]["text"] == "تابع: شروط القبول › يشترط ما يلي"


def test_heading_at_part_tail_moves_to_next_part() -> None:
    first = [{"kind": "text", "text": "نص"}, {"kind": "sub", "text": "عنوان"}]
    second = [{"kind": "text", "text": "متن العنوان"}]
    parts = [first, second]
    move_dangling_headings(parts)
    assert [item["text"] for item in parts[0]] == ["نص"]
    assert [item["text"] for item in parts[1]] == ["عنوان", "متن العنوان"]


def test_short_unit_is_not_split() -> None:
    items = [{"kind": "text", "text": "نص قصير"}]
    assert split_items(items, max_chars=300) == [items]


def test_repeated_heading_continues_the_current_unit() -> None:
    blocks = [
        {"t": "h", "kind": "sub", "text": "حقوق الطالب"},
        {"t": "p", "text": "1. الحق الأول"},
        {"t": "h", "kind": "sub", "text": "حقوق الطالب"},
        {"t": "p", "text": "2. الحق الثاني"},
        {"t": "h", "kind": "sub", "text": "واجبات الطالب"},
        {"t": "p", "text": "1. الواجب الأول"},
    ]
    units = units_from_blocks(blocks, has_structure=False)
    assert [unit["label"] for unit in units] == ["حقوق الطالب", "واجبات الطالب"]
    assert [item["text"] for item in units[0]["items"]] == ["1. الحق الأول", "2. الحق الثاني"]
    assert [item["text"] for item in units[1]["items"]] == ["1. الواجب الأول"]


def test_repeated_heading_uses_arabic_normalization() -> None:
    blocks = [
        {"t": "h", "kind": "sub", "text": "إجراءات الطالب"},
        {"t": "p", "text": "الإجراء الأول"},
        {"t": "h", "kind": "sub", "text": "اِجراءات الطالب"},
        {"t": "p", "text": "الإجراء الثاني"},
    ]
    units = units_from_blocks(blocks, has_structure=False)
    assert len(units) == 1
    assert units[0]["label"] == "إجراءات الطالب"
    assert units[0]["items"] == [
        {"kind": "text", "text": "الإجراء الأول"},
        {"kind": "text", "text": "الإجراء الثاني"},
    ]


def test_repeated_breaking_heading_starts_a_new_unit() -> None:
    for kind in ("bab", "qism", "appendix", "article", "exec"):
        blocks = [
            {"t": "h", "kind": kind, "text": "عنوان", "no": 1},
            {"t": "p", "text": "المتن الأول"},
            {"t": "h", "kind": kind, "text": "عنوان", "no": 1},
            {"t": "p", "text": "المتن الثاني"},
        ]
        units = units_from_blocks(blocks, has_structure=True)
        assert len(units) == 2
        assert [unit["label"] for unit in units] == ["عنوان", "عنوان"]
        assert units[0]["items"] == [{"kind": "text", "text": "المتن الأول"}]
        assert units[1]["items"] == [{"kind": "text", "text": "المتن الثاني"}]


def test_cover_stops_at_first_full_sentence() -> None:
    from chunk import is_cover

    assert is_cover("جامعة الملك عبدالعزيز عمادة القبول والتسجيل")
    assert is_cover("للعام الجامعي 1449/1448هـ الموافق 2027/2026")
    assert not is_cover("هذا الدليل سوف يطبق بنهاية الفصل الدراسي الثاني لجميع الطلاب.")


def calendar_doc(doc_type: str) -> dict:
    return {
        "uni": "KSU",
        "slug": "taqwim",
        "title": "التقويم الدراسي",
        "doc_type": doc_type,
        "source_file": "taqwim.txt",
        "blocks": [
            {
                "t": "table",
                "header": ["الحدث", "اليوم", "التاريخ"],
                "rows": [
                    ["بداية الدراسة", "الأحد", "2026/08/23"],
                    ["بداية الاختبارات النهائية", "الأحد", "2026/12/13"],
                ],
                "unit_label": "الفصل الدراسي الأول",
            },
            {"t": "p", "text": "التواريخ قابلة للتعديل بقرار من المجلس."},
        ],
    }


def test_each_calendar_event_is_its_own_chunk() -> None:
    from chunk import build_chunks

    chunks = build_chunks(calendar_doc("تقويم"))
    assert [chunk["chunk_id"][-4:] for chunk in chunks[1:]] == ["_r01", "_r02"]
    assert chunks[0]["body"] == "التواريخ قابلة للتعديل بقرار من المجلس."
    assert chunks[1]["body"] == "الحدث: بداية الدراسة؛ اليوم: الأحد؛ التاريخ: 2026/08/23."
    assert all(chunk["article_label"] == "الفصل الدراسي الأول" for chunk in chunks)
    assert chunks[2]["text"].splitlines()[1:] == [
        "الفصل الدراسي الأول",
        "الحدث: بداية الاختبارات النهائية؛ اليوم: الأحد؛ التاريخ: 2026/12/13.",
    ]


def test_other_documents_keep_table_rows_together() -> None:
    from chunk import build_chunks, is_event_line

    assert len(build_chunks(calendar_doc("دليل"))) == 1
    assert not is_event_line("المادة: نص بلا فاصلة منقوطة")
