"""اختبارات تقسيم الأدلة (المرحلة 5): قوائم العناوين، والكلمات العامة، وجدول لكل فصل أو كلية.

التشغيل:
    python -m pytest tests/test_segmentation.py
"""

from chunk import build_chunks
from bs4 import BeautifulSoup

from parse import (
    blocks_from_soup,
    blocks_from_text,
    classify_heading,
    join_split_titles,
    program_tables,
)
from table_fixes import one_table_per_college

LONG_TEXT = "يجوز للطالب أن يتقدم بطلبه وفق الشروط المعتمدة. " * 12


def make_guide(blocks: list[dict]) -> dict:
    return {
        "uni": "KSU",
        "slug": "guide",
        "title": "دليل تجريبي",
        "doc_type": "دليل",
        "source_file": "guide.txt",
        "blocks": blocks,
    }


def test_bare_semester_heading_is_a_subheading() -> None:
    assert classify_heading("الفصل الأول") == ("sub", None, "الفصل الأول")
    assert classify_heading("الفصل الثاني") == ("sub", None, "الفصل الثاني")


def test_titled_chapter_and_academic_semester_keep_their_structure() -> None:
    assert classify_heading("الفصل الأول: التعريفات") == (
        "bab", None, "الفصل الأول: التعريفات"
    )
    assert classify_heading("الفصل الدراسي الأول") == ("bab", None, "الفصل الدراسي الأول")
    assert classify_heading("الفصل الدراسي الثاني") == ("bab", None, "الفصل الدراسي الثاني")


def test_only_listed_lines_are_section_headings() -> None:
    raw = "القسم الأول\nنص.\n\n\nسطر يبدو عنواناً\nنص آخر.\n\n\nالفصل من الجامعة\nحالاته."
    blocks = blocks_from_text(raw, headings=["القسم الأول", "الفصل من الجامعة"])
    top_headings = [
        block["text"] for block in blocks if block["t"] == "h" and block.get("level") == 1
    ]
    assert top_headings == ["القسم الأول", "الفصل من الجامعة"]
    assert blocks[-2]["kind"] == "sub"


def test_generic_word_stays_inside_its_section() -> None:
    blocks = [
        {"t": "h", "kind": "sub", "level": 1, "text": "التحويل بين الكليات:"},
        {"t": "p", "text": LONG_TEXT},
        {"t": "h", "kind": "sub", "level": 1, "text": "ملاحظة:"},
        {"t": "p", "text": "يتعين على طلاب المنح تعبئة النموذج."},
    ]
    chunks = build_chunks(make_guide(blocks))
    assert len(chunks) == 1
    assert chunks[0]["article_label"] == "التحويل بين الكليات:"
    assert "يتعين على طلاب المنح" in chunks[0]["body"]


def test_housing_deposit_stays_out_of_prohibitions() -> None:
    blocks = [
        {"t": "h", "kind": "sub", "level": 1, "text": "المحظورات والمخالفات"},
        {"t": "p", "text": "1. الإنذار بالفصل"},
        {
            "t": "h", "kind": "sub", "level": 1,
            "text": "إجراءات التقديم والتسكين ومبلغ التأمين:",
        },
        {"t": "p", "text": "يشترط سداد مبلغ التأمين وقدره (1000) ريال."},
    ]
    doc = make_guide(blocks)
    doc.update(uni="KAU", slug="sakan")
    chunks = build_chunks(doc)
    assert len(chunks) == 2
    assert chunks[0]["article_label"] == "المحظورات والمخالفات"
    assert "1000" not in chunks[0]["body"]
    assert chunks[1]["article_label"] == "إجراءات التقديم والتسكين ومبلغ التأمين:"
    assert "1000" in chunks[1]["body"]


def test_outbound_transfer_stays_separate_from_internal_transfer() -> None:
    blocks = [
        {
            "t": "h", "kind": "sub", "level": 1,
            "text": "ثانياً: التحويل من جامعة الملك عبدالعزيز الى جامعة اخرى",
        },
        {"t": "p", "text": "الجامعة تقدم المستندات الرسمية المطلوبة فقط."},
        {
            "t": "h", "kind": "sub", "level": 1,
            "text": "التحويل من برنامج إلى آخر داخل الجامعة:",
        },
        {"t": "p", "text": LONG_TEXT},
    ]
    doc = make_guide(blocks)
    doc.update(uni="KAU", slug="tahwil_ulya")
    chunks = build_chunks(doc)
    assert len(chunks) == 2
    assert chunks[0]["body"] == "الجامعة تقدم المستندات الرسمية المطلوبة فقط."
    assert chunks[1]["article_label"] == "التحويل من برنامج إلى آخر داخل الجامعة:"


def test_labelled_tables_become_separate_chunks() -> None:
    header = ["الحدث", "التاريخ"]
    blocks = [
        {
            "t": "table",
            "header": header,
            "rows": [["بداية الدراسة", "2026/08/23"]],
            "unit_label": "الفصل الدراسي الأول",
        },
        {
            "t": "table",
            "header": header,
            "rows": [["بداية الدراسة", "2027/01/10"]],
            "unit_label": "الفصل الدراسي الثاني",
        },
    ]
    chunks = build_chunks(make_guide(blocks))
    assert [chunk["article_label"] for chunk in chunks] == [
        "الفصل الدراسي الأول",
        "الفصل الدراسي الثاني",
    ]


def test_one_table_per_college_keeps_both_paths_together() -> None:
    header = ["الكلية", "المعدل"]
    first = {
        "t": "table",
        "header": header,
        "rows": [["الطب", "4.5"], ["التمريض (من كليات صحية)", "3.5"]],
    }
    second = {
        "t": "table",
        "header": header,
        "rows": [["التمريض (من كليات غير صحية)", "4.5"], ["الهندسة", "4"]],
    }
    tables = one_table_per_college([first, second])
    assert [table["unit_label"] for table in tables] == [
        "كلية الطب",
        "كلية التمريض",
        "كلية الهندسة",
    ]
    assert len(tables[1]["rows"]) == 2


def heading(text: str) -> dict:
    return {"t": "h", "kind": "sub", "text": text}


def test_split_title_is_joined_and_repeated_tail_dropped() -> None:
    blocks = join_split_titles(
        [
            heading("القبول المباشر"),
            heading("لبرامج البكالوريوس"),
            heading("برامج القبول المباشر في مرحلة البكالوريوس"),
            heading("مرحلة البكالوريوس"),
            heading("شروط القبول العامة"),
            heading("يشترط لقبول الطالب ما يلي:"),
            heading("نبذة عن الخدمات"),
            heading("المعسكرات التدريبية"),
        ]
    )
    assert [block["text"] for block in blocks] == [
        "القبول المباشر لبرامج البكالوريوس",
        "برامج القبول المباشر في مرحلة البكالوريوس",
        "شروط القبول العامة",
        "يشترط لقبول الطالب ما يلي:",
        "نبذة عن الخدمات",
        "المعسكرات التدريبية",
    ]
    assert blocks[3]["t"] == "p"


def test_each_college_list_becomes_its_own_chunk() -> None:
    html = (
        "<h2>الكليات الإنسانية</h2><p>معايير القبول:</p><ul><li>40% تحصيلي</li></ul>"
        "<h2>كلية الحقوق</h2><ul><li>أنظمة</li></ul>"
        "<h2>كلية السياحة</h2><ul><li>إدارة السفر والسياحة (طلاب)</li><li>إدارة الضيافة</li></ul>"
        "<p>مقابلة شخصية</p>"
        "<h2>كلية الهندسة التطبيقية</h2><p>معايير القبول:</p><ul><li>30% قدرات</li></ul>"
        "<ul><li>الهندسة الكهربائية التطبيقية</li></ul>"
        "<h2>الشروط العامة</h2><p>نص.</p>"
    )
    blocks = blocks_from_soup(BeautifulSoup(html, "html.parser"), mark_lists=True)
    blocks = program_tables(
        blocks, groups=["الكليات الإنسانية", "كلية الهندسة التطبيقية"], end="الشروط العامة"
    )
    for block in blocks:
        block.pop("list", None)
    chunks = build_chunks(make_guide(blocks))
    by_label = {chunk["article_label"]: chunk["body"] for chunk in chunks}
    assert by_label["كلية الحقوق"] == (
        "المجموعة: الكليات الإنسانية؛ الكلية: كلية الحقوق؛ البرنامج: أنظمة."
    )
    assert "إدارة السفر والسياحة (طلاب)" in by_label["كلية السياحة"]
    assert "مقابلة شخصية" in by_label["كلية السياحة"]
    assert "كلية السياحة" not in by_label["كلية الحقوق"]
    assert by_label["كلية الهندسة التطبيقية"].startswith(
        "الكلية: كلية الهندسة التطبيقية؛ البرنامج: الهندسة الكهربائية التطبيقية."
    )
    assert "30% قدرات" in by_label["كلية الهندسة التطبيقية"]
    assert "40% تحصيلي" in by_label["الكليات الإنسانية"]
    assert by_label["الكليات الإنسانية"].endswith("الكليات: كلية الحقوق، كلية السياحة.")
    assert "الكليات:" not in by_label["كلية الهندسة التطبيقية"]


def test_short_sections_stay_separate_and_empty_heading_is_kept() -> None:
    blocks = [
        heading("المطعم المركزي"),
        {"t": "p", "text": "توفير ثلاث وجبات يومياً."},
        heading("دراسة طالب زائر:"),
        heading("اولاً: طالب مسجل بجامعة أخرى"),
        {"t": "p", "text": "يقدم طلبه قبل بداية الفصل."},
    ]
    chunks = build_chunks(make_guide(blocks))
    assert [chunk["article_label"] for chunk in chunks] == [
        "المطعم المركزي",
        "اولاً: طالب مسجل بجامعة أخرى",
    ]
    assert chunks[1]["body"] == "دراسة طالب زائر:\nيقدم طلبه قبل بداية الفصل."


def test_long_paragraph_ending_with_colon_stays_its_own_chunk() -> None:
    intro = "توفر الجامعة خيارات متنوعة للقبول الأكاديمي. " * 5 + "فيما يلي تفاصيل خيارات القبول المتاحة:"
    blocks = [
        heading("برامج البكالوريوس والدبلوم المتاحة"),
        {"t": "p", "text": intro},
        heading("برامج البكالوريوس في الكليات الصحية"),
        {"t": "p", "text": "الكليات: كلية الطب، كلية الصيدلة."},
    ]
    chunks = build_chunks(make_guide(blocks))
    assert [chunk["article_label"] for chunk in chunks] == [
        "برامج البكالوريوس والدبلوم المتاحة",
        "برامج البكالوريوس في الكليات الصحية",
    ]

