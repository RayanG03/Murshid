"""اختبارات ثبات معرّفات المقاطع (chunk.py): إضافة كتلة لا تغيّر معرّفات غيرها.

التشغيل:
    python -m pytest tests/test_chunk_ids.py
"""

from chunk import build_chunks

LONG_TEXT = "يجوز للطالب أن يتقدم بطلبه وفق الشروط المعتمدة. " * 12


def make_doc(blocks: list[dict]) -> dict:
    return {
        "uni": "KSU",
        "slug": "test",
        "title": "لائحة تجريبية",
        "doc_type": "لائحة",
        "source_file": "test.txt",
        "blocks": blocks,
    }


def article(number: int, name: str) -> list[dict]:
    return [
        {"t": "h", "kind": "article", "text": f"المادة {name}", "no": number},
        {"t": "p", "text": f"نص المادة {name}. {LONG_TEXT}"},
    ]


def section(title: str) -> list[dict]:
    return [
        {"t": "h", "kind": "sub", "level": 1, "text": title},
        {"t": "p", "text": f"{title}: {LONG_TEXT}"},
    ]


def chunk_ids(blocks: list[dict]) -> list[str]:
    return [chunk["chunk_id"] for chunk in build_chunks(make_doc(blocks))]


def test_articles_are_named_by_their_number() -> None:
    assert chunk_ids(article(1, "الأولى") + article(25, "الخامسة والعشرون")) == [
        "ksu_test_art001",
        "ksu_test_art025",
    ]


def test_adding_an_executive_rule_keeps_article_ids() -> None:
    before = chunk_ids(article(1, "الأولى") + article(2, "الثانية"))
    rule = [
        {"t": "h", "kind": "exec", "text": "القاعدة التنفيذية للمادة الأولى", "no": 1},
        {"t": "p", "text": "نص القاعدة. " + LONG_TEXT},
    ]
    after = chunk_ids(article(1, "الأولى") + rule + article(2, "الثانية"))
    assert after == ["ksu_test_art001", "ksu_test_exec001", "ksu_test_art002"]
    for chunk_id in before:
        assert chunk_id in after


def test_adding_a_section_keeps_other_section_ids() -> None:
    before = chunk_ids(section("البطاقة الجامعية") + section("المكافآت"))
    after = chunk_ids(section("البطاقة الجامعية") + section("السكن الجامعي") + section("المكافآت"))
    assert len(after) == 3
    assert before == [after[0], after[2]]


def test_part_suffix_only_for_split_units() -> None:
    long_article = [{"t": "h", "kind": "article", "text": "المادة الأولى", "no": 1}]
    for number in range(1, 8):
        long_article.append({"t": "p", "text": f"{number}. {LONG_TEXT}"})
    ids = chunk_ids(long_article + article(2, "الثانية"))
    assert ids[0] == "ksu_test_art001_p1"
    assert ids[1] == "ksu_test_art001_p2"
    assert ids[-1] == "ksu_test_art002"


def test_same_section_title_gets_a_counter() -> None:
    ids = chunk_ids(section("حقوق الطالب") + section("واجبات الطالب") + section("حقوق الطالب"))
    assert ids[2] == ids[0] + "_2"
