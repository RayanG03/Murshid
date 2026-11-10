"""اختبارات نص البحث في المحادثة (answer.py)، والرفض والاستشهاد (prompt_parts.py)،
وقراءة رد المرتِّب (openrouter.py).

التشغيل:
    python -m pytest tests/test_answer.py
"""

from answer import search_query
from openrouter import is_valid_rerank_results
from prompt_parts import GREETING, REFUSAL, cited_passages, is_greeting, is_refusal, passage_header


def test_a_new_question_is_searched_as_written() -> None:
    assert search_query("كم  مرة أأجل؟", []) == "كم مرة أأجل؟"


def test_a_follow_up_is_searched_with_the_earlier_questions() -> None:
    history = [
        {"question": "أنا طالب ماجستير، كم أقصى مدة؟", "answer": "ستة فصول [1]."},
        {"question": "وإذا تجاوزتها؟", "answer": "يجوز منح مهلة [1]."},
    ]
    assert search_query("كم مرة يمددون؟", history) == (
        "أنا طالب ماجستير، كم أقصى مدة؟ وإذا تجاوزتها؟ كم مرة يمددون؟"
    )


def test_full_refusal_is_detected() -> None:
    assert is_refusal(REFUSAL)


def test_partial_answer_phrase_is_not_a_refusal() -> None:
    assert not is_refusal("يحق لك التأجيل فصلين [1]. ولم تذكر المصادر المتاحة أثره على المكافأة.")


def test_refusal_phrase_inside_an_answer_is_not_a_refusal() -> None:
    assert not is_refusal("لا يوجد في المصادر المتاحة تاريخ محدد، والطلب قبل النهائي [1].")


def test_english_refusal_starts_with_the_arabic_sentence() -> None:
    assert is_refusal(REFUSAL + "\nNot found in the available sources.")


def test_fixed_greeting_is_recognised() -> None:
    assert is_greeting(GREETING)
    assert not is_greeting("يحق لك الاعتذار عن مقررين [1].")


def test_passage_header_shows_type_path_and_part() -> None:
    hit = {
        "university_ar": "جامعة الملك سعود",
        "doc_title": "اللائحة التنظيمية",
        "doc_type": "لائحة",
        "bab": "الباب الثالث",
        "qism": "القسم الرابع",
        "appendix": None,
        "article_label": "المادة الخامسة والعشرون",
        "part": "2/3",
    }
    assert passage_header(1, hit) == (
        "[1] جامعة الملك سعود — اللائحة التنظيمية (لائحة) — الباب الثالث › القسم الرابع"
        " › المادة الخامسة والعشرون (الجزء 2 من 3)"
    )



def test_passage_header_shows_the_version_when_the_source_gives_one() -> None:
    hit = {
        "university_ar": "جامعة الملك عبدالعزيز",
        "doc_title": "دليل القبول",
        "doc_type": "دليل",
        "version": "1449/1448هـ",
        "bab": None,
        "qism": None,
        "appendix": None,
        "article_label": "كلية السياحة",
        "part": "1/1",
    }
    assert passage_header(1, hit) == (
        "[1] جامعة الملك عبدالعزيز — دليل القبول (دليل، النسخة: 1449/1448هـ) — كلية السياحة"
    )

def test_citations_outside_the_passages_are_ignored() -> None:
    assert cited_passages("الجواب [1] وكذلك [3][9] و[1]", passage_count=4) == [1, 3]


def test_rerank_results_are_validated() -> None:
    assert is_valid_rerank_results([{"index": 0, "relevance_score": 0.9}])
    assert not is_valid_rerank_results([{"index": "0", "relevance_score": 0.9}])
    assert not is_valid_rerank_results(None)
