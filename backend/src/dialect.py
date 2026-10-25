"""Saudi-dialect -> MSA normalization for SEARCH queries only.

Students write "وش شروط التحويل" or "ابي احذف ماده" while the corpus is
formal Arabic. Replacing a few high-frequency dialect words with their MSA
equivalents moves the query embedding closer to the regulations. Applied
only to the retrieval query (settings.dialect_normalization); the answer
prompt always keeps the student's own wording.
"""

import re

# Whole-word replacements (dialect -> MSA). Keep this list short and
# unambiguous: every entry changes retrieval for all queries containing it.
DIALECT_TO_MSA: dict[str, str] = {
    "وش": "ما",
    "وشو": "ما",
    "ايش": "ما",
    "إيش": "ما",
    "شنو": "ما",
    "ليش": "لماذا",
    "ليه": "لماذا",
    "وين": "أين",
    "ابي": "أريد",
    "أبي": "أريد",
    "ابغى": "أريد",
    "أبغى": "أريد",
    "ابغا": "أريد",
    "ودي": "أريد",
    "يبي": "يريد",
    "يبغى": "يريد",
    "اقدر": "أستطيع",
    "أقدر": "أستطيع",
    "يقدر": "يستطيع",
    "اسوي": "أعمل",
    "أسوي": "أعمل",
    "يسوي": "يعمل",
    "الحين": "الآن",
    "عشان": "لكي",
    "علشان": "لكي",
    "زي": "مثل",
    "مو": "ليس",
    "مب": "ليس",
    "كذا": "هكذا",
    "اطلع": "أستخرج",
    "اطلّع": "أستخرج",
    # Student vocabulary (live tests 2026-10-25): "ابي اأجل الترم كيف" was
    # refused (rerank 0.12) — "الترم" is the colloquial word for semester.
    "الترم": "الفصل الدراسي",
    "ترم": "فصل دراسي",
    "الترمات": "الفصول الدراسية",
    "الكورس": "المقرر",
    "كورس": "مقرر",
    "الكورسات": "المقررات",
    # Frequent misspellings of key regulation words.
    "اأجل": "أؤجل",
    "أأجل": "أؤجل",
    "اءجل": "أؤجل",
    "انزار": "إنذار",
    "الانزار": "الإنذار",
    "مكافئة": "مكافأة",
    "مكافئه": "مكافأة",
    "المكافئة": "المكافأة",
    "المكافئه": "المكافأة",
    "الاختبرات": "الاختبارات",
    "اختبرات": "اختبارات",
}

_WORD_RE = re.compile(r"[؀-ۿ]+")


def normalize_dialect(query: str) -> str:
    """Replace whole dialect words with MSA; everything else untouched."""
    return _WORD_RE.sub(lambda m: DIALECT_TO_MSA.get(m.group(0), m.group(0)), query)
