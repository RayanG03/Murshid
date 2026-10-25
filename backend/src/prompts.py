"""Prompts for grounded Arabic QA generation.

All entry points are parametrized by university; the KAU strings are
preserved byte-for-byte as the default (``university="kau"``). KSU wording
deliberately avoids the "موثّق" claim (the KSU dataset is not formally
verified like the KAU 371). Keeping every university-dependent string in
this module means src.context_assembly and src.pipeline can never render
disagreeing prompt text.
"""

from src.config import UNIVERSITY_DISPLAY_NAMES

# Rule-2 citation example per university (rest of the rules stay identical).
_CITATION_EXAMPLES: dict[str, str] = {
    "kau": "دليل التحويل الداخلي — صفحة 7، أو رابط المصدر إن وُجد",
    "ksu": "اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية "
    "— المادة الرابعة عشر، أو سؤال رقم 12",
}

# Rule-1 Q&A clause and rule-2 Q&A tag phrasing per university.
_QA_CONTEXT_CLAUSES: dict[str, str] = {
    "kau": "أسئلة وإجابات موثّقة",
    "ksu": "أسئلة وإجابات",
}
_QA_TAG_CLAUSES: dict[str, str] = {
    "kau": "رقم السؤال الموثّق (مثل: سؤال موثّق رقم 12)",
    "ksu": "رقم السؤال (مثل: سؤال رقم 12)",
}

_QA_BLOCK_TITLES: dict[str, str] = {
    "kau": "أسئلة وإجابات موثّقة من جامعة الملك عبدالعزيز:",
    "ksu": "أسئلة وإجابات من قاعدة أسئلة جامعة الملك سعود:",
}

DOC_BLOCK_TITLE = "مقتطفات من اللوائح والأدلة الرسمية:"

_SYSTEM_TEMPLATE = (
    "أنت مساعد جامعي تجيب عن أسئلة الطلاب وفق لوائح {university}. "
    "التزم حرفيًا بالقواعد التالية:\n"
    "1. أجب فقط من السياق المقدم لك ({qa_clause} و/أو مقتطفات من "
    "اللوائح والأدلة الرسمية). لا تستخدم أي معرفة خارجية ولا تخمين.\n"
    "2. اذكر المصدر بعد الإجابة: {qa_tag_clause} "
    "أو المصدر/الصفحة للمقتطف (مثل: {citation_example}).\n"
    "3. اكتب الإجابة بالعربية وبأسلوب واضح ومختصر.\n"
    "4. إذا كان السياق فارغًا أو غير كافٍ للإجابة عن السؤال، اكتب حصرًا "
    "العبارة التالية دون أي إضافة: لا يمكنني تأكيد الإجابة من المصادر المتاحة.\n"
    "5. لا تختلق أرقامًا أو تواريخ أو نسبًا أو إجراءات غير موجودة في السياق.\n"
    "6. اكتب نصًا عاديًا دون صيغ رياضية أو رموز LaTeX، واكتب أي معادلة "
    "بالكلمات (مثل: مجموع النقاط ÷ مجموع الساعات)."
)

# Back-compat alias: the exact KAU wording as it always was.
SYSTEM_PROMPT = (
    "أنت مساعد جامعي تجيب عن أسئلة الطلاب وفق لوائح جامعة الملك عبدالعزيز. "
    "التزم حرفيًا بالقواعد التالية:\n"
    "1. أجب فقط من السياق المقدم لك (أسئلة وإجابات موثّقة و/أو مقتطفات من "
    "اللوائح والأدلة الرسمية). لا تستخدم أي معرفة خارجية ولا تخمين.\n"
    "2. اذكر المصدر بعد الإجابة: رقم السؤال الموثّق (مثل: سؤال موثّق رقم 12) "
    "أو المصدر/الصفحة للمقتطف (مثل: دليل التحويل الداخلي — صفحة 7، أو رابط "
    "المصدر إن وُجد).\n"
    "3. اكتب الإجابة بالعربية وبأسلوب واضح ومختصر.\n"
    "4. إذا كان السياق فارغًا أو غير كافٍ للإجابة عن السؤال، اكتب حصرًا "
    "العبارة التالية دون أي إضافة: لا يمكنني تأكيد الإجابة من المصادر المتاحة.\n"
    "5. لا تختلق أرقامًا أو تواريخ أو نسبًا أو إجراءات غير موجودة في السياق.\n"
    "6. اكتب نصًا عاديًا دون صيغ رياضية أو رموز LaTeX، واكتب أي معادلة "
    "بالكلمات (مثل: مجموع النقاط ÷ مجموع الساعات)."
)

# Cross-check the template against the historical constant at import time.
_KAU_SYSTEM_FROM_TEMPLATE = _SYSTEM_TEMPLATE.format(
    university=UNIVERSITY_DISPLAY_NAMES["kau"],
    qa_clause=_QA_CONTEXT_CLAUSES["kau"],
    qa_tag_clause=_QA_TAG_CLAUSES["kau"],
    citation_example=_CITATION_EXAMPLES["kau"],
)
assert _KAU_SYSTEM_FROM_TEMPLATE == SYSTEM_PROMPT

QA_BLOCK_TITLE = _QA_BLOCK_TITLES["kau"]


def build_system_prompt(university: str = "kau") -> str:
    """System prompt for one university (KAU keeps the historical wording)."""
    if university not in _CITATION_EXAMPLES:
        raise ValueError(f"Unknown university {university!r}")
    return _SYSTEM_TEMPLATE.format(
        university=UNIVERSITY_DISPLAY_NAMES[university],
        qa_clause=_QA_CONTEXT_CLAUSES[university],
        qa_tag_clause=_QA_TAG_CLAUSES[university],
        citation_example=_CITATION_EXAMPLES[university],
    )


def qa_block_title(university: str = "kau") -> str:
    """Q&A context-block title for a university."""
    if university not in _QA_BLOCK_TITLES:
        raise ValueError(f"Unknown university {university!r}")
    return _QA_BLOCK_TITLES[university]


def build_user_prompt(
    question: str, qa_block: str, doc_block: str, university: str = "kau",
    standalone_question: str | None = None,
) -> str:
    """Assemble the labeled context blocks and the question.

    ``standalone_question`` (follow-ups only) is the history-aware rewrite
    used for retrieval; it is shown right under the user's wording so the
    answer model knows what a short follow-up refers to. Omitted when it
    is missing or identical to the question (prompt unchanged).
    """
    parts = [f"السؤال: {question}"]
    if standalone_question and standalone_question.strip() != question.strip():
        parts.append(
            f"المقصود بالسؤال (صيغة مستقلة): {standalone_question}"
        )
    if qa_block:
        parts.append(qa_block_title(university) + "\n" + qa_block)
    if doc_block:
        parts.append(DOC_BLOCK_TITLE + "\n" + doc_block)
    parts.append("أجب وفق القواعد المذكورة، مع ذكر المصدر.")
    return "\n\n".join(parts)
