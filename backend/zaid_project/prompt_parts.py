"""رسالة نموذج الإجابة وفحص رده: ترويسة كل مقطع، والمحادثة السابقة، والرفض والتحية
والاستشهادات، وبطاقات المصادر التي تعرضها الواجهة.
"""

import re

REFUSAL = "لا يوجد في المصادر المتاحة"
GREETING = "أهلاً بك، أنا مساعد يجيب عن أسئلتك في لوائح جامعتك وأدلتها وتقويمها الأكاديمي."
CITATION = re.compile(r"\[(\d+)\]")


def passage_header(number: int, hit: dict) -> str:
    """[n] الجامعة — الوثيقة (النوع، النسخة) — الباب › القسم › العنوان (الجزء 2 من 3).

    كان النموذج يرى اسم المادة فقط، فيضيع الباب والقسم ونوع الوثيقة (لائحة أم دليل)
    ورقم الجزء من المادة المقسومة. النسخة (العام الجامعي أو تاريخ الإصدار) تُذكر إن ذكرها
    المصدر، فيستطيع النموذج أن يقول «حسب دليل القبول 1449/1448هـ».
    """
    kind = hit["doc_type"]
    if hit.get("version"):
        kind += f"، النسخة: {hit['version']}"
    header = f"[{number}] {hit['university_ar']} — {hit['doc_title']} ({kind})"
    path = []
    for part in (
        hit.get("bab"),
        hit.get("qism"),
        hit.get("appendix"),
        hit.get("article_label") or hit.get("unit_kind"),
    ):
        if part and part not in path:
            path.append(part)
    if path:
        header += " — " + " › ".join(path)
    part_number, part_count = hit["part"].split("/")
    if part_count != "1":
        header += f" (الجزء {part_number} من {part_count})"
    return header


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for number, hit in enumerate(hits, 1):
        blocks.append(passage_header(number, hit) + "\n" + hit["body"])
    return "المقاطع:\n\n" + "\n\n".join(blocks) + f"\n\nالسؤال: {question}"


def history_messages(history: list[dict]) -> list[dict]:
    """الأدوار السابقة لنموذج الإجابة رسائلَ عادية بلا مقاطعها القديمة."""
    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def first_line(answer: str) -> str:
    lines = answer.strip().splitlines()
    return lines[0].strip() if lines else ""


def is_refusal(answer: str) -> bool:
    """رفض كامل = السطر الأول جملة الرفض وحدها. الجواب الجزئي فيه «ولم تذكر المصادر
    المتاحة…»، وقد يذكر الجملة وسط الكلام («لا يوجد في المصادر المتاحة تاريخ محدد»)."""
    return first_line(answer).rstrip(".").strip() == REFUSAL


def is_greeting(answer: str) -> bool:
    return first_line(answer).startswith(GREETING.rstrip("."))


def cited_passages(answer: str, passage_count: int) -> list[int]:
    numbers = set()
    for match in CITATION.findall(answer):
        number = int(match)
        if 1 <= number <= passage_count:
            numbers.add(number)
    return sorted(numbers)


def source_cards(hits: list[dict]) -> list[dict]:
    """المصادر كما تعرضها الواجهة."""
    cards = []
    for number, hit in enumerate(hits, 1):
        cards.append(
            {
                "n": number,
                "chunk_id": hit["chunk_id"],
                "university": hit["university_ar"],
                "doc_title": hit["doc_title"],
                "label": hit.get("article_label"),
                "body": hit["body"],
                "source_url": hit.get("source_url"),
            }
        )
    return cards
