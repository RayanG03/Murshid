"""المرحلة 2 من خط البناء: يحوّل بنية الوثائق إلى مقاطع (chunks) جاهزة للـ embedding.

المدخلات: data/structured/*.json (من parse.py). المخرجات: data/chunks/{ksu,kau}.jsonl

المقطِّع يقرأ بنيةً صريحة أنتجها parse.py، فلا يخمّن العناوين بأنماط عربية، وهذا
ما يمنع نسبة الملاحق والأقسام خطأً إلى المادة السابقة.

حدود التقطيع: مادة | قاعدة تنفيذية | ملحق | قسم | باب. العناوين الفرعية داخل اللوائح
لا تكسر المادة؛ أما في الأدلة، وهي بلا مواد، فهي الحدود الوحيدة المتاحة.

كل مقطع يبدأ بترويسة سياق داخل نصه نفسه، لأن حقول الـ metadata لا تدخل في المتجه،
فبدونها يعجز البحث عن التفريق بين وثيقتين متشابهتي الصياغة (التحويل الداخلي والخارجي).

التقويم حدثٌ في كل مقطع: جدول الفصل كله في مقطع واحد فيه عشرون حدثاً متجهه عام، فتسبقه
مقاطع شروط الإجراء نفسه عند سؤال «متى أقدم على التحويل؟»؛ سطر الحدث وحده يطابق السؤال.

التشغيل:
    python chunk.py        ثم python embedding.py
"""

import hashlib
import json
import re

import config
from arabic_text import normalize

SENTENCE_END = re.compile(r"(?<=[.؟!])\s+")
SENTENCE_LINE_END = re.compile(r"[.؟!]\s*$")
FULL_DATE = re.compile(r"[\d٠-٩]{1,4}\s*[/\-]\s*[\d٠-٩]{1,2}\s*[/\-]\s*[\d٠-٩]{1,4}")
BREAKING_KINDS = {"bab", "qism", "appendix", "article", "exec"}
SECTION_KINDS = {"preamble", "section", "qism", "bab"}
KIND_NAMES = {
    "article": "مادة",
    "exec": "قاعدة تنفيذية",
    "appendix": "ملحق",
    "qism": "قسم",
    "bab": "باب",
    "section": "قسم",
    "preamble": "مقدمة",
    "entry": "قسم",
}
STUB_HINTS = ("اضغط هنا", "إضغط هنا", "الرابط أدناه", ".pdf")
CALENDAR_TYPE = "تقويم"
EVENT_PREFIXES = (normalize("الحدث"), normalize("الحركة الأكاديمية"))
EMPTY_CELL = {"", "-", "–", "∅"}

RULING_WORDS = re.compile(
    r"يجوز|يجب|يشترط|يلتزم|يحق|يتم |يمكن|تُعلن|تعلن|تُصرف|تصرف|مقدارها|"
    r"الشروط|مُعد|يُمنح|يمنح|يُحرم|يحرم|يُفصل|يفصل|لا تقل|لا يقل|"
    r"بقرار|قرار |يوافق|بتاريخ|توفر"
)

ORDINAL_SUB = re.compile(
    r"^(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|"
    r"عاشراً|الضوابط والشروط)"
)
GENERIC_LABELS = {
    normalize(word)
    for word in (
        "ملاحظة",
        "ملاحظات",
        "ملاحظة هامة",
        "ملاحظات هامة",
        "ملاحظات مهمة",
        "تنبيه",
        "تنويه",
        "معلومات عن البرنامج",
        "أولاً",
        "ثانياً",
        "ثالثاً",
        "رابعاً",
        "خامساً",
        "سادساً",
        "سابعاً",
        "ثامناً",
        "تاسعاً",
        "عاشراً",
    )
}
PROCEDURE_LINE = re.compile(r"^(إجراءات|اجراءات|شروط|آلية|الية)\s")
LIST_MARK = re.compile(r"^\s*(?:(\([\d٠-٩]+\)|[\d٠-٩]+\s*[.\-–)])|([أ-ي]\s*[.\-–)])|([-•·*]))")


def new_unit(kind: str, label: str | None, number: int | None, context: dict) -> dict:
    return {
        "kind": kind,
        "label": label,
        "no": number,
        "bab": context.get("bab"),
        "qism": context.get("qism"),
        "appendix": context.get("appendix"),
        "items": [],
    }


def is_generic_label(text: str) -> bool:
    return normalize(text).strip().rstrip(":：").strip() in GENERIC_LABELS


def close_unit(units: list[dict], unit: dict) -> None:
    if unit["items"] or unit["label"]:
        units.append(unit)


def exec_label(text: str, article_label: str | None) -> str:
    """عنوان القاعدة التنفيذية مع اسم مادتها إن لم يذكره.

    قواعد لائحة KAU كلها بعنوان واحد «القواعد التنفيذية لجامعة الملك عبدالعزيز»، فلا يعرف
    البحث ولا نموذج الإجابة لأي مادة هي (f44: ظنّ قواعد طي القيد مادة أخرى غير الانسحاب).
    القاعدة في المصدر تلي مادتها مباشرة، فنضيف اسمها بين قوسين ونُبقي العنوان الأصلي.
    عناوين KSU تذكر المادة («القاعدة التنفيذية للمادة الخامسة والأربعين») فتبقى كما هي.
    """
    if not article_label or normalize("ماده") in normalize(text):
        return text
    return f"{text} ({article_label})"


def start_unit(
    kind: str,
    text: str,
    number: int | None,
    last_article: int | None,
    context: dict,
    last_article_label: str | None = None,
) -> dict:
    if kind == "exec":
        return new_unit(
            "exec", exec_label(text, last_article_label), number or last_article, context
        )
    if kind in ("bab", "qism"):
        return new_unit(kind, text, None, context)
    if kind in ("appendix", "article"):
        return new_unit(kind, text, number, context)
    return new_unit("section", text, None, context)


def next_context(kind: str, text: str, context: dict) -> dict:
    """الباب والقسم والملحق الذي تقع فيه الوحدات بعد عنوان بنيوي."""
    if kind == "bab":
        return {"bab": text, "qism": None, "appendix": None}
    if kind == "qism":
        return {"bab": context["bab"], "qism": text, "appendix": None}
    if kind == "appendix":
        return {"bab": context["bab"], "qism": context["qism"], "appendix": text}
    return context


def units_from_blocks(blocks: list[dict], has_structure: bool) -> list[dict]:
    """يجمّع الكتل في وحدات دلالية حسب حدود البنية."""
    units = []
    context = {"bab": None, "qism": None, "appendix": None}
    unit = new_unit("preamble", None, None, context)
    last_article = None
    last_article_label = None
    for block in blocks:
        if block["t"] == "p":
            unit["items"].append({"kind": "text", "text": block["text"]})
            continue
        if block["t"] == "table":
            if block.get("unit_label"):
                close_unit(units, unit)
                unit = new_unit("entry", block["unit_label"], None, context)
            for line in linearize_table(block):
                unit["items"].append({"kind": "text", "text": line})
            continue
        kind, text, number = block["kind"], block["text"], block.get("no")
        if (
            kind not in BREAKING_KINDS
            and unit["label"]
            and normalize(text) == normalize(unit["label"])
        ):
            continue
        if kind == "exec" and unit["kind"] == "article" and not unit["items"]:
            unit["items"].append({"kind": "text", "text": text})
            continue
        is_sub = has_structure or block.get("level", 1) == 2 or is_generic_label(text)
        if kind not in BREAKING_KINDS and is_sub:
            unit["items"].append({"kind": "sub", "text": text})
            continue
        close_unit(units, unit)
        context = next_context(kind, text, context)
        if kind == "article":
            last_article = number
            last_article_label = text
        unit = start_unit(kind, text, number, last_article, context, last_article_label)
    close_unit(units, unit)
    return units


def linearize_table(block: dict) -> list[str]:
    """يحوّل الجدول إلى جمل يحمل كل منها عنوان العمود بجانب قيمته.

    الصف «الطب | 4.5 | الكليات الصحية | أحياء 112 | 90» بعنوانه في سطر بعيد جعل
    النموذج يقرأ المعدل وحده؛ الصفوف المتتالية للمدخل نفسه (العمود الأول) تُجمع
    في سطر واحد تُذكر فيه القيم الثابتة مرة وما يتغير لكل صف.
    """
    header = block.get("header") or []
    rows = []
    for row in block["rows"]:
        rows.append([cell.strip() for cell in row])
    if not header:
        return lines_without_header(rows)
    lines = []
    for kind, content in group_rows(rows, len(header)):
        if kind == "rows":
            lines.append(row_group_sentence(header, content))
        elif content:
            lines.append(content)
    return lines


def lines_without_header(rows: list[list[str]]) -> list[str]:
    lines = []
    for row in rows:
        if any(row):
            filled = [cell for cell in row if cell not in EMPTY_CELL]
            lines.append(" | ".join(filled))
    return lines


def group_rows(rows: list[list[str]], column_count: int) -> list[tuple]:
    """يجمع الصفوف المتتالية للمدخل نفسه: [("rows", صفوف)] أو [("text", سطر حاشية)]."""
    groups = []
    for row in rows:
        filled = [cell for cell in row if cell not in EMPTY_CELL]
        if len(filled) <= 1 and len(row) < column_count:
            groups.append(("text", filled[0] if filled else ""))
            continue
        row = (row + [""] * column_count)[:column_count]
        is_same_entry = bool(
            groups and groups[-1][0] == "rows" and groups[-1][1][0][0] == row[0] and row[0]
        )
        if is_same_entry:
            groups[-1][1].append(row)
        else:
            groups.append(("rows", [row]))
    return groups


def row_group_sentence(header: list[str], rows: list[list[str]]) -> str:
    constant_columns = []
    varying_columns = []
    for column in range(len(header)):
        values = {row[column] for row in rows}
        if len(values) == 1:
            constant_columns.append(column)
        else:
            varying_columns.append(column)

    parts = []
    for column in constant_columns:
        if rows[0][column] not in EMPTY_CELL:
            parts.append(f"{header[column]}: {rows[0][column]}")
    if len(varying_columns) == 1:
        column = varying_columns[0]
        values = [row[column] for row in rows if row[column] not in EMPTY_CELL]
        if values:
            parts.append(f"{header[column]}: " + "، ".join(values))
    elif varying_columns:
        variants = []
        for row in rows:
            pieces = []
            for column in varying_columns:
                if row[column] not in EMPTY_CELL:
                    pieces.append(f"{header[column]}: {row[column]}")
            if pieces:
                variants.append("، ".join(pieces))
        if len(variants) > 1:
            for number, variant in enumerate(variants, 1):
                parts.append(f"({number}) {variant}")
        else:
            parts += variants
    sentence = "؛ ".join(part for part in parts if part)
    return sentence.rstrip(".") + "."


def is_heading_only(unit: dict) -> bool:
    """وحدة بلا محتوى، أو كل ما فيها جملة تمهيد قصيرة تنتهي بنقطتين وقائمتها في الوحدة التالية
    («التميز الأكاديمي للبرامج الدراسية» ثم «برامج وشروط قبول الطلبة المتميزين أكاديمياً:»).
    الفقرة الطويلة التي تنتهي بنقطتين («… فيما يلي تفاصيل خيارات القبول المتاحة:») محتوى،
    فتبقى وحدها ولا تُلصق فوق المجموعة الأولى بعدها.
    """
    for item in unit["items"]:
        if item["kind"] == "sub":
            continue
        is_lead = item["text"].rstrip().endswith(":") and len(item["text"]) <= config.LEAD_MAX
        if not is_lead:
            return False
    return True


def attach_empty_headings(units: list[dict]) -> list[dict]:
    """عنوان بلا محتوى في الأدلة («دراسة طالب زائر:» ثم «اولاً: طالب مسجل بجامعة أخرى…»)
    يُكتب في أول الوحدة التالية، وتبقى الوحدة باسمها.

    كان الدمج يلصق كل وحدة أقصر من 400 حرف بالتالية تحت اسم الأولى، فصارت «كلية السياحة»
    داخل مقطع «كلية الحقوق» (f30). الوحدة القصيرة التي فيها محتوى معلومة كاملة فتبقى وحدها.
    قبل جدول مسمّى (عنوان مجموعة كليات) لا يُلصق شيء، فاسم المجموعة في صفوف الجدول،
    وتبقى جملة التمهيد إن وُجدت وحدةً كما هي.
    """
    result = []
    pending = []
    for unit in units:
        if unit["kind"] not in SECTION_KINDS:
            result += [waiting for waiting in pending if waiting["items"]]
            pending = []
            result.append(unit)
            continue
        if is_heading_only(unit):
            pending.append(unit)
            continue
        parents = []
        for waiting in pending:
            if waiting["label"]:
                parents.append({"kind": "sub", "text": waiting["label"]})
            parents += waiting["items"]
        unit["items"] = parents + unit["items"]
        pending = []
        result.append(unit)
    result += [waiting for waiting in pending if waiting["items"]]
    return result


def split_items(items: list[dict], max_chars: int) -> list[list[dict]]:
    """يقسم الوحدة أجزاءً عند حدود العناصر؛ الفقرة الأطول من السقف تُفتَّت عند حدود الجُمل."""
    pieces = []
    for item in items:
        if item["kind"] == "text" and len(item["text"]) > max_chars:
            for sentence in SENTENCE_END.split(item["text"]):
                if sentence.strip():
                    pieces.append({"kind": "text", "text": sentence.strip()})
        else:
            pieces.append(item)

    parts = []
    current = []
    size = 0
    for item in pieces:
        item_size = len(item["text"]) + 1
        if current and size + item_size > max_chars:
            parts.append(current)
            current = []
            size = 0
        current.append(item)
        size += item_size
    if current:
        parts.append(current)
    if len(parts) > 1:
        return carry_context(parts)
    return parts


def opens_list(item: dict) -> bool:
    return item["kind"] == "sub" or item["text"].rstrip().endswith(":")


def list_marker(text: str) -> str | None:
    """نوع ترقيم البند: num (رقم) | letter (حرف) | dash (شرطة) | None لجملة عادية."""
    match = LIST_MARK.match(text)
    if match is None:
        return None
    if match.group(1) is not None:
        return "num"
    if match.group(2) is not None:
        return "letter"
    return "dash"


def shorten(text: str) -> str:
    text = text.rstrip(":").strip()
    if len(text) <= config.LEAD_MAX:
        return text
    return text[: config.LEAD_MAX].rsplit(" ", 1)[0] + "…"


def move_dangling_headings(parts: list[list[dict]]) -> None:
    """عنوانٌ أو تمهيدٌ في ذيل جزء ينتقل إلى الجزء التالي حيث متنه."""
    for index in range(len(parts) - 1):
        current_part = parts[index]
        next_part = parts[index + 1]
        while len(current_part) > 1 and opens_list(current_part[-1]):
            next_part.insert(0, current_part.pop())


def update_open_context(state: dict, part: list[dict]) -> None:
    """يتتبع ما يبقى مفتوحاً بعد الجزء: العنوان الفرعي وجملة التمهيد ونوع بنودها."""
    for item in part:
        if item["kind"] == "sub":
            if ORDINAL_SUB.match(item["text"]):
                state["ordinal_heading"] = item["text"]
            else:
                state["heading"] = item["text"]
                state["ordinal_heading"] = None
            state["lead"] = None
        elif item["kind"] == "text" and item["text"].rstrip().endswith(":"):
            state["lead"] = item["text"]
            state["lead_marker"] = "?"
        elif state["lead"] and state["lead_marker"] == "?" and item["kind"] == "text":
            state["lead_marker"] = list_marker(item["text"])


def continuation_line(part: list[dict], state: dict) -> str | None:
    trail = []
    for heading in (state["heading"], state["ordinal_heading"]):
        if heading:
            trail.append(heading)
    lead = state["lead"]
    if lead and not opens_list(part[0]):
        if list_marker(lead) is None or list_marker(part[0]["text"]) == state["lead_marker"]:
            trail.append(lead)
    if not trail:
        return None
    return "تابع: " + " › ".join(shorten(text) for text in trail)


def carry_context(parts: list[list[dict]]) -> list[list[dict]]:
    """الجزء الثاني فما بعده يبدأ وسط قائمة: «٥. يُشترط أن تكون الجهة الشريكة مرخصة»
    لا يقول أي برنامج يخص، والقاعدة التنفيذية للمادة ١٤ تجمع أربعة برامج بترقيم
    يتكرر. فيُسبق كل جزء بسطر «تابع:» يحمل العنوان الفرعي المفتوح وجملة التمهيد.
    عدد الأجزاء لا يتغير.
    """
    move_dangling_headings(parts)
    state = {"heading": None, "ordinal_heading": None, "lead": None, "lead_marker": None}
    result = []
    for index, part in enumerate(parts):
        if index > 0:
            line = continuation_line(part, state)
            if line:
                part = [{"kind": "ctx", "text": line}] + part
        result.append(part)
        update_open_context(state, part)
    return result


def chunk_header(doc: dict, unit: dict) -> str:
    """ترويسة السياق: [الجامعة · الوثيقة · المرحلة · الباب · القسم · الملحق]."""
    bits = [config.UNIVERSITY_NAMES[doc["uni"]], doc["title"]]
    stage = doc.get("stage")
    if stage and stage != config.UNDERGRADUATE_STAGE and stage not in doc["title"]:
        bits.append(stage)
    for key in ("bab", "qism", "appendix"):
        if unit.get(key) and unit[key] != unit["label"]:
            bits.append(unit[key])
    return "[" + " · ".join(bits) + "]"


def unit_id(doc: dict, unit: dict) -> str:
    """معرّف من بنية الوحدة لا من موضعها، فإضافة كتلة لا تغيّر معرّفات غيرها:
    مادة ← art025، قاعدة تنفيذية ← exec025، ملحق ← appx01، وغيرها ← sec_ وبصمة
    من الباب والقسم والعنوان."""
    prefix = f"{doc['uni'].lower()}_{doc['slug']}"
    number = unit["no"]
    if unit["kind"] == "article" and number:
        return f"{prefix}_art{number:03d}"
    if unit["kind"] == "exec" and number:
        return f"{prefix}_exec{number:03d}"
    if unit["kind"] == "appendix" and number:
        return f"{prefix}_appx{number:02d}"
    key = "|".join(part or "" for part in (unit["bab"], unit["qism"], unit["label"]))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_sec_{digest}"


def unique_id(base_id: str, used_ids: set) -> str:
    """عدّاد فقط عند تصادم حقيقي (دليل حقوق KAU فيه ثلاثة أقسام بعنوان «حقوق الطالب»)."""
    chunk_id = base_id
    counter = 1
    while chunk_id in used_ids:
        counter += 1
        chunk_id = f"{base_id}_{counter}"
    used_ids.add(chunk_id)
    return chunk_id


def make_chunk(
    doc: dict, unit: dict, part: list[dict], part_number: int, part_count: int, chunk_id: str
) -> dict | None:
    label = unit["label"] or ""
    if part_count > 1:
        part_label = f"({part_number}/{part_count})"
        label = f"{label} {part_label}" if label else part_label
    body = "\n".join(item["text"] for item in part).strip()
    if not body and not label:
        return None
    lines = [chunk_header(doc, unit)]
    if label:
        lines.append(label)
    if body:
        lines.append(body)
    text = "\n".join(lines)
    return {
        "chunk_id": chunk_id,
        "university": doc["uni"],
        "university_ar": config.UNIVERSITY_NAMES[doc["uni"]],
        "doc_title": doc["title"],
        "doc_type": doc["doc_type"],
        "stage": doc.get("stage", config.UNDERGRADUATE_STAGE),
        "version": doc.get("version"),
        "source_file": doc["source_file"],
        "bab": unit.get("bab"),
        "qism": unit.get("qism"),
        "appendix": unit.get("appendix"),
        "unit_kind": KIND_NAMES.get(unit["kind"], unit["kind"]),
        "article_label": unit["label"],
        "part": f"{part_number}/{part_count}",
        "text": text,
        "body": body,
        "is_stub": is_link_stub(body),
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "indexable": doc.get("indexable", True),
        "source_url": doc.get("source_url"),
    }


def is_link_stub(body: str) -> bool:
    """مقطع قصير لا يحوي إلا إحالة («اضغط هنا»، رابط PDF) لا معلومة."""
    if len(body) >= config.STUB_MAX:
        return False
    for hint in STUB_HINTS:
        if hint in body:
            return True
    return False


def document_units(doc: dict) -> list[dict]:
    has_structure = False
    for block in doc["blocks"]:
        if block["t"] == "h" and block["kind"] in ("article", "qism", "bab", "appendix"):
            has_structure = True
    units = units_from_blocks(doc["blocks"], has_structure)
    if not has_structure:
        units = attach_empty_headings(units)
    return units


def review_notes(doc: dict) -> list[str]:
    """وحدات تحتاج نظرة بشرية: اسم عام، أو أكثر من عنوان إجراء في وحدة واحدة."""
    notes = []
    for unit in document_units(doc):
        label = unit["label"] or ""
        if label and is_generic_label(label):
            notes.append(f"اسم عام «{label}»")
        procedures = []
        if PROCEDURE_LINE.match(label):
            procedures.append(label)
        for item in unit["items"]:
            is_heading_like = item["kind"] == "sub" or len(item["text"]) <= 90
            if is_heading_like and PROCEDURE_LINE.match(item["text"]):
                procedures.append(item["text"])
        if len(procedures) > 1:
            notes.append(
                f"«{label[:50]}» فيها {len(procedures)} عناوين إجراءات: "
                + " | ".join(text[:40] for text in procedures)
            )
    return notes


def is_cover(body: str) -> bool:
    """غلاف الوثيقة (اسم الجهة والعام الجامعي): قصير، بلا حكم ولا جملة تامة ولا تاريخ كامل.
    الغلاف قد يقع في أكثر من مقطع أول («جامعة الملك عبدالعزيز عمادة القبول والتسجيل» ثم
    «للعام الجامعي…»)؛ وسطر «بداية الفصل الدراسي الأول - الأحد ٢٠٢٦/٠٨/٢٣م» معلومة لا غلاف."""
    if len(body) >= config.COVER_MAX or RULING_WORDS.search(body) or FULL_DATE.search(body):
        return False
    return not any(SENTENCE_LINE_END.search(line) for line in body.splitlines())


def is_event_line(text: str) -> bool:
    """سطر حدث في جدول تقويم: «الحدث (…): اسمه؛ التاريخ…» أو «الحركة الأكاديمية: اسمها؛ …»."""
    field, separator, rest = text.partition(":")
    if not separator or "؛" not in rest:
        return False
    if not normalize(field.strip()).startswith(EVENT_PREFIXES):
        return False
    return bool(rest.split("؛", 1)[0].strip())


def build_chunks(doc: dict) -> list[dict]:
    units = document_units(doc)
    chunks = []
    used_ids = set()
    for unit in units:
        items = [item for item in unit["items"] if item["text"].strip()]
        if not items:
            continue
        base_id = unique_id(unit_id(doc, unit), used_ids)
        events = []
        if doc["doc_type"] == CALENDAR_TYPE:
            events = [item for item in items if is_event_line(item["text"].strip())]
            items = [item for item in items if item not in events]
        if items:
            parts = split_items(items, doc.get("max_chars", config.MAX_CHARS))
            for part_number, part in enumerate(parts, 1):
                chunk_id = base_id if len(parts) == 1 else f"{base_id}_p{part_number}"
                chunk = make_chunk(doc, unit, part, part_number, len(parts), chunk_id)
                if chunk:
                    chunks.append(chunk)
        for number, item in enumerate(events, 1):
            chunk = make_chunk(doc, unit, [item], 1, 1, f"{base_id}_r{number:02d}")
            if chunk:
                chunks.append(chunk)

    for chunk in chunks:
        if not is_cover(chunk["body"]):
            break
        chunk["is_stub"] = True
    return chunks


def print_summary(uni: str, chunks: list[dict], path_name: str) -> None:
    sizes = sorted(len(chunk["text"]) for chunk in chunks)
    articles = sum(1 for chunk in chunks if chunk["unit_kind"] == "مادة")
    stubs = sum(1 for chunk in chunks if chunk["is_stub"])
    print(
        f"  {path_name}: {len(chunks)} chunk | {sum(sizes):,} حرف | مواد {articles} | "
        f"وسيط {sizes[len(sizes) // 2]} | أكبر {sizes[-1]} | stub {stubs}"
    )


def main() -> None:
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_by_university = {"KSU": [], "KAU": []}
    notes = []
    for path in sorted(config.STRUCTURED_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for note in review_notes(doc):
            notes.append(f"{doc['uni']} {doc['title']}: {note}")
        chunks = build_chunks(doc)
        chunks_by_university[doc["uni"]].extend(chunks)
        sizes = sorted(len(chunk["text"]) for chunk in chunks)
        status = " | تحتاج مراجعة" if not doc.get("indexable", True) else ""
        print(
            f"  {doc['uni']} {doc['title'][:42]:<44} {len(chunks):>4} chunk | "
            f"وسيط {sizes[len(sizes) // 2]:>5} | أكبر {sizes[-1]:>5}{status}"
        )

    print()
    for uni, chunks in chunks_by_university.items():
        out_path = config.CHUNKS_DIR / f"{uni.lower()}.jsonl"
        with out_path.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print_summary(uni, chunks, out_path.name)
    print(f"\nللمراجعة البشرية ({len(notes)}):")
    for note in notes:
        print("  - " + note)
    print(f"\nالمجلد: {config.CHUNKS_DIR}")


if __name__ == "__main__":
    main()
