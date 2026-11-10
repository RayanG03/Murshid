"""تصحيح الجداول التي أفسدها الاستخراج، قبل تقطيعها.

يستعمله parse.py بعد قراءة كل وثيقة. كل إصلاح هنا سببه خلل محدد في الجدول المستخرج
قورن بالـ PDF الأصلي: أعمدة مُزاحة، أو عناوين ضائعة، أو خلايا فارغة تعني «كالسطر
السابق». بعد الإصلاح يحوّل chunk.py كل صف إلى جملة تحمل عنوان كل عمود بجانب قيمته.

الاستخدام:
    import table_fixes
    blocks = table_fixes.apply("KSU", "sihhiya", blocks)
"""

import re

GENDER = re.compile(r"طلاب|طالبات")


def tables_in(blocks: list[dict]) -> list[dict]:
    tables = []
    for block in blocks:
        if block["t"] == "table":
            tables.append(block)
    return tables


def pad_row(row: list[str], length: int) -> list[str]:
    """يكمل الصف بخلايا فارغة أو يقصّه حتى يصير بطول length."""
    return (list(row) + [""] * length)[:length]


SIHHIYA_HEADER = [
    "تصنيف حالة الإعاقة",
    "أمثلة للحالات الفرعية",
    "محدد التصنيف على النظام الأكاديمي",
]
SIHHIYA_ROWS = [
    [
        "الإعاقة السمعية (صمم)",
        "بكم مع فقدان السمع العميق بالأذنين، بكم مع فقدان السمع بالأذنين / زراعة قوقعة",
        "حاجة الطالب إلى ترجمة لغة الإشارة للتواصل وفي مواقف التعلم (كالمحاضرات).",
    ],
    [
        "الإعاقة السمعية (ضعف السمع)",
        "ضعف بالسمع أذن واحدة، زراعة القوقعة الإلكترونية، فقدان السمع العميق بالأذنين، "
        "ضعف سمع بالأذنين، صمم أذن واحدة، فقدان السمع العميق بالأذنين / زراعة قوقعة",
        "لا يحتاج الطالب إلى ترجمة لغة الإشارة للتواصل وفي مواقف التعلم (كالمحاضرات). "
        "قدرة الطالب على التواصل (الاستقبال والتعبير) من خلال اللغة المنطوقة، "
        "وبالاستفادة من المعينات السمعية أو زراعة القوقعة أو قراءة الشفاه أو غيرها من "
        "الأدوات أو الأجهزة أو الطرق المساعدة.",
    ],
    [
        "الإعاقة البصرية (كف البصر)",
        "كف إبصار قانوني بالعينين، كف إبصار بالعينين/مهاق",
        "عدم قدرة الطالب على القيام بأي مهمة تتطلب الرؤية بشكل أساسي.",
    ],
    [
        "الإعاقة البصرية (ضعف بصر)",
        "كف إبصار عين واحدة، ضعف إبصار بالعينين، ضعف إبصار عين واحدة، كف إبصار عين "
        "واحدة وضعف إبصار عين أخرى",
        "قدرة الطالب المحدودة على أداء بعض المهام التي تتطلب الرؤية بشكل أساسي "
        "وبالاستعانة بالوسائل المعينة كالأدوات أو الأجهزة المكبرة.",
    ],
    [
        "الحالات الصحية الأخرى (المزمنة أو الشديدة)",
        "أمراض القلب/ فشل عضلة القلب، قصور التنفس، فشل كلوي نهائي، فشل كلوي، فشل بالكبد "
        "على قائمة الزراعة، ورم خبيث، استسقاء بالدماغ، التلاسيميا الكبرى، أنيميا "
        "منجلية، حامل أنيميا منجلية، فقر الدم اللامصنع، نقص المناعة الدائم، عسر تكون "
        "العظم والغضروف، روماتويد، الرنح العائلي، متلازمة الصبغي الهش، الذئبة الحمراء، "
        "التهاب مفاصل روماتيزمي، زراعة أحد الأعضاء، سيولة في الدم، الإسهال المضيع "
        "للكلور، التغذية المحيطية الوريدية، التغذية المحيطية غير الوريدية، الأمراض "
        "الاستقلابية الوراثية، أمراض المناعة الذاتية المزمنة، مرض جلدي، بهاق، داء كرون "
        "والتهاب القولون القرحي، الحساسية لمادة الغلوتين، صدفية جلدية، داء بهجت، التصلب "
        "المتعدد، التليف الكيسي، حمى البحر الأبيض المتوسط العائلية، الفوال، فقد الإحساس "
        "والتعرق، فشل النمو وسوء التغذية المعند، الورم العصبي الليفي، صرع مستعصي، عدم "
        "التحكم بالمخارج، متلازمة مارفان، تجلط الدم، الزهايمر المبكر، سرطان الدم، "
        "التسمم بالرصاص، الهموفيليا (عدم تخثر الدم)، متلازمة توريت، السكري، الربو، "
        "متلازمة شوارتز جامبل (شد مستمر في عضلات الوجه)",
        "الطالب تحت العلاج أو المتابعة للحالة الصحية، وتسبب الحالة الصحية بعض القيود في الأداء.",
    ],
    [
        "الإعاقة الجسمية والحركية",
        "شلل طرف علوي أيمن/أيسر، شلل طرف سفلي أيمن/أيسر، ضعف شقي أيمن/أيسر، شلل طرفين "
        "علويين/سفليين، شلل طرفين (علوي أيمن وسفلي أيسر)، شلل طرفين (علوي أيسر وسفلي "
        "أيمن)، شلل شقي أيمن/أيسر، شلل ثلاثي/رباعي وإقعاد تام، بتر طرف علوي أيمن/أيسر، "
        "بتر طرف سفلي أيمن/أيسر، بتر طرفين علويين/سفليين، بتر طرفين (علوي أيمن وسفلي "
        "أيسر)، بتر طرفين (علوي أيسر وسفلي أيمن)، بتر شقي أيمن/أيسر، بتر ثلاثي/رباعي، "
        "ضعف ثلاثي/رباعي (الأطراف الأربعة)، بتر جزئي طرف علوي أيمن/أيسر، بتر جزئي طرف "
        "سفلي أيمن، بتر جزئي طرف علوي أيمن، ضعف طرفين سفليين وصعوبة شديدة بالمشي، ضعف "
        "طرفين علويين، ضعف طرفين (علوي أيمن وسفلي أيسر)، ضعف طرفين (علوي أيسر وسفلي "
        "أيمن)، ضعف بطرف علوي/سفلي، قصر طرف سفلي شديد، تشوهات شديدة بالمظهر الخارجي "
        "(مشوهة) قد تكون بالأعضاء الداخلية، تشوهات بسيطة، قزامة، الكسور أو الحروق التي "
        "تؤدي إلى تقلصات، شلل الأطفال، سل العظام، ضعف العظام الناتج عن الشلل الدماغي، "
        "العظم الزجاجي",
        "",
    ],
    [
        "الاضطرابات السلوكية والانفعالية",
        "كاضطرابات القلق والاكتئاب واضطراب الشخصية الحدية واضطراب ثنائي القطب",
        "",
    ],
    [
        "اضطرابات التواصل (اضطراب اللغة والكلام)",
        "عيوب نطق شديدة، اضطرابات الكلام (النطق، الطلاقة، الصوت)، اضطرابات اللغة "
        "(التعبيرية أو الاستقبالية)",
        "",
    ],
    [
        "الإعاقات المزدوجة والمتعددة",
        "ضعف بالأطراف الأربعة/ صعوبة بالنطق، بكم مع فقدان السمع العميق بالأذنين/ ضعف "
        "إبصار بالعينين، كف إبصار بالعينين/مهاق/تشوهات شديدة بالمظهر الخارجي، كف إبصار "
        "بالعينين/ ورم خبيث تحت العلاج بالكيماوي، كف إبصار بالعينين/ ضعف بالأطراف/ "
        "صعوبات تعلم، شلل شقي أيمن/ استسقاء بالدماغ/ صعوبة بالنطق",
        "",
    ],
    [
        "اضطراب طيف التوحد",
        "",
        "",
    ],
    [
        "صعوبات التعلم",
        "",
        "",
    ],
    [
        "اضطراب فرط الحركة وتشتت الانتباه",
        "",
        "",
    ],
]


def fix_ksu_sihhiya(blocks: list[dict]) -> list[dict]:
    """يستبدل جداول التصنيف كلها بالجدول المنقول من الـ PDF، في موضع أولها."""
    first_table = 0
    while blocks[first_table]["t"] != "table":
        first_table += 1
    kept = []
    for block in blocks:
        if block["t"] != "table":
            kept.append(block)
    rows = []
    for row in SIHHIYA_ROWS:
        rows.append(list(row))
    fixed = {"t": "table", "header": SIHHIYA_HEADER, "rows": rows, "curated": True}
    return kept[:first_table] + [fixed] + kept[first_table:]


KSU_COLLEGES = {
    "العلوم الإنسانية والاجتماعية",
    "التربية",
    "اللغات وعلومها",
    "الفنون",
    "الحقوق والعلوم السياسية",
    "السياحة والآثار",
    "علوم الرياضة والنشاط البدني",
    "الطب",
    "طب الأسنان",
    "الصيدلة",
    "الأمير سلطان للخدمات الطبية الطارئة",
    "التمريض",
    "العلوم الطبية التطبيقية",
    "الهندسة",
    "العلوم",
    "العمارة والتخطيط",
    "علوم الحاسب والمعلومات",
    "علوم الأغذية والزراعة",
    "إدارة الأعمال",
    "الهندسة التطبيقية (المزامنية)",
    "علوم الحاسب التطبيقي (المزامنية)",
    "إدارة الأعمال التطبيقية (المزامنية)",
}


def find_gender(cells: list[str]) -> str:
    for cell in cells:
        if GENDER.search(cell):
            return cell
    return ""


def find_college(cells: list[str]) -> str:
    for cell in cells:
        if cell in KSU_COLLEGES:
            return cell
    return ""


def find_program(cells: list[str], gender: str, college: str) -> str:
    """البرنامج هو الخلية الوحيدة التي ليست جنساً ولا كلية."""
    for cell in cells:
        if cell and cell not in (gender, college):
            return cell
    return ""


def fix_ksu_qabul(blocks: list[dict]) -> list[dict]:
    college = ""
    gender = ""
    for table in tables_in(blocks):
        rows = []
        for row in table["rows"]:
            cells = [cell.strip() for cell in pad_row(row, 3)]
            if not any(cells):
                continue
            row_gender = find_gender(cells)
            row_college = find_college(cells)
            college = row_college or college
            gender = row_gender or gender
            program = find_program(cells, row_gender, row_college)
            if program:
                rows.append([college, program, gender])
        table["header"] = ["الكلية", "البرنامج", "الجنس"]
        table["rows"] = rows
        table["curated"] = True
    return blocks


def fix_ksu_taqwim(blocks: list[dict]) -> list[dict]:
    """كل جدول فصلٌ دراسي: يصير مقطعاً مستقلاً باسم فصله، ويدخل اسم الفصل عمود الحدث.

    قبلُ كان التقويم كله وحدة تُقسم بالحجم، فجمع الجزء الأول الفصل الأول وبعض الثاني.
    """
    for table in tables_in(blocks):
        semester = table["header"][0]
        table["header"][0] = f"الحدث ({semester})"
        table["unit_label"] = semester
    return blocks


IELTS_HEADER = [
    "مسار مقررات معهد اللغة الإنجليزية",
    "IELTS",
    "TOEFL iBT (المجموع الكلي الأدنى وفق النطاق القديم)",
    "TOEFL iBT (الدرجة الكلية الأدنى وفق النطاق الجديد)",
    "PTE Academic (الدرجة الأدنى)",
]
MEDICALLY_FIT = "أن يكون الطالب لائقاً طبياً"
STAR_EXTRAS = {
    "الطب": ["إجراء مقابلة شخصية", MEDICALLY_FIT],
    "طب الأسنان": [
        "إجراء مقابلة شخصية",
        MEDICALLY_FIT,
        "اجتياز اختبار Dental Admissions Test - DAT",
    ],
    "الصيدلة": [MEDICALLY_FIT],
    "التمريض": [MEDICALLY_FIT],
    "علوم التأهيل الطبي": [MEDICALLY_FIT],
    "العلوم الطبية التطبيقية": [MEDICALLY_FIT],
    "الدراسات البحرية": ["أن لا يعاني الطالب من رهاب البحر"],
    "علوم البحار": ["أن لا يعاني الطالب من رهاب البحر"],
}
EXTRAS_COLUMN = "شروط إضافية لتثبيت التحويل"
HEADER_RENAME = {
    "الدرجة الأدنى": "الدرجة الأدنى المطلوبة في مقررات المعايير",
    "المعايير الخاصة بالكلية": "المعايير الخاصة بالكلية (مقررات مطلوبة)",
}


def split_star_extras(college: str) -> tuple[str, str]:
    """«الطب*» ← («الطب»، «إجراء مقابلة شخصية، أن يكون الطالب لائقاً طبياً»)."""
    if "*" not in college:
        return college, ""
    name = college.replace("*", "").replace("  ", " ").strip()
    base_name = name.split(" (")[0].strip()
    return name, "، ".join(STAR_EXTRAS.get(base_name, []))


def fix_english_exemption_table(table: dict) -> list[dict]:
    """يعيد أعمدة جدول الإعفاء من الإنجليزية إلى مواضعها؛ الصفوف ذات الخلية الواحدة حواشٍ."""
    rows = []
    notes = []
    for row in table["rows"]:
        if len(row) == 1:
            notes.append(row[0])
        elif not row[0].startswith("الدرجة"):
            pte, toefl_new, toefl_old, ielts, track = pad_row(row, 5)
            rows.append([track, ielts, toefl_old, toefl_new, pte])
    fixed = [{"t": "table", "header": IELTS_HEADER, "rows": rows, "curated": True}]
    for note in notes:
        fixed.append({"t": "p", "text": note})
    return fixed


def fill_college_rows(table: dict) -> None:
    """يملأ الخلايا الفارغة من السطر السابق، ويضيف عمود الشروط الإضافية."""
    previous = None
    rows = []
    for row in table["rows"]:
        row = [cell.strip() for cell in pad_row(row, 5)]
        if previous and row[0].startswith("اجتياز المقررات"):
            row = [previous[0], previous[1], row[2] or previous[2], row[0], row[4]]
        elif previous and not row[0]:
            row = [previous[0], previous[1], previous[2], row[3], row[4]]
        elif previous and row[0] == previous[0] and not row[1]:
            row[1] = previous[1]
        rows.append(row)
        previous = row
    for row in rows:
        row[0], extras = split_star_extras(row[0])
        row.append(extras)
    header = []
    for name in pad_row(table["header"], 5):
        header.append(HEADER_RENAME.get(name.strip(), name))
    table["header"] = header + [EXTRAS_COLUMN]
    table["rows"] = rows


def one_table_per_college(tables: list[dict]) -> list[dict]:
    """جدول لكل كلية فتصير كل كلية مقطعاً بشروطها كاملة، كما في صفحة KSU.

    صفوف الكلية الواحدة قد تقع في جدولين (فاصل صفحة)، ومسارا الكلية الواحدة («التمريض
    (من كليات صحية)» و«(من كليات غير صحية)») يبقيان معاً في مقطعها.
    """
    colleges = []
    rows_by_college = {}
    for table in tables:
        for row in table["rows"]:
            college = row[0].split(" (")[0].strip()
            if college not in rows_by_college:
                colleges.append(college)
                rows_by_college[college] = []
            rows_by_college[college].append(row)
    college_tables = []
    for college in colleges:
        college_tables.append(
            {
                "t": "table",
                "header": tables[0]["header"],
                "rows": rows_by_college[college],
                "unit_label": f"كلية {college}",
                "curated": True,
            }
        )
    return college_tables


def fix_kau_tahwil_dakhili(blocks: list[dict]) -> list[dict]:
    fixed = []
    college_tables = []
    first_college_position = None
    for block in blocks:
        if block["t"] != "table":
            fixed.append(block)
            continue
        header = block.get("header") or []
        if header and header[0] == "PTE Academic":
            fixed += fix_english_exemption_table(block)
            continue
        if header and header[0] == "الكلية":
            fill_college_rows(block)
            if first_college_position is None:
                first_college_position = len(fixed)
            college_tables.append(block)
            continue
        fixed.append(block)
    if first_college_position is None:
        return fixed
    before = fixed[:first_college_position]
    after = fixed[first_college_position:]
    return before + one_table_per_college(college_tables) + after


def fix_kau_taqwim(blocks: list[dict]) -> list[dict]:
    for table in tables_in(blocks):
        if table["rows"] and table["rows"][0][:2] == ["الحدث", "الأسبوع"]:
            table["header"] = [
                "الحدث",
                "من (الأسبوع)",
                "من (اليوم والتاريخ)",
                "إلى (الأسبوع)",
                "إلى (اليوم والتاريخ)",
            ]
            table["rows"] = table["rows"][1:]
    return blocks


def fix_kau_qabul(blocks: list[dict]) -> list[dict]:
    for table in tables_in(blocks):
        if not table.get("header") and table["rows"] and len(table["rows"][0]) == 3:
            table["header"] = [
                "النوع",
                "مسارات الثانوية العامة",
                "مسار قبول ذوي الإعاقة السمعية الشديدة",
            ]
    return blocks


def fix_kau_tahwil_khariji(blocks: list[dict]) -> list[dict]:
    for table in tables_in(blocks):
        if not table.get("header"):
            table["header"] = ["الإجراء", "الموعد"]
    return blocks


EMPLOYER_APPROVAL_JOBS = ["معيد / محاضر", "القضاة", "جميع الموظفين (عدا الفئات المذكورة أعلاه)"]


def fix_kau_ulya_qabul(blocks: list[dict]) -> list[dict]:
    """جدول «موافقة جهة العمل» (ص 10): عمود الوظيفة مُزاح صفاً عن عمود الموافقة،
    و«القضاة» قُرئت «القضية»، وعنوان «أ- الموظفين:» فوق الجدول صار رأس عمود."""
    for table in tables_in(blocks):
        if table["header"] != ["موافقة جهة العمل المطلوبة", "أ- الموظفين:"]:
            continue
        approvals = [row[0] for row in table["rows"][:3]]
        if len(approvals) != 3 or not all(approvals):
            raise ValueError("جدول موافقة جهة العمل KAU تغيّر — راجع fix_kau_ulya_qabul")
        table["header"] = ["الوظيفة (الموظفون)", "موافقة جهة العمل المطلوبة"]
        table["rows"] = [[job, approval] for job, approval in zip(EMPLOYER_APPROVAL_JOBS, approvals)]
    return blocks


DIPLOMA_FEE = "3,000 ريال غير شاملة ضريبة القيمة المضافة** (3,450 ريال شاملة ضريبة القيمة المضافة)"


def fix_kau_diploma_bued(blocks: list[dict]) -> list[dict]:
    """جدول الرسوم (ص 31): عمود الرسوم في الـ PDF خلية واحدة لكل البرامج فيها السعر
    مرتين، 3,000 غير شاملة الضريبة و3,450 شاملة (3,000 × 1.15)، فوزّعهما Chandra على
    الصفوف كأن لنصف البرامج سعراً وللنصف الآخر سعراً. ورمز الريال صورة سقطت."""
    for table in tables_in(blocks):
        header = table["header"] or []
        if not header or "الرسوم الدراسية" not in header[0]:
            continue
        if len(table["rows"]) != 24 or not all(row[0][:5] in ("3,000", "3,450") for row in table["rows"]):
            raise ValueError("جدول رسوم الدبلومات عن بعد KAU تغيّر — راجع fix_kau_diploma_bued")
        for row in table["rows"]:
            row[0] = DIPLOMA_FEE
    return blocks


PROGRAMS_COLUMNS = 5
COLLEGE_OF_DEPARTMENT = {"الدراسات الإسلامية": "التربية"}
PROGRAM_STARTS = ("ماجستير", "دكتوراه", "الماجستير", "الدكتوراه", "الدبلوم")


def fix_ksu_ulya_baramij(blocks: list[dict]) -> list[dict]:
    """جدول برامج الدراسات العليا KSU، جدول لكل صفحة. خلية الكلية (وأحياناً القسم واللغة)
    ممتدة على صفوف كثيرة، فتأتي فارغة في أول الصفحة التالية: تُملأ من الصف السابق.
    وقسم الدراسات الإسلامية (ص 2) تبدأ خلية كليته «التربية» عنده وكتبها Chandra في صف
    علم النفس بعده، فالسابق هنا خطأ. وآخر خمسة صفوف في ص 13 زاد في أولها عمود فارغ
    فانزاحت الأعمدة، وكذلك صف الرياضيات آخر ص 7 (أنهى Chandra خلية «العلوم» قبله فانزاح
    يساراً: البرنامج في عمود القسم). وبعض الصفحات كتب Chandra أعمدتها من اليسار («لغة البرنامج» أولاً)،
    فتُقلب."""
    header = ["الكلية", "القسم", "البرامج", "التخصص/ المسار", "لغة البرنامج"]
    previous = [""] * PROGRAMS_COLUMNS
    for table in tables_in(blocks):
        is_reversed = table["header"] == header[::-1]
        if table["header"] != header and not is_reversed:
            continue
        table["header"] = header
        rows = []
        for row in table["rows"]:
            if len(row) == PROGRAMS_COLUMNS + 1 and not row[0].strip():
                row = row[1:]
            row = pad_row(row, PROGRAMS_COLUMNS)
            if is_reversed:
                row = row[::-1]
            if row[1].startswith(PROGRAM_STARTS):
                row = [""] + row[:-1]
            for column in range(PROGRAMS_COLUMNS):
                if not row[column].strip():
                    row[column] = previous[column]
            if row[1] in COLLEGE_OF_DEPARTMENT:
                row[0] = COLLEGE_OF_DEPARTMENT[row[1]]
            rows.append(row)
            previous = row
        table["rows"] = rows
    return programs_by_college(blocks, header)


def programs_by_college(blocks: list[dict], header: list[str]) -> list[dict]:
    """جدول مسمّى لكل كلية بدل جدول لكل صفحة، كأدلة البكالوريوس: الكلية في اسم المقطع،
    والصفوف تُجمع بالقسم، فتنقسم الكلية الكبيرة (التربية 59 صفاً) عند حدود الأقسام.
    جدول الصفحة بالكلية في عمودها الأول جعل كل الكلية جملة واحدة طولها 4,600 حرف."""
    colleges = {}
    first_index = None
    kept = []
    for block in blocks:
        if block["t"] == "table" and block["header"] == header:
            if first_index is None:
                first_index = len(kept)
            for row in block["rows"]:
                colleges.setdefault(row[0], []).append(row[1:])
            continue
        kept.append(block)
    if first_index is None:
        return blocks
    tables = []
    for college, rows in colleges.items():
        label = college if college.startswith("عمادة") else f"كلية {college}"
        tables.append({"t": "table", "header": header[1:], "rows": rows, "unit_label": label})
    return kept[:first_index] + tables + kept[first_index:]


FIXES = {
    ("KAU", "ulya_qabul"): fix_kau_ulya_qabul,
    ("KAU", "diploma_bued"): fix_kau_diploma_bued,
    ("KSU", "ulya_baramij"): fix_ksu_ulya_baramij,
    ("KSU", "sihhiya"): fix_ksu_sihhiya,
    ("KSU", "qabul"): fix_ksu_qabul,
    ("KSU", "taqwim"): fix_ksu_taqwim,
    ("KAU", "tahwil_dakhili"): fix_kau_tahwil_dakhili,
    ("KAU", "taqwim"): fix_kau_taqwim,
    ("KAU", "qabul"): fix_kau_qabul,
    ("KAU", "tahwil_khariji"): fix_kau_tahwil_khariji,
}


def apply(university: str, slug: str, blocks: list[dict]) -> list[dict]:
    fix = FIXES.get((university, slug))
    if fix is None:
        return blocks
    return fix(blocks)
