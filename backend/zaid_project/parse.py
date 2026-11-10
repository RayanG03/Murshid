"""المرحلة 1 من خط البناء: يقرأ مصادر الوثائق ويحفظ بنيتها بدل تسطيحها.

المدخلات: data/raw/{ksu,kau}/ (نسخ المصادر) و data/web_sources/ (صفحات رسمية منقولة).
المخرجات لكل وثيقة:
  1. data/{uni}/result_*.txt            نص نظيف للمراجعة البشرية
  2. data/structured/{uni}_{slug}.json   شجرة الكتل (عنوان / فقرة / جدول) للتقطيع

لماذا JSON إضافي: وسم HTML يعرف يقيناً أي سطر عنوان (h1..h4)، أما النص المسطَّح
فيضطر المقطِّع لتخمين البنية، وهذا ما نسب جداول التقديرات خطأً إلى المادة الأخيرة سابقاً.

كل تصحيح يدوي للنص في curated_fixes.py، وتصحيح الجداول في table_fixes.py.

نسخة الوثيقة (version في DOCS) كما يذكرها مصدرها: العام الجامعي أو تاريخ الإصدار أو آخر
تحديث للصفحة، وتغيب حين لا يذكرها المصدر. ملف تقويم KSU لا يذكر عامه، فعامه من عنوان
صفحته الرسمية (source_url): «التقويم الدراسي للعام الجامعي 2026-2027م».

التشغيل:
    python parse.py        ثم python chunk.py
"""

import json
import re
from importlib.util import find_spec

from bs4 import BeautifulSoup, Tag

import config
import curated_fixes
import table_fixes

HTML_PARSER = "lxml" if find_spec("lxml") else "html.parser"

SOURCE_DIRS = {"KSU": config.RAW_DIR / "ksu", "KAU": config.RAW_DIR / "kau"}

KSU_SERVICES_HEADINGS = [
    "البطاقات الجامعية",
    "إجراءات إصدار البطاقة الجامعية - طلاب",
    "إجراءات استخراج بدل مفقود للبطاقة الجامعية",
    "إجراءات استخراج البطاقة الجامعية - طالبات",
    "المكافآت",
    "المكافآت وطريقة صرفها",
    "تسليم المكافاة",
    "مكافآت الطلاب والطالبات ذوي الإعاقة",
    "إجراءات صرف بدل إعاقة",
    "مكافأة التفوق",
    "استحقاق مكافأة التفوق",
    "طلبة الدراسات العليا",
    "الحذف والإضافة",
    "أولاً : شروط الحذف",
    "ثانياً : شروط الإضافة",
    "اجراءات الحذف والاضافة",
    "تأجيل الدراسة للفصل الدراسي",
    "تقييم المحاضرين",
    "التحويل من كلية لأخرى داخل الجامعة",
    "التحويل بين مسارات السنة الأولى المشتركة:",
    "التحويل بين الكليات:",
    "اجراءات معادلة لمحول من داخل الجامعة",
    "الحرمان من دخول الاختبار النهائي",
    "رفع الحرمان",
    "اجراءات رفع الحرمان",
    "تسجيل الطالب الزائر لجامعة أخرى",
    "الفصل من الجامعة",
    "منح الفرصة للطلبة المفصولين من الجامعة لأسباب أكاديمية",
    "الانسحاب من الجامعة",
    "ملاحظات عن إخلاء الطرف:",
    "الطلبة المتوقع تخرجهم :",
    "مراتب الشرف",
]
KAU_GRADUATE_TRANSFER_HEADINGS = [
    "دراسة طالب زائر:",
    "اولاً: طالب مسجل بجامعة أخرى يرغب التسجيل بجامعة الملك عبدالعزيز كطالب زائر لفصل دراسي واحد",
    "ثانياً: طالب مسجل بجامعة الملك عبدالعزيز يرغب التسجيل بجامعة اخرى كطالب زائر",
    "التحويل الى جامعة الملك عبدالعزيز:",
    "اولاً: التحويل من جامعة اخرى الى جامعة الملك عبدالعزيز.",
    "ثانياً: التحويل من جامعة الملك عبدالعزيز الى جامعة اخرى",
    "التحويل من برنامج إلى آخر داخل الجامعة:",
    "الية التحويل :",
    "التحويل من نظام الدراسة بالرسالة الى المشروع البحثي أو العكس",
]
KAU_HOUSING_HEADINGS = [
    "ضوابط الالتحاق بالإسكان الطلابي",
    "شروط القبول في السكن الجامعي المدفوع:",
    "ضوابط استحقاق التقديم في الإسكان الطلابي للطلاب المتزوجين، طلاب المنح الدوليين ، "
    "منح داخلية وخارجية فقط.",
    "ثانياً: ضوابط نقاط المفاضلة بين الطلبة المتقدمين بالإسكان الطلابي والمنطبقة "
    "عليهم معايير القبول:",
    "رسوم السكن وسدادها:",
    "توزيع الغرف السكنية:",
    "أولاً: توزيع الغرف السكنية وفقاً لمعايير معلنة تضعها العمادة بما يحقق الأهداف الآتية :",
    "إخلاء الغرف والمغادرة:",
    "أولاً: يُطلب من الطالبـ/ ـة إخلاء السكن ومغادرته في الحالات التالية:",
    "البقاء في الإسكان الطلابي خلال الإجازات:",
    "حقوق والتزامات الطالبـ/ـة الساكن/ة",
    "المحظورات والمخالفات",
    "إجراءات التقديم والتسكين ومبلغ التأمين:",
    "أولا : إجراءات المتبعة في حال الضبط ( جميع المخالفات في الدليل )",
]

KSU_QABUL_PROGRAMS = dict(
    groups=[
        "الكليات الصحية",
        "الكليات العلمية",
        "كلية إدارة الأعمال",
        "الكليات الإنسانية",
        "كلية الهندسة التطبيقية",
        "كلية علوم الحاسب التطبيقي",
        "كلية إدارة الأعمال التطبيقية",
    ],
    end="برامج درجة الدبلوم (المتوسط -المشارك) بمقابل مالي",
)
KAU_QABUL_PROGRAMS = dict(
    groups=[
        "برامج البكالوريوس في الكليات الصحية",
        "برامج البكالوريوس في الكليات العلمية",
        "برامج البكالوريوس في الكليات الأدبية والإنسانية",
        "برامج البكالوريوس في الاقتصاد والإدارة",
        "برامج القبول المباشر في مرحلة البكالوريوس",
        "برامج القبول المباشر في مرحلة الدبلوم",
    ],
    end="مسارات القبول في الجامعة",
)
KAU_EXTERNAL_TRANSFER_PROGRAMS = dict(
    groups=["برامج البكالوريوس المتاحة"],
    end="الشروط العامة للتحويل الخارجي",
)

DOCS = [
    dict(
        uni="KSU",
        slug="shuun",
        src="اللائحة التنظيمية للشؤون الدراسية KSU.txt",
        out="result_لائحة_الشؤون_الدراسية.txt",
        title="اللائحة التنظيمية للشؤون الدراسية",
        version="٢٤/٧/١٤٤٥هـ الموافق ٥/٢/٢٠٢٤م",
        doc_type="لائحة",
    ),
    dict(
        uni="KSU",
        slug="qabul_dalil",
        src="دليل القبول 2026-2027 KSU.txt",
        out="result_دليل_القبول.txt",
        title="دليل القبول للبرامج الجامعية 2026-2027",
        version="2026 - 2027م",
        doc_type="دليل",
        programs=KSU_QABUL_PROGRAMS,
    ),
    dict(
        uni="KSU",
        slug="sihhiya",
        src="تصنيف الحالة الصحية للطلبة KSU.txt",
        out="result_الحالة_الصحية.txt",
        title="تصنيف الحالة الصحية للطلبة",
        version="٢٠٢٥م",
        doc_type="دليل",
    ),
    dict(
        uni="KSU",
        slug="qabul",
        src="القبول في جامعة الملك سعود KSU.txt",
        out="result_القبول_KSU.txt",
        title="القبول في جامعة الملك سعود 2026-2027",
        version="2026/2027م",
        doc_type="دليل",
        mixed=True,
    ),
    dict(
        uni="KSU",
        slug="taqwim",
        src="التقويم الدراسي KSU.txt",
        out="result_التقويم_الدراسي_KSU.txt",
        title="التقويم الدراسي 2026-2027",
        version="2026-2027م",
        doc_type="تقويم",
        ksu_calendar=True,
        source_url="https://cacsm.ksu.edu.sa/ar/node/3041",
    ),
    dict(
        uni="KSU",
        slug="harakat",
        src="جدول الحركات الأكاديمية KSU.txt",
        out="result_جدول_الحركات_الأكاديمية.txt",
        title="جدول الحركات الأكاديمية 2026-2027",
        version="٢٠٢٦/٢٧م",
        doc_type="تقويم",
    ),
    dict(
        uni="KSU",
        slug="khadamat",
        src="الخدمات الطلابية KSU.txt",
        out="result_الخدمات_الطلابية.txt",
        title="الخدمات والإجراءات الطلابية",
        doc_type="دليل",
        ksu_services=True,
        headings=KSU_SERVICES_HEADINGS,
    ),
    dict(
        uni="KSU",
        slug="suluk",
        src="قواعد السلوك والانضباط الطلابي KSU.txt",
        out="result_قواعد_السلوك_والانضباط_KSU.txt",
        title="قواعد السلوك والانضباط الطلابي",
        version="الطبعة الأولى 2022",
        doc_type="لائحة",
    ),
    dict(
        uni="KSU",
        slug="zaer",
        src="ksu_visitor_in.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_الطالب_الزائر_KSU.txt",
        title="الدراسة كطالب زائر في جامعة الملك سعود",
        version="آخر تحديث للصفحة: يناير 9, 2025",
        doc_type="دليل",
        source_url="https://dar.ksu.edu.sa/ar/visitor",
    ),
    dict(
        uni="KSU",
        slug="tahwil_khariji",
        src="ksu_transfer_in.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_التحويل_الخارجي_KSU.txt",
        title="التحويل من خارج الجامعة إلى جامعة الملك سعود",
        version="آخر تحديث للصفحة: أغسطس 24, 2026",
        doc_type="دليل",
        source_url="https://dar.ksu.edu.sa/ar/node/646",
    ),
    dict(
        uni="KSU",
        slug="tahwil_dakhili",
        src="ksu_internal_transfer.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_التحويل_الداخلي_KSU.txt",
        title="شروط التحويل الداخلي للكليات",
        version="2026/2027م (1448هـ)",
        doc_type="دليل",
        transfer=True,
        max_chars=config.KSU_TRANSFER_MAX_CHARS,
        source_url="https://dar.ksu.edu.sa/ar/internal_transfer_terms",
    ),
    dict(
        uni="KSU",
        slug="iskan",
        src="ksu_housing.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_الإسكان_KSU.txt",
        title="شروط الإقامة بالإسكان الجامعي",
        version="آخر تحديث للصفحة: سبتمبر 21, 2026",
        doc_type="دليل",
        source_url="https://sa.ksu.edu.sa/ar/node/1013",
    ),
    dict(
        uni="KSU",
        slug="ulya_qawaid",
        src="اللائحة المنظمة للدراسات العليا والقواعد التنفيذية KSU.txt",
        out="result_اللائحة_المنظمة_للدراسات_العليا_KSU.txt",
        title="اللائحة المنظمة للدراسات العليا والقواعد التنفيذية بجامعة الملك سعود",
        version="إصدار رقم 4 (28/02/1445هـ)",
        doc_type="لائحة",
        stage="الدراسات العليا",
        source_url=(
            "https://cba.ksu.edu.sa/sites/cba.ksu.edu.sa/files/users/user2343/"
            "%D8%A7%D9%84%D9%82%D9%88%D8%A7%D8%B9%D8%AF%20%D8%A7%D9%84%D8%AA%D9%86%D9%81%D9%8A"
            "%D8%B0%D9%8A%D8%A9%20%D9%84%D9%84%D8%A7%D8%A6%D8%AD%D8%A9%20%D8%A7%D9%84%D8%AF"
            "%D8%B1%D8%A7%D8%B3%D8%A7%D8%AA%20%D8%A7%D9%84%D8%B9%D9%84%D9%8A%D8%A7%20%D8%A7"
            "%D9%84%D8%A5%D8%B5%D8%AF%D8%A7%D8%B1%20%D8%A7%D9%84%D8%B1%D8%A7%D8%A8%D8%B9.pdf"
        ),
    ),
    dict(
        uni="KSU",
        slug="ulya_qabul",
        src="ksu_grad_admission.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_شروط_القبول_الدراسات_العليا_KSU.txt",
        title="شروط القبول في برامج الدراسات العليا بجامعة الملك سعود",
        version="آخر تحديث للصفحة: أكتوبر 23, 2026",
        doc_type="دليل",
        stage="الدراسات العليا",
        source_url="https://sa.ksu.edu.sa/ar/node/4362",
    ),
    dict(
        uni="KSU",
        slug="ulya_baramij",
        src="برامج الدراسات العليا 1448 KSU.txt",
        out="result_برامج_الدراسات_العليا_1448_KSU.txt",
        title="برامج الدراسات العليا الاعتيادية للعام الجامعي 1448هـ",
        version="1448هـ (2026/2027م)",
        doc_type="دليل",
        stage="الدراسات العليا",
        source_url=(
            "https://graduatestudies.ksu.edu.sa/sites/graduatestudies.ksu.edu.sa/files/users/"
            "user168/%D8%A8%D8%B1%D8%A7%D9%85%D8%AC%20%D8%A7%D9%84%D8%AF%D8%B1%D8%A7%D8%B3"
            "%D8%A7%D8%AA%20%D8%A7%D9%84%D8%B9%D9%84%D9%8A%D8%A7%20%D8%A7%D9%84%D9%85%D8%AA"
            "%D8%A7%D8%AD%D8%A9%20%D9%84%D9%84%D8%B9%D8%A7%D9%85%20%D8%A7%D9%84%D8%AC%D8%A7"
            "%D9%85%D8%B9%D9%8A%201448%D9%87_0.pdf"
        ),
    ),
    dict(
        uni="KSU",
        slug="diploma_qabul",
        src="ksu_diploma_admission.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_شروط_القبول_الدبلوم_KSU.txt",
        title="شروط القبول في برامج الدبلوم — كلية الدراسات التطبيقية",
        version="آخر تحديث للصفحة: أكتوبر 18, 2026",
        doc_type="دليل",
        stage="الدبلوم",
        source_url="https://cas.ksu.edu.sa/ar/node/3068",
    ),
    dict(
        uni="KAU",
        slug="dirasa",
        src="لائحة الدراسة والاختبارات للمرحلة KAU.txt",
        out="result_لائحة_الدراسة_KAU.txt",
        title="لائحة الدراسة والاختبارات للمرحلة الجامعية",
        version="1444/1/3هـ، وآخر تعديل 1447/7/3هـ",
        doc_type="لائحة",
    ),
    dict(
        uni="KAU",
        slug="suluk",
        src="لائحة ضبط سلوك الطلاب والطالبات KAU.txt",
        out="result_ضبط_السلوك_KAU.txt",
        title="لائحة ضبط سلوك الطلاب والطالبات",
        version="١٤٤٠/٣/٣ هـ",
        doc_type="لائحة",
    ),
    dict(
        uni="KAU",
        slug="qabul",
        src="دليل القبول KAU.txt",
        out="result_دليل_القبول_KAU.txt",
        title="دليل القبول",
        version="1449/1448هـ",
        doc_type="دليل",
        programs=KAU_QABUL_PROGRAMS,
    ),
    dict(
        uni="KAU",
        slug="tahwil_dakhili",
        src="دليل التحويل الداخلي KAU.txt",
        out="result_دليل_التحويل_KAU.txt",
        title="دليل التحويل الداخلي",
        version="1448/1447 هـ",
        doc_type="دليل",
    ),
    dict(
        uni="KAU",
        slug="tahwil_khariji",
        src="دليل التحويل الخارجي KAU.txt",
        out="result_دليل_التحويل_الخارجي_KAU.txt",
        title="دليل التحويل الخارجي",
        version="1449/1448هـ الموافق 2027/2026",
        doc_type="دليل",
        programs=KAU_EXTERNAL_TRANSFER_PROGRAMS,
    ),
    dict(
        uni="KAU",
        slug="huquq",
        src="دليل حقوق وواجبات الطالب KAU.txt",
        out="result_حقوق_وواجبات_الطالب_KAU.txt",
        title="دليل حقوق وواجبات الطالب",
        version="1447 هـ - 2025م",
        doc_type="دليل",
    ),
    dict(
        uni="KAU",
        slug="taqwim",
        src="التقويم الأكاديمي KAU.txt",
        out="result_التقويم_الأكاديمي_KAU.txt",
        title="التقويم الأكاديمي",
        version="1449/1448هـ - 2027/2026م",
        doc_type="تقويم",
    ),
    dict(
        uni="KAU",
        slug="tahwil_ulya",
        src="التحويل داخل او إلى الجامعة والطالب الزائر KAU.txt",
        out="result_التحويل_والزائر_دراسات_عليا_KAU.txt",
        title="التحويل والطالب الزائر — الدراسات العليا",
        doc_type="دليل",
        stage="دراسات عليا",
        plain=True,
        headings=KAU_GRADUATE_TRANSFER_HEADINGS,
    ),
    dict(
        uni="KAU",
        slug="taqwim_sana",
        src="kau_calendar_year.txt",
        src_dir=config.WEB_SOURCES_DIR,
        out="result_التقويم_السنوي_KAU.txt",
        title="التقويم الأكاديمي للعام الجامعي 1448/1449هـ",
        version="1448/1449هـ (2026/2027م)",
        doc_type="تقويم",
        source_url="https://kau.edu.sa/ar/academic-calendar",
    ),
    dict(
        uni="KAU",
        slug="sakan",
        src="ضوابط الالتحاق بالإسكان الطلابي KAU.txt",
        out="result_ضوابط_الإسكان_KAU.txt",
        title="ضوابط الالتحاق بالإسكان الطلابي",
        doc_type="دليل",
        plain=True,
        headings=KAU_HOUSING_HEADINGS,
    ),
    dict(
        uni="KAU",
        slug="maliya",
        src="الخدمات المالية KAU.txt",
        out="result_الخدمات_المالية_KAU.txt",
        title="الخدمات المالية والمكافآت الطلابية",
        doc_type="دليل",
        plain=True,
    ),
    dict(
        uni="KAU",
        slug="ulya_qawaid",
        src="اللائحة المنظمة للدراسات العليا والقواعد التنفيذية KAU.txt",
        out="result_اللائحة_المنظمة_للدراسات_العليا_KAU.txt",
        title="اللائحة المنظمة للدراسات العليا بالجامعات والقواعد التنفيذية بجامعة الملك عبدالعزيز",
        version="محرم 1444هـ / أغسطس 2022م",
        doc_type="لائحة",
        stage="الدراسات العليا",
        source_url=(
            "https://kau.edu.sa/ar/internal-regulations/regulation/9c7f807c-3fe2-49b9-96a1-1b19f98854f5"
        ),
    ),
    dict(
        uni="KAU",
        slug="ulya_qabul",
        src="دليل القبول للدراسات العليا 1448 KAU.txt",
        out="result_دليل_القبول_للدراسات_العليا_KAU.txt",
        title="دليل القبول لبرامج الدراسات العليا الأكاديمية 1448هـ",
        version="1448هـ (مارس 2026)",
        doc_type="دليل",
        stage="الدراسات العليا",
        source_url=(
            "https://www.kau.edu.sa/web/static?path=uploads%2F2026%2FMar%2Fmediacenter%2F"
            "%D8%B9%D9%85%D8%A7%D8%AF%D8%A9_%D8%A7%D9%84%D8%AF%D8%B1%D8%A7%D8%B3%D8%A7%D8%AA_"
            "%D8%A7%D9%84%D8%B9%D9%84%D9%8A%D8%A7_-_%D8%A8%D8%B1%D8%A7%D9%85%D8%AC_%D8%A7%D9%84"
            "%D8%AF%D9%83%D8%AA%D9%88%D8%B1%D8%A9%2F4UGNDZDebNpB7YSG.pdf"
        ),
    ),
    dict(
        uni="KAU",
        slug="diploma_bued",
        src="دليل القبول لدبلومات التعليم عن بعد KAU.txt",
        out="result_دليل_القبول_دبلومات_التعليم_عن_بعد_KAU.txt",
        title="دليل القبول لبرامج دبلومات التعليم عن بعد",
        version="1449/1448هـ، الإصدار الثاني (أغسطس 2026)",
        doc_type="دليل",
        stage="الدبلوم",
        source_url="https://www.kau.edu.sa/web/static/ade44658-f908-472b-ad11-33456093dc5f",
    ),
]

IMAGE_DESCRIPTION_STARTS = [
    "شريط",
    "شعار",
    "رسم",
    "لوحة",
    "خلفية",
    "زخرف",
    "أيقونة",
    "إطار",
    "بنر",
    "يتضمن شجرة",
    "نمط متكرر",
    "رمز استجابة سريعة يتوسط",
]
KSU_LOGO_TEXT = "logo of king saud university"
OCR_BROKEN = re.compile(r"(اء){3,}")
HTML_TABLE = re.compile(r"(<table\b[^>]*>.*?</table>)", re.DOTALL | re.IGNORECASE)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

TABLE_OF_CONTENTS = re.compile(
    r"^(فهرس\s+الم.تويات|قائمة\s+الم.تويات|الم.تويات|الفهرس|محتويات\s+الدليل)$"
)
BAB = re.compile(r"^(?:الباب|الفصل)\s+([^\s:：\-–—]+(?:\s+عشر)?)\s*[:：\-–—]?\s*(.{0,80})$")
QISM = re.compile(r"^القسم\s+([^\s:：\-–—]+(?:\s+عشر)?)\s*[:：\-–—]?\s*(.{0,80})$")
APPENDIX_NUMBER = re.compile(r"^(?:الملحق|ملحق)\s*\(\s*([٠-٩\d]+)\s*\)\s*(.{0,60})$")
APPENDIX_WORD = re.compile(r"^(?:الملحق|ملحق)\s+([ء-ي]+)\s*(.{0,60})$")
EXEC_RULE = re.compile(r"^القاعدة\s+التنفيذية\s+للمادة\s+([؀-ۿ\s]{1,45})$")
EXEC_RULES_GROUP = re.compile(r"^(?:القواعد|القاعدة|قاعدة)\s+(?:التنفيذية|الإجرائية|تنفيذية)\b")
ARTICLE_WORD = re.compile(r"^المادة\s+([؀-ۿ\s]{1,45}?)\s*[:：]?$")
ARTICLE_NUMBER = re.compile(r"^مادة\s*\(\s*([٠-٩\d]+)\s*\)$")
ORDINAL_WORDS = (
    r"(?:[أا]ولا|[أا]ولاً|[أا]ولًا|ثانيا|ثانياً|ثانيًا|ثالثا|ثالثاً|ثالثًا|رابعا|"
    r"رابعاً|رابعًا|خامسا|خامساً|خامسًا|سادسا|سادساً|سابعا|سابعاً|ثامنا|ثامناً|"
    r"تاسعا|تاسعاً|عاشرا|عاشراً)"
)
ORDINAL_MARK = re.compile(rf"^{ORDINAL_WORDS}\s*[:ـ]")

ONES = {
    "الأولى": 1,
    "الأول": 1,
    "الحادية": 1,
    "الحادي": 1,
    "الواحدة": 1,
    "الواحد": 1,
    "الثانية": 2,
    "الثاني": 2,
    "الثالثة": 3,
    "الثالث": 3,
    "الرابعة": 4,
    "الرابع": 4,
    "الخامسة": 5,
    "الخامس": 5,
    "السادسة": 6,
    "السادس": 6,
    "السابعة": 7,
    "السابع": 7,
    "الثامنة": 8,
    "الثامن": 8,
    "التاسعة": 9,
    "التاسع": 9,
    "العاشرة": 10,
    "العاشر": 10,
}
TENS = {
    "عشرة": 10,
    "عشر": 10,
    "العشرون": 20,
    "العشرين": 20,
    "الثلاثون": 30,
    "الثلاثين": 30,
    "الأربعون": 40,
    "الأربعين": 40,
    "الخمسون": 50,
    "الخمسين": 50,
    "الستون": 60,
    "الستين": 60,
    "السبعون": 70,
    "السبعين": 70,
    "الثمانون": 80,
    "الثمانين": 80,
    "التسعون": 90,
    "التسعين": 90,
}

TRANSFER_COLLEGES = 21
BANNER_LINE = re.compile(r"^=+$")
TRANSFER_TABLE_HEADER = re.compile(r"^(\S.*? / .*?):$")
GPA_VALUE = re.compile(r"^\d\.\d{2}$")


def ordinal_to_int(phrase: str) -> int | None:
    """«الرابعة عشرة» ← 14، «الحادية والعشرون» ← 21، وإلا None."""
    total = 0
    has_number = False
    for word in phrase.split():
        word = word.strip("،:.()")
        if word.startswith("و") and (word[1:] in ONES or word[1:] in TENS):
            word = word[1:]
        if word in ONES:
            total += ONES[word]
            has_number = True
        elif word in TENS:
            total += TENS[word]
            has_number = True
        else:
            return None
    if has_number:
        return total
    return None


def classify_heading(text: str) -> tuple[str, int | None, str]:
    """نوع العنوان ورقمه: (kind, number, text). kind = toc | appendix | exec | article |
    qism | bab | sub (عنوان فرعي عادي)."""
    text = re.sub(r"\s+", " ", text).strip()
    if TABLE_OF_CONTENTS.match(text):
        return "toc", None, text

    match = APPENDIX_NUMBER.match(text)
    if match:
        return "appendix", int(match.group(1).translate(ARABIC_DIGITS)), text
    match = APPENDIX_WORD.match(text)
    if match and ordinal_to_int(match.group(1)) is not None:
        return "appendix", ordinal_to_int(match.group(1)), text

    match = EXEC_RULE.match(text)
    if match and ordinal_to_int(match.group(1)) is not None:
        return "exec", ordinal_to_int(match.group(1)), text
    if EXEC_RULES_GROUP.match(text):
        return "exec", None, text

    match = ARTICLE_WORD.match(text)
    if match and ordinal_to_int(match.group(1)) is not None:
        return "article", ordinal_to_int(match.group(1)), text
    match = ARTICLE_NUMBER.match(text)
    if match:
        return "article", int(match.group(1).translate(ARABIC_DIGITS)), text

    if QISM.match(text):
        return "qism", None, text
    match = BAB.match(text)
    if match:
        if text.startswith("الفصل ") and not match.group(2):
            return "sub", None, text
        return "bab", None, text
    return "sub", None, text


def heading_block(
    kind: str, text: str, number: int | None = None, level: int | None = None
) -> dict:
    block = {"t": "h", "kind": kind}
    if level is not None:
        block["level"] = level
    block["text"] = text
    if number:
        block["no"] = number
    return block


def is_image_description(text: str) -> bool:
    stripped = text.strip()
    for start in IMAGE_DESCRIPTION_STARTS:
        if stripped.startswith(start):
            return True
    return False


def is_broken(text: str) -> bool:
    return bool(OCR_BROKEN.search(text))


def own_text(tag: Tag) -> str:
    """نص العنصر دون القوائم المتداخلة بداخله (وإلا تكرر النص الفرعي مرتين)."""
    parts = []
    for child in tag.children:
        if getattr(child, "name", None) in ("ol", "ul", "li", "p", "table"):
            continue
        if hasattr(child, "get_text"):
            text = child.get_text(" ", strip=True)
        else:
            text = str(child).strip()
        if text:
            parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def read_table(table_tag: Tag) -> tuple[list[str] | None, list[list[str]]]:
    """يستخرج الجدول (الرؤوس، الصفوف) مع توسيع الخلايا الممتدة على أكثر من صف."""
    header = None
    rows = []
    spanning = {}
    for row_tag in table_tag.find_all("tr"):
        cells = list(row_tag.find_all(["td", "th"]))
        if not cells:
            continue
        values = {}
        still_spanning = {}
        for column, (rows_left, text) in spanning.items():
            values[column] = text
            if rows_left - 1 > 0:
                still_spanning[column] = (rows_left - 1, text)
        spanning = still_spanning

        column = 0
        for cell in cells:
            while column in values:
                column += 1
            values[column] = cell.get_text(" ", strip=True)
            row_span = int(cell.get("rowspan", 1))
            if row_span > 1:
                spanning[column] = (row_span - 1, values[column])
            column += 1

        row = []
        for index in range(max(values) + 1):
            value = values.get(index, "")
            row.append("" if is_broken(value) else value)
        if not any(value.strip() for value in row):
            continue
        is_header_row = all(cell.name == "th" for cell in cells)
        if header is None and is_header_row:
            header = row
        else:
            rows.append(row)
    return header, rows


def table_block(table_tag: Tag) -> dict | None:
    header, rows = read_table(table_tag)
    if not rows and not header:
        return None
    return {"t": "table", "header": header, "rows": rows}


def drop_image_descriptions(soup: BeautifulSoup) -> None:
    """يحذف ما كتبه Chandra داخل الصور: <div class="img-description">.

    فيها وصف بالإنجليزية للصور الزخرفية («Illustration: 3D rendering of an office desk…»)،
    ولقطات بوابة القبول ببيانات مثال، وجداول لوحات وهمية. والنص العربي الحقيقي فيها (صفحتا
    الدبلوم في اللغة الصينية والتحول الرقمي، وعنوانا «الرسوم الدراسية» و«الأسئلة الشائعة»)
    كتبه Chandra نصاً عادياً بجانبها أيضاً، فإبقاؤه يكرره.
    """
    for box in soup.select("div.img-description"):
        box.decompose()


def blocks_from_soup(soup: BeautifulSoup, mark_lists: bool = False) -> list[dict]:
    """يحوّل شجرة HTML إلى كتل مصنّفة.

    mark_lists: كل بند <li> يحمل "list" برقم قائمته، كي يميّز program_tables قائمة
    البرامج من الفقرات حولها. يُحذف الحقل بعد ذلك.
    """
    blocks = []
    list_numbers = {}
    is_in_toc = False
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if tag.find_parent("table") is not None:
            continue
        if tag.name in ("h1", "h2", "h3", "h4"):
            text = tag.get_text(" ", strip=True)
            if not text or is_broken(text):
                continue
            kind, number, text = classify_heading(text)
            if kind == "toc":
                is_in_toc = True
                continue
            is_in_toc = False
            blocks.append(heading_block(kind, text, number))
            continue
        if is_in_toc:
            continue
        if tag.name == "table":
            block = table_block(tag)
            if block:
                blocks.append(block)
            continue
        if tag.find_parent(["h1", "h2", "h3", "h4"]):
            continue
        text = own_text(tag)
        if not text or is_image_description(text) or is_broken(text):
            continue
        block = {"t": "p", "text": text}
        if mark_lists and tag.name == "li":
            parent = tag.find_parent(["ul", "ol"])
            block["list"] = list_numbers.setdefault(id(parent), len(list_numbers))
        blocks.append(block)
    return blocks


SPLIT_TITLE_TAIL = re.compile(r"^(?:في\s|ل)")


def join_split_titles(blocks: list[dict]) -> list[dict]:
    """عنوان قسّمه الـ OCR إلى عناوين متتالية بلا محتوى بينها يعود عنواناً واحداً.

    «القبول المباشر» + «لبرامج البكالوريوس» ← «القبول المباشر لبرامج البكالوريوس».
    عنوان تالٍ هو ذيل العنوان قبله («برامج القبول المباشر في مرحلة البكالوريوس» ثم
    «مرحلة البكالوريوس») يُحذف لأنه تكرار. وعنوان تالٍ ينتهي بنقطتين («يشترط لقبول
    الطالب المستجد في الجامعة ما يلي:») جملة تمهيد لا عنوان، فيصير فقرة.
    """
    result = []
    for block in blocks:
        previous = result[-1] if result else None
        if block["t"] != "h" or previous is None or previous["t"] != "h":
            result.append(block)
            continue
        if previous["text"].endswith(block["text"]):
            continue
        if SPLIT_TITLE_TAIL.match(block["text"]):
            previous["text"] = f"{previous['text']} {block['text']}"
            continue
        if block["text"].endswith(":"):
            result.append({"t": "p", "text": block["text"]})
            continue
        result.append(block)
    return result


def college_name(name: str) -> str:
    """«الطب» ← «كلية الطب»: أدلة KSU تكتب الكلية باسمها تحت «الكليات الصحية»."""
    if name.startswith(("كلية ", "الكلية ")):
        return name
    return f"كلية {name}"


def program_table(group: str, college: str, programs: list[str], label: str) -> dict:
    """جدول (المجموعة | الكلية | البرنامج)؛ العمود الفارغ أو المكرر («كلية إدارة الأعمال»
    مجموعةً وكليةً) لا يُكتب."""
    if group == college:
        group = ""
    header = []
    fixed = []
    for name, value in (("المجموعة", group), ("الكلية", college)):
        if value:
            header.append(name)
            fixed.append(value)
    rows = [fixed + [program] for program in programs]
    return {"t": "table", "header": header + ["البرنامج"], "rows": rows, "unit_label": label}


def program_tables(blocks: list[dict], groups: list[str], end: str) -> list[dict]:
    """قوائم «الكلية ← برامجها» في الأدلة تصير جدولاً لكل كلية، فتصير كل كلية مقطعاً.

    في الكتيّب كل كلية عنوان تحته قائمة برامج قصيرة، فكانت كل كلية وحدة صغيرة يدمجها
    chunk.py مع الكلية بعدها تحت اسم الأولى (f30: «كلية السياحة» داخل مقطع «كلية الحقوق»).
    الجدول (المجموعة | الكلية | البرنامج) بعنوان وحدة (unit_label) لا يُدمج أبداً، كما في
    دليل التحويل الداخلي KAU. الأسماء تبقى كما في المصدر، ومعها (طلاب) و(طالبات).

    groups: عناوين المجموعات («الكليات الصحية»)، تبقى عناوين وتدخل عمود المجموعة.
    end: أول عنوان بعد القوائم. قائمة البرامج <ul> تلي عنوان الكلية مباشرة؛ أما القائمة
    التي تسبقها جملة تنتهي بنقطتين («معايير القبول:») فهي بنود تلك الجملة.
    مجموعة بلا كليات تحتها («كلية الهندسة التطبيقية») تصير هي المقطع: جدولها في موضع
    عنوانها، وفقراتها (المسارات ومعايير القبول) بعده.

    المجموعة التي تحتها كليتان أو أكثر يُكتب في مقطعها سطر «الكليات: …» بأسماء كلياتها
    قبل أول كلية، فسؤال القائمة («وش الكليات العلمية؟») يجد الأسماء كلها في مقطع واحد،
    والمقاطع الأربعة لا تتسع لتسع كليات.
    """
    result = []
    group = None
    group_position = None
    college = None
    tables = 0
    colleges_by_group = {}
    first_college_position = {}
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block["t"] == "h" and block["text"] == end:
            group = None
        if group is None and not (block["t"] == "h" and block["text"] in groups):
            result.append(block)
            index += 1
            continue
        if block["t"] == "h":
            if block["text"] in groups:
                group = block["text"]
                group_position = len(result)
                result.append(block)
            else:
                college = college_name(block["text"])
            index += 1
            continue
        if "list" not in block:
            if college is not None:
                raise ValueError(f"الكلية «{college}» بلا قائمة برامج بعد عنوانها")
            result.append(block)
            index += 1
            continue
        previous = blocks[index - 1]
        is_lead_list = previous["t"] == "p" and "list" not in previous and previous["text"].endswith(":")
        items = []
        while index < len(blocks) and blocks[index].get("list") == block["list"]:
            items.append(blocks[index])
            index += 1
        if is_lead_list:
            result += items
            continue
        programs = [item["text"] for item in items]
        if college is not None:
            first_college_position.setdefault(group_position, len(result))
            colleges_by_group.setdefault(group_position, []).append(college)
            result.append(program_table(group, college, programs, college))
            college = None
        elif group.startswith("كلية "):
            result[group_position] = program_table("", group, programs, group)
        else:
            result[group_position] = program_table(group, "", programs, group)
        tables += 1
    if tables == 0:
        raise ValueError(f"لا قوائم برامج تحت {groups[0]} — تغيّرت بنية المصدر")
    for position in sorted(first_college_position, reverse=True):
        colleges = colleges_by_group[position]
        if len(colleges) >= 2:
            card = {"t": "p", "text": "الكليات: " + "، ".join(colleges) + "."}
            result.insert(first_college_position[position], card)
    return result


def blocks_from_mixed(raw: str) -> list[dict]:
    """ملف نصي عادي تتخلله جداول HTML (بلا وسم جذر)."""
    blocks = []
    for part in HTML_TABLE.split(raw):
        if part.lstrip().lower().startswith("<table"):
            block = table_block(BeautifulSoup(part, HTML_PARSER).find("table"))
            if block:
                blocks.append(block)
            continue
        for line in part.splitlines():
            line = line.strip()
            if not line or is_broken(line):
                continue
            if ORDINAL_MARK.match(line) or (len(line) <= 120 and line.endswith(":")):
                kind, number, text = classify_heading(line)
                blocks.append(heading_block(kind, text, number))
            else:
                blocks.append({"t": "p", "text": line})
    return blocks


def count_blank_lines_before(lines: list[str], index: int) -> int:
    blank_lines = 0
    previous = index - 1
    while previous >= 0 and not lines[previous].strip():
        blank_lines += 1
        previous -= 1
    return blank_lines


def blocks_from_text(raw: str, headings: list[str] | tuple = ()) -> list[dict]:
    """ملف نصي خالص بلا HTML: البنية مستنتجة من الأسطر الفارغة.

    عنوان رئيس (level 1): يسبقه سطران فارغان أو أكثر، أو يبدأ الملف، أو يبدأ بترتيب
    (أولاً/ثانياً). وإن مرّرت الوثيقة قائمة عناوينها (headings) فهي وحدها العناوين الرئيسة.
    عنوان فرعي (level 2): يسبقه سطر فارغ واحد، قصير ولا ينتهي بنقطة.
    الرئيس يبدأ وحدة جديدة، والفرعي يبقى داخلها فلا تبتعد «شروط الحصول» عن الخدمة.
    """
    lines = raw.split("\n")
    blocks = []
    has_seen_content = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or is_broken(line):
            continue
        blank_lines = count_blank_lines_before(lines, index)
        ends_sentence = line.endswith((".", "،", "؛"))
        if headings:
            is_top = line in headings
        else:
            is_top = not has_seen_content or blank_lines >= 2 or bool(ORDINAL_MARK.match(line))
        is_sub = blank_lines == 1 and len(line) <= 100 and not ends_sentence
        has_seen_content = True

        if is_top and (len(line) <= 140 or line.endswith(":")) and headings:
            text = re.sub(r"\s+", " ", line.lstrip("* ")).strip()
            blocks.append(heading_block("sub", text, level=1))
        elif is_top and (len(line) <= 140 or line.endswith(":")):
            kind, number, text = classify_heading(line.lstrip("* ").strip())
            blocks.append(heading_block(kind, text, number, level=1))
        elif is_sub:
            blocks.append(heading_block("sub", line.lstrip("* ").strip(), level=2))
        else:
            blocks.append({"t": "p", "text": line})
    return blocks


def transfer_table_row(line: str) -> list[str]:
    """صف «علوم الحاسب | 4.00 | —»: «—» خلية فارغة."""
    row = []
    for cell in line.split(" | "):
        row.append("" if cell.strip() == "—" else cell.strip())
    return row


def read_transfer_lines(raw: str) -> tuple[list[dict], list[dict]]:
    """يقرأ صفحة التحويل سطراً سطراً: (كتل الكليات، فقرات المقدمة قبل أول مجموعة)."""
    blocks = []
    preamble = []
    table = None
    is_in_banner = False
    is_after_blank = True
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if BANNER_LINE.match(line):
            is_in_banner = not is_in_banner
            continue
        if not line:
            is_after_blank = True
            continue
        if is_in_banner:
            blocks.append(heading_block("bab", line))
            if line.startswith("كلية "):
                blocks.append(heading_block("qism", line))
            continue
        is_new_paragraph = is_after_blank
        is_after_blank = False
        if table is not None and " | " in line:
            table["rows"].append(transfer_table_row(line))
            continue
        table = None
        header_match = TRANSFER_TABLE_HEADER.match(line)
        if header_match:
            header = [name.strip() for name in header_match.group(1).split(" / ")]
            table = {"t": "table", "header": header, "rows": []}
            blocks.append(table)
        elif not blocks:
            preamble.append({"t": "p", "text": line})
        elif is_new_paragraph and line.startswith("كلية "):
            blocks.append(heading_block("qism", line))
        else:
            blocks.append({"t": "p", "text": line})
    return blocks, preamble


def name_gpa_columns(blocks: list[dict]) -> None:
    """العمود الذي كله معدلات يُسمّى «الحد الأدنى للمعدل التراكمي (…)» كي لا يُقرأ
    «4.00 | 3.50» رقمين بلا معنى."""
    for table in blocks:
        if table["t"] != "table":
            continue
        for column in range(1, len(table["header"])):
            values = []
            for row in table["rows"]:
                if column < len(row) and row[column]:
                    values.append(row[column])
            if values and all(GPA_VALUE.match(value) for value in values):
                table["header"][column] = f"الحد الأدنى للمعدل التراكمي ({table['header'][column]})"


def blocks_from_ksu_transfer(raw: str) -> list[dict]:
    """صفحة شروط التحويل الداخلي KSU (نص منقول بلا HTML): مجموعة ← كلية ← شروطها.

    المجموعة («الكليات الصحية») بين سطرين «====» فتصير باباً، والسطر الذي يبدأ بـ«كلية»
    بعد سطر فارغ قسماً، فيقطع chunk.py عند كل كلية ويضع مجموعتها في الترويسة.
    الجدول سطر «البرنامج / من خارج الكلية / من داخل الكلية:» تليه صفوف « | »،
    و«—» خلية فارغة.

    المقدمة (وفيها حكم الطلاب ذوي الإعاقة) قسم أول بعنوان «مقدمة الشروط»، وقسما الطب
    وطب الأسنان (curated_fixes.py) أول مجموعة «الكليات الصحية».
    """
    blocks, preamble = read_transfer_lines(raw)
    college_count = 0
    group_index = None
    for index, block in enumerate(blocks):
        if block["t"] == "h" and block["kind"] == "qism":
            college_count += 1
        if block["t"] == "h" and block["kind"] == "bab":
            if block["text"] == curated_fixes.NO_TRANSFER_GROUP:
                group_index = index
    if college_count != TRANSFER_COLLEGES or group_index is None:
        raise ValueError(
            f"صفحة التحويل الداخلي: {college_count} كلية بدل {TRANSFER_COLLEGES}"
            " أو بلا «الكليات الصحية» — راجع "
            "data/web_sources/ksu_internal_transfer.txt"
        )
    name_gpa_columns(blocks)
    no_transfer = curated_fixes.no_transfer_college_blocks()
    blocks = blocks[: group_index + 1] + no_transfer + blocks[group_index + 1 :]
    introduction = [heading_block("bab", "مقدمة الشروط"), heading_block("qism", "مقدمة الشروط")]
    return introduction + preamble + blocks


def ksu_services_blocks(service_lines: list[str], headings: list[str]) -> list[dict]:
    """صفحة الخدمات: مقاطع HTML (<article>) يتخللها نص خالص."""
    blocks = []
    plain_lines = []
    for line in service_lines:
        if line.lstrip().lower().startswith("<article"):
            blocks += blocks_from_text("".join(plain_lines), headings)
            plain_lines = []
            blocks += blocks_from_soup(BeautifulSoup(line, HTML_PARSER))
        else:
            plain_lines.append(line)
    blocks += blocks_from_text("".join(plain_lines), headings)
    blocks = curated_fixes.remove_stale_sections(blocks)
    for block in blocks:
        if block["t"] == "p":
            restored = curated_fixes.restore_riyal(block["text"])
            if restored != block["text"]:
                block["text"] = restored
                block["curated"] = True
    return blocks


def ksu_calendar_blocks(raw: str) -> list[dict]:
    """تقويم KSU (صفحة HTML): تواريخه مصححة يدوياً في curated_fixes، ووصف الشعار يُحذف."""
    blocks = blocks_from_soup(BeautifulSoup(raw, HTML_PARSER))
    blocks = curated_fixes.fix_ksu_calendar(blocks)
    kept = []
    for block in blocks:
        if block.get("text", "").lower() != KSU_LOGO_TEXT:
            kept.append(block)
    return kept


def drop_repeated_headings(blocks: list[dict]) -> list[dict]:
    """يحذف ترويسة الصفحة المتكررة (عنوان الوثيقة يتكرر أعلى كل صفحة)."""
    page_titles = set()
    for block in blocks[:4]:
        if block["t"] == "h":
            page_titles.add(block["text"])
    seen_titles = set()
    kept = []
    for block in blocks:
        if block["t"] == "h" and block["text"] in page_titles:
            if block["text"] in seen_titles:
                continue
            seen_titles.add(block["text"])
        if kept and block == kept[-1]:
            continue
        kept.append(block)
    return kept


def render_text(blocks: list[dict]) -> str:
    lines = []
    for block in blocks:
        if block["t"] == "h":
            lines.append(f"\n{block['text']}\n")
        elif block["t"] == "p":
            lines.append(block["text"])
        else:
            if block.get("unit_label"):
                lines.append(f"\n{block['unit_label']}\n")
            if block["header"]:
                lines.append(" | ".join(block["header"]))
            for row in block["rows"]:
                lines.append(" | ".join(row))
    text = re.sub(r"[ \t]+", " ", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
SPACE_RUN = re.compile(r"[ 	 ]{2,}")


def clean_spaces(blocks: list[dict]) -> None:
    """يحذف الحروف المخفية (‌ قبل «أ-» في دليل الدراسات العليا) والمسافات المتتالية
    التي تركها الـ OCR بعد علامات القوائم («-          الفئة الأولى»)."""
    for block in blocks:
        if block["t"] == "table":
            rows = [block.get("header") or []] + block["rows"]
            for row in rows:
                for index, cell in enumerate(row):
                    row[index] = SPACE_RUN.sub(" ", INVISIBLE.sub("", cell))
        else:
            block["text"] = SPACE_RUN.sub(" ", INVISIBLE.sub("", block["text"])).strip()


def read_blocks(doc: dict) -> list[dict]:
    """يقرأ مصدر الوثيقة بالطريقة التي تناسبه ويطبّق التصحيحات اليدوية."""
    source_dir = doc.get("src_dir", SOURCE_DIRS[doc["uni"]])
    raw = (source_dir / doc["src"]).read_text(encoding="utf-8")
    raw = curated_fixes.apply_text_fixes(doc["src"], raw)

    if doc.get("ksu_calendar"):
        blocks = ksu_calendar_blocks(raw)
    elif doc.get("ksu_services"):
        blocks = ksu_services_blocks(raw.splitlines(keepends=True), doc["headings"])
    elif doc.get("transfer"):
        blocks = blocks_from_ksu_transfer(raw)
    elif doc.get("plain"):
        blocks = blocks_from_text(raw, doc.get("headings", ()))
    elif doc.get("mixed"):
        blocks = blocks_from_mixed(raw)
    else:
        if not raw.lstrip().startswith("<"):
            raw = "<" + raw.lstrip()
        soup = BeautifulSoup(raw, HTML_PARSER)
        drop_image_descriptions(soup)
        blocks = blocks_from_soup(soup, "programs" in doc)

    if not doc.get("transfer"):
        blocks = drop_repeated_headings(blocks)
    if "programs" in doc:
        blocks = join_split_titles(blocks)
        blocks = program_tables(blocks, **doc["programs"])
        for block in blocks:
            block.pop("list", None)
    blocks = table_fixes.apply(doc["uni"], doc["slug"], blocks)
    blocks = curated_fixes.remove_screenshots(doc["src"], blocks)
    clean_spaces(blocks)
    curated_fixes.mark_text_fixes(doc["src"], blocks)
    return curated_fixes.manual_intro_blocks(doc["src"]) + blocks


def write_outputs(doc: dict, blocks: list[dict]) -> str:
    text = render_text(blocks)
    text_path = config.DATA_DIR / doc["uni"].lower() / doc["out"]
    text_path.write_text(text, encoding="utf-8")

    payload = {
        "uni": doc["uni"],
        "slug": doc["slug"],
        "title": doc["title"],
        "doc_type": doc["doc_type"],
        "stage": doc.get("stage", config.UNDERGRADUATE_STAGE),
        "version": doc.get("version"),
        "source_file": doc["out"],
        "blocks": blocks,
        "indexable": doc.get("indexable", True),
    }
    if "max_chars" in doc:
        payload["max_chars"] = doc["max_chars"]
    if "source_url" in doc:
        payload["source_url"] = doc["source_url"]
    json_path = config.STRUCTURED_DIR / f"{doc['uni'].lower()}_{doc['slug']}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return text


def count_kinds(blocks: list[dict]) -> dict:
    counts = {}
    for block in blocks:
        kind = block["kind"] if block["t"] == "h" else block["t"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def main() -> None:
    config.STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"{'الوثيقة':<44}{'حرف':>8}{'باب':>5}{'قسم':>5}{'مادة':>6}{'قاعدة':>7}"
        f"{'ملحق':>6}{'جدول':>6}"
    )
    print("-" * 88)
    for doc in DOCS:
        blocks = read_blocks(doc)
        text = write_outputs(doc, blocks)
        counts = count_kinds(blocks)
        print(
            f"{doc['uni'] + ' ' + doc['title'][:39]:<44}{len(text):>8,}"
            f"{counts.get('bab', 0):>5}{counts.get('qism', 0):>5}{counts.get('article', 0):>6}"
            f"{counts.get('exec', 0):>7}{counts.get('appendix', 0):>6}"
            f"{counts.get('table', 0):>6}"
        )
    print(f"\nالبنية في: {config.STRUCTURED_DIR}")


if __name__ == "__main__":
    main()
