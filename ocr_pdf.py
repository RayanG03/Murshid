"""يحوّل ملفات PDF في data/source_pdfs إلى HTML بنموذج Chandra عبر واجهة Datalab.

لماذا OCR لا نص الـ PDF: طبقة النص في هذه الملفات عربيتها مكسورة («األكاديمية»
بدل «الأكاديمية»، و«لربامج» بدل «لبرامج»)، والبحث لا يجد الكلمة المكسورة. صورة الصفحة
يقرؤها Chandra صحيحة، ويخرجها HTML (عناوين وفقرات وجداول) بالصيغة نفسها التي يقرؤها
parse.py من ملفات data/raw الأخرى.

المدخل: data/source_pdfs/{ksu,kau}/<اسم>.pdf
المخرج: data/raw/{ksu,kau}/<العنوان> <KSU|KAU>.txt  وفيه <article dir="rtl"> … </article>
        و data/source_pdfs/ocr/<اسم>.json  الاستجابة كاملة (للمراجعة والتكلفة)

المفتاح في datalab_key.txt (خارج git) أو في متغير البيئة DATALAB_API_KEY.
الوضع accurate (Chandra)؛ تكلفته 10 دولارات لكل 1000 صفحة.

التشغيل:
    python ocr_pdf.py              يحوّل ما لم يُحوَّل بعد
    python ocr_pdf.py --force      يعيد الكل
"""

import argparse
import json
import os
import time

import httpx

import config

CONVERT_URL = "https://www.datalab.to/api/v1/convert"
KEY_FILE = config.ROOT / "datalab_key.txt"
PDF_DIR = config.DATA_DIR / "source_pdfs"
OCR_DIR = PDF_DIR / "ocr"
POLL_SECONDS = 5
POLL_LIMIT = 360

# اسم الملف ← اسم ملف المصدر في data/raw (ينتهي بـ KSU أو KAU كباقي الملفات)
PDFS = {
    "ksu/grad_qawaid_isdar4.pdf": "اللائحة المنظمة للدراسات العليا والقواعد التنفيذية KSU.txt",
    "ksu/grad_programs_1448.pdf": "برامج الدراسات العليا 1448 KSU.txt",
    "kau/grad_qawaid.pdf": "اللائحة المنظمة للدراسات العليا والقواعد التنفيذية KAU.txt",
    "kau/grad_dalil_1448.pdf": "دليل القبول للدراسات العليا 1448 KAU.txt",
    "kau/diploma_online_1448.pdf": "دليل القبول لدبلومات التعليم عن بعد KAU.txt",
}


def load_key() -> str:
    key = os.environ.get("DATALAB_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("مفتاح Datalab غير موجود — ضعه في datalab_key.txt أو في DATALAB_API_KEY")
    return key


def convert(pdf_path, key: str) -> dict:
    headers = {"X-API-Key": key}
    with open(pdf_path, "rb") as handle:
        response = httpx.post(
            CONVERT_URL,
            headers=headers,
            files={"file": (pdf_path.name, handle, "application/pdf")},
            data={"output_format": "html", "mode": "accurate", "disable_image_extraction": "true"},
            timeout=300,
        )
    response.raise_for_status()
    check_url = response.json()["request_check_url"]
    for _ in range(POLL_LIMIT):
        time.sleep(POLL_SECONDS)
        result = httpx.get(check_url, headers=headers, timeout=60).json()
        if result.get("status") == "complete":
            if not result.get("success"):
                raise RuntimeError(f"{pdf_path.name}: {result.get('error')}")
            return result
    raise TimeoutError(f"{pdf_path.name}: لم يكتمل التحويل")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    key = load_key()
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    for name, raw_name in PDFS.items():
        uni = name.split("/")[0]
        raw_path = config.RAW_DIR / uni / raw_name
        json_path = OCR_DIR / (name.replace("/", "_").removesuffix(".pdf") + ".json")
        if raw_path.exists() and not args.force:
            print(f"موجود: {raw_path.name}")
            continue
        result = convert(PDF_DIR / name, key)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        raw_path.write_text(
            f'<article dir="rtl" lang="ar">\n{result["html"]}\n</article>\n', encoding="utf-8"
        )
        print(
            f"{raw_path.name}: {result.get('page_count')} صفحة، "
            f"الجودة {result.get('parse_quality_score')}، التكلفة {result.get('cost_breakdown')}"
        )


if __name__ == "__main__":
    main()
