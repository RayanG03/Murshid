"""فحص النصوص الناتجة من parse.py قبل التقطيع: OCR مكسور، أو HTML متبقٍّ، أو فهرس.

المدخلات: data/ksu/result_*.txt و data/kau/result_*.txt. لا يغيّر أي ملف.

التشغيل:
    python check_all.py
"""

import re
import sys

import config

FOLDERS = [("الملك سعود", config.DATA_DIR / "ksu"), ("الملك عبدالعزيز", config.DATA_DIR / "kau")]

OCR_BROKEN = re.compile(r"(اء){3,}")
HTML_LEFTOVER = re.compile(r"<[^>]+>")
TABLE_OF_CONTENTS = re.compile(
    r"^\s*(فهرس\s+الم.تويات|قائمة\s+الم.تويات|الم.تويات|الفهرس|محتويات\s+الدليل)\s*$"
)


def line_numbers(lines: list[str], pattern: re.Pattern) -> list[int]:
    numbers = []
    for number, line in enumerate(lines, 1):
        if pattern.search(line):
            numbers.append(number)
    return numbers


def find_problems(lines: list[str]) -> list[str]:
    problems = []
    broken = line_numbers(lines, OCR_BROKEN)
    if broken:
        problems.append(f"OCR مكسور في {len(broken)} سطر (أولها: {broken[0]})")
    html = line_numbers(lines, HTML_LEFTOVER)
    if html:
        problems.append(f"HTML متبقٍّ في {len(html)} سطر (أولها: {html[0]})")
    contents = line_numbers([line.strip() for line in lines], TABLE_OF_CONTENTS)
    if contents:
        problems.append(f"فهرس متبقٍّ في السطر {contents[0]}")
    return problems


def repeated_lines(lines: list[str]) -> list[str]:
    """للعلم فقط: شروط الجداول المشتركة بين الكليات تتكرر بشكل مشروع."""
    seen = set()
    repeated = []
    for line in lines:
        line = line.strip()
        if len(line) > 40:
            if line in seen:
                repeated.append(line)
            seen.add(line)
    return repeated


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    is_clean = True
    total_chars = 0
    total_files = 0
    for university, folder in FOLDERS:
        print(f"\n{'=' * 58}\n{university}\n{'=' * 58}")
        for path in sorted(folder.glob("result_*.txt")):
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            problems = find_problems(lines)
            repeated = repeated_lines(lines)
            content_lines = sum(1 for line in lines if line.strip())
            total_chars += len(text)
            total_files += 1
            print(f"\n  [{'نظيف' if not problems else 'عيب'}] {path.name}")
            print(f"     {len(text):,} حرف | {content_lines} سطر محتوى")
            for problem in problems:
                is_clean = False
                print(f"     ! {problem}")
            if repeated:
                print(f"     ~ {len(repeated)} سطر مكرر (متوقع: {repeated[0][:45]}...)")
    print(f"\n{'=' * 58}")
    print(f"الإجمالي: {total_files} ملف | {total_chars:,} حرف")
    print("كل الملفات نظيفة" if is_clean else "فيه تحذيرات فوق")


if __name__ == "__main__":
    main()
