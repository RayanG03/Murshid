"""Step D diagnostics: pdfplumber vs pypdfium2 on the 4 corrupted KAU PDFs.

Checks per (file, library):
- lam-alef ligature reversal (scrambled vs proper counts)
- probe phrases found (normalized: tatweel/space-stripped, incl. lam-alef words)
- suspicious non-Arabic/non-ASCII chars inside Arabic text (wrong codepoints)
- digit-Arabic gluing (numbers detached from labels)
- sample lines + calendar table lines

Writes NO files. Run: uv run python -X utf8 -m src.data_prep.diag_libraries
"""

import re
from collections import Counter
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

SCRAMBLED_RE = re.compile(r"\u0627[\u0623\u0625\u0622]\u0644")
PROPER_RE = re.compile(r"\u0627\u0644[\u0623\u0625\u0622]")
ARABIC_WORD_RE = re.compile(r"[\u0621-\u064A]")
ALLOWED = set(
    "\n\t "
    "0123456789"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "،؛؟«»“”‘’–—…·"
)
GLUED_RE = re.compile(r"([\u0600-\u06FF][0-9]|[0-9][\u0600-\u06FF])")

PROBES_ALL = ["جامعة الملك عبدالعزيز", "القبول والتسجيل"]
PROBES_SPECIFIC = {
    "تقويم": ["مواعيد التقويم الأكاديمي", "للانتظام"],
    "الخارجي": ["التراكمي", "لبرامج"],
    "البكالوريوس": ["لبرامج", "الموافق"],
    "التحويل الداخلي": ["المسارات الأكاديمية", "الكليات الصحية"],
}


def normalize(text: str) -> str:
    """Strip tatweel and collapse whitespace for probe matching."""
    return re.sub(r"[\u0640\s]+", "", text)


def extract_pdfplumber(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_pypdfium2(path: Path) -> str:
    parts: list[str] = []
    doc = pdfium.PdfDocument(str(path))
    for page in doc:
        textpage = page.get_textpage()
        parts.append(textpage.get_text_range())
        textpage.close()
        page.close()
    doc.close()
    return "\n".join(parts)


def check_text(text: str, label: str, probes: list[str], show_table: bool) -> None:
    print(f"\n=== {label} ===")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    arabic = [l for l in lines if ARABIC_WORD_RE.search(l)]
    print(f"chars={len(text)} lines={len(lines)} arabic_lines={len(arabic)}")
    print(f"lam-alef: scrambled={len(SCRAMBLED_RE.findall(text))} "
          f"proper={len(PROPER_RE.findall(text))}")

    norm = normalize(text)
    for p in probes:
        found = normalize(p) in norm
        print(f"probe '{p}': {'FOUND' if found else 'not found'}")

    suspicious = Counter(
        ch for ch in text if ch not in ALLOWED and ch != "\r"
        and not ("\u0600" <= ch <= "\u06FF")
    )
    if suspicious:
        top = ", ".join(f"{c!r}(U+{ord(c):04X})x{n}" for c, n in suspicious.most_common(6))
        print(f"suspicious chars: {top}")
    else:
        print("suspicious chars: none")

    glued = len(GLUED_RE.findall(text))
    print(f"digit-glued-to-Arabic occurrences: {glued}")

    print("sample lines:")
    for l in arabic[:5]:
        print(f"  | {l[:150]}")
    digit_lines = [l for l in lines if re.search(r"[0-9]", l)]
    if show_table:
        print(f"calendar/table digit lines (first 6 of {len(digit_lines)}):")
        for l in digit_lines[:6]:
            print(f"  # {l[:150]}")
    else:
        dec = [l for l in lines if re.search(r"\d+[.,]\d+", l)]
        print(f"decimal-number lines (first 4 of {len(dec)}):")
        for l in dec[:4]:
            print(f"  # {l[:150]}")


def main() -> None:
    from src.config import get_settings

    kau = get_settings().sources_dir / "KAU"
    targets = [
        "تقويم 26-27.pdf",
        "دليل التحويل الداخلي.pdf",
        "دليل القبول لبرامج البكالوريوس والدبلوم.pdf",
        "دليل القبول للتحويل الخارجي إلى الجامعة.pdf",
    ]
    for name in targets:
        path = kau / name
        key = next((k for k in PROBES_SPECIFIC if k in name), "")
        probes = PROBES_ALL + PROBES_SPECIFIC.get(key, [])
        is_calendar = "تقويم" in name
        t1 = extract_pdfplumber(path)
        check_text(t1, f"pdfplumber :: {name}", probes, is_calendar)
        t2 = extract_pypdfium2(path)
        check_text(t2, f"pypdfium2 :: {name}", probes, is_calendar)


if __name__ == "__main__":
    main()
