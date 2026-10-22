"""Stage 2: extract text from source PDFs (PyMuPDF) with per-PDF reports.

For each PDF prints: page count, total characters, 10 sample Arabic lines,
every page under 50 characters, and lines containing numbers (integrity check).
Flags presentation-form characters (a sign of broken Arabic shaping).
Extracted text is saved to backend/data/extracted/ for later stages.

Run from backend/:
  uv run python -X utf8 -m src.data_prep.extract_pdfs [name-filter ...]
"""

import re
import sys
from pathlib import Path

import fitz

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
PRESENTATION_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
ARABIC_INDIC_NUM_RE = re.compile(r"[\u0660-\u0669]")
NUMBER_LINE_RE = re.compile(r"(\d+[.,]?\d*\s*[%٪]|ساع[ةات]\s*\D{0,12}\d|\d+[.,]\d+)")
LAM_ALEF_SCRAMBLED_RE = re.compile(r"\u0627[\u0623\u0625\u0622]\u0644")
LAM_ALEF_PROPER_RE = re.compile(r"\u0627\u0644[\u0623\u0625\u0622]")

LOW_CHAR_THRESHOLD = 50
SAMPLE_LINE_COUNT = 10
NUMBER_SAMPLE_COUNT = 8


def extract_pdf(pdf_path: Path, out_txt: Path) -> dict[str, object]:
    """Extract one PDF, save its text, and return quality stats."""
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text() for page in doc]
    total_chars = sum(len(t) for t in page_texts)
    low_pages = [
        (i + 1, len(t)) for i, t in enumerate(page_texts)
        if len(t.strip()) < LOW_CHAR_THRESHOLD
    ]

    lines = [ln.strip() for t in page_texts for ln in t.splitlines() if ln.strip()]
    arabic_lines = [ln for ln in lines if ARABIC_RE.search(ln)]
    number_lines = [ln for ln in lines if NUMBER_LINE_RE.search(ln)]

    joined = "\n".join(lines)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(joined, encoding="utf-8")
    doc.close()

    return {
        "path": pdf_path,
        "pages": len(page_texts),
        "total_chars": total_chars,
        "low_pages": low_pages,
        "sample_lines": arabic_lines[:SAMPLE_LINE_COUNT],
        "number_lines": number_lines[:NUMBER_SAMPLE_COUNT],
        "number_line_count": len(number_lines),
        "arabic_line_count": len(arabic_lines),
        "presentation_forms": bool(PRESENTATION_RE.search(joined)),
        "arabic_indic_digits": bool(ARABIC_INDIC_NUM_RE.search(joined)),
        "lam_alef_scrambled": len(LAM_ALEF_SCRAMBLED_RE.findall(joined)),
        "lam_alef_proper": len(LAM_ALEF_PROPER_RE.findall(joined)),
    }


def print_report(stats: dict[str, object]) -> None:
    """Print the per-PDF extraction report."""
    p = stats["path"]
    print(f"\n=== {p.parent.name}/{p.name} ===")
    print(f"pages={stats['pages']}  chars={stats['total_chars']}  "
          f"arabic_lines={stats['arabic_line_count']}  "
          f"number_lines={stats['number_line_count']}")
    flags = []
    if stats["presentation_forms"]:
        flags.append("PRESENTATION-FORMS (possible broken shaping!)")
    if stats["arabic_line_count"] == 0:
        flags.append("NO-ARABIC-TEXT")
    print(f"flags: {flags if flags else 'none'}")
    print(f"arabic-indic digits present: {stats['arabic_indic_digits']}")
    print(f"lam-alef: scrambled={stats['lam_alef_scrambled']} "
          f"proper={stats['lam_alef_proper']}"
          + ("  <-- GARBLED" if stats["lam_alef_scrambled"] > 0
             and stats["lam_alef_proper"] == 0 else ""))
    low = stats["low_pages"]
    if low:
        print(f"pages under {LOW_CHAR_THRESHOLD} chars: {low}")
    else:
        print(f"pages under {LOW_CHAR_THRESHOLD} chars: none")
    print("sample Arabic lines:")
    for ln in stats["sample_lines"]:
        print(f"  | {ln[:160]}")
    print("lines containing numbers:")
    for ln in stats["number_lines"]:
        print(f"  # {ln[:160]}")


def main() -> None:
    """Extract all PDFs under sources/KAU and sources/KSU (optional filters)."""
    from src.config import get_settings

    settings = get_settings()
    filters = sys.argv[1:]
    pdfs = sorted(
        list((settings.sources_dir / "KAU").glob("*.pdf"))
        + list((settings.sources_dir / "KSU").glob("*.pdf"))
    )
    if filters:
        pdfs = [p for p in pdfs if any(f in p.name for f in filters)]
    if not pdfs:
        raise SystemExit("No PDFs matched.")

    out_root = settings.data_dir / "extracted"
    for pdf in pdfs:
        out_txt = out_root / pdf.parent.name / f"{pdf.stem}.txt"
        stats = extract_pdf(pdf, out_txt)
        print_report(stats)
        print(f"saved -> {out_txt}")


if __name__ == "__main__":
    main()
