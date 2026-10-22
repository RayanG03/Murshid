"""Finalize دليل التحويل الداخلي.pdf extraction from verified Gemini pages.

Applies two manually-verified corrections to the cached Gemini page outputs
(backend/data/extracted/_diagnostics/دليل التحويل الداخلي/):
1. Academic-year order: the source document prints "1448/1447" everywhere;
   Gemini flipped it to "1447/1448" on pages 02 and 04 and in the running
   page footer on pages 05-09. All occurrences are normalized to "1448/1447".
2. Page 12 footer: Gemini dropped the contact block (social handles and
   email); the verified content is appended.

Then assembles the 12 corrected pages into the final extraction file at
backend/data/extracted/KAU/دليل التحويل الداخلي.txt (UTF-8, pages joined by
blank lines, matching the extract_pdfs.py output convention).

Provenance: text-layer extraction (PyMuPDF / pypdfium2 / pdfplumber) failed on
this PDF due to corrupted ToUnicode glyph-to-Unicode maps; the text below comes
from Gemini vision transcription (gemini-2.5-flash) of 300-DPI page renders,
cross-checked against Tesseract OCR output, with the two corrections above
applied after manual spot-checks against the source PDF.

Run from backend/:
  uv run python -X utf8 -m src.data_prep.finalize_internal_guide
"""

import re
from pathlib import Path

YEAR_WRONG = "1447/1448"
YEAR_RIGHT = "1448/1447"

VERIFIED_P12_FOOTER = """Instagram: KAUAdmission
YouTube: kauadmission
X: admissionweb
Snapchat: kauadmission
البريد الإلكتروني: dar@kau.edu.sa"""


def correct_year_order(page_dir: Path) -> list[str]:
    """Normalize the flipped academic-year pair in every cached page."""
    fixed: list[str] = []
    for f in sorted(page_dir.glob("page_*.gemini.txt")):
        text = f.read_text(encoding="utf-8")
        if YEAR_WRONG in text:
            count = text.count(YEAR_WRONG)
            f.write_text(text.replace(YEAR_WRONG, YEAR_RIGHT), encoding="utf-8")
            fixed.append(f"{f.name}: {count}x")
    return fixed


def append_p12_footer(page_dir: Path) -> bool:
    """Append the manually-verified footer content to page 12's output."""
    f = page_dir / "page_12.gemini.txt"
    text = f.read_text(encoding="utf-8")
    if "dar@kau.edu.sa" in text:
        return False
    f.write_text(text.rstrip() + "\n" + VERIFIED_P12_FOOTER + "\n",
                 encoding="utf-8")
    return True


def assemble_final(page_dir: Path, out_path: Path) -> int:
    """Join the 12 corrected pages into the final extraction file."""
    pages = []
    for f in sorted(page_dir.glob("page_*.gemini.txt")):
        body = f.read_text(encoding="utf-8").strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        pages.append(body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(pages) + "\n", encoding="utf-8")
    return len("\n\n".join(pages))


def main() -> None:
    from src.config import get_settings

    settings = get_settings()
    page_dir = (settings.data_dir / "extracted" / "_diagnostics"
                / "دليل التحويل الداخلي")
    out_path = (settings.data_dir / "extracted" / "KAU"
                / "دليل التحويل الداخلي.txt")

    fixed = correct_year_order(page_dir)
    print("year-order corrections applied:", fixed if fixed else "none")
    added = append_p12_footer(page_dir)
    print("p12 verified footer appended:", added)
    total = assemble_final(page_dir, out_path)
    print(f"final extraction -> {out_path} ({total} chars, 12 pages)")


if __name__ == "__main__":
    main()
