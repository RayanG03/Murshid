"""Stage-2b (KSU): write per-page KSU PDF text into the committed transcribed layout.

For every PDF under sources/KSU/, each page is extracted with PyMuPDF,
NFKC-folded (repairs the regulation's Arabic presentation forms losslessly),
stripped of TOC dot-leader lines and the trailing bare page-number footer,
and saved as:

    sources/KSU/transcribed/<pdf-stem>/page_NN.txt

Unlike the KAU layout, blank lines are NOT preserved — the KSU ingestor
works line-by-line (the regulation PDF has no blank lines), so each page
file holds only its non-empty cleaned lines, one per line. Cover/TOC pages
that end up empty are written as empty files to keep page numbering intact.
All pages are written (front matter included); the front-matter SKIP for
the 2024 regulation happens at ingest time, not here.

The report flags, per document, whether the text layer is usable
(arabic lines, low-char pages, presentation forms, lam-alef sanity,
visual-order reversal). Documents flagged NO-ARABIC-TEXT / GARBLED /
many low-char pages are OCR candidates: OCR their pages into the same
folder as page_NN.txt (see sources/KSU/transcribed/README.md).

Run from backend/:
  uv run python -X utf8 -m src.data_prep.extract_ksu_pages [--overwrite] [name-filter ...]
"""

import sys
from pathlib import Path

import fitz

from src.ingest.ksu_pdf import FOOTER_NUM_RE, TOC_DOTS_RE
from src.data_prep.extract_pdfs import (
    ARABIC_RE,
    LAM_ALEF_PROPER_RE,
    LAM_ALEF_SCRAMBLED_RE,
    PRESENTATION_RE,
)
# Shared visual-order detector (defined once with the KAU extractor).
from src.data_prep.extract_kau_pages import LOW_CHAR_THRESHOLD, looks_visual_order

import unicodedata


def clean_page_lines(raw: str) -> list[str]:
    """NFKC-fold one page and return its kept lines (KSU line-based layout).

    Mirrors src.ingest.ksu_pdf.extract_pages' per-page cleaning: drop empty
    lines and TOC dot-leader lines, then drop one trailing bare footer
    page-number line.
    """
    lines = [
        unicodedata.normalize("NFKC", ln.strip())
        for ln in raw.splitlines()
        if ln.strip() and not TOC_DOTS_RE.search(ln)
    ]
    if lines and FOOTER_NUM_RE.match(lines[-1]):
        lines.pop()
    return lines


def write_doc_pages(pdf_path: Path, out_dir: Path) -> dict[str, object]:
    """Extract one PDF into page_NN.txt files and return quality stats."""
    doc = fitz.open(pdf_path)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        total_chars = 0
        low_pages: list[int] = []
        joined_parts: list[str] = []
        for page_index, page in enumerate(doc):
            kept = clean_page_lines(page.get_text())
            text = "\n".join(kept)
            if text:
                total_chars += len(text)
                joined_parts.append(text)
            if len(text) < LOW_CHAR_THRESHOLD:
                low_pages.append(page_index + 1)
            (out_dir / f"page_{page_index + 1:02d}.txt").write_text(
                text, encoding="utf-8"
            )
            written += 1
    finally:
        doc.close()

    joined = "\n".join(joined_parts)
    arabic_lines = [ln for ln in joined.splitlines() if ARABIC_RE.search(ln)]
    scrambled = len(LAM_ALEF_SCRAMBLED_RE.findall(joined))
    proper = len(LAM_ALEF_PROPER_RE.findall(joined))
    flags: list[str] = []
    if not arabic_lines:
        flags.append("NO-ARABIC-TEXT")
    if PRESENTATION_RE.search(joined):
        flags.append("PRESENTATION-FORMS (NFKC should have fixed these!)")
    if scrambled > 0 and proper == 0:
        flags.append("GARBLED (lam-alef scrambled)")
    if looks_visual_order(joined):
        flags.append("VISUAL-ORDER (reversed words!) — do not ingest")
    low_ratio = len(low_pages) / written if written else 0.0
    verdict = "text-layer OK"
    if flags or low_ratio > 0.3:
        verdict = "OCR CANDIDATE"
    return {
        "pages": written,
        "total_chars": total_chars,
        "arabic_line_count": len(arabic_lines),
        "low_pages": low_pages,
        "flags": flags,
        "verdict": verdict,
    }


def print_report(pdf_path: Path, stats: dict[str, object]) -> None:
    """Print the per-PDF extraction report."""
    print(f"\n=== {pdf_path.parent.name}/{pdf_path.name} ===")
    print(
        f"pages={stats['pages']}  chars={stats['total_chars']}  "
        f"arabic_lines={stats['arabic_line_count']}"
    )
    flags = stats["flags"]
    print(f"flags: {flags if flags else 'none'}")
    low = stats["low_pages"]
    if low:
        print(f"pages under {LOW_CHAR_THRESHOLD} chars: {low}")
    else:
        print(f"pages under {LOW_CHAR_THRESHOLD} chars: none")
    print(f"verdict: {stats['verdict']}")


def main() -> None:
    """Extract KSU PDFs into the committed transcribed/ layout."""
    from src.config import get_settings

    settings = get_settings()
    args = sys.argv[1:]
    overwrite = "--overwrite" in args
    filters = [a for a in args if not a.startswith("--")]

    ksu_dir = settings.sources_dir / "KSU"
    out_root = ksu_dir / "transcribed"
    pdfs = sorted(ksu_dir.glob("*.pdf"))
    if filters:
        pdfs = [p for p in pdfs if any(f in p.name for f in filters)]
    if not pdfs:
        raise SystemExit("No PDFs matched.")

    for pdf_path in pdfs:
        stem = pdf_path.stem
        out_dir = out_root / stem
        existing = list(out_dir.glob("page_*.txt")) if out_dir.is_dir() else []
        if existing and not overwrite:
            print(
                f"\n=== skipping {stem} ({len(existing)} pages already in "
                "transcribed/; use --overwrite to redo) ==="
            )
            continue

        stats = write_doc_pages(pdf_path, out_dir)
        print_report(pdf_path, stats)
        print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
