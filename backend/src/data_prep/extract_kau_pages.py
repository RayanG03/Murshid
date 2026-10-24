"""Stage-2b: write per-page KAU PDF text into the committed transcribed/ layout.

For every PDF under sources/KAU/ (except the internal transfer guide, whose
human-verified transcription already exists — use --all to include it), each
page is extracted with PyMuPDF, NFKC-folded (repairs Arabic presentation
forms losslessly), stripped of repeated title headers and trailing bare
page-number footers, and saved as:

    sources/KAU/transcribed/<pdf-stem>/page_NN.txt

Blank lines are PRESERVED — the kau_pdf ingestor groups blocks on them.

The report flags, per document, whether the text layer is usable
(arabic lines, low-char pages, presentation forms, lam-alef sanity).
Documents flagged NO-ARABIC-TEXT / GARBLED / many low-char pages are OCR
candidates: OCR their pages into the same folder as page_NN.txt (see
sources/KAU/transcribed/README.md) and re-run with --overwrite.

Run from backend/:
  uv run python -X utf8 -m src.data_prep.extract_kau_pages [--all] [--overwrite] [name-filter ...]
"""

import re
import sys
import unicodedata
from pathlib import Path

import fitz

from src.ingest.kau_pdf import DOC_TITLES, LEGACY_DIAGNOSTICS_STEM
from src.data_prep.extract_pdfs import (
    ARABIC_RE,
    LAM_ALEF_PROPER_RE,
    LAM_ALEF_SCRAMBLED_RE,
    PRESENTATION_RE,
)

LOW_CHAR_THRESHOLD = 50
DEFAULT_HEADER = "دليل التحويل الداخلي بين الكليات"


def clean_page_text(raw: str, title: str) -> str:
    """NFKC-fold one page and drop title headers / trailing page numbers."""
    text = unicodedata.normalize("NFKC", raw)
    header_re = re.compile(rf"^{re.escape(title)}\s*[\d/]+\s*هـ\s*$")
    lines = [ln for ln in text.splitlines() if not header_re.match(ln.strip())]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and re.fullmatch(r"[0-9\u0660-\u0669]{1,3}", lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def looks_visual_order(joined: str, min_words: int = 20) -> bool:
    """Heuristic: did the text layer come out visually (right-to-left) reversed?

    In logical-order Arabic a large share of words start with the definite
    article 'ال' (المادة، اللائحة، الطلاب، ...). Visually reversed extraction
    almost never produces word-initial 'ال'. Only trusted when there are
    enough Arabic words to judge (>= min_words).
    """
    words = [w for w in joined.split() if ARABIC_RE.search(w)]
    if len(words) < min_words:
        return False
    prefix_al = sum(1 for w in words if w.startswith("\u0627\u0644"))
    return prefix_al / len(words) < 0.02


def write_doc_pages(pdf_path: Path, out_dir: Path, title: str) -> dict[str, object]:
    """Extract one PDF into page_NN.txt files and return quality stats."""
    doc = fitz.open(pdf_path)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        total_chars = 0
        low_pages: list[int] = []
        joined_parts: list[str] = []
        for page_index, page in enumerate(doc):
            cleaned = clean_page_text(page.get_text(), title)
            if cleaned:
                total_chars += len(cleaned)
                joined_parts.append(cleaned)
            if len(cleaned) < LOW_CHAR_THRESHOLD:
                low_pages.append(page_index + 1)
            (out_dir / f"page_{page_index + 1:02d}.txt").write_text(
                cleaned, encoding="utf-8"
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
        flags.append("PRESENTATION-FORMS (possible broken shaping!)")
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
    """Extract KAU PDFs into the committed transcribed/ layout."""
    from src.config import get_settings

    settings = get_settings()
    args = sys.argv[1:]
    include_all = "--all" in args
    overwrite = "--overwrite" in args
    filters = [a for a in args if not a.startswith("--")]

    kau_dir = settings.sources_dir / "KAU"
    out_root = kau_dir / "transcribed"
    pdfs = sorted(kau_dir.glob("*.pdf"))
    if filters:
        pdfs = [p for p in pdfs if any(f in p.name for f in filters)]
    if not pdfs:
        raise SystemExit("No PDFs matched.")

    for pdf_path in pdfs:
        stem = pdf_path.stem
        out_dir = out_root / stem
        existing = list(out_dir.glob("page_*.txt")) if out_dir.is_dir() else []
        if stem == LEGACY_DIAGNOSTICS_STEM and not include_all:
            print(
                f"\n=== skipping {stem} (verified legacy transcription; "
                "use --all to force) ==="
            )
            continue
        if existing and not overwrite:
            print(
                f"\n=== skipping {stem} ({len(existing)} pages already in "
                "transcribed/; use --overwrite to redo) ==="
            )
            continue

        title = DOC_TITLES.get(stem, stem)
        stats = write_doc_pages(pdf_path, out_dir, title)
        print_report(pdf_path, stats)
        print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
