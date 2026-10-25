"""Convert Mistral OCR playground output into transcribed page_NN.txt files.

The Mistral OCR playground (free web UI) downloads a zip with this shape:

    <doc-name>.pdf/
        markdown.md                  (combined pages, ignored here)
        pages/page-N/markdown.md     (per-page transcription, N is 1-based)
        pages/page-N/page-metadata.json
        pages/page-N/img-*.jpeg      (page images, ignored here)

This tool turns that into the layout the KAU/KSU PDF ingestors consume:

    <out-dir>/<doc-stem>/page_NN.txt   (NN zero-padded, 1-based page order)

Cleaning is verbatim-preserving: markdown image refs are dropped, heading
markers (#) are stripped while keeping the heading text, outer blank lines
are trimmed. Nothing else is touched — no "correction", no reordering.

Usage:
    python mistral_to_pages.py <mistral.zip-or-extracted-dir> <out-dir>

Process every <doc>.pdf/ folder found in the input. Pages with no Arabic
characters are flagged loudly (they may need a re-OCR) but still written.

Typical follow-up (from backend/):
    uv run python -X utf8 -m src.ingest.run_kau --only pdf   (or run_ksu)
"""

import re
import sys
import zipfile
from pathlib import Path

# Markdown image reference anywhere in a line: ![alt](target)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Leading heading markers: "#### المادة الثامنة:" -> "المادة الثامنة:"
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
# Any Arabic letter (used for the no-Arabic-page warning).
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
# Arabic presentation forms — the "garbled" signature we are escaping from.
PRESENTATION_FORMS_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")


def clean_page_md(raw: str) -> str:
    """Markdown page -> plain text page (verbatim-preserving)."""
    lines_out: list[str] = []
    for line in raw.splitlines():
        if IMAGE_RE.search(line):
            rest = IMAGE_RE.sub("", line).strip()
            if not rest:
                continue
            line = rest
        line = HEADING_RE.sub("", line).rstrip()
        lines_out.append(line)
    text = "\n".join(lines_out)
    return text.strip("\n") + "\n" if text.strip() else ""


def convert_doc(doc_dir: Path, out_dir: Path) -> bool:
    """One <doc>.pdf/ folder -> <out>/<stem>/page_NN.txt. Returns OK flag."""
    pages_root = doc_dir / "pages"
    if not pages_root.is_dir():
        print(f"[skip] {doc_dir.name}: no pages/ inside")
        return False
    stem = doc_dir.name.removesuffix(".pdf")
    target = out_dir / stem
    target.mkdir(parents=True, exist_ok=True)

    page_dirs = sorted(
        (p for p in pages_root.iterdir() if p.is_dir() and p.name.startswith("page-")),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not page_dirs:
        print(f"[skip] {doc_dir.name}: pages/ has no page-N folders")
        return False

    ok = True
    total_chars = 0
    for page_dir in page_dirs:
        n = int(page_dir.name.split("-")[1])
        src = page_dir / "markdown.md"
        if not src.exists():
            print(f"[warn] {doc_dir.name}/{page_dir.name}: markdown.md missing")
            ok = False
            continue
        text = clean_page_md(src.read_text(encoding="utf-8"))
        if PRESENTATION_FORMS_RE.search(text):
            print(
                f"[warn] {doc_dir.name}/page_{n:02d}: presentation-form "
                "codepoints found — unexpected for Mistral output"
            )
            ok = False
        if text.strip() and not ARABIC_RE.search(text):
            print(f"[warn] {doc_dir.name}/page_{n:02d}: NO Arabic characters")
            ok = False
        if not text.strip():
            print(f"[warn] {doc_dir.name}/page_{n:02d}: EMPTY page")
            ok = False
        (target / f"page_{n:02d}.txt").write_text(text, encoding="utf-8")
        total_chars += len(text)
    print(
        f"[done] {stem}: pages={len(page_dirs)} total_chars={total_chars} "
        f"-> {target}"
    )
    return ok


def main() -> None:
    """Unzip (if needed), find <doc>.pdf/ folders, convert each one."""
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    root: Path
    tmp: Path | None = None
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = out_dir.parent / f"_mistral_extract_{src.stem}"
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        print(f"[unzip] {src.name} -> {tmp}")
        root = tmp
    elif src.is_dir():
        root = src
    else:
        raise SystemExit(f"input not found: {src}")

    doc_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(".pdf")]
    if not doc_dirs:
        raise SystemExit(f"no <doc>.pdf/ folders under {root}")
    all_ok = True
    for doc_dir in sorted(doc_dirs):
        all_ok = convert_doc(doc_dir, out_dir) and all_ok
    if tmp:
        print(f"[note] extracted input kept at {tmp} (delete it anytime)")
    if not all_ok:
        print("[!] warnings above — review flagged pages before ingesting")
    print(
        "\nNext: copy <out-dir>/<stem>/ over "
        "backend/sources/<UNI>/transcribed/<stem>/ then re-ingest that "
        "university's PDF collection."
    )


if __name__ == "__main__":
    main()
