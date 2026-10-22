"""Diagnostics ONLY for KAU/تقويم 26-27.pdf extraction modes (Step A).

Compares extraction approaches on the calendar PDF:
- "text" default (baseline, known garbled)
- "text" with sort=True (layout-aware ordering)
- span-level reconstruction from get_text("dict") bboxes (RTL span sort)
- flag variations (TEXT_PRESERVE_LIGATURES on/off, TEXT_DEHYPHENATE)

For each mode prints: total chars, scrambled-vs-proper lam-alef counts,
and sample lines. Writes NO files.
"""

import re
from pathlib import Path

import fitz

SCRAMBLED_RE = re.compile(r"\u0627[\u0623\u0625\u0622]\u0644")
PROPER_RE = re.compile(r"\u0627\u0644[\u0623\u0625\u0622]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def lam_alef_counts(text: str) -> tuple[int, int]:
    return len(SCRAMBLED_RE.findall(text)), len(PROPER_RE.findall(text))


def show(text: str, label: str) -> None:
    scrambled, proper = lam_alef_counts(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    arabic = [l for l in lines if ARABIC_RE.search(l)]
    interesting = [l for l in lines if re.search(r"[\u0623\u0625\u0622]", l)]
    print(f"\n--- {label} ---")
    print(f"chars={len(text)}  lines={len(lines)}  "
          f"scrambled-lam-alef={scrambled}  proper-lam-alef={proper}")
    print("lines containing hamza-forms (ligature test):")
    for l in interesting[:5]:
        print(f"  | {l[:150]}")
    digit_lines = [l for l in lines if re.search(r"[0-9\u0660-\u0669]", l)]
    print(f"digit-bearing lines: {len(digit_lines)} (first 4):")
    for l in digit_lines[:4]:
        print(f"  # {l[:150]}")


def span_reconstruct(page: fitz.Page) -> str:
    """Rebuild text from 'dict' spans, ordering spans right-to-left per line."""
    out_lines: list[str] = []
    d = page.get_text("dict")
    for block in d["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s["text"].strip()]
            if not spans:
                continue
            spans.sort(key=lambda s: -s["bbox"][0])
            out_lines.append(" ".join(s["text"] for s in spans))
    return "\n".join(out_lines)


def main() -> None:
    from src.config import get_settings

    pdf = get_settings().sources_dir / "KAU" / "تقويم 26-27.pdf"
    doc = fitz.open(pdf)
    page = doc[0]

    show(page.get_text("text"), 'MODE 1: get_text("text") default [baseline]')
    show(page.get_text("text", sort=True), 'MODE 2: get_text("text", sort=True)')

    default_flags = fitz.TEXTFLAGS_TEXT
    combos = [
        ("no PRESERVE_LIGATURES", default_flags & ~fitz.TEXT_PRESERVE_LIGATURES),
        ("DEHYPHENATE", default_flags | fitz.TEXT_DEHYPHENATE),
        (
            "no LIGATURES + sort",
            (default_flags & ~fitz.TEXT_PRESERVE_LIGATURES) | 0,
        ),
    ]
    for name, flags in combos:
        show(
            page.get_text("text", flags=flags,
                          sort=name.endswith("sort")),
            f"MODE 3: flags={name}",
        )

    show(span_reconstruct(page), "MODE 4: span-level RTL reconstruction")

    raw = page.get_text("rawdict")
    print("\n--- rawdict char detail for one ligature word ---")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = [c for c in span.get("chars", []) if c["c"].strip()]
                text = "".join(c["c"] for c in chars)
                if SCRAMBLED_RE.search(text):
                    print(f"span text: {text[:80]}")
                    for c in chars[:8]:
                        print(f"  char {c['c']!r} bbox_x0={c['bbox'][0]:.1f}")
                    break
            else:
                continue
            break
        else:
            continue
        break
    doc.close()


if __name__ == "__main__":
    main()
