"""Post-generation checks: numeric grounding and programmatic sources.

The prompt already forbids invented numbers, but that relies on the model
obeying. ``ungrounded_numbers`` finds numbers in the answer (GPA values,
hours, percentages, dates, article numbers) that appear nowhere in the
question or the retrieved context, so they can be logged or refused
(settings.numeric_guard_mode).
"""

import re

from src.text import normalize_digits_for_search

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[المصدر:[^\]]*\]")
_LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}\s*[.)\-–]\s", re.M)

# Small numbers are often written in words in the sources ("ثلاث ساعات");
# a digit 1-10 counts as grounded when a matching word stem is present.
_NUMBER_WORDS: dict[str, tuple[str, ...]] = {
    "1": ("واحد", "أول", "اول"),
    "2": ("اثن", "ثنت", "ثاني", "فصلين", "سنتين", "مرتين"),
    "3": ("ثلاث", "ثالث"),
    "4": ("أربع", "اربع", "رابع"),
    "5": ("خمس", "خامس"),
    "6": ("ست", "سادس"),
    "7": ("سبع", "سابع"),
    "8": ("ثمان", "ثامن"),
    "9": ("تسع", "تاسع"),
    "10": ("عشر", "عاشر"),
}


def _canonical(number: str) -> str:
    """'4,50' / '4.50' -> '4.5'; '08' -> '8'."""
    n = number.replace(",", ".")
    if "." in n:
        whole, frac = n.split(".", 1)
        frac = frac.rstrip("0")
        whole = whole.lstrip("0") or "0"
        return f"{whole}.{frac}" if frac else whole
    return n.lstrip("0") or "0"


def numbers_in(text: str) -> set[str]:
    """Canonical numbers found in text (Arabic-Indic digits normalized)."""
    return {_canonical(m) for m in _NUM_RE.findall(normalize_digits_for_search(text))}


def ungrounded_numbers(answer: str, context: str, question: str = "") -> list[str]:
    """Numbers in the answer that the context/question cannot back.

    Citation tags ("[المصدر: ...]") and list markers ("1. ", "2- ") are
    ignored; digits 1-10 also match their Arabic word forms in the context.
    """
    body = _CITATION_RE.sub(" ", answer)
    body = _LIST_MARKER_RE.sub(" ", normalize_digits_for_search(body))
    backed = numbers_in(context) | numbers_in(question)
    missing: list[str] = []
    for num in sorted(numbers_in(body), key=lambda s: (len(s), s)):
        if num in backed:
            continue
        words = _NUMBER_WORDS.get(num)
        if words and any(w in context for w in words):
            continue
        missing.append(num)
    return missing
