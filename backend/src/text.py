"""Text utilities shared by ingestion modules."""

import re

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ARABIC_INDIC_EXTENDED = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DECIMAL_SEP = "٫"
ARABIC_THOUSANDS_SEP = "٬"

_DIGIT_TABLE = str.maketrans(
    ARABIC_INDIC_DIGITS + ARABIC_INDIC_EXTENDED + ARABIC_DECIMAL_SEP + ARABIC_THOUSANDS_SEP,
    "0123456789" + "0123456789" + "." + ",",
)


def normalize_digits_for_search(text: str) -> str:
    """Convert Arabic-Indic digits/separators to Western forms.

    Used ONLY when building the text that gets embedded (search copy);
    original text is always kept for display, per AGENTS.md.
    """
    return text.translate(_DIGIT_TABLE)


TATWEEL = "\u0640"

# Arabic diacritics (fathatan..sukun), a PDF text-layer extraction artifact.
_DIACRITICS = "".join(chr(cp) for cp in range(0x064B, 0x0653))

_CLEAN_TABLE: dict[int, str | None] = {ord(TATWEEL): None}
_CLEAN_TABLE.update({cp: None for cp in range(0x064B, 0x0653)})


def clean_search_text(text: str) -> str:
    """Remove ONLY U+0640 (tatweel) and U+064B-U+0652 (Arabic diacritics).

    Both are PDF text-layer extraction artifacts rather than source
    content, so cleaning is applied to the chunk text that is embedded
    AND stored (display). Nothing else is touched: no letter
    normalization, no reordering, no digit work.
    """
    return text.translate(_CLEAN_TABLE)
