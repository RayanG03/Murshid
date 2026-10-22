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
