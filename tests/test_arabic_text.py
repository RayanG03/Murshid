"""اختبارات تطبيع النص العربي (arabic_text.py).

التشغيل:
    python -m pytest tests/test_arabic_text.py
"""

from arabic_text import normalize


def test_normalize_unifies_hamza_carriers_and_ta_marbuta() -> None:
    assert normalize("مكافأة") == "مكافاه"
    assert normalize("مكافئه") == "مكافاه"
    assert normalize("مسئول") == normalize("مسؤول")


def test_normalize_removes_diacritics_and_converts_digits() -> None:
    assert normalize("المادةُ ٢٥") == "الماده 25"
    assert normalize("يُفصَل") == "يفصل"
