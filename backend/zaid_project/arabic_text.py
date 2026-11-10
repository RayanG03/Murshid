"""تطبيع النص العربي للمقارنة: حذف التشكيل، وتوحيد الهمزات والياء والتاء المربوطة.

يستعمله answer.py (هل طلبت الصياغة «التقويم الأكاديمي») و chunk.py (العناوين العامة
مثل «ملاحظة» و«تنبيه»). البحث اللفظي نفسه يأخذ أوزان كلماته من bge-m3 (retrieve.py).

الاستخدام:
    from arabic_text import normalize
    normalize("الإنذارات") == normalize("الانذارات")
"""

import re

DIACRITICS = re.compile(r"[ً-ْـ]")
HAMZA_CARRIERS = re.compile(r"[إأآاؤئ]")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text: str) -> str:
    text = DIACRITICS.sub("", text).translate(ARABIC_DIGITS)
    text = HAMZA_CARRIERS.sub("ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")
