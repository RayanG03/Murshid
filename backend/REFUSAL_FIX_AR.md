# تجربة تقليل الامتناع مع تشخيص الدليل

السجل يثبت أن 16 من الامتناعات الـ21 وقعت قبل DeepSeek، و5 وقعت بعد اجتياز بوابة الاسترجاع. لا توجد أخطاء استشهاد مسجلة في هذه الحالات. التفاصيل وأرقام الطلبات في data/refusal_diagnosis.

هذه حزمة تشخيص وتجربة استرجاع، وليست إثباتاً أن الامتناعات حُلّت. لا يغيّر نسخ الملفات وحده إعداد البحث. لا توجد مفاتيح API أو قواعد بيانات أو مصادر معدلة في الحزمة، ولا تحتاج إلى إعادة الفهرسة.

## التثبيت

1. أوقف الباك إند باستخدام Ctrl+C.
2. احتفظ بنسخة من backend/src الحالي للرجوع لها.
3. انسخ مجلد backend الموجود في الحزمة فوق backend في مشروع Murshid، ووافق على استبدال الملفات الموجودة. انسخ المجلد بالكامل ليصل الملف الجديد src/retrieval_diagnostics.py. لا تستبدل .env ولا تحذف flat_index.
4. افتح PowerShell ونفذ:

```powershell
cd "C:\Users\Moe\Desktop\Murshid\backend"
uv run python apply_refusal_profile.py --profile hybrid
```

الأمر يغيّر إعدادات الاسترجاع والتشخيص والكاش فقط، ويترك مفاتيح المزود وVECTOR_BACKEND وEVIDENCE_MODE وسائر إعداداتك كما هي. يحفظ نسخة كاملة من .env داخل data/config_backups، وهو مجلد بيانات محلي. احتفظ بالنسخة محلياً لأنها تحتوي مفتاحك، ولا ترسلها ضمن نتائج الاختبار.

التجربة تستخدم RETRIEVAL_MODE=hybrid وHYBRID_LEXICAL_EXTRA=4 وRETRIEVAL_TOP_K=8 ومجموعة ترتيب من 8 لكل جامعة. يبقى حد الرفض الحالي دون تعديل؛ لا يُسمح بمرور مرشح لمجرد أنه ظهر في بحث الكلمات. يوقف الكاش مؤقتاً حتى لا يعاد استخدام أجوبة تشغيل سابق، ويفعل تسجيل التشخيص. لم نرفع عدد مقاطع السياق ولم نغير تعليمات الإجابة في هذه التجربة لفصل أثر الاسترجاع.

5. تحقق من الإعدادات الفعلية دون إظهار المفتاح:

```powershell
uv run python -c "from src.config import get_settings; s=get_settings(); print(s.vector_backend, s.evidence_mode, s.retrieval_mode, s.retrieval_top_k, s.hybrid_lexical_extra, s.retrieval_diagnostics); print(s.ksu_rerank_refuse_threshold, s.kau_rerank_refuse_threshold)"
```

المتوقع flat documents_only hybrid 8 4 True. يفترض أن تبقى عتبتا الرفض عند 0.4 إن لم تكن عدلتهما سابقاً. إن ظهر إعداد مختلف، فتذكر أن متغيرات البيئة في نافذة PowerShell تتقدم على .env.

6. شغل السيرفر واترك النافذة مفتوحة:

```powershell
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

7. في نافذة ثانية أعد تشغيل الاختبار نفسه:

```powershell
cd "C:\Users\Moe\Desktop\Murshid\backend"
uv run python ..\benchmark100\run_benchmark.py
```

أرسل results.csv الجديد وbackend/data/logs/ask.jsonl. الملف الأخير يحتوي نص الأسئلة ومقاطع المصادر والرد الخام عند تفعيل التشخيص؛ لا ترسل .env أو مجلد config_backups. نطابق الطلبات بواسطة request_id، فلا يلزم حذف السجل القديم.

قارن الـ100 سؤال كلها، بما فيها أسئلة الامتناع العشرة والأجوبة التي كانت صحيحة، وسجل أخطاء الخدمة والزمن أيضاً. وجود نص مصدر وحده لا يثبت صحة الحكم، وزيادة عدد الإجابات وحدها ليست تحسناً. الاختبارات المحلية تتحقق من منطق التعديل، ولا تقيس دقة النموذج الحقيقي أو أثر التجربة على هذه الأسئلة.

## الرجوع

للمقارنة مع إعدادات الاسترجاع السابقة مع إبقاء التشخيص مفعلاً:

```powershell
uv run python apply_refusal_profile.py --profile baseline
```

هذا يحدد dense، وtop_k=5، ومجموعة ترتيب 6، وبدون إضافات كلمات. لا يعيد باقي إعداداتك إلى نسخة قديمة، ولا يغير عتبة الرفض. أعد تشغيل السيرفر بعد أي تغيير.

لإعادة ملف .env كما كان تماماً قبل التجربة، استخدم مسار النسخة الذي طبعه الأمر الأول:

```powershell
uv run python apply_refusal_profile.py --restore "مسار نسخة الإعدادات الذي ظهر لك"
```

بعد انتهاء التشخيص يمكن جعل RETRIEVAL_DIAGNOSTICS=false وإعادة قيمة ANSWER_CACHE_SIZE المرغوبة، ثم إعادة تشغيل الخادم.
