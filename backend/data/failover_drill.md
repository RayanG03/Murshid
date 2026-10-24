# Failover drill (real providers)

## 1-normal-chain

key overrides: none

startup: 44.7s (models loaded once)

### kau (first-cold)
- HTTP 200, 2.74s, generation=1.153s, budget_14s=held
- attempts: [('groq', 'ok', 1.14)]
- stage_seconds: {'search': 1.585, 'rerank': 0.0, 'generation': 1.153, 'total': 2.737}

```
يتطلب حذف المادة بعد انتهاء فترة الحذف والإضافة تقديم طلب اعتذار عن المقرر عبر البوابة، وسيُسجَّل لك علامة **W** في السجل.  
[المصدر: سؤال موثّق رقم 61]
```

### ksu (second)
- HTTP 200, 11.53s, generation=1.917s, budget_14s=held
- attempts: [('groq', 'ok', 1.92)]
- stage_seconds: {'search': 0.407, 'rerank': 9.199, 'generation': 1.917, 'total': 11.524}

```
شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول هي:

1. **المدة الزمنية لتقديم الطلب**  
   - للطالب بنظام الفصول (النظام الفصلي): يجب تقديم طلب الاعتذار قبل بداية فترة الاختبارات النهائية بأربعة أسابيع للفصل الدراسي، وأسبوعين للفصل الصيفي.  
   - للطالب بنظام السنة الدراسية: يجب تقديم الطلب قبل بداية فترة الاختبارات النهائية بأثمان أسابيع.  
   *(المصدر: الطلاب > الاعتذار؛ المادة السادسة والخمسون — صفحة 39)*  

2. **عدد المقررات المسموح بالاعتذار عنها**  
   - في الفصل الدراسي أو الصيفي الواحد: حد أقصى مقررين (نظام الفصول) أو مقرر واحد (حسب بعض النصوص).  
   - في السنة الدراسية الواحدة: حد أقصى ثلاث أو أربع مقررات حسب النظام (نظام السنة الدراسية: ثلاث مقررات؛ نظام الفصول: أربع مقررات طيلة بقاء الطالب في الجامعة).  
   *(المصدر: الطلاب > الاعتذار؛ المادة السادسة والخمسون — صفحة 39؛ المادة الخامسة والخمسون — صفحة 38)*  

3. **الحد الأدنى للعبء الدراسي**  
   - لا يجوز أن يقل عدد الساعات المسجلة في الفصل عن الحد الأدنى (12 ساعة) بعد الاعتذار.  
   *(المصدر: الطلاب > الاعتذار)*  

4. **إجراءات تقديم الطلب**  
   - يُقدَّم الطلب عبر بوابة النظام الأكاديمي خلال الفترة المحددة.  
   - يختار الطالب “حركات أكاديمية” ثم “اعتذار عن مقرر”.  
   - عميد الكلية يتخذ القرار خلال خمسة أيام؛ إذا لم يرد خلال هذه المدة يُنفّذ الاعتذار آليًا.  
   *(المصدر: الطلاب > الاعتذار)*  

5. **النتيجة الأكاديمية**  
   - يُسجَّل للطالب تقدير (ع) أو (W) في المقرر الذي تم الاعتذار عنه.  

هذه الشروط هي ما يُستند إليه في الجامعة للاعتذار عن اختبار المقرر النهائي بعذر مقبول.  

**المصدر:** الطلاب > الاعتذار؛ المادة الخامسة والخمسون — صفحة 38؛ المادة السادسة والخمسون — صفحة 39.
```

### kau (repeat-warm)
- HTTP 200, 0.92s, generation=0.686s, budget_14s=held
- attempts: [('groq', 'ok', 0.69)]
- stage_seconds: {'search': 0.232, 'rerank': 0.0, 'generation': 0.686, 'total': 0.918}

```
يتطلب حذف المادة بعد انتهاء فترة الحذف والإضافة تقديم طلب اعتذار عن المقرر عبر البوابة، وسيُسجل لك علامة **W** في السجل.  
[المصدر: سؤال موثّق رقم 61]
```

### ksu (repeat-warm)
- HTTP 200, 10.14s, generation=2.849s, budget_14s=held
- attempts: [('groq', 'ok', 2.86)]
- stage_seconds: {'search': 0.23, 'rerank': 7.058, 'generation': 2.849, 'total': 10.137}

```
شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول هي:

1. **المدة الزمنية لتقديم الطلب**  
   - طالب النظام الفصلي: يجب تقديم طلب الاعتذار قبل بداية فترة الاختبارات النهائية بأربعة أسابيع.  
   - طالب النظام الصيفي: يجب تقديم الطلب قبل بداية فترة الاختبارات النهائية بأسبوعين.  
   - طالب النظام السنوي: يجب تقديم الطلب قبل بداية فترة الاختبارات النهائية بثمانية أسابيع.  
   (المصدر: الطلاب > الاعتذار؛ اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية — المادة السادسة والخمسون (صفحة 39))

2. **عدد المقررات المسموح بالاعتذار عنها**  
   - في الفصل الدراسي أو الصيفي الواحد: حد أقصى مقررين (نظام فصلي) أو مقرر واحد (نظام صيفي).  
   - في السنة الدراسية الواحدة (نظام سنوي): حد أقصى أربع مقررات.  
   - طوال مسار الطالب في الجامعة لا يجوز أن يتجاوز عدد المقررات المعتذر عنها أربعة مقررات.  
   (المصدر: الطلاب > الاعتذار؛ اللائحة التنظيمية للشؤون الدراسية والأكاديمية للمرحلة الجامعية — المادة السادسة والخمسون (صفحة 39))

3. **الحد الأدنى للعبء الدراسي**  
   - لا يجوز أن يقل عدد الساعات المسجلة في الفصل عن الحد الأدنى (12 ساعة) بعد الاعتذار.  
   (المصدر: الطلاب > الاعتذار)

4. **إجراءات تقديم الطلب**  
   - تقديم الطلب عبر بوابة النظام الأكاديمي خلال الفترة المحددة.  
   - يختار الطالب “اعتذار عن مقرر” ويحدد المقرر.  
   - عميد الكلية يتخذ القرار خلال خمسة أيام؛ إذا لم يرد خلال هذه المدة يُنفّذ الاعتذار آلياً.  
   (المصدر: الطلاب > الاعتذار)

5. **النتيجة الأكاديمية**  
   - يُرصد للطالب تقدير (ع) أو (W) في المقرر المعتذر عنه.  
   (المصدر: الطلاب > الاعتذار)
```

### cold-start comparison (scenario 1)
- first /ask (kau): 2.74s
- second /ask (ksu): 11.53s
- warm repeat /ask (kau): 0.92s
- cold-start penalty over warm repeat: 1.82s

## 2-groq-invalid-key

key overrides: ['GROQ_API_KEY']

startup: 38.6s (models loaded once)

### kau (first-cold)
- HTTP 502, 7.44s, generation=n/a, budget_14s=held
- attempts: [('groq', 'http_401', 0.72), ('mistral', 'http_429', 1.06), ('mistral', 'http_429', 0.74)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[كيف احذف ماده بعد انتهاء فتره ]': groq[http_401] last: _HttpStatusError: HTTP 401: {\"error\":{\"message\":\"Invalid API Key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key; mistral[http_429,http_429] last: _HttpStatusError: HTTP 429: {\"object\":\"error\",\"message\":\"Rate limit exceeded\",\"type\":\"rate_limited\",\"param\":null,\"code\":; gemini[skipped_budget] last: remaining 7.5s < per_attempt 8.0s"}
```

### ksu (second)
- HTTP 502, 22.51s, generation=n/a, budget_14s=EXCEEDED
- attempts: [('mistral', 'http_429', 0.78), ('mistral', 'http_429', 0.66)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[وش شروط الاعتذار عن اختبار الم]': groq[auth_skipped] last: skipped (earlier 401/403); mistral[http_429,http_429] last: _HttpStatusError: HTTP 429: {\"object\":\"error\",\"message\":\"Rate limit exceeded\",\"type\":\"rate_limited\",\"param\":null,\"code\":; gemini[skipped_budget] last: remaining 0.6s < per_attempt 8.0s"}
```

## 3-groq-mistral-invalid

key overrides: ['GROQ_API_KEY', 'MISTRAL_API_KEY']

startup: 32.5s (models loaded once)

### kau (first-cold)
- HTTP 502, 14.47s, generation=n/a, budget_14s=EXCEEDED
- attempts: [('groq', 'http_401', 0.48), ('mistral', 'http_401', 0.61), ('gemini', 'timeout', 8.47)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[كيف احذف ماده بعد انتهاء فتره ]': groq[http_401] last: _HttpStatusError: HTTP 401: {\"error\":{\"message\":\"Invalid API Key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key; mistral[http_401] last: _HttpStatusError: HTTP 401: {\"detail\":\"Invalid API Key\"}\n; gemini[timeout] last: ReadTimeout: The read operation timed out"}
```

### ksu (second)
- HTTP 502, 19.71s, generation=n/a, budget_14s=EXCEEDED
- attempts: [('gemini', 'http_5xx', 6.73)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[وش شروط الاعتذار عن اختبار الم]': groq[auth_skipped] last: skipped (earlier 401/403); mistral[auth_skipped] last: skipped (earlier 401/403); gemini[http_5xx] last: _HttpStatusError: HTTP 503: {\n  \"error\": {\n    \"code\": 503,\n    \"message\": \"This model is currently experiencing high de"}
```

## 4-all-three-invalid

key overrides: ['GEMINI_API_KEY', 'GROQ_API_KEY', 'MISTRAL_API_KEY']

startup: 32.5s (models loaded once)

### kau (first-cold)
- HTTP 502, 7.34s, generation=n/a, budget_14s=held
- attempts: [('groq', 'http_401', 0.48), ('mistral', 'http_401', 0.58), ('gemini', 'http_400', 1.25)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[كيف احذف ماده بعد انتهاء فتره ]': groq[http_401] last: _HttpStatusError: HTTP 401: {\"error\":{\"message\":\"Invalid API Key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key; mistral[http_401] last: _HttpStatusError: HTTP 401: {\"detail\":\"Invalid API Key\"}\n; gemini[http_400] last: _HttpStatusError: HTTP 400: {\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"API key not valid. Please pass a valid API k"}
```

### ksu (second)
- HTTP 502, 22.1s, generation=n/a, budget_14s=EXCEEDED
- attempts: [('gemini', 'http_400', 0.61), ('gemini', 'http_400', 1.28)]
- stage_seconds: None

```
{"error": "all generation providers failed: failover chain exhausted on 'answer[وش شروط الاعتذار عن اختبار الم]': groq[auth_skipped] last: skipped (earlier 401/403); mistral[auth_skipped] last: skipped (earlier 401/403); gemini[http_400,http_400] last: _HttpStatusError: HTTP 400: {\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"API key not valid. Please pass a valid API k"}
```
