# ربط أسئلة التقييم بمقاطع التقويم الجديدة (2026-10-10)

صار التقويم مقطعاً لكل حدث (`chunk.py`): مقطع الفصل `…_sec_<بصمة>` صار أسطراً `…_sec_<بصمة>_rNN`،
وما في الجدول من غير الأحداث (مثل «بداية الفصل الدراسي الأول - الأحد ٢٠٢٦/٠٨/٢٣م») بقي في المعرّف القديم.

القاعدة: كل مقطع فصل في الجواب المعتمد يُستبدل بأسطر الأحداث منه التي فيها تاريخ مذكور في الجواب المتوقع
(بأرقام التاريخ، أياً كان ترتيبها أو نوع أرقامها). أربعة أسئلة طلاب جوابها المتوقع بأسماء الشهور
(«23 أغسطس») فاختيرت أسطرها يدوياً بالتاريخ نفسه. لم يُضف تقويم لم يكن في الجواب المعتمد قبل،
فالتقويم الثاني لـ KAU يذكر بعض التواريخ نفسها ولا يُعدّ صحيحاً.

| الملف | السؤال | قبل | بعد |
|---|---|---|---|
| test_questions_50.jsonl | t01 | ksu_taqwim_sec_03f386c9، ksu_harakat_sec_f44bd9d7 | ksu_taqwim_sec_03f386c9_r01، ksu_harakat_sec_f44bd9d7 |
| test_questions_50.jsonl | t02 | ksu_taqwim_sec_03f386c9، ksu_harakat_sec_f44bd9d7 | ksu_taqwim_sec_03f386c9_r07، ksu_taqwim_sec_03f386c9_r08، ksu_harakat_sec_f44bd9d7_r18 |
| test_questions_50.jsonl | t03 | ksu_harakat_sec_2d63285d، ksu_taqwim_sec_210f2c7c | ksu_harakat_sec_2d63285d_r11، ksu_harakat_sec_2d63285d_r12، ksu_taqwim_sec_210f2c7c_r02، ksu_taqwim_sec_210f2c7c_r03 |
| test_questions_50.jsonl | t04 | ksu_harakat_sec_f44bd9d7، ksu_shuun_art056 | ksu_harakat_sec_f44bd9d7_r10، ksu_harakat_sec_f44bd9d7_r11، ksu_harakat_sec_f44bd9d7_r16، ksu_shuun_art056 |
| test_questions_50.jsonl | t05 | ksu_harakat_sec_2d63285d | ksu_harakat_sec_2d63285d_r09 |
| test_questions_50.jsonl | t06 | ksu_taqwim_sec_9813db3d، ksu_harakat_sec_fb4e991e | ksu_taqwim_sec_9813db3d_r01، ksu_taqwim_sec_9813db3d_r02، ksu_taqwim_sec_9813db3d_r03، ksu_harakat_sec_fb4e991e_r01، ksu_harakat_sec_fb4e991e_r02، ksu_harakat_sec_fb4e991e_r06، ksu_harakat_sec_fb4e991e |
| test_questions_50.jsonl | t07 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r01 |
| test_questions_50.jsonl | t08 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r11 |
| test_questions_50.jsonl | t09 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r10 |
| test_questions_50.jsonl | t10 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r01، kau_taqwim_sec_13052e22_r07، kau_taqwim_sec_13052e22_r17، kau_taqwim_sec_13052e22_r18، kau_taqwim_sec_13052e22_r19 |
| test_questions_50.jsonl | r04 | kau_taqwim_sana_sec_ba7d35eb | kau_taqwim_sana_sec_ba7d35eb_r07، kau_taqwim_sana_sec_ba7d35eb_r08 |
| test_questions_new50.jsonl | n01 | ksu_harakat_sec_f44bd9d7 | ksu_harakat_sec_f44bd9d7_r14، ksu_harakat_sec_f44bd9d7_r15 |
| test_questions_new50.jsonl | n02 | ksu_harakat_sec_f44bd9d7 | ksu_harakat_sec_f44bd9d7_r01، ksu_harakat_sec_f44bd9d7_r02، ksu_harakat_sec_f44bd9d7_r04، ksu_harakat_sec_f44bd9d7_r05، ksu_harakat_sec_f44bd9d7_r06 |
| test_questions_new50.jsonl | n03 | ksu_harakat_sec_2d63285d | ksu_harakat_sec_2d63285d_r13 |
| test_questions_new50.jsonl | n04 | ksu_taqwim_sec_03f386c9 | ksu_taqwim_sec_03f386c9_r04، ksu_taqwim_sec_03f386c9_r05 |
| test_questions_new50.jsonl | m01 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r09 |
| test_questions_new50.jsonl | m02 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r16 |
| test_questions_new50.jsonl | m03 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r08، kau_taqwim_sec_13052e22_r14، kau_taqwim_sec_13052e22_r15 |
| test_questions_new50.jsonl | m04 | kau_taqwim_sec_13052e22 | kau_taqwim_sec_13052e22_r13 |
| student_questions.jsonl | student_07 (يدوي) | ksu_taqwim_sec_03f386c9 | ksu_harakat_sec_f44bd9d7، ksu_taqwim_sec_03f386c9_r01 |
| student_questions.jsonl | student_08 (يدوي) | ksu_taqwim_sec_03f386c9 | ksu_harakat_sec_f44bd9d7_r12، ksu_harakat_sec_f44bd9d7_r13، ksu_taqwim_sec_03f386c9_r02، ksu_taqwim_sec_03f386c9_r03 |
| student_questions.jsonl | student_09 (يدوي) | ksu_harakat_sec_2d63285d | ksu_harakat_sec_fb4e991e، ksu_harakat_sec_fb4e991e_r01، ksu_harakat_sec_fb4e991e_r02، ksu_harakat_sec_fb4e991e_r06، ksu_taqwim_sec_9813db3d_r01، ksu_taqwim_sec_9813db3d_r02، ksu_taqwim_sec_9813db3d_r03 |
| student_questions.jsonl | student_10 (يدوي) | ksu_harakat_sec_f44bd9d7 | ksu_harakat_sec_f44bd9d7_r02 |
| fresh50.jsonl | f13 | ksu_harakat_sec_f44bd9d7 | ksu_harakat_sec_f44bd9d7_r14، ksu_harakat_sec_f44bd9d7_r15 |
| fresh50.jsonl | f15 | ksu_taqwim_sec_cc8ca341 | ksu_taqwim_sec_cc8ca341_r01 |
| fresh50.jsonl | f39 | kau_taqwim_sana_sec_20922d1f، kau_taqwim_sec_13052e22 | kau_taqwim_sana_sec_20922d1f_r03، kau_taqwim_sana_sec_20922d1f_r04، kau_taqwim_sec_13052e22_r08، kau_taqwim_sec_13052e22_r14، kau_taqwim_sec_13052e22_r15 |

26 سؤالاً (test_questions_50.jsonl: 11، test_questions_new50.jsonl: 8، student_questions.jsonl: 4، fresh50.jsonl: 3). fresh50 رُبط ولم يُشغَّل.
