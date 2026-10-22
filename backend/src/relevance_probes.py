"""Relevance-gate calibration probes: relevant vs off-topic vs domain-adjacent.

Measures top-3 cosine scores per side (QA vs docs) for three query categories
and prints s1 / s1-s2 / s1-s3 so floor and gap thresholds can be chosen from
data. Run from backend/:

  uv run python -X utf8 -m src.relevance_probes
"""

import logging

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.retrieval import Retriever

logging.basicConfig(level=logging.WARNING)

RELEVANT = [
    "وشو شروط التخصيص",
    "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه",
    "شروط التحويل الداخلي الى كلية الطب",
    "السكن الجامعي وش يوفر",
    "كيف اطلع بطاقه جامعيه بدل فاقد",
    "وش شروط السكن الجامعي ومين يقدر يسكن فيه؟",
    "ما وسائل التواصل مع عمادة شؤون الطلاب والبريد الإلكتروني للتواصل؟",
    "ما الحد الأدنى للمعدل التراكمي للتحويل إلى كلية الحاسبات وتقنية المعلومات؟",
    "وش شروط التحويل الداخلي إلى كلية العمارة والتخطيط؟",
    "كيف أقدم على بدل كتب ومراجع الدراسات العليا؟",
]

OFF_TOPIC = [
    "كيف اسوي كيك الشوكولاتة",
    "ما هو سعر البيتكوين اليوم",
    "How do I train for a marathon?",
    "افضل مطاعم في الرياض",
    "اشرح خوارزمية الفرز السريع",
    "ما عاصمة اليابان",
]

# Same domain, but the corpus contains no correct answer for these
# (calendar/admission-guide PDFs deferred, bus table is an image,
# housing/health detail pages not captured, international fees absent).
DOMAIN_ADJACENT_WRONG = [
    "متى تبدأ اختبارات الدوري الأول للفصل الدراسي الأول في التقويم الجامعي الجديد؟",
    "ما آخر يوم للتسجيل في مقررات الفصل الدراسي الثاني حسب التقويم الجامعي؟",
    "متى إجازة نهاية الفصل الدراسي الثاني في التقويم الجامعي؟",
    "ما مواعيد التسجيل للفصل الصيفي في التقويم الجامعي؟",
    "ما النسبة الموزونة المطلوبة للقبول في كلية الحاسبات وتقنية المعلومات للطلبة الجدد؟",
    "ما شروط القبول في برامج الدبلوم بالجامعة؟",
    "متى يفتح باب التحويل الخارجي للجامعة ومتى آخر موعد للتقديم؟",
    "كيف تحسب النسبة الموزونة للقبول بالجامعة للثانوية والقدرات؟",
    "ما جدول مواعيد الحافلات الجامعية من السكن الجامعي إلى الكليات؟",
    "كم رسوم السكن الجامعي الشهرية وكيف أدفعها؟",
    "ما خدمات العيادة الجامعية ومواعيد عملها؟",
    "كم رسوم الدراسة للطلاب الدوليين في الجامعة؟",
]


def spreads(results: list) -> tuple[float, float, float]:
    """Return (s1, s1-s2, s1-s3) with missing ranks treated as score 0."""
    scores = [r.score for r in results] + [0.0, 0.0, 0.0]
    s1 = scores[0]
    return s1, s1 - scores[1], s1 - scores[2]


def main() -> None:
    """Print per-category score/gap distributions for both retrieval sides."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)

    for category, queries in [
        ("RELEVANT", RELEVANT),
        ("OFF-TOPIC", OFF_TOPIC),
        ("DOMAIN-ADJACENT-WRONG", DOMAIN_ADJACENT_WRONG),
    ]:
        print(f"\n=== {category} ===")
        qa_s1, doc_s1, qa_d3, doc_d3 = [], [], [], []
        for q in queries:
            qa, docs = retriever.retrieve_split(q)
            qs1, qd2, qd3 = spreads(qa)
            ds1, dd2, dd3 = spreads(docs)
            qa_s1.append(qs1)
            doc_s1.append(ds1)
            qa_d3.append(qd3)
            doc_d3.append(dd3)
            print(f"qa {qs1:.3f} (d2 {qd2:+.3f} d3 {qd3:+.3f}) | "
                  f"doc {ds1:.3f} (d2 {dd2:+.3f} d3 {dd3:+.3f}) | {q[:60]}")
        print(f"-- qa  s1 range: {min(qa_s1):.3f}..{max(qa_s1):.3f}; "
              f"s1-s3 range: {min(qa_d3):+.3f}..{max(qa_d3):+.3f}")
        print(f"-- doc s1 range: {min(doc_s1):.3f}..{max(doc_s1):.3f}; "
              f"s1-s3 range: {min(doc_d3):+.3f}..{max(doc_d3):+.3f}")


if __name__ == "__main__":
    main()
