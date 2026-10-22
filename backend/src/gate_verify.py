"""Stage 4 gate verification: refusal must fire BEFORE any LLM call.

Uses an exploding generator: if the gate wrongly passes context, generation
would be invoked and the run fails loudly. Expected-refusing queries must
return refused=True with the deterministic refusal string.

Run from backend/:
  uv run python -X utf8 -m src.gate_verify
"""

import logging

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.generators import BaseGenerator
from src.pipeline import CANNOT_CONFIRM, answer_question
from src.retrieval import Retriever

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

EXPECT_REFUSE = [
    ("(d) calendar", "متى تبدأ اختبارات الدوري الأول للفصل الدراسي الأول في التقويم الجامعي الجديد؟"),
    ("new: bus timetable", "ما جدول مواعيد الحافلات الجامعية من السكن الجامعي إلى الكليات؟"),
    ("new: diploma admission", "ما شروط القبول في برامج الدبلوم بالجامعة؟"),
    ("new: summer registration", "ما مواعيد التسجيل للفصل الصيفي في التقويم الجامعي؟"),
]

EXPECT_ANSWER = [
    ("(a) strong QA", "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه"),
    ("(b) doc-only contacts", "ما وسائل التواصل مع عمادة شؤون الطلاب والبريد الإلكتروني للتواصل؟"),
    ("(c) housing both sides", "وش شروط السكن الجامعي ومين يقدر يسكن فيه؟"),
]


class ExplodingGenerator(BaseGenerator):
    """Fails the run if the gate ever hands context to generation."""

    def __init__(self) -> None:
        super().__init__("exploding", backoff_base=0.01, max_attempts=1)

    def _invoke(self, system: str, prompt: str) -> str:
        raise AssertionError("LLM was called during a gate-verification run")


def main() -> None:
    """Verify gate decisions for all cases without calling any LLM."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)
    generator = ExplodingGenerator()

    failures = 0
    for label, question in EXPECT_REFUSE:
        r = answer_question(question, retriever, generator)
        ok = r.refused and r.answer == CANNOT_CONFIRM
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} refuse | {label} | refused={r.refused} "
              f"| answer={r.answer[:50]}")

    for label, question in EXPECT_ANSWER:
        from src.generators import GenerationError

        try:
            r = answer_question(question, retriever, generator)
            failures += 1
            print(f"FAIL answer | {label} | gate refused a relevant query "
                  f"(qa={r.qa_items and len(r.qa_items)}, "
                  f"doc={r.doc_items and len(r.doc_items)})")
        except GenerationError as exc:
            if "LLM was called" in str(exc):
                print(f"PASS answer | {label} | context reached generation as expected")
            else:
                raise

    print(f"\n{'ALL GATE CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
