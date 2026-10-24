"""Stage 4 gate verification: refusal must fire BEFORE any LLM call.

Uses an exploding generator: if the gate wrongly passes context, generation
would be invoked and the run fails loudly. Expected-refusing queries must
return refused=True with the deterministic refusal string.

Probe lists updated for the Mistral-OCR corpus (all KAU/KSU PDFs ingested
clean): the admission guide now answers the diploma-admission question, so
that probe moved from EXPECT_REFUSE to EXPECT_ANSWER. The refuse loop no
longer crashes when a probe becomes answerable — it prints FAIL and moves
on, so future corpus improvements degrade to a readable report.

Run from backend/:
  uv run python -X utf8 -m src.gate_verify
"""

import logging

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.generators import BaseGenerator, GenerationError
from src.pipeline import CANNOT_CONFIRM, answer_question
from src.retrieval import Retriever

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

EXPECT_REFUSE = [
    # Off-topic: must be refused BEFORE generation (below-floor on both sides).
    ("off-topic: cake", "كيف اسوي كيك الشوكولاتة"),
    ("off-topic: marathon", "How do I train for a marathon?"),
]

EXPECT_ANSWER = [
    # --- gate passes context; generation decides the final answer ---
    # Calendar answers live in تقويم 26-27 (OCR'd + ingested). Gate passes
    # since the 2026-10-25 recalibration (leader 0.832 / gap 0.012).
    ("(d) calendar", "متى تبدأ اختبارات الدوري الأول للفصل الدراسي الأول في التقويم الجامعي الجديد؟"),
    ("new: summer registration", "ما مواعيد التسجيل للفصل الصيفي في التقويم الجامعي؟"),
    # Admission guide (دليل القبول) ingested clean — gate passes, gen answers.
    ("new: diploma admission", "ما شروط القبول في برامج الدبلوم بالجامعة؟"),
    ("(a) strong QA", "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه"),
    ("(b) doc-only contacts", "ما وسائل التواصل مع عمادة شؤون الطلاب والبريد الإلكتروني للتواصل؟"),
    ("(c) housing both sides", "وش شروط السكن الجامعي ومين يقدر يسكن فيه؟"),
    # BUS: gate passes it (bus-service chunks ~0.867 leader) but the corpus
    # does NOT contain the timetable (image in PDF). The exploding stub here
    # cannot test grounding, so this probe only asserts "context reaches
    # generation". The REAL refusal is UI-verified (see runbook).
    ("new: bus timetable", "ما جدول مواعيد الحافلات الجامعية من السكن الجامعي إلى الكليات؟"),
]


class ExplodingGenerator(BaseGenerator):
    """Fails the run if the gate ever hands context to generation."""

    def __init__(self) -> None:
        super().__init__("exploding", backoff_base=0.01, max_attempts=1)

    def _invoke_once(
        self, system: str, prompt: str, timeout_seconds: float
    ) -> tuple[str, bool]:
        raise AssertionError("LLM was called during a gate-verification run")


def _is_exploded(exc: GenerationError) -> bool:
    """True when the failure came from the exploding generator itself."""
    return "LLM was called" in str(exc)


def main() -> None:
    """Verify gate decisions for all cases without calling any LLM."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)
    generator = ExplodingGenerator()

    failures = 0
    for label, question in EXPECT_REFUSE:
        try:
            r = answer_question(question, retriever, generator)
        except GenerationError as exc:
            if _is_exploded(exc):
                failures += 1
                print(f"FAIL refuse | {label} | gate PASSED a refuse-probe "
                      "(corpus now answers it) — move it to EXPECT_ANSWER")
                continue
            raise
        ok = r.refused and r.answer == CANNOT_CONFIRM
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} refuse | {label} | refused={r.refused} "
              f"| answer={r.answer[:50]}")

    for label, question in EXPECT_ANSWER:
        try:
            r = answer_question(question, retriever, generator)
            failures += 1
            print(f"FAIL answer | {label} | gate refused a relevant query "
                  f"(qa={r.qa_items and len(r.qa_items)}, "
                  f"doc={r.doc_items and len(r.doc_items)})")
        except GenerationError as exc:
            if _is_exploded(exc):
                print(f"PASS answer | {label} | context reached generation as expected")
            else:
                raise

    print(f"\n{'ALL GATE CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
