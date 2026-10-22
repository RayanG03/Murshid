"""Stage 4 smoke runner: the four test cases (a)-(d) with full context shown.

Run from backend/:
  uv run python -X utf8 -m src.generation_smoke --provider gemini
"""

import argparse
import logging

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.generators import PROVIDERS, get_generator
from src.pipeline import answer_question
from src.retrieval import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)

CASES = [
    ("(a) strong QA match", "كيف احذف ماده بعد انتهاء فتره الحذف والاضافه"),
    ("(b) strong doc-only match", "ما وسائل التواصل مع عمادة شؤون الطلاب والبريد الإلكتروني للتواصل؟"),
    ("(c) both sides", "وش شروط السكن الجامعي ومين يقدر يسكن فيه؟"),
    ("(d) known gap (calendar)", "متى تبدأ اختبارات الدوري الأول للفصل الدراسي الأول في التقويم الجامعي الجديد؟"),
]


def main() -> None:
    """Run all four cases and print context + answer for each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    args = parser.parse_args()

    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)
    generator = get_generator(args.provider, settings)

    for label, question in CASES:
        result = answer_question(question, retriever, generator)
        print(f"\n{'=' * 70}\nCASE {label}\nQ: {question}")
        qa_scores = [f"{r.score:.3f}" for r in result.qa_items]
        doc_scores = [f"{r.score:.3f}" for r in result.doc_items]
        print(f"qa_scores={qa_scores}  doc_scores={doc_scores}  "
              f"refused={result.refused}")
        print(f"--- assembled context ---\n{result.context_text or '<EMPTY>'}")
        print(f"--- answer ({args.provider}) ---\n{result.answer}")


if __name__ == "__main__":
    main()
