"""KSU end-to-end smoke: 5 questions through the full KSU pipeline.

Answers (Arabic) go to data/e2e_ksu.md (UTF-8); the console shows
ASCII progress lines only.

run:  uv run python -X utf8 -m src.ksu_e2e_smoke --provider default
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.generators import PROVIDERS, get_generator
from src.pipeline import answer_question
from src.retrieval import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)

QUESTIONS: tuple[tuple[str, str], ...] = (
    ("q1", "وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟"),
    ("q2", "وش نظام نقاط الخروج في البرامج الدراسية؟"),
    ("q3", "وش الفرق بين الدرجة المزدوجة والدرجة المشتركة؟"),
    ("q4", "كيف احصل على البطاقة الجامعية الجديدة؟"),
    ("q5", "وش اجراءات معادلة المواد للمتحول من داخل الجامعة؟"),
)
OUT_NAME = "e2e_ksu.md"


def main() -> None:
    """Answer 5 KSU questions end to end; answers only into the md file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=(*PROVIDERS, "default"), default="default"
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)
    generator = (
        None if args.provider == "default"
        else get_generator(args.provider, settings)
    )
    label = args.provider if args.provider != "default" else (
        "default chain: " + " -> ".join(settings.default_generation_providers)
    )

    lines = ["# KSU end-to-end smoke", "", f"provider: {label}", ""]
    t_start = time.perf_counter()
    for qi, question in QUESTIONS:
        t0 = time.perf_counter()
        result = answer_question(
            question, retriever, generator, university="ksu",
            reranker=reranker,
        )
        seconds = time.perf_counter() - t0
        print(
            f"[{qi}] refused={result.refused} provider={result.provider} "
            f"truncated={result.truncated} seconds={seconds:.1f}"
        )
        lines += [
            f"## {qi}",
            "",
            f"question: {question}",
            "",
            f"refused: {result.refused}",
            f"provider: {result.provider}",
            f"model: {result.model}",
            f"truncated: {result.truncated}",
            "",
            "context:",
            "```",
            result.context_text or "<EMPTY>",
            "```",
            "",
            "answer:",
            "```",
            result.answer,
            "```",
            "",
        ]
    out_path = Path(settings.data_dir) / OUT_NAME
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"5 questions in {time.perf_counter() - t_start:.1f}s "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
