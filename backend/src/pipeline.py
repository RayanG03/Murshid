"""Question-answering pipeline: retrieval -> context assembly -> generation."""

import logging
from dataclasses import dataclass

from src.config import get_settings, get_university_gates
from src.context_assembly import _doc_line, _qa_line, assemble_context, render_context
from src.generators import BaseGenerator
from src.prompts import build_system_prompt, build_user_prompt
from src.retrieval import Retrieved, Retriever

logger = logging.getLogger("capstone.pipeline")

CANNOT_CONFIRM = "لا يمكنني تأكيد الإجابة من المصادر المتاحة."


@dataclass
class AnswerResult:
    """One answered question with its assembled evidence."""

    question: str
    answer: str
    context_text: str
    qa_items: list[Retrieved]
    doc_items: list[Retrieved]
    refused: bool


def answer_question(
    question: str,
    retriever: Retriever,
    generator: BaseGenerator | None,
    university: str = "kau",
) -> AnswerResult:
    """Answer a question from retrieval evidence; refuse when no context.

    When nothing clears the relevance floors the refusal is returned
    deterministically without calling any LLM (never guess, per AGENTS.md).
    ``university`` decides the searched collections, the context gates and
    every prompt string — all sourced from one config entry point each so
    the three can never disagree for the same request.
    """
    settings = get_settings()
    gates = get_university_gates(university)
    qa_results, doc_results = retriever.retrieve_split(
        question, university=university
    )
    context = assemble_context(
        qa_results,
        doc_results,
        floor_qa=gates["floor_qa"],
        floor_docs=gates["floor_docs"],
        max_qa=settings.context_max_qa,
        max_docs=settings.context_max_docs,
        strong_leader=gates["strong_leader"],
        gap=gates["gap"],
    )
    context_text = render_context(context, university)

    if context.is_empty:
        logger.info(
            "[pipeline] gate refused (university=%s, qa=%s, doc=%s) "
            "-> cannot confirm: %s",
            university, context.qa_gate, context.doc_gate, question,
        )
        return AnswerResult(
            question=question, answer=CANNOT_CONFIRM, context_text="",
            qa_items=[], doc_items=[], refused=True,
        )

    if generator is None:
        raise ValueError("A generator is required when context is non-empty.")

    qa_block = "\n".join(_qa_line(i, university) for i in context.qa_items)
    doc_block = "\n".join(_doc_line(i, university) for i in context.doc_items)
    user_prompt = build_user_prompt(question, qa_block, doc_block, university)
    answer = generator.generate(
        build_system_prompt(university), user_prompt,
        label=f"answer[{question[:30]}]",
    )
    return AnswerResult(
        question=question, answer=answer, context_text=context_text,
        qa_items=context.qa_items, doc_items=context.doc_items, refused=False,
    )
