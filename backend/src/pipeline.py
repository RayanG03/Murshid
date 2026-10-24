"""Question-answering pipeline: retrieval -> context assembly -> generation."""

import logging
import time
from dataclasses import dataclass

from src.config import get_settings, get_university_gates
from src.context_assembly import (
    AssembledContext,
    _doc_line,
    _qa_line,
    assemble_context,
    render_context,
)
from src.generators import (
    BaseGenerator,
    generate_with_meta,
    get_generation_chain,
)
from src.ksu_gate import KSU_QA_SOURCE, RerankerLike, rerank_and_gate
from src.prompts import build_system_prompt, build_user_prompt
from src.query_rewrite import Turn, rewrite_question
from src.retrieval import Retrieved, Retriever

logger = logging.getLogger("capstone.pipeline")

CANNOT_CONFIRM = "لا يمكنني تأكيد الإجابة من المصادر المتاحة."


@dataclass
class AnswerResult:
    """One answered question with its assembled evidence.

    provider/model/truncated come from the generation result (refusals
    never call an LLM, so provider is "none" there).
    """

    question: str
    answer: str
    context_text: str
    qa_items: list[Retrieved]
    doc_items: list[Retrieved]
    refused: bool
    provider: str = "none"
    model: str = ""
    truncated: bool = False
    # Stage timings (seconds) for API logging; 0.0 when the stage did not
    # run (refusals skip generation, KAU has no rerank stage, questions
    # without history skip the rewrite).
    rewrite_seconds: float = 0.0
    search_seconds: float = 0.0
    rerank_seconds: float = 0.0
    generation_seconds: float = 0.0


def answer_question(
    question: str,
    retriever: Retriever,
    generator: BaseGenerator | list[BaseGenerator] | None,
    university: str = "kau",
    reranker: RerankerLike | None = None,
    deadline: float | None = None,
    history: list[Turn] | None = None,
) -> AnswerResult:
    """Answer a question from retrieval evidence; refuse when no context.

    When nothing clears the relevance floors the refusal is returned
    deterministically without calling any LLM (never guess, per AGENTS.md).
    ``generator`` is one provider generator or a failover chain (list of
    BaseGenerator); None means the DEFAULT failover chain from config
    (groq -> mistral -> gemini). ``university`` decides the searched
    collections, the context gates and every prompt string — all sourced
    from one config entry point each so the three can never disagree for
    the same request. KAU keeps the floor/leader/gap quartet path
    unchanged; KSU instead reranks the top dense pool and refuses below
    the threshold (reranker required). ``deadline`` is the absolute
    time.monotonic() request deadline anchored at /ask start; None
    (offline callers) anchors it at generation start from config.
    ``history`` is the recent (question, answer) turns sent by the
    frontend; when present, the follow-up is first rewritten into a
    standalone search query (src/query_rewrite.py) and RETRIEVAL + RERANK
    run against that query while the answer prompt keeps the user's
    original wording. Rewrite failure degrades to the raw question.
    """
    settings = get_settings()
    search_query, rewrite_seconds = _maybe_rewrite(
        question, generator, history, deadline, settings
    )
    if university == "ksu":
        return _answer_question_ksu(
            question, retriever, generator, reranker, deadline=deadline,
            search_query=search_query, rewrite_seconds=rewrite_seconds,
        )
    gates = get_university_gates(university)
    t0 = time.perf_counter()
    qa_results, doc_results = retriever.retrieve_split(
        search_query, university=university
    )
    search_seconds = time.perf_counter() - t0
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
            rewrite_seconds=rewrite_seconds,
            search_seconds=search_seconds,
        )

    return _generate_answer(
        question, context, generator, university, search_seconds,
        deadline=deadline, rewrite_seconds=rewrite_seconds,
    )


def _maybe_rewrite(
    question: str,
    generator: BaseGenerator | list[BaseGenerator] | None,
    history: list[Turn] | None,
    deadline: float | None,
    settings: object,
) -> tuple[str, float]:
    """Rewrite conversational follow-ups; pass plain questions through.

    Normalizes the generator to a chain exactly like _generate_answer does
    so the rewrite walks the SAME failover chain as the answer. Returns
    (search_query, rewrite_seconds); seconds are 0.0 when no rewrite ran.
    """
    if not history:
        return question, 0.0
    chain = [generator] if isinstance(generator, BaseGenerator) else (
        list(generator) if generator is not None else get_generation_chain()
    )
    return rewrite_question(
        chain, history, question, deadline=deadline,
        settings=settings,  # type: ignore[arg-type]
    )


def _answer_question_ksu(
    question: str,
    retriever: Retriever,
    generator: BaseGenerator | None,
    reranker: RerankerLike | None,
    deadline: float | None = None,
    search_query: str | None = None,
    rewrite_seconds: float = 0.0,
) -> AnswerResult:
    """KSU branch: rerank gate on the merged dense pool, then Q&A-first context.

    The top pool candidates by dense score are reranked (probabilities used
    as received); if the top probability is below the threshold the request
    refuses with CANNOT_CONFIRM before any LLM call. Context is built from
    the reranked items, Q&A items ahead of document items, capped by
    context_max_qa / context_max_docs.
    """
    if reranker is None:
        raise ValueError(
            "university='ksu' requires a reranker (loaded at startup)"
        )
    settings = get_settings()
    gates = get_university_gates("ksu")
    t0 = time.perf_counter()
    qa_results, doc_results = retriever.retrieve_split(
        search_query if search_query is not None else question,
        university="ksu",
    )
    search_seconds = time.perf_counter() - t0
    gate = rerank_and_gate(
        search_query if search_query is not None else question,
        qa_results,
        doc_results,
        reranker,
        pool_size=settings.ksu_rerank_pool_size,
        threshold=gates["rerank_threshold"],
    )
    logger.info(
        "[pipeline] ksu rerank gate (university=ksu, pool=%d, "
        "top=%.4f, decision=%s, rerank_seconds=%.2f): %s",
        gate.pool_size, gate.top_score,
        "pass" if gate.passed else "refuse",
        gate.rerank_seconds, question,
    )
    if not gate.passed:
        return AnswerResult(
            question=question, answer=CANNOT_CONFIRM, context_text="",
            qa_items=[], doc_items=[], refused=True,
            rewrite_seconds=rewrite_seconds,
            search_seconds=search_seconds,
            rerank_seconds=gate.rerank_seconds,
        )
    qa_items = [
        r for r in gate.ranked if r.source_type == KSU_QA_SOURCE
    ][:settings.context_max_qa]
    doc_items = [
        r for r in gate.ranked if r.source_type != KSU_QA_SOURCE
    ][:settings.context_max_docs]
    context = AssembledContext(
        qa_items=qa_items,
        doc_items=doc_items,
        qa_gate="rerank-pass",
        doc_gate="rerank-pass",
    )
    return _generate_answer(
        question, context, generator, "ksu",
        search_seconds, gate.rerank_seconds,
        deadline=deadline, rewrite_seconds=rewrite_seconds,
    )


def _generate_answer(
    question: str,
    context: AssembledContext,
    generator: BaseGenerator | list[BaseGenerator] | None,
    university: str,
    search_seconds: float = 0.0,
    rerank_seconds: float = 0.0,
    deadline: float | None = None,
    rewrite_seconds: float = 0.0,
) -> AnswerResult:
    """Shared generation tail: labeled context -> failover chain -> result.

    The generator argument is normalized to a chain (single generator ->
    [generator]; None -> the default chain from config) and sent through
    generate_with_meta so provider/model/truncated always travel with the
    answer.
    """
    context_text = render_context(context, university)
    chain = [generator] if isinstance(generator, BaseGenerator) else (
        list(generator) if generator is not None else get_generation_chain()
    )

    qa_block = "\n".join(_qa_line(i, university) for i in context.qa_items)
    doc_block = "\n".join(_doc_line(i, university) for i in context.doc_items)
    user_prompt = build_user_prompt(question, qa_block, doc_block, university)
    t0 = time.perf_counter()
    result = generate_with_meta(
        chain,
        build_system_prompt(university),
        user_prompt,
        label=f"answer[{question[:30]}]",
        deadline=deadline,
    )
    generation_seconds = time.perf_counter() - t0
    return AnswerResult(
        question=question, answer=result.text, context_text=context_text,
        qa_items=context.qa_items, doc_items=context.doc_items, refused=False,
        provider=result.provider, model=result.model,
        truncated=result.truncated,
        rewrite_seconds=rewrite_seconds,
        search_seconds=search_seconds, rerank_seconds=rerank_seconds,
        generation_seconds=generation_seconds,
    )