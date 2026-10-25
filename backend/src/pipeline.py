"""Question-answering pipeline: retrieval -> context assembly -> generation."""

import logging
import time
from dataclasses import dataclass, field

from src.config import UNIVERSITY_COLLECTIONS, get_settings, get_university_gates
from src.context_assembly import (
    AssembledContext,
    _doc_line,
    _kau_doc_source,
    _ksu_doc_source,
    _qa_line,
    assemble_context,
    render_context,
)
from src.dialect import normalize_dialect
from src.generators import (
    BaseGenerator,
    generate_with_meta,
    get_generation_chain,
)
from src.grounding import ungrounded_numbers
from src.ksu_gate import RerankerLike, filter_ranked, rerank_and_gate
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
    # Diagnostics for logging / evaluation.
    search_query: str = ""
    gate: str = ""
    gate_top_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    numeric_guard_refused: bool = False


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

    When nothing clears the relevance gates the refusal is returned
    deterministically without calling any LLM (never guess, per AGENTS.md).
    ``generator`` is one provider generator or a failover chain (list of
    BaseGenerator); None means the DEFAULT failover chain from config
    (groq -> mistral -> gemini). KAU uses the calibrated floor/leader/gap
    quartet unless settings.kau_gate_mode == "rerank"; KSU always reranks
    the top dense pool and refuses below the threshold (reranker
    required). ``deadline`` is the absolute time.monotonic() request
    deadline anchored at /ask start. ``history`` triggers the follow-up
    rewrite (src/query_rewrite.py): retrieval + rerank use the standalone
    query, and the answer prompt shows it under the user's wording.
    """
    settings = get_settings()
    search_query, rewrite_seconds = _maybe_rewrite(
        question, generator, history, deadline, settings
    )
    rewritten = search_query != question
    if settings.dialect_normalization:
        search_query = normalize_dialect(search_query)

    result = _answer_for_query(
        question, search_query, retriever, generator, university,
        reranker, deadline, rewrite_seconds,
    )
    # Rewrite topic-drift safety net: a gate refusal on the rewritten query
    # gets one retry with the user's own wording when time allows.
    remaining = None if deadline is None else deadline - time.monotonic()
    if (
        rewritten and result.refused and result.provider == "none"
        and (remaining is None
             or remaining >= settings.rewrite_fallback_min_remaining_s)
    ):
        raw_query = (
            normalize_dialect(question) if settings.dialect_normalization
            else question
        )
        logger.info(
            "[pipeline] rewritten query refused (%r); retrying with the raw "
            "question: %s", search_query[:60], question,
        )
        retry = _answer_for_query(
            question, raw_query, retriever, generator, university,
            reranker, deadline, rewrite_seconds,
        )
        retry.search_seconds += result.search_seconds
        retry.rerank_seconds += result.rerank_seconds
        if not retry.refused:
            retry.gate += " (raw-question retry)"
        result = retry
    return _maybe_fallback(question, university, generator, deadline, result)


def _maybe_fallback(
    question: str,
    university: str,
    generator: BaseGenerator | list[BaseGenerator] | None,
    deadline: float | None,
    result: AnswerResult,
) -> AnswerResult:
    """Labeled web / general-knowledge answer after a refusal (opt-in).

    Qualifies: a GATE refusal (provider "none"), or the answer model itself
    replying with the refusal sentence because the passed context did not
    contain the answer (live 2026-10-25: "الكليات في المزاحمية" — chunks
    mention the branch but not its colleges). Numeric-guard refusals and
    real answers pass through untouched.
    """
    settings = get_settings()
    gate_refused = result.refused and result.provider == "none"
    model_refused = (
        not result.numeric_guard_refused
        and result.answer.strip() == CANNOT_CONFIRM
    )
    if settings.general_fallback_mode == "off" or not (gate_refused or model_refused):
        return result
    from src.general_fallback import try_fallback

    fb = try_fallback(
        question, university, _as_chain(generator), deadline, settings,
        search_query=result.search_query or question,
    )
    if fb is None:
        return result
    result.answer = fb.text
    result.refused = False
    result.provider, result.model = fb.provider, fb.model
    result.sources = fb.sources
    result.generation_seconds += fb.seconds
    result.gate += f" +fallback:{fb.kind}"
    return result


def _answer_for_query(
    question: str,
    search_query: str,
    retriever: Retriever,
    generator: BaseGenerator | list[BaseGenerator] | None,
    university: str,
    reranker: RerankerLike | None,
    deadline: float | None,
    rewrite_seconds: float,
) -> AnswerResult:
    """Gate + generation for one search query (see answer_question)."""
    settings = get_settings()
    use_rerank = university == "ksu" or (
        university == "kau" and settings.kau_gate_mode == "rerank"
    )
    if use_rerank:
        if reranker is None and university == "kau":
            logger.warning(
                "[pipeline] kau_gate_mode=rerank but no reranker; "
                "falling back to the cosine gate"
            )
        else:
            return _answer_question_rerank(
                question, retriever, generator, reranker, university,
                deadline=deadline, search_query=search_query,
                rewrite_seconds=rewrite_seconds,
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
    top = max([r.score for r in qa_results[:1] + doc_results[:1]], default=0.0)
    gate_label = f"qa={context.qa_gate},doc={context.doc_gate}"

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
            search_query=search_query, gate=gate_label, gate_top_score=top,
        )

    result = _generate_answer(
        question, context, generator, university, search_seconds,
        deadline=deadline, rewrite_seconds=rewrite_seconds,
        search_query=search_query,
    )
    result.gate, result.gate_top_score = gate_label, top
    return result


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
    chain = _as_chain(generator)
    return rewrite_question(
        chain, history, question, deadline=deadline,
        settings=settings,  # type: ignore[arg-type]
    )


def _as_chain(
    generator: BaseGenerator | list[BaseGenerator] | None,
) -> list[BaseGenerator]:
    """Single generator -> [generator]; None -> default chain from config."""
    if isinstance(generator, BaseGenerator):
        return [generator]
    return list(generator) if generator is not None else get_generation_chain()


def _rerank_params(university: str) -> tuple[int, float]:
    """(pool_size, threshold) for the rerank gate of a university."""
    settings = get_settings()
    if university == "ksu":
        gates = get_university_gates("ksu")
        return settings.ksu_rerank_pool_size, gates["rerank_threshold"]
    return settings.kau_rerank_pool_size, settings.kau_rerank_refuse_threshold


def _answer_question_rerank(
    question: str,
    retriever: Retriever,
    generator: BaseGenerator | list[BaseGenerator] | None,
    reranker: RerankerLike | None,
    university: str = "ksu",
    deadline: float | None = None,
    search_query: str | None = None,
    rewrite_seconds: float = 0.0,
) -> AnswerResult:
    """Rerank gate on the merged dense pool, then Q&A-first context.

    The top pool candidates by dense score (plus optional keyword
    candidates, settings.hybrid_lexical_extra) are reranked; if the top
    probability is below the threshold the request refuses with
    CANNOT_CONFIRM before any LLM call. Context is built from the reranked
    items that clear the context filter, Q&A items ahead of document
    items, capped by context_max_qa / context_max_docs.
    """
    if reranker is None:
        raise ValueError(
            f"university={university!r} rerank gate requires a reranker "
            "(loaded at startup)"
        )
    settings = get_settings()
    pool_size, threshold = _rerank_params(university)
    query = search_query if search_query is not None else question
    t0 = time.perf_counter()
    qa_results, doc_results = retriever.retrieve_split(query, university=university)
    extra: list[Retrieved] = []
    if settings.hybrid_lexical_extra > 0 and hasattr(retriever, "lexical_split"):
        lex_qa, lex_docs = retriever.lexical_split(
            query, settings.hybrid_lexical_extra, university=university
        )
        extra = lex_qa + lex_docs
    search_seconds = time.perf_counter() - t0
    gate = rerank_and_gate(
        query, qa_results, doc_results, reranker,
        pool_size=pool_size, threshold=threshold, extra_candidates=extra,
    )
    logger.info(
        "[pipeline] rerank gate (university=%s, pool=%d, top=%.4f, "
        "decision=%s, rerank_seconds=%.2f): %s",
        university, gate.pool_size, gate.top_score,
        "pass" if gate.passed else "refuse", gate.rerank_seconds, question,
    )
    remaining = None if deadline is None else deadline - time.monotonic()
    logger.info(
        "[timing] %s search=%.2fs rerank=%.2fs remaining_for_answer=%s",
        university, search_seconds, gate.rerank_seconds,
        "n/a" if remaining is None else f"{remaining:.1f}s",
    )
    gate_label = "rerank-pass" if gate.passed else "rerank-refuse"
    if not gate.passed:
        return AnswerResult(
            question=question, answer=CANNOT_CONFIRM, context_text="",
            qa_items=[], doc_items=[], refused=True,
            rewrite_seconds=rewrite_seconds,
            search_seconds=search_seconds,
            rerank_seconds=gate.rerank_seconds,
            search_query=query, gate=gate_label, gate_top_score=gate.top_score,
        )
    ranked = (
        filter_ranked(
            gate, settings.rerank_context_min_score,
            settings.rerank_context_relative,
        )
        if settings.rerank_context_filter else gate.ranked
    )
    qa_source = UNIVERSITY_COLLECTIONS[university][0]
    context = AssembledContext(
        qa_items=[r for r in ranked if r.source_type == qa_source][
            :settings.context_max_qa
        ],
        doc_items=[r for r in ranked if r.source_type != qa_source][
            :settings.context_max_docs
        ],
        qa_gate=gate_label,
        doc_gate=gate_label,
    )
    result = _generate_answer(
        question, context, generator, university,
        search_seconds, gate.rerank_seconds,
        deadline=deadline, rewrite_seconds=rewrite_seconds,
        search_query=query,
    )
    result.gate, result.gate_top_score = gate_label, gate.top_score
    return result


# Back-compat name (older scripts import the KSU branch directly).
def _answer_question_ksu(question, retriever, generator, reranker, **kwargs):
    """KSU rerank branch (kept for existing callers)."""
    return _answer_question_rerank(
        question, retriever, generator, reranker, "ksu", **kwargs
    )


def context_sources(context: AssembledContext, university: str) -> list[str]:
    """Citation strings for the evidence actually sent to the LLM."""
    out: list[str] = []
    for item in context.qa_items:
        q_id = item.metadata.get("q_id", "?")
        out.append(
            f"سؤال موثّق رقم {q_id}" if university == "kau" else f"سؤال رقم {q_id}"
        )
    for item in context.doc_items:
        md = item.metadata
        out.append(_ksu_doc_source(md) if university == "ksu" else _kau_doc_source(md))
    seen: set[str] = set()
    return [s for s in out if s and not (s in seen or seen.add(s))]


def _generate_answer(
    question: str,
    context: AssembledContext,
    generator: BaseGenerator | list[BaseGenerator] | None,
    university: str,
    search_seconds: float = 0.0,
    rerank_seconds: float = 0.0,
    deadline: float | None = None,
    rewrite_seconds: float = 0.0,
    search_query: str | None = None,
) -> AnswerResult:
    """Shared generation tail: labeled context -> failover chain -> result.

    After generation the numeric guard compares the numbers in the answer
    with the context (settings.numeric_guard_mode).
    """
    settings = get_settings()
    context_text = render_context(context, university)
    chain = _as_chain(generator)

    qa_block = "\n".join(_qa_line(i, university) for i in context.qa_items)
    doc_block = "\n".join(_doc_line(i, university) for i in context.doc_items)
    user_prompt = build_user_prompt(
        question, qa_block, doc_block, university,
        standalone_question=(
            search_query if settings.pass_standalone_question else None
        ),
    )
    t0 = time.perf_counter()
    result = generate_with_meta(
        chain,
        build_system_prompt(university),
        user_prompt,
        label=f"answer[{question[:30]}]",
        deadline=deadline,
    )
    generation_seconds = time.perf_counter() - t0

    answer = result.text
    missing: list[str] = []
    guard_refused = False
    if settings.numeric_guard_mode in ("log", "refuse") and answer.strip() != CANNOT_CONFIRM:
        missing = ungrounded_numbers(
            answer, context_text, f"{question} {search_query or ''}"
        )
        if missing:
            logger.warning(
                "[pipeline] numeric guard: answer numbers not in context %s "
                "(mode=%s): %s", missing, settings.numeric_guard_mode, question,
            )
            if settings.numeric_guard_mode == "refuse":
                answer, guard_refused = CANNOT_CONFIRM, True

    return AnswerResult(
        question=question, answer=answer, context_text=context_text,
        qa_items=context.qa_items, doc_items=context.doc_items,
        refused=guard_refused,
        provider=result.provider, model=result.model,
        truncated=result.truncated,
        rewrite_seconds=rewrite_seconds,
        search_seconds=search_seconds, rerank_seconds=rerank_seconds,
        generation_seconds=generation_seconds,
        search_query=search_query or question,
        sources=[] if guard_refused else context_sources(context, university),
        ungrounded_numbers=missing,
        numeric_guard_refused=guard_refused,
    )
