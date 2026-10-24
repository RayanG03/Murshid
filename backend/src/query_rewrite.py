"""History-aware query rewriting for conversational follow-ups.

A follow-up like "لخّص الإجابة بنقاط" or "هل هناك مواعيد محددة؟" carries no
standalone searchable content: fed to retrieval as-is it matches nothing,
the precision gates correctly refuse, and the user sees CANNOT_CONFIRM.
This module rewrites such follow-ups into standalone search queries using
the recent turns the frontend sends with the request:

    "هل هناك مواعيد محددة؟" + (previous turn about CS admission) ->
    "ما هي مواعيد القبول في كلية علوم الحاسب؟"

Hard safety rules (mirroring the rest of the system):
  - The rewrite goes through the SAME failover chain and the SAME absolute
    request deadline as the answer, additionally capped by
    settings.query_rewrite_max_seconds so it can never starve the answer.
  - ANY failure (GenerationError, empty/garbage output, disabled flag,
    missing history) degrades to the raw question — a follow-up must never
    fail because its rewrite failed.
  - No answer content is produced here; refusals stay deterministic.
"""

import logging
import re
import time
from typing import Sequence

from src.config import Settings, get_settings
from src.generators import GenerationError, generate_with_meta

logger = logging.getLogger("capstone.rewrite")

Turn = tuple[str, str]  # (question, answer)

_SYSTEM_PROMPT = (
    "أنت مُعيد صياغة أسئلة البحث في نظام أسئلة الجامعات السعودية. "
    "مهمتك: تحويل سؤال المتابعة إلى سؤال بحثي عربي مستقل وكامل، "
    "بالاعتماد على سياق المحادثة السابقة. "
    "القواعد: "
    "1) أخرج السؤال المعاد صياغته فقط، دون أي شرح أو مقدمة أو علامات تنصيص. "
    "2) إذا كان سؤال المتابعة مستقلاً بذاته أعده كما هو دون تغيير. "
    "3) إذا كانت آخر إجابة في المحادثة رفضاً أو غير مفيدة، اعتمد على السؤال نفسه. "
    "4) لا تجب على السؤال إطلاقاً، فقط أعد صياغته كسؤال بحثي."
)

_USER_TEMPLATE = (
    "المحادثة السابقة:\n{history_block}\n\n"
    "سؤال المتابعة: {question}\n\n"
    "أعد صياغة سؤال المتابعة كسؤال بحثي مستقل كامل (السؤال فقط):"
)

_TURN_TEMPLATE = "[{n}] المستخدم: {q}\n    المساعد: {a}"

# Leading labels a model might add despite the instruction to output only
# the rewritten query.
_LABEL_PREFIXES = (
    "السؤال:", "الاستعلام:", "السؤال المعاد صياغته:",
    "الاستعلام المعاد صياغته:", "question:", "query:", "search query:",
)

# Refusal-ish openers: the model answered instead of rewriting.
_REFUSAL_PREFIXES = (
    "لا يمكنني", "لا أستطيع", "عذراً", "أعتذر",
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_history(
    raw: object,
    max_turns: int = 2,
    max_chars: int = 1200,
) -> list[Turn]:
    """Validate and clip the `history` field sent by the frontend.

    Accepts only [{"question": str, "answer": str}, ...]; anything else
    (None, wrong types, empty strings) is dropped silently. Keeps the LAST
    `max_turns` turns (the most recent context) and clips both texts.
    """
    if not isinstance(raw, list):
        return []
    turns: list[Turn] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q, a = item.get("question"), item.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        q, a = q.strip(), a.strip()
        if not q or not a:
            continue
        turns.append((q[:max_chars], a[:max_chars]))
    return turns[-max_turns:] if max_turns > 0 else turns


def build_rewrite_prompts(history: Sequence[Turn], question: str) -> tuple[str, str]:
    """Build (system, user) prompts for the rewrite call."""
    lines = [
        _TURN_TEMPLATE.format(n=i + 1, q=q, a=a)
        for i, (q, a) in enumerate(history)
    ]
    user = _USER_TEMPLATE.format(
        history_block="\n".join(lines), question=question
    )
    return _SYSTEM_PROMPT, user


def sanitize_rewrite(text: str, fallback: str, max_chars: int = 300) -> str:
    """Clean a model-produced rewrite; fall back to the raw question when
    the output is empty, a refusal, or absurdly long after clipping."""
    if not isinstance(text, str):
        return fallback
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    for prefix in _LABEL_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    cleaned = cleaned.strip("\"'`«»“”‘’")
    if (
        not cleaned
        or len(cleaned) > max_chars
        or cleaned.startswith(_REFUSAL_PREFIXES)
    ):
        return fallback
    return cleaned


def rewrite_question(
    chain: list,
    history: Sequence[Turn],
    question: str,
    deadline: float | None = None,
    settings: Settings | None = None,
) -> tuple[str, float]:
    """Rewrite a follow-up into a standalone search query.

    Returns (search_query, rewrite_seconds). `deadline` is the absolute
    request deadline anchored at /ask start; the rewrite is additionally
    capped at query_rewrite_max_seconds from NOW so the answer always
    keeps its share. Every failure path returns the raw question.
    """
    s = settings or get_settings()
    if not history or not s.query_rewrite_enabled:
        return question, 0.0

    now = time.monotonic()
    budget_end = now + s.query_rewrite_max_seconds
    effective_deadline = (
        min(deadline, budget_end) if deadline is not None else budget_end
    )
    # The chain refuses to START an attempt unless the remaining time
    # covers per_attempt_timeout_s — with the rewrite capped at 4s and the
    # production per-attempt timeout at 8s, every rewrite would die as
    # skipped_budget. Give the rewrite its own clamped per-attempt timeout
    # slightly BELOW its cap (absolute deadline still wins when lower);
    # the headroom absorbs the microseconds between anchoring the budget
    # and the chain's start check (remaining 4.0 < 4.0 boundary race).
    headroom = min(0.5, s.query_rewrite_max_seconds * 0.1)
    attempt_budget = max(s.query_rewrite_max_seconds - headroom, 0.05)
    rewrite_settings = s.model_copy(update={
        "per_attempt_timeout_s": min(s.per_attempt_timeout_s, attempt_budget),
    })
    t0 = time.perf_counter()
    system, user = build_rewrite_prompts(history, question)
    try:
        result = generate_with_meta(
            chain,
            system,
            user,
            label=f"rewrite[{question[:30]}]",
            settings=rewrite_settings,
            deadline=effective_deadline,
        )
    except GenerationError as exc:
        logger.warning(
            "[rewrite] chain failed for %r (%s); falling back to the "
            "raw question", question[:60], exc,
        )
        return question, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - rewrite must never fail a request
        logger.warning(
            "[rewrite] unexpected error (%s); falling back to the raw "
            "question", type(exc).__name__,
        )
        return question, time.perf_counter() - t0

    rewritten = sanitize_rewrite(
        result.text, question, s.query_rewrite_max_query_chars
    )
    seconds = time.perf_counter() - t0
    if rewritten == question:
        logger.info(
            "[rewrite] kept raw question (%.2fs): %r", seconds, question[:60],
        )
    else:
        logger.info(
            "[rewrite] %r -> %r (%.2fs, provider=%s)",
            question[:60], rewritten[:60], seconds, result.provider,
        )
    return rewritten, seconds
