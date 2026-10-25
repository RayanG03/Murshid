"""Opt-in fallback when the official corpus cannot answer a question.

The core design is unchanged: answers come from the ingested official
documents and anything the gates cannot back is refused. This module only
runs AFTER a gate refusal and only when settings.general_fallback_mode is
not "off". Every fallback answer is clearly labeled so a student can never
mistake it for a sourced answer:

- "web": Gemini with the Google Search tool, instructed to use the
  university's official domain only; the answer is accepted only when at
  least one grounding source is on that domain (kau.edu.sa / ksu.edu.sa),
  and those links are shown.
- "llm": the model's own general knowledge, ONLY for general questions
  about the university (not regulations/fees/procedures, not off-topic),
  prefixed with a "not verified, may be outdated" warning.
- "web_then_llm": web first, llm when the web found nothing.

NOTE: "web" makes live requests at answer time, which AGENTS.md's "never
fetch" rule did not foresee; keep it off unless the team agreed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import UNIVERSITY_DISPLAY_NAMES, Settings, get_settings
from src.generators import BaseGenerator, GenerationError, generate_with_meta
from src.grounding import ungrounded_numbers

logger = logging.getLogger("capstone.fallback")

MODES = ("off", "web", "llm", "web_then_llm")
NO_ANSWER = "NO_ANSWER"

OFFICIAL_DOMAINS: dict[str, tuple[str, ...]] = {
    "kau": ("kau.edu.sa",),
    "ksu": ("ksu.edu.sa",),
}

WEB_PREFIX = (
    "🔎 لم أجد الإجابة في المصادر المعتمدة في النظام، وهذه نتيجة بحث مباشر "
    "في الموقع الرسمي لـ{university} (تحقق من الرابط):"
)
LLM_PREFIX = (
    "⚠️ معلومة عامة من معرفة النموذج، وليست من المصادر الرسمية المعتمدة، "
    "وقد تكون قديمة أو غير دقيقة:"
)

_WEB_SYSTEM = (
    "أنت مساعد يجيب عن أسئلة الطلاب حول {university}. ابحث في الإنترنت مقتصراً "
    "على الموقع الرسمي للجامعة (site:{domain}) وأجب بالعربية باختصار اعتماداً "
    "على نتائج هذا الموقع فقط. إذا لم تجد الإجابة في صفحات {domain}، أو كان "
    "السؤال لا علاقة له بالجامعة، اكتب فقط: NO_ANSWER. لا تختلق أرقاماً أو أسماء."
)

_LLM_SYSTEM = (
    "أنت مساعد جامعي. سؤال الطالب عن {university} لم تُوجد إجابته في "
    "المصادر الرسمية المعتمدة. اتبع القواعد بدقة:\n"
    "1. إذا كان السؤال عن لوائح أو أنظمة أو إجراءات أكاديمية أو مالية (مثل "
    "المعدلات، الإنذار، الحرمان، التسجيل، الحذف والإضافة، المكافآت، الرسوم، "
    "شروط القبول أو التحويل، المواعيد)، اكتب فقط: NO_ANSWER\n"
    "2. إذا كان السؤال لا علاقة له بالجامعة، اكتب فقط: NO_ANSWER\n"
    "3. غير ذلك (معلومات عامة عن الجامعة مثل تاريخها أو موقعها أو قيادتها)، "
    "أجب بجملة أو جملتين من معرفتك العامة، وإذا لم تكن متأكداً اكتب فقط: NO_ANSWER\n"
    "4. اكتب نصاً عادياً دون رموز رياضية."
)


_PAGES_SYSTEM = (
    "أنت مساعد يجيب عن سؤال طالب حول {university} اعتماداً فقط على "
    "المقتطفات التالية من الموقع الرسمي للجامعة. أجب بالعربية باختصار، "
    "وانقل الأرقام والتواريخ كما وردت حرفياً. إذا لم تحتوِ المقتطفات على "
    "الإجابة، أو كان السؤال لا علاقة له بالجامعة، اكتب فقط: NO_ANSWER. "
    "إذا كانت المقتطفات تغطي جزءاً من الإجابة فقط (مثلاً قائمة ناقصة)، "
    "فاذكر ما وجدته ثم قل صراحة إن القائمة قد لا تكون كاملة وإن التفاصيل "
    "في الرابط. "
    "إذا ظهرت معلومات لأكثر من عام دراسي، اعتمد الأحدث فقط واذكر العام "
    "الدراسي في الإجابة. اكتب نصاً عادياً دون رموز رياضية. في سطر أخير "
    "منفصل اكتب أرقام المقتطفات التي أخذت منها الإجابة بهذا الشكل: "
    "المقتطفات: 1"
)

_USED_RE = re.compile(r"^\s*المقتطفات\s*[:：]\s*([0-9٠-٩،,\s]+)\s*$", re.M)


def split_used_snippets(text: str, count: int) -> tuple[str, list[int]]:
    """Remove the trailing 'المقتطفات: 1, 2' line; return (text, 1-based ids).

    Invalid or missing ids -> [] (the caller then lists every page)."""
    match = None
    for match in _USED_RE.finditer(text):
        pass
    if match is None:
        return text.strip(), []
    digits = match.group(1).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    ids = sorted({int(n) for n in re.findall(r"\d+", digits) if 1 <= int(n) <= count})
    cleaned = (text[: match.start()] + text[match.end():]).strip()
    return cleaned, ids


def _provider(settings: Settings) -> str:
    choice = settings.web_search_provider
    if choice == "auto":
        return "tavily" if settings.tavily_api_key else "gemini"
    return choice


def tavily_search(
    question: str, domains: tuple[str, ...], settings: Settings, timeout_s: float,
) -> list[dict]:
    """Official-domain search results [{title, url, text}] (empty on error)."""
    payload = {
        "api_key": settings.tavily_api_key,
        "query": question,
        "include_domains": list(domains),
        "search_depth": settings.tavily_search_depth,
        "max_results": settings.tavily_max_results,
        **({"chunks_per_source": settings.tavily_chunks_per_source}
           if settings.tavily_search_depth == "advanced" else {}),
        "include_raw_content": True,
        "include_answer": False,
    }
    try:
        resp = httpx.post(
            "https://api.tavily.com/search", json=payload, timeout=timeout_s,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}",
                     "Content-Type": "application/json"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[fallback] tavily failed: %s", type(exc).__name__)
        return []
    from src.usage_tracker import get_tracker

    if resp.status_code != 200:
        get_tracker().record_error("tavily", resp.status_code, str(resp.text)[:200],
                                   getattr(resp, "headers", None))
        logger.warning("[fallback] tavily HTTP %d: %s", resp.status_code,
                       " ".join(str(resp.text).split())[:200])
        return []
    get_tracker().record_response(
        "tavily", settings.tavily_search_depth, getattr(resp, "headers", None),
        credits=2 if settings.tavily_search_depth == "advanced" else 1)
    out = []
    for r in resp.json().get("results", []) or []:
        url = str(r.get("url", ""))
        if not any(d in url.lower() for d in domains):
            continue  # belt and braces: official domain only
        snippet = str(r.get("content") or "").strip()
        raw = str(r.get("raw_content") or "").strip()
        if snippet or raw:
            out.append({"title": str(r.get("title", "")), "url": url,
                        "snippet": snippet,
                        "text": (raw or snippet)[: settings.tavily_max_chars_per_page]})
    return out


def split_passages(text: str, max_chars: int) -> list[str]:
    """Paragraph-ish passages of at most ~max_chars (never empty)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n(?=[-•*#|])", text) if b.strip()]
    out: list[str] = []
    current = ""
    for block in blocks:
        while len(block) > max_chars:  # very long paragraph: hard wrap
            out.append(block[:max_chars])
            block = block[max_chars:]
        if current and len(current) + len(block) + 1 > max_chars:
            out.append(current)
            current = block
        else:
            current = f"{current}\n{block}" if current else block
    if current:
        out.append(current)
    return out


def select_passages(query: str, pages: list[dict], max_chars: int, limit: int) -> list[dict]:
    """Best passages across all pages for the query.

    Tavily's own relevance snippet of each page is always kept (it is the
    search engine's pick); remaining slots go to the highest-BM25 passages
    from the full page texts. Each item keeps its page title and URL.
    """
    from src.lexical import Bm25Index

    chosen: list[dict] = []
    seen: set[str] = set()
    for p in pages:
        if p.get("snippet"):
            chosen.append({"title": p["title"], "url": p["url"], "text": p["snippet"]})
            seen.add(p["snippet"][:200])
    candidates: list[dict] = []
    for p in pages:
        for passage in split_passages(p.get("text", ""), max_chars):
            if passage[:200] not in seen:
                candidates.append({"title": p["title"], "url": p["url"], "text": passage})
    if candidates and len(chosen) < limit:
        index = Bm25Index([c["text"] for c in candidates], [{}] * len(candidates))
        for i, _score in index.search(query, limit - len(chosen)):
            chosen.append(candidates[i])
    return chosen[:limit] if chosen else candidates[:limit]


def answer_from_pages(
    question: str, university: str, pages: list[dict],
    chain: list[BaseGenerator], deadline: float | None, settings: Settings,
) -> FallbackAnswer | None:
    """Answer strictly from official pages; rejects invented numbers."""
    if not pages or not chain:
        return None
    name = UNIVERSITY_DISPLAY_NAMES.get(university, university)
    context = "\n\n".join(
        f"[{i}] {p['title']} — {p['url']}\n{p['text']}" for i, p in enumerate(pages, 1)
    )
    started = time.monotonic()
    try:
        result = generate_with_meta(
            chain, _PAGES_SYSTEM.format(university=name),
            f"السؤال: {question}\n\nمقتطفات من الموقع الرسمي:\n{context}",
            label=f"web[{question[:30]}]", settings=settings, deadline=deadline,
        )
    except GenerationError as exc:
        logger.warning("[fallback] web answer failed: %s", str(exc)[:160])
        return None
    text, used = split_used_snippets(result.text.strip(), len(pages))
    if not text or NO_ANSWER in text:
        logger.info("[fallback] official pages do not answer %r", question[:60])
        return None
    if used:
        pages = [pages[i - 1] for i in used]
    missing = ungrounded_numbers(text, context, question)
    if missing:
        logger.warning("[fallback] web answer has numbers not on the pages %s "
                       "-> rejected", missing)
        return None
    links: list[str] = []
    for p in pages:  # several passages can come from one page
        line = f"{p['title']}: {p['url']}".strip(": ")
        if line not in links:
            links.append(line)
    body = (WEB_PREFIX.format(university=name) + "\n\n" + text
            + "\n\nالمصادر:\n" + "\n".join(links))
    return FallbackAnswer(
        text=body, kind="web", provider=f"tavily+{result.provider}",
        model=result.model, sources=links, seconds=time.monotonic() - started,
    )


@dataclass
class FallbackAnswer:
    """A labeled non-corpus answer."""

    text: str
    kind: str  # "web" | "llm"
    provider: str
    model: str
    sources: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _is_official(chunk_web: dict, domains: tuple[str, ...]) -> bool:
    title = str(chunk_web.get("title", "")).lower()
    uri = str(chunk_web.get("uri", "")).lower()
    return any(title.endswith(d) or d in uri for d in domains)


def parse_search_response(
    data: dict[str, Any], domains: tuple[str, ...],
) -> tuple[str, list[dict]] | None:
    """(answer text, official grounding sources) or None when unusable."""
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
    except (KeyError, IndexError, TypeError):
        return None
    text = "".join(str(p.get("text", "")) for p in parts).strip()
    if not text or NO_ANSWER in text:
        return None
    chunks = candidate.get("groundingMetadata", {}).get("groundingChunks", []) or []
    official = [c["web"] for c in chunks if "web" in c and _is_official(c["web"], domains)]
    if not official:
        return None  # not backed by the official site -> reject
    return text, official


def _error_summary(resp: Any) -> str:
    """One-line Gemini error: message + violated quota metrics (no key)."""
    try:
        err = resp.json().get("error", {})
    except Exception:  # noqa: BLE001
        return " ".join(str(resp.text).split())[:300]
    parts = [" ".join(str(err.get("message", "")).split())[:220]]
    for detail in err.get("details", []) or []:
        for v in detail.get("violations", []) or []:
            metric = v.get("quotaMetric") or v.get("quotaId")
            if metric:
                parts.append(f"quota={metric} value={v.get('quotaValue', '?')}")
        if detail.get("retryDelay"):
            parts.append(f"retry_after={detail['retryDelay']}")
    return " | ".join(p for p in parts if p)


def search_official_site(
    question: str, university: str, settings: Settings, timeout_s: float,
    chain: list[BaseGenerator] | None = None, deadline: float | None = None,
    search_query: str | None = None,
) -> FallbackAnswer | None:
    """Official-website answer via the configured search backend.

    tavily: search restricted to the official domain, answered from the
    page text by the normal provider chain. gemini: Google Search
    grounding, accepted only with an official grounding source.
    """
    domains = OFFICIAL_DOMAINS.get(university, ())
    if not domains:
        return None
    if _provider(settings) == "tavily":
        if not settings.tavily_api_key:
            return None
        name = UNIVERSITY_DISPLAY_NAMES.get(university, university)
        query = f"{search_query or question} {name}"
        pages = tavily_search(query, domains, settings, timeout_s)
        if not pages:
            logger.info("[fallback] tavily: no official pages for %r", question[:60])
            return None
        pages = select_passages(
            query, pages, settings.web_passage_chars, settings.web_context_passages)
        from src.generators import get_generation_chain

        return answer_from_pages(
            question, university, pages,
            chain or get_generation_chain(settings=settings), deadline, settings,
        )
    if not settings.gemini_api_key:
        return None
    name = UNIVERSITY_DISPLAY_NAMES.get(university, university)
    payload = {
        "system_instruction": {"parts": [{"text": _WEB_SYSTEM.format(
            university=name, domain=domains[0])}]},
        "contents": [{"parts": [{"text": f"{question}\n(site:{domains[0]})"}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    url = (f"{settings.gemini_base_url.rstrip('/')}/models/"
           f"{settings.general_fallback_search_model}:generateContent")
    started = time.monotonic()
    try:
        resp = httpx.post(
            url, json=payload, timeout=timeout_s,
            headers={"x-goog-api-key": settings.gemini_api_key,
                     "Content-Type": "application/json"},
        )
    except Exception as exc:  # noqa: BLE001 - fallback must never fail /ask
        logger.warning("[fallback] web search failed: %s", type(exc).__name__)
        return None
    from src.usage_tracker import get_tracker

    if resp.status_code != 200:
        get_tracker().record_error("gemini", resp.status_code, _error_summary(resp),
                                   getattr(resp, "headers", None))
        logger.warning("[fallback] web search HTTP %d: %s",
                       resp.status_code, _error_summary(resp))
        return None
    get_tracker().record_response("gemini", settings.general_fallback_search_model,
                                  getattr(resp, "headers", None))
    parsed = parse_search_response(resp.json(), domains)
    if parsed is None:
        logger.info("[fallback] web search: no official answer for %r", question[:60])
        return None
    text, official = parsed
    links = []
    for web in official[:3]:
        line = f"{web.get('title', domains[0])}: {web.get('uri', '')}".strip()
        if line not in links:
            links.append(line)
    body = (WEB_PREFIX.format(university=name) + "\n\n" + text
            + "\n\nالمصادر:\n" + "\n".join(links))
    return FallbackAnswer(
        text=body, kind="web", provider="gemini-search",
        model=settings.general_fallback_search_model, sources=links,
        seconds=time.monotonic() - started,
    )


def general_knowledge_answer(
    question: str, university: str, chain: list[BaseGenerator],
    deadline: float | None, settings: Settings,
) -> FallbackAnswer | None:
    """Short labeled answer from model knowledge for general questions only."""
    name = UNIVERSITY_DISPLAY_NAMES.get(university, university)
    started = time.monotonic()
    try:
        result = generate_with_meta(
            chain, _LLM_SYSTEM.format(university=name), f"السؤال: {question}",
            label=f"fallback[{question[:30]}]", settings=settings,
            deadline=deadline,
        )
    except GenerationError as exc:
        logger.warning("[fallback] llm failed: %s", str(exc)[:160])
        return None
    text = result.text.strip()
    if not text or NO_ANSWER in text:
        logger.info("[fallback] llm declined (regulation/off-topic): %r", question[:60])
        return None
    return FallbackAnswer(
        text=f"{LLM_PREFIX}\n\n{text}", kind="llm", provider=result.provider,
        model=result.model, seconds=time.monotonic() - started,
    )


def try_fallback(
    question: str, university: str, chain: list[BaseGenerator],
    deadline: float | None, settings: Settings | None = None,
    search_query: str | None = None,
) -> FallbackAnswer | None:
    """Run the configured fallback within the remaining request budget."""
    s = settings or get_settings()
    mode = s.general_fallback_mode
    if mode not in MODES or mode == "off":
        return None

    def remaining() -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    if mode in ("web", "web_then_llm"):
        left = remaining()
        if left is None or left >= s.min_attempt_seconds:
            timeout = s.per_attempt_timeout_s if left is None else min(
                s.per_attempt_timeout_s, left - 0.2)
            answer = search_official_site(
                question, university, s, timeout, chain=chain, deadline=deadline,
                search_query=search_query)
            if answer is not None:
                return answer
    if mode in ("llm", "web_then_llm"):
        left = remaining()
        if left is None or left >= s.min_attempt_seconds:
            return general_knowledge_answer(question, university, chain, deadline, s)
    return None


def main() -> None:
    """Live smoke test from backend/:
    python -m uv run python -X utf8 -m src.general_fallback "سؤال" kau [web|llm]
    """
    import sys

    from src.generators import get_generation_chain

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    question = sys.argv[1] if len(sys.argv) > 1 else "من هو رئيس الجامعة؟"
    university = sys.argv[2] if len(sys.argv) > 2 else "kau"
    which = sys.argv[3] if len(sys.argv) > 3 else "web"
    s = get_settings()
    started = time.monotonic()
    if which == "web":
        answer = search_official_site(question, university, s, 20.0)
    else:
        answer = general_knowledge_answer(
            question, university, get_generation_chain(settings=s), None, s)
    print(f"\n[{which}] {time.monotonic() - started:.1f}s")
    print(answer.text if answer else "(no answer: declined, nothing official found, or error — see log)")


if __name__ == "__main__":
    main()
