"""وقت السؤال: يسترجع المقاطع المناسبة ثم يولّد الإجابة عبر OpenRouter.

خطوات السؤال الواحد:
  1. بحث هجين (معنى وكلمات) بسؤال الطالب كما كتبه؛ في المحادثة تسبقه أسئلته السابقة.
  2. المرتِّب (qwen3-reranker-8b) يرتّب أول 25 نتيجة.
  3. Sonnet يكتب الإجابة من المقاطع الأربعة الأولى مع أرقام الاستشهاد [n].

لا إعادة صياغة للسؤال ولا معاملة خاصة لأسئلة المواعيد: التقويم مقطّع حدثاً حدثاً
(chunk.py)، فسطر الموعد مقطع عادي يصل إليه البحث، ويرفعه المرتِّب إن سأل الطالب عن موعد.

كل سؤال يُسجَّل سطراً في logs/requests.jsonl مع كلفته. رسالة نموذج الإجابة وفحص ردها
في prompt_parts.py.

التشغيل:
    python answer.py "كم إنذاراً يؤدي إلى الفصل؟" --university KAU
    from answer import Answerer;  Answerer().ask("...", university="KAU")
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import openrouter
from prompt_parts import (
    REFUSAL,
    build_prompt,
    cited_passages,
    history_messages,
    is_greeting,
    is_refusal,
    source_cards,
)
from retrieve import Retriever

logger = logging.getLogger(__name__)
REQUEST_LOG = logging.getLogger("rag.requests")

MISSING_UNIVERSITY = "الرجاء تحديد الجامعة (KSU أو KAU) قبل السؤال."
CHINESE_JAPANESE_KOREAN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"ملف التعليمات {path.name} غير موجود في مجلد prompts")
    return path.read_text(encoding="utf-8").strip()


def setup_request_log(path: Path = config.REQUEST_LOG_FILE) -> None:
    """يكتب سطر JSON لكل سؤال في logs/requests.jsonl (يُستدعى مرة عند التشغيل)."""
    if REQUEST_LOG.handlers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOG.addHandler(handler)
    REQUEST_LOG.setLevel(logging.INFO)
    REQUEST_LOG.propagate = False


def rerank_documents(hits: list[dict]) -> list[str]:
    return [hit["text"][: config.RERANK_TEXT_MAX] for hit in hits]


def rerank_cache_key(question: str, documents: list[str]) -> str:
    text = json.dumps([config.RERANK_MODEL, question, documents], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recent_turns(history: list[dict] | None) -> list[dict]:
    """آخر HISTORY_TURNS أدوار من المحادثة: تكفي لفهم سؤال المتابعة."""
    return (history or [])[-config.HISTORY_TURNS :]


def search_query(question: str, history: list[dict]) -> str:
    """نص البحث: السؤال وحده، أو أسئلة المحادثة السابقة ثم السؤال.

    سؤال المتابعة («وكم مرة أقدر أسويه؟»، «وفي الماجستير؟») لا يذكر موضوعه، فيُبحث به مع
    ما سبقه من أسئلة الطالب. إجابات المساعد لا تدخل: هي نص طويل يغلب على السؤال."""
    questions = [turn["question"] for turn in history] + [question]
    return " ".join(" ".join(text.split()) for text in questions)


def order_by_reranker(candidates: list[dict], order: list[int]) -> list[dict]:
    ordered = [candidates[index] for index in order]
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


class Answerer:
    def __init__(
        self,
        retriever: Retriever | None = None,
        use_reranker: bool = True,
        rerank_cache: dict | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.use_reranker = use_reranker
        self.system_prompt = load_prompt(config.SYSTEM_PROMPT_FILE)
        self.rerank_cache = rerank_cache
        self.retrieve_cache = {}

    def rerank(self, question: str, candidates: list[dict]) -> tuple[list[dict] | None, float]:
        """المرشحون بترتيب المرتِّب وكلفته، أو None إن فشل فيبقى ترتيب البحث."""
        documents = rerank_documents(candidates)
        key = rerank_cache_key(question, documents)
        if self.rerank_cache is not None and key in self.rerank_cache:
            return order_by_reranker(candidates, self.rerank_cache[key]), 0.0
        try:
            reply = openrouter.rerank(config.RERANK_MODEL, question, documents)
        except RuntimeError as error:
            logger.warning("تعذّر الاتصال بالمرتِّب (%s)", error)
            return None, 0.0
        if reply is None:
            return None, 0.0
        if self.rerank_cache is not None:
            self.rerank_cache[key] = reply["order"]
        return order_by_reranker(candidates, reply["order"]), reply["cost"]

    def run_retrieval(
        self,
        question: str,
        university: str,
        k: int,
        history: list[dict],
    ) -> dict:
        query = search_query(question, history)
        rest = self.retriever.search(
            query, k=config.RERANK_CANDIDATES, university=university
        )
        cost = 0.0
        is_reranked = False
        has_rerank_failed = False
        if self.use_reranker and rest:
            reranked, rerank_cost = self.rerank(query, rest)
            cost += rerank_cost
            if reranked is None:
                has_rerank_failed = True
            else:
                rest = reranked
                is_reranked = True
        return {
            "hits": rest[:k],
            "candidates": rest,
            "search_query": None if query == question else query,
            "reranked": is_reranked,
            "rerank_failed": has_rerank_failed,
            "cost": cost,
        }

    def retrieve(
        self,
        question: str,
        university: str,
        k: int = config.TOP_K,
        history: list[dict] | None = None,
    ) -> dict:
        """المقاطع الأربعة (hits) وكل المرشحين بترتيبهم (candidates) ووصف ما جرى."""
        history = recent_turns(history)
        turns = tuple((turn["question"], turn["answer"]) for turn in history)
        key = (question, university, k, turns)
        if key not in self.retrieve_cache:
            if len(self.retrieve_cache) >= config.RETRIEVE_CACHE_SIZE:
                del self.retrieve_cache[list(self.retrieve_cache)[0]]
            self.retrieve_cache[key] = self.run_retrieval(
                question, university, k, history
            )
        return self.retrieve_cache[key]

    def generate_answer(self, question: str, hits: list[dict], history: list[dict]) -> dict:
        """الجواب والنموذج الذي كتبه وكلفته. إن نفد الرصيد عن Sonnet (402) يجيب qwen."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += history_messages(history)
        messages.append({"role": "user", "content": build_prompt(question, hits)})
        model = config.ANSWER_MODEL
        max_tokens = config.ANSWER_MAX_TOKENS
        try:
            reply = openrouter.chat(model, messages, max_tokens, config.ANSWER_TEMPERATURE)
        except openrouter.NoCreditError:
            logger.warning(
                "رصيد OpenRouter لا يكفي لـ %s (402) — يجيب %s", model, config.FALLBACK_ANSWER_MODEL
            )
            model = config.FALLBACK_ANSWER_MODEL
            max_tokens = config.FALLBACK_MAX_TOKENS
            reply = openrouter.chat(model, messages, max_tokens, config.ANSWER_TEMPERATURE)
        cost = reply["cost"]
        if CHINESE_JAPANESE_KOREAN.search(reply["content"]):
            reply = openrouter.chat(model, messages, max_tokens, config.ANSWER_TEMPERATURE)
            cost += reply["cost"]
        return {"answer": reply["content"], "model_used": model, "cost": cost}

    def ask(
        self,
        question: str,
        university: str | None,
        k: int = config.TOP_K,
        history: list[dict] | None = None,
    ) -> dict:
        started = time.time()
        if university not in config.UNIVERSITY_NAMES:
            return {
                "answer": MISSING_UNIVERSITY,
                "sources": [],
                "grounded": True,
                "refused": True,
                "cited": [],
            }
        history = recent_turns(history)
        retrieval = self.retrieve(question, university, k, history)
        hits = retrieval["hits"]
        result = {
            "answer": REFUSAL,
            "model_used": None,
            "search_query": retrieval["search_query"],
            "grounded": True,
            "refused": True,
            "cited": [],
            "sources": [],
            "reranked": retrieval["reranked"],
            "rerank_failed": retrieval["rerank_failed"],
            "cost": retrieval["cost"],
        }
        if hits:
            generated = self.generate_answer(question, hits, history)
            cited = cited_passages(generated["answer"], len(hits))
            refused = is_refusal(generated["answer"])
            greeting = is_greeting(generated["answer"])
            result.update(
                {
                    "answer": generated["answer"],
                    "model_used": generated["model_used"],
                    "grounded": refused or greeting or bool(cited),
                    "refused": refused,
                    "greeting": greeting,
                    "cited": cited,
                    "sources": source_cards(hits),
                    "cost": retrieval["cost"] + generated["cost"],
                }
            )
        log_request(question, university, result, hits, len(history), time.time() - started)
        return result


def log_request(
    question: str,
    university: str,
    result: dict,
    hits: list[dict],
    history_turns: int,
    seconds: float,
) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "university": university,
        "question": question,
        "history_turns": history_turns,
        "search_query": result["search_query"],
        "reranked": result["reranked"],
        "rerank_failed": result["rerank_failed"],
        "chunk_ids": [hit["chunk_id"] for hit in hits],
        "model_used": result["model_used"],
        "seconds": round(seconds, 1),
        "refused": result["refused"],
        "cost": round(result["cost"], 6),
    }
    REQUEST_LOG.info(json.dumps(record, ensure_ascii=False))


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--university", choices=list(config.UNIVERSITY_NAMES), required=True)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    setup_request_log()
    result = Answerer().ask(args.question, university=args.university, k=args.k)
    print("\n" + result["answer"] + "\n")
    print("─" * 70)
    for source in result["sources"]:
        mark = "★" if source["n"] in result["cited"] else " "
        line = f"{mark} [{source['n']}] {source['chunk_id']} — {source['doc_title']}"
        if source["label"]:
            line += f" — {source['label']}"
        print(line)
    if not result["grounded"]:
        print("\n⚠ الجواب بلا استشهاد — غير مستند إلى المقاطع.")


if __name__ == "__main__":
    main()
