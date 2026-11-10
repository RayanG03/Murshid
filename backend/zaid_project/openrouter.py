"""اتصالات OpenRouter: المحادثة (chat) والترتيب (rerank) وقراءة الرصيد.

كل طلب يرجع كلفته من حقل usage، ويعيد المحاولة عند الأخطاء المؤقتة
(429 و 5xx وانقطاع الرد) بانتظار يطول مع كل محاولة.

المفتاح في api_key.txt أو في متغير البيئة RAG_OPENROUTER_KEY.

الاستخدام:
    import openrouter
    reply = openrouter.chat("qwen/qwen3-32b", [{"role": "user", "content": "..."}], 200, 0.1)
    reply["content"], reply["cost"]
"""

import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

import config

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = {0, 429, 500, 502, 503, 504}
NO_CREDIT_STATUS = 402
THINKING = re.compile(r"<think>.*?</think>\s*", re.S)


class NoCreditError(RuntimeError):
    """OpenRouter ردّ بـ 402: الرصيد لا يكفي لهذا الطلب."""


def load_api_key() -> str:
    key = os.environ.get("RAG_OPENROUTER_KEY", "").strip()
    if key:
        return key
    if config.API_KEY_FILE.exists():
        key = config.API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key and not key.startswith("PASTE_"):
            return key
    raise RuntimeError(
        "مفتاح OpenRouter غير موجود — ضعه في api_key.txt أو في متغير البيئة RAG_OPENROUTER_KEY"
    )


def send_request(url: str, payload: dict | None, timeout: int) -> tuple[int, dict | None]:
    """طلب واحد (POST إن وُجد payload وإلا GET). يرجع (رمز الحالة، الرد)."""
    headers = {
        "Authorization": f"Bearer {load_api_key()}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "RAG University Regulations",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, read_error_body(error)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        logger.warning("لم يصل رد من OpenRouter: %s", error)
        return 0, None
    except json.JSONDecodeError:
        logger.warning("رد OpenRouter ليس JSON")
        return 0, None


def read_error_body(error: urllib.error.HTTPError) -> dict | None:
    try:
        return json.loads(error.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def send_with_retries(
    url: str, payload: dict, timeout: int, retries: int
) -> tuple[int, dict | None]:
    """يعيد المحاولة عند 429 و 5xx وانقطاع الرد: الانتظار 2 ث ثم 4 ث."""
    status, body = send_request(url, payload, timeout)
    for attempt in range(1, retries + 1):
        if status not in RETRYABLE_STATUSES:
            break
        wait = config.RETRY_WAIT_SECONDS * attempt
        logger.warning("OpenRouter ردّ بـ %s — محاولة جديدة بعد %s ث", status, wait)
        time.sleep(wait)
        status, body = send_request(url, payload, timeout)
    return status, body


def error_message(status: int, body: dict | None) -> str:
    if status == 0:
        return "لم يصل رد (انقطاع الاتصال أو انتهاء المهلة)"
    if body and isinstance(body.get("error"), dict):
        return f"{status}: {body['error'].get('message', '')}"
    return f"رمز {status}"


def chat(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    retries: int = config.CHAT_RETRIES,
) -> dict:
    """رد النموذج: {"content": النص، "model": النموذج، "cost": الكلفة بالدولار}.

    يرفع NoCreditError عند 402، وRuntimeError بالعربية عند أي فشل آخر.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    status, body = send_with_retries(
        config.OPENROUTER_CHAT_URL, payload, config.CHAT_TIMEOUT, retries
    )
    if status == NO_CREDIT_STATUS:
        raise NoCreditError(f"رصيد OpenRouter لا يكفي لـ {model} — اشحن الرصيد")
    if status != 200 or not body or not body.get("choices"):
        raise RuntimeError(f"تعذّر الحصول على رد من {model}: {error_message(status, body)}")
    choice = body["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    content = THINKING.sub("", content).strip()
    if not content or choice.get("finish_reason") == "length":
        raise RuntimeError("انقطعت إجابة النموذج قبل اكتمالها — أعد المحاولة")
    cost = (body.get("usage") or {}).get("cost") or 0.0
    return {"content": content, "model": model, "cost": cost}


def rerank(
    model: str, query: str, documents: list[str], retries: int = config.RERANK_RETRIES
) -> dict | None:
    """ترتيب المستندات من الأنسب إلى الأبعد: {"order": أرقامها، "cost": الكلفة}.

    يرجع None إن فشل الطلب بعد إعادة المحاولة؛ المستدعي يُبقي ترتيب البحث.
    """
    payload = {"model": model, "query": query, "documents": documents}
    status, body = send_with_retries(
        config.OPENROUTER_RERANK_URL, payload, config.RERANK_TIMEOUT, retries
    )
    if status != 200 or not body or not is_valid_rerank_results(body.get("results")):
        logger.warning("فشل المرتِّب (%s) — يبقى ترتيب البحث", error_message(status, body))
        return None
    results = sorted(body["results"], key=lambda result: -result["relevance_score"])
    order = []
    for result in results:
        if 0 <= result["index"] < len(documents):
            order.append(result["index"])
    cost = (body.get("usage") or {}).get("cost") or 0.0
    return {"order": order, "cost": cost}


def is_valid_rerank_results(results: object) -> bool:
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, dict):
            return False
        if not isinstance(result.get("index"), int):
            return False
        if not isinstance(result.get("relevance_score"), (int, float)):
            return False
    return True


def read_remaining_credit() -> float | None:
    """الرصيد المتبقي بالدولار، أو None إن تعذّرت القراءة."""
    try:
        status, body = send_request(config.OPENROUTER_CREDITS_URL, None, config.CREDITS_TIMEOUT)
    except RuntimeError:
        return None
    if status != 200 or not body or "data" not in body:
        return None
    return body["data"]["total_credits"] - body["data"]["total_usage"]
