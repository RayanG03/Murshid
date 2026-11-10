"""واجهة الويب المحلية (FastAPI): صفحة web/index.html ونقاط API للبحث والإجابة.

الواجهة (web/index.html + web/static/app.js) تطلب /api/search أولاً، ثم /api/answer للإجابة. نتيجة
الاسترجاع محفوظة في كاش صغير داخل Answerer، فلا يتكرر البحث ولا المرتِّب للسؤال نفسه.

التشغيل:
    python app.py           ثم افتح http://127.0.0.1:8000
"""

import json
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from answer import MISSING_UNIVERSITY, Answerer, setup_request_log
from prompt_parts import source_cards

logger = logging.getLogger(__name__)
UNEXPECTED_ERROR = "حدث خطأ غير متوقع في الخادم — أعد المحاولة بعد قليل."

app = FastAPI(title="مساعد اللوائح الجامعية")
app.mount("/static", StaticFiles(directory=config.WEB_DIR / "static"), name="static")


class Query(BaseModel):
    question: str
    university: str | None = None
    k: int = config.TOP_K
    history: list[dict[str, str]] = []


class Feedback(BaseModel):
    university: str | None = None
    question: str = ""
    answer: str = ""
    feedback: str | None = None


def get_answerer() -> Answerer:
    return app.state.answerer


def history_turns(query: Query) -> list[dict]:
    """الأدوار السابقة الصالحة فقط (سؤال وجواب غير فارغين)."""
    turns = []
    for turn in query.history:
        question = turn.get("question", "").strip()
        answer = turn.get("answer", "").strip()
        if question and answer:
            turns.append({"question": question, "answer": answer})
    return turns


def query_problem(query: Query) -> str | None:
    """رسالة بالعربية إن كان الطلب ناقصاً، وإلا None."""
    if not query.question.strip():
        return "اكتب سؤالك أولاً."
    if query.university not in config.UNIVERSITY_NAMES:
        return MISSING_UNIVERSITY
    return None


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.error("خطأ غير متوقع في %s", request.url.path, exc_info=error)
    return JSONResponse(
        status_code=500,
        content={
            "error": UNEXPECTED_ERROR,
            "answer": UNEXPECTED_ERROR,
            "sources": [],
            "cited": [],
            "grounded": False,
        },
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"chunks": len(get_answerer().retriever.chunks), "model": config.ANSWER_MODEL}


@app.get("/api/corpus")
def corpus() -> dict:
    return get_answerer().retriever.corpus()


@app.post("/api/search")
def search(query: Query) -> dict:
    """المرحلة السريعة: المقاطع وحدها، دون انتظار النموذج."""
    problem = query_problem(query)
    if problem:
        return {"sources": [], "error": problem}
    retrieval = get_answerer().retrieve(
        query.question.strip(),
        query.university,
        query.k,
        history_turns(query),
    )
    return {"sources": source_cards(retrieval["hits"]), "rerank_failed": retrieval["rerank_failed"]}


@app.post("/api/answer")
def answer(query: Query) -> dict:
    problem = query_problem(query)
    if problem:
        return {"answer": problem, "error": True, "sources": [], "cited": [], "grounded": False}
    try:
        return get_answerer().ask(
            query.question.strip(),
            query.university,
            query.k,
            history_turns(query),
        )
    except RuntimeError as error:
        logger.warning("تعذّرت الإجابة: %s", error)
        return {"answer": str(error), "error": True, "sources": [], "cited": [], "grounded": False}


@app.post("/api/feedback")
def feedback(item: Feedback) -> dict:
    """تقييم المستخدم للإجابة (👍/👎) — سطر JSON في logs/feedback.jsonl."""
    if item.feedback not in ("up", "down", None):
        return {"ok": False}
    record = {"time": datetime.now().isoformat(timespec="seconds"), **item.model_dump()}
    config.FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    setup_request_log()
    print("تحميل الفهرس ونموذج bge-m3 ...")
    answerer = Answerer()
    answerer.retriever.encode(["تهيئة"])
    app.state.answerer = answerer
    print("جاهز — افتح http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
