"""تقييم الإجابات: يشغّل Answerer.ask ويكتب ملفاً للتصحيح اليدوي.

يحفظ في data/eval/results/<الاسم>.jsonl و .md لكل سؤال: السؤال، والجواب المتوقع،
وجواب النظام، ومعرّفات المقاطع التي استشهد بها. ويطبع فحوصاً آلية بسيطة، بلا نموذج
حَكَم: هل الرفض كما هو متوقع، وهل في الجواب استشهاد، وهل أرقام الجواب المتوقع موجودة فيه.

التشغيل (Sonnet يكلّف قرابة 1–2$ لكل 100 سؤال):
    python eval/eval_answers.py --ids t01,t02,t03 --out answers_probe
    python eval/eval_answers.py --sets main --out answers_full
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openrouter
from answer import Answerer, setup_request_log
from eval_retrieval import (
    QUESTION_SETS,
    RERANK_CACHE_FILE,
    RESULTS_DIR,
    load_cache,
    load_questions,
    save_new_cache_entries,
)
from retrieve import Retriever

REFUSAL_TYPE = "رفض"
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    return NUMBER.findall(text.translate(ARABIC_DIGITS))


def check_answer(question: dict, result: dict) -> dict:
    """الفحوص الآلية؛ الحكم النهائي للتصحيح اليدوي."""
    expects_refusal = question.get("type") == REFUSAL_TYPE
    expected_numbers = numbers_in(question.get("expected_answer", ""))
    answer_numbers = set(numbers_in(result["answer"]))
    found = [number for number in expected_numbers if number in answer_numbers]
    return {
        "refusal_ok": result["refused"] == expects_refusal,
        "has_citation": bool(result["cited"]),
        "numbers_found": f"{len(found)}/{len(expected_numbers)}",
    }


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [f"# تصحيح يدوي — {path.stem}", "", "اكتب في «الحكم»: صحيح / ناقص / خطأ.", ""]
    for row in rows:
        checks = row["checks"]
        lines += [
            f"## {row['id']} ({row.get('type') or '—'})",
            "",
            f"**السؤال:** {row['question']}",
            "",
            f"**المتوقع:** {row['expected_answer']}",
            "",
            f"**الجواب:**\n\n{row['answer']}",
            "",
            f"- المستشهد بها: {', '.join(row['cited_chunks']) or '—'}",
            f"- الرفض كما هو متوقع: {checks['refusal_ok']} · استشهاد: "
            f"{checks['has_citation']} · الأرقام: {checks['numbers_found']}",
            "- الحكم: ",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_questions(set_names: list[str], ids: str | None) -> list[dict]:
    questions = []
    for set_name in set_names:
        for file_name in QUESTION_SETS[set_name]:
            questions += load_questions(file_name, only_with_gold=False)
    if not ids:
        return questions
    wanted = ids.split(",")
    return [question for question in questions if question["id"] in wanted]


def answer_row(answerer: Answerer, question: dict) -> dict:
    started = time.time()
    result = answerer.ask(question["question"], question["university"])
    cited_chunks = []
    for source in result["sources"]:
        if source["n"] in result["cited"]:
            cited_chunks.append(source["chunk_id"])
    return {
        "id": question["id"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question.get("expected_answer", ""),
        "answer": result["answer"],
        "refused": result["refused"],
        "model_used": result["model_used"],
        "cost": round(result["cost"], 6),
        "cited_chunks": cited_chunks,
        "sources": [source["chunk_id"] for source in result["sources"]],
        "seconds": round(time.time() - started, 1),
        "checks": check_answer(question, result),
    }


def error_row(question: dict, error: Exception) -> dict:
    """سؤال تعذّرت إجابته (رد مقطوع أو انقطاع): يُسجَّل بالخطأ ويكمل التشغيل."""
    return {
        "id": question["id"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question.get("expected_answer", ""),
        "answer": str(error),
        "error": True,
        "refused": False,
        "model_used": None,
        "cost": 0.0,
        "cited_chunks": [],
        "sources": [],
        "seconds": 0.0,
        "checks": {"refusal_ok": False, "has_citation": False, "numbers_found": "—"},
    }


def run_questions(answerer: Answerer, questions: list[dict], out_path: Path) -> list[dict]:
    """يكتب كل سطر فور وصوله: توقف التشغيل في المنتصف كان يُضيع إجابات مدفوعة."""
    rows = []
    with out_path.open("w", encoding="utf-8") as stream:
        for number, question in enumerate(questions, 1):
            try:
                row = answer_row(answerer, question)
            except RuntimeError as error:
                row = error_row(question, error)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{number}/{len(questions)} {question['id']}: {row['checks']} ({row['cost']:.4f}$)"
            )
    return rows


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sets", nargs="+", choices=list(QUESTION_SETS), default=["main"])
    parser.add_argument("--ids", help="أسئلة محددة مفصولة بفواصل، مثل t01,n05")
    parser.add_argument("--out", required=True, help="اسم ملف النتيجة داخل data/eval/results")
    args = parser.parse_args()

    questions = select_questions(args.sets, args.ids)
    rerank_cache = load_cache(RERANK_CACHE_FILE)
    old_reranks = dict(rerank_cache)
    answerer = Answerer(Retriever(), rerank_cache=rerank_cache)
    setup_request_log()

    credit_before = openrouter.read_remaining_credit()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        rows = run_questions(answerer, questions, RESULTS_DIR / f"{args.out}.jsonl")
    finally:
        save_new_cache_entries(RERANK_CACHE_FILE, rerank_cache, old_reranks)
    credit_after = openrouter.read_remaining_credit()
    write_markdown(RESULTS_DIR / f"{args.out}.md", rows)

    refusals_ok = sum(1 for row in rows if row["checks"]["refusal_ok"])
    cited = sum(1 for row in rows if row["checks"]["has_citation"])
    errors = [row["id"] for row in rows if row.get("error")]
    print(f"\nالرفض كما هو متوقع: {refusals_ok}/{len(rows)} · فيها استشهاد: {cited}/{len(rows)}")
    if errors:
        print(f"تعذّرت إجابتها: {', '.join(errors)}")
    print(f"الكلفة من usage: {sum(row['cost'] for row in rows):.4f}$")
    if credit_before is not None and credit_after is not None:
        print(f"الرصيد: {credit_before:.4f}$ → {credit_after:.4f}$")
    print(f"للتصحيح اليدوي: {RESULTS_DIR / (args.out + '.md')}")


if __name__ == "__main__":
    main()
