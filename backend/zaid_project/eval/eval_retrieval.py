"""تقييم الاسترجاع: هل يصل المقطع الصحيح (gold) إلى المراتب الأولى؟

يشغّل خط الاسترجاع الحقيقي (Answerer.retrieve) على الأسئلة التي لها gold_chunks،
ويحسب Hit@1 و Hit@4 و Hit@15 و MRR@10 لكل ملف ولكل نوع سؤال.

ترتيب المرتِّب يُحفظ في data/eval/cache، فالتشغيل الثاني مجاني
ونتيجته ثابتة. النتائج تُحفظ في data/eval/results/<الاسم>.json و .md

التشغيل:
    python eval/eval_retrieval.py --sets main --no-rerank --out baseline_norerank
    python eval/eval_retrieval.py --sets main --out rerank_run --compare data/eval/results/a.json
    python eval/eval_retrieval.py --diff data/eval/results/a.json data/eval/results/b.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openrouter
from answer import Answerer
from retrieve import Retriever

EVAL_DIR = ROOT / "data" / "eval"
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"
RERANK_CACHE_FILE = CACHE_DIR / "rerank.jsonl"

QUESTION_SETS = {
    "main": ["test_questions_50.jsonl", "test_questions_new50.jsonl"],
    "dev": ["questions.jsonl", "student_questions.jsonl"],
    "mixed": ["mixed50.jsonl"],
    "fresh": ["fresh50.jsonl"],
    "colleges": ["colleges20.jsonl"],
    "grad": ["grad_diploma27.jsonl"],
}
FINAL_K = 4
DEEP_K = 15
DIFF_K = 10


def load_questions(file_name: str, only_with_gold: bool = True) -> list[dict]:
    questions = []
    path = EVAL_DIR / file_name
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line)
        has_gold = bool(question.get("gold_chunks") or question.get("gold_doc"))
        if only_with_gold and not has_gold:
            continue
        question["file"] = file_name
        questions.append(question)
    return questions


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["key"]] = entry["value"]
    return cache


def save_new_cache_entries(path: Path, cache: dict, old_cache: dict) -> int:
    """يضيف إلى الملف ما جدّ أو تغيّر (صياغة صُححت)؛ عند القراءة يغلب آخر سطر للمفتاح."""
    new_keys = [key for key in cache if old_cache.get(key) != cache[key]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for key in new_keys:
            entry = {"key": key, "value": cache[key]}
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(new_keys)


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip()


def first_gold_rank(ranked: list[str], gold: list[str]) -> int | None:
    for position, chunk_id in enumerate(ranked, 1):
        if chunk_id in gold:
            return position
    return None


def first_gold_document_rank(titles: list[str], gold_doc: str) -> int | None:
    """أول مقطع من الوثيقة المطلوبة؛ gold_doc جزء من عنوانها، والبدائل مفصولة بـ «|»."""
    alternatives = gold_doc.split("|")
    for position, title in enumerate(titles, 1):
        for alternative in alternatives:
            if alternative and alternative in title:
                return position
    return None


def compute_metrics(rows: list[dict]) -> dict:
    ranks = [row["gold_rank"] for row in rows]
    hit_1 = sum(1 for rank in ranks if rank is not None and rank <= 1)
    hit_4 = sum(1 for rank in ranks if rank is not None and rank <= 4)
    hit_15 = sum(1 for rank in ranks if rank is not None and rank <= 15)
    reciprocal_ranks = [1 / rank for rank in ranks if rank is not None and rank <= 10]
    mrr_10 = sum(reciprocal_ranks) / len(rows) if rows else 0.0
    return {
        "questions": len(rows),
        "hit@1": hit_1,
        "hit@4": hit_4,
        "hit@15": hit_15,
        "mrr@10": round(mrr_10, 3),
    }


def group_metrics(rows: list[dict], field: str) -> dict:
    groups = {}
    for row in rows:
        name = row.get(field) or "—"
        groups.setdefault(name, []).append(row)
    return {name: compute_metrics(groups[name]) for name in sorted(groups)}


def format_metrics(metrics: dict) -> str:
    count = metrics["questions"]
    return (
        f"Hit@1 {metrics['hit@1']}/{count} · Hit@4 {metrics['hit@4']}/{count} · "
        f"Hit@15 {metrics['hit@15']}/{count} · MRR@10 {metrics['mrr@10']}"
    )


def metrics_table(title: str, grouped: dict) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| | أسئلة | Hit@1 | Hit@4 | Hit@15 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for name, metrics in grouped.items():
        lines.append(
            f"| {name} | {metrics['questions']} | {metrics['hit@1']} | "
            f"{metrics['hit@4']} | {metrics['hit@15']} | {metrics['mrr@10']} |"
        )
    return lines + [""]


def write_markdown(path: Path, result: dict) -> None:
    settings = result["settings"]
    lines = [
        f"# تقييم الاسترجاع — {path.stem}",
        "",
        f"- التاريخ: {settings['created']} · commit {settings['git_commit']}",
        f"- المرتِّب: {'مفعّل' if settings['rerank'] else 'معطّل'}",
        f"- المجموع: {format_metrics(result['metrics']['overall'])}",
        "",
    ]
    lines += metrics_table("لكل ملف", result["metrics"]["by_file"])
    lines += metrics_table("لكل نوع سؤال", result["metrics"]["by_type"])
    misses = [
        row for row in result["questions"] if row["gold_rank"] is None or row["gold_rank"] > FINAL_K
    ]
    lines += [f"### خارج الأربعة الأولى ({len(misses)})", ""]
    for row in misses:
        rank = row["gold_rank"] or "—"
        lines.append(f"- {row['id']} (المرتبة {rank}): {row['question']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_rank_changes(rows: list[dict], previous_path: Path) -> None:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_ranks = {row["id"]: row["gold_rank"] for row in previous["questions"]}
    changes = []
    for row in rows:
        if row["id"] not in previous_ranks:
            continue
        old_rank, new_rank = previous_ranks[row["id"]], row["gold_rank"]
        if old_rank != new_rank:
            changes.append(f"  {row['id']}: {old_rank or '—'} → {new_rank or '—'}")
    print(f"\nتغيّرت مرتبة {len(changes)} سؤالاً مقارنةً بـ {previous_path.name}:")
    for line in changes:
        print(line)


def diff_runs(first_path: Path, second_path: Path) -> int:
    """يقارن أول 10 مقاطع لكل سؤال بين تشغيلين؛ إعادة الهيكلة يجب أن تعطي صفر فرق."""
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = {}
    for row in json.loads(second_path.read_text(encoding="utf-8"))["questions"]:
        second[row["id"]] = row
    differences = 0
    compared = 0
    for row in first["questions"]:
        other = second.get(row["id"])
        if other is None:
            continue
        compared += 1
        if row["ranked"][:DIFF_K] != other["ranked"][:DIFF_K]:
            differences += 1
            print(f"{row['id']}:\n  {row['ranked'][:DIFF_K]}\n  {other['ranked'][:DIFF_K]}")
    print(f"\nقورن {compared} سؤالاً: {compared - differences} متطابق، {differences} مختلف")
    return differences


def evaluate(answerer: Answerer, questions: list[dict]) -> tuple[list[dict], float]:
    """يرجع صفاً لكل سؤال، ومجموع كلفة الاستدعاءات الجديدة من حقل usage."""
    rows = []
    total_cost = 0.0
    for number, question in enumerate(questions, 1):
        result = answerer.retrieve(question["question"], question["university"])
        total_cost += result["cost"]
        top_hits = result["candidates"][:DEEP_K]
        ranked = [hit["chunk_id"] for hit in top_hits]
        if question.get("gold_chunks"):
            gold = question["gold_chunks"]
            gold_rank = first_gold_rank(ranked, gold)
        else:
            gold = question["gold_doc"]
            gold_rank = first_gold_document_rank([hit["doc_title"] for hit in top_hits], gold)
        rows.append(
            {
                "id": question["id"],
                "file": question["file"],
                "type": question.get("type"),
                "university": question["university"],
                "question": question["question"],
                "search_query": result["search_query"],
                "reranked": result["reranked"],
                "rerank_failed": result["rerank_failed"],
                "gold": gold,
                "ranked": ranked,
                "gold_rank": gold_rank,
            }
        )
        print(f"\r{number}/{len(questions)}", end="", flush=True)
    print()
    return rows, total_cost


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sets", nargs="+", choices=list(QUESTION_SETS), default=["main"])
    parser.add_argument("--no-rerank", action="store_true", help="بدون المرتِّب (مجاني)")
    parser.add_argument("--out", help="اسم ملف النتيجة داخل data/eval/results")
    parser.add_argument("--compare", type=Path, help="نتيجة سابقة لطباعة الأسئلة التي تغيرت")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    if not args.diff and not args.out:
        parser.error("--out مطلوب")
    return args


def load_sets(set_names: list[str]) -> list[dict]:
    questions = []
    for set_name in set_names:
        for file_name in QUESTION_SETS[set_name]:
            questions += load_questions(file_name)
    return questions


def run_evaluation(args: argparse.Namespace) -> dict:
    """يشغّل الأسئلة بالكاش، ويحفظ الاستدعاءات الجديدة فيه، ويرجع النتيجة كاملة."""
    rerank_cache = load_cache(RERANK_CACHE_FILE)
    old_reranks = dict(rerank_cache)
    answerer = Answerer(
        Retriever(), use_reranker=not args.no_rerank, rerank_cache=rerank_cache
    )

    credit_before = openrouter.read_remaining_credit()
    started = time.time()
    rows, usage_cost = evaluate(answerer, load_sets(args.sets))
    new_reranks = save_new_cache_entries(RERANK_CACHE_FILE, rerank_cache, old_reranks)
    return {
        "settings": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": current_git_commit(),
            "rerank": not args.no_rerank,
            "sets": args.sets,
            "new_rerank_calls": new_reranks,
            "usage_cost": round(usage_cost, 6),
            "credit_before": credit_before,
            "credit_after": openrouter.read_remaining_credit(),
            "seconds": round(time.time() - started),
        },
        "metrics": {
            "overall": compute_metrics(rows),
            "by_file": group_metrics(rows, "file"),
            "by_type": group_metrics(rows, "type"),
        },
        "questions": rows,
    }


def print_summary(result: dict, json_path: Path) -> None:
    settings = result["settings"]
    print(f"المجموع: {format_metrics(result['metrics']['overall'])}")
    for name, metrics in result["metrics"]["by_file"].items():
        print(f"  {name}: {format_metrics(metrics)}")
    print(
        f"استدعاءات المرتِّب الجديدة: {settings['new_rerank_calls']} — "
        f"كلفتها {settings['usage_cost']:.4f}$"
    )
    if settings["credit_before"] is not None and settings["credit_after"] is not None:
        print(f"الرصيد: {settings['credit_before']:.4f}$ → {settings['credit_after']:.4f}$")
    print(f"النتيجة: {json_path}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_arguments()
    if args.diff:
        diff_runs(*args.diff)
        return
    result = run_evaluation(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{args.out}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(RESULTS_DIR / f"{args.out}.md", result)
    print_summary(result, json_path)
    if args.compare:
        print_rank_changes(result["questions"], args.compare)


if __name__ == "__main__":
    main()
