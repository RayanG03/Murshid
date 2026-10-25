"""Held-out evaluation: retrieval hits, refusal accuracy, latency, answers.

Unlike src.retrieval_eval (self-retrieval: the indexed questions are used
as queries, so Hit@k is near-trivially high), this suite runs
backend/eval/questions.csv — paraphrased / dialect student questions plus
questions that must be refused — through the REAL pipeline.

Modes (run from backend/):
  gate-only (default, no LLM calls, free):
      python -m uv run python -X utf8 -m src.eval_suite
  one university, with config overrides (compare a change with baseline):
      ... -m src.eval_suite --university kau --set kau_gate_mode=rerank
  comparison table of the main configurations (gate-only):
      ... -m src.eval_suite --matrix
  full answers through the provider chain (costs API calls):
      ... -m src.eval_suite --university ksu --generate

Metrics (per university):
  answerable_pass  share of `answer` rows that passed the gate
  hit_rate         share of `answer` rows whose evidence contains a keyword
  refuse_correct   share of `refuse` rows refused
  top-score window max gate score among refuse rows vs min among hits,
                   to calibrate thresholds (rerank or cosine)
With --generate also: keyword-in-answer, ungrounded numbers, providers,
latency. Every run writes a CSV to backend/data/eval/.
"""

import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

from src.config import BACKEND_DIR, get_settings
from src.generators import BaseGenerator, GenerationError
from src.pipeline import answer_question

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

QUESTIONS = BACKEND_DIR / "eval" / "questions.csv"

# Named configurations for --matrix (gate-only). Each is a set of
# settings overrides applied on top of config/.env.
MATRIX: dict[str, dict[str, str]] = {
    "baseline": {},
    "kau_rerank": {"kau_gate_mode": "rerank"},
    "dialect": {"dialect_normalization": "true"},
    "hybrid": {"hybrid_lexical_extra": "3"},
    # hybrid roughly doubled KSU rerank time on a laptop CPU (bigger pool)
    # without gains in the first run, so "all" leaves it out.
    "rerank+dialect": {"kau_gate_mode": "rerank", "dialect_normalization": "true"},
}


class _GateOnlyGenerator(BaseGenerator):
    """Stand-in 'LLM' for gate-only runs: never calls the network."""

    def __init__(self) -> None:
        super().__init__("gate_only", max_attempts=1)
        self.model = "none"

    def _invoke_once(self, system, prompt, timeout_seconds):
        return "(gate-only run: no generation)", False


def apply_overrides(pairs: dict[str, str]) -> dict[str, object]:
    """Set settings fields in place (typed); returns previous values."""
    settings = get_settings()
    previous: dict[str, object] = {}
    for key, raw in pairs.items():
        if not hasattr(settings, key):
            raise SystemExit(f"Unknown setting: {key}")
        current = getattr(settings, key)
        if isinstance(current, bool):
            value: object = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        previous[key] = current
        setattr(settings, key, value)
    return previous


def load_rows(path: Path, university: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if university in ("all", r["university"])]


def _evidence_text(result) -> str:
    parts = []
    for item in result.qa_items + result.doc_items:
        parts.append(item.document)
        parts.append(str(item.metadata.get("answer", "")))
        parts.append(" ".join(str(v) for v in item.metadata.values()))
    return "\n".join(parts)


def _keywords(row: dict) -> list[str]:
    return [k.strip() for k in row.get("expect_any", "").split("|") if k.strip()]


def _norm(text: str) -> str:
    """Spelling-insensitive form for keyword matching (ة/ه, أ/ا, ى/ي...).

    The Q&A data is written informally ("بطاقه جامعيه"), so a raw
    substring test missed correct evidence in the first run.
    """
    import re

    from src.lexical import normalize_token

    return " ".join(normalize_token(t) for t in re.findall(r"\S+", text))


def _contains_any(text: str, keys: list[str]) -> bool:
    normalized = _norm(text)
    return any(k in text or _norm(k) in normalized for k in keys)


def run(rows: list[dict], retriever, reranker, generate: bool) -> list[dict]:
    """Run every row through answer_question; returns result records."""
    chain = None if generate else [_GateOnlyGenerator()]
    out: list[dict] = []
    for row in rows:
        history = None
        if row.get("history_question") and row.get("history_answer"):
            if not generate:
                out.append({**row, "status": "skipped (follow-up needs --generate)"})
                continue
            history = [(row["history_question"], row["history_answer"])]
        started = time.perf_counter()
        # Gate-only runs must not inherit the request deadline: on a slow
        # CPU the stub "LLM" was skipped for budget and the row was lost
        # (first run). Real --generate runs keep the production deadline.
        deadline = (
            time.monotonic() + get_settings().request_deadline_s
            if generate else None
        )
        try:
            res = answer_question(
                row["question"], retriever, chain, university=row["university"],
                reranker=reranker, history=history, deadline=deadline,
            )
        except GenerationError as exc:
            out.append({**row, "status": f"generation_error: {exc}"[:200]})
            continue
        seconds = time.perf_counter() - started
        evidence = _evidence_text(res)
        keys = _keywords(row)
        hit = (not res.refused) and _contains_any(evidence, keys)
        rec = {
            **row,
            "status": "ok",
            "refused": res.refused,
            "gate": res.gate,
            "gate_top_score": round(res.gate_top_score, 4),
            "hit": hit if row["expect"] == "answer" else "",
            "correct": (
                hit if row["expect"] == "answer" else res.refused
            ),
            "search_query": res.search_query,
            "sources": " ; ".join(res.sources),
            "seconds": round(seconds, 3),
        }
        if generate:
            rec.update({
                "answer": res.answer,
                "provider": res.provider,
                "keyword_in_answer": _contains_any(res.answer, keys),
                "ungrounded_numbers": " ".join(res.ungrounded_numbers),
            })
        out.append(rec)
    return out


def summarize(records: list[dict], label: str = "") -> dict[str, dict]:
    """Print and return per-university metrics."""
    by_uni: dict[str, dict] = {}
    for uni in sorted({r["university"] for r in records}):
        recs = [r for r in records if r["university"] == uni and r.get("status") == "ok"]
        ans = [r for r in recs if r["expect"] == "answer"]
        ref = [r for r in recs if r["expect"] == "refuse"]
        hits = [r for r in ans if r["hit"] is True]
        m = {
            "n": len(recs),
            "answerable_pass": _rate([not r["refused"] for r in ans]),
            "hit_rate": _rate([r["hit"] is True for r in ans]),
            "refuse_correct": _rate([r["refused"] for r in ref]),
            "max_refuse_score": max((r["gate_top_score"] for r in ref), default=None),
            "min_hit_score": min((r["gate_top_score"] for r in hits), default=None),
            "median_seconds": (
                round(statistics.median(r["seconds"] for r in recs), 2) if recs else None
            ),
        }
        if any("answer" in r for r in recs):
            m["keyword_in_answer"] = _rate([r.get("keyword_in_answer") is True for r in ans])
            m["answers_with_ungrounded_numbers"] = sum(
                1 for r in recs if r.get("ungrounded_numbers")
            )
        by_uni[uni] = m
        print(f"\n[{label or 'run'}] {uni}: " + ", ".join(f"{k}={v}" for k, v in m.items()))
        misses = [r["id"] for r in ans if r["hit"] is not True]
        wrong_answers = [r["id"] for r in ref if not r["refused"]]
        if misses:
            print(f"  missed answerable: {', '.join(misses)}")
        if wrong_answers:
            print(f"  answered but should refuse: {', '.join(wrong_answers)}")
    return by_uni


def _rate(flags: list[bool]) -> str:
    return f"{sum(flags)}/{len(flags)}" if flags else "-"


def write_csv(records: list[dict], tag: str) -> Path:
    out_dir = get_settings().data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    keys: list[str] = []
    for r in records:
        keys.extend(k for k in r if k not in keys)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--university", default="all", choices=("all", "kau", "ksu"))
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--generate", action="store_true",
                        help="call the real provider chain (costs API calls)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="settings override, repeatable")
    parser.add_argument("--matrix", action="store_true",
                        help="gate-only comparison of the MATRIX configurations")
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    from src.chroma_store import get_client
    from src.embeddings import E5Embedder
    from src.retrieval import Retriever

    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
    retriever = Retriever(get_client(settings.chroma_dir), embedder)
    rows = load_rows(args.questions, args.university)
    print(f"{len(rows)} questions from {args.questions}")

    if args.matrix:
        apply_overrides({"general_fallback_mode": "off"})
        table: dict[str, dict] = {}
        for name, overrides in MATRIX.items():
            previous = apply_overrides(overrides)
            records = run(rows, retriever, reranker, generate=False)
            table[name] = summarize(records, name)
            write_csv(records, name)
            apply_overrides({k: str(v) for k, v in previous.items()})
        print("\n=== comparison (hits / correct refusals / median seconds) ===")
        for name, unis in table.items():
            cells = [f"{u}: hit {m['hit_rate']} refuse {m['refuse_correct']} "
                     f"{m['median_seconds']}s"
                     for u, m in unis.items()]
            print(f"{name:12s} " + " | ".join(cells))
        return

    overrides = dict(pair.split("=", 1) for pair in args.set)
    if not args.generate:
        # Gate-only runs measure the official-source gates; a live web /
        # LLM fallback would both cost calls and hide refusals.
        overrides.setdefault("general_fallback_mode", "off")
    apply_overrides(overrides)
    records = run(rows, retriever, reranker, generate=args.generate)
    summarize(records, "generate" if args.generate else "gate-only")
    tag = "gen" if args.generate else "gate"
    print(f"\nCSV: {write_csv(records, tag)}")


if __name__ == "__main__":
    main()
