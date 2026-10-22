"""KSU retrieval-gate calibration probes (step 2 of the KSU stage).

Probe categories for university="ksu":
- RELEVANT: paraphrase topics verified present in ksu_qa / ksu_txt / ksu_pdf
  (Q&A-flavored and doc-flavored topics are mixed on purpose).
- OFF-TOPIC: shared with src.relevance_probes (same six queries).
- DOMAIN_ADJACENT_WRONG: same-domain topics verified ABSENT from all three
  current collections. The weighted-percentage candidate was dropped after
  verification (ksu_txt_0000 answers 90%/95% for the تميّز programs) and the
  external-transfer candidate was dropped (the 43-row التحويل الداخلي
  والخارجي ksu_qa category covers it).

modes (from backend/):
  uv run python -X utf8 -m src.ksu_probes --verify     # per-query top-hit evidence
  uv run python -X utf8 -m src.ksu_probes --calibrate  # seed-42 TUNE/TEST report
"""

import argparse
import random
from collections.abc import Sequence

from src.chroma_store import get_client
from src.config import get_settings
from src.context_assembly import gate_side
from src.embeddings import E5Embedder
from src.relevance_probes import OFF_TOPIC
from src.retrieval import Retrieved, Retriever

SEED = 42

RELEVANT_KSU: list[str] = [
    "وش شروط التحويل من كلية لكلية ثانية داخل جامعة الملك سعود؟",
    "وشو المعدل الفصلي ووش الفرق بينه وبين المعدل التراكمي؟",
    "متى يحرم الطالب من دخول الاختبار النهائي في المقرر؟",
    "كم نسبة الغياب المسموحة في المقرر قبل الحرمان من الاختبار النهائي؟",
    "كيف اقدم طلب تأجيل دراستي للفصل الدراسي؟",
    "وش شروط التخصيص في الكليات بعد السنة الأولى المشتركة؟",
    "وش اجراءات معادلة المواد للمتحول من داخل الجامعة؟",
    "كيف احصل على البطاقة الجامعية الجديدة؟",
    "متى تنقطع المكافأة الشهرية عن الطالب؟",
    "وش شروط مرتبة الشرف عند التخرج؟",
    "كم الحد الأقصى لعدد الوحدات الدراسية في الفصل الدراسي؟",
    "وش شروط الالتحاق ببرنامج التخصص الفرعي؟",
    "وش الفرق بين الدرجة المزدوجة والدرجة المشتركة؟",
    "وش نظام نقاط الخروج في البرامج الدراسية؟",
    "وش شروط الاعتذار عن اختبار المقرر النهائي بعذر مقبول؟",
]

# Evidence notes (verified 2026-10-19 against the current collections;
# keyword-presence + doc-head inspection over questions AND answers):
DOMAIN_ADJACENT_WRONG_KSU: list[str] = [
    # Calendar dates: "تقويم" appears only as the generic phrase (ksu_qa_44 /
    # ksu_qa_124 "راجع التقويم") and in semester-term definitions in ksu_pdf;
    # the actual calendar document is NOT ingested (AGENTS known gap).
    "متى تبدأ اختبارات الدوري الأول للفصل الدراسي الأول في التقويم الجامعي؟",
    "متى إجازة نهاية الفصل الدراسي الثاني في التقويم الجامعي؟",
    "ما آخر يوم للحذف والإضافة في التقويم الدراسي للفصل الثاني؟",
    "ما مواعيد التسجيل للفصل الصيفي في التقويم الجامعي؟",
    # Results dates: the two ksu_qa "نتائج" hits carry no dates.
    "متى تنزل نتائج الاختبارات النهائية؟",
    # Diploma fees: "دبلوم" hits are مكافأة-QA (ksu_qa_107), regulation
    # definitions (ksu_pdf المادة الأولى), and acceptance rules — no fees.
    "كم رسوم برامج الدبلوم في جامعة الملك سعود؟",
    # International students: ksu_pdf hits are المادة 14/16 shared-program
    # wording; ksu_qa hits are unrelated. No student-fee content anywhere.
    "كم رسوم الدراسة للطلاب الدوليين؟",
    # Housing fees: ksu_qa_503 only says "رسوم رمزيه" with no amounts.
    "كم رسوم السكن الجامعي الشهرية وكيف أدفعها؟",
    # Buses: حافل/نقل keyword scan finds no timetable content.
    "ما جدول مواعيد الحافلات الجامعية؟",
    # Clinic: "عياد" appears once in ksu_qa (rewards context) and once in
    # ksu_pdf (definition block); no services/hours anywhere.
    "ما خدمات العيادة الجامعية ومواعيد عملها؟",
    # Library: ksu_txt_0038 lists the university card's uses (incl. the
    # library) but has no location/hours content.
    "وين مكتبة الجامعة ومواعيد فتحها؟",
    # Coop training: only passing mentions inside المادة 14 program rules.
    "وش شروط التدريب التعاوني وساعاته؟",
]

UNIVERSITY = "ksu"

OFF_TOPIC_KSU: list[str] = list(OFF_TOPIC)


def split_tune_test(
    queries: Sequence[str], seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 TUNE/TEST split (per category)."""
    items = sorted(queries)
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = (len(items) + 1) // 2
    return items[:cut], items[cut:]


def _scores(retriever: Retriever, query: str, k: int = 3) -> dict[str, list[float]]:
    """Top-k cosine scores per side (qa / docs) for one probe."""
    qa, docs = retriever.retrieve_split(query, k=k, university=UNIVERSITY)
    return {
        "qa": [r.score for r in qa],
        "docs": [r.score for r in docs],
    }


def _top_evidence(retriever: Retriever, query: str, k: int = 3) -> dict[str, list]:
    """Top-k (collection, id, score, doc_head) per side for verification."""
    qa, docs = retriever.retrieve_split(query, k=k, university=UNIVERSITY)
    return {
        "qa": [(r.source_type, r.metadata.get("q_id", "?"), r.score, r.document[:70]) for r in qa],
        "docs": [(r.source_type, r.metadata.get("unit") or r.metadata.get("page_title")
                  or r.metadata.get("source_file", "?"), r.score, r.document[:70]) for r in docs],
    }


def _gate_pass(
    scores: list[float], floor: float, strong: float, gap: float
) -> tuple[bool, str]:
    """Apply the context gate to a bare score list (max_items irrelevant)."""
    items = [
        Retrieved(source_type="probe", score=s, document="", metadata={}) for s in scores
    ]
    kept, reason = gate_side(items, floor, strong, gap, max_items=999)
    return bool(kept), reason


def _side_pass(sides: dict[str, list[float]], gates: dict[str, float]) -> tuple[bool, str, str]:
    """Answer-level verdict for one query under a gate quartet."""
    qa_ok, qa_reason = _gate_pass(
        sides["qa"], gates["floor_qa"], gates["strong_leader"], gates["gap"]
    )
    doc_ok, doc_reason = _gate_pass(
        sides["docs"], gates["floor_docs"], gates["strong_leader"], gates["gap"]
    )
    reasons = []
    if not qa_ok:
        reasons.append(f"qa:{qa_reason}")
    if not doc_ok:
        reasons.append(f"doc:{doc_reason}")
    answered = qa_ok or doc_ok
    return answered, ("pass" if answered else ",".join(reasons)), ""


def run_verify() -> None:
    """Print per-query top-hit evidence for every probe category."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    retriever = Retriever(get_client(settings.chroma_dir), embedder)
    for category, queries in (
        ("RELEVANT", RELEVANT_KSU),
        ("OFF-TOPIC", OFF_TOPIC_KSU),
        ("DOMAIN-ADJACENT-WRONG", DOMAIN_ADJACENT_WRONG_KSU),
    ):
        print(f"\n=== {category} ===")
        for query in queries:
            print(f"\nQ: {query}")
            evidence = _top_evidence(retriever, query)
            for side in ("qa", "docs"):
                hits = evidence[side]
                if not hits:
                    print(f"  {side}: (no hits)")
                    continue
                for col, ident, score, head in hits:
                    print(f"  {side} | {col} | {ident} | {score:.4f} | {head!r}")


def _report_table(
    retriever: Retriever, queries: list[str], title: str
) -> None:
    """Print per-query per-side top-3 scores."""
    print(f"\n--- {title} ---")
    for q in queries:
        scores = _scores(retriever, q)
        qa_head = ",".join(f"{s:.3f}" for s in scores["qa"][:3])
        doc_head = ",".join(f"{s:.3f}" for s in scores["docs"][:3])
        print(f"qa[{qa_head}] doc[{doc_head}] | {q[:64]}")


def _tune_margin(
    tune_rows: dict[str, dict[str, dict[str, list[float]]]],
    floors: dict[str, float],
) -> float:
    """Floor-distance margin of a candidate quartet on the TUNE slice.

    Heuristic on floors only (the grid itself already enforces the exact
    leader/gap behaviour for every TUNE query):
    - relevant margin: min over RELEVANT of max over sides of (top - floor);
    - refusal margin: min over OFF/DAW of min over non-empty sides of
      (floor - top).
    Empty sides contribute nothing on their side.
    """
    margin = 1.0
    for sides in tune_rows["RELEVANT"].values():
        per_side = []
        for side, floor in (("qa", floors["floor_qa"]), ("docs", floors["floor_docs"])):
            if sides[side]:
                per_side.append(max(sides[side]) - floor)
        margin = min(margin, max(per_side))
    for category in ("OFF-TOPIC", "DOMAIN-ADJACENT-WRONG"):
        for sides in tune_rows[category].values():
            for side, floor in (("qa", floors["floor_qa"]), ("docs", floors["floor_docs"])):
                if sides[side]:
                    margin = min(margin, floor - max(sides[side]))
    return margin


def _choose_thresholds(
    tune_rows: dict[str, dict[str, dict[str, list[float]]]],
) -> dict | None:
    """Grid-search gate values on TUNE only.

    Constraint: every TUNE RELEVANT query must pass on >= 1 side; every
    OFF-TOPIC / DOMAIN-ADJACENT-WRONG query must be refused on both sides.
    Among satisfying quartets, maximize the floor margin and prefer the
    middle of that optimum.
    """
    floors = [0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86]
    leaders = [0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]
    gaps = [0.03, 0.04, 0.05, 0.06]
    tuning_queries = {
        "RELEVANT": tune_rows["RELEVANT"],
        "REFUSAL": {
            q: sides
            for category in ("OFF-TOPIC", "DOMAIN-ADJACENT-WRONG")
            for q, sides in tune_rows[category].items()
        },
    }

    passing: list[tuple[float, float, float, float, float]] = []
    for fq in floors:
        for fd in floors:
            for leader in leaders:
                for gap in gaps:
                    gates = {
                        "floor_qa": fq, "floor_docs": fd,
                        "strong_leader": leader, "gap": gap,
                    }
                    ok = True
                    for sides in tuning_queries["RELEVANT"].values():
                        qa_ok, _ = _gate_pass(sides["qa"], fq, leader, gap)
                        doc_ok, _ = _gate_pass(sides["docs"], fd, leader, gap)
                        if not (qa_ok or doc_ok):
                            ok = False
                            break
                    if ok:
                        for sides in tuning_queries["REFUSAL"].values():
                            qa_ok, _ = _gate_pass(sides["qa"], fq, leader, gap)
                            doc_ok, _ = _gate_pass(sides["docs"], fd, leader, gap)
                            if qa_ok or doc_ok:
                                ok = False
                                break
                    if ok:
                        margin = _tune_margin(tune_rows, gates)
                        passing.append((margin, fq, fd, leader, gap))
    if not passing:
        return None
    best = max(p[0] for p in passing)
    optimal = [p for p in passing if abs(p[0] - best) < 1e-9]
    # Middle-most optimum keeps the gate centered in its feasible window.
    margin, fq, fd, leader, gap = sorted(
        optimal, key=lambda p: (abs(p[1] - 0.83), abs(p[2] - 0.83), p[4])
    )[0]
    return {
        "floor_qa": fq, "floor_docs": fd, "strong_leader": leader, "gap": gap,
        "margin": margin, "feasible_tuples": len(optimal),
    }


def run_calibrate() -> None:
    """Seed-42 TUNE/TEST calibration with the KAU baseline comparison."""
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    retriever = Retriever(get_client(settings.chroma_dir), embedder)

    split_map: dict[str, tuple[list[str], list[str]]] = {}
    tune_rows: dict[str, dict[str, dict[str, list[float]]]] = {}
    for category, queries in (
        ("RELEVANT", RELEVANT_KSU),
        ("OFF-TOPIC", OFF_TOPIC_KSU),
        ("DOMAIN-ADJACENT-WRONG", DOMAIN_ADJACENT_WRONG_KSU),
    ):
        tune, test = split_tune_test(queries)
        split_map[category] = (tune, test)
        print(f"{category}: {len(tune)} TUNE / {len(test)} TEST")
        tune_rows[category] = {q: _scores(retriever, q) for q in tune}
        _report_table(retriever, tune, f"{category} TUNE (per-side top-3)")

    for category, (_tune, test) in split_map.items():
        _report_table(retriever, test, f"{category} TEST (per-side top-3)")

    choice = _choose_thresholds(tune_rows)
    if choice is None:
        print("\nNO CLEAN THRESHOLD on TUNE: distributions overlap -> "
              "'ksu' gates stay un-calibrated (config keeps raising).")
    else:
        print(f"\nCHOSEN on TUNE: floor_qa={choice['floor_qa']} "
              f"floor_docs={choice['floor_docs']} "
              f"strong_leader={choice['strong_leader']} gap={choice['gap']} "
              f"| min floor margin {choice['margin']:.4f} "
              f"| feasible quartets {choice['feasible_tuples']}")

    def evaluate_test(category: str, queries: list[str], gates: dict[str, float]) -> tuple[int, list[str]]:
        """(wrong_query_count, per-query lines) over the TEST slice."""
        wrong = 0
        lines: list[str] = []
        for q in queries:
            sides = _scores(retriever, q)
            answered, why, _state = _side_pass(sides, gates)
            if category == "RELEVANT":
                if not answered:
                    wrong += 1
                    lines.append(f"REFUSED | {q[:64]} | {why}")
            elif answered:
                wrong += 1
                lines.append(
                    f"PASSED-GATE | {q[:64]} "
                    f"| qa top {sides['qa'][:1]} doc top {sides['docs'][:1]}"
                )
        return wrong, lines

    if choice is None:
        print("\n=== TEST with TUNE-chosen KSU gates: skipped (none chosen) ===")
    else:
        gates = {
            key: choice[key]
            for key in ("floor_qa", "floor_docs", "strong_leader", "gap")
        }
        print(f"\n=== TEST with TUNE-chosen KSU gates {gates} ===")
        for category, (_tune, test) in split_map.items():
            wrong, lines = evaluate_test(category, test, gates)
            print(f"{category} TEST | n={len(test)} | wrong={wrong}")
            for line in lines:
                print("   ", line)

    kau_gates = {
        "floor_qa": settings.relevance_floor_qa,
        "floor_docs": settings.relevance_floor_docs,
        "strong_leader": settings.gate_strong_leader,
        "gap": settings.gate_gap,
    }
    print(f"\n=== TEST with KAU thresholds {kau_gates} (baseline) ===")
    for category, (_tune, test) in split_map.items():
        wrong, lines = evaluate_test(category, test, kau_gates)
        print(f"{category} TEST | n={len(test)} | wrong={wrong}")
        for line in lines:
            print("   ", line)


def main() -> None:
    """CLI entry: --verify or --calibrate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="per-query evidence")
    parser.add_argument("--calibrate", action="store_true", help="TUNE/TEST report")
    args = parser.parse_args()
    if not (args.verify or args.calibrate):
        parser.error("provide --verify or --calibrate")
    if args.verify:
        run_verify()
    else:
        run_calibrate()


if __name__ == "__main__":
    main()
