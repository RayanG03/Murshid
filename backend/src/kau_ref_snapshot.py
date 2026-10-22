"""KAU retrieval regression snapshot over every query in relevance_probes.

Captures, for each probe query on both sides (qa / docs): collection name,
score, metadata id fields and the first 80 chars of the document. The KAU
collections are never rebuilt, so an identical snapshot after a refactor
proves behavior is unchanged.

Usage from backend/:
  uv run python -X utf8 -m src.kau_ref_snapshot --out data/kau_probe_snapshot.json
  uv run python -X utf8 -m src.kau_ref_snapshot --check data/kau_probe_snapshot.json
"""

import argparse
import json
from pathlib import Path

from src.chroma_store import get_client
from src.config import get_settings
from src.embeddings import E5Embedder
from src.relevance_probes import DOMAIN_ADJACENT_WRONG, OFF_TOPIC, RELEVANT
from src.retrieval import Retriever

CRYPTO = "sha256"


def _side_payload(results) -> list[dict]:
    """Snapshot one retrieval side: id-ish fields, score, document head."""
    payload: list[dict] = []
    for r in results:
        md = r.metadata
        identifier = (
            md.get("q_id")
            if md.get("type") in ("qa", None)
            else None
        )
        payload.append(
            {
                "collection": r.source_type,
                "score": round(r.score, 6),
                "q_id": identifier,
                "unit": md.get("unit") or md.get("page_title") or md.get("page"),
                "doc": r.document[:80],
            }
        )
    return payload


def capture_snapshot(retriever: Retriever) -> dict:
    """Return the full snapshot dict for the three probe categories."""
    snapshot: dict[str, dict] = {}
    for category, queries in (
        ("RELEVANT", RELEVANT),
        ("OFF-TOPIC", OFF_TOPIC),
        ("DOMAIN-ADJACENT-WRONG", DOMAIN_ADJACENT_WRONG),
    ):
        snapshot[category] = {}
        for query in queries:
            qa, docs = retriever.retrieve_split(query, university="kau")
            snapshot[category][query] = {
                "qa": _side_payload(qa),
                "docs": _side_payload(docs),
            }
    return snapshot


def diff(snapshot_a: dict, snapshot_b: dict) -> list[str]:
    """Return a list of human-readable differences (empty when equal)."""
    problems: list[str] = []
    for category in sorted(set(snapshot_a) | set(snapshot_b)):
        for query in sorted(
            set(snapshot_a.get(category, {})) | set(snapshot_b.get(category, {}))
        ):
            a = snapshot_a.get(category, {}).get(query, {})
            b = snapshot_b.get(category, {}).get(query, {})
            for side in ("qa", "docs"):
                if a.get(side) != b.get(side):
                    problems.append(f"{category} | {query[:50]} | side {side} differs")
                    if a.get(side) != b.get(side):
                        for x, y in zip(
                            a.get(side, []), b.get(side, [])
                        ):
                            if x != y:
                                problems.append(f"    got: {x}\n    want: {y}")
        if category not in snapshot_a or category not in snapshot_b:
            problems.append(f"category {category} missing from one snapshot")
    return problems


def main() -> None:
    """Write or check the KAU retrieval snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="write snapshot JSON here")
    parser.add_argument("--check", type=Path, help="compare live retrieval to this file")
    args = parser.parse_args()
    if not (args.out or args.check):
        parser.error("provide --out or --check")

    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    retriever = Retriever(client, embedder)
    live = capture_snapshot(retriever)

    if args.out:
        out_path = args.out
        out_path.write_text(
            json.dumps(live, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        total = sum(len(v) for cat in live.values() for v in cat.values())
        print(f"snapshot written: {out_path} ({total} side-lists)")
        return
    baseline = json.loads(args.check.read_text(encoding="utf-8"))
    problems = diff(baseline, live)
    if problems:
        print(f"DIFF: {len(problems)} problems")
        for p in problems:
            print("  ", p)
        raise SystemExit(1)
    print("DIFF: EMPTY (KAU retrieval regression clean)")


if __name__ == "__main__":
    main()
