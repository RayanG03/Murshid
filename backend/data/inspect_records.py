"""Read-only inspection of KSU Chroma records for literal search needles.

Prints every matching record (ids, metadata, FULL document text, no
truncation) from ksu_qa, ksu_txt and ksu_pdf, grouped by needle then
collection, into data/inspect_records.md (UTF-8) plus a match-count table
on stdout. Purely diagnostic: reads via get_existing_collection, writes
nothing else, modifies no data.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.chroma_store import COL_KSU_PDF, COL_KSU_QA, COL_KSU_TXT
from src.chroma_store import get_client, get_existing_collection
from src.config import get_settings

NEEDLES: dict[str, list[str]] = {
    "1) نتائج": ["متى تظهر النتائج", "النتائج النهائية"],
    "2) صيفي": ["المواد الصيفيه", "الفصل الصيفي"],
    "3) مزدوجة/مشتركة": ["المزدوجة", "المشتركة", "درجة مزدوجة", "درجة مشتركة"],
}
COLLECTIONS: list[str] = [COL_KSU_QA, COL_KSU_TXT, COL_KSU_PDF]
OUT_PATH = "inspect_records.md"


def fetch_matches(collection, needle: str) -> tuple[list[str], list[dict], list[str]]:
    """(ids, metadatas, documents) of every record containing the needle."""
    res = collection.get(
        where_document={"$contains": needle},
        include=["documents", "metadatas"],
    )
    return list(res["ids"]), list(res["metadatas"]), list(res["documents"])


def main() -> None:
    settings = get_settings()
    client = get_client(settings.chroma_dir)
    collections: dict[str, object] = {}
    for name in COLLECTIONS:
        col = get_existing_collection(client, name)
        if col is None:
            raise SystemExit(f"collection missing: {name}")
        collections[name] = col

    lines: list[str] = ["# KSU record inspection (literal `$contains` needles)", ""]
    counts: dict[str, dict[str, int]] = {}

    for needle_group, group_needles in NEEDLES.items():
        lines.append(f"## {needle_group}")
        lines.append("")
        for needle in group_needles:
            lines.append(f"### needle: {needle!r}")
            lines.append("")
            for name in COLLECTIONS:
                ids, metas, docs = fetch_matches(collections[name], needle)
                counts.setdefault(needle, {})[name] = len(ids)
                lines.append(
                    f"match count: {len(ids)} in {name} for needle {needle!r}"
                )
                lines.append("")
                for rid, md, doc in zip(ids, metas, docs):
                    lines.append(f"- record id: `{rid}`")
                    lines.append(
                        "- metadata: "
                        + json.dumps(md, ensure_ascii=False, sort_keys=True)
                    )
                    lines.append("- document (FULL, untruncated):")
                    lines.append("```")
                    lines.append(str(doc))
                    lines.append("```")
                    lines.append("")
    lines.append("## Match count per needle per collection")
    lines.append("")
    lines.append("| needle | " + " | ".join(COLLECTIONS) + " | total |")
    lines.append("|---|---|---|---|")
    for needle in [n for ns in NEEDLES.values() for n in ns]:
        per = counts[needle]
        row_total = sum(per.values())
        lines.append(
            f"| {needle!r} | "
            + " | ".join(str(per[c]) for c in COLLECTIONS)
            + f" | {row_total} |"
        )

    out_path = Path(settings.data_dir) / OUT_PATH
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out_path}")
    print()
    print("| needle | " + " | ".join(COLLECTIONS) + " | total |")
    print("|---|---|---|---|")
    for needle in [n for ns in NEEDLES.values() for n in ns]:
        per = counts[needle]
        print(
            f"| {needle!r} | "
            + " | ".join(str(per[c]) for c in COLLECTIONS)
            + f" | {sum(per.values())} |"
        )


if __name__ == "__main__":
    main()
