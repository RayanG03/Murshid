"""يترجم معرّفات gold في ملفات التقييم إلى معرّفات المقاطع الحالية، بمطابقة النص حرفياً.

إعادة التقطيع قد تغيّر المعرّفات. هذا السكربت لا يخمّن: كل معرّف لا يطابق نصُّه مقطعاً
واحداً بالضبط يُطبع ويبقى كما هو ليُسأل عنه صاحب المشروع. خريطة المقاطع التي تغيّر نصها
(بالمقطع الجديد الذي فيه معلومة الجواب المتوقع) تُمرَّر بـ --manual كملف JSON:
{"رقم السؤال": {"المعرّف القديم": ["الجديد", ...]}}.

ويضيف gold_refs بجانب gold_chunks: اسم الوثيقة والمادة أو القسم، ليقرأها الإنسان.

التشغيل:
    git show HEAD:data/chunks/ksu.jsonl HEAD:data/chunks/kau.jsonl > old_chunks.jsonl
    python eval/migrate_gold.py --old old_chunks.jsonl              عرض الخريطة فقط
    python eval/migrate_gold.py --old old_chunks.jsonl --write      تحديث ملفات التقييم
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

GOLD_FILES = [
    "test_questions_50.jsonl",
    "test_questions_new50.jsonl",
    "questions.jsonl",
    "student_questions.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_current_chunks() -> dict[str, dict]:
    chunks = {}
    for path in sorted(config.CHUNKS_DIR.glob("*.jsonl")):
        for chunk in read_jsonl(path):
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def ids_by_text(chunks: dict[str, dict]) -> dict[str, list[str]]:
    by_text = {}
    for chunk_id, chunk in chunks.items():
        by_text.setdefault(chunk["text"], []).append(chunk_id)
    return by_text


def map_gold_id(
    gold_id: str, old_chunks: dict, new_chunks: dict, new_by_text: dict, manual: dict
) -> list[str] | None:
    """المعرّفات الجديدة لمعرّف gold واحد، أو None إن لم يمكن تحديدها يقيناً.

    manual هنا خريطة السؤال نفسه: {"المعرّف القديم": ["الجديد", ...]}.
    """
    if gold_id in manual:
        return manual[gold_id]
    old_chunk = old_chunks.get(gold_id)
    if old_chunk is None:
        return [gold_id] if gold_id in new_chunks else None
    matches = new_by_text.get(old_chunk["text"], [])
    if len(matches) == 1:
        return matches
    return None


def gold_reference(chunk: dict) -> str:
    reference = f"{chunk['doc_title']} — {chunk.get('article_label') or chunk['unit_kind']}"
    if chunk["part"] != "1/1" and chunk.get("article_label"):
        reference += f" ({chunk['part']})"
    return reference


def migrate_question(
    question: dict, old_chunks: dict, new_chunks: dict, new_by_text: dict, manual: dict
) -> tuple[dict, list[str]]:
    """السؤال بمعرّفات gold الجديدة ومراجعها، وقائمة المعرّفات التي لم تُترجم."""
    new_gold = []
    unmapped = []
    question_manual = manual.get(question["id"], {})
    for gold_id in question.get("gold_chunks") or []:
        mapped = map_gold_id(gold_id, old_chunks, new_chunks, new_by_text, question_manual)
        if mapped is None:
            unmapped.append(gold_id)
            mapped = [gold_id]
        for chunk_id in mapped:
            if chunk_id not in new_gold:
                new_gold.append(chunk_id)
    migrated = {}
    for key, value in question.items():
        if key == "gold_refs":
            continue
        migrated[key] = new_gold if key == "gold_chunks" else value
        if key == "gold_chunks":
            references = []
            for chunk_id in new_gold:
                if chunk_id in new_chunks:
                    references.append(gold_reference(new_chunks[chunk_id]))
            migrated["gold_refs"] = references
    return migrated, unmapped


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--old", type=Path, required=True, help="مقاطع الفهرس قبل التغيير")
    parser.add_argument("--manual", type=Path, help="خريطة مؤكدة يدوياً (JSON)")
    parser.add_argument("--write", action="store_true", help="اكتب ملفات التقييم")
    args = parser.parse_args()

    old_chunks = {chunk["chunk_id"]: chunk for chunk in read_jsonl(args.old)}
    new_chunks = read_current_chunks()
    new_by_text = ids_by_text(new_chunks)
    manual = {}
    if args.manual:
        manual = json.loads(args.manual.read_text(encoding="utf-8"))

    all_unmapped = []
    for file_name in GOLD_FILES:
        path = config.EVAL_DIR / file_name
        migrated_rows = []
        for question in read_jsonl(path):
            migrated, unmapped = migrate_question(
                question, old_chunks, new_chunks, new_by_text, manual
            )
            migrated_rows.append(migrated)
            if question.get("gold_chunks") != migrated.get("gold_chunks"):
                print(f"{question['id']}: {question['gold_chunks']} → {migrated['gold_chunks']}")
            for gold_id in unmapped:
                all_unmapped.append(f"{file_name} {question['id']}: {gold_id}")
        if args.write:
            lines = [json.dumps(row, ensure_ascii=False) for row in migrated_rows]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nلم يُترجم {len(all_unmapped)} معرّفاً (يحتاج قرار صاحب المشروع):")
    for line in all_unmapped:
        print("  " + line)
    if not args.write:
        print("\nعرض فقط — أضف --write لتحديث ملفات التقييم")


if __name__ == "__main__":
    main()
