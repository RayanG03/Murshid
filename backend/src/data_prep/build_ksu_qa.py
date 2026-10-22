"""Build the KSU Q&A dataset from the original columns of the audit workbook.

All 574 rows use the KSU answer set (``Answer``): the workbook audits the KSU
dataset against KAU regulations, and the 146 rows marked "غير مطابق" carry a
separate KAU replacement answer, which is NOT copied here. Rows marked
"غير موثّق رسميًا" are unverified for KAU only and are included. The KAU
source URL ("المصدر الرسمي") is a kau.edu.sa citation and is dropped.

Writes backend/data/ksu_qa.csv.

Run from backend/: uv run python -X utf8 -m src.data_prep.build_ksu_qa
"""

import pandas as pd

from src.config import get_settings
from src.data_prep.workbook import (
    COL_ANSWER,
    COL_CATEGORY,
    COL_Q_ID,
    COL_QUESTION,
    COL_QUESTION_NORM,
    load_audit,
)

EXPECTED_ROWS = 574


def build_ksu_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the five KSU columns for all rows (no KAU audit columns)."""
    records = [
        {
            "Q_ID": row[COL_Q_ID],
            "question_original": row[COL_QUESTION],
            "question_normalized": row[COL_QUESTION_NORM],
            "answer": row[COL_ANSWER],
            "category": row[COL_CATEGORY],
        }
        for row in df.to_dict("records")
    ]
    return pd.DataFrame(records)


def print_report(ksu: pd.DataFrame) -> None:
    """Print totals, empty answers, duplicate questions, category counts."""
    print(f"\nKSU Q&A rows: {len(ksu)}")
    if len(ksu) != EXPECTED_ROWS:
        raise SystemExit(
            f"FATAL: expected {EXPECTED_ROWS} KSU rows, found {len(ksu)}."
        )
    empty_answers = int(ksu["answer"].isna().sum())
    empty_answers += int((ksu["answer"].astype(str).str.strip() == "").sum())
    print(f"empty/NaN answers: {empty_answers}")
    dup_questions = int(ksu["question_original"].duplicated().sum())
    print(f"duplicate question_original rows: {dup_questions}")
    print("per-category counts:")
    for category, count in ksu["category"].value_counts().items():
        print(f"  {category}: {count}")


def main() -> None:
    """Load the workbook, build the KSU Q&A dataset, write and report."""
    settings = get_settings()
    xlsx_path = settings.sources_dir / "Final-Dataset_KAU_Audit.xlsx"
    if not xlsx_path.exists():
        raise SystemExit(f"FATAL: workbook not found at {xlsx_path}")

    df = load_audit(xlsx_path)
    ksu = build_ksu_dataset(df)
    print_report(ksu)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.data_dir / "ksu_qa.csv"
    ksu.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
