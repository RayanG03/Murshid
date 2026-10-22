"""Load and validate the KAU audit workbook (read-only, backend/sources/)."""

from pathlib import Path

import pandas as pd

AUDIT_SHEET = "مراجعة عبدالعزيز"

COL_Q_ID = "Q_ID"
COL_QUESTION = "Question_Original"
COL_QUESTION_NORM = "Question_Normalized"
COL_ANSWER = "Answer"
COL_CATEGORY = "Category"
COL_STATUS = "حالة المطابقة مع جامعة الملك عبدالعزيز"
COL_KAU_ANSWER = "إجابة جامعة الملك عبدالعزيز"
COL_SOURCE = "المصدر الرسمي"

STATUS_MATCH = "مطابق بالمعنى"
STATUS_DIFFERENT = "غير مطابق"
STATUS_UNVERIFIED = "غير موثّق رسميًا"

EXPECTED_COUNTS = {
    STATUS_MATCH: 225,
    STATUS_DIFFERENT: 146,
    STATUS_UNVERIFIED: 203,
}


def load_audit(xlsx_path: Path) -> pd.DataFrame:
    """Load the audit sheet and verify the category split matches AGENTS.md."""
    df = pd.read_excel(xlsx_path, sheet_name=AUDIT_SHEET)
    required = [
        COL_Q_ID, COL_QUESTION, COL_QUESTION_NORM, COL_ANSWER, COL_CATEGORY,
        COL_STATUS, COL_KAU_ANSWER, COL_SOURCE,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Workbook missing expected columns: {missing}")

    counts = df[COL_STATUS].value_counts().to_dict()
    print(f"Workbook rows: {len(df)}")
    for status, expected in EXPECTED_COUNTS.items():
        actual = counts.get(status, 0)
        print(f"  {status}: {actual} (expected {expected})")
    if counts != EXPECTED_COUNTS:  # type: ignore[comparison-overlap]
        raise SystemExit(
            "FATAL: category counts do not match AGENTS.md "
            f"(225/146/203). Found: {counts}. Refusing to build datasets."
        )
    if df[COL_Q_ID].duplicated().any():
        raise SystemExit("FATAL: duplicate Q_ID values in audit sheet.")
    return df


def build_kau_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the audit into the verified KAU Q&A set and the unverified set.

    Verified answers follow AGENTS.md: use the original ``Answer`` for
    matching rows, and the KAU reviewer answer for non-matching rows.
    """
    verified_rows: list[dict] = []
    unverified_rows: list[dict] = []

    for row in df.to_dict("records"):
        record = {
            "Q_ID": row[COL_Q_ID],
            "question_original": row[COL_QUESTION],
            "question_normalized": row[COL_QUESTION_NORM],
            "category": row[COL_CATEGORY],
            "match_status": row[COL_STATUS],
            "answer_ksu_original": row[COL_ANSWER],
            "answer_kau_reviewer": row[COL_KAU_ANSWER],
            "source_url": row[COL_SOURCE],
        }
        if row[COL_STATUS] == STATUS_UNVERIFIED:
            record["answer"] = row[COL_ANSWER]
            unverified_rows.append(record)
        else:
            record["answer"] = (
                row[COL_ANSWER]
                if row[COL_STATUS] == STATUS_MATCH
                else row[COL_KAU_ANSWER]
            )
            verified_rows.append(record)

    verified = pd.DataFrame(verified_rows)
    unverified = pd.DataFrame(unverified_rows)
    return verified, unverified
