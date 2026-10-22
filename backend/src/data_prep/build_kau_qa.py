"""Stage 2 entry point: build the KAU Q&A datasets from the audit workbook.

Writes:
- backend/data/kau_qa.csv          (371 verified pairs)
- backend/data/kau_unverified.csv  (203 unverified pairs, excluded from retrieval)

Run from backend/: uv run python -X utf8 -m src.data_prep.build_kau_qa
"""

from src.config import get_settings
from src.data_prep.workbook import (
    STATUS_DIFFERENT,
    STATUS_MATCH,
    build_kau_datasets,
    load_audit,
)


def main() -> None:
    """Build and write both KAU datasets, printing the category report."""
    settings = get_settings()
    xlsx_path = settings.sources_dir / "Final-Dataset_KAU_Audit.xlsx"
    if not xlsx_path.exists():
        raise SystemExit(f"FATAL: workbook not found at {xlsx_path}")

    df = load_audit(xlsx_path)
    verified, unverified = build_kau_datasets(df)

    from_original = int((verified["match_status"] == STATUS_MATCH).sum())
    from_kau = int((verified["match_status"] == STATUS_DIFFERENT).sum())

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = settings.data_dir / "kau_qa.csv"
    unv_path = settings.data_dir / "kau_unverified.csv"
    verified.to_csv(qa_path, index=False, encoding="utf-8-sig")
    unverified.to_csv(unv_path, index=False, encoding="utf-8-sig")

    print(f"\nVerified KAU Q&A: {len(verified)} pairs -> {qa_path}")
    print(f"  answers from original `Answer`      : {from_original}")
    print(f"  answers from KAU reviewer answer    : {from_kau}")
    print(f"Unverified (excluded from retrieval)  : {len(unverified)} -> {unv_path}")
    total = len(verified) + len(unverified)
    print(f"Total rows accounted for             : {total}")
    if total != len(df):
        raise SystemExit("FATAL: row accounting mismatch.")


if __name__ == "__main__":
    main()
