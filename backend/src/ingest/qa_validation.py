"""Validate entire QA batches before embedding or touching an existing index."""
import hashlib

from src.data_prep.import_questions import official_urls


def qa_record_id(university: str, q_id: str) -> str:
    value = str(q_id)
    suffix = value if value.isdigit() else hashlib.sha256(value.encode()).hexdigest()[:24]
    return f'{university}_qa_{suffix}'


def validate_qa(df, university: str, expected_rows: int | None = None) -> None:
    required = {'Q_ID', 'question_original', 'answer'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError('Missing QA fields or zero rows; index left untouched')
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, got {len(df)}')
    for col in required:
        if df[col].astype(str).str.strip().eq('').any():raise ValueError(f'Blank {col}')
    if df.Q_ID.duplicated().any():raise ValueError('Duplicate Q_ID')
    if 'university' in df and not df.university.eq(university).all():raise ValueError('Mixed universities')
    if university == 'kau':
        for row in df.to_dict('records'):
            status = row.get('evidence_status') or row.get('match_status', '')
            eligible = status.startswith(('موثق','موثّق')) or status in ('verified','verified_conditional','مطابق بالمعنى','غير مطابق','مراجعة سابقة — مطابق بالمعنى','مراجعة سابقة — غير مطابق','مراجعة سابقة — إجابة معدلة لعبدالعزيز')
            if not eligible or not official_urls(str(row.get('source_url','')), university):
                raise ValueError(f'Unverified KAU record {row["Q_ID"]}')
