"""Import CSV/TSV/XLSX QA batches; preserve all rows and evidence provenance.

Dry-run by default. Only eligible KAU answers enter the serving CSV.
URLs and status labels are declarations from the input, not new validation.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from src.config import get_settings
from src.text import normalize_digits_for_search

FIELDS = ['Q_ID', 'question_original', 'question_normalized', 'answer', 'category',
          'university', 'match_status', 'source_url', 'evidence_status', 'source_locator',
          'original_question', 'source_codes', 'verification_date', 'input_row',
          'review_reason', 'raw_record_json']
ALIASES = {
 'Q_ID': ['المعرف الموحد', 'Q_ID', 'id'],
 'question_original': ['question_original', 'Question', 'question', 'السؤال'],
 'answer': ['answer', 'Answer', 'الإجابة'],
 'category': ['التصنيف السابق', 'category', 'Category'],
 'source_url': ['روابط المصادر', 'المصدر الرسمي', 'source_url'],
 'source_locator': ['موضع الدليل', 'source_locator'],
 'evidence_status': ['حالة التوثيق والمراجعة', 'evidence_status'],
 'match_status': ['حالة المراجعة الأصلية', 'حالة المطابقة مع جامعة الملك عبدالعزيز', 'match_status'],
 'original_question': ['السؤال الأصلي كاملًا', 'original_question'],
 'source_codes': ['رموز المصادر', 'source_codes'],
 'verification_date': ['تاريخ التحقق المسجل', 'verification_date'],
}

def official_urls(value: str, university: str) -> list[str]:
    """Validate the exact official host suffix, never an arbitrary substring."""
    out = []
    for u in re.findall(r'https?://[^\s<>]+', value):
        host = (urlparse(u).hostname or '').lower()
        domain = university + '.edu.sa'
        if host == domain or host.endswith('.' + domain):
            out.append(u)
    return out

def read_rows(path: Path, university: str, sheet: str | None = None) -> tuple[list[dict], list[dict]]:
    """Read questions and source catalog; detect the header in first 30 rows."""
    sources = []
    if path.suffix.lower() == '.xlsx':
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        preferred = ['أسئلة سعود'] if university == 'ksu' else ['أسئلة عبدالعزيز', 'مراجعة عبدالعزيز']
        ws = wb[sheet] if sheet else next((wb[n] for n in preferred if n in wb.sheetnames), wb.worksheets[0])
        values = list(ws.values)
        for other in wb:
            if other.title == ws.title: continue
            vals = list(other.values)
            for i, row in enumerate(vals[:30]):
                if 'رمز المصدر' in row:
                    sources.extend({str(k): '' if v is None else str(v) for k, v in zip(row, r) if k}
                                   for r in vals[i+1:] if any(v is not None for v in r))
                    break
        wb.close()
    else:
        with path.open(encoding='utf-8-sig', newline='') as f:
            values = list(csv.reader(f, delimiter='\t' if path.suffix.lower() == '.tsv' else ','))
    for i, row in enumerate(values[:30]):
        names = {str(v) for v in row if v is not None}
        if names & {'المعرف الموحد', 'Q_ID', 'question_original', 'Question', 'question'}:
            headers = [str(v).strip() if v is not None else '' for v in row]
            records = []
            for n, vals in enumerate(values[i+1:], i+2):
                if not any(v is not None and str(v).strip() for v in vals): continue
                records.append({**{k: '' if v is None else str(v).strip() for k, v in zip(headers, vals) if k}, '_input_row': n})
            return records, sources
    raise ValueError('No recognized question header in first 30 rows')

def _pick(raw: dict, names: list[str]) -> str:
    return next((str(raw[n]).strip() for n in names if raw.get(n) is not None and str(raw[n]).strip()), '')

def prepare_rows(rows: list[dict], university: str, allow_legacy_reviewed: bool = False) -> tuple[list[dict], list[dict]]:
    """Account for every input row. Conflicts are quarantined on both sides."""
    prepared = []
    for n, raw in enumerate(rows, 1):
        row = {f: _pick(raw, ALIASES.get(f, [f])) for f in FIELDS}
        if university == 'ksu':
            row['question_original'] = _pick(raw, ['سؤال سعود الأصلي كاملًا', 'السؤال الأصلي كاملًا', 'Question', 'question_original', 'question'])
            row['answer'] = _pick(raw, ['إجابة سعود الأصلية', 'Answer', 'answer'])
            row['evidence_status'] = _pick(raw, ['حالة توثيق إجابة سعود', 'ksu_evidence_status']) or 'original_unreviewed'
            # A comparison workbook does NOT establish a KSU source.
            row['source_url'] = '\n'.join(official_urls(_pick(raw, ['روابط مصادر سعود', 'ksu_source_url', 'source_url']), 'ksu'))
            row['source_locator'] = _pick(raw, ['موضع دليل سعود', 'ksu_source_locator'])
            row['source_codes'] = _pick(raw, ['رموز مصادر سعود'])
            row['verification_date'] = _pick(raw, ['تاريخ تحقق سعود'])
            row['match_status'] = ''
        else:
            row['question_original'] = _pick(raw, ['سؤال عبدالعزيز', 'question_original', 'Question', 'question'])
            row['answer'] = _pick(raw, ['إجابة عبدالعزيز', 'إجابة جامعة الملك عبدالعزيز', 'answer', 'Answer'])
        row['Q_ID'] = row['Q_ID'] or university.upper()+'-'+hashlib.sha256(row['question_original'].encode()).hexdigest()[:16]
        row['university'] = university
        row['input_row'] = str(raw.get('_input_row', n))
        row['question_normalized'] = normalize_digits_for_search(row['question_original'])
        row['original_question'] = row['original_question'] or row['question_original']
        row['raw_record_json'] = json.dumps(raw, ensure_ascii=False)
        reasons = []
        if not row['question_original'] or not row['answer']: reasons.append('missing_question_or_answer')
        declared = raw.get('university', university)
        if declared != university or row['Q_ID'].upper().startswith(('KSU-', 'KAU-')) and not row['Q_ID'].upper().startswith(university.upper()+'-'):
            reasons.append('university_mismatch')
        if university == 'kau':
            status = row['evidence_status']
            current = status.startswith(('موثق', 'موثّق')) or status in ('verified', 'verified_conditional')
            legacy = status in ('مراجعة سابقة — مطابق بالمعنى', 'مراجعة سابقة — غير مطابق', 'مراجعة سابقة — إجابة معدلة لعبدالعزيز') or (not status and row['match_status'] in ('مطابق بالمعنى', 'غير مطابق'))
            if legacy and not status:
                row['evidence_status'] = row['match_status']
            if not current and not (allow_legacy_reviewed and legacy and 'غير موث' not in status):
                reasons.append('evidence_not_eligible')
            if not official_urls(row['source_url'], 'kau'): reasons.append('missing_official_kau_url')
        row['review_reason'] = ';'.join(reasons)
        prepared.append(row)
    by_id, by_question = defaultdict(list), defaultdict(list)
    for row in prepared:
        by_id[row['Q_ID']].append(row)
        question_key = ' '.join(row['question_normalized'].split())
        if question_key:
            by_question[question_key].append(row)
    for group in by_id.values():
        if len(group)>1:
            for row in group: row['review_reason'] += ';duplicate_id'
    for group in by_question.values():
        if len({r['answer'] for r in group}) > 1:
            for row in group: row['review_reason'] += ';conflicting_answers'
    return [r for r in prepared if not r['review_reason']], [r for r in prepared if r['review_reason']]

def write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n");w.writeheader();w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()

def import_file(path: Path, university: str, output_dir: Path, *, write: bool = False,
                allow_legacy_reviewed: bool = False, sheet: str | None = None) -> dict:
    rows, sources = read_rows(path, university, sheet)
    accepted, review = prepare_rows(rows, university, allow_legacy_reviewed)
    report = {'file': path.name, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'university': university, 'input_rows': len(rows), 'accepted': len(accepted),
              'review': len(review), 'source_catalog_rows': len(sources),
              'allow_legacy_reviewed': allow_legacy_reviewed,
              'status_counts': dict(Counter(r['evidence_status'] for r in accepted+review)),
              'reason_counts': dict(Counter(reason for r in review for reason in r['review_reason'].split(';') if reason)),
              'note': 'Input evidence labels preserved; no new official verification. KSU comparison links are not KSU evidence.'}
    assert len(accepted)+len(review)==len(rows)
    if write:
        if not accepted:raise ValueError('No eligible records; existing serving CSV left untouched')
        write_csv(output_dir/f'{university}_qa.csv', accepted)
        write_csv(output_dir/f'{university}_import_review.csv', review)
        write_csv(output_dir/f'{university}_all_questions.csv', sorted(accepted+review, key=lambda r: int(r['input_row'])))
        if sources:write_csv(output_dir/f'{university}_source_catalog.csv', sources, list(sources[0]))
        (output_dir/f'{university}_import_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input', type=Path)
    p.add_argument('--university',required=True,choices=['ksu','kau']);p.add_argument('--sheet')
    p.add_argument('--output-dir',type=Path,default=get_settings().data_dir)
    p.add_argument('--write',action='store_true');p.add_argument('--allow-legacy-reviewed',action='store_true')
    a=p.parse_args();print(json.dumps(import_file(a.input,a.university,a.output_dir,write=a.write,allow_legacy_reviewed=a.allow_legacy_reviewed,sheet=a.sheet),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
