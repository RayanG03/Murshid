"""Read-only OCR audit; optional human reference WER and numeric occurrence metrics."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.config import get_settings
from src.text import normalize_digits_for_search


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right)+1))
    for i, x in enumerate(left, 1):
        current = [i]
        for j, y in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def compare(text: str, reference: str) -> dict:
    words=reference.split();observed=text.split()
    nums=lambda t: Counter(re.findall(r'\d+(?:[.,]\d+)?',normalize_digits_for_search(t)))
    expected, actual=nums(reference),nums(text);matches=sum((expected&actual).values())
    return {'wer':edit_distance(words,observed)/len(words) if words else None,
            'numeric_recall': matches/sum(expected.values()) if expected else None,
            'numeric_precision':matches/sum(actual.values()) if actual else None,
            'missing_numbers':list((expected-actual).elements()),'extra_numbers':list((actual-expected).elements())}


def audit(university: str, output: Path, reference_dir: Path | None = None) -> dict:
    base=get_settings().sources_dir/university.upper()/'transcribed'
    if output.resolve().is_relative_to(get_settings().sources_dir.resolve()):
        raise ValueError('Audit output must be outside read-only sources')
    pages=[]
    for p in sorted(base.rglob('page_*.txt')):
        t=p.read_text(encoding='utf-8');rel=p.relative_to(base)
        flags=[]
        if not t.strip():flags.append('blank')
        if '[ILLEGIBLE]' in t:flags.append('illegible')
        if '\ufffd' in t or '\x00' in t:flags.append('corrupt_characters')
        row={'path':str(rel),'characters':len(t),'flags':flags}
        ref=reference_dir/rel if reference_dir else None
        if ref and ref.is_file():row.update(compare(t,ref.read_text(encoding='utf-8')))
        pages.append(row)
    report={'university':university,'pages':len(pages),'warnings':sum(bool(p['flags']) for p in pages),
            'manual_references':sum('wer' in p for p in pages),'details':pages,
            'note':'Numeric overlap does not establish the rule attached to a number. Blank pages can be covers.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--university',required=True,choices=['ksu','kau'])
    p.add_argument('--reference-dir',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=audit(a.university,a.output or get_settings().data_dir/'quality'/f'{a.university}_ocr_audit.json',a.reference_dir)
    print(json.dumps({k:v for k,v in r.items() if k!='details'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
