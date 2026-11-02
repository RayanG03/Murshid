"""Run questions against the local Murshid API. Standard library only.

Only question/university/history are transmitted, never the answer key.
Results are saved after every request. No automatic semantic grading.
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

def request_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def run(base_url, questions, out, timeout=120):
    out.mkdir(parents=True, exist_ok=True)
    fields=['test_id','university','question','status','http_status','answer','refused','sources','source_details','seconds','request_id','error']
    records=[]
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as csvfile, (out/'results.jsonl').open('w',encoding='utf-8') as jsonfile:
        writer=csv.DictWriter(csvfile,fieldnames=fields); writer.writeheader(); csvfile.flush()
        for index,q in enumerate(questions,1):
            # Explicit allow-list: answer_key fields can never enter a request.
            payload={'question':q['question'],'university':q['university'],'history':[]}
            record={key:'' for key in fields}
            record.update(test_id=q['test_id'],university=q['university'],question=q['question'])
            started=time.perf_counter()
            try:
                response=request_json(base_url.rstrip('/')+'/ask',payload,timeout)
                if not isinstance(response,dict) or not isinstance(response.get('answer'),str) or not response['answer'].strip():
                    raise ValueError('Invalid response: missing or empty answer')
                record.update(status='ok',http_status=200,answer=response['answer'],refused=response.get('refused',''),sources=response.get('sources',[]),source_details=response.get('source_details',[]),request_id=response.get('request_id',''))
            except urllib.error.HTTPError as error:
                record.update(status='http_error',http_status=error.code,error=error.read().decode('utf-8',errors='replace')[:3000])
            except Exception as error:
                record.update(status='request_error',error=f'{type(error).__name__}: {error}')
            record['seconds']=round(time.perf_counter()-started,3)
            records.append(record)
            jsonfile.write(json.dumps(record,ensure_ascii=False)+'\n'); jsonfile.flush()
            csv_record={key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in record.items()}
            writer.writerow(csv_record); csvfile.flush()
            print(f"[{index:03}/{len(questions)}] {q['test_id']} {q['university']} {record['status']} {record['seconds']:.1f}s",flush=True)
    (out/'run_summary.json').write_text(json.dumps(dict(total=len(records),successful_http_answers=sum(r['status']=='ok' for r in records),request_errors=sum(r['status']!='ok' for r in records),note='HTTP success is not answer accuracy. Review answers and cited passages manually. Request errors count as failures in end-to-end reporting.'),ensure_ascii=False,indent=2),encoding='utf-8')
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--limit',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.limit<=100 or args.timeout<=0: parser.error('limit must be 1..100 and timeout must be positive')
    try:
        health=request_json(args.base_url.rstrip('/')+'/health',timeout=10)
    except Exception as error:
        print(f'Cannot reach backend. Start it first. {type(error).__name__}: {error}',file=sys.stderr)
        return 1
    details={}
    try: details=request_json(args.base_url.rstrip('/')+'/health/details',timeout=10)
    except Exception: pass
    questions=json.loads((HERE/'questions_only.json').read_text(encoding='utf-8'))[:args.limit]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=HERE/'results'/stamp
    out.mkdir(parents=True)
    (out/'run_settings.json').write_text(json.dumps(dict(start_utc=stamp,base_url=args.base_url,health=health,health_details=details,total=len(questions),history='empty for every request',execution='sequential, no automatic retries',answer_cache='not cleared; elapsed times may include cache hits'),ensure_ascii=False,indent=2),encoding='utf-8')
    run(args.base_url,questions,out,args.timeout)
    print(f'\nSaved results: {out}\nSend results.csv back for source-based review. No accuracy score has been assigned automatically.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
