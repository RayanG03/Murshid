"""Regression tests for evidence integrity and the supplied batches."""
import csv
import shutil
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.config import BACKEND_DIR, get_settings
from src.data_prep.import_questions import import_file, official_urls, prepare_rows, read_rows
from src.data_prep.audit_transcriptions import compare, audit
from src.ingest.qa_validation import validate_qa, qa_record_id
from src.lexical import tokenize, Bm25Index
from src.pipeline import answer_question, CANNOT_CONFIRM, AnswerResult
from src.generators import BaseGenerator, generate_with_meta
from src.main import create_app
from src.api_logging import JsonlLogger
from test_api_layer import _FakeRetriever, _FakeReranker, _configure

@pytest.mark.parametrize('text,token', [('المادة ٤','4'),('المعدل ٣٫٧٥','3.75'),('المعدل 3,750','3.75'),('المادة ١٠','10'),('شَرْط','شرط'),('لا يجوز','لا'),('لم يكمل','لم'),('لن يقبل','لن')])
def test_numeric_and_negative_tokens(text,token):
    assert token in tokenize(text)

def test_bm25_differentiates_number():
    idx=Bm25Index(['معدل 3.75','معدل 2.75'],[{},{}]);assert idx.search('معدل 3.75',1)[0][0]==0

class Gen(BaseGenerator):
    def __init__(self,text,truncated=False):
        super().__init__('fake',max_attempts=1,backoff_base=.001)
        self.text=text;self.truncated=truncated;self.calls=0
    def _invoke_once(self,*args):
        self.calls+=1;return self.text,self.truncated

@pytest.mark.parametrize('text,truncated',[(CANNOT_CONFIRM,False),('نص مقطوع',True)])
def test_refusal_and_truncation(text,truncated):
    result=answer_question('سؤال',_FakeRetriever(qa=[.99]),Gen(text,truncated))
    assert result.refused and result.answer==CANNOT_CONFIRM and result.sources==[]

def test_empty_provider_uses_alternative():
    first,last=Gen('  '),Gen('جواب')
    assert generate_with_meta([first,last],'sys','prompt').text=='جواب';assert first.calls==last.calls==1

def test_strict_citation_guard(monkeypatch):
    monkeypatch.setattr(get_settings(),'citation_guard_mode','refuse')
    result=answer_question('سؤال',_FakeRetriever(qa=[.99]),Gen('جواب [المصدر: مرجع مخترع]'))
    assert result.refused and result.citation_issues==['مرجع مخترع']

def test_lexical_does_not_embed(monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_mode','lexical');monkeypatch.setattr(s,'kau_gate_mode','rerank')
    class R(_FakeRetriever):
        def retrieve_split(self,*args,**kwargs):raise AssertionError('dense path used')
        def lexical_split(self,*args,**kwargs):return self.qa_hits,[]
    assert not answer_question('سؤال',R(qa=[.99]),Gen('جواب'),reranker=_FakeReranker([.99])).refused

def test_lexical_cannot_use_cosine_gate(monkeypatch):
    monkeypatch.setattr(get_settings(),'retrieval_mode','lexical');monkeypatch.setattr(get_settings(),'kau_gate_mode','cosine')
    with pytest.raises(ValueError):answer_question('سؤال',_FakeRetriever(),Gen('جواب'))

def test_request_id_disambiguates_same_answer(tmp_path):
    logger=JsonlLogger(tmp_path,'ask')
    for i in (1,2):logger.write({'university':'ksu','answer':'جواب','question':f'سؤال{i}','request_id':str(i)})
    assert logger.last_assembly('ksu','جواب','1')=='سؤال1';assert logger.last_assembly('ksu','جواب','bad') is None

def test_busy_slot_and_release():
    app=create_app(load_resources=False)
    with TestClient(app) as client:
        _configure(app.state,retriever=_FakeRetriever(qa=[.99]));slots=app.state.question_slots
        for _ in range(get_settings().max_concurrent_questions):slots.acquire()
        assert client.post('/ask',json={'question':'سؤال','university':'kau'}).status_code==429
        for _ in range(get_settings().max_concurrent_questions):slots.release()
        assert client.post('/ask',json={'question':'سؤال','university':'kau'}).status_code==200
        assert slots.acquire(blocking=False);slots.release()

def test_source_host_spoof():
    assert not official_urls('https://kau.edu.sa.attacker.test/a','kau');assert official_urls('https://dar.kau.edu.sa/a','kau')

def test_ksu_never_inherits_kau_evidence():
    raw={'المعرف الموحد':'KSU-0001','سؤال سعود الأصلي كاملًا':'سؤال','إجابة سعود الأصلية':'جواب','روابط مصادر عبدالعزيز':'https://kau.edu.sa/a','حالة توثيق ومراجعة عبدالعزيز':'موثق'}
    accepted,review=prepare_rows([raw],'ksu');assert not review and accepted[0]['source_url']=='';assert accepted[0]['evidence_status']=='original_unreviewed'

def test_conflicting_question_quarantines_both():
    accepted,review=prepare_rows([{'Q_ID':str(i),'Question':'نفس السؤال','Answer':str(i)} for i in (1,2)],'ksu')
    assert not accepted and len(review)==2

def test_mixed_universities():
    accepted,review=prepare_rows([{'Q_ID':'KAU-01','Question':'سؤال','Answer':'جواب'}],'ksu')
    assert not accepted and 'university_mismatch' in review[0]['review_reason']

def test_zero_import_preserves_existing(tmp_path):
    p=tmp_path/'x.tsv';p.write_text('Q_ID\tQuestion\tAnswer\n1\tسؤال\t\n');target=tmp_path/'ksu_qa.csv';target.write_text('old')
    with pytest.raises(ValueError):import_file(p,'ksu',tmp_path,write=True)
    assert target.read_text()=='old'

def test_quoted_multiline_tsv(tmp_path):
    p=tmp_path/'x.tsv';p.write_text('Q_ID\tQuestion\tAnswer\n1\t"سؤال\nكامل"\tجواب\n')
    rows,_=read_rows(p,'ksu');assert rows[0]['Question']=='سؤال\nكامل'

def test_variable_batch_and_record_ids():
    accepted,review=prepare_rows([{'Q_ID':f'KSU-{i:04}','Question':f'سؤال {i}','Answer':'جواب'} for i in range(1074)],'ksu')
    assert len(accepted)==1074 and not review;validate_qa(pd.DataFrame(accepted),'ksu')
    assert qa_record_id('ksu','12')=='ksu_qa_12';assert len({qa_record_id('ksu',r['Q_ID']) for r in accepted})==1074

@pytest.mark.parametrize('uni,nactive,nreview',[('ksu',1074,0),('kau',658,416)])
def test_delivered_dataset_integrity(uni,nactive,nreview):
    data=BACKEND_DIR/'data';read=lambda name:list(csv.DictReader((data/name).open(encoding='utf-8-sig')))
    allrows=read(f'{uni}_all_questions.csv');active=read(f'{uni}_qa.csv');review=read(f'{uni}_import_review.csv')
    assert len(allrows)==1074 and len(active)==nactive and len(review)==nreview;assert len({r['Q_ID'] for r in allrows})==1074
    assert {r['Q_ID'] for r in active}.isdisjoint(r['Q_ID'] for r in review);validate_qa(pd.DataFrame(active),uni)
    assert len(read(f'{uni}_source_catalog.csv'))==23
    if uni=='ksu':assert all(not r['source_url'] and 'لم تُراجع' in r['evidence_status'] for r in active)

def test_ingest_preserves_real_batch_metadata(monkeypatch,tmp_path):
    from src.ingest.ksu_qa import ingest_ksu_qa
    from src.ingest.kau_qa import ingest_kau_qa
    monkeypatch.setattr(get_settings(),'data_dir',tmp_path)
    class Embedder:
        def embed_passages(self,docs):return [[0.,1.] for _ in docs]
    class Collection:
        def __init__(self):self.records=[]
        def add(self,**kwargs):self.records.extend(zip(kwargs['ids'],kwargs['metadatas']))
    class Client:
        def __init__(self):self.collections={}
        def delete_collection(self,name):self.collections.pop(name,None)
        def create_collection(self,name,metadata):
            c=Collection();self.collections[name]=c;return c
    client=Client()
    for uni,fn,n in [('ksu',ingest_ksu_qa,1074),('kau',ingest_kau_qa,658)]:
        shutil.copyfile(BACKEND_DIR/'data'/f'{uni}_qa.csv',tmp_path/f'{uni}_qa.csv');assert fn(client,Embedder())==n
    assert all(md['q_id'].startswith('KAU-') and md['source_url'] and md['evidence_status'] for _,md in client.collections['kau_qa_verified'].records)

def test_gold_ids_override_misleading_keywords(monkeypatch):
    import src.eval_suite as suite
    from src.retrieval import Retrieved
    result=AnswerResult('q','a','',[],[Retrieved('kau_web',.9,'correct keyword',{'source_id':'wrong'})],False)
    monkeypatch.setattr(suite,'answer_question',lambda *args,**kwargs:result)
    rows=[{'id':'1','question':'q','university':'kau','expect':'answer','expect_any':'keyword','expected_source_ids':'gold'}]
    record=suite.run(rows,None,None,False)[0];assert record['hit'] is False and record['hit_method']=='gold_source_ids'

def test_evaluation_errors_count_in_denominator(monkeypatch):
    import src.eval_suite as suite
    def fail(*args,**kwargs):raise RuntimeError('offline')
    monkeypatch.setattr(suite,'answer_question',fail)
    stats=suite.summarize(suite.run([{'id':'1','question':'q','university':'kau','expect':'answer'}],None,None,False))['kau']
    assert stats['errors']==1 and stats['correct_over_all_attempts']=='0/1'

def test_ocr_reference_detects_numeric_loss():
    result=compare('معدل 2.75','معدل 3.75');assert result['wer']==.5 and result['missing_numbers']==['3.75'] and result['numeric_recall']==0

def test_ocr_source_directory_protected():
    with pytest.raises(ValueError):audit('kau',get_settings().sources_dir/'audit.json')


def test_old_match_cannot_override_new_unverified_status():
    raw={'Q_ID':'KAU-1','Question':'سؤال','Answer':'جواب','evidence_status':'يحتاج تحقق رسمي إضافي',
         'match_status':'مطابق بالمعنى','source_url':'https://kau.edu.sa/a'}
    active,review=prepare_rows([raw],'kau',allow_legacy_reviewed=True)
    assert not active and len(review)==1


def test_rejected_output_cannot_trigger_external_fallback(monkeypatch):
    import src.general_fallback as fb
    monkeypatch.setattr(get_settings(),'general_fallback_mode','web')
    def unexpected(*args,**kwargs):raise AssertionError('guard refusal bypassed')
    monkeypatch.setattr(fb,'try_fallback',unexpected)
    result=answer_question('سؤال',_FakeRetriever(qa=[.99]),Gen('نص مقطوع',True))
    assert result.refused and result.output_guard_refused
