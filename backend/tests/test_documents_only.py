"""Document-only answering must exclude QA from retrieval, gates and generation."""
import pytest
from fastapi.testclient import TestClient
from src.config import Settings, get_settings, UNIVERSITY_COLLECTIONS
from src.context_assembly import AssembledContext, _ksu_doc_source, _kau_doc_source
from src.flat_store import FlatClient
from src.retrieval import Retriever, Retrieved
from src.pipeline import answer_question, _generate_answer, CANNOT_CONFIRM
from src.generators import BaseGenerator
from src.grounding import render_document_citations
from src.main import create_app
from src.answer_cache import cache_key
from src.prompts import build_system_prompt, build_user_prompt


@pytest.fixture(autouse=True)
def documents_mode(monkeypatch):
    monkeypatch.setattr(get_settings(), 'evidence_mode', 'documents_only')


def doc(uni, score=.99):
    if uni=='ksu':
        return Retrieved('ksu_pdf',score,'يشترط الحصول على موافقة الكلية.',
                         {'type':'pdf','doc':'اللائحة التنظيمية للشؤون الدراسية','unit':'المادة 14','page':20,'page_offset':6,'source_id':'ksu_pdf_14'})
    return Retrieved('kau_web',score,'يشترط الحصول على موافقة الكلية.',
                     {'type':'web','page_title':'دليل التحويل','section':'الشروط','source_url':'https://kau.edu.sa/a','source_id':'kau_web_1'})


def qa(uni):
    return Retrieved(UNIVERSITY_COLLECTIONS[uni][0],1.,'جاهز_SECRET_QUESTION',
                     {'type':'qa','q_id':uni.upper()+'-0522','answer':'جاهز_SECRET_ANSWER'})


class Ret:
    def __init__(self,uni,docs=True):
        self.uni=uni;self.docs=[doc(uni)] if docs else []
    def retrieve_split(self,*args,**kwargs):return [qa(self.uni)],self.docs
    def lexical_split(self,*args,**kwargs):return [qa(self.uni)],self.docs


class Gen(BaseGenerator):
    def __init__(self,text='يشترط الحصول على موافقة الكلية. [المصدر: مرجع 1]'):
        super().__init__('test_docs',max_attempts=1);self.model='stub';self.text=text;self.calls=0;self.prompts=[]
    def _invoke_once(self,system,prompt,timeout_seconds):
        self.calls+=1;self.prompts.append((system,prompt));return self.text,False


class Reranker:
    def __init__(self,score=.99):self.score=score;self.pairs=[]
    def predict(self,pairs,**kwargs):self.pairs.extend(pairs);return [self.score]*len(pairs)


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('uni',['ksu','kau'])
def test_model_refusal_retains_opt_in_selected_context(enabled,uni,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_diagnostics',enabled)
    monkeypatch.setattr(s,'kau_gate_mode','rerank')
    result=answer_question('ما الشرط؟',Ret(uni),Gen(CANNOT_CONFIRM),university=uni,reranker=Reranker())
    assert result.refused and result.refusal_reason=='model_refusal'
    assert result.provider=='test_docs' and not result.citation_issues
    if enabled:
        assert result.diagnostics['passed']
        assert result.diagnostics['raw_generation_text']==CANNOT_CONFIRM
        assert result.diagnostics['selected_context'][0]['document']==doc(uni).document
        assert all('SECRET' not in row['document'] for row in result.diagnostics['ranked_candidates'])
    else: assert result.diagnostics=={}


@pytest.mark.parametrize('uni',['ksu','kau'])
def test_gate_refusal_logs_candidate_scores_without_calling_provider(uni,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_diagnostics',True)
    monkeypatch.setattr(s,'kau_gate_mode','rerank')
    generator=Gen();result=answer_question('هل يوجد دليل؟',Ret(uni),generator,university=uni,reranker=Reranker(.01))
    assert result.refused and result.refusal_reason=='retrieval_gate' and generator.calls==0
    trace=result.diagnostics
    assert not trace['passed'] and trace['threshold']==.40
    assert trace['ranked_candidates'][0]['rerank_score']==.01
    assert trace['ranked_candidates'][0]['metadata']['source_id']==doc(uni).metadata['source_id']
    assert 'raw_generation_text' not in trace


def test_invalid_citation_reason_is_distinct_from_model_refusal(monkeypatch):
    monkeypatch.setattr(get_settings(),'retrieval_diagnostics',True)
    result=answer_question('شرط؟',Ret('ksu'),Gen('يشترط موافقة الكلية. [المصدر: مرجع 99]'),university='ksu',reranker=Reranker())
    assert result.refused and result.refusal_reason=='citation_guard'
    assert 'مرجع 99' in result.diagnostics['raw_generation_text']
    assert result.citation_issues and result.diagnostics['selected_context']


def test_numeric_refusal_reason_is_separate(monkeypatch):
    monkeypatch.setattr(get_settings(),'numeric_guard_mode','refuse')
    result=answer_question('ما النسبة؟',Ret('ksu'),Gen('النسبة 99.99%. [المصدر: مرجع 1]'),university='ksu',reranker=Reranker())
    assert result.refused and result.refusal_reason=='numeric_guard'


def test_evidence_trace_is_bounded_and_whitelists_metadata():
    from src.retrieval_diagnostics import evidence_trace
    item=doc('ksu');item.document='نص '*10000
    item.metadata['API_KEY']='DO_NOT_LOG';item.metadata['answer']='OLD_READY_ANSWER'
    trace=evidence_trace([item],[.8],max_chars=123)[0]
    assert len(trace['document'])==123 and trace['document_truncated_in_log']
    assert 'API_KEY' not in trace['metadata'] and 'answer' not in trace['metadata']


@pytest.mark.parametrize('uni',['ksu','kau'])
def test_hybrid_adds_missing_document_but_still_requires_same_gate(uni,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_mode','hybrid')
    monkeypatch.setattr(s,'hybrid_lexical_extra',4)
    monkeypatch.setattr(s,'kau_gate_mode','rerank')
    monkeypatch.setattr(s,'retrieval_diagnostics',True)
    class HybridRet(Ret):
        def retrieve_split(self,*args,**kwargs):return [qa(uni)],[]
        def lexical_split(self,*args,**kwargs):return [qa(uni)],[doc(uni)]
    passing=answer_question('ما الشرط؟',HybridRet(uni),Gen(),university=uni,reranker=Reranker(.8))
    refusing_gen=Gen()
    refusing=answer_question('ما الشرط؟',HybridRet(uni),refusing_gen,university=uni,reranker=Reranker(.1))
    assert not passing.refused and passing.refusal_reason==''
    assert refusing.refused and refusing.refusal_reason=='retrieval_gate' and refusing_gen.calls==0
    assert all('SECRET' not in i['document'] for i in passing.diagnostics['ranked_candidates'])


def test_default_mode_is_documents(monkeypatch):
    monkeypatch.delenv('EVIDENCE_MODE',raising=False)
    assert Settings(_env_file=None).evidence_mode=='documents_only'


@pytest.mark.parametrize('uni',['ksu','kau'])
@pytest.mark.parametrize('mode',['dense','lexical','hybrid'])
def test_qa_never_reaches_reranker_prompt_or_sources(uni,mode,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_mode',mode);monkeypatch.setattr(s,'kau_gate_mode','rerank')
    gen=Gen();rank=Reranker();result=answer_question('ما شروط التحويل؟',Ret(uni),gen,university=uni,reranker=rank)
    assert not result.refused and not result.qa_items
    assert all('SECRET' not in p[1] for p in rank.pairs)
    assert 'SECRET' not in gen.prompts[0][1]
    assert '[المصدر: مرجع 1]' not in result.answer
    assert 'سؤال رقم' not in result.answer and '0522' not in result.answer
    assert result.sources and all('سؤال' not in source for source in result.sources)
    assert ('صفحة PDF 20' if uni=='ksu' else 'https://kau.edu.sa/a') in result.answer


@pytest.mark.parametrize('uni',['ksu','kau'])
@pytest.mark.parametrize('mode',['dense','lexical','hybrid'])
def test_qa_alone_cannot_authorize_an_answer(uni,mode,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'retrieval_mode',mode);monkeypatch.setattr(s,'kau_gate_mode','rerank')
    gen=Gen();rank=Reranker();result=answer_question('سؤال جاهز',Ret(uni,False),gen,university=uni,reranker=rank)
    assert result.refused and result.answer==CANNOT_CONFIRM and not result.sources
    assert gen.calls==0 and not rank.pairs


@pytest.mark.parametrize('uni',['ksu','kau'])
def test_strong_qa_cannot_rescue_weak_documents(uni):
    rank=Reranker(score=.01);gen=Gen();result=answer_question('سؤال',Ret(uni),gen,university=uni,reranker=rank)
    assert result.refused and gen.calls==0
    assert rank.pairs and all('SECRET' not in pair[1] for pair in rank.pairs)


def test_cosine_gate_does_not_use_qa(monkeypatch):
    monkeypatch.setattr(get_settings(),'kau_gate_mode','cosine')
    gen=Gen();result=answer_question('سؤال',Ret('kau',False),gen,university='kau')
    assert result.refused and gen.calls==0


def test_generation_safety_net_strips_preassembled_qa():
    gen=Gen();result=_generate_answer('سؤال',AssembledContext(qa_items=[qa('ksu')]),gen,'ksu')
    assert result.refused and gen.calls==0


@pytest.mark.parametrize('foreign',[Retrieved('kau_pdf',1.,'FOREIGN',{}),Retrieved('ksu_pdf',1.,'QA_IN_DOC_SLOT',{'type':'qa'}),Retrieved('ksu_pdf',1.,'MIXED',{'university':'kau'})])
def test_foreign_or_mislabelled_qa_cannot_supply_evidence(foreign):
    ret=Ret('ksu',False);ret.docs=[foreign];gen=Gen()
    result=answer_question('سؤال',ret,gen,university='ksu',reranker=Reranker())
    assert result.refused and gen.calls==0


@pytest.mark.parametrize('text',['جواب بلا مصدر','جواب [المصدر: سؤال رقم KSU-0522]','جواب [المصدر: مرجع 99]','جواب [المصدر: مرجع 1] [المصدر: مجهول]'])
def test_invalid_or_missing_citations_refuse_even_in_log_mode(text,monkeypatch):
    monkeypatch.setattr(get_settings(),'citation_guard_mode','log')
    result=answer_question('سؤال',Ret('ksu'),Gen(text),university='ksu',reranker=Reranker())
    assert result.refused and result.answer==CANNOT_CONFIRM and result.citation_issues and not result.sources


def test_exact_document_label_citation_also_accepted():
    label=_ksu_doc_source(doc('ksu').metadata)
    result=answer_question('سؤال',Ret('ksu'),Gen(f'جواب [المصدر: {label}]'),university='ksu',reranker=Reranker())
    assert not result.refused and label in result.answer


def test_fallback_cannot_bypass_document_refusal(monkeypatch):
    import src.general_fallback as fallback
    monkeypatch.setattr(get_settings(),'general_fallback_mode','llm')
    def fail(*args,**kwargs):raise AssertionError('external fallback used')
    monkeypatch.setattr(fallback,'try_fallback',fail)
    gen=Gen();result=answer_question('سؤال',Ret('ksu',False),gen,university='ksu',reranker=Reranker())
    assert result.refused and gen.calls==0


def test_prompts_ignore_qa_and_require_original_reference():
    prompt=build_user_prompt('سؤال','SECRET_QA','DOCUMENT','ksu')
    assert 'SECRET_QA' not in prompt and 'DOCUMENT' in prompt
    system=build_system_prompt('ksu')
    assert 'لا تخترع' in system and '[المصدر: مرجع 1]' in system


def test_text_source_label_preserves_file_and_section_without_inventing_url():
    label=_ksu_doc_source({'type':'txt','source_file':'university documents.txt','main_section':'القبول','service':'التسجيل'})
    assert 'university documents.txt' in label and 'القبول' in label and 'التسجيل' in label
    assert 'https://' not in label


def test_alias_expansion_does_not_create_a_source():
    rendered,issues=render_document_citations('حكم [المصدر: مرجع 1]',['وثيقة أ، صفحة 5'])
    assert not issues and rendered=='حكم [المصدر: وثيقة أ، صفحة 5]'
    _,issues=render_document_citations('حكم [المصدر: مرجع 2]',['وثيقة أ، صفحة 5'])
    assert issues


@pytest.mark.parametrize('uni',['ksu','kau'])
def test_real_retriever_never_opens_qa_collection(tmp_path,uni):
    client=FlatClient(tmp_path);calls=[]
    for name in UNIVERSITY_COLLECTIONS[uni]:
        c=client.create_collection(name)
        is_qa=name==UNIVERSITY_COLLECTIONS[uni][0]
        c.add(ids=[name],documents=['SECRET' if is_qa else 'يشترط الحصول على موافقة الكلية'],
              metadatas=[qa(uni).metadata if is_qa else doc(uni).metadata],embeddings=[[1.,0.]])
    class TrackedClient:
        def get_collection(self,name):calls.append(name);return client.get_collection(name)
    class E:
        def embed_queries(self,texts):return [[1.,0.]]
    ret=Retriever(TrackedClient(),E())
    dense_qa,dense_docs=ret.retrieve_split('موافقة الكلية',university=uni)
    lexical_qa,lexical_docs=ret.lexical_split('موافقة الكلية',3,university=uni)
    ret.warm_lexical(uni)
    assert not dense_qa and not lexical_qa and dense_docs and lexical_docs
    assert UNIVERSITY_COLLECTIONS[uni][0] not in calls


def test_api_returns_document_sources_and_ignores_legacy_cache():
    app=create_app(load_resources=False)
    with TestClient(app) as client:
        gen=Gen();app.state.retriever=Ret('ksu');app.state.reranker=Reranker();app.state.generation_chain=[gen]
        key=cache_key('ksu:qa_and_documents','سؤال')
        app.state.answer_cache.put(key,{'answer':'OLD_QUESTION_ANSWER','refused':False,'sources':['سؤال رقم KSU-0522']})
        response=client.post('/ask',json={'question':'سؤال','university':'ksu'})
        assert response.status_code==200
        body=response.json();assert not body['refused'] and 'OLD_QUESTION' not in body['answer']
        assert body['sources'] and all('سؤال' not in s for s in body['sources'])
        assert all(not source['q_id'] for source in body['source_details'])
        assert gen.calls==1


def test_api_keeps_diagnostic_evidence_in_local_log_only(monkeypatch):
    import json
    s=get_settings();monkeypatch.setattr(s,'retrieval_diagnostics',True)
    app=create_app(load_resources=False)
    with TestClient(app) as client:
        app.state.retriever=Ret('ksu');app.state.reranker=Reranker(.01)
        app.state.generation_chain=[Gen()]
        response=client.post('/ask',json={'question':'سؤال تشخيصي','university':'ksu'})
        assert response.status_code==200 and response.json()['refused']
        assert 'diagnostics' not in response.json()
        assert 'SECRET' not in response.text
    record=json.loads((s.data_dir/'logs'/'ask.jsonl').read_text().splitlines()[-1])
    assert record['request_id']==response.json()['request_id']
    assert record['refusal_reason']=='retrieval_gate'
    assert record['diagnostics']['ranked_candidates'][0]['document']==doc('ksu').document


@pytest.mark.parametrize('tag',['[المصدر: مرجع ١]','[ المصدر : مرجع1]','[المصدر: مرجع 01]','[المصدر: مرجع ١ ]'])
def test_alias_spacing_and_arabic_digits_are_not_false_refusals(tag):
    rendered,issues=render_document_citations('حكم '+tag,['وثيقة أ، صفحة 5'])
    assert not issues and rendered=='حكم [المصدر: وثيقة أ، صفحة 5]'
