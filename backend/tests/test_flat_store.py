"""Persistence, semantic ranking and ingestion without a Chroma reader."""
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from src.flat_store import FlatClient
from src.chroma_store import get_client, collection_counts
from src.config import BACKEND_DIR, get_settings, UNIVERSITY_COLLECTIONS
from src.retrieval import Retriever
from src.pipeline import answer_question
from test_api_layer import _FakeChainGenerator


def seed(tmp_path, n=3, dim=2):
    client=FlatClient(tmp_path);col=client.create_collection('test_vectors')
    vectors=np.random.default_rng(12).normal(size=(n,dim)).tolist()
    col.add(ids=[str(i) for i in range(n)],documents=[f'doc{i}' for i in range(n)],
            metadatas=[{'q_id':f'KSU-{i}','answer':'جواب','source_url':'https://ksu.edu.sa/a'} for i in range(n)],embeddings=vectors)
    return client,col,vectors


def test_roundtrip_and_metadata(tmp_path):
    _,col,vectors=seed(tmp_path)
    loaded=FlatClient(tmp_path).get_collection('test_vectors')
    data=loaded.get(include=['documents','metadatas','embeddings'])
    assert loaded.count()==3 and data['ids']==['0','1','2']
    assert data['documents']==['doc0','doc1','doc2']
    assert data['metadatas'][0]['source_url']=='https://ksu.edu.sa/a'
    assert np.allclose(data['embeddings'],vectors)
    assert sorted(p.suffix for p in tmp_path.iterdir())==['.npz']


def test_rankings_against_independent_cosine_after_restart(tmp_path):
    _,_,vectors=seed(tmp_path,n=1100,dim=12)
    queries=np.random.default_rng(19).normal(size=(4,12))
    col=FlatClient(tmp_path).get_collection('test_vectors')
    result=col.query(query_embeddings=queries,n_results=20)
    for j,q in enumerate(queries):
        scores=[float(sum(x*y for x,y in zip(v,q))/(sum(x*x for x in v)**.5*sum(y*y for y in q)**.5)) for v in vectors]
        order=sorted(range(len(vectors)),key=lambda i:-scores[i])[:20]
        assert result['ids'][j]==list(map(str,order))
        assert result['distances'][j]==pytest.approx([1-scores[i] for i in order],abs=1e-6)


def test_multiple_batches_and_cache_invalidation(tmp_path):
    _,col,_=seed(tmp_path)
    col.query(query_embeddings=[[1.,0.]],n_results=1)
    col.add(ids=['new'],documents=['newdoc'],metadatas=[{}],embeddings=[[1.,0.]])
    assert col.query(query_embeddings=[[1.,0.]],n_results=1)['ids'][0]==['new']
    assert FlatClient(tmp_path).get_collection(col.name).count()==4


def test_slicing_and_returned_values_not_mutable(tmp_path):
    _,col,_=seed(tmp_path)
    data=col.get(offset=1,limit=1,include=['metadatas','embeddings'])
    data['metadatas'][0]['answer']='changed';data['embeddings'][0]=[999,999]
    assert col.get(offset=1,limit=1)['metadatas'][0]['answer']=='جواب'
    assert col.get(offset=99,limit=1)['ids']==[]


@pytest.mark.parametrize('vectors',[[[0.,0.]],[[float('nan'),1.]],[[float('inf'),1.]]])
def test_invalid_vectors_are_rejected(tmp_path,vectors):
    col=FlatClient(tmp_path).create_collection('test_invalid')
    with pytest.raises(ValueError):col.add(ids=['a'],documents=['d'],metadatas=[{}],embeddings=vectors)
    assert col.count()==0


@pytest.mark.parametrize('query',[[[0.,0.]],[[1.,0.,1.]],[[float('nan'),0.]]])
def test_invalid_queries_rejected(tmp_path,query):
    _,col,_=seed(tmp_path)
    with pytest.raises(ValueError):col.query(query_embeddings=query)


def test_duplicate_ids_and_dimensions_do_not_change_files(tmp_path):
    _,col,_=seed(tmp_path);old=col.path.read_bytes()
    for args in [dict(ids=['0'],embeddings=[[1.,0.]]),dict(ids=['new'],embeddings=[[1.,0.,1.]])]:
        with pytest.raises(ValueError):col.add(documents=['d'],metadatas=[{}],**args)
    assert col.path.read_bytes()==old


def test_atomic_write_failure_rolls_back_memory_and_disk(tmp_path,monkeypatch):
    import src.flat_store as store
    _,col,_=seed(tmp_path);old=col.path.read_bytes()
    def fail(*args):raise OSError('simulated replace failure')
    monkeypatch.setattr(store.os,'replace',fail)
    with pytest.raises(OSError):col.add(ids=['new'],documents=['d'],metadatas=[{}],embeddings=[[1.,0.]])
    assert col.count()==3 and col.path.read_bytes()==old
    assert not list(tmp_path.glob('*.tmp'))


def test_concurrent_readers(tmp_path):
    _,col,_=seed(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:col.query(query_embeddings=[[1.,0.]],n_results=2),range(16)))
    assert all(r['ids']==results[0]['ids'] for r in results)


def test_corrupt_format_detected(tmp_path):
    np.savez_compressed(tmp_path/'test_bad.npz',records=np.asarray(json.dumps({'version':99})),embeddings=np.empty((0,0)))
    with pytest.raises(ValueError,match='format'):FlatClient(tmp_path).get_collection('test_bad')


def test_delete_create_missing_and_unsafe_names(tmp_path):
    client,col,_=seed(tmp_path)
    with pytest.raises(ValueError):client.create_collection(col.name)
    client.delete_collection(col.name)
    with pytest.raises(KeyError):client.get_collection(col.name)
    with pytest.raises(ValueError):client.create_collection('../escape')
    assert not tmp_path.parent.joinpath('escape.npz').exists()


def test_backend_routing_never_imports_chroma(tmp_path,monkeypatch):
    s=get_settings();monkeypatch.setattr(s,'vector_backend','flat');monkeypatch.setattr(s,'flat_index_dir',tmp_path)
    monkeypatch.setitem(sys.modules,'chromadb',None)
    assert isinstance(get_client(tmp_path/'unused_chroma'),FlatClient)
    assert not (tmp_path/'unused_chroma').exists()


def test_six_collections_after_process_style_reopen(tmp_path):
    client=FlatClient(tmp_path)
    for uni,names in UNIVERSITY_COLLECTIONS.items():
        for name in names:
            c=client.create_collection(name)
            c.add(ids=[name],documents=['دليل'],metadatas=[{'university':uni}],embeddings=[[1.,2.]])
    for _ in range(3):
        reopened=FlatClient(tmp_path)
        for names in UNIVERSITY_COLLECTIONS.values():
            assert set(collection_counts(reopened,names).values())=={1}
            for name in names:assert reopened.get_collection(name).query(query_embeddings=[[1.,2.]])['ids'][0]==[name]


@pytest.mark.parametrize('uni,n',[('ksu',1074),('kau',658)])
def test_actual_dataset_ingestion_search_and_generation_after_reopen(tmp_path,monkeypatch,uni,n):
    from src.ingest.ksu_qa import ingest_ksu_qa
    from src.ingest.kau_qa import ingest_kau_qa
    s=get_settings();data=tmp_path/'data';data.mkdir();monkeypatch.setattr(s,'data_dir',data)
    shutil.copyfile(BACKEND_DIR/'data'/f'{uni}_qa.csv',data/f'{uni}_qa.csv')
    class Embedder:
        def embed_passages(self,docs):return [[1.,2.] for _ in docs]
        def embed_queries(self,docs):return [[1.,2.] for _ in docs]
    class Reranker:
        def predict(self,pairs,**kwargs):return [.99]*len(pairs)
    client=FlatClient(tmp_path/'index');embedder=Embedder()
    assert (ingest_ksu_qa if uni=='ksu' else ingest_kau_qa)(client,embedder)==n
    reopened=FlatClient(tmp_path/'index');ret=Retriever(reopened,embedder)
    qa,_=ret.retrieve_split('ما شروط التسجيل؟',university=uni)
    assert qa and qa[0].metadata['source_id'] and qa[0].metadata['q_id'].startswith(uni.upper()+'-')
    result=answer_question('ما شروط التسجيل؟',ret,_FakeChainGenerator(),university=uni,reranker=Reranker())
    assert not result.refused and result.answer and result.sources


def test_six_collections_search_from_fresh_python_process(tmp_path):
    import subprocess
    client=FlatClient(tmp_path)
    for names in UNIVERSITY_COLLECTIONS.values():
        for name in names:
            col=client.create_collection(name)
            col.add(ids=['first','second'],documents=['أ','ب'],metadatas=[{},{}],embeddings=[[1.,0.],[0.,1.]])
    code='''import json,sys
from pathlib import Path
from src.flat_store import FlatClient
from src.config import UNIVERSITY_COLLECTIONS
client=FlatClient(Path(sys.argv[1]))
out=[]
for _ in range(3):
 for names in UNIVERSITY_COLLECTIONS.values():
  for name in names:
   col=client.get_collection(name)
   result=col.query(query_embeddings=[[0.,2.]],n_results=1)
   assert result['ids'][0]==['second']
   out.append(col.count())
print(json.dumps(out))
'''
    run=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=BACKEND_DIR,text=True,capture_output=True,timeout=30,check=True)
    assert json.loads(run.stdout)==[2]*18
