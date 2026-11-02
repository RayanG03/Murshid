from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from src.retrieval import Retriever

class InternalError(Exception):pass

class Broken:
    name='ksu_pdf'
    def __init__(self,vectors=None):
        self.vectors=[[2.,0.],[0.,3.],[-1.,0.]] if vectors is None else vectors
        self.queries=self.reads=0
    def count(self):return len(self.vectors)
    def query(self,**kwargs):
        self.queries+=1
        raise InternalError('Error creating hnsw segment reader: Nothing found on disk')
    def get(self,limit,offset,include):
        self.reads+=1;end=min(offset+limit,len(self.vectors))
        return {'ids':[str(i) for i in range(offset,end)],'documents':[f'doc{i}' for i in range(offset,end)],
                'metadatas':[{'page':i+1,'source_url':'https://ksu.edu.sa/a'} for i in range(offset,end)],
                'embeddings':np.asarray(self.vectors[offset:end],dtype=np.float32)}

def test_scores_metadata_and_cache():
    c=Broken();r=Retriever(None,None);hits=r._query_collection(c,[5.,0.],10)
    assert [h.score for h in hits]==pytest.approx([1,0,-1])
    assert [h.metadata['source_id'] for h in hits]==['0','1','2']
    assert hits[0].document=='doc0' and hits[0].metadata['page']==1
    hits[0].metadata['page']=999
    assert r._query_collection(c,[1.,0.],1)[0].metadata['page']==1
    assert r._query_collection(c,[0.,9.],1)[0].metadata['source_id']=='1'
    assert c.queries==c.reads==1

def test_independent_cosine_and_pagination():
    rng=np.random.default_rng(17);vectors=rng.normal(size=(1100,12));q=rng.normal(size=12)
    c=Broken(vectors.tolist());hits=Retriever(None,None)._query_collection(c,q,20)
    scores=[sum(x*y for x,y in zip(v,q))/(sum(x*x for x in v)**.5*sum(y*y for y in q)**.5) for v in vectors]
    gold=sorted(range(len(scores)),key=lambda i:-scores[i])[:20]
    assert [h.metadata['source_id'] for h in hits]==list(map(str,gold))
    assert [h.score for h in hits]==pytest.approx([scores[i] for i in gold],abs=1e-6)
    assert c.reads==3

def test_cache_count_change():
    c=Broken();r=Retriever(None,None);r._query_collection(c,[1.,0.],1);c.vectors.append([1.,1.])
    assert len(r._query_collection(c,[1.,0.],10))==4 and c.reads==2

def test_concurrent_cache_read():
    c=Broken();r=Retriever(None,None);r._query_collection(c,[1.,0.],1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hits=list(pool.map(lambda _:r._query_collection(c,[0.,1.],1),range(12)))
    assert all(h[0].metadata['source_id']=='1' for h in hits) and c.reads==1

@pytest.mark.parametrize('error',[ValueError('dimension mismatch'),InternalError('permission denied'),RuntimeError('disk full')])
def test_unrelated_errors_propagate(error):
    c=Broken()
    def fail(**kwargs):raise error
    c.query=fail
    with pytest.raises(type(error),match=str(error)):Retriever(None,None)._query_collection(c,[1.,0.],1)
    assert c.reads==0

@pytest.mark.parametrize('query',[[0.,0.],[1.,0.,0.],[float('nan'),0.]])
def test_invalid_query(query):
    with pytest.raises(ValueError):Retriever(None,None)._query_collection(Broken(),query,1)

def test_corrupt_vector_is_not_dropped():
    with pytest.raises(ValueError,match='stored vectors'):
        Retriever(None,None)._query_collection(Broken([[0.,0.]]),[1.,0.],1)

def test_missing_embeddings():
    c=Broken();c.get=lambda **kwargs:{'ids':['1'],'documents':['d'],'metadatas':[{}],'embeddings':None}
    with pytest.raises(ValueError,match='Incomplete'):Retriever(None,None)._query_collection(c,[1.,0.],1)

def test_incomplete_batch():
    c=Broken();get=c.get;c.get=lambda **kwargs:get(limit=1,offset=kwargs['offset'],include=kwargs['include'])
    with pytest.raises(ValueError,match='Collection changed'):Retriever(None,None)._query_collection(c,[1.,0.],1)

def test_normal_hnsw_path():
    c=Broken();c.query=lambda **kwargs:{'ids':[['ok']],'documents':[['d']],'metadatas':[[{'page':3}]],'distances':[[.2]]}
    hit=Retriever(None,None)._query_collection(c,[1.,0.],1)[0]
    assert hit.score==pytest.approx(.8) and hit.metadata['source_id']=='ok' and c.reads==0
