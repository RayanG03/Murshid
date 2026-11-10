"""البحث: يدمج البحث الدلالي مع البحث اللفظي داخل جامعة واحدة، وكلاهما من bge-m3.

المدخلات: data/index/ksu/ و data/index/kau/ (مقاطع كل جامعة ومتجهاتها وأوزان كلماتها)
ودولابا Qdrant (ksu_rules و kau_rules)؛ إن كان Qdrant مطفأً فالمتجهات من embeddings.npy
في مجلد الجامعة.
المخرجات: المقاطع الأقرب للسؤال، مع درجة الدمج في الحقل score.

- الدلالي (dense): تشابه متجه السؤال ومتجه المقطع.
- اللفظي (sparse): bge-m3 يعطي كل كلمة في السؤال والمقطع وزناً، والدرجة مجموع
  (وزنها في السؤال × وزنها في المقطع) للكلمات المشتركة. حلّ محل BM25 وتجذيعنا اليدوي.
كل نوع درجة يُطبَّع من 0 إلى 1 داخل المقاطع المسموح بها ثم تُجمع: 0.6 دلالي + 0.4 لفظي.

التشغيل:
    python retrieve.py "كم عدد الإنذارات التي تؤدي للفصل؟" --university KAU
    from retrieve import Retriever;  Retriever().search("...", k=5, university="KAU")
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

import config
from embedding import (
    index_paths,
    index_universities,
    lexical_score,
    load_embedding_model,
    load_sparse_head,
    sparse_weights,
)
from index_qdrant import point_id

logger = logging.getLogger(__name__)


def scale_to_unit(scores: np.ndarray) -> np.ndarray:
    """تطبيع من 0 إلى 1 كي يُجمع النوعان بأوزان ثابتة."""
    lowest, highest = float(scores.min()), float(scores.max())
    if highest > lowest:
        return (scores - lowest) / (highest - lowest)
    return np.zeros_like(scores)


def read_index_chunks(path: Path) -> list[dict]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(json.loads(line))
    return chunks


def read_sparse_weights(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise RuntimeError("أوزان الكلمات غير موجودة — شغّل: python embedding.py")
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(index_dir: Path) -> dict:
    """الفهرس يجب أن يكون مبنياً بنموذج config نفسه، وإلا فمتجه السؤال لا يقارن به."""
    path = index_dir / "manifest.json"
    if not path.exists():
        raise RuntimeError("الفهرس غير موجود — شغّل: python embedding.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("model") != config.EMBEDDING_MODEL
        or manifest.get("revision") != config.EMBEDDING_REVISION
    ):
        raise RuntimeError(
            f"الفهرس مبني بنموذج {manifest.get('model')} "
            f"({manifest.get('revision')}) لا {config.EMBEDDING_MODEL} — "
            "شغّل: python embedding.py --full"
        )
    if "universities" not in manifest:
        raise RuntimeError("الفهرس بالتنظيم القديم (ملف واحد) — شغّل: python embedding.py")
    return manifest


def read_index(index_dir: Path) -> tuple[list[dict], np.ndarray, list[dict]]:
    """فهرس كل جامعة من مجلدها، مضموماً في الذاكرة بترتيب index_universities: المقاطع
    ومتجهاتها وأوزان كلماتها."""
    manifest = read_manifest(index_dir)
    chunks = []
    vectors = []
    weights = []
    for university in index_universities():
        paths = index_paths(index_dir, university)
        university_chunks = read_index_chunks(paths["chunks"])
        university_vectors = np.load(paths["vectors"])
        university_weights = read_sparse_weights(paths["weights"])
        expected = manifest["universities"][university]["count"]
        if not (
            len(university_chunks)
            == university_vectors.shape[0]
            == len(university_weights)
            == expected
        ):
            raise RuntimeError(
                f"عدد مقاطع {university} لا يطابق عدد المتجهات أو أوزان الكلمات — "
                "شغّل: python embedding.py"
            )
        chunks += university_chunks
        vectors.append(university_vectors)
        weights += university_weights
    return chunks, np.concatenate(vectors), weights


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR, use_qdrant: bool = True) -> None:
        self.chunks, self.vectors, self.sparse_weights = read_index(index_dir)
        self.universities = np.array([chunk.get("university", "") for chunk in self.chunks])
        self.stubs = np.array([bool(chunk.get("is_stub")) for chunk in self.chunks])
        self.point_ids = [point_id(chunk["chunk_id"]) for chunk in self.chunks]
        self.university_rows = {}
        for university in config.QDRANT_COLLECTIONS:
            self.university_rows[university] = np.flatnonzero(self.universities == university)
        self.model = None
        self.sparse_head = None
        self.query_weights = {}
        self.qdrant = self.connect_qdrant() if use_qdrant else None

    def connect_qdrant(self) -> "QdrantClient | None":
        """يتصل بـ Qdrant ويتأكد أن كل دولاب فيه مقاطع جامعته المحلية نفسها.

        Qdrant مطفأ ← البحث من ملف المتجهات مع تنبيه. Qdrant شغال لكن دولابه ناقص أو من
        تقطيع أقدم (نُسي index_qdrant.py بعد إعادة التقطيع) ← خطأ صريح، لأن النتائج
        ستشير إلى نصوص لا تطابق ما يُعرض.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import ApiException

        client = QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
        try:
            collections = client.get_collections().collections
        except (ApiException, OSError) as error:
            logger.warning(
                "Qdrant غير متاح (%s) — البحث من ملفات المتجهات في data/index", type(error).__name__
            )
            return None
        existing = {collection.name for collection in collections}
        for university, rows in self.university_rows.items():
            self.check_collection(client, existing, university, rows)
        return client

    def check_collection(
        self, client: "QdrantClient", existing: set, university: str, rows: np.ndarray
    ) -> None:
        name = config.QDRANT_COLLECTIONS[university]
        stored = {}
        if name in existing:
            points, _ = client.scroll(
                name, limit=len(rows) + 1, with_payload=["text"], with_vectors=False
            )
            for point in points:
                stored[str(point.id)] = point.payload["text"]
        expected = {}
        for row in rows:
            expected[self.point_ids[row]] = self.chunks[row]["text"]
        if stored != expected:
            raise RuntimeError(
                f"دولاب {name} في Qdrant لا يطابق data/index — شغّل: python index_qdrant.py"
            )

    def dense_scores(
        self,
        vector: np.ndarray,
        university: str,
        rows: np.ndarray,
        exclude_stubs: bool,
    ) -> np.ndarray:
        """تشابه السؤال مع مقاطع الصفوف المطلوبة، مرتبةً مثلها."""
        if self.qdrant is None:
            return self.vectors[rows] @ vector
        from qdrant_client.http.exceptions import ApiException

        try:
            return self.qdrant_dense_scores(vector, university, rows, exclude_stubs)
        except (ApiException, OSError) as error:
            logger.warning(
                "تعذّر البحث في Qdrant (%s) — التحويل إلى ملفات المتجهات في data/index",
                type(error).__name__,
            )
            self.qdrant = None
            return self.vectors[rows] @ vector

    def qdrant_dense_scores(
        self,
        vector: np.ndarray,
        university: str,
        rows: np.ndarray,
        exclude_stubs: bool,
    ) -> np.ndarray:
        """Qdrant يطبّق الفلاتر نفسها ويرجع كل ما يطابقها لا أفضله فقط، لأن الدمج مع
        البحث اللفظي يطبّع الدرجات على المجموعة كاملة. ونتأكد أنه أرجع المقاطع نفسها."""
        from qdrant_client import models

        conditions = []
        if exclude_stubs:
            conditions.append(
                models.FieldCondition(key="is_stub", match=models.MatchValue(value=False))
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        points = self.qdrant.query_points(
            config.QDRANT_COLLECTIONS[university],
            query=vector.tolist(),
            query_filter=query_filter,
            limit=len(rows),
            with_payload=False,
        ).points
        scores = {str(point.id): point.score for point in points}
        expected_ids = [self.point_ids[row] for row in rows]
        if set(scores) != set(expected_ids):
            raise RuntimeError(
                "فلتر Qdrant أرجع مقاطع غير التي حسبها بايثون — شغّل: python index_qdrant.py"
            )
        return np.array([scores[pid] for pid in expected_ids], dtype=np.float32)

    def load_model(self) -> None:
        if self.model is None:
            self.model = load_embedding_model(device="cpu")
            self.sparse_head = load_sparse_head(self.model[0].auto_model.config.hidden_size)

    def encode_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """أوزان كلمات كل نص (رأس sparse في bge-m3)."""
        self.load_model()
        return sparse_weights(self.model, self.sparse_head, texts)

    def query_sparse_weights(self, query: str) -> dict[str, float]:
        """أوزان كلمات السؤال، محفوظة لآخر الأسئلة: السؤال يُبحث به أكثر من مرة."""
        if query not in self.query_weights:
            if len(self.query_weights) >= config.QUERY_SPARSE_CACHE_SIZE:
                del self.query_weights[next(iter(self.query_weights))]
            self.query_weights[query] = self.encode_sparse([query])[0]
        return self.query_weights[query]

    def lexical_scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        weights = self.query_sparse_weights(query)
        return np.array(
            [lexical_score(weights, self.sparse_weights[row]) for row in rows], dtype=np.float32
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        self.load_model()
        return self.model.encode(
            texts,
            batch_size=config.QUERY_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def filter_mask(self, rows: np.ndarray, exclude_stubs: bool) -> np.ndarray:
        mask = np.ones(len(rows), dtype=bool)
        if exclude_stubs:
            mask &= ~self.stubs[rows]
        return mask

    def top_hits(self, rows: np.ndarray, combined: np.ndarray, k: int) -> list[dict]:
        """أفضل k مقاطع؛ النص المكرر في وثيقتين يظهر مرة واحدة."""
        hits = []
        seen_hashes = set()
        for position in np.argsort(-combined):
            chunk = self.chunks[rows[position]]
            digest = chunk.get("content_hash")
            if digest and digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            hits.append({**chunk, "score": float(combined[position])})
            if len(hits) == k:
                break
        return hits

    def search(
        self,
        query: str,
        k: int = 5,
        university: str | None = None,
        query_vector: np.ndarray | None = None,
        exclude_stubs: bool = True,
    ) -> list[dict]:
        if query_vector is None:
            query_vector = self.encode([query])[0]
        if university is None:
            return self.search_both_universities(query, k, query_vector, exclude_stubs)
        university = university.upper()
        rows = self.university_rows[university]
        mask = self.filter_mask(rows, exclude_stubs)
        if not mask.any():
            return []
        rows = rows[mask]
        sparse = self.lexical_scores(query, rows)
        dense = self.dense_scores(query_vector, university, rows, exclude_stubs)
        combined = config.DENSE_WEIGHT * scale_to_unit(
            dense
        ) + config.SPARSE_WEIGHT * scale_to_unit(sparse)
        return self.top_hits(rows, combined, k)

    def search_both_universities(
        self,
        query: str,
        k: int,
        query_vector: np.ndarray,
        exclude_stubs: bool,
    ) -> list[dict]:
        """سؤال بلا جامعة (سطر الأوامر فقط؛ الموقع يشترطها): درجة كل دولاب مطبّعة
        من 0 إلى 1 فتُدمج القائمتان مباشرة."""
        hits = []
        for university in config.QDRANT_COLLECTIONS:
            hits += self.search(
                query,
                k=k,
                university=university,
                query_vector=query_vector,
                exclude_stubs=exclude_stubs,
            )
        hits.sort(key=lambda hit: -hit["score"])
        return hits[:k]

    def corpus(self) -> dict:
        """الوثائق المفهرسة لكل جامعة وعدد مقاطعها — لقائمة الوثائق في الواجهة."""
        universities = {}
        for chunk in self.chunks:
            if chunk.get("is_stub"):
                continue
            university = universities.setdefault(
                chunk["university"], {"name": chunk["university_ar"], "docs": {}}
            )
            doc = university["docs"].setdefault(
                chunk["doc_title"],
                {"chunks": 0, "type": chunk.get("doc_type")},
            )
            doc["chunks"] += 1
        result = {}
        for code, university in universities.items():
            docs = []
            for title, doc in university["docs"].items():
                docs.append({"title": title, **doc})
            docs.sort(key=lambda doc: -doc["chunks"])
            total = sum(doc["chunks"] for doc in docs)
            result[code] = {"name": university["name"], "chunks": total, "docs": docs}
        return result


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--university", choices=list(config.UNIVERSITY_NAMES))
    args = parser.parse_args()

    hits = Retriever().search(args.query, k=args.k, university=args.university)
    for rank, hit in enumerate(hits, 1):
        print(f"\n[{rank}] {hit['chunk_id']}  score={hit['score']:.3f}  ({hit['university']})")
        print(f"    {hit['doc_title']} — {hit.get('article_label') or '—'}")
        print("   ", hit["body"][:400].replace("\n", " "))


if __name__ == "__main__":
    main()
