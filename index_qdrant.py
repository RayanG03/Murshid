"""المرحلة 4 من خط البناء: ينقل الفهرس الجاهز (data/index/ksu و data/index/kau) إلى Qdrant دون
إعادة حساب المتجهات.

دولاب (collection) لكل جامعة: ksu_rules و kau_rules، بمسافة cosine. معرّف كل نقطة
uuid5 مشتق من chunk_id فيبقى ثابتاً ما بقي المعرّف، والمقطع كاملاً في الـ payload.
كل تشغيل يحذف الدولابين ويعيد إنشاءهما.

التشغيل (Qdrant يعمل في Docker على localhost:6333):
    python index_qdrant.py --check     يجهّز النقاط ويتحقق منها دون اتصال
    python index_qdrant.py             يرفعها إلى Qdrant
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

import config
from embedding import index_paths, index_universities, sha256_file

FILTER_FIELDS = {"is_stub": "bool"}


def point_id(chunk_id: str) -> str:
    """Qdrant لا يقبل نصاً معرّفاً، فالمعرّف uuid5 ثابت مشتق من chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_index(index_dir: Path) -> tuple[list[dict], np.ndarray, dict]:
    """مقاطع كل جامعة ومتجهاتها من مجلدها، بعد مطابقة بصماتها مع الـ manifest."""
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if "universities" not in manifest:
        raise ValueError("الفهرس بالتنظيم القديم (ملف واحد) — شغّل: python embedding.py")
    chunks = []
    vectors = []
    for university in index_universities():
        paths = index_paths(index_dir, university)
        fingerprints = manifest["universities"][university]["sha256"]
        for path in (paths["chunks"], paths["vectors"]):
            if sha256_file(path) != fingerprints[path.name]:
                raise ValueError(f"{path} لا يطابق الـ manifest — شغّل: python embedding.py")
        university_chunks = []
        for line in paths["chunks"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                university_chunks.append(json.loads(line))
        university_vectors = np.load(paths["vectors"])
        if university_vectors.shape != (len(university_chunks), manifest["dimensions"]):
            raise ValueError(
                f"شكل متجهات {university} {university_vectors.shape} "
                f"لا يطابق {len(university_chunks)} مقطعاً"
            )
        chunks += university_chunks
        vectors.append(university_vectors)
    return chunks, np.concatenate(vectors), manifest


def build_points(chunks: list[dict], vectors: np.ndarray) -> list[dict]:
    points = []
    seen_ids = set()
    for row, chunk in enumerate(chunks):
        if chunk["chunk_id"] in seen_ids:
            raise ValueError(f"معرّف مكرر: {chunk['chunk_id']}")
        seen_ids.add(chunk["chunk_id"])
        if chunk["university"] not in config.QDRANT_COLLECTIONS:
            raise ValueError(f"جامعة بلا دولاب: {chunk['university']}")
        points.append(
            {"id": point_id(chunk["chunk_id"]), "vector": vectors[row].tolist(), "payload": chunk}
        )
    return points


def upload(points: list[dict], dimensions: int, url: str) -> None:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=url)
    for university, name in config.QDRANT_COLLECTIONS.items():
        university_points = []
        for point in points:
            if point["payload"]["university"] == university:
                university_points.append(models.PointStruct(**point))
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )
        for field, schema in FILTER_FIELDS.items():
            client.create_payload_index(name, field, field_schema=models.PayloadSchemaType(schema))
        client.upload_points(name, university_points, batch_size=64, wait=True)
        stored = client.count(name, exact=True).count
        if stored != len(university_points):
            raise ValueError(f"{name}: رُفع {stored} من أصل {len(university_points)}")
        print(f"رُفعت {stored} نقطة إلى {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="تجهيز وتحقق فقط، دون رفع")
    args = parser.parse_args()

    chunks, vectors, manifest = load_index(config.INDEX_DIR)
    points = build_points(chunks, vectors)
    print(f"{len(points)} نقطة جاهزة، طول المتجه {manifest['dimensions']}")
    for university, name in config.QDRANT_COLLECTIONS.items():
        count = sum(1 for point in points if point["payload"]["university"] == university)
        print(f"  {name}: {count} نقطة")
    if not args.check:
        upload(points, manifest["dimensions"], config.QDRANT_URL)


if __name__ == "__main__":
    main()
