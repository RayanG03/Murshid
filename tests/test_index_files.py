"""اختبارات تنظيم الفهرس: مجلد لكل جامعة (save_index في embedding.py)، يقرؤه البحث
و index_qdrant.py والتحديث التزايدي بترتيب ثابت (KAU ثم KSU).

لا تحمّل النموذج: تكتب فهرساً صغيراً في مجلد مؤقت ثم تقرؤه.

التشغيل:
    python -m pytest tests/test_index_files.py
"""

import json
from pathlib import Path

import numpy as np

import config
from embedding import index_paths, read_old_index, save_index
from index_qdrant import load_index
from retrieve import Retriever


def make_chunk(chunk_id: str, university: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "university": university,
        "doc_title": "دليل تجريبي",
        "stage": config.UNDERGRADUATE_STAGE,
        "is_stub": False,
        "text": text,
        "body": text,
    }


def build_index(index_dir: Path) -> np.ndarray:
    chunks = [
        make_chunk("ksu_a", "KSU", "نص أ"),
        make_chunk("kau_b", "KAU", "نص ب"),
        make_chunk("ksu_c", "KSU", "نص ج"),
    ]
    matrix = np.eye(3, 4, dtype=np.float32)
    weights = [{"1": 0.1}, {"2": 0.2}, {"3": 0.3}]
    save_index(index_dir, chunks, matrix, weights, [])
    return matrix


def test_each_university_gets_its_own_folder(tmp_path: Path) -> None:
    build_index(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["universities"]["KSU"]["count"] == 2
    assert manifest["universities"]["KAU"]["count"] == 1
    for university in ("KSU", "KAU"):
        for path in index_paths(tmp_path, university).values():
            assert path.exists()
    ksu = index_paths(tmp_path, "KSU")
    lines = ksu["chunks"].read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["chunk_id"] for line in lines] == ["ksu_a", "ksu_c"]
    assert np.load(ksu["vectors"]).shape == (2, 4)


def test_retriever_joins_universities_in_fixed_order(tmp_path: Path) -> None:
    matrix = build_index(tmp_path)
    retriever = Retriever(tmp_path, use_qdrant=False)
    assert [chunk["chunk_id"] for chunk in retriever.chunks] == ["kau_b", "ksu_a", "ksu_c"]
    assert np.array_equal(retriever.vectors, matrix[[1, 0, 2]])
    assert retriever.university_rows["KSU"].tolist() == [1, 2]


def test_qdrant_loader_and_incremental_reuse_read_the_folders(tmp_path: Path) -> None:
    matrix = build_index(tmp_path)
    chunks, vectors, _ = load_index(tmp_path)
    assert [chunk["chunk_id"] for chunk in chunks] == ["kau_b", "ksu_a", "ksu_c"]
    assert np.array_equal(vectors, matrix[[1, 0, 2]])
    old_vectors, old_weights = read_old_index(tmp_path)
    assert np.array_equal(old_vectors["نص ج"], matrix[2])
    assert old_weights["نص ب"] == {"2": 0.2}
