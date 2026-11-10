"""المرحلة 3 من خط البناء: يحسب متجهات المقاطع وأوزان كلماتها بـ bge-m3 ويحفظ الفهرس.

المدخلات: data/chunks/*.jsonl (من chunk.py).
المخرجات في data/index/، مجلد لكل جامعة (ksu/ و kau/) فيه:
- chunks.jsonl و embeddings.npy (البحث الدلالي) و sparse_weights.json (البحث اللفظي: وزن
  كل رمز في المقطع)، بنفس ترتيب الصفوف.
ومعها data/index/manifest.json: النموذج ومراجعته، وأعداد كل جامعة وبصمات ملفاتها.

كل حساب بالنموذج هنا وقت البناء، لا وقت السؤال. افتراضياً يعيد استعمال متجه كل نص وأوزانه
إن لم يتغير حرفياً عن الفهرس الحالي، ويحسب الجديد والمتغير فقط: إعادة البناء كاملة تأخذ
قرابة ساعة على هذا المعالج.

التشغيل:
    python embedding.py            تحديث تزايدي (الافتراضي)، ثم python index_qdrant.py
    python embedding.py --full     حساب كل المتجهات من جديد
    python embedding.py --check    فحص المدخلات فقط، دون تحميل النموذج
    python embedding.py --download تنزيل ملفات bge-m3 أول مرة على جهاز جديد (كل تشغيل غيره
                                   يقرأ النموذج من الجهاز وحده، بلا إنترنت)
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        block = stream.read(1024 * 1024)
        while block:
            digest.update(block)
            block = stream.read(1024 * 1024)
    return digest.hexdigest()


MODEL_MISSING = (
    "ملفات bge-m3 غير موجودة على هذا الجهاز — نزّلها مرة واحدة (تحتاج إنترنت): "
    "python embedding.py --download"
)


def load_embedding_model(
    device: str | None = None, local_only: bool = True
) -> "SentenceTransformer":
    """bge-m3 بالمراجعة المثبّتة في config. الاستيراد هنا لأن torch بطيء التحميل.

    من ملفات الجهاز وحدها افتراضياً: المراجعة مثبّتة فلا نسخة أحدث تُطلب، وكانت المكتبة
    تسأل Hugging Face عند كل تحميل، فإذا انقطع الاتصال توقف التشغيل كله.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_REVISION,
            device=device,
            local_files_only=local_only,
        )
    except OSError as error:
        raise RuntimeError(MODEL_MISSING) from error


def load_sparse_head(hidden_size: int, local_only: bool = True) -> "torch.nn.Linear":
    """طبقة خطية صغيرة من مستودع bge-m3 تعطي كل رمز وزناً (من ملفات الجهاز افتراضياً)."""
    import torch
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            config.EMBEDDING_MODEL,
            config.SPARSE_HEAD_FILE,
            revision=config.EMBEDDING_REVISION,
            local_files_only=local_only,
        )
    except OSError as error:
        raise RuntimeError(MODEL_MISSING) from error
    head = torch.nn.Linear(hidden_size, 1)
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return head


def token_weights(
    token_ids: list[int], values: list[float], skip_ids: set[int]
) -> dict[str, float]:
    """وزن كل رمز في النص: أعلى وزن إن تكرر، بلا الرموز الخاصة ولا الأوزان الصفرية.

    المفتاح نص لأن ملف الأوزان JSON."""
    weights = {}
    for token_id, value in zip(token_ids, values):
        if token_id in skip_ids or value <= 0:
            continue
        key = str(token_id)
        weights[key] = max(weights.get(key, 0.0), round(value, 5))
    return weights


def sparse_weights(
    model: "SentenceTransformer", head: "torch.nn.Linear", texts: list[str]
) -> list[dict[str, float]]:
    """أوزان كلمات كل نص بترتيبه. نفس طريقة bge-m3: relu(head(آخر طبقة)) لكل رمز."""
    import torch

    transformer, tokenizer = model[0].auto_model, model[0].tokenizer
    device = transformer.device
    head = head.to(device)
    skip_ids = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    }
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    result = [None] * len(texts)
    with torch.no_grad():
        for start in range(0, len(order), config.SPARSE_BATCH_SIZE):
            batch_indices = order[start : start + config.SPARSE_BATCH_SIZE]
            batch = tokenizer(
                [texts[index] for index in batch_indices],
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(device)
            hidden = transformer(**batch).last_hidden_state
            values = torch.relu(head(hidden)).squeeze(-1)
            for index, ids, row in zip(batch_indices, batch["input_ids"].tolist(), values.tolist()):
                result[index] = token_weights(ids, row, skip_ids)
    return result


def lexical_score(query: dict[str, float], document: dict[str, float]) -> float:
    """تطابق السؤال والمقطع: مجموع (وزن الرمز في السؤال × وزنه في المقطع) للرموز المشتركة."""
    score = 0.0
    for token, weight in query.items():
        if token in document:
            score += weight * document[token]
    return score


def read_chunk_file(path: Path, seen_ids: set) -> tuple[list[dict], int]:
    """مقاطع الملف القابلة للفهرسة، وعدد المستبعَد منها (indexable = false)."""
    chunks = []
    excluded = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSON غير صالح في {path}:{line_number} — أعد تشغيل: python chunk.py"
            ) from error
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or not str(chunk.get("text", "")).strip():
            raise ValueError(f"مقطع بلا معرّف أو نص في {path}:{line_number}")
        if chunk_id in seen_ids:
            raise ValueError(f"معرّف مكرر: {chunk_id} — راجع chunk.py")
        seen_ids.add(chunk_id)
        if chunk.get("indexable", True):
            chunks.append(chunk)
        else:
            excluded += 1
    return chunks, excluded


def load_chunks(input_dir: Path) -> tuple[list[dict], list[dict]]:
    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"لا توجد ملفات مقاطع في {input_dir} — شغّل: python chunk.py")
    chunks = []
    sources = []
    seen_ids = set()
    for path in paths:
        file_chunks, excluded = read_chunk_file(path, seen_ids)
        chunks += file_chunks
        sources.append(
            {
                "file": path.name,
                "chunks": len(file_chunks),
                "excluded_review_chunks": excluded,
                "sha256": sha256_file(path),
            }
        )
    if not chunks:
        raise ValueError("لا توجد مقاطع قابلة للفهرسة")
    return chunks, sources


def index_matches_model(index_dir: Path) -> bool:
    """الفهرس الحالي مبني بنموذج config نفسه، فتصلح متجهاته لإعادة الاستعمال."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("model") != config.EMBEDDING_MODEL
        or manifest.get("revision") != config.EMBEDDING_REVISION
    ):
        print("الفهرس الحالي مبني بنموذج آخر — سيُحسب كل شيء")
        return False
    return True


def index_paths(index_dir: Path, university: str) -> dict[str, Path]:
    """ملفات فهرس جامعة واحدة في مجلدها: data/index/ksu/ أو data/index/kau/."""
    folder = index_dir / university.lower()
    return {name: folder / file_name for name, file_name in config.INDEX_FILES.items()}


def index_universities() -> list[str]:
    """ترتيب الجامعات الثابت في الفهرس (KAU ثم KSU)، وبه تُضم صفوفها في الذاكرة."""
    return sorted(config.UNIVERSITY_NAMES)


def read_old_index(index_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """متجهات الفهرس الحالي وأوزان كلماته مفهرسةً بنص المقطع."""
    if not index_matches_model(index_dir):
        return {}, {}
    old_vectors = {}
    old_weights = {}
    for university in index_universities():
        paths = index_paths(index_dir, university)
        if not paths["chunks"].exists() or not paths["vectors"].exists():
            continue
        texts = []
        for line in paths["chunks"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
        old_vectors.update(zip(texts, np.load(paths["vectors"])))
        if paths["weights"].exists():
            weights = json.loads(paths["weights"].read_text(encoding="utf-8"))
            old_weights.update(zip(texts, weights))
    return old_vectors, old_weights


def loaded_model(models: dict, device: str | None) -> "SentenceTransformer":
    """النموذج يُحمَّل مرة واحدة، وفقط إن وُجد نص جديد يحتاجه."""
    if "model" not in models:
        print(f"تحميل {config.EMBEDDING_MODEL} ...")
        models["model"] = load_embedding_model(device)
        hidden_size = models["model"][0].auto_model.config.hidden_size
        models["head"] = load_sparse_head(hidden_size)
    return models["model"]


def embed_texts(
    texts: list[str],
    old_vectors: dict[str, np.ndarray],
    old_weights: dict[str, dict],
    device: str | None,
    models: dict,
) -> tuple[np.ndarray, list[dict], int]:
    """متجه وأوزان كلمات لكل نص بترتيبه، وعدد النصوص التي حُسبت من جديد."""
    unique_texts = list(dict.fromkeys(texts))
    vector_texts = [text for text in unique_texts if text not in old_vectors]
    weight_texts = [text for text in unique_texts if text not in old_weights]
    new_vectors = {}
    new_weights = {}
    if vector_texts:
        model = loaded_model(models, device)
        encoded = model.encode(
            vector_texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        new_vectors = dict(zip(vector_texts, encoded))
    if weight_texts:
        model = loaded_model(models, device)
        new_weights = dict(zip(weight_texts, sparse_weights(model, models["head"], weight_texts)))
    vectors = [old_vectors.get(text, new_vectors.get(text)) for text in texts]
    weights = [old_weights.get(text, new_weights.get(text)) for text in texts]
    new_count = len(set(vector_texts) | set(weight_texts))
    return np.asarray(vectors, dtype=np.float32), weights, new_count


def check_index_data(chunks: list[dict], matrix: np.ndarray, weights: list[dict]) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("عدد المتجهات لا يطابق عدد المقاطع")
    if not np.isfinite(matrix).all():
        raise ValueError("في المتجهات قيم غير منتهية")
    if len(weights) != len(chunks) or any(weight is None for weight in weights):
        raise ValueError("عدد أوزان الكلمات لا يطابق عدد المقاطع")
    for chunk in chunks:
        if chunk["university"] not in config.UNIVERSITY_NAMES:
            raise ValueError(f"جامعة بلا مجلد فهرس: {chunk['university']}")


def write_index_files(
    temp_paths: dict[str, Path], chunks: list[dict], matrix: np.ndarray, weights: list[dict]
) -> None:
    with temp_paths["chunks"].open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with temp_paths["weights"].open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n" + ",\n".join(json.dumps(weight) for weight in weights) + "\n]\n")
    with temp_paths["vectors"].open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)


def save_index(
    output_dir: Path,
    chunks: list[dict],
    matrix: np.ndarray,
    weights: list[dict],
    sources: list[dict],
) -> None:
    """يكتب فهرس كل جامعة في مجلدها ومعه manifest واحد: ملفات مؤقتة أولاً ثم تُستبدل كلها
    معاً، فلا يبقى فهرس نصف مكتوب."""
    check_index_data(chunks, matrix, weights)
    university_paths = {}
    for university in index_universities():
        university_paths[university] = index_paths(output_dir, university)
    manifest_path = output_dir / "manifest.json"
    final_paths = [manifest_path]
    for paths in university_paths.values():
        final_paths += paths.values()
    temp_paths = {path: path.with_name(path.name + ".tmp") for path in final_paths}
    try:
        universities = {}
        for university, paths in university_paths.items():
            rows = [row for row, chunk in enumerate(chunks) if chunk["university"] == university]
            paths["chunks"].parent.mkdir(parents=True, exist_ok=True)
            write_index_files(
                {name: temp_paths[path] for name, path in paths.items()},
                [chunks[row] for row in rows],
                matrix[rows],
                [weights[row] for row in rows],
            )
            universities[university] = {
                "folder": paths["chunks"].parent.name,
                "count": len(rows),
                "sha256": {path.name: sha256_file(temp_paths[path]) for path in paths.values()},
            }
        manifest = {
            "model": config.EMBEDDING_MODEL,
            "revision": config.EMBEDDING_REVISION,
            "count": len(chunks),
            "dimensions": matrix.shape[1],
            "normalized": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "sparse_head": config.SPARSE_HEAD_FILE,
            "universities": universities,
        }
        with temp_paths[manifest_path].open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for path in final_paths:
            os.replace(temp_paths[path], path)
    finally:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="حساب كل المتجهات من جديد")
    parser.add_argument("--check", action="store_true", help="فحص المدخلات دون تحميل النموذج")
    parser.add_argument("--device", help="cpu أو cuda (افتراضياً: يختار النموذج)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="تنزيل ملفات bge-m3 من Hugging Face (مرة على كل جهاز جديد)",
    )
    args = parser.parse_args()

    if args.download:
        model = load_embedding_model("cpu", local_only=False)
        load_sparse_head(model[0].auto_model.config.hidden_size, local_only=False)
        print(f"نُزّلت ملفات {config.EMBEDDING_MODEL} ({config.EMBEDDING_REVISION[:8]})")
        return
    chunks, sources = load_chunks(config.CHUNKS_DIR)
    for source in sources:
        print(
            f"{source['file']}: {source['chunks']} مقطع، "
            f"{source['excluded_review_chunks']} مستبعد للمراجعة"
        )
    if args.check:
        return
    old_vectors, old_weights = ({}, {}) if args.full else read_old_index(config.INDEX_DIR)
    models = {}
    matrix, weights, new_count = embed_texts(
        [chunk["text"] for chunk in chunks], old_vectors, old_weights, args.device, models
    )
    save_index(config.INDEX_DIR, chunks, matrix, weights, sources)
    print(f"الفهرس: {len(chunks)} مقطع — حُسب من جديد {new_count} نصاً، والباقي أُعيد استعماله")
    print("التالي: python index_qdrant.py")


if __name__ == "__main__":
    main()
