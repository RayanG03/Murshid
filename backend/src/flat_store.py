"""Small-corpus persistent vector store without Chroma/HNSW.

Stores original vectors, documents, IDs and metadata in atomic NPZ files.
Search is exact cosine similarity; no third-party service or re-embedding
is used at query time. Restart serving after every external ingest.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Sequence

import numpy as np


def _name(name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,128}', name):
        raise ValueError('Invalid vector collection name')
    return name


class FlatCollection:
    """Minimal collection contract used by Murshid's ingestion/retrieval."""

    def __init__(self, name: str, path: Path, *, create: bool = False) -> None:
        self.name = _name(name)
        self.path = path
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._normalized: np.ndarray | None = None
        if create:
            self._save()
        else:
            with np.load(path, allow_pickle=False) as data:
                payload = json.loads(str(data['records'].item()))
                self._vectors = np.asarray(data['embeddings'], dtype=np.float32)
            if payload.get('version') != 1:
                raise ValueError(f'Unsupported flat index format: {name}')
            self._ids = payload['ids']
            self._documents = payload['documents']
            self._metadatas = payload['metadatas']
            if not (len(self._ids) == len(self._documents) == len(self._metadatas) == len(self._vectors)):
                raise ValueError(f'Incomplete flat index: {name}')
            if len(set(self._ids)) != len(self._ids):
                raise ValueError(f'Duplicate IDs in flat index: {name}')
            self._validate_vectors(self._vectors)

    @staticmethod
    def _validate_vectors(vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError('Embedding matrix must be finite and two-dimensional')
        if len(vectors):
            norms = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError('Embeddings must have finite, positive norms')

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'version': 1, 'ids': self._ids,
                              'documents': self._documents,
                              'metadatas': self._metadatas}, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as fh:
                np.savez_compressed(fh, records=np.asarray(payload), embeddings=self._vectors)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink()

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def add(self, *, ids: Sequence[str], documents: Sequence[str],
            metadatas: Sequence[dict], embeddings: Sequence[Sequence[float]]) -> None:
        incoming = np.asarray(embeddings, dtype=np.float32)
        new_ids = [str(i) for i in ids]
        self._validate_vectors(incoming)
        if not (len(new_ids) == len(documents) == len(metadatas) == len(incoming)):
            raise ValueError('Batch field lengths differ')
        with self._lock:
            if len(set(new_ids)) != len(new_ids) or set(new_ids) & set(self._ids):
                raise ValueError('Duplicate vector IDs')
            if len(self._ids) and incoming.shape[1] != self._vectors.shape[1]:
                raise ValueError('Embedding dimension mismatch')
            old = (self._ids, self._documents, self._metadatas, self._vectors, self._normalized)
            self._ids = self._ids + new_ids
            self._documents = self._documents + list(documents)
            self._metadatas = self._metadatas + [dict(m) for m in metadatas]
            self._vectors = np.vstack((self._vectors, incoming)) if old[0] else incoming
            self._normalized = None
            try:
                self._save()
            except Exception:
                self._ids, self._documents, self._metadatas, self._vectors, self._normalized = old
                raise

    def get(self, *, include: Sequence[str] = ('documents', 'metadatas'),
            limit: int | None = None, offset: int = 0) -> dict:
        if not set(include) <= {'documents', 'metadatas', 'embeddings'}:
            raise ValueError('Unsupported get include field')
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError('Negative get offset/limit')
        with self._lock:
            end = None if limit is None else offset + limit
            result = {'ids': list(self._ids[offset:end])}
            if 'documents' in include:
                result['documents'] = list(self._documents[offset:end])
            if 'metadatas' in include:
                result['metadatas'] = [dict(m) for m in self._metadatas[offset:end]]
            if 'embeddings' in include:
                result['embeddings'] = self._vectors[offset:end].copy()
            return result

    def query(self, *, query_embeddings: Sequence[Sequence[float]], n_results: int = 10,
              include: Sequence[str] = ('documents', 'metadatas', 'distances')) -> dict:
        if not set(include) <= {'documents', 'metadatas', 'distances', 'embeddings'}:
            raise ValueError('Unsupported query include field')
        if n_results <= 0:
            raise ValueError('n_results must be positive')
        queries = np.asarray(query_embeddings, dtype=np.float32)
        self._validate_vectors(queries)
        with self._lock:
            if len(self._ids) == 0:
                return {key: [[] for _ in queries] for key in ('ids', *include)}
            if queries.shape[1] != self._vectors.shape[1]:
                raise ValueError('Query embedding dimension mismatch')
            if self._normalized is None:
                self._normalized = self._vectors / np.linalg.norm(self._vectors, axis=1)[:, None]
            scores = np.clip((queries / np.linalg.norm(queries, axis=1)[:, None]) @ self._normalized.T, -1., 1.)
            result = {key: [] for key in ('ids', *include)}
            for row in scores:
                order = np.argsort(-row, kind='stable')[:min(n_results, len(self._ids))]
                result['ids'].append([self._ids[int(i)] for i in order])
                if 'documents' in include:
                    result['documents'].append([self._documents[int(i)] for i in order])
                if 'metadatas' in include:
                    result['metadatas'].append([dict(self._metadatas[int(i)]) for i in order])
                if 'distances' in include:
                    result['distances'].append([float(1. - row[i]) for i in order])
                if 'embeddings' in include:
                    result['embeddings'].append(self._vectors[order].copy())
            return result


class FlatClient:
    """A process-local collection registry backed by portable NPZ files."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, FlatCollection] = {}
        self._lock = threading.RLock()

    def _path(self, name: str) -> Path:
        return self.directory / (_name(name) + '.npz')

    def get_collection(self, name: str) -> FlatCollection:
        with self._lock:
            if name not in self._collections:
                path = self._path(name)
                if not path.is_file():
                    raise KeyError(f'Flat collection not found: {name}')
                self._collections[name] = FlatCollection(name, path)
            return self._collections[name]

    def create_collection(self, name: str, metadata: dict | None = None) -> FlatCollection:
        with self._lock:
            path = self._path(name)
            if path.exists():
                raise ValueError(f'Collection already exists: {name}')
            collection = FlatCollection(name, path, create=True)
            self._collections[name] = collection
            return collection

    def delete_collection(self, name: str) -> None:
        with self._lock:
            self._path(name).unlink(missing_ok=True)
            self._collections.pop(name, None)
