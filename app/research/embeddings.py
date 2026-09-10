"""API-based embeddings and a persistent numpy vector index.

Uses the configured LLM endpoint's embedding model (OpenAI-compatible
embeddings API). Vectors are cached on disk under
data/research/embeddings/<model>/ so index rebuilds are cheap.

Note: local embedding models (sentence-transformers etc.) would need
torch, whose DLL fails to load on this machine, hence the API path.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.llm import LLM

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "research"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
BATCH_SIZE = 16


async def embed_texts(
    texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """Embed texts via the API in batches, returning an (n, dim) array."""
    llm = LLM()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = await llm.client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return np.asarray(vectors, dtype=np.float32)


class VectorIndex:
    """Cosine-similarity index over chunk embeddings, persisted to disk."""

    def __init__(self, cache_dir: Optional[Path] = None, model: str = DEFAULT_EMBEDDING_MODEL):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_ROOT / "embeddings" / model
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ build

    async def build(self, ids: List[str], texts: List[str]) -> "VectorIndex":
        """Embed texts and persist ids + vectors to the cache dir."""
        vectors = await embed_texts(texts, model=self.model)
        self.ids = list(ids)
        self.vectors = vectors
        self._norms = np.linalg.norm(vectors, axis=1)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )
        np.save(self.cache_dir / "vectors.npy", vectors)
        return self

    @classmethod
    def load(cls, cache_dir: Optional[Path] = None, model: str = DEFAULT_EMBEDDING_MODEL) -> "VectorIndex":
        index = cls(cache_dir=cache_dir, model=model)
        ids_path = index.cache_dir / "ids.json"
        vectors_path = index.cache_dir / "vectors.npy"
        if not ids_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"No cached vectors in {index.cache_dir}")
        index.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        index.vectors = np.load(vectors_path)
        index._norms = np.linalg.norm(index.vectors, axis=1)
        return index

    def exists(self) -> bool:
        return (self.cache_dir / "ids.json").exists() and (
            self.cache_dir / "vectors.npy"
        ).exists()

    # ------------------------------------------------------------------ search

    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Return top_k (chunk_id, cosine score) pairs for a query."""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        query_vec = (await embed_texts([query], model=self.model))[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        scores = (self.vectors @ query_vec) / (self._norms * query_norm)
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
