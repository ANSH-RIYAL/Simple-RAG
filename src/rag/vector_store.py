import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class SimpleVectorStore:
    """Minimal vector store with on-disk persistence.

    Files written under index_dir:
      - embeddings.npy (float32, shape [N, D])
      - items.jsonl (one JSON per line with id, text, metadata)
      - config.json (contains embedding_dim and counts)
    """

    def __init__(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self._embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self._items: List[Dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item_id: str, text: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"vector shape {vector.shape} != ({self.embedding_dim},)")
        self._embeddings = np.vstack([self._embeddings, vector.reshape(1, -1)])
        self._items.append({"id": item_id, "text": text, "metadata": metadata or {}})

    def save(self, index_dir: str) -> None:
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "embeddings.npy", self._embeddings)
        with (p / "items.jsonl").open("w", encoding="utf-8") as f:
            for item in self._items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (p / "config.json").open("w", encoding="utf-8") as f:
            json.dump({"embedding_dim": self.embedding_dim, "count": len(self._items)}, f)

    @classmethod
    def load(cls, index_dir: str) -> "SimpleVectorStore":
        p = Path(index_dir)
        with (p / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        store = cls(embedding_dim=int(cfg["embedding_dim"]))
        store._embeddings = np.load(p / "embeddings.npy").astype(np.float32)
        store._items = []
        with (p / "items.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                store._items.append(json.loads(line))
        return store

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self._embeddings.size == 0:
            return []
        if query_vector.shape != (self.embedding_dim,):
            raise ValueError(f"query_vector shape {query_vector.shape} != ({self.embedding_dim},)")
        # Cosine similarity with L2-normalized vectors
        A = self._embeddings
        q = query_vector.reshape(1, -1)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = (A_norm @ q_norm.T).reshape(-1)
        idx = np.argsort(-sims)[:top_k]
        results: List[Dict] = []
        for i in idx:
            item = self._items[int(i)]
            results.append({
                "id": item["id"],
                "score": float(sims[int(i)]),
                "text": item["text"],
                "metadata": item["metadata"],
            })
        return results
