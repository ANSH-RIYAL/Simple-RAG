import os
import time
import json
from typing import List, Optional

import numpy as np
import requests


class MistralEmbeddingsClient:
    """Thin client for Mistral embeddings API.

    Reads API key from MISTRAL_API_KEY env var unless provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: float = 30.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set")
        self.model = model or os.getenv("MISTRAL_EMBEDDINGS_MODEL", "mistral-embed")
        self.base_url = base_url or os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai")
        self.timeout = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        mat = np.vstack(vectors).astype(np.float32)
        return mat

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    return [np.array(item["embedding"], dtype=np.float32) for item in data]
                # Retry on 429/5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.retry_backoff_s * (2 ** attempt))
        raise RuntimeError(f"Failed to fetch embeddings after retries: {last_exc}")
