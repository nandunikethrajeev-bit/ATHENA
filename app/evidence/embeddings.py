"""Local and mock embedding generation for ATHENA (Milestone M3).

Provides a model-agnostic, deterministic embedding abstraction:
- MockEmbeddingClient: Fully deterministic, offline, zero-network, pure-Python hash-based vectors.
- OllamaEmbeddingClient: Connects to local Ollama (/api/embeddings) using existing httpx.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any, Protocol, runtime_checkable

import httpx


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Metadata identifying the embedding model and vector characteristics."""

    name: str
    dimension: int
    version: str


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for model-agnostic dense text embedding backends."""

    def embed_text(self, text: str) -> list[float]:
        """Compute a dense vector embedding for a single text string."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute dense vector embeddings for a sequence of text strings."""
        ...

    def get_model_info(self) -> EmbeddingModelInfo:
        """Return metadata for the active embedding model."""
        ...


class MockEmbeddingClient:
    """Deterministic, zero-network embedding client for offline testing and development.

    Uses deterministic token hashing and L2 normalization to produce unit-length vectors.
    Identical texts produce identical vectors; texts sharing vocabulary produce higher similarity.
    """

    def __init__(self, dimension: int = 64, model_name: str = "mock-hash-embed-v1") -> None:
        """Initialize MockEmbeddingClient.

        Args:
            dimension: Fixed length of output vector (default: 64).
            model_name: Descriptive identifier for the mock model.
        """
        self.dimension = dimension
        self.model_name = model_name
        self._info = EmbeddingModelInfo(name=model_name, dimension=dimension, version="1.0.0")

    def get_model_info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized embedding vector for text."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        # Tokenize simply into lowercase words and n-grams
        words = text.lower().split()
        vec = [0.0] * self.dimension

        for word in words:
            # Deterministic hash to dimension index and sign
            h = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dimension
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign

            # Also hash 3-grams within words to capture subword similarity
            for i in range(len(word) - 2):
                tri = word[i : i + 3]
                th = hashlib.sha256(tri.encode("utf-8")).digest()
                tidx = int.from_bytes(th[:4], "big") % self.dimension
                tsign = 0.5 if (th[4] % 2 == 0) else -0.5
                vec[tidx] += tsign

        # L2 normalize vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            return [x / norm for x in vec]
        return [0.0] * self.dimension

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of texts."""
        return [self.embed_text(t) for t in texts]


class OllamaEmbeddingClient:
    """Client for local Ollama embedding endpoints (e.g. nomic-embed-text, qwen2.5:7b)."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        """Initialize OllamaEmbeddingClient.

        Args:
            model: Embedding model tag in Ollama.
            base_url: Base URL where Ollama is listening.
            timeout: HTTP timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cached_dim: int | None = None

    def get_model_info(self) -> EmbeddingModelInfo:
        dim = self._cached_dim or 1536
        return EmbeddingModelInfo(name=self.model, dimension=dim, version="ollama-local")

    def embed_text(self, text: str) -> list[float]:
        """Request embedding from Ollama /api/embeddings."""
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError(f"Ollama response missing 'embedding' list: {data}")
                self._cached_dim = len(embedding)
                return [float(x) for x in embedding]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch embeddings from local Ollama ({endpoint}): {exc}"
            ) from exc

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_client(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> EmbeddingClient:
    """Factory creating an EmbeddingClient instance.

    Supported providers:
        - 'mock' (default for offline testing)
        - 'ollama' (connects to local Ollama server)
    """
    resolved_provider = (
        provider
        or os.environ.get("ATHENA_EMBEDDING_PROVIDER")
        or "mock"
    ).strip().lower()

    if resolved_provider == "mock":
        return MockEmbeddingClient()

    if resolved_provider == "ollama":
        target_model = model or os.environ.get("ATHENA_EMBEDDING_MODEL") or "qwen2.5:7b"
        target_url = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        return OllamaEmbeddingClient(model=target_model, base_url=target_url)

    raise ValueError(
        f"Unsupported embedding provider: '{resolved_provider}'. Supported: 'mock', 'ollama'."
    )
