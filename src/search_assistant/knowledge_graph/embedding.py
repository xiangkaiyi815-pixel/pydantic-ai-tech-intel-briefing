from __future__ import annotations

import hashlib
import os
from typing import Protocol


def _embedding_cache_key(text: str) -> str:
    """Return a stable hash for a piece of text used as an embedding cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without extra dependencies."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b) ** 0.5


class EmbeddingProvider(Protocol):
    """Protocol for turning a batch of texts into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class NullEmbeddingProvider:
    """Provider that returns empty vectors, effectively disabling semantic matching."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider (also works with compatible endpoints)."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI embedding API supports up to 2048 texts per request; keep a safe batch size.
        batch_size = 64
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            results.extend(item.embedding for item in response.data)
        return results


def build_embedding_provider(
    enabled: bool | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    """Build an embedding provider from explicit arguments or environment variables.

    Args:
        enabled: Whether semantic matching is enabled. Defaults to True unless
            ``SEARCH_ASSISTANT_EMBEDDING_ENABLED`` is set to ``false``/``0``/``no``.
        api_key: OpenAI-compatible API key. Falls back to ``OPENAI_API_KEY``.
        base_url: OpenAI-compatible base URL. Falls back to ``OPENAI_BASE_URL``.
        model: Embedding model name. Falls back to ``OPENAI_EMBEDDING_MODEL``
            then ``text-embedding-3-small``.
    """
    if enabled is None:
        raw = os.getenv("SEARCH_ASSISTANT_EMBEDDING_ENABLED", "true").strip().lower()
        enabled = raw not in {"false", "0", "no", "off"}

    if not enabled:
        return NullEmbeddingProvider()

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return NullEmbeddingProvider()

    return OpenAIEmbeddingProvider(
        api_key=key,
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
