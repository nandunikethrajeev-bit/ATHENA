"""Dense vector indexing and cosine semantic retrieval for ATHENA (Milestone M3).

Provides in-memory and persistent vector index capabilities using pure-Python
vector math. Computes exact cosine similarities with zero external vector database engines.
"""

from collections.abc import Sequence
import math
from typing import Any

from app.evidence.embeddings import EmbeddingClient, get_embedding_client
from app.evidence.models import EvidenceChunk, EvidenceMatch, EvidenceType


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity score in range [-1.0, 1.0], or 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class VectorIndex:
    """Dense vector index supporting top-K semantic nearest-neighbor retrieval."""

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        """Initialize VectorIndex.

        Args:
            embedding_client: EmbeddingClient to use for computing vectors when not pre-embedded.
        """
        self.embedding_client = embedding_client or get_embedding_client()
        self._chunks: dict[str, EvidenceChunk] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def total_vectors(self) -> int:
        """Return the number of indexed vectors."""
        return len(self._vectors)

    def clear(self) -> None:
        """Clear all indexed vectors and chunks."""
        self._chunks.clear()
        self._vectors.clear()

    def add_chunk(
        self,
        chunk: EvidenceChunk,
        vector: Sequence[float] | None = None,
    ) -> None:
        """Index a single EvidenceChunk and its vector representation.

        Args:
            chunk: The EvidenceChunk to index.
            vector: Optional precomputed embedding vector. If None, computes vector from text.
        """
        if vector is None:
            if chunk.embedding is not None:
                vector = chunk.embedding
            else:
                combined_text = f"{chunk.source.paper_title or ''} {chunk.text}"
                vector = self.embedding_client.embed_text(combined_text)
                chunk.embedding = list(vector)

        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = [float(x) for x in vector]

    def add_chunks(
        self,
        chunks: Sequence[EvidenceChunk],
        vectors: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Index multiple EvidenceChunks."""
        if vectors is not None:
            for chunk, vec in zip(chunks, vectors):
                self.add_chunk(chunk, vector=vec)
        else:
            for chunk in chunks:
                self.add_chunk(chunk)

    def retrieve_semantic(
        self,
        query: str | None = None,
        query_vector: Sequence[float] | None = None,
        top_k: int = 5,
        evidence_types: Sequence[EvidenceType] | None = None,
    ) -> list[EvidenceMatch]:
        """Retrieve top-K evidence chunks closest to query using cosine similarity.

        Args:
            query: Query string to embed and match.
            query_vector: Optional precomputed query vector.
            top_k: Maximum number of matches to return (must be > 0).
            evidence_types: Optional filter for specific EvidenceType(s).

        Returns:
            List of EvidenceMatch objects with similarity scores, sorted descending.
        """
        if top_k <= 0 or not self._vectors:
            return []

        if query_vector is None:
            if not query or not query.strip():
                return []
            query_vector = self.embedding_client.embed_text(query)

        type_filter = set(evidence_types) if evidence_types is not None else None
        scored_matches: list[EvidenceMatch] = []

        for cid, chunk_vec in self._vectors.items():
            chunk = self._chunks[cid]
            if type_filter is not None and chunk.evidence_type not in type_filter:
                continue

            sim = cosine_similarity(query_vector, chunk_vec)
            # Only consider positive similarity
            if sim > 0.0:
                scored_matches.append(
                    EvidenceMatch(
                        chunk=chunk,
                        score=round(sim, 4),
                        matched_terms=["semantic_match"],
                        retrieval_mode="semantic",
                    )
                )

        scored_matches.sort(key=lambda m: m.score, reverse=True)
        return scored_matches[:top_k]
