"""Hybrid evidence retrieval combining BM25 lexical ranking and dense vector similarity for ATHENA (Milestone M3).

Uses Reciprocal Rank Fusion (RRF) to merge lexical keyword signals with semantic conceptual
embeddings, producing an optimal ranking of evidence chunks.
"""

from collections.abc import Sequence
from typing import Literal

from app.evidence.embeddings import EmbeddingClient, get_embedding_client
from app.evidence.models import EvidenceChunk, EvidenceMatch, EvidenceType
from app.evidence.retriever import EvidenceIndex
from app.evidence.vector_index import VectorIndex


def compute_rrf_score(
    rank_lex: int | None,
    rank_sem: int | None,
    k: int = 60,
    weight_lex: float = 1.0,
    weight_sem: float = 1.0,
) -> float:
    """Compute Reciprocal Rank Fusion (RRF) score for a document.

    Formula:
        Score(d) = (weight_lex / (k + rank_lex)) + (weight_sem / (k + rank_sem))

    Args:
        rank_lex: 1-based rank in lexical results (None if not ranked).
        rank_sem: 1-based rank in semantic results (None if not ranked).
        k: Smoothing constant (standard: 60).
        weight_lex: Multiplier for lexical rank.
        weight_sem: Multiplier for semantic rank.

    Returns:
        Combined RRF score float.
    """
    score = 0.0
    if rank_lex is not None:
        score += weight_lex / (k + rank_lex)
    if rank_sem is not None:
        score += weight_sem / (k + rank_sem)
    return score


class HybridRetriever:
    """Combines BM25Okapi lexical retrieval with dense vector cosine retrieval."""

    def __init__(
        self,
        bm25_index: EvidenceIndex | None = None,
        vector_index: VectorIndex | None = None,
        embedding_client: EmbeddingClient | None = None,
        rrf_k: int = 60,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            bm25_index: Optional pre-configured EvidenceIndex.
            vector_index: Optional pre-configured VectorIndex.
            embedding_client: EmbeddingClient used to initialize VectorIndex if omitted.
            rrf_k: Reciprocal Rank Fusion smoothing parameter (default: 60).
        """
        self.bm25_index = bm25_index or EvidenceIndex()
        self.vector_index = vector_index or VectorIndex(embedding_client=embedding_client or get_embedding_client())
        self.rrf_k = rrf_k
        self._chunks: dict[str, EvidenceChunk] = {}

    def index_chunks(self, chunks: Sequence[EvidenceChunk]) -> None:
        """Index evidence chunks into both the BM25 store and dense vector index.

        Args:
            chunks: Sequence of EvidenceChunk instances.
        """
        self._chunks.clear()
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

        self.bm25_index.index_chunks(chunks)
        self.vector_index.clear()
        self.vector_index.add_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        evidence_types: Sequence[EvidenceType] | None = None,
        mode: Literal["bm25", "semantic", "hybrid"] = "hybrid",
        weight_lexical: float = 1.0,
        weight_semantic: float = 1.0,
    ) -> list[EvidenceMatch]:
        """Retrieve and rank evidence chunks using the specified retrieval strategy.

        Args:
            query: The scientific question or search query.
            top_k: Number of ranked chunks to return.
            evidence_types: Optional filter for EvidenceType.
            mode: 'bm25' (lexical only), 'semantic' (dense vector only), or 'hybrid' (RRF).
            weight_lexical: Weight factor for lexical ranking in hybrid mode.
            weight_semantic: Weight factor for semantic ranking in hybrid mode.

        Returns:
            List of EvidenceMatch instances sorted by score descending.
        """
        if not query or not query.strip() or top_k <= 0 or not self._chunks:
            return []

        if mode == "bm25":
            return self.bm25_index.retrieve(query=query, top_k=top_k, evidence_types=evidence_types)

        if mode == "semantic":
            return self.vector_index.retrieve_semantic(
                query=query, top_k=top_k, evidence_types=evidence_types
            )

        # Hybrid mode: Retrieve candidates from both systems with broader depth
        retrieval_depth = max(top_k * 3, 20)
        lex_matches = self.bm25_index.retrieve(
            query=query, top_k=retrieval_depth, evidence_types=evidence_types
        )
        sem_matches = self.vector_index.retrieve_semantic(
            query=query, top_k=retrieval_depth, evidence_types=evidence_types
        )

        lex_rank_map: dict[str, int] = {m.chunk.chunk_id: idx for idx, m in enumerate(lex_matches, 1)}
        sem_rank_map: dict[str, int] = {m.chunk.chunk_id: idx for idx, m in enumerate(sem_matches, 1)}
        lex_matched_terms: dict[str, list[str]] = {m.chunk.chunk_id: m.matched_terms for m in lex_matches}

        all_candidate_ids = set(lex_rank_map.keys()) | set(sem_rank_map.keys())
        scored_candidates: list[EvidenceMatch] = []

        for cid in all_candidate_ids:
            chunk = self._chunks.get(cid)
            if chunk is None:
                continue

            r_lex = lex_rank_map.get(cid)
            r_sem = sem_rank_map.get(cid)

            combined_score = compute_rrf_score(
                rank_lex=r_lex,
                rank_sem=r_sem,
                k=self.rrf_k,
                weight_lex=weight_lexical,
                weight_sem=weight_semantic,
            )

            matched_terms = lex_matched_terms.get(cid, [])
            if r_sem is not None:
                matched_terms = list(matched_terms) + ["semantic"]

            scored_candidates.append(
                EvidenceMatch(
                    chunk=chunk,
                    score=round(combined_score, 6),
                    matched_terms=matched_terms,
                    retrieval_mode="hybrid",
                )
            )

        scored_candidates.sort(key=lambda m: m.score, reverse=True)
        return scored_candidates[:top_k]
