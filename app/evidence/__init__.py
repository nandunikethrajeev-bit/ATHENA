"""Evidence processing, persistent storage, and hybrid retrieval subsystem for ATHENA (Milestones M2 & M3).

Converts retrieved scientific literature into traceable, normalized evidence items,
provides persistent SQLite storage, local dense vector embeddings, and hybrid
(BM25 + dense vector RRF) retrieval ready for downstream synthesis and citation.
"""

from app.evidence.chunker import (
    chunk_document,
    chunk_documents,
    create_metadata_chunk,
    split_scientific_sentences,
)
from app.evidence.cleaner import clean_scientific_text
from app.evidence.context import build_evidence_context, format_evidence_item
from app.evidence.embeddings import (
    EmbeddingClient,
    EmbeddingModelInfo,
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    get_embedding_client,
)
from app.evidence.hybrid import HybridRetriever, compute_rrf_score
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)
from app.evidence.normalizer import normalize_paper
from app.evidence.retriever import EvidenceIndex, tokenize
from app.evidence.store import EvidenceStore
from app.evidence.vector_index import VectorIndex, cosine_similarity

__all__ = [
    "EmbeddingClient",
    "EmbeddingModelInfo",
    "EvidenceChunk",
    "EvidenceIndex",
    "EvidenceMatch",
    "EvidenceSource",
    "EvidenceStore",
    "EvidenceType",
    "HybridRetriever",
    "MockEmbeddingClient",
    "NormalizedDocument",
    "OllamaEmbeddingClient",
    "VectorIndex",
    "build_evidence_context",
    "chunk_document",
    "chunk_documents",
    "clean_scientific_text",
    "compute_rrf_score",
    "cosine_similarity",
    "create_metadata_chunk",
    "format_evidence_item",
    "get_embedding_client",
    "normalize_paper",
    "split_scientific_sentences",
    "tokenize",
]
