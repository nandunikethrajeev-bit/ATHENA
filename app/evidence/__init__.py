"""Evidence processing and retrieval subsystem for ATHENA (Milestone M2).

Converts retrieved scientific literature into traceable, normalized evidence items
ready for downstream synthesis and citation.
"""

from app.evidence.chunker import (
    chunk_document,
    chunk_documents,
    create_metadata_chunk,
    split_scientific_sentences,
)
from app.evidence.cleaner import clean_scientific_text
from app.evidence.context import build_evidence_context, format_evidence_item
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)
from app.evidence.normalizer import normalize_paper
from app.evidence.retriever import EvidenceIndex, tokenize

__all__ = [
    "EvidenceChunk",
    "EvidenceIndex",
    "EvidenceMatch",
    "EvidenceSource",
    "EvidenceType",
    "NormalizedDocument",
    "build_evidence_context",
    "chunk_document",
    "chunk_documents",
    "clean_scientific_text",
    "create_metadata_chunk",
    "format_evidence_item",
    "normalize_paper",
    "split_scientific_sentences",
    "tokenize",
]
