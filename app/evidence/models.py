"""Data models for document normalization and evidence extraction in ATHENA (Milestone M2).

Every evidence item retains strict provenance back to its original scientific source,
and distinguishes between metadata, abstract, and full-text evidence.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    """Classification of evidence origin.

    Crucial for scientific integrity: abstracts must NEVER be conflated with full-text study bodies.
    """

    METADATA = "metadata"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"


@dataclass(frozen=True)
class EvidenceSource:
    """Immutable provenance record connecting an evidence chunk to its originating paper."""

    openalex_id: str
    paper_title: str | None
    publication_year: int | None
    doi: str | None
    landing_page_url: str | None
    venue: str | None = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    source_database: str = "OpenAlex"

    @property
    def short_id(self) -> str:
        """Extract short identifier from full OpenAlex URI (e.g. 'W12345' from 'https://openalex.org/W12345')."""
        if self.openalex_id and "/" in self.openalex_id:
            return self.openalex_id.rstrip("/").split("/")[-1]
        return self.openalex_id or ""

    def to_dict(self) -> dict[str, Any]:
        """Convert EvidenceSource to dictionary."""
        data = asdict(self)
        data["authors"] = list(self.authors)
        data["short_id"] = self.short_id
        return data


@dataclass
class NormalizedDocument:
    """Canonical normalized representation of a scientific publication in ATHENA."""

    document_id: str
    title: str | None
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    doi: str | None = None
    venue: str | None = None
    landing_page_url: str | None = None
    abstract_text: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    cited_by_count: int | None = None
    is_open_access: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None
    source_database: str = "OpenAlex"
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        """Extract short identifier (e.g. 'W12345')."""
        if self.document_id and "/" in self.document_id:
            return self.document_id.rstrip("/").split("/")[-1]
        return self.document_id or ""

    @property
    def has_abstract(self) -> bool:
        """Return True if an abstract is available and non-empty."""
        return bool(self.abstract_text and self.abstract_text.strip())

    @property
    def has_full_text(self) -> bool:
        """Return True if any full-text body sections are present."""
        return bool(self.sections)

    def to_source(self) -> EvidenceSource:
        """Construct an immutable EvidenceSource provenance record for this document."""
        return EvidenceSource(
            openalex_id=self.document_id,
            paper_title=self.title,
            publication_year=self.publication_year,
            doi=self.doi,
            landing_page_url=self.landing_page_url,
            venue=self.venue,
            authors=tuple(self.authors),
            source_database=self.source_database,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert NormalizedDocument to a JSON-serializable dictionary."""
        return {
            "document_id": self.document_id,
            "short_id": self.short_id,
            "title": self.title,
            "authors": self.authors,
            "publication_year": self.publication_year,
            "doi": self.doi,
            "venue": self.venue,
            "landing_page_url": self.landing_page_url,
            "abstract_text": self.abstract_text,
            "sections": self.sections,
            "cited_by_count": self.cited_by_count,
            "is_open_access": self.is_open_access,
            "oa_status": self.oa_status,
            "oa_url": self.oa_url,
            "source_database": self.source_database,
            "has_abstract": self.has_abstract,
            "has_full_text": self.has_full_text,
        }


@dataclass
class EvidenceChunk:
    """A discrete, traceable evidence item extracted from a scientific document.

    Retains complete provenance back to the originating paper, the specific section,
    and the evidence classification (metadata, abstract, or full_text).
    """

    chunk_id: str
    document_id: str
    source: EvidenceSource
    evidence_type: EvidenceType
    section: str
    text: str
    char_count: int
    word_count: int
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert EvidenceChunk to a JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source.to_dict(),
            "evidence_type": self.evidence_type.value,
            "section": self.section,
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "chunk_index": self.chunk_index,
        }


@dataclass
class EvidenceMatch:
    """A retrieved evidence chunk scored against a user query."""

    chunk: EvidenceChunk
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert EvidenceMatch to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
        }
