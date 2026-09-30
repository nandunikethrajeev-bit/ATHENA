"""Data models for scientific literature retrieval in ATHENA."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpenAccessInfo:
    """Information regarding open access status of a scientific paper."""

    is_oa: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert OpenAccessInfo to a dictionary."""
        return asdict(self)


@dataclass
class Paper:
    """Internal canonical representation of a scientific paper retrieved by ATHENA.

    Captures bibliographical metadata, access information, and reconstructed abstract
    while preserving provenance and source identity back to the original scholarly record.
    Missing attributes are represented as None rather than fabricated.
    """

    title: str | None
    openalex_id: str
    source: str = "OpenAlex"
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    venue: str | None = None
    cited_by_count: int | None = None
    publication_type: str | None = None
    landing_page_url: str | None = None
    open_access: OpenAccessInfo | None = None
    raw_id: str | None = None

    @property
    def short_id(self) -> str:
        """Extract short identifier from full OpenAlex URI (e.g. 'W12345' from 'https://openalex.org/W12345')."""
        if self.openalex_id and "/" in self.openalex_id:
            return self.openalex_id.rstrip("/").split("/")[-1]
        return self.openalex_id or ""

    def to_dict(self) -> dict[str, Any]:
        """Convert Paper instance to a JSON-serializable dictionary."""
        data = asdict(self)
        if self.open_access is not None:
            data["open_access"] = self.open_access.to_dict()
        return data
