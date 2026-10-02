"""Data models for scientific evidence synthesis in ATHENA (Milestone M3).

All synthesis models strictly isolate retrieved scientific evidence from
model-generated syntheses and claims, preserving complete provenance
back to originating scientific publications.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.evidence.models import EvidenceChunk, EvidenceMatch


class ClaimStatus(str, Enum):
    """Integrity status of a synthesized scientific claim."""

    SUPPORTED = "supported"              # All cited evidence IDs exist in retrieved evidence
    PARTIALLY_VALID = "partially_valid"  # Some cited evidence IDs exist, some are invalid
    UNSUPPORTED = "unsupported"          # No valid evidence IDs cited (or zero evidence cited)


@dataclass(frozen=True)
class EvidenceReference:
    """Traceable bibliographic citation record for an evidence chunk cited in synthesis.

    Directly links an evidence identifier (e.g. 'W12345-abs-1') to its full scholarly metadata
    without fabricating any missing fields.
    """

    evidence_id: str
    paper_title: str
    authors: list[str]
    publication_year: int | None
    venue: str | None
    doi: str | None
    openalex_id: str
    source_url: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert EvidenceReference to a JSON-serializable dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "paper_title": self.paper_title,
            "authors": list(self.authors),
            "publication_year": self.publication_year,
            "venue": self.venue,
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "source_url": self.source_url,
        }

    @classmethod
    def from_chunk(cls, chunk: EvidenceChunk) -> "EvidenceReference":
        """Construct an EvidenceReference from an existing M2 EvidenceChunk."""
        src = chunk.source
        return cls(
            evidence_id=chunk.chunk_id,
            paper_title=src.paper_title or "Untitled",
            authors=list(src.authors),
            publication_year=src.publication_year,
            venue=src.venue,
            doi=src.doi,
            openalex_id=src.openalex_id,
            source_url=src.landing_page_url or src.doi,
        )


@dataclass
class SynthesizedClaim:
    """A discrete scientific finding or statement traceable to evidence chunk IDs."""

    claim_id: str
    text: str
    evidence_ids: list[str]
    status: ClaimStatus = ClaimStatus.SUPPORTED
    valid_evidence_ids: list[str] = field(default_factory=list)
    invalid_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert SynthesizedClaim to a JSON-serializable dictionary."""
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "status": self.status.value,
            "valid_evidence_ids": list(self.valid_evidence_ids),
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
        }


@dataclass
class ValidationReport:
    """Audit report evaluating the grounding integrity of the synthesis."""

    total_claims: int
    supported_claims: int
    partially_valid_claims: int
    unsupported_claims: int
    total_citations: int
    valid_citations: int
    invalid_citations: int
    grounding_score: float
    invalid_evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert ValidationReport to a JSON-serializable dictionary."""
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "partially_valid_claims": self.partially_valid_claims,
            "unsupported_claims": self.unsupported_claims,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "grounding_score": round(self.grounding_score, 4),
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
        }


@dataclass
class EvidenceContext:
    """Container holding ranked evidence matches and metadata for synthesis."""

    query: str
    matches: list[EvidenceMatch] = field(default_factory=list)
    total_papers: int | None = None
    total_chunks: int | None = None

    @property
    def chunks(self) -> list[EvidenceChunk]:
        """Return the list of EvidenceChunks from the matches."""
        return [m.chunk for m in self.matches]

    def to_dict(self) -> dict[str, Any]:
        """Convert EvidenceContext to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "total_papers": self.total_papers,
            "total_chunks": self.total_chunks,
            "matches": [m.to_dict() for m in self.matches],
        }


@dataclass
class ResearchSynthesis:
    """Structured, evidence-grounded scientific synthesis produced in Milestone M3."""

    research_question: str
    overview: str
    key_findings: list[SynthesizedClaim]
    conflicting_findings: list[SynthesizedClaim]
    limitations: list[str]
    claims: list[SynthesizedClaim]
    validation_report: ValidationReport
    model_name: str
    evidence_references: dict[str, EvidenceReference] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ResearchSynthesis to a JSON-serializable dictionary."""
        return {
            "research_question": self.research_question,
            "overview": self.overview,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "conflicting_findings": [f.to_dict() for f in self.conflicting_findings],
            "limitations": list(self.limitations),
            "claims": [c.to_dict() for c in self.claims],
            "validation_report": self.validation_report.to_dict(),
            "model_name": self.model_name,
            "evidence_references": {k: v.to_dict() for k, v in self.evidence_references.items()},
        }

    def to_markdown(self) -> str:
        """Render a structured, clean terminal- and publication-readable report."""
        divider = "=" * 76
        sub_divider = "-" * 76

        lines = [
            divider,
            f"ATHENA Scientific Evidence Synthesis (Milestone M3)",
            f"Research Question: {self.research_question}",
            f"Model: {self.model_name}",
            divider,
            "",
            "### Overview",
            self.overview.strip() if self.overview else "No overview provided.",
            "",
            "### Key Findings",
        ]

        if not self.key_findings:
            lines.append("No supported key findings generated.")
        else:
            for idx, finding in enumerate(self.key_findings, 1):
                ev_str = ", ".join(f"[{eid}]" for eid in finding.evidence_ids) if finding.evidence_ids else "None"
                lines.append(f"{idx}. [{finding.claim_id}] {finding.text}")
                lines.append(f"   Evidence: {ev_str} (Status: {finding.status.value.upper()})")

        lines.extend(["", "### Conflicting Findings"])
        if not self.conflicting_findings:
            lines.append("No explicit conflicting findings identified across the retrieved evidence.")
        else:
            for idx, conflict in enumerate(self.conflicting_findings, 1):
                ev_str = ", ".join(f"[{eid}]" for eid in conflict.evidence_ids) if conflict.evidence_ids else "None"
                lines.append(f"{idx}. [{conflict.claim_id}] {conflict.text}")
                lines.append(f"   Evidence: {ev_str} (Status: {conflict.status.value.upper()})")

        lines.extend(["", "### Limitations"])
        if not self.limitations:
            lines.append("No explicit limitations reported.")
        else:
            for idx, lim in enumerate(self.limitations, 1):
                lines.append(f"- {lim}")

        lines.extend([
            "",
            sub_divider,
            "### Grounding & Validation Audit",
            f"Grounding Score:           {self.validation_report.grounding_score * 100:.1f}%",
            f"Total Claims:              {self.validation_report.total_claims} "
            f"({self.validation_report.supported_claims} Supported, "
            f"{self.validation_report.partially_valid_claims} Partially Valid, "
            f"{self.validation_report.unsupported_claims} Unsupported)",
            f"Citations Checked:         {self.validation_report.total_citations} "
            f"({self.validation_report.valid_citations} Valid, {self.validation_report.invalid_citations} Invalid)",
        ])

        if self.validation_report.invalid_evidence_ids:
            lines.append(f"Flagged Invalid IDs:       {', '.join(self.validation_report.invalid_evidence_ids)}")

        lines.extend(["", sub_divider, "### Evidence Sources"])
        if not self.evidence_references:
            lines.append("No evidence sources cited.")
        else:
            for eid, ref in sorted(self.evidence_references.items()):
                authors_str = ", ".join(ref.authors[:3]) + (f" et al. ({len(ref.authors)} total)" if len(ref.authors) > 3 else "")
                year_str = str(ref.publication_year) if ref.publication_year is not None else "N/A"
                venue_str = f" | {ref.venue}" if ref.venue else ""
                doi_str = f" | DOI: {ref.doi}" if ref.doi else ""
                lines.append(f"[{eid}] {ref.paper_title}")
                lines.append(f"      Authors: {authors_str or 'Unknown'} ({year_str}{venue_str}){doi_str}")
                lines.append(f"      OpenAlex ID: {ref.openalex_id}")

        lines.append(divider)
        return "\n".join(lines)
