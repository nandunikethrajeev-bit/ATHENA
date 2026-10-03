"""Data models for candidate research-gap analysis in ATHENA (Milestone M4).

Defines structured representations for scientific gaps, gap classification taxonomies,
grounding validation reports, and complete candidate research gap analyses.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.synthesis.models import EvidenceReference


class GapType(str, Enum):
    """Classification of scientific research gaps discovered across literature."""

    CONTRADICTION = "contradiction"          # Contradictory findings or unresolved disputes between studies
    METHODOLOGICAL = "methodological"        # Methodological limits (e.g. small sample sizes, in vitro only, lack of controls)
    COVERAGE_SCOPE = "coverage_scope"        # Unexplored materials, untested conditions, operational regimes, demographics
    UNVERIFIED_CLAIM = "unverified_claim"    # Hypothesized mechanisms or claims lacking empirical verification


@dataclass
class CandidateGap:
    """A discrete candidate research gap derived from synthesized evidence."""

    gap_id: str
    title: str
    description: str
    gap_type: GapType
    rationale: str
    source_claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    valid_evidence_ids: list[str] = field(default_factory=list)
    invalid_evidence_ids: list[str] = field(default_factory=list)
    valid_claim_ids: list[str] = field(default_factory=list)
    invalid_claim_ids: list[str] = field(default_factory=list)
    is_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateGap to a JSON-serializable dictionary."""
        return {
            "gap_id": self.gap_id,
            "title": self.title,
            "description": self.description,
            "gap_type": self.gap_type.value,
            "rationale": self.rationale,
            "source_claim_ids": list(self.source_claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "valid_evidence_ids": list(self.valid_evidence_ids),
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
            "valid_claim_ids": list(self.valid_claim_ids),
            "invalid_claim_ids": list(self.invalid_claim_ids),
            "is_grounded": self.is_grounded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateGap":
        """Reconstruct CandidateGap from dictionary."""
        raw_type = data.get("gap_type", "coverage_scope")
        gap_type = GapType(raw_type) if isinstance(raw_type, str) else raw_type

        return cls(
            gap_id=str(data.get("gap_id") or ""),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            gap_type=gap_type,
            rationale=str(data.get("rationale") or ""),
            source_claim_ids=list(data.get("source_claim_ids") or []),
            evidence_ids=list(data.get("evidence_ids") or []),
            valid_evidence_ids=list(data.get("valid_evidence_ids") or []),
            invalid_evidence_ids=list(data.get("invalid_evidence_ids") or []),
            valid_claim_ids=list(data.get("valid_claim_ids") or []),
            invalid_claim_ids=list(data.get("invalid_claim_ids") or []),
            is_grounded=bool(data.get("is_grounded", True)),
        )


@dataclass
class GapValidationReport:
    """Audit report evaluating the grounding integrity of candidate research gaps."""

    total_gaps: int
    grounded_gaps: int
    unsupported_gaps: int
    total_evidence_citations: int
    valid_evidence_citations: int
    invalid_evidence_citations: int
    invalid_evidence_ids: list[str]
    invalid_claim_ids: list[str]
    grounding_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert GapValidationReport to a JSON-serializable dictionary."""
        return {
            "total_gaps": self.total_gaps,
            "grounded_gaps": self.grounded_gaps,
            "unsupported_gaps": self.unsupported_gaps,
            "total_evidence_citations": self.total_evidence_citations,
            "valid_evidence_citations": self.valid_evidence_citations,
            "invalid_evidence_citations": self.invalid_evidence_citations,
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
            "invalid_claim_ids": list(self.invalid_claim_ids),
            "grounding_score": round(self.grounding_score, 4),
        }


@dataclass
class ResearchGapAnalysis:
    """Structured collection of candidate research gaps with complete provenance."""

    research_question: str
    gaps: list[CandidateGap]
    validation_report: GapValidationReport
    evidence_references: dict[str, EvidenceReference]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert ResearchGapAnalysis to a JSON-serializable dictionary."""
        return {
            "research_question": self.research_question,
            "total_gaps": len(self.gaps),
            "gaps": [g.to_dict() for g in self.gaps],
            "validation_report": self.validation_report.to_dict(),
            "evidence_references": {k: v.to_dict() for k, v in self.evidence_references.items()},
            "model_name": self.model_name,
        }

    def to_markdown(self) -> str:
        """Render a clean, human-readable terminal/markdown gap report."""
        divider = "=" * 76
        sub_divider = "-" * 76

        lines = [
            divider,
            f"ATHENA Candidate Research-Gap Analysis (Milestone M4)",
            f"Research Question: {self.research_question}",
            f"Model: {self.model_name}",
            divider,
            "",
            "> [!NOTE]",
            "> Candidate gaps indicate unexplored boundaries in the retrieved literature subset.",
            "> They do not constitute absolute proof of novelty or universal scientific absence.",
            "",
            "### Identified Candidate Gaps",
        ]

        if not self.gaps:
            lines.append("No candidate research gaps identified from the available evidence.")
        else:
            for idx, gap in enumerate(self.gaps, 1):
                claims_str = ", ".join(f"[{cid}]" for cid in gap.source_claim_ids) if gap.source_claim_ids else "None"
                ev_str = ", ".join(f"[{eid}]" for eid in gap.evidence_ids) if gap.evidence_ids else "None"
                status_str = "GROUNDED" if gap.is_grounded else "UNSUPPORTED CITATION(S)"

                lines.extend([
                    f"{idx}. [{gap.gap_id}] {gap.title}",
                    f"   Type:        {gap.gap_type.value.upper()}",
                    f"   Anchors:     Claims {claims_str} | Evidence: {ev_str} (Status: {status_str})",
                    f"   Description: {gap.description}",
                    f"   Rationale:   {gap.rationale}",
                    "",
                ])

        lines.extend([
            sub_divider,
            "### Gap Grounding & Audit",
            f"Grounding Score:           {self.validation_report.grounding_score * 100:.1f}%",
            f"Total Gaps:                {self.validation_report.total_gaps} "
            f"({self.validation_report.grounded_gaps} Grounded, {self.validation_report.unsupported_gaps} Unsupported)",
            f"Evidence Citations:        {self.validation_report.valid_evidence_citations} Valid, "
            f"{self.validation_report.invalid_evidence_citations} Invalid",
        ])

        if self.validation_report.invalid_evidence_ids:
            lines.append(f"Flagged Evidence IDs:      {', '.join(self.validation_report.invalid_evidence_ids)}")
        if self.validation_report.invalid_claim_ids:
            lines.append(f"Flagged Claim IDs:         {', '.join(self.validation_report.invalid_claim_ids)}")

        lines.extend(["", sub_divider, "### Evidence Sources Cited"])
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

        lines.append(divider)
        return "\n".join(lines)
