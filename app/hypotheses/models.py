"""Data models for candidate hypothesis generation in ATHENA (Milestone M5).

Defines structured representations for candidate scientific hypotheses, experimental variables,
falsification criteria, validation auditing, and complete candidate hypothesis sets.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from app.synthesis.models import EvidenceReference


@dataclass
class CandidateHypothesis:
    """A discrete, testable candidate scientific hypothesis addressing a research gap."""

    hypothesis_id: str
    target_gap_id: str
    title: str
    statement: str
    rationale: str
    proposed_mechanism: str
    independent_variables: list[str] = field(default_factory=list)
    dependent_variables: list[str] = field(default_factory=list)
    falsification_criteria: str = ""
    source_claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    valid_claim_ids: list[str] = field(default_factory=list)
    invalid_claim_ids: list[str] = field(default_factory=list)
    valid_evidence_ids: list[str] = field(default_factory=list)
    invalid_evidence_ids: list[str] = field(default_factory=list)
    valid_gap_id: bool = True
    is_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateHypothesis to a JSON-serializable dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "target_gap_id": self.target_gap_id,
            "title": self.title,
            "statement": self.statement,
            "rationale": self.rationale,
            "proposed_mechanism": self.proposed_mechanism,
            "independent_variables": list(self.independent_variables),
            "dependent_variables": list(self.dependent_variables),
            "falsification_criteria": self.falsification_criteria,
            "source_claim_ids": list(self.source_claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "valid_claim_ids": list(self.valid_claim_ids),
            "invalid_claim_ids": list(self.invalid_claim_ids),
            "valid_evidence_ids": list(self.valid_evidence_ids),
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
            "valid_gap_id": self.valid_gap_id,
            "is_grounded": self.is_grounded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateHypothesis":
        """Reconstruct CandidateHypothesis from a dictionary."""
        return cls(
            hypothesis_id=str(data.get("hypothesis_id") or ""),
            target_gap_id=str(data.get("target_gap_id") or ""),
            title=str(data.get("title") or ""),
            statement=str(data.get("statement") or ""),
            rationale=str(data.get("rationale") or ""),
            proposed_mechanism=str(data.get("proposed_mechanism") or ""),
            independent_variables=list(data.get("independent_variables") or []),
            dependent_variables=list(data.get("dependent_variables") or []),
            falsification_criteria=str(data.get("falsification_criteria") or ""),
            source_claim_ids=list(data.get("source_claim_ids") or []),
            evidence_ids=list(data.get("evidence_ids") or []),
            valid_claim_ids=list(data.get("valid_claim_ids") or []),
            invalid_claim_ids=list(data.get("invalid_claim_ids") or []),
            valid_evidence_ids=list(data.get("valid_evidence_ids") or []),
            invalid_evidence_ids=list(data.get("invalid_evidence_ids") or []),
            valid_gap_id=bool(data.get("valid_gap_id", True)),
            is_grounded=bool(data.get("is_grounded", True)),
        )


@dataclass
class HypothesisValidationReport:
    """Audit report evaluating the grounding and falsifiability of candidate hypotheses."""

    total_hypotheses: int
    grounded_hypotheses: int
    unsupported_hypotheses: int
    falsifiable_count: int
    total_citations: int
    valid_citations: int
    invalid_citations: int
    invalid_gap_ids: list[str]
    invalid_claim_ids: list[str]
    invalid_evidence_ids: list[str]
    grounding_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert HypothesisValidationReport to a JSON-serializable dictionary."""
        return {
            "total_hypotheses": self.total_hypotheses,
            "grounded_hypotheses": self.grounded_hypotheses,
            "unsupported_hypotheses": self.unsupported_hypotheses,
            "falsifiable_count": self.falsifiable_count,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "invalid_gap_ids": list(self.invalid_gap_ids),
            "invalid_claim_ids": list(self.invalid_claim_ids),
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
            "grounding_score": round(self.grounding_score, 4),
        }


@dataclass
class CandidateHypothesisSet:
    """Structured collection of candidate hypotheses with provenance and validation audit."""

    research_question: str
    hypotheses: list[CandidateHypothesis]
    validation_report: HypothesisValidationReport
    evidence_references: dict[str, EvidenceReference]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert CandidateHypothesisSet to a JSON-serializable dictionary."""
        return {
            "research_question": self.research_question,
            "total_hypotheses": len(self.hypotheses),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "validation_report": self.validation_report.to_dict(),
            "evidence_references": {k: v.to_dict() for k, v in self.evidence_references.items()},
            "model_name": self.model_name,
        }

    def to_markdown(self) -> str:
        """Render a clean, human-readable terminal/markdown hypothesis report."""
        divider = "=" * 76
        sub_divider = "-" * 76

        lines = [
            divider,
            f"ATHENA Candidate Hypothesis Generation (Milestone M5)",
            f"Research Question: {self.research_question}",
            f"Model: {self.model_name}",
            divider,
            "",
            "> [!WARNING]",
            "> CANDIDATE HYPOTHESES: These propositions are AI-generated scientific conjectures derived from literature gaps.",
            "> They do NOT constitute established scientific facts, validated theories, or empirical discoveries.",
            "> They require experimental verification and expert human peer review before any practical or laboratory implementation.",
            "",
            "### Formulated Candidate Hypotheses",
        ]

        if not self.hypotheses:
            lines.append("No candidate hypotheses formulated from the available research gaps.")
        else:
            for idx, hyp in enumerate(self.hypotheses, 1):
                claims_str = ", ".join(f"[{cid}]" for cid in hyp.source_claim_ids) if hyp.source_claim_ids else "None"
                ev_str = ", ".join(f"[{eid}]" for eid in hyp.evidence_ids) if hyp.evidence_ids else "None"
                status_str = "GROUNDED" if hyp.is_grounded else "UNSUPPORTED CITATION(S)"
                iv_str = ", ".join(hyp.independent_variables) if hyp.independent_variables else "Not specified"
                dv_str = ", ".join(hyp.dependent_variables) if hyp.dependent_variables else "Not specified"

                lines.extend([
                    f"{idx}. [{hyp.hypothesis_id}] {hyp.title}",
                    f"   Target Gap:  [{hyp.target_gap_id}]",
                    f"   Anchors:     Claims: {claims_str} | Evidence: {ev_str} (Status: {status_str})",
                    f"   Statement:   {hyp.statement}",
                    f"   Rationale:   {hyp.rationale}",
                    f"   Mechanism:   {hyp.proposed_mechanism}",
                    f"   Variables:   Independent: [{iv_str}] | Dependent: [{dv_str}]",
                    f"   Falsify If:  {hyp.falsification_criteria}",
                    "",
                ])

        lines.extend([
            sub_divider,
            "### Hypothesis Grounding & Falsifiability Audit",
            f"Grounding Score:           {self.validation_report.grounding_score * 100:.1f}%",
            f"Total Hypotheses:          {self.validation_report.total_hypotheses} "
            f"({self.validation_report.grounded_hypotheses} Grounded, {self.validation_report.unsupported_hypotheses} Unsupported)",
            f"Falsifiable Hypotheses:    {self.validation_report.falsifiable_count} / {self.validation_report.total_hypotheses}",
            f"Citations Checked:        {self.validation_report.valid_citations} Valid, "
            f"{self.validation_report.invalid_citations} Invalid",
        ])

        if self.validation_report.invalid_gap_ids:
            lines.append(f"Flagged Gap IDs:           {', '.join(self.validation_report.invalid_gap_ids)}")
        if self.validation_report.invalid_claim_ids:
            lines.append(f"Flagged Claim IDs:         {', '.join(self.validation_report.invalid_claim_ids)}")
        if self.validation_report.invalid_evidence_ids:
            lines.append(f"Flagged Evidence IDs:      {', '.join(self.validation_report.invalid_evidence_ids)}")

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
