"""Candidate research-gap analyzer orchestrator for ATHENA (Milestone M4).

Coordinates:
ResearchSynthesis + EvidenceChunks -> Strict Gap Prompt -> LLMClient -> Safe Parsing -> Validation -> ResearchGapAnalysis
"""

from collections.abc import Sequence
import json
from typing import Any

from app.evidence.models import EvidenceChunk
from app.gaps.models import (
    CandidateGap,
    GapType,
    GapValidationReport,
    ResearchGapAnalysis,
)
from app.gaps.prompts import GAP_SYSTEM_PROMPT, build_gap_analysis_prompt
from app.gaps.validators import validate_gap_analysis
from app.synthesis.providers import LLMClient, get_llm_client
from app.synthesis.synthesizer import SynthesisParsingError, extract_json_payload


class GapAnalysisError(Exception):
    """Raised when candidate research-gap analysis fails."""


class GapAnalyzer:
    """Orchestrator for discovering candidate research gaps from synthesized literature."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialize GapAnalyzer with an optional LLMClient.

        If client is None, get_llm_client() is called during analyze().
        """
        self.client = client

    def analyze(
        self,
        synthesis: Any,
        evidence_chunks: Sequence[EvidenceChunk] | None = None,
        client: LLMClient | None = None,
        filter_type: str | GapType | None = None,
    ) -> ResearchGapAnalysis:
        """Analyze synthesized evidence to discover structured candidate research gaps.

        Args:
            synthesis: Validated ResearchSynthesis instance from M3.
            evidence_chunks: Optional sequence of supporting EvidenceChunks.
            client: Optional LLMClient override.
            filter_type: Optional GapType or str to filter results by gap taxonomy.

        Returns:
            Validated ResearchGapAnalysis instance.
        """
        # Edge case: If synthesis has no claims, return empty analysis without calling LLM
        has_claims = bool(synthesis.key_findings or synthesis.conflicting_findings or synthesis.claims)
        if not has_claims:
            empty_report = GapValidationReport(
                total_gaps=0,
                grounded_gaps=0,
                unsupported_gaps=0,
                total_evidence_citations=0,
                valid_evidence_citations=0,
                invalid_evidence_citations=0,
                invalid_evidence_ids=[],
                invalid_claim_ids=[],
                grounding_score=1.0,
            )
            return ResearchGapAnalysis(
                research_question=synthesis.research_question,
                gaps=[],
                validation_report=empty_report,
                evidence_references={},
                model_name="none (no claims to analyze)",
            )

        # Resolve LLM client
        active_client = client or self.client
        if active_client is None:
            active_client = get_llm_client()

        model_name = getattr(active_client, "model", None)
        if not model_name:
            model_name = (
                "mock"
                if active_client.__class__.__name__ == "MockLLMClient"
                else active_client.__class__.__name__
            )

        # Build deterministic prompt
        prompt = build_gap_analysis_prompt(synthesis=synthesis, evidence_chunks=evidence_chunks)

        # Invoke model
        raw_response = active_client.complete(prompt, system_prompt=GAP_SYSTEM_PROMPT)

        # Extract and parse JSON
        data = extract_json_payload(raw_response)

        raw_gaps = data.get("gaps") or []
        candidate_gaps: list[CandidateGap] = []

        for idx, item in enumerate(raw_gaps, 1):
            if isinstance(item, dict):
                gid = str(item.get("gap_id") or f"G{idx}")
                title = str(item.get("title") or f"Candidate Gap {idx}").strip()
                desc = str(item.get("description") or "").strip()
                raw_type = str(item.get("gap_type") or "coverage_scope").strip().lower()

                # Validate or default gap_type
                try:
                    gap_type = GapType(raw_type)
                except ValueError:
                    gap_type = GapType.COVERAGE_SCOPE

                rationale = str(item.get("rationale") or "").strip()
                source_claims = [str(c).strip() for c in item.get("source_claim_ids") or []]
                evidence_ids = [str(e).strip() for e in item.get("evidence_ids") or []]

                candidate_gaps.append(
                    CandidateGap(
                        gap_id=gid,
                        title=title,
                        description=desc,
                        gap_type=gap_type,
                        rationale=rationale,
                        source_claim_ids=source_claims,
                        evidence_ids=evidence_ids,
                    )
                )

        if filter_type is not None:
            target_type = filter_type.value if isinstance(filter_type, GapType) else str(filter_type).strip().lower()
            candidate_gaps = [g for g in candidate_gaps if g.gap_type.value == target_type]

        dummy_report = GapValidationReport(
            total_gaps=len(candidate_gaps),
            grounded_gaps=0,
            unsupported_gaps=0,
            total_evidence_citations=0,
            valid_evidence_citations=0,
            invalid_evidence_citations=0,
            invalid_evidence_ids=[],
            invalid_claim_ids=[],
            grounding_score=0.0,
        )

        analysis = ResearchGapAnalysis(
            research_question=synthesis.research_question,
            gaps=candidate_gaps,
            validation_report=dummy_report,
            evidence_references={},
            model_name=model_name,
        )

        # Validate claims and citations
        validate_gap_analysis(analysis, synthesis, evidence_chunks=evidence_chunks)

        return analysis
