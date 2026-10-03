"""Candidate hypothesis generator orchestrator for ATHENA (Milestone M5).

Coordinates:
ResearchGapAnalysis + ResearchSynthesis + EvidenceChunks
  -> Deterministic Hypothesis Prompt -> LLMClient -> Safe JSON Parsing -> Validation -> CandidateHypothesisSet
"""

from collections.abc import Sequence
import json
from typing import Any

from app.evidence.models import EvidenceChunk
from app.gaps.models import ResearchGapAnalysis
from app.hypotheses.models import (
    CandidateHypothesis,
    CandidateHypothesisSet,
    HypothesisValidationReport,
)
from app.hypotheses.prompts import HYPOTHESIS_SYSTEM_PROMPT, build_hypothesis_prompt
from app.hypotheses.validators import validate_hypothesis_set
from app.synthesis.models import ResearchSynthesis
from app.synthesis.providers import LLMClient, get_llm_client
from app.synthesis.synthesizer import extract_json_payload


class HypothesisGenerationError(Exception):
    """Raised when candidate hypothesis generation fails."""


class HypothesisGenerator:
    """Orchestrator for formulating candidate hypotheses from candidate research gaps."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialize HypothesisGenerator with an optional LLMClient.

        If client is None, get_llm_client() is called during generate().
        """
        self.client = client

    def generate(
        self,
        gap_analysis: ResearchGapAnalysis,
        synthesis: ResearchSynthesis,
        evidence_chunks: Sequence[EvidenceChunk] | None = None,
        client: LLMClient | None = None,
    ) -> CandidateHypothesisSet:
        """Formulate testable candidate hypotheses addressing identified research gaps.

        Args:
            gap_analysis: Validated ResearchGapAnalysis instance from M4.
            synthesis: Validated ResearchSynthesis instance from M3.
            evidence_chunks: Optional sequence of supporting EvidenceChunks.
            client: Optional LLMClient override.

        Returns:
            Validated CandidateHypothesisSet instance.
        """
        # Edge case: If there are no candidate research gaps, exit early without calling LLM
        if not gap_analysis.gaps:
            empty_report = HypothesisValidationReport(
                total_hypotheses=0,
                grounded_hypotheses=0,
                unsupported_hypotheses=0,
                falsifiable_count=0,
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                invalid_gap_ids=[],
                invalid_claim_ids=[],
                invalid_evidence_ids=[],
                grounding_score=1.0,
            )
            return CandidateHypothesisSet(
                research_question=synthesis.research_question,
                hypotheses=[],
                validation_report=empty_report,
                evidence_references={},
                model_name="none (no gaps to address)",
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
        prompt = build_hypothesis_prompt(
            gap_analysis=gap_analysis,
            synthesis=synthesis,
            evidence_chunks=evidence_chunks,
        )

        # Invoke model
        raw_response = active_client.complete(prompt, system_prompt=HYPOTHESIS_SYSTEM_PROMPT)

        # Extract and parse JSON
        data = extract_json_payload(raw_response)

        raw_hypotheses = data.get("hypotheses") or []
        candidate_hypotheses: list[CandidateHypothesis] = []

        for idx, item in enumerate(raw_hypotheses, 1):
            if isinstance(item, dict):
                hid = str(item.get("hypothesis_id") or f"H{idx}")
                target_gid = str(item.get("target_gap_id") or "G1").strip()
                title = str(item.get("title") or f"Candidate Hypothesis {idx}").strip()
                statement = str(item.get("statement") or "").strip()
                rationale = str(item.get("rationale") or "").strip()
                mechanism = str(item.get("proposed_mechanism") or "").strip()

                iv = [str(x).strip() for x in (item.get("independent_variables") or []) if str(x).strip()]
                dv = [str(y).strip() for y in (item.get("dependent_variables") or []) if str(y).strip()]
                falsify = str(item.get("falsification_criteria") or "").strip()

                source_claims = [str(c).strip() for c in (item.get("source_claim_ids") or []) if str(c).strip()]
                evidence_ids = [str(e).strip() for e in (item.get("evidence_ids") or []) if str(e).strip()]

                candidate_hypotheses.append(
                    CandidateHypothesis(
                        hypothesis_id=hid,
                        target_gap_id=target_gid,
                        title=title,
                        statement=statement,
                        rationale=rationale,
                        proposed_mechanism=mechanism,
                        independent_variables=iv,
                        dependent_variables=dv,
                        falsification_criteria=falsify,
                        source_claim_ids=source_claims,
                        evidence_ids=evidence_ids,
                    )
                )

        dummy_report = HypothesisValidationReport(
            total_hypotheses=len(candidate_hypotheses),
            grounded_hypotheses=0,
            unsupported_hypotheses=0,
            falsifiable_count=0,
            total_citations=0,
            valid_citations=0,
            invalid_citations=0,
            invalid_gap_ids=[],
            invalid_claim_ids=[],
            invalid_evidence_ids=[],
            grounding_score=0.0,
        )

        hypothesis_set = CandidateHypothesisSet(
            research_question=synthesis.research_question,
            hypotheses=candidate_hypotheses,
            validation_report=dummy_report,
            evidence_references={},
            model_name=model_name,
        )

        # Validate against gaps, claims, and evidence
        validate_hypothesis_set(
            hypothesis_set=hypothesis_set,
            gap_analysis=gap_analysis,
            synthesis=synthesis,
            evidence_chunks=evidence_chunks,
        )

        return hypothesis_set
