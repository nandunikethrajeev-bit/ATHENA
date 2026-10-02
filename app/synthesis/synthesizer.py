"""Synthesizer orchestrator for ATHENA scientific evidence synthesis (Milestone M3).

Coordinates the M2 -> M3 pipeline:
EvidenceContext -> Strict Prompt -> LLM Client -> Safe Parsing -> Validation -> ResearchSynthesis
"""

import json
import re
from typing import Any, Sequence

from app.evidence.models import EvidenceChunk, EvidenceMatch
from app.synthesis.models import (
    ClaimStatus,
    EvidenceContext,
    ResearchSynthesis,
    SynthesizedClaim,
    ValidationReport,
)
from app.synthesis.prompts import SYSTEM_PROMPT, build_synthesis_prompt
from app.synthesis.providers import LLMClient, OpenAILikeClient, get_llm_client
from app.synthesis.validators import validate_synthesis


class SynthesisParsingError(Exception):
    """Raised when an LLM response cannot be parsed into the required synthesis schema."""


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Safely extract and parse JSON from model output, handling markdown fences and extraneous whitespace.

    Args:
        raw_text: Raw output string returned by the LLM.

    Returns:
        Parsed JSON dictionary.

    Raises:
        SynthesisParsingError: If valid JSON cannot be found or parsed.
    """
    text = raw_text.strip()
    if not text:
        raise SynthesisParsingError("LLM response was empty.")

    # Strip markdown code fences if present (e.g. ```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, text)
    if fence_match:
        text = fence_match.group(1).strip()

    # If surrounding conversational chatter exists, locate outermost JSON braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except Exception as exc:
        snippet = raw_text[:200] + ("..." if len(raw_text) > 200 else "")
        raise SynthesisParsingError(
            f"Failed to parse LLM response as JSON: {exc}. Response snippet: {snippet!r}"
        ) from exc

    if not isinstance(data, dict):
        raise SynthesisParsingError(f"Expected a JSON object from LLM, got {type(data).__name__}.")

    return data


class Synthesizer:
    """Orchestrates evidence-grounded scientific synthesis."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialize Synthesizer with an optional LLMClient.

        If client is None, get_llm_client() will be called during synthesize().
        """
        self.client = client

    def _extract_chunks(self, evidence_context: Any) -> list[EvidenceChunk]:
        """Extract a flat list of EvidenceChunk instances from various context representations."""
        if evidence_context is None:
            return []

        if isinstance(evidence_context, EvidenceContext):
            return list(evidence_context.chunks)

        if isinstance(evidence_context, Sequence):
            chunks: list[EvidenceChunk] = []
            for item in evidence_context:
                if isinstance(item, EvidenceChunk):
                    chunks.append(item)
                elif isinstance(item, EvidenceMatch):
                    chunks.append(item.chunk)
            return chunks

        return []

    def synthesize(
        self,
        research_question: str,
        evidence_context: Any,
        client: LLMClient | None = None,
    ) -> ResearchSynthesis:
        """Execute structured evidence synthesis on ranked retrieval items.

        Args:
            research_question: User's scientific research question.
            evidence_context: EvidenceContext, Sequence of EvidenceMatch, or Sequence of EvidenceChunk.
            client: Optional LLMClient to override the instance client.

        Returns:
            Validated ResearchSynthesis instance.
        """
        chunks = self._extract_chunks(evidence_context)

        # 1. Edge case: Zero evidence retrieved -> return structured report without calling LLM
        if not chunks:
            empty_report = ValidationReport(
                total_claims=0,
                supported_claims=0,
                partially_valid_claims=0,
                unsupported_claims=0,
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                grounding_score=1.0,
                invalid_evidence_ids=[],
            )
            return ResearchSynthesis(
                research_question=research_question,
                overview="Insufficient evidence: no relevant scientific evidence chunks were retrieved to synthesize findings for this research question.",
                key_findings=[],
                conflicting_findings=[],
                limitations=["No matching evidence retrieved from literature index."],
                claims=[],
                validation_report=empty_report,
                model_name="none (insufficient evidence)",
                evidence_references={},
            )

        # 2. Resolve active LLM client
        active_client = client or self.client
        if active_client is None:
            active_client = get_llm_client()

        model_name = getattr(active_client, "model", None)
        if not model_name:
            model_name = "mock" if active_client.__class__.__name__ == "MockLLMClient" else active_client.__class__.__name__

        # 3. Build deterministic prompt
        prompt = build_synthesis_prompt(research_question, chunks)

        # 4. Invoke LLM
        raw_response = active_client.complete(prompt, system_prompt=SYSTEM_PROMPT)

        # 5. Parse JSON payload
        data = extract_json_payload(raw_response)

        overview = str(data.get("overview") or "").strip()
        raw_key_findings = data.get("key_findings") or []
        raw_conflicting_findings = data.get("conflicting_findings") or []
        raw_limitations = data.get("limitations") or []

        key_findings: list[SynthesizedClaim] = []
        for idx, item in enumerate(raw_key_findings, 1):
            if isinstance(item, dict):
                cid = str(item.get("claim_id") or f"KF{idx}")
                text = str(item.get("text") or "").strip()
                eids = [str(e).strip() for e in item.get("evidence_ids") or []]
                key_findings.append(SynthesizedClaim(claim_id=cid, text=text, evidence_ids=eids))

        conflicting_findings: list[SynthesizedClaim] = []
        for idx, item in enumerate(raw_conflicting_findings, 1):
            if isinstance(item, dict):
                cid = str(item.get("claim_id") or f"CF{idx}")
                text = str(item.get("text") or "").strip()
                eids = [str(e).strip() for e in item.get("evidence_ids") or []]
                conflicting_findings.append(SynthesizedClaim(claim_id=cid, text=text, evidence_ids=eids))

        limitations: list[str] = []
        for lim in raw_limitations:
            if isinstance(lim, str) and lim.strip():
                limitations.append(lim.strip())

        all_claims = list(key_findings) + list(conflicting_findings)

        dummy_report = ValidationReport(
            total_claims=len(all_claims),
            supported_claims=0,
            partially_valid_claims=0,
            unsupported_claims=0,
            total_citations=0,
            valid_citations=0,
            invalid_citations=0,
            grounding_score=0.0,
            invalid_evidence_ids=[],
        )

        synthesis = ResearchSynthesis(
            research_question=research_question,
            overview=overview,
            key_findings=key_findings,
            conflicting_findings=conflicting_findings,
            limitations=limitations,
            claims=all_claims,
            validation_report=dummy_report,
            model_name=model_name,
            evidence_references={},
        )

        # 6. Validate claims and resolve provenance
        validate_synthesis(synthesis, chunks)

        return synthesis
