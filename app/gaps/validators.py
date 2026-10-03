"""Candidate research-gap grounding and citation validator for ATHENA (Milestone M4).

Cross-references candidate gaps against synthesized claim IDs and M2/M3 evidence chunks,
ensuring gaps are anchored to actual scientific findings rather than unconstrained hallucinations.
"""

from collections.abc import Sequence
from typing import Any

from app.evidence.models import EvidenceChunk
from app.gaps.models import CandidateGap, GapValidationReport, ResearchGapAnalysis
from app.synthesis.models import EvidenceReference, ResearchSynthesis


def validate_gap_analysis(
    analysis: ResearchGapAnalysis,
    synthesis: ResearchSynthesis,
    evidence_chunks: Sequence[EvidenceChunk] | None = None,
) -> tuple[GapValidationReport, dict[str, EvidenceReference]]:
    """Validate all candidate research gaps against synthesized claims and evidence.

    Args:
        analysis: The ResearchGapAnalysis instance to validate.
        synthesis: The source ResearchSynthesis from which gaps were derived.
        evidence_chunks: Optional sequence of valid EvidenceChunks.

    Returns:
        Tuple of (GapValidationReport, dict of evidence_id -> EvidenceReference).
    """
    # 1. Valid Claim IDs
    valid_claim_ids: set[str] = {c.claim_id for c in synthesis.claims}
    if not valid_claim_ids:
        valid_claim_ids = {f.claim_id for f in synthesis.key_findings} | {
            c.claim_id for c in synthesis.conflicting_findings
        }

    # 2. Valid Evidence IDs & Provenance Map
    valid_chunks_map: dict[str, EvidenceChunk] = {}
    if evidence_chunks:
        for chunk in evidence_chunks:
            valid_chunks_map[chunk.chunk_id] = chunk

    # Also include any pre-resolved evidence references from synthesis
    existing_references: dict[str, EvidenceReference] = dict(synthesis.evidence_references)

    total_citations = 0
    valid_citations = 0
    invalid_citations = 0

    global_invalid_eids: set[str] = set()
    global_invalid_cids: set[str] = set()

    grounded_gaps_count = 0
    unsupported_gaps_count = 0

    resolved_references: dict[str, EvidenceReference] = {}

    for gap in analysis.gaps:
        # Check source claims
        valid_cids: list[str] = []
        invalid_cids: list[str] = []
        for cid in gap.source_claim_ids:
            clean_cid = cid.strip()
            total_citations += 1
            if clean_cid in valid_claim_ids:
                valid_citations += 1
                valid_cids.append(clean_cid)
            else:
                invalid_citations += 1
                invalid_cids.append(clean_cid)
                global_invalid_cids.add(clean_cid)

        gap.valid_claim_ids = valid_cids
        gap.invalid_claim_ids = invalid_cids

        # Check evidence citations
        valid_eids: list[str] = []
        invalid_eids: list[str] = []
        for eid in gap.evidence_ids:
            clean_eid = eid.strip()
            total_citations += 1
            if clean_eid in valid_chunks_map:
                valid_citations += 1
                valid_eids.append(clean_eid)
                if clean_eid not in resolved_references:
                    chunk = valid_chunks_map[clean_eid]
                    resolved_references[clean_eid] = EvidenceReference.from_chunk(chunk)
            elif clean_eid in existing_references:
                valid_citations += 1
                valid_eids.append(clean_eid)
                if clean_eid not in resolved_references:
                    resolved_references[clean_eid] = existing_references[clean_eid]
            else:
                invalid_citations += 1
                invalid_eids.append(clean_eid)
                global_invalid_eids.add(clean_eid)

        gap.valid_evidence_ids = valid_eids
        gap.invalid_evidence_ids = invalid_eids

        # Grounding status
        has_anchor = bool(valid_cids or valid_eids)
        has_invalid = bool(invalid_cids or invalid_eids)

        if has_anchor and not has_invalid:
            gap.is_grounded = True
            grounded_gaps_count += 1
        else:
            gap.is_grounded = False
            unsupported_gaps_count += 1

    grounding_score = (
        valid_citations / max(1, total_citations)
        if total_citations > 0
        else (1.0 if not analysis.gaps else 0.0)
    )

    report = GapValidationReport(
        total_gaps=len(analysis.gaps),
        grounded_gaps=grounded_gaps_count,
        unsupported_gaps=unsupported_gaps_count,
        total_evidence_citations=total_citations,
        valid_evidence_citations=valid_citations,
        invalid_evidence_citations=invalid_citations,
        invalid_evidence_ids=sorted(list(global_invalid_eids)),
        invalid_claim_ids=sorted(list(global_invalid_cids)),
        grounding_score=grounding_score,
    )

    analysis.validation_report = report
    analysis.evidence_references = resolved_references

    return report, resolved_references
