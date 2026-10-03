"""Candidate hypothesis grounding, citation, and falsifiability validator for ATHENA (Milestone M5).

Cross-references candidate hypotheses against candidate research gaps (M4), synthesized claims (M3),
and primary evidence chunks (M2/M3), ensuring hypotheses are grounded and testable.
"""

from collections.abc import Sequence
from typing import Any

from app.evidence.models import EvidenceChunk
from app.gaps.models import ResearchGapAnalysis
from app.hypotheses.models import (
    CandidateHypothesis,
    CandidateHypothesisSet,
    HypothesisValidationReport,
)
from app.synthesis.models import EvidenceReference, ResearchSynthesis


def validate_hypothesis_set(
    hypothesis_set: CandidateHypothesisSet,
    gap_analysis: ResearchGapAnalysis,
    synthesis: ResearchSynthesis,
    evidence_chunks: Sequence[EvidenceChunk] | None = None,
) -> tuple[HypothesisValidationReport, dict[str, EvidenceReference]]:
    """Validate all candidate hypotheses against candidate gaps, synthesized claims, and evidence.

    Args:
        hypothesis_set: The CandidateHypothesisSet instance to validate.
        gap_analysis: The source ResearchGapAnalysis containing candidate gaps.
        synthesis: The source ResearchSynthesis from which claims were derived.
        evidence_chunks: Optional sequence of valid EvidenceChunks.

    Returns:
        Tuple of (HypothesisValidationReport, dict of evidence_id -> EvidenceReference).
    """
    # 1. Valid Gap IDs
    valid_gap_ids: set[str] = {g.gap_id for g in gap_analysis.gaps}

    # 2. Valid Claim IDs
    valid_claim_ids: set[str] = {c.claim_id for c in synthesis.claims}
    if not valid_claim_ids:
        valid_claim_ids = {f.claim_id for f in synthesis.key_findings} | {
            c.claim_id for c in synthesis.conflicting_findings
        }

    # 3. Valid Evidence IDs & Provenance Map
    valid_chunks_map: dict[str, EvidenceChunk] = {}
    if evidence_chunks:
        for chunk in evidence_chunks:
            valid_chunks_map[chunk.chunk_id] = chunk

    existing_references: dict[str, EvidenceReference] = dict(synthesis.evidence_references)

    total_citations = 0
    valid_citations = 0
    invalid_citations = 0

    global_invalid_gids: set[str] = set()
    global_invalid_cids: set[str] = set()
    global_invalid_eids: set[str] = set()

    grounded_count = 0
    unsupported_count = 0
    falsifiable_count = 0

    resolved_references: dict[str, EvidenceReference] = {}

    for hyp in hypothesis_set.hypotheses:
        # Check target gap ID
        clean_gid = hyp.target_gap_id.strip()
        total_citations += 1
        if clean_gid and clean_gid in valid_gap_ids:
            valid_citations += 1
            hyp.valid_gap_id = True
        else:
            invalid_citations += 1
            hyp.valid_gap_id = False
            if clean_gid:
                global_invalid_gids.add(clean_gid)
            else:
                global_invalid_gids.add("MISSING_GAP_ID")

        # Check source claims
        valid_cids: list[str] = []
        invalid_cids: list[str] = []
        for cid in hyp.source_claim_ids:
            clean_cid = cid.strip()
            total_citations += 1
            if clean_cid in valid_claim_ids:
                valid_citations += 1
                valid_cids.append(clean_cid)
            else:
                invalid_citations += 1
                invalid_cids.append(clean_cid)
                global_invalid_cids.add(clean_cid)

        hyp.valid_claim_ids = valid_cids
        hyp.invalid_claim_ids = invalid_cids

        # Check evidence citations
        valid_eids: list[str] = []
        invalid_eids: list[str] = []
        for eid in hyp.evidence_ids:
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

        hyp.valid_evidence_ids = valid_eids
        hyp.invalid_evidence_ids = invalid_eids

        # Check falsifiability criteria
        falsification_text = hyp.falsification_criteria.strip() if hyp.falsification_criteria else ""
        if len(falsification_text) >= 10:
            falsifiable_count += 1

        # Grounding status
        has_anchor = bool(hyp.valid_gap_id and (valid_cids or valid_eids))
        has_invalid = bool(not hyp.valid_gap_id or invalid_cids or invalid_eids)

        if has_anchor and not has_invalid:
            hyp.is_grounded = True
            grounded_count += 1
        else:
            hyp.is_grounded = False
            unsupported_count += 1

    grounding_score = (
        valid_citations / max(1, total_citations)
        if total_citations > 0
        else (1.0 if not hypothesis_set.hypotheses else 0.0)
    )

    report = HypothesisValidationReport(
        total_hypotheses=len(hypothesis_set.hypotheses),
        grounded_hypotheses=grounded_count,
        unsupported_hypotheses=unsupported_count,
        falsifiable_count=falsifiable_count,
        total_citations=total_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        invalid_gap_ids=sorted(list(global_invalid_gids)),
        invalid_claim_ids=sorted(list(global_invalid_cids)),
        invalid_evidence_ids=sorted(list(global_invalid_eids)),
        grounding_score=grounding_score,
    )

    hypothesis_set.validation_report = report
    hypothesis_set.evidence_references = resolved_references

    return report, resolved_references
