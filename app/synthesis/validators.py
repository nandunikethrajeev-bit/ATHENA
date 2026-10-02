"""Scientific claim and evidence grounding validation layer for ATHENA (Milestone M3).

This is a critical safety and verification boundary:
- Never trusts LLM citations at face value.
- Cross-references every cited evidence ID against genuine M2 EvidenceChunks.
- Flags phantom / hallucinated citations.
- Determines whether scientific claims are fully supported, partially valid, or unsupported.
- Resolves immutable provenance records for all valid evidence citations.
"""

from typing import Sequence

from app.evidence.models import EvidenceChunk
from app.synthesis.models import (
    ClaimStatus,
    EvidenceReference,
    ResearchSynthesis,
    SynthesizedClaim,
    ValidationReport,
)


def validate_synthesis(
    synthesis: ResearchSynthesis,
    evidence_chunks: Sequence[EvidenceChunk],
) -> tuple[ValidationReport, dict[str, EvidenceReference]]:
    """Validate all claims and citations in a ResearchSynthesis against provided EvidenceChunks.

    Args:
        synthesis: The ResearchSynthesis instance to validate.
        evidence_chunks: Sequence of valid M2 EvidenceChunks supplied to the LLM.

    Returns:
        A tuple of (ValidationReport, dict of evidence_id -> EvidenceReference).
    """
    valid_id_map: dict[str, EvidenceChunk] = {chunk.chunk_id: chunk for chunk in evidence_chunks}
    valid_ids_set = set(valid_id_map.keys())

    all_claims: list[SynthesizedClaim] = list(synthesis.claims)
    if not all_claims:
        # If claims list is empty, aggregate key_findings and conflicting_findings
        all_claims = list(synthesis.key_findings) + list(synthesis.conflicting_findings)
        synthesis.claims = all_claims

    total_citations = 0
    valid_citations = 0
    invalid_citations = 0
    global_invalid_ids: set[str] = set()

    supported_count = 0
    partially_valid_count = 0
    unsupported_count = 0

    referenced_sources: dict[str, EvidenceReference] = {}

    for claim in all_claims:
        cited_ids = list(claim.evidence_ids)
        claim_valid_ids: list[str] = []
        claim_invalid_ids: list[str] = []

        for cid in cited_ids:
            clean_id = cid.strip()
            total_citations += 1
            if clean_id in valid_ids_set:
                valid_citations += 1
                claim_valid_ids.append(clean_id)
                # Resolve provenance
                if clean_id not in referenced_sources:
                    chunk = valid_id_map[clean_id]
                    referenced_sources[clean_id] = EvidenceReference.from_chunk(chunk)
            else:
                invalid_citations += 1
                claim_invalid_ids.append(clean_id)
                global_invalid_ids.add(clean_id)

        claim.valid_evidence_ids = claim_valid_ids
        claim.invalid_evidence_ids = claim_invalid_ids

        # Evaluate claim status
        if not cited_ids:
            # Claim made with zero evidence citations
            claim.status = ClaimStatus.UNSUPPORTED
            unsupported_count += 1
        elif not claim_valid_ids:
            # All cited IDs were invalid / hallucinated
            claim.status = ClaimStatus.UNSUPPORTED
            unsupported_count += 1
        elif claim_invalid_ids:
            # Mixed valid and invalid citations
            claim.status = ClaimStatus.PARTIALLY_VALID
            partially_valid_count += 1
        else:
            # All cited IDs are valid
            claim.status = ClaimStatus.SUPPORTED
            supported_count += 1

    grounding_score = (
        valid_citations / max(1, total_citations)
        if total_citations > 0
        else (1.0 if not all_claims else 0.0)
    )

    report = ValidationReport(
        total_claims=len(all_claims),
        supported_claims=supported_count,
        partially_valid_claims=partially_valid_count,
        unsupported_claims=unsupported_count,
        total_citations=total_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        grounding_score=grounding_score,
        invalid_evidence_ids=sorted(list(global_invalid_ids)),
    )

    # Attach to synthesis object
    synthesis.validation_report = report
    synthesis.evidence_references = referenced_sources

    return report, referenced_sources
