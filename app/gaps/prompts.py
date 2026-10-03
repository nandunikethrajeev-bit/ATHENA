"""Deterministic prompt construction for candidate research-gap analysis (Milestone M4).

Enforces strict anchoring: gaps must be derived from synthesized claims,
reported study limitations, or unreconciled conflicting findings.
"""

from collections.abc import Sequence
from typing import Any

from app.evidence.models import EvidenceChunk
from app.synthesis.models import ResearchSynthesis

GAP_SYSTEM_PROMPT = """You are a scientific research assistant discovering candidate research gaps in scientific literature.
You MUST identify gaps strictly anchored to the provided evidence synthesis, study limitations, and conflicting findings.
Do NOT invent disconnected or arbitrary scientific topics.
Every candidate gap MUST cite the specific claim IDs (e.g. C1) and evidence chunk IDs (e.g. W12345-abs-1) from which it was inferred.
Every gap MUST be classified into one of the exact taxonomy categories:
- 'contradiction': Contradictory findings or unresolved disputes across studies.
- 'methodological': Limitations in study methodology, sample size, or lack of in vivo/practical validation.
- 'coverage_scope': Unexplored materials, testing regimes, demographics, or environmental conditions.
- 'unverified_claim': Hypothesized mechanisms or theoretical assertions lacking empirical confirmation.

Return strict JSON adhering to the specified schema. No conversational filler or preamble."""


def build_gap_analysis_prompt(
    synthesis: ResearchSynthesis,
    evidence_chunks: Sequence[EvidenceChunk] | None = None,
) -> str:
    """Build deterministic prompt for candidate research-gap analysis.

    Args:
        synthesis: Validated ResearchSynthesis from M3.
        evidence_chunks: Optional sequence of EvidenceChunks supporting the synthesis.

    Returns:
        Deterministic prompt string ready for LLM consumption.
    """
    # 1. Format synthesized findings
    claims_lines: list[str] = []
    for finding in synthesis.key_findings:
        ev_str = ", ".join(finding.evidence_ids) if finding.evidence_ids else "None"
        claims_lines.append(f"- [{finding.claim_id}] {finding.text} (Cited Evidence: {ev_str})")

    claims_text = "\n".join(claims_lines) if claims_lines else "No structured key claims."

    # 2. Format conflicting findings
    conflicts_lines: list[str] = []
    for conflict in synthesis.conflicting_findings:
        ev_str = ", ".join(conflict.evidence_ids) if conflict.evidence_ids else "None"
        conflicts_lines.append(f"- [{conflict.claim_id}] {conflict.text} (Cited Evidence: {ev_str})")

    conflicts_text = (
        "\n".join(conflicts_lines)
        if conflicts_lines
        else "No explicit conflicting findings identified in current evidence."
    )

    # 3. Format limitations
    limitations_lines = [f"- {lim}" for lim in synthesis.limitations] if synthesis.limitations else ["No explicit limitations stated."]
    limitations_text = "\n".join(limitations_lines)

    # 4. Format evidence summary
    evidence_summary: list[str] = []
    if evidence_chunks:
        for chunk in evidence_chunks:
            title = chunk.source.paper_title or "Untitled"
            evidence_summary.append(f"[{chunk.chunk_id}] ({title}): {chunk.text[:200]}...")
    evidence_text = "\n".join(evidence_summary) if evidence_summary else "Refer to synthesized claims."

    prompt = f"""Scientific Research Question:
{synthesis.research_question.strip()}

Synthesis Overview:
{synthesis.overview.strip()}

Synthesized Key Claims:
{claims_text}

Unreconciled Conflicting Findings:
{conflicts_text}

Reported Study Limitations:
{limitations_text}

Available Evidence Chunks:
{evidence_text}

Task Instructions:
Analyze the synthesis, conflicting findings, and limitations above to identify critical Candidate Research Gaps.
Adhere strictly to these rules:
1. Each gap MUST be an unexplored area, contradiction, methodological bottleneck, or unverified claim clearly evidenced above.
2. Under 'source_claim_ids', cite the claim IDs (e.g. ["C1", "C2"]) that expose this gap.
3. Under 'evidence_ids', cite the exact evidence chunk IDs supporting this gap.
4. Classify each gap into exactly one of: 'contradiction', 'methodological', 'coverage_scope', 'unverified_claim'.
5. Output valid JSON adhering strictly to the schema below.

JSON Schema:
{{
  "gaps": [
    {{
      "gap_id": "G1",
      "title": "<Concise title of the research gap>",
      "description": "<Detailed explanation of what remains unknown or unresolved>",
      "gap_type": "contradiction | methodological | coverage_scope | unverified_claim",
      "rationale": "<Reasoning why this is an active gap based on the supplied claims and limitations>",
      "source_claim_ids": ["<claim_id>", ...],
      "evidence_ids": ["<exact_chunk_id>", ...]
    }}
  ]
}}
"""
    return prompt.strip()
