"""Deterministic prompt construction for candidate hypothesis generation (Milestone M5).

Enforces tripartite anchoring: every hypothesis must resolve an identified candidate gap (Gk),
derive its mechanistic rationale from synthesized claims (Ci), and cite underlying evidence chunks (W...).
Requires explicit independent/dependent variables and falsification criteria.
"""

from collections.abc import Sequence
from typing import Any

from app.evidence.models import EvidenceChunk
from app.gaps.models import ResearchGapAnalysis
from app.synthesis.models import ResearchSynthesis

HYPOTHESIS_SYSTEM_PROMPT = """You are a rigorous scientific research assistant generating candidate scientific hypotheses.
You MUST formulate testable, falsifiable candidate hypotheses strictly anchored to the provided candidate research gaps, synthesized claims, and evidence chunks.
Do NOT invent arbitrary, disconnected scientific claims.
Every candidate hypothesis MUST:
1. Cite the exact 'target_gap_id' (e.g. G1) it proposes to resolve.
2. Cite the specific 'source_claim_ids' (e.g. ["C1", "C2"]) grounding its mechanistic premise.
3. Cite the exact 'evidence_ids' (e.g. ["W12345-abs-1"]) supporting the underlying data.
4. Specify a clear 'statement' of the hypothesis.
5. Provide a mechanistic 'rationale' explaining how it addresses the target gap.
6. Detail the causal/physical/biochemical 'proposed_mechanism'.
7. Enumerate manipulated 'independent_variables' and measured 'dependent_variables'.
8. Specify explicit 'falsification_criteria': concrete experimental observations or quantitative thresholds that would empirically refute the hypothesis.

Return strict JSON adhering to the specified schema. No conversational preamble or postscript."""


def build_hypothesis_prompt(
    gap_analysis: ResearchGapAnalysis,
    synthesis: ResearchSynthesis,
    evidence_chunks: Sequence[EvidenceChunk] | None = None,
) -> str:
    """Build deterministic prompt for candidate hypothesis generation.

    Args:
        gap_analysis: Validated ResearchGapAnalysis from M4.
        synthesis: Validated ResearchSynthesis from M3.
        evidence_chunks: Optional sequence of EvidenceChunks supporting the synthesis.

    Returns:
        Deterministic prompt string ready for LLM consumption.
    """
    # 1. Format candidate research gaps
    gap_lines: list[str] = []
    for gap in gap_analysis.gaps:
        claims_str = ", ".join(gap.source_claim_ids) if gap.source_claim_ids else "None"
        ev_str = ", ".join(gap.evidence_ids) if gap.evidence_ids else "None"
        gap_lines.extend([
            f"- [{gap.gap_id}] {gap.title} (Type: {gap.gap_type.value.upper()})",
            f"  Description: {gap.description}",
            f"  Rationale:   {gap.rationale}",
            f"  Anchors:     Claims: [{claims_str}] | Evidence: [{ev_str}]",
        ])

    gaps_text = "\n".join(gap_lines) if gap_lines else "No structured candidate research gaps identified."

    # 2. Format synthesized key claims
    claims_lines: list[str] = []
    for finding in synthesis.key_findings:
        ev_str = ", ".join(finding.evidence_ids) if finding.evidence_ids else "None"
        claims_lines.append(f"- [{finding.claim_id}] {finding.text} (Evidence: {ev_str})")

    claims_text = "\n".join(claims_lines) if claims_lines else "No structured key claims."

    # 3. Format conflicting findings
    conflicts_lines: list[str] = []
    for conflict in synthesis.conflicting_findings:
        ev_str = ", ".join(conflict.evidence_ids) if conflict.evidence_ids else "None"
        conflicts_lines.append(f"- [{conflict.claim_id}] {conflict.text} (Evidence: {ev_str})")

    conflicts_text = (
        "\n".join(conflicts_lines)
        if conflicts_lines
        else "No explicit conflicting findings identified in current evidence."
    )

    # 4. Format limitations
    limitations_lines = [f"- {lim}" for lim in synthesis.limitations] if synthesis.limitations else ["No explicit limitations stated."]
    limitations_text = "\n".join(limitations_lines)

    # 5. Format evidence excerpts
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

Candidate Research Gaps to Address:
{gaps_text}

Synthesized Key Claims:
{claims_text}

Unreconciled Conflicting Findings:
{conflicts_text}

Reported Study Limitations:
{limitations_text}

Available Evidence Chunks:
{evidence_text}

Task Instructions:
Formulate discrete, testable, and falsifiable Candidate Hypotheses to resolve the identified Candidate Research Gaps above.
Adhere strictly to these scientific rules:
1. Each hypothesis MUST directly target one identified research gap by citing 'target_gap_id' (e.g. "G1").
2. Anchor each hypothesis in synthesized literature: cite 'source_claim_ids' and 'evidence_ids'.
3. Detail the 'proposed_mechanism' explaining why the hypothesized relationship is expected to hold.
4. Specify operational experimental variables: 'independent_variables' (what is varied) and 'dependent_variables' (what is measured).
5. Formulate unambiguous 'falsification_criteria': what specific observation, null result, or quantitative measurement would prove the hypothesis false.
6. Output valid JSON adhering strictly to the schema below.

JSON Schema:
{{
  "hypotheses": [
    {{
      "hypothesis_id": "H1",
      "target_gap_id": "G1",
      "title": "Concise hypothesis title",
      "statement": "Precise testable proposition",
      "rationale": "Why this resolves the target gap",
      "proposed_mechanism": "Causal / physical / chemical / biological mechanism",
      "independent_variables": ["Variable A", "Condition B"],
      "dependent_variables": ["Outcome X", "Metric Y"],
      "falsification_criteria": "The hypothesis is refuted if...",
      "source_claim_ids": ["C1"],
      "evidence_ids": ["chunk_id_1"]
    }}
  ]
}}"""

    return prompt
