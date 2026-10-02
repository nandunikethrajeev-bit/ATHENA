"""Deterministic prompt templates and construction for scientific evidence synthesis (Milestone M3).

Enforces strict scientific grounding: the LLM is instructed to synthesize solely from
supplied evidence chunks, cite exact chunk identifiers, report uncertainties/conflicts,
and output strictly structured JSON.
"""

from typing import Sequence

from app.evidence.models import EvidenceChunk, EvidenceType

SYSTEM_PROMPT = """You are a scientific research assistant synthesizing scientific evidence.
You MUST use ONLY the evidence supplied in the context.
Every factual claim MUST reference one or more exact evidence IDs.
Do NOT invent citations.
Do NOT invent papers, authors, DOI values, numerical results, or experimental findings.
If evidence conflicts, explicitly report the disagreement.
If the available evidence is insufficient, explicitly state that.
Do not treat abstract evidence as full-text evidence.
Return strict JSON matching the required schema. No conversational filler or preamble."""


def format_evidence_block(chunk: EvidenceChunk) -> str:
    """Format a single EvidenceChunk with provenance and integrity constraints."""
    source = chunk.source
    authors_str = ", ".join(source.authors[:3]) + (f" et al. ({len(source.authors)} total)" if len(source.authors) > 3 else "")
    if not authors_str:
        authors_str = "Unknown"

    year_str = str(source.publication_year) if source.publication_year is not None else "N/A"
    venue_str = source.venue or "N/A"
    doi_str = source.doi or "N/A"
    url_str = source.landing_page_url or "N/A"

    integrity_note = ""
    if chunk.evidence_type == EvidenceType.ABSTRACT:
        integrity_note = "[Integrity Note: Evidence sourced from peer-reviewed abstract; full-text body not indexed. Do not extrapolate study methodology beyond what is stated.]"
    elif chunk.evidence_type == EvidenceType.METADATA:
        integrity_note = "[Integrity Note: Bibliographic metadata only; abstract and full-text body not indexed.]"
    elif chunk.evidence_type == EvidenceType.FULL_TEXT:
        integrity_note = "[Integrity Note: Full-text section body.]"

    lines = [
        f"=== EVIDENCE ITEM: {chunk.chunk_id} ===",
        f"Paper Title:   {source.paper_title or 'Untitled'}",
        f"Authors:       {authors_str}",
        f"Year / Venue:  {year_str} | {venue_str}",
        f"OpenAlex ID:   {source.openalex_id}",
        f"DOI / URL:     {doi_str} | {url_str}",
        f"Evidence Type: {chunk.evidence_type.value.upper()} (Section: {chunk.section})",
    ]
    if integrity_note:
        lines.append(integrity_note)

    lines.append("Content:")
    lines.append(chunk.text.strip())
    lines.append("=" * 40)

    return "\n".join(lines)


def build_synthesis_prompt(research_question: str, chunks: Sequence[EvidenceChunk]) -> str:
    """Assemble a deterministic prompt containing research question, ranked evidence, and schema instructions.

    Args:
        research_question: The user's scientific research question.
        chunks: Sequence of EvidenceChunk instances retrieved by M2.

    Returns:
        Deterministic prompt string ready for LLM consumption.
    """
    evidence_blocks: list[str] = []
    for chunk in chunks:
        evidence_blocks.append(format_evidence_block(chunk))

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "NO EVIDENCE RETRIEVED."

    prompt = f"""Scientific Research Question:
{research_question.strip()}

Available Evidence Chunks ({len(chunks)} items):
{evidence_text}

Synthesis Instructions:
Synthesize the evidence above to answer the research question.
You MUST adhere to these constraints:
1. Base all statements ONLY on the provided evidence chunks above.
2. Every item in 'key_findings' and 'conflicting_findings' MUST cite the exact evidence IDs (e.g. ["{chunks[0].chunk_id if chunks else 'EID'}"]) supporting it.
3. Do NOT cite any evidence ID not explicitly listed in the Available Evidence Chunks above.
4. If two or more studies present contradictory or differing conclusions, explicitly report them under 'conflicting_findings'.
5. Report any limitations of the available evidence under 'limitations' (e.g., lack of full text, limited sample sizes, unaddressed sub-questions).
6. Output MUST be valid JSON adhering strictly to the schema below.

JSON Schema:
{{
  "overview": "<Comprehensive, evidence-based synthesis overview answering the question>",
  "key_findings": [
    {{
      "claim_id": "C1",
      "text": "<Specific finding backed by evidence>",
      "evidence_ids": ["<exact_chunk_id>", ...]
    }}
  ],
  "conflicting_findings": [
    {{
      "claim_id": "C2",
      "text": "<Contradictory or divergent finding observed across evidence>",
      "evidence_ids": ["<exact_chunk_id>", ...]
    }}
  ],
  "limitations": [
    "<Scientific limitation, evidence gap, or methodology caveat>"
  ]
}}
"""
    return prompt.strip()
