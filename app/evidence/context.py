"""Evidence context formatting and prompt assembly for ATHENA (Milestone M2).

Formats retrieved EvidenceMatch items into structured, citation-traceable context blocks
for downstream synthesis, verification, or user presentation.
"""

from typing import Sequence

from app.evidence.models import EvidenceMatch, EvidenceType


def format_evidence_item(index: int, match: EvidenceMatch) -> str:
    """Format a single EvidenceMatch with complete provenance and evidence classification."""
    chunk = match.chunk
    source = chunk.source

    authors_str = ", ".join(source.authors[:3]) + (f" et al. ({len(source.authors)} total)" if len(source.authors) > 3 else "")
    if not authors_str:
        authors_str = "Unknown"

    year_str = str(source.publication_year) if source.publication_year is not None else "N/A"
    doi_str = source.doi or "N/A"
    url_str = source.landing_page_url or "N/A"
    matched_terms_str = ", ".join(match.matched_terms) if match.matched_terms else "N/A"

    header_lines = [
        f"[{index}] EVIDENCE ITEM: {chunk.chunk_id}",
        f"Paper:           {source.paper_title or 'Untitled'}",
        f"Authors:         {authors_str}",
        f"Year / Venue:    {year_str} | {source.venue or 'N/A'}",
        f"OpenAlex ID:     {source.openalex_id}",
        f"DOI:             {doi_str}",
        f"Source URL:      {url_str}",
        f"Evidence Type:   {chunk.evidence_type.value.upper()} (Section: {chunk.section})",
        f"Relevance Score: {match.score:.3f} (Matched terms: {matched_terms_str})",
    ]

    # Explicit warning if abstract or metadata
    if chunk.evidence_type == EvidenceType.ABSTRACT:
        header_lines.append("[Integrity Note: Evidence sourced from peer-reviewed abstract; full-text body not indexed.]")
    elif chunk.evidence_type == EvidenceType.METADATA:
        header_lines.append("[Integrity Note: Bibliographic metadata only; abstract and full-text not indexed.]")

    header = "\n".join(header_lines)
    content = f"Content:\n{chunk.text}"

    return f"{header}\n\n{content}"


def build_evidence_context(
    query: str,
    matches: Sequence[EvidenceMatch],
    total_papers: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Assemble a formatted evidence context report from ranked retrieval matches.

    Args:
        query: Research query that produced the retrieval results.
        matches: Ranked list of EvidenceMatch instances.
        total_papers: Optional count of papers analyzed.
        total_chunks: Optional count of total evidence chunks indexed.

    Returns:
        Structured markdown/text block ready for terminal display or LLM grounding.
    """
    if not matches:
        return (
            f"### Evidence Context for Query: {query!r}\n\n"
            "No relevant evidence chunks matched the query criteria."
        )

    stats_line = f"Retrieved {len(matches)} relevant evidence chunk(s)"
    if total_chunks is not None and total_papers is not None:
        stats_line += f" from {total_chunks} indexed chunk(s) across {total_papers} paper(s)"
    stats_line += "."

    divider = "=" * 76
    sub_divider = "-" * 76

    blocks = [
        f"### ATHENA Evidence Context (Milestone M2)",
        f"Query: {query!r}",
        stats_line,
        divider,
    ]

    for i, match in enumerate(matches, 1):
        blocks.append(format_evidence_item(i, match))
        blocks.append(sub_divider)

    return "\n".join(blocks)
