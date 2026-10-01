"""ATHENA application entry point.

Milestones supported:
- M0: Project foundation
- M1: Scientific literature retrieval from OpenAlex
- M2: Evidence retrieval & document processing

Usage:
    python -m app.main                                  # Verify system startup / banner
    python -m app.main "research question"              # Retrieve scientific papers
    python -m app.main "research question" --extract-evidence   # Extract and retrieve evidence
    python -m app.main "research question" --max-results 5 --export data/results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from app import __version__
    from app.evidence import (
        EvidenceIndex,
        build_evidence_context,
        chunk_documents,
        normalize_paper,
    )
    from app.retrieval import (
        OpenAlexRateLimitError,
        OpenAlexNetworkError,
        QueryValidationError,
        RetrievalError,
        search_papers,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    # Put the project root on sys.path so the "app" package resolves
    # when this file is executed as a plain script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import __version__
    from app.evidence import (
        EvidenceIndex,
        build_evidence_context,
        chunk_documents,
        normalize_paper,
    )
    from app.retrieval import (
        OpenAlexRateLimitError,
        OpenAlexNetworkError,
        QueryValidationError,
        RetrievalError,
        search_papers,
    )


def format_paper_summary(index: int, paper) -> str:
    """Format a single retrieved Paper for compact terminal output."""
    title = paper.title or "Untitled"
    if paper.authors:
        if len(paper.authors) > 3:
            authors_str = ", ".join(paper.authors[:3]) + f" et al. ({len(paper.authors)} total)"
        else:
            authors_str = ", ".join(paper.authors)
    else:
        authors_str = "Unknown / Not specified"

    year_str = str(paper.publication_year) if paper.publication_year is not None else "N/A"
    doi_str = paper.doi if paper.doi else "N/A"
    citations_str = str(paper.cited_by_count) if paper.cited_by_count is not None else "N/A"
    venue_str = f"\nVenue:       {paper.venue}" if paper.venue else ""
    openalex_id = paper.openalex_id or "N/A"

    return (
        f"[{index}]\n"
        f"Title:       {title}\n"
        f"Authors:     {authors_str}\n"
        f"Year:        {year_str}{venue_str}\n"
        f"DOI:         {doi_str}\n"
        f"Citations:   {citations_str}\n"
        f"OpenAlex ID: {openalex_id}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """ATHENA CLI entry point supporting system verification and literature retrieval."""
    if argv is None:
        argv = []

    parser = argparse.ArgumentParser(
        prog="athena",
        description="ATHENA — Autonomous Scientific Intelligence",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Scientific research question or keywords to search (e.g. 'machine learning Alzheimer\\'s disease')",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of papers to retrieve (1 to 50, default: 10)",
    )
    parser.add_argument(
        "--extract-evidence",
        action="store_true",
        default=False,
        help="Extract, chunk, index, and retrieve relevant evidence chunks (Milestone M2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top evidence chunks to retrieve with --extract-evidence (default: 5)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export retrieved paper metadata and evidence as a JSON file",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Contact email for OpenAlex polite pool access",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAlex API key for uninterrupted access",
    )

    args = parser.parse_args(argv)

    # Bare invocation: print system startup banner and usage instructions
    if args.query is None:
        print(f"ATHENA v{__version__} - Autonomous Scientific Intelligence")
        print("Status: project foundation (M0) active. Literature retrieval (M1) ready. Evidence retrieval (M2) ready.")
        print('Usage: python -m app.main "<research question>" [--extract-evidence] [--max-results N] [--export output.json]')
        return 0

    print(f"ATHENA v{__version__} — Scientific Research Assistant")
    print(f"Searching OpenAlex for: {args.query!r} (max_results={args.max_results})...\n")

    try:
        papers = search_papers(
            query=args.query,
            max_results=args.max_results,
            email=args.email,
            api_key=args.api_key,
        )
    except QueryValidationError as exc:
        print(f"Error: Invalid query - {exc}", file=sys.stderr)
        return 2
    except OpenAlexRateLimitError as exc:
        print(f"\n[OpenAlex Rate Limit] {exc}", file=sys.stderr)
        return 1
    except OpenAlexNetworkError as exc:
        print(f"\n[Network Error] {exc}", file=sys.stderr)
        return 1
    except RetrievalError as exc:
        print(f"\n[Retrieval Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Unexpected Error] An unexpected error occurred: {exc}", file=sys.stderr)
        return 1

    count = len(papers)
    print(f"Retrieved {count} paper(s) from OpenAlex:")
    print("=" * 72)

    if count == 0:
        print("No papers found matching the query criteria.")
        return 0

    for i, paper in enumerate(papers, 1):
        print(format_paper_summary(i, paper))
        print("-" * 72)

    # M2 Evidence extraction and retrieval
    extracted_docs = []
    all_chunks = []
    retrieved_matches = []

    if args.extract_evidence:
        print(f"\n[M2] Normalizing documents and extracting evidence chunks...")
        extracted_docs = [normalize_paper(p) for p in papers]
        all_chunks = chunk_documents(extracted_docs)
        print(f"[M2] Extracted {len(all_chunks)} chunk(s) across {len(extracted_docs)} paper(s).")

        index = EvidenceIndex()
        index.index_chunks(all_chunks)
        retrieved_matches = index.retrieve(query=args.query, top_k=args.top_k)

        context_output = build_evidence_context(
            query=args.query,
            matches=retrieved_matches,
            total_papers=count,
            total_chunks=len(all_chunks),
        )
        print(f"\n{context_output}")

    if args.export:
        export_path = Path(args.export)
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {
                "query": args.query,
                "retrieved_count": count,
                "source": "OpenAlex",
                "papers": [p.to_dict() for p in papers],
            }
            if args.extract_evidence:
                export_data["evidence_chunks_count"] = len(all_chunks)
                export_data["evidence_chunks"] = [c.to_dict() for c in all_chunks]
                export_data["retrieved_matches"] = [m.to_dict() for m in retrieved_matches]

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved results for {count} paper(s) to: {export_path}")
        except Exception as exc:
            print(f"Warning: Failed to export results to {export_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
