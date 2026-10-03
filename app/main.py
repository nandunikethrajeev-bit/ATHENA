"""ATHENA application entry point.

Milestones supported:
- M0: Project foundation
- M1: Scientific literature retrieval from OpenAlex
- M2: Evidence retrieval & document processing
- M3: Scientific evidence synthesis & claim grounding
- M4: Candidate research-gap analysis & validation

Usage:
    python -m app.main                                          # Verify system startup / banner
    python -m app.main "research question"                      # Retrieve scientific papers (M1)
    python -m app.main "research question" --extract-evidence   # Extract and retrieve evidence (M2)
    python -m app.main "research question" --extract-evidence --synthesize --llm-provider mock  # Synthesize (M3)
    python -m app.main "research question" --extract-evidence --synthesize --analyze-gaps --llm-provider mock  # Gaps (M4)
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
        EvidenceStore,
        HybridRetriever,
        build_evidence_context,
        chunk_documents,
        get_embedding_client,
        normalize_paper,
    )
    from app.gaps import (
        GapAnalyzer,
        GapType,
        ResearchGapAnalysis,
    )
    from app.retrieval import (
        OpenAlexRateLimitError,
        OpenAlexNetworkError,
        QueryValidationError,
        RetrievalError,
        search_papers,
    )
    from app.synthesis import (
        EvidenceContext,
        LLMConfigurationError,
        Synthesizer,
        get_llm_client,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    # Put the project root on sys.path so the "app" package resolves
    # when this file is executed as a plain script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import __version__
    from app.evidence import (
        EvidenceIndex,
        EvidenceStore,
        HybridRetriever,
        build_evidence_context,
        chunk_documents,
        get_embedding_client,
        normalize_paper,
    )
    from app.gaps import (
        GapAnalyzer,
        GapType,
        ResearchGapAnalysis,
    )
    from app.retrieval import (
        OpenAlexRateLimitError,
        OpenAlexNetworkError,
        QueryValidationError,
        RetrievalError,
        search_papers,
    )
    from app.synthesis import (
        EvidenceContext,
        LLMConfigurationError,
        Synthesizer,
        get_llm_client,
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
    """ATHENA CLI entry point supporting system verification, literature retrieval, and evidence synthesis."""
    if argv is None:
        argv = []

    # Configure UTF-8 encoding with fallback replacement for Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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
        "--synthesize",
        action="store_true",
        default=False,
        help="Synthesize retrieved evidence into structured scientific claims (Milestone M3)",
    )
    parser.add_argument(
        "--analyze-gaps",
        action="store_true",
        default=False,
        help="Analyze synthesized evidence to discover candidate research gaps (Milestone M4)",
    )
    parser.add_argument(
        "--gap-type",
        type=str,
        default=None,
        choices=["contradiction", "methodological", "coverage_scope", "unverified_claim"],
        help="Filter candidate research gaps by taxonomy type (Milestone M4)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider for synthesis: mock, openai, ollama, openrouter (default: from ATHENA_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name for synthesis (e.g. gpt-4o-mini, llama3.1:8b)",
    )
    parser.add_argument(
        "--persist-evidence",
        action="store_true",
        default=False,
        help="Persist extracted evidence chunks and provenance to SQLite database (Milestone M3)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/athena_evidence.db",
        help="Path to SQLite evidence database (default: data/athena_evidence.db)",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="bm25",
        choices=["bm25", "semantic", "hybrid"],
        help="Evidence retrieval mode: bm25, semantic, or hybrid (default: bm25)",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="mock",
        help="Embedding provider: mock (default for offline/testing) or ollama",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model tag (e.g. nomic-embed-text, qwen2.5:7b)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export retrieved paper metadata, evidence, and synthesis as a JSON file",
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
        print("Status: project foundation (M0) active. Literature retrieval (M1) ready. Evidence retrieval (M2) ready. Evidence synthesis (M3) ready. Research gaps (M4) ready.")
        print('Usage: python -m app.main "<research question>" [--extract-evidence] [--synthesize] [--analyze-gaps] [--llm-provider mock] [--max-results N] [--export output.json]')
        return 0

    # Auto-enable prerequisite stages
    if args.analyze_gaps:
        args.synthesize = True
        args.extract_evidence = True
    elif args.synthesize and not args.extract_evidence:
        args.extract_evidence = True

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

        if args.persist_evidence:
            print(f"[M3] Persisting {len(all_chunks)} chunk(s) to SQLite store ({args.db_path})...")
            store = EvidenceStore(args.db_path)
            store.save_chunks(all_chunks)
            store.close()

        if args.retrieval_mode in ("semantic", "hybrid"):
            print(f"[M3] Initializing {args.retrieval_mode} retrieval with embedding provider: {args.embedding_provider}...")
            embedder = get_embedding_client(
                provider=args.embedding_provider,
                model=args.embedding_model,
            )
            retriever = HybridRetriever(embedding_client=embedder)
            retriever.index_chunks(all_chunks)
            retrieved_matches = retriever.retrieve(
                query=args.query,
                top_k=args.top_k,
                mode=args.retrieval_mode,
            )
        else:
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

    # M3 Evidence synthesis
    synthesis = None
    if args.synthesize:
        print(f"\n[M3] Synthesizing evidence using LLM provider...")
        try:
            client = get_llm_client(
                provider=args.llm_provider,
                model=args.llm_model,
            )
            synthesizer = Synthesizer(client=client)
            evidence_ctx = EvidenceContext(
                query=args.query,
                matches=retrieved_matches,
                total_papers=count,
                total_chunks=len(all_chunks),
            )
            synthesis = synthesizer.synthesize(
                research_question=args.query,
                evidence_context=evidence_ctx,
            )
            print(f"\n{synthesis.to_markdown()}")
        except LLMConfigurationError as exc:
            print(f"\n[LLM Configuration Error] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[Synthesis Error] Failed to synthesize evidence: {exc}", file=sys.stderr)
            return 1

    # M4 Candidate Research-Gap Analysis
    gap_analysis = None
    if args.analyze_gaps and synthesis is not None:
        print(f"\n[M4] Analyzing candidate research gaps from scientific synthesis...")
        try:
            client = get_llm_client(
                provider=args.llm_provider,
                model=args.llm_model,
            )
            analyzer = GapAnalyzer(client=client)
            gap_analysis = analyzer.analyze(
                synthesis=synthesis,
                evidence_chunks=all_chunks,
                filter_type=args.gap_type,
            )
            print(f"\n{gap_analysis.to_markdown()}")
        except LLMConfigurationError as exc:
            print(f"\n[LLM Configuration Error] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[Gap Analysis Error] Failed to analyze research gaps: {exc}", file=sys.stderr)
            return 1

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
            if args.synthesize and synthesis is not None:
                export_data["synthesis"] = synthesis.to_dict()
            if args.analyze_gaps and gap_analysis is not None:
                export_data["research_gaps"] = gap_analysis.to_dict()

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved results for {count} paper(s) to: {export_path}")
        except Exception as exc:
            print(f"Warning: Failed to export results to {export_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
