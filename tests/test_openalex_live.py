"""Live API integration test for ATHENA literature retrieval against OpenAlex.

This test contacts the official OpenAlex REST API over the network.
It is marked with `@pytest.mark.integration` and excluded from default
offline unit test runs.

Run explicitly with:
    python -m pytest -m integration
    python tests/test_openalex_live.py
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app.retrieval import OpenAlexRateLimitError, OpenAlexNetworkError, search_papers


@pytest.mark.integration
def test_live_openalex_search():
    """Execute a live search against OpenAlex for real scientific literature."""
    query = "machine learning Alzheimer's disease"
    print(f"\n[LIVE API TEST] Contacting OpenAlex for query: {query!r}...")

    try:
        papers = search_papers(query=query, max_results=3, timeout=25.0)
    except OpenAlexRateLimitError as exc:
        pytest.skip(
            f"OpenAlex cluster is currently rate-limiting anonymous search: {exc}. "
            "Set OPENALEX_API_KEY for uninterrupted live access."
        )
    except OpenAlexNetworkError as exc:
        pytest.skip(f"Network connectivity to OpenAlex failed: {exc}")

    assert len(papers) > 0, "Expected at least 1 paper from OpenAlex."

    first_paper = papers[0]
    print(f"\nRetrieved {len(papers)} paper(s). First paper:")
    print(f"Title:       {first_paper.title}")
    print(f"Authors:     {first_paper.authors}")
    print(f"Year:        {first_paper.publication_year}")
    print(f"OpenAlex ID: {first_paper.openalex_id}")
    print(f"DOI:         {first_paper.doi}")
    print(f"Abstract:    {(first_paper.abstract[:100] + '...') if first_paper.abstract else None}")

    # Verify data integrity
    assert first_paper.source == "OpenAlex"
    assert first_paper.openalex_id.startswith("https://openalex.org/W")
    assert isinstance(first_paper.title, str) and len(first_paper.title) > 0
    assert isinstance(first_paper.authors, list)
    if first_paper.publication_year is not None:
        assert isinstance(first_paper.publication_year, int)
        assert 1900 <= first_paper.publication_year <= 2030

    # Ensure serialization
    data = first_paper.to_dict()
    assert data["openalex_id"] == first_paper.openalex_id
    assert data["source"] == "OpenAlex"


if __name__ == "__main__":
    print("Running live OpenAlex integration test...")
    try:
        test_live_openalex_search()
        print("\nLive OpenAlex integration test PASSED.")
    except pytest.skip.Exception as s:
        print(f"\nLive OpenAlex test skipped: {s}")
    except Exception as e:
        print(f"\nLive OpenAlex test FAILED: {e}", file=sys.stderr)
        sys.exit(1)
