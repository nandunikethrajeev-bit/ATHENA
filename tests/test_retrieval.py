"""Unit tests for ATHENA literature retrieval subsystem (Milestone M1).

All tests in this file are deterministic, fully offline, and do NOT make
network requests to OpenAlex.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.main import format_paper_summary, main
from app.retrieval.exceptions import (
    OpenAlexAPIError,
    OpenAlexNetworkError,
    OpenAlexRateLimitError,
    QueryValidationError,
    RetrievalError,
)
from app.retrieval.models import OpenAccessInfo, Paper
from app.retrieval.openalex import OpenAlexClient, search_papers
from app.retrieval.utils import reconstruct_abstract, validate_query


# ==============================================================================
# 1. Query Validation Tests
# ==============================================================================

class TestQueryValidation:
    def test_valid_query_normalizes_whitespace(self):
        cleaned, count = validate_query("   machine learning Alzheimer's   ", max_results=15)
        assert cleaned == "machine learning Alzheimer's"
        assert count == 15

    def test_research_question_with_question_mark_sanitized(self):
        cleaned, count = validate_query(
            "Can machine learning improve early detection of Alzheimer's disease?",
            max_results=5,
        )
        assert cleaned == "Can machine learning improve early detection of Alzheimer's disease"
        assert count == 5

    def test_wildcards_removed_cleanly(self):
        cleaned, _ = validate_query("Alzheimer* AND dementia?")
        assert cleaned == "Alzheimer AND dementia"

    def test_default_max_results(self):
        cleaned, count = validate_query("CRISPR gene editing")
        assert cleaned == "CRISPR gene editing"
        assert count == 10

    @pytest.mark.parametrize("invalid_query", ["", "   ", "\t\n  \r"])
    def test_reject_empty_or_whitespace_query(self, invalid_query):
        with pytest.raises(QueryValidationError, match="empty or contain only whitespace"):
            validate_query(invalid_query)

    def test_reject_non_string_query(self):
        with pytest.raises(QueryValidationError, match="Query must be a string"):
            validate_query(12345)  # type: ignore

    @pytest.mark.parametrize("invalid_count", [0, -1, -100])
    def test_reject_non_positive_max_results(self, invalid_count):
        with pytest.raises(QueryValidationError, match="positive integer greater than 0"):
            validate_query("valid query", max_results=invalid_count)

    def test_reject_excessive_max_results(self):
        with pytest.raises(QueryValidationError, match="exceeds the safe maximum limit"):
            validate_query("valid query", max_results=51)

    @pytest.mark.parametrize("invalid_type", ["10", 5.5, True, False, None])
    def test_reject_non_integer_max_results(self, invalid_type):
        with pytest.raises(QueryValidationError, match="max_results must be an integer"):
            validate_query("valid query", max_results=invalid_type)  # type: ignore


# ==============================================================================
# 2. Abstract Reconstruction Tests
# ==============================================================================

class TestAbstractReconstruction:
    def test_ordered_reconstruction(self):
        inverted = {
            "Machine": [0],
            "learning": [1],
            "improves": [2],
            "detection.": [3],
        }
        reconstructed = reconstruct_abstract(inverted)
        assert reconstructed == "Machine learning improves detection."

    def test_unordered_reconstruction(self):
        inverted = {
            "detection.": [3],
            "improves": [2],
            "Machine": [0],
            "learning": [1],
        }
        reconstructed = reconstruct_abstract(inverted)
        assert reconstructed == "Machine learning improves detection."

    def test_repeated_tokens_at_different_positions(self):
        inverted = {
            "the": [0, 4],
            "model": [1],
            "evaluated": [2],
            "against": [3],
            "baseline.": [5],
        }
        reconstructed = reconstruct_abstract(inverted)
        assert reconstructed == "the model evaluated against the baseline."

    @pytest.mark.parametrize("empty_val", [None, {}, "", 123, []])
    def test_empty_or_non_dict_returns_none(self, empty_val):
        assert reconstruct_abstract(empty_val) is None  # type: ignore

    def test_inverted_index_with_malformed_entries(self):
        inverted = {
            "valid": [0],
            "corrupt": "not a list",
            None: [1],
            "negative": [-5],
            "target": [2],
        }
        reconstructed = reconstruct_abstract(inverted)  # type: ignore
        assert reconstructed == "valid target"

    def test_whitespace_only_reconstruction_returns_none(self):
        inverted = {"   ": [0], "": [1]}
        assert reconstruct_abstract(inverted) is None


# ==============================================================================
# 3. OpenAlex Response Parsing Tests
# ==============================================================================

class TestOpenAlexResponseParsing:
    @pytest.fixture
    def sample_work(self):
        return {
            "id": "https://openalex.org/W2741809807",
            "title": "Machine learning in Alzheimer's disease diagnosis",
            "publication_year": 2021,
            "doi": "https://doi.org/10.1038/s41598-021-98765-4",
            "type": "journal-article",
            "cited_by_count": 87,
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A111",
                        "display_name": "Alice Smith",
                    },
                    "author_position": "first",
                },
                {
                    "author": {
                        "id": "https://openalex.org/A222",
                        "display_name": "Bob Jones",
                    },
                    "author_position": "middle",
                },
            ],
            "primary_location": {
                "source": {
                    "id": "https://openalex.org/S100",
                    "display_name": "Scientific Reports",
                },
                "landing_page_url": "https://www.nature.com/articles/s41598-021-98765-4",
            },
            "open_access": {
                "is_oa": True,
                "oa_status": "gold",
                "oa_url": "https://www.nature.com/articles/s41598-021-98765-4.pdf",
            },
            "abstract_inverted_index": {
                "Alzheimer's": [0],
                "disease": [1],
                "diagnosis": [2],
                "remains": [3],
                "challenging.": [4],
            },
        }

    def test_parse_complete_work(self, sample_work):
        client = OpenAlexClient()
        paper = client.parse_work(sample_work)

        assert isinstance(paper, Paper)
        assert paper.title == "Machine learning in Alzheimer's disease diagnosis"
        assert paper.openalex_id == "https://openalex.org/W2741809807"
        assert paper.short_id == "W2741809807"
        assert paper.source == "OpenAlex"
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.publication_year == 2021
        assert paper.doi == "https://doi.org/10.1038/s41598-021-98765-4"
        assert paper.abstract == "Alzheimer's disease diagnosis remains challenging."
        assert paper.venue == "Scientific Reports"
        assert paper.cited_by_count == 87
        assert paper.publication_type == "journal-article"
        assert paper.landing_page_url == "https://www.nature.com/articles/s41598-021-98765-4"
        assert paper.open_access == OpenAccessInfo(
            is_oa=True,
            oa_status="gold",
            oa_url="https://www.nature.com/articles/s41598-021-98765-4.pdf",
        )
        assert paper.raw_id == "https://openalex.org/W2741809807"

    def test_paper_to_dict_serializable(self, sample_work):
        client = OpenAlexClient()
        paper = client.parse_work(sample_work)
        d = paper.to_dict()

        assert d["title"] == paper.title
        assert d["openalex_id"] == "https://openalex.org/W2741809807"
        assert d["source"] == "OpenAlex"
        assert d["authors"] == ["Alice Smith", "Bob Jones"]
        assert d["open_access"]["is_oa"] is True
        # Verify valid JSON serialization
        serialized = json.dumps(d)
        assert "Alice Smith" in serialized


# ==============================================================================
# 4. Missing Fields Handling Tests
# ==============================================================================

class TestMissingFieldsHandling:
    def test_work_with_minimal_data(self):
        minimal_work = {
            "id": "https://openalex.org/W999",
        }
        client = OpenAlexClient()
        paper = client.parse_work(minimal_work)

        assert paper.openalex_id == "https://openalex.org/W999"
        assert paper.short_id == "W999"
        assert paper.title is None
        assert paper.authors == []
        assert paper.publication_year is None
        assert paper.doi is None
        assert paper.abstract is None
        assert paper.venue is None
        assert paper.cited_by_count is None
        assert paper.publication_type is None
        assert paper.landing_page_url == "https://openalex.org/W999"
        assert paper.open_access is None
        assert paper.source == "OpenAlex"

    def test_venue_fallback_to_host_venue(self):
        work = {
            "id": "https://openalex.org/W1",
            "primary_location": None,
            "host_venue": {"display_name": "Journal of Neuroscience"},
        }
        paper = OpenAlexClient().parse_work(work)
        assert paper.venue == "Journal of Neuroscience"

    def test_venue_fallback_to_locations_list(self):
        work = {
            "id": "https://openalex.org/W2",
            "primary_location": None,
            "host_venue": None,
            "locations": [
                {"source": None},
                {"source": {"display_name": "Nature Medicine"}},
            ],
        }
        paper = OpenAlexClient().parse_work(work)
        assert paper.venue == "Nature Medicine"

    def test_authors_empty_when_authorships_missing_or_malformed(self):
        work = {
            "id": "https://openalex.org/W3",
            "authorships": [
                {},
                {"author": None},
                {"author": {"display_name": ""}},
                {"author": {"display_name": "Dr. Valid"}},
            ],
        }
        paper = OpenAlexClient().parse_work(work)
        assert paper.authors == ["Dr. Valid"]


# ==============================================================================
# 5. Empty Search Results Tests
# ==============================================================================

class TestEmptySearchResults:
    def test_empty_results_list(self):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"meta": {"count": 0}, "results": []})
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        papers = client.search("nonexistent topic query 12345XYZ")
        assert papers == []

    def test_missing_results_key(self):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"meta": {"count": 0}})
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        papers = client.search("query with malformed envelope")
        assert papers == []


# ==============================================================================
# 6. Network and API Failure Handling Tests
# ==============================================================================

class TestNetworkAndAPIFailures:
    def test_http_429_rate_limiting_with_retry_after_header(self):
        headers = {"retry-after": "45"}
        body = {"error": "Rate limit exceeded", "message": "Cluster under elevated load"}
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(429, headers=headers, json=body)
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexRateLimitError) as exc_info:
            client.search("machine learning Alzheimer")

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 45.0
        assert "Retry after 45.0 seconds" in str(exc_info.value)
        assert "OPENALEX_API_KEY" in str(exc_info.value)

    def test_http_429_with_body_retry_after(self):
        body = {"error": "Rate limit exceeded", "retryAfter": 30}
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(429, json=body)
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexRateLimitError) as exc_info:
            client.search("machine learning Alzheimer")

        assert exc_info.value.retry_after == 30.0

    def test_http_500_server_error(self):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(500, text="Internal Server Error")
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexAPIError) as exc_info:
            client.search("Alzheimer biomarkers")

        assert exc_info.value.status_code == 500
        assert "HTTP 500" in str(exc_info.value)

    def test_http_404_error(self):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(404, json={"message": "Not Found"})
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexAPIError) as exc_info:
            client.search("test")

        assert exc_info.value.status_code == 404

    def test_malformed_json_response(self):
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="<html>502 Bad Gateway</html>")
        )
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexAPIError, match="Malformed JSON"):
            client.search("test")

    def test_connection_timeout(self):
        def raise_timeout(req):
            raise httpx.ConnectTimeout("Connection handshake timed out")

        mock_transport = httpx.MockTransport(raise_timeout)
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexNetworkError, match="timed out"):
            client.search("test")

    def test_connection_error(self):
        def raise_connect_error(req):
            raise httpx.ConnectError("Failed to establish a new connection")

        mock_transport = httpx.MockTransport(raise_connect_error)
        mock_client = httpx.Client(transport=mock_transport)
        client = OpenAlexClient(client=mock_client)

        with pytest.raises(OpenAlexNetworkError, match="Failed to communicate"):
            client.search("test")


# ==============================================================================
# 7. CLI Demonstration and Formatting Tests
# ==============================================================================

class TestCLI:
    def test_cli_bare_invocation_shows_banner_and_exits_0(self, capsys):
        code = main([])
        assert code == 0
        captured = capsys.readouterr()
        assert "ATHENA" in captured.out
        assert "Literature retrieval (M1) ready" in captured.out

    def test_format_paper_summary_with_all_fields(self):
        paper = Paper(
            title="Early Diagnosis of Alzheimer's",
            openalex_id="https://openalex.org/W100",
            authors=["Alice", "Bob", "Charlie", "David"],
            publication_year=2023,
            venue="Neurology",
            doi="https://doi.org/10.1000/1",
            cited_by_count=15,
        )
        summary = format_paper_summary(1, paper)
        assert "[1]" in summary
        assert "Early Diagnosis of Alzheimer's" in summary
        assert "Alice, Bob, Charlie et al. (4 total)" in summary
        assert "2023" in summary
        assert "Venue:       Neurology" in summary
        assert "https://doi.org/10.1000/1" in summary
        assert "Citations:   15" in summary
        assert "OpenAlex ID: https://openalex.org/W100" in summary

    def test_format_paper_summary_with_missing_fields(self):
        paper = Paper(
            title=None,
            openalex_id="https://openalex.org/W200",
        )
        summary = format_paper_summary(2, paper)
        assert "[2]" in summary
        assert "Title:       Untitled" in summary
        assert "Authors:     Unknown / Not specified" in summary
        assert "Year:        N/A" in summary
        assert "DOI:         N/A" in summary
        assert "Citations:   N/A" in summary

    @patch("app.main.search_papers")
    def test_cli_search_success_with_json_export(self, mock_search, tmp_path, capsys):
        mock_paper = Paper(
            title="Test Paper",
            openalex_id="https://openalex.org/W1",
            authors=["Researcher One"],
            publication_year=2024,
            cited_by_count=3,
        )
        mock_search.return_value = [mock_paper]

        export_file = tmp_path / "exported_results.json"
        exit_code = main(["machine learning Alzheimer", "--export", str(export_file)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Retrieved 1 paper(s) from OpenAlex" in captured.out
        assert "Test Paper" in captured.out
        assert export_file.is_file()

        with open(export_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["retrieved_count"] == 1
        assert data["papers"][0]["title"] == "Test Paper"
        assert data["papers"][0]["source"] == "OpenAlex"

    def test_cli_invalid_query_returns_code_2(self, capsys):
        code = main(["   "])
        assert code == 2
        captured = capsys.readouterr()
        assert "Error: Invalid query" in captured.err

    @patch("app.main.search_papers")
    def test_cli_rate_limit_returns_code_1_with_friendly_message(self, mock_search, capsys):
        mock_search.side_effect = OpenAlexRateLimitError("Rate limit reached", retry_after=30.0)
        code = main(["valid query"])
        assert code == 1
        captured = capsys.readouterr()
        assert "[OpenAlex Rate Limit]" in captured.err
