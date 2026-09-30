"""OpenAlex literature retrieval client and parser for ATHENA."""

import os
from typing import Any

import httpx

from app import __version__
from app.retrieval.exceptions import (
    OpenAlexAPIError,
    OpenAlexNetworkError,
    OpenAlexRateLimitError,
)
from app.retrieval.models import OpenAccessInfo, Paper
from app.retrieval.utils import (
    DEFAULT_MAX_RESULTS,
    reconstruct_abstract,
    validate_query,
)

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"


class OpenAlexClient:
    """Client for retrieving scholarly works metadata from OpenAlex REST API."""

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        base_url: str = OPENALEX_WORKS_ENDPOINT,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize OpenAlexClient.

        Args:
            email: Contact email for the polite pool (defaults to OPENALEX_EMAIL env var).
            api_key: OpenAlex API key (defaults to OPENALEX_API_KEY env var).
            base_url: API endpoint for works retrieval.
            timeout: HTTP request timeout in seconds.
            client: Optional preconfigured httpx.Client (primarily for testing and mocking).
        """
        self.email = email or os.environ.get("OPENALEX_EMAIL")
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self._external_client = client

    def _build_headers(self) -> dict[str, str]:
        """Construct request headers including polite User-Agent and optional auth."""
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if self.email:
            headers["User-Agent"] = f"ATHENA/{__version__} (https://github.com/athena-project; mailto:{self.email})"
        else:
            headers["User-Agent"] = f"ATHENA/{__version__} (https://github.com/athena-project)"

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def parse_work(self, work: dict[str, Any]) -> Paper:
        """Convert a single OpenAlex work record dictionary into the canonical Paper model.

        Missing or unavailable fields are kept as None rather than fabricated.

        Args:
            work: Dictionary of work metadata from OpenAlex results.

        Returns:
            A structured Paper instance.
        """
        raw_id = work.get("id")
        openalex_id = str(raw_id) if raw_id is not None else ""

        # Parse title
        raw_title = work.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else None

        # Parse authors
        authors: list[str] = []
        raw_authorships = work.get("authorships")
        if isinstance(raw_authorships, list):
            for authorship in raw_authorships:
                if isinstance(authorship, dict):
                    author_obj = authorship.get("author")
                    if isinstance(author_obj, dict):
                        author_name = author_obj.get("display_name")
                        if isinstance(author_name, str) and author_name.strip():
                            authors.append(author_name.strip())

        # Publication year
        raw_year = work.get("publication_year")
        publication_year = int(raw_year) if isinstance(raw_year, int) else None

        # DOI
        raw_doi = work.get("doi")
        doi = str(raw_doi).strip() if isinstance(raw_doi, str) else None

        # Reconstructed abstract
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        # Venue / Journal extraction
        venue: str | None = None
        primary_loc = work.get("primary_location")
        if isinstance(primary_loc, dict):
            source = primary_loc.get("source")
            if isinstance(source, dict):
                v_name = source.get("display_name")
                if isinstance(v_name, str) and v_name.strip():
                    venue = v_name.strip()

        if venue is None:
            # Fallback 1: check host_venue (legacy OpenAlex field)
            host_venue = work.get("host_venue")
            if isinstance(host_venue, dict):
                v_name = host_venue.get("display_name")
                if isinstance(v_name, str) and v_name.strip():
                    venue = v_name.strip()

        if venue is None:
            # Fallback 2: search alternative locations
            locations = work.get("locations")
            if isinstance(locations, list):
                for loc in locations:
                    if isinstance(loc, dict):
                        source = loc.get("source")
                        if isinstance(source, dict):
                            v_name = source.get("display_name")
                            if isinstance(v_name, str) and v_name.strip():
                                venue = v_name.strip()
                                break

        # Cited by count
        raw_citations = work.get("cited_by_count")
        cited_by_count = int(raw_citations) if isinstance(raw_citations, int) else None

        # Publication type
        raw_type = work.get("type")
        publication_type = str(raw_type).strip() if isinstance(raw_type, str) else None

        # Landing page URL
        landing_page_url: str | None = None
        if isinstance(primary_loc, dict):
            raw_url = primary_loc.get("landing_page_url")
            if isinstance(raw_url, str) and raw_url.strip():
                landing_page_url = raw_url.strip()

        if landing_page_url is None:
            landing_page_url = doi or openalex_id or None

        # Open access information
        open_access: OpenAccessInfo | None = None
        raw_oa = work.get("open_access")
        if isinstance(raw_oa, dict):
            open_access = OpenAccessInfo(
                is_oa=raw_oa.get("is_oa") if isinstance(raw_oa.get("is_oa"), bool) else None,
                oa_status=str(raw_oa.get("oa_status")) if raw_oa.get("oa_status") is not None else None,
                oa_url=str(raw_oa.get("oa_url")) if raw_oa.get("oa_url") is not None else None,
            )

        return Paper(
            title=title,
            openalex_id=openalex_id,
            source="OpenAlex",
            authors=authors,
            publication_year=publication_year,
            doi=doi,
            abstract=abstract,
            venue=venue,
            cited_by_count=cited_by_count,
            publication_type=publication_type,
            landing_page_url=landing_page_url,
            open_access=open_access,
            raw_id=str(raw_id) if raw_id is not None else None,
        )

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
        """Search OpenAlex works for the given scientific research question or query.

        Args:
            query: The research question or query terms to search.
            max_results: Maximum number of works to return (safe limit 1-50).

        Returns:
            A list of Paper models representing retrieved scientific works.

        Raises:
            QueryValidationError: If query or max_results is invalid.
            OpenAlexRateLimitError: If OpenAlex returns HTTP 429.
            OpenAlexAPIError: If OpenAlex returns an HTTP error or malformed response.
            OpenAlexNetworkError: If a connection error or timeout occurs.
        """
        cleaned_query, validated_max = validate_query(query, max_results)

        params: dict[str, Any] = {
            "search": cleaned_query,
            "per_page": validated_max,
        }
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        headers = self._build_headers()

        try:
            if self._external_client is not None:
                response = self._external_client.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(
                        self.base_url,
                        params=params,
                        headers=headers,
                    )
        except httpx.TimeoutException as exc:
            raise OpenAlexNetworkError(
                f"Connection to OpenAlex timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise OpenAlexNetworkError(
                f"Failed to communicate with OpenAlex API: {exc}"
            ) from exc

        # Handle rate limiting (429)
        if response.status_code == 429:
            retry_after: float | None = None
            raw_retry = response.headers.get("retry-after")
            if raw_retry:
                try:
                    retry_after = float(raw_retry)
                except ValueError:
                    retry_after = None

            body_data: Any = None
            try:
                body_data = response.json()
                if retry_after is None and isinstance(body_data, dict):
                    body_retry = body_data.get("retryAfter")
                    if isinstance(body_retry, (int, float)):
                        retry_after = float(body_retry)
            except Exception:
                body_data = response.text

            msg = "OpenAlex rate limit exceeded."
            if retry_after is not None:
                msg += f" Retry after {retry_after} seconds."
            msg += (
                " Tip: Set the OPENALEX_API_KEY environment variable for free, "
                "uninterrupted API access (https://openalex.org/settings/api)."
            )

            raise OpenAlexRateLimitError(
                message=msg,
                status_code=429,
                retry_after=retry_after,
                response_body=body_data,
            )

        # Handle other HTTP error status codes
        if response.status_code >= 400:
            error_detail = response.text
            try:
                err_json = response.json()
                if isinstance(err_json, dict) and "message" in err_json:
                    error_detail = err_json["message"]
            except Exception:
                pass
            raise OpenAlexAPIError(
                f"OpenAlex API returned HTTP {response.status_code}: {error_detail}",
                status_code=response.status_code,
                response_body=response.text,
            )

        # Parse JSON response
        try:
            payload = response.json()
        except Exception as exc:
            raise OpenAlexAPIError(
                f"Malformed JSON response received from OpenAlex: {exc}",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc

        if not isinstance(payload, dict):
            raise OpenAlexAPIError(
                "Unexpected response structure from OpenAlex (expected JSON object).",
                status_code=response.status_code,
                response_body=payload,
            )

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            # Empty results or unexpected results field
            return []

        papers: list[Paper] = []
        for item in raw_results:
            if isinstance(item, dict):
                papers.append(self.parse_work(item))

        return papers


def search_papers(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    email: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> list[Paper]:
    """Retrieve scientific literature relevant to a query from OpenAlex.

    Convenience function that instantiates OpenAlexClient and executes the search.

    Args:
        query: Research question or topic string.
        max_results: Maximum number of papers to retrieve (default 10, safe upper bound 50).
        email: Contact email for OpenAlex polite pool.
        api_key: OpenAlex API key.
        timeout: Request timeout in seconds.

    Returns:
        A list of canonical Paper models.
    """
    client = OpenAlexClient(email=email, api_key=api_key, timeout=timeout)
    return client.search(query=query, max_results=max_results)
