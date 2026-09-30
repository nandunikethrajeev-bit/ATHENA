"""Domain exceptions for the ATHENA retrieval module."""

from typing import Any


class RetrievalError(Exception):
    """Base exception for all retrieval operations in ATHENA."""


class QueryValidationError(RetrievalError, ValueError):
    """Raised when a research query or query parameter is invalid."""


class OpenAlexAPIError(RetrievalError):
    """Raised when the OpenAlex API returns an unexpected or error response."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class OpenAlexRateLimitError(OpenAlexAPIError):
    """Raised when OpenAlex returns HTTP 429 Too Many Requests."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        retry_after: float | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.retry_after = retry_after


class OpenAlexNetworkError(RetrievalError):
    """Raised when a network-level failure (e.g. timeout, connection error) occurs."""
