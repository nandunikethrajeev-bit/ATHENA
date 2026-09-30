"""Literature retrieval subsystem for ATHENA (Milestone M1)."""

from app.retrieval.exceptions import (
    OpenAlexAPIError,
    OpenAlexNetworkError,
    OpenAlexRateLimitError,
    RetrievalError,
    QueryValidationError,
)
from app.retrieval.models import OpenAccessInfo, Paper
from app.retrieval.openalex import OpenAlexClient, search_papers
from app.retrieval.utils import (
    DEFAULT_MAX_RESULTS,
    MAX_ALLOWED_RESULTS,
    reconstruct_abstract,
    validate_query,
)

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "MAX_ALLOWED_RESULTS",
    "OpenAccessInfo",
    "OpenAlexAPIError",
    "OpenAlexClient",
    "OpenAlexNetworkError",
    "OpenAlexRateLimitError",
    "Paper",
    "QueryValidationError",
    "RetrievalError",
    "reconstruct_abstract",
    "search_papers",
    "validate_query",
]
