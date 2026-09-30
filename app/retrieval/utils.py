"""Utility functions for literature retrieval in ATHENA."""

from typing import Any

from app.retrieval.exceptions import QueryValidationError

DEFAULT_MAX_RESULTS = 10
MAX_ALLOWED_RESULTS = 50


def reconstruct_abstract(inverted_index: dict[str, Any] | None) -> str | None:
    """Reconstruct human-readable abstract from an OpenAlex inverted index structure.

    OpenAlex represents paper abstracts as a dictionary where keys are words and
    values are lists of integer token positions in the document, e.g.:
        {"Machine": [0], "learning": [1], "improves": [2]}

    If no abstract is available, or if the inverted index is empty or invalid,
    this function returns None. No text is generated using an LLM.

    Args:
        inverted_index: Mapping of word tokens to lists of integer position indices.

    Returns:
        The reconstructed text string, or None if unavailable.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None

    # Collect valid (position, word) pairs
    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str):
            continue
        if not isinstance(positions, (list, tuple)):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                positioned_words.append((pos, word))

    if not positioned_words:
        return None

    # Sort tokens deterministically by token position
    positioned_words.sort(key=lambda item: item[0])

    # Join reconstructed tokens
    reconstructed = " ".join(word for _, word in positioned_words).strip()
    return reconstructed if reconstructed else None


def validate_query(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_allowed: int = MAX_ALLOWED_RESULTS,
) -> tuple[str, int]:
    """Validate and normalize a user research query and result count.

    Args:
        query: The raw search string or research question.
        max_results: The requested number of papers to retrieve.
        max_allowed: Safe upper ceiling to prevent unbounded queries.

    Returns:
        A tuple of (cleaned_query, validated_max_results).

    Raises:
        QueryValidationError: If query is empty, whitespace-only, not a string,
            or if max_results is invalid or exceeds safe limits.
    """
    if not isinstance(query, str):
        raise QueryValidationError("Query must be a string.")

    # Remove wildcard characters (? and *) that cause OpenAlex HTTP 400 in stemmed search,
    # and normalize whitespace. Research questions frequently end with '?'.
    sanitized = query.replace("?", " ").replace("*", " ")
    cleaned_query = " ".join(sanitized.split())

    if not cleaned_query:
        raise QueryValidationError("Query must not be empty or contain only whitespace.")

    # Guard against booleans since bool is a subclass of int in Python (e.g. True == 1)
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise QueryValidationError("max_results must be an integer.")

    if max_results <= 0:
        raise QueryValidationError("max_results must be a positive integer greater than 0.")

    if max_results > max_allowed:
        raise QueryValidationError(
            f"max_results ({max_results}) exceeds the safe maximum limit of {max_allowed}."
        )

    return cleaned_query, max_results
