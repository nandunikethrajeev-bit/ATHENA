"""Deterministic in-memory evidence retrieval engine for ATHENA (Milestone M2).

Uses BM25Okapi lexical relevance scoring implemented entirely in pure Python standard library.
Requires zero external vector databases, zero paid APIs, and zero heavy dependencies.
"""

from collections import Counter
import math
import re
from typing import Sequence

from app.evidence.chunker import chunk_documents
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceType,
    NormalizedDocument,
)

# Standard English stop words to filter out during BM25 tokenization
_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})

# Token pattern: matches alphanumeric terms and hyphenated scientific terms (e.g. p53, covid-19)
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b")


def tokenize(text: str, filter_stopwords: bool = True) -> list[str]:
    """Tokenize scientific text into normalized, lowercased terms.

    Supports both hyphenated compound scientific terms (e.g. 'crispr-cas9')
    and their sub-components ('crispr', 'cas9').

    Args:
        text: Text to tokenize.
        filter_stopwords: If True, filters out standard functional stop words.

    Returns:
        List of cleaned token strings.
    """
    if not text:
        return []

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        tokens.append(token)
        if "-" in token:
            for sub_term in token.split("-"):
                if sub_term and sub_term != token:
                    tokens.append(sub_term)

    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return tokens


class EvidenceIndex:
    """In-memory, deterministic BM25Okapi index for scientific EvidenceChunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize EvidenceIndex.

        Args:
            k1: BM25 term frequency saturation parameter (standard: 1.5).
            b: BM25 document length normalization parameter (standard: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: list[EvidenceChunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_tf: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    @property
    def total_chunks(self) -> int:
        """Return total number of indexed evidence chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Clear all indexed chunks."""
        self.chunks.clear()
        self._doc_tokens.clear()
        self._doc_tf.clear()
        self._doc_lengths.clear()
        self._df.clear()
        self._avgdl = 0.0

    def index_chunks(self, chunks: Sequence[EvidenceChunk]) -> None:
        """Index a collection of EvidenceChunks into the in-memory BM25 store.

        Args:
            chunks: Sequence of EvidenceChunk instances.
        """
        self.clear()
        if not chunks:
            return

        total_length = 0
        for chunk in chunks:
            self.chunks.append(chunk)
            # Tokenize chunk text plus title for higher keyword matching relevance
            combined_text = f"{chunk.source.paper_title or ''} {chunk.text}"
            tokens = tokenize(combined_text)
            self._doc_tokens.append(tokens)

            tf = Counter(tokens)
            self._doc_tf.append(tf)

            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            total_length += doc_len

            # Update document frequency (term present in document)
            for term in tf:
                self._df[term] += 1

        self._avgdl = total_length / len(self.chunks) if self.chunks else 0.0

    def index_documents(self, documents: Sequence[NormalizedDocument]) -> None:
        """Chunk and index a collection of NormalizedDocument instances."""
        chunks = chunk_documents(documents)
        self.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        evidence_types: Sequence[EvidenceType] | None = None,
    ) -> list[EvidenceMatch]:
        """Retrieve and rank evidence chunks relevant to the scientific query.

        Args:
            query: The research query or question.
            top_k: Maximum number of ranked results to return (must be > 0).
            evidence_types: Optional filter to restrict results to specific EvidenceType(s).

        Returns:
            List of EvidenceMatch objects sorted by relevance score descending.
        """
        if not self.chunks or not query or not query.strip() or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.chunks)
        scores: list[float] = [0.0] * n_docs
        matched_terms_per_doc: list[set[str]] = [set() for _ in range(n_docs)]

        # Precompute IDF for query terms
        idf_map: dict[str, float] = {}
        for q_term in set(query_tokens):
            df = self._df.get(q_term, 0)
            if df > 0:
                # Standard smoothed BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                idf_map[q_term] = max(0.0, idf)

        # Score documents
        type_filter = set(evidence_types) if evidence_types is not None else None

        for idx, chunk in enumerate(self.chunks):
            if type_filter is not None and chunk.evidence_type not in type_filter:
                continue

            doc_len = self._doc_lengths[idx]
            tf_map = self._doc_tf[idx]

            # Length normalization denominator
            denom = self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl)) if self._avgdl > 0 else 1.0

            doc_score = 0.0
            for q_term in query_tokens:
                if q_term not in tf_map:
                    continue

                tf = tf_map[q_term]
                idf = idf_map.get(q_term, 0.0)

                term_score = idf * (tf * (self.k1 + 1.0)) / (tf + denom)
                doc_score += term_score
                matched_terms_per_doc[idx].add(q_term)

            scores[idx] = doc_score

        # Collect matches with positive scores
        matches: list[EvidenceMatch] = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                matches.append(
                    EvidenceMatch(
                        chunk=self.chunks[idx],
                        score=score,
                        matched_terms=sorted(matched_terms_per_doc[idx]),
                    )
                )

        # Sort matches by score descending
        matches.sort(key=lambda m: m.score, reverse=True)

        return matches[:top_k]
