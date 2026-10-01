"""Scientific evidence chunking and sentence boundary detection for ATHENA (Milestone M2).

Splits normalized scientific documents into discrete, verifiable EvidenceChunk objects
while preserving strict provenance and explicitly distinguishing evidence classes
(metadata vs. abstract vs. full-text).
"""

import re
from typing import Sequence

from app.evidence.models import (
    EvidenceChunk,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)

# Common scientific and scholarly abbreviations that should NOT cause sentence breaks
_SCIENTIFIC_ABBREVIATIONS = (
    "et al",
    "e.g",
    "i.e",
    "fig",
    "figs",
    "tab",
    "ref",
    "refs",
    "vs",
    "vol",
    "no",
    "p",
    "pp",
    "approx",
    "ca",
    "dr",
    "prof",
    "ed",
    "eds",
    "eq",
    "eqs",
    "al",
)

# Regex to safely protect abbreviations before sentence splitting
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(abbr) for abbr in _SCIENTIFIC_ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
_ABBREV_TOKEN = "@@DOT@@"

# Regex for numbers with decimals (e.g. 0.05, 3.14)
_DECIMAL_PATTERN = re.compile(r"(\d+)\.(\d+)")
_DECIMAL_TOKEN = r"\1@@DECDOT@@\2"

# Sentence boundary delimiter: period, exclamation, or question mark followed by whitespace and an uppercase letter or quote
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“0-9])')


def split_scientific_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting scientific abbreviations and decimal numbers.

    Args:
        text: Input scientific paragraph or abstract.

    Returns:
        List of trimmed, non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    # 1. Protect decimal numbers: 0.05 -> 0@@DECDOT@@05
    protected = _DECIMAL_PATTERN.sub(_DECIMAL_TOKEN, text)

    # 2. Protect scientific abbreviations: e.g. -> e@@DOT@@g@@DOT@@
    # We replace abbreviation periods iteratively
    def _mask_abbrev(match: re.Match) -> str:
        return match.group(1) + _ABBREV_TOKEN

    protected = _ABBREV_PATTERN.sub(_mask_abbrev, protected)

    # 3. Split on sentence boundaries
    raw_sentences = _SENTENCE_SPLIT_RE.split(protected)

    # 4. Restore masked dots and clean up
    sentences: list[str] = []
    for s in raw_sentences:
        restored = s.replace(_ABBREV_TOKEN, ".").replace("@@DECDOT@@", ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def create_metadata_chunk(doc: NormalizedDocument, source: EvidenceSource) -> EvidenceChunk:
    """Construct an explicit metadata evidence chunk for a normalized document.

    Ensures papers without indexed abstracts still provide traceable bibliographic evidence.
    """
    short_id = doc.short_id or "doc"
    chunk_id = f"{short_id}-meta-0"

    authors_str = ", ".join(doc.authors) if doc.authors else "Unknown"
    year_str = str(doc.publication_year) if doc.publication_year is not None else "N/A"
    venue_str = doc.venue or "Unknown venue"
    doi_str = doc.doi or "N/A"
    citations_str = str(doc.cited_by_count) if doc.cited_by_count is not None else "N/A"
    oa_str = "Open Access" if doc.is_open_access else "Closed / Subscription"

    lines = [
        f"Title: {doc.title or 'Untitled'}",
        f"Authors: {authors_str}",
        f"Publication: {venue_str} ({year_str})",
        f"DOI: {doi_str} | Citations: {citations_str} | Access: {oa_str}",
    ]

    if not doc.has_abstract:
        lines.append("[Note: Abstract text is not indexed in the source repository for this work.]")

    text = "\n".join(lines)
    words = text.split()

    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=doc.document_id,
        source=source,
        evidence_type=EvidenceType.METADATA,
        section="metadata",
        text=text,
        char_count=len(text),
        word_count=len(words),
        chunk_index=0,
    )


def chunk_document(
    doc: NormalizedDocument,
    max_words_per_chunk: int = 120,
    min_words_per_chunk: int = 25,
) -> list[EvidenceChunk]:
    """Extract structured, traceable evidence chunks from a NormalizedDocument.

    Strategy:
    1. Always creates an initial EvidenceType.METADATA chunk capturing paper provenance.
    2. If the document has an abstract:
       - Splits abstract into scientific sentences.
       - Groups sentences into cohesive chunks (~50-120 words).
       - Classifies chunks strictly as EvidenceType.ABSTRACT.
       - NEVER classifies an abstract as full-text.
    3. If the document contains full-text body sections:
       - Chunks each section and classifies as EvidenceType.FULL_TEXT.

    Args:
        doc: The NormalizedDocument to chunk.
        max_words_per_chunk: Upper target word count for a chunk before starting a new one.
        min_words_per_chunk: Minimum word threshold to avoid tiny orphan chunks.

    Returns:
        A list of EvidenceChunk instances.
    """
    chunks: list[EvidenceChunk] = []
    source = doc.to_source()
    short_id = doc.short_id or "doc"

    # 1. Metadata Chunk
    chunks.append(create_metadata_chunk(doc, source))

    # 2. Abstract Chunks
    if doc.has_abstract and doc.abstract_text:
        sentences = split_scientific_sentences(doc.abstract_text)
        current_chunk_sentences: list[str] = []
        current_word_count = 0
        abs_index = 1

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > max_words_per_chunk and current_word_count >= min_words_per_chunk:
                # Flush current chunk
                chunk_text = " ".join(current_chunk_sentences).strip()
                words = chunk_text.split()
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{short_id}-abs-{abs_index}",
                        document_id=doc.document_id,
                        source=source,
                        evidence_type=EvidenceType.ABSTRACT,
                        section="abstract",
                        text=chunk_text,
                        char_count=len(chunk_text),
                        word_count=len(words),
                        chunk_index=abs_index,
                    )
                )
                abs_index += 1
                current_chunk_sentences = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_words

        # Flush remaining sentences in abstract
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            words = chunk_text.split()
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{short_id}-abs-{abs_index}",
                    document_id=doc.document_id,
                    source=source,
                    evidence_type=EvidenceType.ABSTRACT,
                    section="abstract",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(words),
                    chunk_index=abs_index,
                )
            )

    # 3. Full-text Chunks (extensible interface for when full text is parsed)
    if doc.has_full_text:
        full_text_index = 1
        for sec_name, sec_text in doc.sections.items():
            sentences = split_scientific_sentences(sec_text)
            current_sentences: list[str] = []
            current_words = 0

            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > max_words_per_chunk and current_words >= min_words_per_chunk:
                    chunk_text = " ".join(current_sentences).strip()
                    words = chunk_text.split()
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{short_id}-{sec_name}-{full_text_index}",
                            document_id=doc.document_id,
                            source=source,
                            evidence_type=EvidenceType.FULL_TEXT,
                            section=sec_name,
                            text=chunk_text,
                            char_count=len(chunk_text),
                            word_count=len(words),
                            chunk_index=full_text_index,
                        )
                    )
                    full_text_index += 1
                    current_sentences = [sentence]
                    current_words = s_words
                else:
                    current_sentences.append(sentence)
                    current_words += s_words

            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                words = chunk_text.split()
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{short_id}-{sec_name}-{full_text_index}",
                        document_id=doc.document_id,
                        source=source,
                        evidence_type=EvidenceType.FULL_TEXT,
                        section=sec_name,
                        text=chunk_text,
                        char_count=len(chunk_text),
                        word_count=len(words),
                        chunk_index=full_text_index,
                    )
                )
                full_text_index += 1

    return chunks


def chunk_documents(documents: Sequence[NormalizedDocument]) -> list[EvidenceChunk]:
    """Chunk multiple NormalizedDocument instances into a flat list of EvidenceChunks."""
    all_chunks: list[EvidenceChunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
