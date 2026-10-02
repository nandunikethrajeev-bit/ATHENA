"""Unit tests for ATHENA persistent SQLite evidence storage (Milestone M3).

All tests are deterministic, offline, and require zero external database servers.
"""

from pathlib import Path
import pytest

from app.evidence.models import (
    EvidenceChunk,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)
from app.evidence.store import EvidenceStore


@pytest.fixture
def memory_store():
    """Create an in-memory EvidenceStore for fast, isolated tests."""
    store = EvidenceStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def sample_source():
    return EvidenceSource(
        openalex_id="https://openalex.org/W12345",
        paper_title="CRISPR Gene Editing for Sickle Cell Disease",
        publication_year=2022,
        doi="https://doi.org/10.1056/NEJMoa12345",
        landing_page_url="https://doi.org/10.1056/NEJMoa12345",
        venue="New England Journal of Medicine",
        authors=("Jennifer Doudna", "David Liu"),
        source_database="OpenAlex",
    )


@pytest.fixture
def sample_chunk(sample_source):
    return EvidenceChunk(
        chunk_id="W12345-abs-1",
        document_id="https://openalex.org/W12345",
        source=sample_source,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Targeted genomic disruption of BCL11A erythroid enhancer reactivates fetal hemoglobin production.",
        char_count=98,
        word_count=12,
        chunk_index=1,
    )


class TestEvidenceStore:
    def test_initialization_creates_tables(self, memory_store):
        assert memory_store.count_documents() == 0
        assert memory_store.count_chunks() == 0

    def test_save_and_retrieve_source_provenance(self, memory_store, sample_source):
        memory_store.save_source(sample_source)
        assert memory_store.count_documents() == 1

        retrieved = memory_store.get_source(sample_source.openalex_id)
        assert retrieved is not None
        assert retrieved.openalex_id == sample_source.openalex_id
        assert retrieved.paper_title == sample_source.paper_title
        assert retrieved.publication_year == 2022
        assert retrieved.doi == sample_source.doi
        assert retrieved.landing_page_url == sample_source.landing_page_url
        assert retrieved.venue == "New England Journal of Medicine"
        assert retrieved.authors == ("Jennifer Doudna", "David Liu")
        assert retrieved.source_database == "OpenAlex"

    def test_save_normalized_document(self, memory_store):
        doc = NormalizedDocument(
            document_id="https://openalex.org/W9999",
            title="Deep Learning in Cardiology",
            authors=["Alice Wonder", "Bob Builder"],
            publication_year=2023,
            doi="https://doi.org/10.1016/j.cardio.2023",
            venue="Lancet Cardiology",
        )
        memory_store.save_document(doc)
        retrieved = memory_store.get_source("https://openalex.org/W9999")
        assert retrieved is not None
        assert retrieved.paper_title == "Deep Learning in Cardiology"
        assert retrieved.authors == ("Alice Wonder", "Bob Builder")

    def test_save_and_retrieve_chunk(self, memory_store, sample_chunk):
        memory_store.save_chunk(sample_chunk)

        assert memory_store.count_chunks() == 1
        assert memory_store.count_documents() == 1

        retrieved = memory_store.get_chunk("W12345-abs-1")
        assert retrieved is not None
        assert retrieved.chunk_id == "W12345-abs-1"
        assert retrieved.document_id == "https://openalex.org/W12345"
        assert retrieved.evidence_type == EvidenceType.ABSTRACT
        assert retrieved.section == "abstract"
        assert "reactivates fetal hemoglobin" in retrieved.text
        assert retrieved.char_count == 98
        assert retrieved.word_count == 12
        assert retrieved.chunk_index == 1
        # Provenance integrity
        assert retrieved.source.paper_title == sample_chunk.source.paper_title
        assert retrieved.source.doi == sample_chunk.source.doi

    def test_save_chunks_batch(self, memory_store, sample_source):
        chunk1 = EvidenceChunk(
            chunk_id="W12345-abs-1",
            document_id=sample_source.openalex_id,
            source=sample_source,
            evidence_type=EvidenceType.ABSTRACT,
            section="abstract",
            text="Sentence one text.",
            char_count=18,
            word_count=3,
            chunk_index=1,
        )
        chunk2 = EvidenceChunk(
            chunk_id="W12345-abs-2",
            document_id=sample_source.openalex_id,
            source=sample_source,
            evidence_type=EvidenceType.ABSTRACT,
            section="abstract",
            text="Sentence two text.",
            char_count=18,
            word_count=3,
            chunk_index=2,
        )

        memory_store.save_chunks([chunk1, chunk2])
        assert memory_store.count_chunks() == 2

        all_chunks = memory_store.list_all_chunks()
        assert len(all_chunks) == 2
        assert {c.chunk_id for c in all_chunks} == {"W12345-abs-1", "W12345-abs-2"}

    def test_idempotent_upsert(self, memory_store, sample_chunk):
        memory_store.save_chunk(sample_chunk)
        # Update text on same chunk_id
        updated_chunk = EvidenceChunk(
            chunk_id=sample_chunk.chunk_id,
            document_id=sample_chunk.document_id,
            source=sample_chunk.source,
            evidence_type=sample_chunk.evidence_type,
            section=sample_chunk.section,
            text="Updated scientific content.",
            char_count=27,
            word_count=3,
            chunk_index=1,
        )
        memory_store.save_chunk(updated_chunk)

        assert memory_store.count_chunks() == 1
        retrieved = memory_store.get_chunk("W12345-abs-1")
        assert retrieved.text == "Updated scientific content."

    def test_embedding_storage_and_update(self, memory_store, sample_chunk):
        sample_chunk.embedding = [0.123, 0.456, 0.789]
        memory_store.save_chunk(sample_chunk)

        retrieved = memory_store.get_chunk(sample_chunk.chunk_id)
        assert retrieved.embedding == [0.123, 0.456, 0.789]

        # Update embedding
        memory_store.update_chunk_embedding(sample_chunk.chunk_id, [0.999, 0.888])
        updated = memory_store.get_chunk(sample_chunk.chunk_id)
        assert updated.embedding == [0.999, 0.888]

    def test_get_nonexistent_chunk_returns_none(self, memory_store):
        assert memory_store.get_chunk("NONEXISTENT-999") is None

    def test_file_based_persistence(self, tmp_path, sample_chunk):
        db_file = tmp_path / "test_evidence.db"
        with EvidenceStore(db_file) as store:
            store.save_chunk(sample_chunk)
            assert store.count_chunks() == 1

        # Reopen file and verify persistence
        with EvidenceStore(db_file) as store:
            assert store.count_chunks() == 1
            chunk = store.get_chunk(sample_chunk.chunk_id)
            assert chunk is not None
            assert chunk.text == sample_chunk.text
