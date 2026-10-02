"""Persistent SQLite evidence storage for ATHENA (Milestone M3).

Stores normalized scientific evidence chunks, full document provenance,
and dense embedding representations using pure standard-library sqlite3.
Guarantees zero external database server requirements and full ACID compliance.
"""

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.evidence.models import (
    EvidenceChunk,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)


class EvidenceStore:
    """ACID-compliant persistent storage for scientific evidence chunks and provenance."""

    def __init__(self, db_path: str | Path = "data/athena_evidence.db") -> None:
        """Initialize EvidenceStore.

        Args:
            db_path: Path to SQLite database file or ':memory:' for transient test stores.
        """
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they do not already exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    paper_title TEXT,
                    publication_year INTEGER,
                    doi TEXT,
                    landing_page_url TEXT,
                    venue TEXT,
                    authors_json TEXT,
                    source_database TEXT
                );

                CREATE TABLE IF NOT EXISTS evidence_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_count INTEGER,
                    word_count INTEGER,
                    chunk_index INTEGER,
                    embedding_json TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON evidence_chunks(document_id);
                """
            )

    def close(self) -> None:
        """Close SQLite database connection."""
        if self._conn:
            self._conn.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def save_source(self, source: EvidenceSource) -> None:
        """Persist an EvidenceSource provenance record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, paper_title, publication_year, doi,
                    landing_page_url, venue, authors_json, source_database
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.openalex_id,
                    source.paper_title,
                    source.publication_year,
                    source.doi,
                    source.landing_page_url,
                    source.venue,
                    json.dumps(list(source.authors), ensure_ascii=False),
                    source.source_database,
                ),
            )

    def save_document(self, doc: NormalizedDocument) -> None:
        """Persist a NormalizedDocument's provenance."""
        self.save_source(doc.to_source())

    def get_source(self, document_id: str) -> EvidenceSource | None:
        """Retrieve an EvidenceSource provenance record by document ID."""
        cursor = self._conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        authors = tuple(json.loads(row["authors_json"]) if row["authors_json"] else [])
        return EvidenceSource(
            openalex_id=row["document_id"],
            paper_title=row["paper_title"],
            publication_year=row["publication_year"],
            doi=row["doi"],
            landing_page_url=row["landing_page_url"],
            venue=row["venue"],
            authors=authors,
            source_database=row["source_database"] or "OpenAlex",
        )

    def save_chunk(self, chunk: EvidenceChunk) -> None:
        """Persist an EvidenceChunk along with its EvidenceSource provenance."""
        self.save_source(chunk.source)
        embedding_json = (
            json.dumps(list(chunk.embedding)) if chunk.embedding is not None else None
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO evidence_chunks (
                    chunk_id, document_id, evidence_type, section,
                    text, char_count, word_count, chunk_index, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.evidence_type.value,
                    chunk.section,
                    chunk.text,
                    chunk.char_count,
                    chunk.word_count,
                    chunk.chunk_index,
                    embedding_json,
                ),
            )

    def save_chunks(self, chunks: Sequence[EvidenceChunk]) -> None:
        """Persist multiple EvidenceChunks in an atomic transaction."""
        for chunk in chunks:
            self.save_chunk(chunk)

    def get_chunk(self, chunk_id: str) -> EvidenceChunk | None:
        """Retrieve a single EvidenceChunk with full provenance by its chunk ID."""
        cursor = self._conn.execute(
            """
            SELECT c.*, d.paper_title, d.publication_year, d.doi,
                   d.landing_page_url, d.venue, d.authors_json, d.source_database
            FROM evidence_chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        authors = tuple(json.loads(row["authors_json"]) if row["authors_json"] else [])
        source = EvidenceSource(
            openalex_id=row["document_id"],
            paper_title=row["paper_title"],
            publication_year=row["publication_year"],
            doi=row["doi"],
            landing_page_url=row["landing_page_url"],
            venue=row["venue"],
            authors=authors,
            source_database=row["source_database"] or "OpenAlex",
        )

        embedding = (
            json.loads(row["embedding_json"]) if row["embedding_json"] is not None else None
        )

        return EvidenceChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            source=source,
            evidence_type=EvidenceType(row["evidence_type"]),
            section=row["section"],
            text=row["text"],
            char_count=row["char_count"],
            word_count=row["word_count"],
            chunk_index=row["chunk_index"],
            embedding=embedding,
        )

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[EvidenceChunk]:
        """Retrieve multiple EvidenceChunks by their chunk IDs."""
        result: list[EvidenceChunk] = []
        for cid in chunk_ids:
            chunk = self.get_chunk(cid)
            if chunk is not None:
                result.append(chunk)
        return result

    def list_all_chunks(self) -> list[EvidenceChunk]:
        """Retrieve all indexed evidence chunks with full provenance."""
        cursor = self._conn.execute("SELECT chunk_id FROM evidence_chunks ORDER BY chunk_id ASC")
        rows = cursor.fetchall()
        return [self.get_chunk(row["chunk_id"]) for row in rows if row["chunk_id"] is not None]  # type: ignore

    def update_chunk_embedding(self, chunk_id: str, embedding: Sequence[float]) -> None:
        """Update or set the dense embedding vector for a chunk."""
        embedding_json = json.dumps(list(embedding))
        with self._conn:
            self._conn.execute(
                "UPDATE evidence_chunks SET embedding_json = ? WHERE chunk_id = ?",
                (embedding_json, chunk_id),
            )

    def count_chunks(self) -> int:
        """Return the total number of stored evidence chunks."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM evidence_chunks")
        return int(cursor.fetchone()[0])

    def count_documents(self) -> int:
        """Return the total number of stored scientific documents."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        return int(cursor.fetchone()[0])
