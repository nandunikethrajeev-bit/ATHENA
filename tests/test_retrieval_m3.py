"""Unit tests for dense vector indexing, mock embeddings, and hybrid retrieval in ATHENA (Milestone M3).

All tests are deterministic, offline, and require zero external API keys or Ollama servers.
"""

import math
import pytest

from app.evidence.chunker import chunk_document
from app.evidence.embeddings import (
    EmbeddingModelInfo,
    MockEmbeddingClient,
    get_embedding_client,
)
from app.evidence.hybrid import HybridRetriever, compute_rrf_score
from app.evidence.models import (
    EvidenceChunk,
    EvidenceSource,
    EvidenceType,
)
from app.evidence.normalizer import normalize_paper
from app.evidence.retriever import EvidenceIndex
from app.evidence.store import EvidenceStore
from app.evidence.vector_index import VectorIndex, cosine_similarity
from app.retrieval.models import Paper
from app.synthesis.models import ClaimStatus
from app.synthesis.providers import MockLLMClient
from app.synthesis.synthesizer import Synthesizer


@pytest.fixture
def mock_embedder():
    return MockEmbeddingClient(dimension=64)


@pytest.fixture
def sample_sources():
    src1 = EvidenceSource(
        openalex_id="https://openalex.org/W101",
        paper_title="Perovskite Stability and Surface Chemistry",
        publication_year=2023,
        doi="https://doi.org/10.1000/101",
        landing_page_url="https://doi.org/10.1000/101",
        venue="Nature Materials",
        authors=("Alice Smith",),
        source_database="OpenAlex",
    )
    src2 = EvidenceSource(
        openalex_id="https://openalex.org/W102",
        paper_title="Silicon Photovoltaics Manufacturing",
        publication_year=2021,
        doi="https://doi.org/10.1000/102",
        landing_page_url="https://doi.org/10.1000/102",
        venue="Solar Cells",
        authors=("Bob Jones",),
        source_database="OpenAlex",
    )
    return src1, src2


@pytest.fixture
def sample_chunks(sample_sources):
    src1, src2 = sample_sources
    chunk1 = EvidenceChunk(
        chunk_id="W101-abs-1",
        document_id=src1.openalex_id,
        source=src1,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Surface passivation with bulky ammonium halides prevents perovskite phase degradation and moisture penetration.",
        char_count=107,
        word_count=13,
        chunk_index=1,
    )
    chunk2 = EvidenceChunk(
        chunk_id="W102-abs-1",
        document_id=src2.openalex_id,
        source=src2,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Industrial monocrystalline silicon ingot casting yields low defect density wafers at commercial scale.",
        char_count=104,
        word_count=13,
        chunk_index=1,
    )
    return chunk1, chunk2


# ==============================================================================
# 1. Embedding Tests
# ==============================================================================

class TestEmbeddings:
    def test_mock_embedder_deterministic(self, mock_embedder):
        text = "perovskite solar cells"
        vec1 = mock_embedder.embed_text(text)
        vec2 = mock_embedder.embed_text(text)

        assert vec1 == vec2
        assert len(vec1) == 64

    def test_mock_embedder_unit_norm(self, mock_embedder):
        vec = mock_embedder.embed_text("quantum dot lasers")
        magnitude = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(magnitude, rel=1e-5) == 1.0

    def test_mock_embedder_model_info(self, mock_embedder):
        info = mock_embedder.get_model_info()
        assert isinstance(info, EmbeddingModelInfo)
        assert info.dimension == 64
        assert info.name == "mock-hash-embed-v1"
        assert info.version == "1.0.0"

    def test_mock_embedder_batch(self, mock_embedder):
        texts = ["Text A", "Text B"]
        vectors = mock_embedder.embed_batch(texts)
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        assert len(vectors[1]) == 64

    def test_get_embedding_client_factory(self):
        client = get_embedding_client(provider="mock")
        assert isinstance(client, MockEmbeddingClient)

        with pytest.raises(ValueError):
            get_embedding_client(provider="unsupported_xyz")


# ==============================================================================
# 2. Vector Index & Cosine Similarity Tests
# ==============================================================================

class TestVectorIndex:
    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert pytest.approx(cosine_similarity(v, v)) == 1.0

    def test_cosine_similarity_orthogonal_vectors(self):
        u = [1.0, 0.0]
        v = [0.0, 1.0]
        assert pytest.approx(cosine_similarity(u, v)) == 0.0

    def test_cosine_similarity_zero_vector(self):
        u = [0.0, 0.0]
        v = [1.0, 2.0]
        assert cosine_similarity(u, v) == 0.0

    def test_vector_index_add_and_retrieve(self, mock_embedder, sample_chunks):
        c1, c2 = sample_chunks
        index = VectorIndex(embedding_client=mock_embedder)
        index.add_chunks([c1, c2])

        assert index.total_vectors == 2

        # Query strongly matching chunk 1
        matches = index.retrieve_semantic("perovskite passivation", top_k=2)
        assert len(matches) >= 1
        assert matches[0].chunk.chunk_id == "W101-abs-1"
        assert matches[0].retrieval_mode == "semantic"

    def test_vector_index_empty_query(self, mock_embedder, sample_chunks):
        index = VectorIndex(embedding_client=mock_embedder)
        index.add_chunks(sample_chunks)
        assert index.retrieve_semantic("") == []
        assert index.retrieve_semantic(None) == []


# ==============================================================================
# 3. Hybrid Retrieval Tests
# ==============================================================================

class TestHybridRetriever:
    def test_rrf_scoring_math(self):
        # Rank 1 in lex and rank 2 in sem with k=60
        score = compute_rrf_score(rank_lex=1, rank_sem=2, k=60, weight_lex=1.0, weight_sem=1.0)
        expected = (1.0 / 61.0) + (1.0 / 62.0)
        assert pytest.approx(score) == expected

    def test_rrf_single_rank(self):
        score = compute_rrf_score(rank_lex=1, rank_sem=None, k=60)
        assert pytest.approx(score) == 1.0 / 61.0

    def test_hybrid_retriever_modes(self, mock_embedder, sample_chunks):
        c1, c2 = sample_chunks
        retriever = HybridRetriever(embedding_client=mock_embedder)
        retriever.index_chunks([c1, c2])

        # BM25 mode
        bm25_matches = retriever.retrieve("silicon ingot", top_k=1, mode="bm25")
        assert len(bm25_matches) == 1
        assert bm25_matches[0].chunk.chunk_id == "W102-abs-1"

        # Semantic mode
        sem_matches = retriever.retrieve("ammonium halides", top_k=1, mode="semantic")
        assert len(sem_matches) == 1
        assert sem_matches[0].chunk.chunk_id == "W101-abs-1"
        assert sem_matches[0].retrieval_mode == "semantic"

        # Hybrid mode
        hybrid_matches = retriever.retrieve("perovskite degradation", top_k=2, mode="hybrid")
        assert len(hybrid_matches) >= 1
        assert hybrid_matches[0].chunk.chunk_id == "W101-abs-1"
        assert hybrid_matches[0].retrieval_mode == "hybrid"


# ==============================================================================
# 4. End-to-End Pipeline: M1 -> M2 -> M3 Storage -> Hybrid -> Synthesis
# ==============================================================================

class TestEndToEndM3Pipeline:
    def test_m1_m2_storage_hybrid_synthesis_flow(self, mock_embedder):
        # 1. M1: Create Paper
        paper = Paper(
            title="Halide Perovskite Passivation Mechanisms",
            openalex_id="https://openalex.org/W9999",
            source="OpenAlex",
            authors=["Elena Rossi", "Marco Rossi"],
            publication_year=2023,
            doi="https://doi.org/10.1000/9999",
            abstract="Interfacial Lewis base passivation suppresses non-radiative recombination and boosts solar efficiency.",
            venue="JACS",
            cited_by_count=45,
        )

        # 2. M2: Normalize and Chunk
        norm_doc = normalize_paper(paper)
        chunks = chunk_document(norm_doc)
        abs_chunk = next(c for c in chunks if c.evidence_type == EvidenceType.ABSTRACT)

        # 3. M3: Store in persistent SQLite
        store = EvidenceStore(":memory:")
        store.save_chunks(chunks)
        assert store.count_chunks() == len(chunks)

        # Retrieve chunk back from store to confirm persistence
        persisted_chunk = store.get_chunk(abs_chunk.chunk_id)
        assert persisted_chunk is not None
        assert persisted_chunk.source.doi == "https://doi.org/10.1000/9999"

        # 4. M3: Hybrid Indexing & Retrieval
        hybrid_retriever = HybridRetriever(embedding_client=mock_embedder)
        hybrid_retriever.index_chunks([persisted_chunk])

        matches = hybrid_retriever.retrieve(
            query="passivation recombination solar",
            top_k=1,
            mode="hybrid",
        )
        assert len(matches) == 1
        assert matches[0].chunk.chunk_id == abs_chunk.chunk_id
        assert matches[0].retrieval_mode == "hybrid"

        # 5. Synthesis: Feed hybrid matches into Synthesizer
        mock_llm_json = {
            "overview": "Lewis base passivation effectively reduces non-radiative losses in perovskites.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Interfacial Lewis bases suppress recombination centers to boost device performance.",
                    "evidence_ids": [abs_chunk.chunk_id],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Evaluated in laboratory test cells."],
        }
        mock_llm = MockLLMClient(response=mock_llm_json)
        synthesizer = Synthesizer(client=mock_llm)

        synthesis = synthesizer.synthesize(
            research_question="How does passivation impact perovskite solar cells?",
            evidence_context=matches,
        )

        # Verify claims and provenance survive intact
        assert synthesis.validation_report.grounding_score == 1.0
        assert synthesis.key_findings[0].status == ClaimStatus.SUPPORTED
        assert abs_chunk.chunk_id in synthesis.evidence_references
        ref = synthesis.evidence_references[abs_chunk.chunk_id]
        assert ref.paper_title == "Halide Perovskite Passivation Mechanisms"
        assert ref.doi == "https://doi.org/10.1000/9999"
        assert ref.authors == ["Elena Rossi", "Marco Rossi"]
