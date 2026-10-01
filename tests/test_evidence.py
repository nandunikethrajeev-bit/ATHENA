"""Unit tests for ATHENA evidence processing and retrieval subsystem (Milestone M2).

All tests are deterministic, fully offline, and do NOT make network requests.
"""

import json
from unittest.mock import patch
import pytest

from app.evidence.chunker import (
    chunk_document,
    chunk_documents,
    create_metadata_chunk,
    split_scientific_sentences,
)
from app.evidence.cleaner import clean_scientific_text
from app.evidence.context import build_evidence_context, format_evidence_item
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)
from app.evidence.normalizer import normalize_paper
from app.evidence.retriever import EvidenceIndex, tokenize
from app.main import main
from app.retrieval.models import OpenAccessInfo, Paper


# ==============================================================================
# 1. Scientific Text Cleaning Tests
# ==============================================================================

class TestTextCleaning:
    def test_clean_none_and_empty_returns_empty_string(self):
        assert clean_scientific_text(None) == ""
        assert clean_scientific_text("") == ""
        assert clean_scientific_text("   \t  \n  ") == ""

    def test_collapses_horizontal_whitespace(self):
        raw = "Machine    learning   for   clinical    biomarkers."
        assert clean_scientific_text(raw) == "Machine learning for clinical biomarkers."

    def test_normalizes_unicode_punctuation_and_spaces(self):
        raw = "“Targeted”\u00a0therapies—evaluated\u2013with\u2018precision\u2019…"
        cleaned = clean_scientific_text(raw)
        assert cleaned == '"Targeted" therapies-evaluated-with\'precision\'...'

    def test_preserves_paragraph_breaks_but_collapses_excessive_newlines(self):
        raw = "Paragraph one.\n\n\n\n\nParagraph two."
        cleaned = clean_scientific_text(raw)
        assert cleaned == "Paragraph one.\n\nParagraph two."

    def test_preserves_scientific_notations_and_statistics(self):
        raw = "Significant improvement observed (p < 0.001, t = 4.32, 95% CI [1.12, 2.45])."
        cleaned = clean_scientific_text(raw)
        assert cleaned == "Significant improvement observed (p < 0.001, t = 4.32, 95% CI [1.12, 2.45])."

    def test_preserves_chemical_and_measurement_units(self):
        raw = "Dosage was 25 mg/kg with H2O solvent at 37 °C; concentration was 10 μg/mL."
        cleaned = clean_scientific_text(raw)
        assert "25 mg/kg" in cleaned
        assert "H2O" in cleaned
        assert "37" in cleaned
        assert "μg/mL" in cleaned


# ==============================================================================
# 2. Document Normalization Tests
# ==============================================================================

class TestDocumentNormalization:
    @pytest.fixture
    def sample_paper(self):
        return Paper(
            title="Deep learning in Alzheimer's diagnosis  ",
            openalex_id="https://openalex.org/W12345",
            source="OpenAlex",
            authors=["Alice Smith", "Bob Jones"],
            publication_year=2021,
            doi="https://doi.org/10.1000/182",
            abstract="Machine learning identifies neurodegenerative patterns early.",
            venue="Nature Medicine",
            cited_by_count=142,
            publication_type="journal-article",
            landing_page_url="https://doi.org/10.1000/182",
            open_access=OpenAccessInfo(is_oa=True, oa_status="gold", oa_url="https://example.com/oa.pdf"),
        )

    def test_normalize_paper_model(self, sample_paper):
        doc = normalize_paper(sample_paper)

        assert isinstance(doc, NormalizedDocument)
        assert doc.document_id == "https://openalex.org/W12345"
        assert doc.short_id == "W12345"
        assert doc.title == "Deep learning in Alzheimer's diagnosis"
        assert doc.authors == ["Alice Smith", "Bob Jones"]
        assert doc.publication_year == 2021
        assert doc.doi == "https://doi.org/10.1000/182"
        assert doc.venue == "Nature Medicine"
        assert doc.landing_page_url == "https://doi.org/10.1000/182"
        assert doc.abstract_text == "Machine learning identifies neurodegenerative patterns early."
        assert doc.has_abstract is True
        assert doc.has_full_text is False
        assert doc.is_open_access is True
        assert doc.oa_status == "gold"
        assert doc.oa_url == "https://example.com/oa.pdf"

    def test_normalize_raw_dict(self):
        data = {
            "openalex_id": "https://openalex.org/W67890",
            "title": "Predictive Genomics",
            "authors": [{"display_name": "Carol Danvers"}],
            "publication_year": "2023",
            "cited_by_count": "55",
            "abstract": "Genomic sequencing accelerates biomarker discovery.",
        }
        doc = normalize_paper(data)

        assert doc.document_id == "https://openalex.org/W67890"
        assert doc.short_id == "W67890"
        assert doc.title == "Predictive Genomics"
        assert doc.authors == ["Carol Danvers"]
        assert doc.publication_year == 2023
        assert doc.cited_by_count == 55
        assert doc.has_abstract is True

    def test_normalize_minimal_paper_without_abstract(self):
        paper = Paper(title=None, openalex_id="https://openalex.org/W000")
        doc = normalize_paper(paper)

        assert doc.document_id == "https://openalex.org/W000"
        assert doc.short_id == "W000"
        assert doc.title is None
        assert doc.authors == []
        assert doc.abstract_text is None
        assert doc.has_abstract is False
        assert doc.has_full_text is False

    def test_reject_invalid_input_type(self):
        with pytest.raises(TypeError, match="Expected Paper or dict"):
            normalize_paper(["invalid", "list"])  # type: ignore

    def test_document_to_dict_and_to_source(self, sample_paper):
        doc = normalize_paper(sample_paper)
        d = doc.to_dict()
        assert d["document_id"] == "https://openalex.org/W12345"
        assert d["has_abstract"] is True
        assert json.dumps(d)  # Ensure JSON serializable

        source = doc.to_source()
        assert isinstance(source, EvidenceSource)
        assert source.openalex_id == doc.document_id
        assert source.paper_title == doc.title
        assert source.authors == ("Alice Smith", "Bob Jones")


# ==============================================================================
# 3. Scientific Sentence Splitting Tests
# ==============================================================================

class TestScientificSentenceSplitting:
    def test_splits_standard_sentences(self):
        text = "First sentence here. Second sentence starts now! Is this the third?"
        sentences = split_scientific_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence here."
        assert sentences[1] == "Second sentence starts now!"
        assert sentences[2] == "Is this the third?"

    def test_respects_scientific_abbreviations(self):
        text = (
            "Smith et al. conducted a trial on Alzheimer's disease. "
            "The effect was notable (e.g. reduced amyloid burden). "
            "See Fig. 2 for the survival curves."
        )
        sentences = split_scientific_sentences(text)
        assert len(sentences) == 3
        assert "Smith et al. conducted" in sentences[0]
        assert "e.g. reduced" in sentences[1]
        assert "Fig. 2" in sentences[2]

    def test_respects_decimal_numbers(self):
        text = "The p-value was p < 0.05 in group A. Group B reached 3.14 units of baseline."
        sentences = split_scientific_sentences(text)
        assert len(sentences) == 2
        assert "p < 0.05" in sentences[0]
        assert "3.14 units" in sentences[1]

    def test_empty_input(self):
        assert split_scientific_sentences("") == []
        assert split_scientific_sentences("   ") == []


# ==============================================================================
# 4. Evidence Chunking & Provenance Tests
# ==============================================================================

class TestEvidenceChunking:
    def test_chunk_paper_with_abstract(self):
        doc = NormalizedDocument(
            document_id="https://openalex.org/W100",
            title="Machine Learning in Dementia",
            authors=["Dr. Alice"],
            publication_year=2022,
            doi="https://doi.org/10.1000/1",
            venue="Lancet Neurology",
            abstract_text=(
                "Early diagnosis of dementia is crucial for patient outcomes. "
                "We applied random forests to MRI scans of 500 patients. "
                "The classifier achieved 94% accuracy in detecting early stage decline."
            ),
        )
        chunks = chunk_document(doc)

        # Must have 1 metadata chunk and at least 1 abstract chunk
        assert len(chunks) >= 2
        meta_chunk = chunks[0]
        assert meta_chunk.evidence_type == EvidenceType.METADATA
        assert meta_chunk.chunk_id == "W100-meta-0"
        assert meta_chunk.section == "metadata"
        assert "Machine Learning in Dementia" in meta_chunk.text
        assert "Dr. Alice" in meta_chunk.text

        # Abstract chunks
        abs_chunk = chunks[1]
        assert abs_chunk.evidence_type == EvidenceType.ABSTRACT
        assert abs_chunk.chunk_id == "W100-abs-1"
        assert abs_chunk.section == "abstract"
        assert "Early diagnosis" in abs_chunk.text

        # NEVER classify abstract as full text
        for chunk in chunks:
            assert chunk.evidence_type != EvidenceType.FULL_TEXT

    def test_chunk_paper_without_abstract_never_fabricates_abstract(self):
        doc = NormalizedDocument(
            document_id="https://openalex.org/W200",
            title="Closed Access Neurogenetics Study",
            authors=["Bob Miller"],
            publication_year=2019,
            doi="https://doi.org/10.1000/2",
            venue="Science",
            abstract_text=None,
        )
        chunks = chunk_document(doc)

        # Must only produce 1 metadata chunk, NO abstract chunks
        assert len(chunks) == 1
        assert chunks[0].evidence_type == EvidenceType.METADATA
        assert chunks[0].chunk_id == "W200-meta-0"
        assert "Abstract text is not indexed" in chunks[0].text

    def test_provenance_preserved_on_all_chunks(self):
        doc = NormalizedDocument(
            document_id="https://openalex.org/W300",
            title="Provenance Verification Test",
            authors=["Scientist One", "Scientist Two"],
            publication_year=2024,
            doi="https://doi.org/10.1000/3",
            landing_page_url="https://nature.com/articles/123",
            venue="Nature",
            abstract_text="A single sentence abstract for provenance tracking.",
        )
        chunks = chunk_document(doc)

        for chunk in chunks:
            src = chunk.source
            assert src.openalex_id == "https://openalex.org/W300"
            assert src.paper_title == "Provenance Verification Test"
            assert src.publication_year == 2024
            assert src.doi == "https://doi.org/10.1000/3"
            assert src.landing_page_url == "https://nature.com/articles/123"
            assert src.venue == "Nature"
            assert src.authors == ("Scientist One", "Scientist Two")

            # Check JSON serialization
            serialized = chunk.to_dict()
            assert serialized["document_id"] == "https://openalex.org/W300"
            assert serialized["source"]["openalex_id"] == "https://openalex.org/W300"
            assert json.dumps(serialized)

    def test_chunk_documents_multiple(self):
        doc1 = NormalizedDocument(document_id="https://openalex.org/W1", title="Paper 1", abstract_text="Abs 1.")
        doc2 = NormalizedDocument(document_id="https://openalex.org/W2", title="Paper 2", abstract_text=None)
        all_chunks = chunk_documents([doc1, doc2])

        assert len(all_chunks) == 3  # (doc1 meta + doc1 abs) + (doc2 meta)


# ==============================================================================
# 5. In-Memory Evidence Retrieval & BM25 Ranking Tests
# ==============================================================================

class TestEvidenceRetrieval:
    @pytest.fixture
    def index_with_corpus(self):
        doc_ad = NormalizedDocument(
            document_id="https://openalex.org/W_AD",
            title="Early Biomarkers of Alzheimer's Disease Using Machine Learning",
            abstract_text=(
                "Machine learning algorithms improve the early detection of Alzheimer's disease "
                "by analyzing cognitive decline and amyloid beta biomarkers in cerebrospinal fluid."
            ),
            publication_year=2022,
        )
        doc_crispr = NormalizedDocument(
            document_id="https://openalex.org/W_CRISPR",
            title="CRISPR-Cas9 Gene Editing in Crop Agriculture",
            abstract_text=(
                "Targeted gene editing utilizing CRISPR-Cas9 endonuclease creates drought-resistant "
                "wheat and rice cultivars without exogenous transgene insertion."
            ),
            publication_year=2021,
        )
        doc_cardio = NormalizedDocument(
            document_id="https://openalex.org/W_CARDIO",
            title="Cardiac Arrhythmia Classification via ECG Waveforms",
            abstract_text=(
                "Convolutional neural networks detect atrial fibrillation and cardiac arrhythmias "
                "from standard 12-lead electrocardiogram signals."
            ),
            publication_year=2023,
        )

        index = EvidenceIndex()
        index.index_documents([doc_ad, doc_crispr, doc_cardio])
        return index

    def test_bm25_retrieval_returns_relevant_chunks(self, index_with_corpus):
        matches = index_with_corpus.retrieve("machine learning Alzheimer detection", top_k=3)

        assert len(matches) > 0
        top_match = matches[0]
        assert top_match.chunk.document_id == "https://openalex.org/W_AD"
        assert top_match.score > 0.0
        assert "alzheimer" in top_match.matched_terms
        assert "machine" in top_match.matched_terms

    def test_crispr_query_retrieves_crispr_paper(self, index_with_corpus):
        matches = index_with_corpus.retrieve("CRISPR gene editing wheat cultivars", top_k=2)
        assert len(matches) > 0
        top_match = matches[0]
        assert top_match.chunk.document_id == "https://openalex.org/W_CRISPR"
        assert "crispr" in top_match.matched_terms

    def test_filter_by_evidence_type(self, index_with_corpus):
        # Retrieve only ABSTRACT evidence
        matches = index_with_corpus.retrieve(
            "Alzheimer biomarkers",
            top_k=5,
            evidence_types=[EvidenceType.ABSTRACT],
        )
        for m in matches:
            assert m.chunk.evidence_type == EvidenceType.ABSTRACT

        # Retrieve only METADATA evidence
        matches_meta = index_with_corpus.retrieve(
            "Alzheimer biomarkers",
            top_k=5,
            evidence_types=[EvidenceType.METADATA],
        )
        for m in matches_meta:
            assert m.chunk.evidence_type == EvidenceType.METADATA

    def test_empty_query_returns_empty_list(self, index_with_corpus):
        assert index_with_corpus.retrieve("", top_k=5) == []
        assert index_with_corpus.retrieve("   ", top_k=5) == []

    def test_unmatched_query_returns_empty_list(self, index_with_corpus):
        assert index_with_corpus.retrieve("quantum astrophysics gravitation blackhole", top_k=5) == []

    def test_empty_index_returns_empty_list(self):
        empty_index = EvidenceIndex()
        assert empty_index.retrieve("Alzheimer", top_k=5) == []

    def test_tokenize_filters_stopwords(self):
        tokens = tokenize("This is a study of the Alzheimer disease with machine learning.")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "of" not in tokens
        assert "alzheimer" in tokens
        assert "disease" in tokens
        assert "machine" in tokens
        assert "learning" in tokens


# ==============================================================================
# 6. Evidence Context Assembly Tests
# ==============================================================================

class TestEvidenceContext:
    def test_format_evidence_item_contains_provenance_and_type(self):
        source = EvidenceSource(
            openalex_id="https://openalex.org/W999",
            paper_title="Biomarker Validation Study",
            publication_year=2023,
            doi="https://doi.org/10.1000/bm",
            landing_page_url="https://doi.org/10.1000/bm",
            venue="JAMA",
            authors=("Alice", "Bob"),
        )
        chunk = EvidenceChunk(
            chunk_id="W999-abs-1",
            document_id="https://openalex.org/W999",
            source=source,
            evidence_type=EvidenceType.ABSTRACT,
            section="abstract",
            text="Biomarkers showed strong diagnostic efficacy.",
            char_count=45,
            word_count=5,
            chunk_index=1,
        )
        match = EvidenceMatch(chunk=chunk, score=2.541, matched_terms=["biomarkers"])

        formatted = format_evidence_item(1, match)
        assert "[1] EVIDENCE ITEM: W999-abs-1" in formatted
        assert "Paper:           Biomarker Validation Study" in formatted
        assert "OpenAlex ID:     https://openalex.org/W999" in formatted
        assert "DOI:             https://doi.org/10.1000/bm" in formatted
        assert "Evidence Type:   ABSTRACT" in formatted
        assert "Relevance Score: 2.541" in formatted
        assert "Biomarkers showed strong diagnostic efficacy." in formatted
        assert "Integrity Note: Evidence sourced from peer-reviewed abstract" in formatted

    def test_build_evidence_context_with_matches(self):
        source = EvidenceSource(
            openalex_id="https://openalex.org/W10",
            paper_title="Study 10",
            publication_year=2020,
            doi=None,
            landing_page_url=None,
        )
        chunk = EvidenceChunk(
            chunk_id="W10-meta-0",
            document_id="https://openalex.org/W10",
            source=source,
            evidence_type=EvidenceType.METADATA,
            section="metadata",
            text="Metadata summary text.",
            char_count=22,
            word_count=3,
            chunk_index=0,
        )
        match = EvidenceMatch(chunk=chunk, score=1.12, matched_terms=["study"])

        context = build_evidence_context("study query", [match], total_papers=1, total_chunks=1)
        assert "ATHENA Evidence Context (Milestone M2)" in context
        assert "Query: 'study query'" in context
        assert "Retrieved 1 relevant evidence chunk(s)" in context
        assert "[1] EVIDENCE ITEM: W10-meta-0" in context

    def test_build_evidence_context_empty(self):
        context = build_evidence_context("unmatched query", [])
        assert "No relevant evidence chunks matched" in context


# ==============================================================================
# 7. CLI Integration with M2 Flag Tests
# ==============================================================================

class TestCLIWithM2:
    @patch("app.main.search_papers")
    def test_cli_extract_evidence_flag_runs_end_to_end(self, mock_search, tmp_path, capsys):
        mock_paper = Paper(
            title="Early Detection of Alzheimer's",
            openalex_id="https://openalex.org/W777",
            authors=["Alice Wonder"],
            publication_year=2022,
            doi="https://doi.org/10.1000/777",
            abstract="Machine learning algorithms detect Alzheimer's markers early.",
            venue="Lancet",
            cited_by_count=25,
        )
        mock_search.return_value = [mock_paper]

        export_file = tmp_path / "m2_results.json"
        code = main([
            "machine learning Alzheimer",
            "--extract-evidence",
            "--top-k", "3",
            "--export", str(export_file),
        ])

        assert code == 0
        captured = capsys.readouterr()
        assert "Retrieved 1 paper(s) from OpenAlex" in captured.out
        assert "[M2] Normalizing documents and extracting evidence chunks" in captured.out
        assert "ATHENA Evidence Context (Milestone M2)" in captured.out
        assert "W777-abs-1" in captured.out or "W777-meta-0" in captured.out

        # Verify export data contains evidence chunks
        assert export_file.is_file()
        with open(export_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "evidence_chunks" in data
        assert "retrieved_matches" in data
        assert len(data["retrieved_matches"]) > 0
