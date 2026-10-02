"""Comprehensive offline tests for ATHENA evidence synthesis (Milestone M3).

All tests are deterministic, offline, and require zero network access or paid API keys.
"""

import json
import os
from unittest.mock import patch
import pytest

from app.evidence.chunker import chunk_document
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceSource,
    EvidenceType,
    NormalizedDocument,
)
from app.evidence.normalizer import normalize_paper
from app.main import main
from app.retrieval.models import OpenAccessInfo, Paper
from app.synthesis.models import (
    ClaimStatus,
    EvidenceContext,
    EvidenceReference,
    ResearchSynthesis,
    SynthesizedClaim,
    ValidationReport,
)
from app.synthesis.prompts import (
    SYSTEM_PROMPT,
    build_synthesis_prompt,
    format_evidence_block,
)
from app.synthesis.providers import (
    LLMConfigurationError,
    LLMProviderError,
    MockLLMClient,
    OpenAILikeClient,
    get_llm_client,
)
from app.synthesis.synthesizer import (
    SynthesisParsingError,
    Synthesizer,
    extract_json_payload,
)
from app.synthesis.validators import validate_synthesis


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_chunk_1():
    src = EvidenceSource(
        openalex_id="https://openalex.org/W1001",
        paper_title="Perovskite Solar Cell Passivation",
        publication_year=2023,
        doi="https://doi.org/10.1000/1001",
        landing_page_url="https://doi.org/10.1000/1001",
        venue="Energy & Environmental Science",
        authors=("Alice Smith", "Bob Jones"),
        source_database="OpenAlex",
    )
    return EvidenceChunk(
        chunk_id="W1001-abs-1",
        document_id="https://openalex.org/W1001",
        source=src,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Surface passivation with 2D perovskite layers improves power conversion efficiency to 25.2%.",
        char_count=92,
        word_count=12,
        chunk_index=1,
    )


@pytest.fixture
def sample_chunk_2():
    src = EvidenceSource(
        openalex_id="https://openalex.org/W1002",
        paper_title="Degradation Pathways in Perovskites",
        publication_year=2022,
        doi="https://doi.org/10.1000/1002",
        landing_page_url="https://doi.org/10.1000/1002",
        venue="Nature Energy",
        authors=("Charlie Brown", "Diana Prince"),
        source_database="OpenAlex",
    )
    return EvidenceChunk(
        chunk_id="W1002-abs-1",
        document_id="https://openalex.org/W1002",
        source=src,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Thermal stress accelerates ion migration, reducing lifetime under continuous 1-sun illumination.",
        char_count=96,
        word_count=11,
        chunk_index=1,
    )


@pytest.fixture
def sample_metadata_chunk():
    src = EvidenceSource(
        openalex_id="https://openalex.org/W1003",
        paper_title="Bibliometrics of Solar Photovoltaics",
        publication_year=2020,
        doi="https://doi.org/10.1000/1003",
        landing_page_url="https://doi.org/10.1000/1003",
        venue="Solar Review",
        authors=("Evan Wright",),
        source_database="OpenAlex",
    )
    return EvidenceChunk(
        chunk_id="W1003-meta-0",
        document_id="https://openalex.org/W1003",
        source=src,
        evidence_type=EvidenceType.METADATA,
        section="metadata",
        text="Title: Bibliometrics of Solar Photovoltaics\nAuthors: Evan Wright\nPublication: Solar Review (2020)",
        char_count=100,
        word_count=14,
        chunk_index=0,
    )


# ==============================================================================
# 1. Model Serialization & Formatting Tests
# ==============================================================================

class TestModels:
    def test_evidence_reference_serialization(self, sample_chunk_1):
        ref = EvidenceReference.from_chunk(sample_chunk_1)
        data = ref.to_dict()

        assert data["evidence_id"] == "W1001-abs-1"
        assert data["paper_title"] == "Perovskite Solar Cell Passivation"
        assert data["authors"] == ["Alice Smith", "Bob Jones"]
        assert data["publication_year"] == 2023
        assert data["venue"] == "Energy & Environmental Science"
        assert data["doi"] == "https://doi.org/10.1000/1001"
        assert data["openalex_id"] == "https://openalex.org/W1001"

    def test_synthesized_claim_serialization(self):
        claim = SynthesizedClaim(
            claim_id="C1",
            text="Passivation improves device stability.",
            evidence_ids=["W1001-abs-1"],
            status=ClaimStatus.SUPPORTED,
            valid_evidence_ids=["W1001-abs-1"],
            invalid_evidence_ids=[],
        )
        data = claim.to_dict()

        assert data["claim_id"] == "C1"
        assert data["text"] == "Passivation improves device stability."
        assert data["evidence_ids"] == ["W1001-abs-1"]
        assert data["status"] == "supported"
        assert data["valid_evidence_ids"] == ["W1001-abs-1"]
        assert data["invalid_evidence_ids"] == []

    def test_validation_report_serialization(self):
        rep = ValidationReport(
            total_claims=3,
            supported_claims=2,
            partially_valid_claims=1,
            unsupported_claims=0,
            total_citations=4,
            valid_citations=3,
            invalid_citations=1,
            grounding_score=0.75,
            invalid_evidence_ids=["W9999-abs-1"],
        )
        data = rep.to_dict()

        assert data["total_claims"] == 3
        assert data["supported_claims"] == 2
        assert data["partially_valid_claims"] == 1
        assert data["unsupported_claims"] == 0
        assert data["total_citations"] == 4
        assert data["valid_citations"] == 3
        assert data["invalid_citations"] == 1
        assert data["grounding_score"] == 0.75
        assert data["invalid_evidence_ids"] == ["W9999-abs-1"]

    def test_evidence_context_serialization(self, sample_chunk_1):
        match = EvidenceMatch(chunk=sample_chunk_1, score=1.85, matched_terms=["passivation"])
        ctx = EvidenceContext(query="passivation efficiency", matches=[match], total_papers=1, total_chunks=1)
        data = ctx.to_dict()

        assert data["query"] == "passivation efficiency"
        assert data["total_papers"] == 1
        assert len(data["matches"]) == 1
        assert ctx.chunks == [sample_chunk_1]

    def test_research_synthesis_to_dict_and_markdown(self, sample_chunk_1):
        ref = EvidenceReference.from_chunk(sample_chunk_1)
        claim = SynthesizedClaim(
            claim_id="C1",
            text="2D perovskite passivation yields 25.2% efficiency.",
            evidence_ids=["W1001-abs-1"],
            status=ClaimStatus.SUPPORTED,
            valid_evidence_ids=["W1001-abs-1"],
        )
        rep = ValidationReport(
            total_claims=1,
            supported_claims=1,
            partially_valid_claims=0,
            unsupported_claims=0,
            total_citations=1,
            valid_citations=1,
            invalid_citations=0,
            grounding_score=1.0,
            invalid_evidence_ids=[],
        )
        synth = ResearchSynthesis(
            research_question="How does passivation impact perovskite efficiency?",
            overview="Passivation significantly enhances both efficiency and stability.",
            key_findings=[claim],
            conflicting_findings=[],
            limitations=["Evaluated primarily on spin-coated small-area cells."],
            claims=[claim],
            validation_report=rep,
            model_name="mock-model",
            evidence_references={"W1001-abs-1": ref},
        )

        d = synth.to_dict()
        assert d["research_question"] == "How does passivation impact perovskite efficiency?"
        assert len(d["key_findings"]) == 1
        assert d["validation_report"]["grounding_score"] == 1.0
        assert "W1001-abs-1" in d["evidence_references"]

        md = synth.to_markdown()
        assert "ATHENA Scientific Evidence Synthesis (Milestone M3)" in md
        assert "Research Question: How does passivation impact perovskite efficiency?" in md
        assert "Model: mock-model" in md
        assert "1. [C1] 2D perovskite passivation yields 25.2% efficiency." in md
        assert "Evidence: [W1001-abs-1] (Status: SUPPORTED)" in md
        assert "Grounding Score:           100.0%" in md
        assert "[W1001-abs-1] Perovskite Solar Cell Passivation" in md


# ==============================================================================
# 2. Prompt Generation Tests
# ==============================================================================

class TestPrompts:
    def test_deterministic_prompt_generation(self, sample_chunk_1, sample_chunk_2):
        q = "perovskite stability"
        prompt1 = build_synthesis_prompt(q, [sample_chunk_1, sample_chunk_2])
        prompt2 = build_synthesis_prompt(q, [sample_chunk_1, sample_chunk_2])

        assert prompt1 == prompt2

    def test_prompt_includes_evidence_ids_provenance_and_integrity_notes(
        self, sample_chunk_1, sample_metadata_chunk
    ):
        q = "solar photovoltaics"
        prompt = build_synthesis_prompt(q, [sample_chunk_1, sample_metadata_chunk])

        assert "W1001-abs-1" in prompt
        assert "W1003-meta-0" in prompt
        assert "Perovskite Solar Cell Passivation" in prompt
        assert "Alice Smith" in prompt
        assert "Energy & Environmental Science" in prompt
        # Integrity notes
        assert "[Integrity Note: Evidence sourced from peer-reviewed abstract; full-text body not indexed." in prompt
        assert "[Integrity Note: Bibliographic metadata only; abstract and full-text body not indexed.]" in prompt

    def test_system_prompt_integrity_clauses(self):
        assert "You MUST use ONLY the evidence supplied in the context." in SYSTEM_PROMPT
        assert "Every factual claim MUST reference one or more exact evidence IDs." in SYSTEM_PROMPT
        assert "Do NOT invent citations." in SYSTEM_PROMPT
        assert "Do NOT invent papers, authors, DOI values" in SYSTEM_PROMPT
        assert "Return strict JSON matching the required schema." in SYSTEM_PROMPT


# ==============================================================================
# 3. Claim / Evidence Validation Tests
# ==============================================================================

class TestValidation:
    def test_all_valid_citations_supported(self, sample_chunk_1, sample_chunk_2):
        claim1 = SynthesizedClaim(claim_id="C1", text="Passivation works.", evidence_ids=["W1001-abs-1"])
        claim2 = SynthesizedClaim(claim_id="C2", text="Thermal degradation occurs.", evidence_ids=["W1002-abs-1"])
        synth = ResearchSynthesis(
            research_question="Q",
            overview="O",
            key_findings=[claim1, claim2],
            conflicting_findings=[],
            limitations=[],
            claims=[claim1, claim2],
            validation_report=None,  # to be computed
            model_name="mock",
        )

        rep, refs = validate_synthesis(synth, [sample_chunk_1, sample_chunk_2])

        assert rep.total_claims == 2
        assert rep.supported_claims == 2
        assert rep.partially_valid_claims == 0
        assert rep.unsupported_claims == 0
        assert rep.total_citations == 2
        assert rep.valid_citations == 2
        assert rep.invalid_citations == 0
        assert rep.grounding_score == 1.0
        assert rep.invalid_evidence_ids == []
        assert claim1.status == ClaimStatus.SUPPORTED
        assert claim2.status == ClaimStatus.SUPPORTED
        assert "W1001-abs-1" in refs
        assert "W1002-abs-1" in refs

    def test_hallucinated_evidence_id_detected(self, sample_chunk_1):
        claim = SynthesizedClaim(
            claim_id="C1",
            text="Unfounded assertion.",
            evidence_ids=["PHANTOM-9999"],
        )
        synth = ResearchSynthesis(
            research_question="Q",
            overview="O",
            key_findings=[claim],
            conflicting_findings=[],
            limitations=[],
            claims=[claim],
            validation_report=None,
            model_name="mock",
        )

        rep, refs = validate_synthesis(synth, [sample_chunk_1])

        assert rep.supported_claims == 0
        assert rep.unsupported_claims == 1
        assert rep.invalid_citations == 1
        assert rep.valid_citations == 0
        assert rep.grounding_score == 0.0
        assert rep.invalid_evidence_ids == ["PHANTOM-9999"]
        assert claim.status == ClaimStatus.UNSUPPORTED
        assert "PHANTOM-9999" not in refs

    def test_partially_valid_claim(self, sample_chunk_1):
        claim = SynthesizedClaim(
            claim_id="C1",
            text="Mixed assertion.",
            evidence_ids=["W1001-abs-1", "FAKE-001"],
        )
        synth = ResearchSynthesis(
            research_question="Q",
            overview="O",
            key_findings=[claim],
            conflicting_findings=[],
            limitations=[],
            claims=[claim],
            validation_report=None,
            model_name="mock",
        )

        rep, refs = validate_synthesis(synth, [sample_chunk_1])

        assert rep.supported_claims == 0
        assert rep.partially_valid_claims == 1
        assert rep.unsupported_claims == 0
        assert rep.valid_citations == 1
        assert rep.invalid_citations == 1
        assert rep.grounding_score == 0.5
        assert rep.invalid_evidence_ids == ["FAKE-001"]
        assert claim.status == ClaimStatus.PARTIALLY_VALID
        assert "W1001-abs-1" in refs

    def test_claim_without_evidence_is_unsupported(self, sample_chunk_1):
        claim = SynthesizedClaim(
            claim_id="C1",
            text="Claim with zero citations.",
            evidence_ids=[],
        )
        synth = ResearchSynthesis(
            research_question="Q",
            overview="O",
            key_findings=[claim],
            conflicting_findings=[],
            limitations=[],
            claims=[claim],
            validation_report=None,
            model_name="mock",
        )

        rep, refs = validate_synthesis(synth, [sample_chunk_1])

        assert rep.unsupported_claims == 1
        assert claim.status == ClaimStatus.UNSUPPORTED


# ==============================================================================
# 4. LLM Providers & Factory Tests
# ==============================================================================

class TestProviders:
    def test_mock_llm_client_static_dict(self):
        payload = {"overview": "Sample overview", "key_findings": []}
        client = MockLLMClient(response=payload)
        resp = client.complete("test prompt", system_prompt="test sys")

        assert json.loads(resp) == payload
        assert len(client.call_history) == 1
        assert client.call_history[0] == ("test prompt", "test sys")

    def test_mock_llm_client_response_generator(self):
        client = MockLLMClient(response_generator=lambda p, s: f"GEN:{len(p)}")
        resp = client.complete("hello", system_prompt="sys")

        assert resp == "GEN:5"

    def test_get_llm_client_mock_provider(self):
        client = get_llm_client(provider="mock", mock_response={"test": True})
        assert isinstance(client, MockLLMClient)
        assert json.loads(client.complete("prompt")) == {"test": True}

    def test_get_llm_client_unconfigured_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMConfigurationError) as exc_info:
                get_llm_client()
            assert "No LLM provider is configured in ATHENA." in str(exc_info.value)

    def test_get_llm_client_unsupported_provider(self):
        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm_client(provider="anthropic_native")
        assert "Unsupported LLM provider: 'anthropic_native'" in str(exc_info.value)

    def test_get_llm_client_ollama_local_config(self):
        client = get_llm_client(provider="ollama", model="qwen2.5:7b")
        assert isinstance(client, OpenAILikeClient)
        assert client.model == "qwen2.5:7b"
        assert client.base_url == "http://localhost:11434/v1"

    def test_get_llm_client_openai_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMConfigurationError) as exc_info:
                get_llm_client(provider="openai")
            assert "OpenAI provider requires an API key" in str(exc_info.value)

    def test_get_llm_client_openrouter_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMConfigurationError) as exc_info:
                get_llm_client(provider="openrouter")
            assert "OpenRouter provider requires an API key" in str(exc_info.value)


# ==============================================================================
# 5. Synthesizer Parsing & Orchestration Tests
# ==============================================================================

class TestSynthesizer:
    def test_extract_json_payload_clean_json(self):
        raw = '{"overview": "Great progress", "key_findings": []}'
        data = extract_json_payload(raw)
        assert data["overview"] == "Great progress"

    def test_extract_json_payload_markdown_fences(self):
        raw = """Here is the synthesis:
```json
{
  "overview": "Synthesized insights",
  "key_findings": [
    {"claim_id": "C1", "text": "Passivation boosts Voc", "evidence_ids": ["W1-abs-1"]}
  ],
  "conflicting_findings": [],
  "limitations": ["Small sample size"]
}
```
Hope this helps!"""
        data = extract_json_payload(raw)
        assert data["overview"] == "Synthesized insights"
        assert len(data["key_findings"]) == 1
        assert data["key_findings"][0]["claim_id"] == "C1"

    def test_extract_json_payload_malformed_raises_synthesis_parsing_error(self):
        with pytest.raises(SynthesisParsingError):
            extract_json_payload("Sorry, I cannot provide JSON because I am an AI.")

    def test_extract_json_payload_empty_string_raises(self):
        with pytest.raises(SynthesisParsingError) as exc:
            extract_json_payload("   \n\t  ")
        assert "response was empty" in str(exc.value)

    def test_synthesizer_empty_evidence_no_llm_call(self):
        mock_client = MockLLMClient()
        synthesizer = Synthesizer(client=mock_client)

        synthesis = synthesizer.synthesize("perovskite stability", evidence_context=[])

        assert "Insufficient evidence" in synthesis.overview
        assert len(synthesis.key_findings) == 0
        assert len(mock_client.call_history) == 0  # LLM must NOT be called!
        assert synthesis.validation_report.grounding_score == 1.0

    def test_synthesizer_successful_orchestration(self, sample_chunk_1, sample_chunk_2):
        llm_output = {
            "overview": "Passivation improves efficiency while thermal stress reduces lifetime.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Surface passivation achieves 25.2% power conversion efficiency.",
                    "evidence_ids": ["W1001-abs-1"],
                }
            ],
            "conflicting_findings": [
                {
                    "claim_id": "C2",
                    "text": "Thermal stress induces rapid degradation contrary to ambient stability reports.",
                    "evidence_ids": ["W1002-abs-1"],
                }
            ],
            "limitations": ["Degradation dynamics examined only under continuous 1-sun."],
        }
        mock_client = MockLLMClient(response=llm_output)
        synthesizer = Synthesizer(client=mock_client)

        ctx = EvidenceContext(query="perovskite passivation", matches=[
            EvidenceMatch(chunk=sample_chunk_1, score=2.1),
            EvidenceMatch(chunk=sample_chunk_2, score=1.8),
        ])

        synthesis = synthesizer.synthesize(
            research_question="What factors govern perovskite solar cell stability?",
            evidence_context=ctx,
        )

        assert len(mock_client.call_history) == 1
        assert len(synthesis.key_findings) == 1
        assert len(synthesis.conflicting_findings) == 1
        assert synthesis.key_findings[0].status == ClaimStatus.SUPPORTED
        assert synthesis.conflicting_findings[0].status == ClaimStatus.SUPPORTED
        assert synthesis.validation_report.grounding_score == 1.0
        assert "W1001-abs-1" in synthesis.evidence_references
        assert "W1002-abs-1" in synthesis.evidence_references


# ==============================================================================
# 6. M1 -> M2 -> M3 Full Pipeline Integration Test (Offline)
# ==============================================================================

class TestPipelineIntegration:
    def test_m1_m2_m3_end_to_end_offline(self):
        # 1. Simulate M1 Paper output
        m1_paper = Paper(
            title="CRISPR Editing of PCSK9 Lowers LDL Cholesterol",
            openalex_id="https://openalex.org/W9999",
            source="OpenAlex",
            authors=["Jennifer Doudna", "Kiran Musunuru"],
            publication_year=2021,
            doi="https://doi.org/10.1056/NEJMoa9999",
            abstract="In vivo adenine base editing of PCSK9 in nonhuman primates precisely reduces blood cholesterol levels by 60%.",
            venue="New England Journal of Medicine",
            cited_by_count=320,
            landing_page_url="https://doi.org/10.1056/NEJMoa9999",
        )

        # 2. Simulate M2 Document Processing and Chunking
        normalized = normalize_paper(m1_paper)
        chunks = chunk_document(normalized)
        assert len(chunks) >= 2  # 1 metadata chunk + 1 abstract chunk

        abs_chunk = next(c for c in chunks if c.evidence_type == EvidenceType.ABSTRACT)
        assert abs_chunk.chunk_id == "W9999-abs-1"
        assert "reduces blood cholesterol levels by 60%" in abs_chunk.text

        # 3. Simulate M3 Evidence Synthesis
        mock_response = {
            "overview": "Adenine base editing targeting PCSK9 yields substantial reductions in circulating LDL cholesterol.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "PCSK9 editing reduced cholesterol by 60% in nonhuman primate models.",
                    "evidence_ids": [abs_chunk.chunk_id],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Evaluated in nonhuman primates; human clinical durability requires further study."],
        }
        client = MockLLMClient(response=mock_response)
        synthesizer = Synthesizer(client=client)

        synthesis = synthesizer.synthesize(
            research_question="Does CRISPR PCSK9 editing effectively lower cholesterol?",
            evidence_context=[abs_chunk],
        )

        # 4. Verify complete provenance preservation
        assert synthesis.validation_report.grounding_score == 1.0
        assert abs_chunk.chunk_id in synthesis.evidence_references

        ref = synthesis.evidence_references[abs_chunk.chunk_id]
        assert ref.paper_title == "CRISPR Editing of PCSK9 Lowers LDL Cholesterol"
        assert ref.authors == ["Jennifer Doudna", "Kiran Musunuru"]
        assert ref.publication_year == 2021
        assert ref.venue == "New England Journal of Medicine"
        assert ref.doi == "https://doi.org/10.1056/NEJMoa9999"
        assert ref.openalex_id == "https://openalex.org/W9999"


# ==============================================================================
# 7. CLI Synthesis Integration Test
# ==============================================================================

class TestCLIWithSynthesis:
    @patch("app.main.search_papers")
    def test_cli_synthesize_with_mock_provider(self, mock_search, capsys, tmp_path):
        sample_paper = Paper(
            title="Passivation in Halide Perovskites",
            openalex_id="https://openalex.org/W8888",
            source="OpenAlex",
            authors=["Elena Rossi"],
            publication_year=2024,
            doi="https://doi.org/10.1000/8888",
            abstract="Lewis base treatment suppresses interfacial trap states and enhances solar efficiency.",
            venue="JACS",
            cited_by_count=15,
        )
        mock_search.return_value = [sample_paper]

        export_file = tmp_path / "synthesis_export.json"

        # Mock response returned by get_llm_client in main CLI
        mock_llm_json = {
            "overview": "Lewis base treatment significantly mitigates defect states.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Interfacial trap suppression enhances solar conversion efficiency.",
                    "evidence_ids": ["W8888-abs-1"],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Limited to small test cells."],
        }

        # Set environment variable so get_llm_client creates MockLLMClient
        with patch("app.synthesis.providers.MockLLMClient.complete", return_value=json.dumps(mock_llm_json)):
            exit_code = main([
                "perovskite passivation",
                "--extract-evidence",
                "--synthesize",
                "--llm-provider", "mock",
                "--export", str(export_file),
            ])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "ATHENA Scientific Evidence Synthesis (Milestone M3)" in captured.out
        assert "Interfacial trap suppression enhances solar conversion efficiency." in captured.out
        assert "Grounding Score:           100.0%" in captured.out
        assert "[W8888-abs-1] Passivation in Halide Perovskites" in captured.out

        # Verify export JSON includes synthesis
        assert export_file.exists()
        with open(export_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "synthesis" in data
        assert data["synthesis"]["research_question"] == "perovskite passivation"
        assert len(data["synthesis"]["key_findings"]) == 1
        assert data["synthesis"]["validation_report"]["grounding_score"] == 1.0
