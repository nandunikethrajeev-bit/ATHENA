"""Comprehensive offline tests for candidate research-gap analysis (Milestone M4).

All tests are deterministic, offline, and require zero external API keys or network access.
"""

import json
from unittest.mock import patch
import pytest

from app.evidence.chunker import chunk_document
from app.evidence.hybrid import HybridRetriever
from app.evidence.models import (
    EvidenceChunk,
    EvidenceMatch,
    EvidenceSource,
    EvidenceType,
)
from app.evidence.normalizer import normalize_paper
from app.evidence.store import EvidenceStore
from app.gaps.analyzer import GapAnalyzer
from app.gaps.models import (
    CandidateGap,
    GapType,
    GapValidationReport,
    ResearchGapAnalysis,
)
from app.gaps.prompts import (
    GAP_SYSTEM_PROMPT,
    build_gap_analysis_prompt,
)
from app.gaps.validators import validate_gap_analysis
from app.main import main
from app.retrieval.models import Paper
from app.synthesis.models import (
    ClaimStatus,
    EvidenceReference,
    ResearchSynthesis,
    SynthesizedClaim,
    ValidationReport,
)
from app.synthesis.providers import MockLLMClient
from app.synthesis.synthesizer import SynthesisParsingError, Synthesizer


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_evidence_chunk():
    src = EvidenceSource(
        openalex_id="https://openalex.org/W2001",
        paper_title="Perovskite Surface Defect Engineering",
        publication_year=2023,
        doi="https://doi.org/10.1000/2001",
        landing_page_url="https://doi.org/10.1000/2001",
        venue="Advanced Materials",
        authors=("Alice Chen", "Bob Wang"),
        source_database="OpenAlex",
    )
    return EvidenceChunk(
        chunk_id="W2001-abs-1",
        document_id="https://openalex.org/W2001",
        source=src,
        evidence_type=EvidenceType.ABSTRACT,
        section="abstract",
        text="Interfacial ammonium passivators reduce lead vacancy defects but accelerate halide segregation under thermal bias above 65 deg C.",
        char_count=130,
        word_count=18,
        chunk_index=1,
    )


@pytest.fixture
def sample_synthesis(sample_evidence_chunk):
    claim1 = SynthesizedClaim(
        claim_id="C1",
        text="Ammonium passivation suppresses lead vacancy defects at cell interfaces.",
        evidence_ids=[sample_evidence_chunk.chunk_id],
        status=ClaimStatus.SUPPORTED,
        valid_evidence_ids=[sample_evidence_chunk.chunk_id],
    )
    claim2 = SynthesizedClaim(
        claim_id="C2",
        text="Halide segregation occurs under thermal bias above 65 deg C.",
        evidence_ids=[sample_evidence_chunk.chunk_id],
        status=ClaimStatus.SUPPORTED,
        valid_evidence_ids=[sample_evidence_chunk.chunk_id],
    )
    report = ValidationReport(
        total_claims=2,
        supported_claims=2,
        partially_valid_claims=0,
        unsupported_claims=0,
        total_citations=2,
        valid_citations=2,
        invalid_citations=0,
        grounding_score=1.0,
        invalid_evidence_ids=[],
    )
    ref = EvidenceReference.from_chunk(sample_evidence_chunk)

    return ResearchSynthesis(
        research_question="What mechanisms govern perovskite interfacial degradation?",
        overview="Surface passivation improves initial efficiency but thermal bias triggers secondary phase instability.",
        key_findings=[claim1, claim2],
        conflicting_findings=[],
        limitations=["Degradation kinetics above 85 deg C were uninvestigated."],
        claims=[claim1, claim2],
        validation_report=report,
        model_name="mock-synthesizer",
        evidence_references={sample_evidence_chunk.chunk_id: ref},
    )


# ==============================================================================
# 1. Model & Serialization Tests
# ==============================================================================

class TestGapModels:
    def test_candidate_gap_serialization(self):
        gap = CandidateGap(
            gap_id="G1",
            title="High-temperature degradation kinetics",
            description="Behavior above 85 °C remains uncharacterized.",
            gap_type=GapType.COVERAGE_SCOPE,
            rationale="Existing studies only evaluate up to 65 °C.",
            source_claim_ids=["C2"],
            evidence_ids=["W2001-abs-1"],
            valid_evidence_ids=["W2001-abs-1"],
            invalid_evidence_ids=[],
            valid_claim_ids=["C2"],
            invalid_claim_ids=[],
            is_grounded=True,
        )
        data = gap.to_dict()

        assert data["gap_id"] == "G1"
        assert data["title"] == "High-temperature degradation kinetics"
        assert data["gap_type"] == "coverage_scope"
        assert data["source_claim_ids"] == ["C2"]
        assert data["is_grounded"] is True

        reconstructed = CandidateGap.from_dict(data)
        assert reconstructed.gap_id == "G1"
        assert reconstructed.gap_type == GapType.COVERAGE_SCOPE
        assert reconstructed.is_grounded is True

    def test_gap_validation_report_serialization(self):
        rep = GapValidationReport(
            total_gaps=2,
            grounded_gaps=2,
            unsupported_gaps=0,
            total_evidence_citations=2,
            valid_evidence_citations=2,
            invalid_evidence_citations=0,
            invalid_evidence_ids=[],
            invalid_claim_ids=[],
            grounding_score=1.0,
        )
        d = rep.to_dict()
        assert d["total_gaps"] == 2
        assert d["grounding_score"] == 1.0

    def test_research_gap_analysis_markdown_rendering(self, sample_synthesis, sample_evidence_chunk):
        gap = CandidateGap(
            gap_id="G1",
            title="Unexplored operational thermal range",
            description="Testing above 85 °C is absent.",
            gap_type=GapType.COVERAGE_SCOPE,
            rationale="Current literature only tests moderate thermal bias.",
            source_claim_ids=["C2"],
            evidence_ids=[sample_evidence_chunk.chunk_id],
            is_grounded=True,
        )
        report = GapValidationReport(
            total_gaps=1,
            grounded_gaps=1,
            unsupported_gaps=0,
            total_evidence_citations=1,
            valid_evidence_citations=1,
            invalid_evidence_citations=0,
            invalid_evidence_ids=[],
            invalid_claim_ids=[],
            grounding_score=1.0,
        )
        ref = EvidenceReference.from_chunk(sample_evidence_chunk)

        analysis = ResearchGapAnalysis(
            research_question="What mechanisms govern perovskite interfacial degradation?",
            gaps=[gap],
            validation_report=report,
            evidence_references={sample_evidence_chunk.chunk_id: ref},
            model_name="mock-analyzer",
        )

        md = analysis.to_markdown()
        assert "ATHENA Candidate Research-Gap Analysis (Milestone M4)" in md
        assert "Candidate gaps indicate unexplored boundaries in the retrieved literature subset." in md
        assert "1. [G1] Unexplored operational thermal range" in md
        assert "Type:        COVERAGE_SCOPE" in md
        assert "Grounding Score:           100.0%" in md
        assert "[W2001-abs-1] Perovskite Surface Defect Engineering" in md


# ==============================================================================
# 2. Prompt Generation Tests
# ==============================================================================

class TestGapPrompts:
    def test_build_gap_analysis_prompt_deterministic(self, sample_synthesis, sample_evidence_chunk):
        prompt1 = build_gap_analysis_prompt(sample_synthesis, [sample_evidence_chunk])
        prompt2 = build_gap_analysis_prompt(sample_synthesis, [sample_evidence_chunk])
        assert prompt1 == prompt2

    def test_gap_prompt_contains_synthesis_components(self, sample_synthesis, sample_evidence_chunk):
        prompt = build_gap_analysis_prompt(sample_synthesis, [sample_evidence_chunk])

        assert "What mechanisms govern perovskite interfacial degradation?" in prompt
        assert "[C1] Ammonium passivation suppresses lead vacancy defects" in prompt
        assert "[C2] Halide segregation occurs under thermal bias" in prompt
        assert "Degradation kinetics above 85 deg C were uninvestigated." in prompt
        assert "W2001-abs-1" in prompt
        assert "contradiction | methodological | coverage_scope | unverified_claim" in prompt

    def test_gap_system_prompt_rules(self):
        assert "discovering candidate research gaps" in GAP_SYSTEM_PROMPT
        assert "Do NOT invent disconnected or arbitrary scientific topics." in GAP_SYSTEM_PROMPT
        assert "Return strict JSON adhering to the specified schema." in GAP_SYSTEM_PROMPT


# ==============================================================================
# 3. Grounding & Citation Validation Tests
# ==============================================================================

class TestGapValidation:
    def test_all_valid_citations(self, sample_synthesis, sample_evidence_chunk):
        gap = CandidateGap(
            gap_id="G1",
            title="Valid Gap",
            description="Legitimate gap description.",
            gap_type=GapType.COVERAGE_SCOPE,
            rationale="Grounded in C1.",
            source_claim_ids=["C1"],
            evidence_ids=[sample_evidence_chunk.chunk_id],
        )
        report = GapValidationReport(0, 0, 0, 0, 0, 0, [], [], 0.0)
        analysis = ResearchGapAnalysis("Q", [gap], report, {}, "mock")

        rep, refs = validate_gap_analysis(analysis, sample_synthesis, [sample_evidence_chunk])

        assert rep.total_gaps == 1
        assert rep.grounded_gaps == 1
        assert rep.unsupported_gaps == 0
        assert rep.grounding_score == 1.0
        assert gap.is_grounded is True
        assert sample_evidence_chunk.chunk_id in refs

    def test_phantom_evidence_id_flagged(self, sample_synthesis):
        gap = CandidateGap(
            gap_id="G1",
            title="Phantom Evidence Gap",
            description="Cites hallucinated chunk ID.",
            gap_type=GapType.METHODOLOGICAL,
            rationale="Unfounded.",
            source_claim_ids=["C1"],
            evidence_ids=["W9999-abs-9"],
        )
        report = GapValidationReport(0, 0, 0, 0, 0, 0, [], [], 0.0)
        analysis = ResearchGapAnalysis("Q", [gap], report, {}, "mock")

        rep, refs = validate_gap_analysis(analysis, sample_synthesis, [])

        assert rep.grounded_gaps == 0
        assert rep.unsupported_gaps == 1
        assert rep.invalid_evidence_ids == ["W9999-abs-9"]
        assert gap.is_grounded is False

    def test_phantom_claim_id_flagged(self, sample_synthesis, sample_evidence_chunk):
        gap = CandidateGap(
            gap_id="G1",
            title="Phantom Claim Gap",
            description="Cites non-existent claim.",
            gap_type=GapType.CONTRADICTION,
            rationale="Unfounded claim.",
            source_claim_ids=["C99"],
            evidence_ids=[sample_evidence_chunk.chunk_id],
        )
        report = GapValidationReport(0, 0, 0, 0, 0, 0, [], [], 0.0)
        analysis = ResearchGapAnalysis("Q", [gap], report, {}, "mock")

        rep, refs = validate_gap_analysis(analysis, sample_synthesis, [sample_evidence_chunk])

        assert rep.grounded_gaps == 0
        assert rep.unsupported_gaps == 1
        assert rep.invalid_claim_ids == ["C99"]
        assert gap.is_grounded is False


# ==============================================================================
# 4. Analyzer & Mock LLM Tests
# ==============================================================================

class TestGapAnalyzer:
    def test_gap_analyzer_with_mock_client(self, sample_synthesis, sample_evidence_chunk):
        mock_payload = {
            "gaps": [
                {
                    "gap_id": "G1",
                    "title": "Unresolved high-temperature degradation mechanism",
                    "description": "Studies reveal phase instability above 65 °C but lack kinetic models at 85 °C.",
                    "gap_type": "coverage_scope",
                    "rationale": "Reported limitation explicitly notes lack of high temperature testing.",
                    "source_claim_ids": ["C2"],
                    "evidence_ids": [sample_evidence_chunk.chunk_id],
                }
            ]
        }
        client = MockLLMClient(response=mock_payload)
        analyzer = GapAnalyzer(client=client)

        analysis = analyzer.analyze(sample_synthesis, [sample_evidence_chunk])

        assert len(analysis.gaps) == 1
        assert analysis.gaps[0].gap_id == "G1"
        assert analysis.gaps[0].gap_type == GapType.COVERAGE_SCOPE
        assert analysis.gaps[0].is_grounded is True
        assert analysis.validation_report.grounding_score == 1.0

    def test_gap_analyzer_markdown_fences(self, sample_synthesis, sample_evidence_chunk):
        raw = """```json
{
  "gaps": [
    {
      "gap_id": "G1",
      "title": "Passivation durability gap",
      "description": "Long-term durability unaddressed.",
      "gap_type": "methodological",
      "rationale": "Short exposure time.",
      "source_claim_ids": ["C1"],
      "evidence_ids": ["W2001-abs-1"]
    }
  ]
}
```"""
        client = MockLLMClient(response=raw)
        analyzer = GapAnalyzer(client=client)
        analysis = analyzer.analyze(sample_synthesis, [sample_evidence_chunk])

        assert len(analysis.gaps) == 1
        assert analysis.gaps[0].title == "Passivation durability gap"

    def test_gap_analyzer_empty_synthesis_no_llm_call(self):
        empty_synthesis = ResearchSynthesis(
            research_question="Question?",
            overview="Insufficient evidence.",
            key_findings=[],
            conflicting_findings=[],
            limitations=[],
            claims=[],
            validation_report=ValidationReport(0, 0, 0, 0, 0, 0, 0, 1.0, []),
            model_name="none",
        )
        mock_client = MockLLMClient()
        analyzer = GapAnalyzer(client=mock_client)

        analysis = analyzer.analyze(empty_synthesis)

        assert len(analysis.gaps) == 0
        assert len(mock_client.call_history) == 0  # LLM must not be called!

    def test_gap_analyzer_malformed_json_raises(self, sample_synthesis):
        client = MockLLMClient(response="This is not valid JSON.")
        analyzer = GapAnalyzer(client=client)

        with pytest.raises(SynthesisParsingError):
            analyzer.analyze(sample_synthesis)


# ==============================================================================
# 5. Full Pipeline Integration: M1 -> M2 -> M3 -> M4
# ==============================================================================

class TestEndToEndM4Pipeline:
    def test_m1_m2_m3_m4_end_to_end(self):
        # 1. M1: OpenAlex Paper
        paper = Paper(
            title="Perovskite Photovoltaics In Situ Degradation",
            openalex_id="https://openalex.org/W5555",
            source="OpenAlex",
            authors=["Prof. Maria Santos", "Dr. John Doe"],
            publication_year=2024,
            doi="https://doi.org/10.1000/5555",
            abstract="In situ X-ray scattering reveals lead iodide phase segregation under light soaking at 45 deg C.",
            venue="Nature Photonics",
            cited_by_count=52,
        )

        # 2. M2: Normalization & Chunking
        norm_doc = normalize_paper(paper)
        chunks = chunk_document(norm_doc)
        abs_chunk = next(c for c in chunks if c.evidence_type == EvidenceType.ABSTRACT)

        # 3. M3: Store & Hybrid Retrieval
        store = EvidenceStore(":memory:")
        store.save_chunks(chunks)
        retriever = HybridRetriever()
        retriever.index_chunks(chunks)
        matches = retriever.retrieve("phase segregation", top_k=1, mode="hybrid")
        assert len(matches) == 1

        # 4. M3: Evidence Synthesis
        synth_payload = {
            "overview": "Light soaking induces lead iodide phase segregation at elevated temperatures.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Light soaking triggers measurable phase segregation at 45 deg C.",
                    "evidence_ids": [abs_chunk.chunk_id],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Degradation examined only up to 45 deg C; commercial operating temperatures reach 85 deg C."],
        }
        synth_client = MockLLMClient(response=synth_payload)
        synthesizer = Synthesizer(client=synth_client)
        synthesis = synthesizer.synthesize("perovskite degradation", matches)
        assert synthesis.validation_report.grounding_score == 1.0

        # 5. M4: Candidate Research-Gap Analysis
        gap_payload = {
            "gaps": [
                {
                    "gap_id": "G1",
                    "title": "Phase stability unexamined at industry standard 85 °C damp heat",
                    "description": "Studies demonstrate segregation at 45 °C but omit the IEC 61215 standard 85 °C / 85% RH regime.",
                    "gap_type": "coverage_scope",
                    "rationale": "Directly exposed by limitation noting tests only reached 45 °C.",
                    "source_claim_ids": ["C1"],
                    "evidence_ids": [abs_chunk.chunk_id],
                }
            ]
        }
        gap_client = MockLLMClient(response=gap_payload)
        gap_analyzer = GapAnalyzer(client=gap_client)
        analysis = gap_analyzer.analyze(synthesis, chunks)

        # 6. Verify cross-milestone provenance preservation
        assert len(analysis.gaps) == 1
        gap = analysis.gaps[0]
        assert gap.is_grounded is True
        assert gap.source_claim_ids == ["C1"]
        assert gap.evidence_ids == [abs_chunk.chunk_id]
        assert analysis.validation_report.grounding_score == 1.0

        # Full provenance preserved
        assert abs_chunk.chunk_id in analysis.evidence_references
        ref = analysis.evidence_references[abs_chunk.chunk_id]
        assert ref.paper_title == "Perovskite Photovoltaics In Situ Degradation"
        assert ref.doi == "https://doi.org/10.1000/5555"
        assert ref.authors == ["Prof. Maria Santos", "Dr. John Doe"]
        assert ref.publication_year == 2024


# ==============================================================================
# 6. CLI Integration Test
# ==============================================================================

class TestCLIWithGapAnalysis:
    @patch("app.main.search_papers")
    def test_cli_analyze_gaps_flag(self, mock_search, capsys, tmp_path):
        sample_paper = Paper(
            title="Halide Perovskite Ingot Casting",
            openalex_id="https://openalex.org/W4444",
            source="OpenAlex",
            authors=["Elena Rossi"],
            publication_year=2024,
            doi="https://doi.org/10.1000/4444",
            abstract="Vertical Bridgman growth produces large single crystals with low trap density.",
            venue="JACS",
            cited_by_count=21,
        )
        mock_search.return_value = [sample_paper]

        export_file = tmp_path / "gap_export.json"

        synth_mock_json = {
            "overview": "Bridgman growth produces high purity perovskite crystals.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Vertical Bridgman growth suppresses trap density.",
                    "evidence_ids": ["W4444-abs-1"],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Crystal slicing yields high kerf loss."],
        }

        gap_mock_json = {
            "gaps": [
                {
                    "gap_id": "G1",
                    "title": "Kerf-loss mitigation in perovskite single crystal wafering",
                    "description": "Wafering methods to minimize material waste remain uninvestigated.",
                    "gap_type": "methodological",
                    "rationale": "Reported limitation explicitly notes high kerf loss during slicing.",
                    "source_claim_ids": ["C1"],
                    "evidence_ids": ["W4444-abs-1"],
                }
            ]
        }

        def mock_llm_responder(prompt, system_prompt):
            if "discovering candidate research gaps" in (system_prompt or ""):
                return json.dumps(gap_mock_json)
            return json.dumps(synth_mock_json)

        with patch("app.synthesis.providers.MockLLMClient.complete", side_effect=mock_llm_responder):
            exit_code = main([
                "perovskite single crystal",
                "--extract-evidence",
                "--synthesize",
                "--analyze-gaps",
                "--llm-provider", "mock",
                "--export", str(export_file),
            ])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "ATHENA Candidate Research-Gap Analysis (Milestone M4)" in captured.out
        assert "Kerf-loss mitigation in perovskite single crystal wafering" in captured.out
        assert "Type:        METHODOLOGICAL" in captured.out

        # Verify export JSON includes research_gaps
        assert export_file.exists()
        with open(export_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "research_gaps" in data
        assert data["research_gaps"]["total_gaps"] == 1
        assert data["research_gaps"]["gaps"][0]["gap_id"] == "G1"
        assert data["research_gaps"]["gaps"][0]["gap_type"] == "methodological"
