"""Offline unit and integration tests for ATHENA Milestone M5 (Candidate Hypothesis Generation).

Verifies:
- Data model creation, serialization, and roundtrip fidelity
- Formatted markdown report rendering with epistemic guardrails
- Deterministic prompt construction with tripartite anchoring
- Citation, gap ID, and falsifiability validation auditing
- HypothesisGenerator orchestrator execution with MockLLMClient
- Full end-to-end integration: M1 -> M2 -> M3 -> M4 -> M5
- CLI invocation with --generate-hypotheses flag and JSON export
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.evidence.chunker import chunk_documents
from app.evidence.models import EvidenceChunk, EvidenceMatch, EvidenceSource
from app.evidence.normalizer import normalize_paper
from app.evidence.retriever import EvidenceIndex
from app.gaps.analyzer import GapAnalyzer
from app.gaps.models import CandidateGap, GapType, GapValidationReport, ResearchGapAnalysis
from app.hypotheses.generator import HypothesisGenerator
from app.hypotheses.models import (
    CandidateHypothesis,
    CandidateHypothesisSet,
    HypothesisValidationReport,
)
from app.hypotheses.prompts import HYPOTHESIS_SYSTEM_PROMPT, build_hypothesis_prompt
from app.hypotheses.validators import validate_hypothesis_set
from app.main import main
from app.retrieval.models import Paper
from app.synthesis.models import (
    EvidenceContext,
    EvidenceReference,
    ResearchSynthesis,
    SynthesizedClaim,
)
from app.synthesis.providers import MockLLMClient
from app.synthesis.synthesizer import Synthesizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evidence_chunk():
    source = EvidenceSource(
        openalex_id="https://openalex.org/W2301656337",
        doi="https://doi.org/10.1039/c5ee03874j",
        landing_page_url="https://doi.org/10.1039/c5ee03874j",
        paper_title="Cesium-containing triple cation perovskite solar cells",
        authors=("Michael Saliba", "Taisuke Matsui"),
        publication_year=2016,
        venue="Energy & Environmental Science",
    )
    return EvidenceChunk(
        chunk_id="W2301656337-abs-1",
        source=source,
        text="Cesium addition improves perovskite thermal stability and suppresses phase segregation.",
        chunk_index=1,
        total_chunks=2,
        section="abstract",
        evidence_type="abstract",
    )


@pytest.fixture
def sample_synthesis(sample_evidence_chunk):
    claim1 = SynthesizedClaim(
        claim_id="C1",
        text="Cesium cations enhance thermal stability and phase purity in triple-cation perovskites.",
        evidence_ids=["W2301656337-abs-1"],
        valid_evidence_ids=["W2301656337-abs-1"],
        invalid_evidence_ids=[],
        is_grounded=True,
    )
    ref = EvidenceReference.from_chunk(sample_evidence_chunk)
    return ResearchSynthesis(
        research_question="perovskite solar cells stability",
        overview="Perovskite solar cell stability improves significantly with compositional tuning.",
        key_findings=[claim1],
        conflicting_findings=[],
        limitations=["Long-term degradation kinetics under 85% RH moisture stress remain uncharacterized."],
        evidence_references={"W2301656337-abs-1": ref},
        model_name="mock",
    )


@pytest.fixture
def sample_gap_analysis(sample_synthesis, sample_evidence_chunk):
    gap = CandidateGap(
        gap_id="G1",
        title="Moisture-induced degradation kinetics under operational humidity",
        description="Degradation mechanisms and phase segregation rates under continuous operational moisture stress remain unquantified.",
        gap_type=GapType.COVERAGE_SCOPE,
        rationale="Synthesized claims show baseline thermal stability, but moisture-induced degradation kinetics are not characterized.",
        source_claim_ids=["C1"],
        evidence_ids=["W2301656337-abs-1"],
        valid_claim_ids=["C1"],
        invalid_claim_ids=[],
        valid_evidence_ids=["W2301656337-abs-1"],
        invalid_evidence_ids=[],
        is_grounded=True,
    )
    report = GapValidationReport(
        total_gaps=1,
        grounded_gaps=1,
        unsupported_gaps=0,
        total_evidence_citations=2,
        valid_evidence_citations=2,
        invalid_evidence_citations=0,
        invalid_evidence_ids=[],
        invalid_claim_ids=[],
        grounding_score=1.0,
    )
    ref = EvidenceReference.from_chunk(sample_evidence_chunk)
    return ResearchGapAnalysis(
        research_question="perovskite solar cells stability",
        gaps=[gap],
        validation_report=report,
        evidence_references={"W2301656337-abs-1": ref},
        model_name="mock",
    )


# ---------------------------------------------------------------------------
# 1. Model & Serialization Tests
# ---------------------------------------------------------------------------

class TestCandidateHypothesisModels:
    def test_candidate_hypothesis_to_from_dict(self):
        hyp = CandidateHypothesis(
            hypothesis_id="H1",
            target_gap_id="G1",
            title="2D Capping Layer Passivation",
            statement="Deposition of a 2D PEAI capping layer on Cs/FA/MA perovskites reduces moisture degradation by >50% over 1,000h.",
            rationale="Addresses Gap G1 by introducing steric moisture resistance grounded in Claim C1.",
            proposed_mechanism="Hydrophobic phenylethylammonium cations form a moisture-resistant Ruddlesden-Popper 2D capping layer.",
            independent_variables=["PEAI concentration (mM)", "Relative humidity (85% RH)"],
            dependent_variables=["PCE retention (%)", "XRD phase segregation intensity"],
            falsification_criteria="If PCE loss exceeds 20% within 200h under 85% RH, the hypothesis is refuted.",
            source_claim_ids=["C1"],
            evidence_ids=["W2301656337-abs-1"],
            valid_claim_ids=["C1"],
            invalid_claim_ids=[],
            valid_evidence_ids=["W2301656337-abs-1"],
            invalid_evidence_ids=[],
            valid_gap_id=True,
            is_grounded=True,
        )

        d = hyp.to_dict()
        assert d["hypothesis_id"] == "H1"
        assert d["target_gap_id"] == "G1"
        assert d["independent_variables"] == ["PEAI concentration (mM)", "Relative humidity (85% RH)"]
        assert "refuted" in d["falsification_criteria"]

        reconstructed = CandidateHypothesis.from_dict(d)
        assert reconstructed.hypothesis_id == hyp.hypothesis_id
        assert reconstructed.target_gap_id == hyp.target_gap_id
        assert reconstructed.statement == hyp.statement
        assert reconstructed.independent_variables == hyp.independent_variables
        assert reconstructed.dependent_variables == hyp.dependent_variables
        assert reconstructed.falsification_criteria == hyp.falsification_criteria
        assert reconstructed.is_grounded is True

    def test_validation_report_to_dict(self):
        report = HypothesisValidationReport(
            total_hypotheses=2,
            grounded_hypotheses=2,
            unsupported_hypotheses=0,
            falsifiable_count=2,
            total_citations=4,
            valid_citations=4,
            invalid_citations=0,
            invalid_gap_ids=[],
            invalid_claim_ids=[],
            invalid_evidence_ids=[],
            grounding_score=1.0,
        )
        d = report.to_dict()
        assert d["total_hypotheses"] == 2
        assert d["falsifiable_count"] == 2
        assert d["grounding_score"] == 1.0

    def test_candidate_hypothesis_set_markdown_rendering(self, sample_evidence_chunk):
        hyp = CandidateHypothesis(
            hypothesis_id="H1",
            target_gap_id="G1",
            title="2D Capping Layer Passivation",
            statement="Deposition of a PEAI capping layer suppresses moisture degradation.",
            rationale="Resolves uncharacterized moisture stability in Gap G1.",
            proposed_mechanism="Steric shielding against moisture ingress.",
            independent_variables=["PEAI concentration"],
            dependent_variables=["PCE retention (%)"],
            falsification_criteria="If degradation rate equals unpassivated control, hypothesis is refuted.",
            source_claim_ids=["C1"],
            evidence_ids=["W2301656337-abs-1"],
            is_grounded=True,
        )
        report = HypothesisValidationReport(
            total_hypotheses=1,
            grounded_hypotheses=1,
            unsupported_hypotheses=0,
            falsifiable_count=1,
            total_citations=2,
            valid_citations=2,
            invalid_citations=0,
            invalid_gap_ids=[],
            invalid_claim_ids=[],
            invalid_evidence_ids=[],
            grounding_score=1.0,
        )
        ref = EvidenceReference.from_chunk(sample_evidence_chunk)
        hyp_set = CandidateHypothesisSet(
            research_question="perovskite solar cells stability",
            hypotheses=[hyp],
            validation_report=report,
            evidence_references={"W2301656337-abs-1": ref},
            model_name="mock",
        )

        md = hyp_set.to_markdown()
        assert "ATHENA Candidate Hypothesis Generation (Milestone M5)" in md
        assert "CANDIDATE HYPOTHESES" in md
        assert "[H1] 2D Capping Layer Passivation" in md
        assert "Target Gap:  [G1]" in md
        assert "Variables:   Independent: [PEAI concentration] | Dependent: [PCE retention (%)]" in md
        assert "Falsify If:  If degradation rate equals unpassivated control, hypothesis is refuted." in md
        assert "Grounding Score:           100.0%" in md
        assert "Falsifiable Hypotheses:    1 / 1" in md
        assert "[W2301656337-abs-1] Cesium-containing triple cation perovskite solar cells" in md

    def test_candidate_hypothesis_set_empty_markdown(self):
        report = HypothesisValidationReport(
            total_hypotheses=0,
            grounded_hypotheses=0,
            unsupported_hypotheses=0,
            falsifiable_count=0,
            total_citations=0,
            valid_citations=0,
            invalid_citations=0,
            invalid_gap_ids=[],
            invalid_claim_ids=[],
            invalid_evidence_ids=[],
            grounding_score=1.0,
        )
        hyp_set = CandidateHypothesisSet(
            research_question="test question",
            hypotheses=[],
            validation_report=report,
            evidence_references={},
            model_name="mock",
        )
        md = hyp_set.to_markdown()
        assert "No candidate hypotheses formulated from the available research gaps." in md


# ---------------------------------------------------------------------------
# 2. Prompt Builder Tests
# ---------------------------------------------------------------------------

class TestHypothesisPromptBuilder:
    def test_build_hypothesis_prompt_anchoring(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        prompt = build_hypothesis_prompt(
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert "Scientific Research Question:" in prompt
        assert "perovskite solar cells stability" in prompt
        assert "Candidate Research Gaps to Address:" in prompt
        assert "[G1] Moisture-induced degradation kinetics under operational humidity" in prompt
        assert "Synthesized Key Claims:" in prompt
        assert "[C1]" in prompt
        assert "Available Evidence Chunks:" in prompt
        assert "[W2301656337-abs-1]" in prompt
        assert "falsification_criteria" in prompt

    def test_system_prompt_rules(self):
        assert "target_gap_id" in HYPOTHESIS_SYSTEM_PROMPT
        assert "falsification_criteria" in HYPOTHESIS_SYSTEM_PROMPT
        assert "independent_variables" in HYPOTHESIS_SYSTEM_PROMPT
        assert "source_claim_ids" in HYPOTHESIS_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 3. Validator Tests
# ---------------------------------------------------------------------------

class TestHypothesisValidator:
    def test_validator_perfect_grounding(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        hyp = CandidateHypothesis(
            hypothesis_id="H1",
            target_gap_id="G1",
            title="Passivation hypothesis",
            statement="Passivation reduces degradation.",
            rationale="Addresses G1 based on C1.",
            proposed_mechanism="Steric barrier mechanism.",
            independent_variables=["Dopant concentration"],
            dependent_variables=["PCE retention"],
            falsification_criteria="Refuted if PCE degradation matches control.",
            source_claim_ids=["C1"],
            evidence_ids=["W2301656337-abs-1"],
        )
        dummy_report = HypothesisValidationReport(1, 0, 0, 0, 0, 0, 0, [], [], [], 0.0)
        hyp_set = CandidateHypothesisSet(
            research_question="perovskite solar cells stability",
            hypotheses=[hyp],
            validation_report=dummy_report,
            evidence_references={},
            model_name="mock",
        )

        report, refs = validate_hypothesis_set(
            hypothesis_set=hyp_set,
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert report.total_hypotheses == 1
        assert report.grounded_hypotheses == 1
        assert report.unsupported_hypotheses == 0
        assert report.falsifiable_count == 1
        assert report.grounding_score == 1.0
        assert len(report.invalid_gap_ids) == 0
        assert len(report.invalid_claim_ids) == 0
        assert len(report.invalid_evidence_ids) == 0
        assert hyp.is_grounded is True
        assert hyp.valid_gap_id is True
        assert "W2301656337-abs-1" in refs

    def test_validator_phantom_gap_detection(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        hyp = CandidateHypothesis(
            hypothesis_id="H1",
            target_gap_id="G999",  # Phantom gap
            title="Passivation hypothesis",
            statement="Passivation reduces degradation.",
            rationale="Addresses G999.",
            proposed_mechanism="Mechanism X.",
            independent_variables=["Var A"],
            dependent_variables=["Var B"],
            falsification_criteria="Refuted if X.",
            source_claim_ids=["C1"],
            evidence_ids=["W2301656337-abs-1"],
        )
        dummy_report = HypothesisValidationReport(1, 0, 0, 0, 0, 0, 0, [], [], [], 0.0)
        hyp_set = CandidateHypothesisSet(
            research_question="perovskite solar cells stability",
            hypotheses=[hyp],
            validation_report=dummy_report,
            evidence_references={},
            model_name="mock",
        )

        report, _ = validate_hypothesis_set(
            hypothesis_set=hyp_set,
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert report.grounded_hypotheses == 0
        assert report.unsupported_hypotheses == 1
        assert "G999" in report.invalid_gap_ids
        assert hyp.valid_gap_id is False
        assert hyp.is_grounded is False

    def test_validator_phantom_claim_and_evidence(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        hyp = CandidateHypothesis(
            hypothesis_id="H1",
            target_gap_id="G1",
            title="Passivation hypothesis",
            statement="Passivation reduces degradation.",
            rationale="Addresses G1.",
            proposed_mechanism="Mechanism X.",
            independent_variables=[],
            dependent_variables=[],
            falsification_criteria="Refuted if X.",
            source_claim_ids=["C999"],  # Phantom claim
            evidence_ids=["W999-abs-1"],  # Phantom evidence
        )
        dummy_report = HypothesisValidationReport(1, 0, 0, 0, 0, 0, 0, [], [], [], 0.0)
        hyp_set = CandidateHypothesisSet(
            research_question="perovskite solar cells stability",
            hypotheses=[hyp],
            validation_report=dummy_report,
            evidence_references={},
            model_name="mock",
        )

        report, _ = validate_hypothesis_set(
            hypothesis_set=hyp_set,
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert "C999" in report.invalid_claim_ids
        assert "W999-abs-1" in report.invalid_evidence_ids
        assert hyp.is_grounded is False
        assert report.grounding_score < 1.0


# ---------------------------------------------------------------------------
# 4. HypothesisGenerator Orchestrator Tests
# ---------------------------------------------------------------------------

class TestHypothesisGenerator:
    def test_generator_empty_gaps_early_exit(self, sample_synthesis):
        empty_gap_analysis = ResearchGapAnalysis(
            research_question="empty question",
            gaps=[],
            validation_report=GapValidationReport(0, 0, 0, 0, 0, 0, [], [], 1.0),
            evidence_references={},
            model_name="none",
        )
        generator = HypothesisGenerator()
        result = generator.generate(gap_analysis=empty_gap_analysis, synthesis=sample_synthesis)

        assert len(result.hypotheses) == 0
        assert result.validation_report.total_hypotheses == 0
        assert "no gaps to address" in result.model_name

    def test_generator_mock_client_complete(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        client = MockLLMClient()
        generator = HypothesisGenerator(client=client)
        result = generator.generate(
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert len(result.hypotheses) == 1
        h1 = result.hypotheses[0]
        assert h1.hypothesis_id == "H1"
        assert h1.target_gap_id == "G1"
        assert "Passivation" in h1.title
        assert len(h1.independent_variables) > 0
        assert len(h1.dependent_variables) > 0
        assert "refuted" in h1.falsification_criteria
        assert result.validation_report.grounding_score == 1.0
        assert result.validation_report.falsifiable_count == 1
        assert h1.is_grounded is True

    def test_generator_custom_json_in_markdown_fence(self, sample_gap_analysis, sample_synthesis, sample_evidence_chunk):
        custom_payload = {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "target_gap_id": "G1",
                    "title": "Dual-halide anion exchange mechanism",
                    "statement": "Bromide incorporation suppresses phase segregation under 1-sun illumination.",
                    "rationale": "Directly tackles coverage gap G1 by introducing halide mixing.",
                    "proposed_mechanism": "Smaller bromide ionic radius contracts the perovskite lattice cage.",
                    "independent_variables": ["Bromide ratio (Br/I)", "Light soaking intensity"],
                    "dependent_variables": ["Lattice constant (Angstroms)", "Trap state emission peak"],
                    "falsification_criteria": "Refuted if trap emission increases monotonically with Br fraction.",
                    "source_claim_ids": ["C1"],
                    "evidence_ids": ["W2301656337-abs-1"],
                }
            ]
        }
        markdown_wrapper = f"Here is the generated scientific hypothesis:\n```json\n{json.dumps(custom_payload)}\n```\nHope this helps!"
        client = MockLLMClient(response=markdown_wrapper)
        generator = HypothesisGenerator(client=client)

        result = generator.generate(
            gap_analysis=sample_gap_analysis,
            synthesis=sample_synthesis,
            evidence_chunks=[sample_evidence_chunk],
        )

        assert len(result.hypotheses) == 1
        h1 = result.hypotheses[0]
        assert h1.title == "Dual-halide anion exchange mechanism"
        assert h1.independent_variables == ["Bromide ratio (Br/I)", "Light soaking intensity"]
        assert h1.target_gap_id == "G1"
        assert h1.is_grounded is True


# ---------------------------------------------------------------------------
# 5. Full End-to-End Pipeline Integration Test (M1 -> M2 -> M3 -> M4 -> M5)
# ---------------------------------------------------------------------------

class TestEndToEndPipelineM1ToM5:
    def test_full_pipeline_flow(self):
        # M1: Paper representation
        paper = Paper(
            title="Halide Perovskite Ingot Growth and Thermomechanical Endurance",
            openalex_id="https://openalex.org/W5555",
            source="OpenAlex",
            authors=["Elena Rossi", "Liam Chen"],
            publication_year=2024,
            doi="https://doi.org/10.1000/5555",
            abstract="Vertical Bridgman growth produces large single crystals with low trap density. Operational shear stresses under thermal cycling induce micro-cracks at boundary dislocations.",
            venue="Nature Materials",
            cited_by_count=45,
        )

        # M2: Document processing & chunking
        norm_doc = normalize_paper(paper)
        chunks = chunk_documents([norm_doc])
        assert len(chunks) >= 1

        # M2: Lexical retrieval
        index = EvidenceIndex()
        index.index_chunks(chunks)
        retrieved_matches = index.retrieve("shear stress thermal cycling", top_k=2)
        assert len(retrieved_matches) >= 1

        # M3: Synthesis
        synth_json = {
            "overview": "Bridgman growth produces low-trap single crystals but experiences shear-induced micro-cracks.",
            "key_findings": [
                {
                    "claim_id": "C1",
                    "text": "Thermal cycling generates shear stresses that nucleate micro-cracks at boundary dislocations.",
                    "evidence_ids": [chunks[0].chunk_id],
                }
            ],
            "conflicting_findings": [],
            "limitations": ["Micro-crack propagation dynamics under long-term thermal gradients remain unmeasured."],
        }

        # M4: Research Gaps
        gap_json = {
            "gaps": [
                {
                    "gap_id": "G1",
                    "title": "Dislocation pinning mechanisms in perovskite ingots",
                    "description": "Techniques to pin dislocations and suppress micro-crack propagation under cyclic shear stress are uninvestigated.",
                    "gap_type": "methodological",
                    "rationale": "C1 establishes that boundary dislocations nucleate micro-cracks during thermal cycling.",
                    "source_claim_ids": ["C1"],
                    "evidence_ids": [chunks[0].chunk_id],
                }
            ]
        }

        # M5: Candidate Hypotheses
        hyp_json = {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "target_gap_id": "G1",
                    "title": "Isovalent B-site alloy pinning for shear stress resilience",
                    "statement": "Incorporating 2 mol% isovalent tin (Sn2+) into lead halide perovskite ingots pins boundary dislocations, decreasing thermal micro-crack density by >60%.",
                    "rationale": "Resolves Gap G1 by introducing local atomic strain fields to arrest dislocation propagation identified in Claim C1.",
                    "proposed_mechanism": "Atomic size mismatch between Sn and Pb creates localized elastic stress fields that impede dislocation motion.",
                    "independent_variables": ["Sn2+ dopant fraction (mol%)", "Thermal cycling count (1 to 500 cycles)"],
                    "dependent_variables": ["Micro-crack density (cracks/mm^2)", "Critical fracture toughness (K_IC)"],
                    "falsification_criteria": "Refuted if micro-crack density after 100 thermal cycles is greater than or equal to pure Pb controls (p > 0.05).",
                    "source_claim_ids": ["C1"],
                    "evidence_ids": [chunks[0].chunk_id],
                }
            ]
        }

        def mock_pipeline_llm(prompt, system_prompt):
            if "hypotheses" in (system_prompt or "").lower() or "hypothesis" in (system_prompt or "").lower():
                return json.dumps(hyp_json)
            if "discovering candidate research gaps" in (system_prompt or "").lower():
                return json.dumps(gap_json)
            return json.dumps(synth_json)

        mock_client = MockLLMClient(response_generator=mock_pipeline_llm)

        # M3: Synthesis execution
        synthesizer = Synthesizer(client=mock_client)
        evidence_ctx = EvidenceContext(
            query="shear stress thermal cycling",
            matches=retrieved_matches,
            total_papers=1,
            total_chunks=len(chunks),
        )
        synthesis = synthesizer.synthesize(
            research_question="perovskite ingot thermal shear stability",
            evidence_context=evidence_ctx,
        )
        assert synthesis.validation_report.grounding_score == 1.0

        # M4: Gap Analysis execution
        gap_analyzer = GapAnalyzer(client=mock_client)
        gap_analysis = gap_analyzer.analyze(
            synthesis=synthesis,
            evidence_chunks=chunks,
        )
        assert gap_analysis.validation_report.grounding_score == 1.0
        assert len(gap_analysis.gaps) == 1

        # M5: Hypothesis Generation execution
        hyp_generator = HypothesisGenerator(client=mock_client)
        hypothesis_set = hyp_generator.generate(
            gap_analysis=gap_analysis,
            synthesis=synthesis,
            evidence_chunks=chunks,
        )

        assert len(hypothesis_set.hypotheses) == 1
        h1 = hypothesis_set.hypotheses[0]
        assert h1.hypothesis_id == "H1"
        assert h1.target_gap_id == "G1"
        assert h1.source_claim_ids == ["C1"]
        assert h1.evidence_ids == [chunks[0].chunk_id]
        assert h1.is_grounded is True
        assert hypothesis_set.validation_report.grounding_score == 1.0
        assert hypothesis_set.validation_report.falsifiable_count == 1
        assert chunks[0].chunk_id in hypothesis_set.evidence_references


# ---------------------------------------------------------------------------
# 6. CLI Integration Test
# ---------------------------------------------------------------------------

class TestCLIWithHypothesisGeneration:
    @patch("app.main.search_papers")
    def test_cli_generate_hypotheses_flag(self, mock_search, capsys, tmp_path):
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

        export_file = tmp_path / "m5_export.json"

        exit_code = main([
            "perovskite single crystal",
            "--generate-hypotheses",
            "--llm-provider", "mock",
            "--export", str(export_file),
        ])

        assert exit_code == 0
        captured = capsys.readouterr()

        # Check that prerequisite milestones ran and were printed
        assert "ATHENA Scientific Evidence Synthesis (Milestone M3)" in captured.out
        assert "ATHENA Candidate Research-Gap Analysis (Milestone M4)" in captured.out
        assert "ATHENA Candidate Hypothesis Generation (Milestone M5)" in captured.out
        assert "CANDIDATE HYPOTHESES" in captured.out
        assert "Passivation layer optimization for prolonged operational resilience" in captured.out
        assert "Target Gap:  [G1]" in captured.out
        assert "Falsify If:" in captured.out

        # Verify export JSON includes candidate_hypotheses
        assert export_file.exists()
        with open(export_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "candidate_hypotheses" in data
        hyp_data = data["candidate_hypotheses"]
        assert hyp_data["total_hypotheses"] == 1
        assert hyp_data["hypotheses"][0]["hypothesis_id"] == "H1"
        assert hyp_data["hypotheses"][0]["target_gap_id"] == "G1"
        assert hyp_data["hypotheses"][0]["is_grounded"] is True
        assert hyp_data["validation_report"]["grounding_score"] == 1.0
