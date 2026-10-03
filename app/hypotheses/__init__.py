"""ATHENA Candidate Hypothesis Generation subsystem (Milestone M5).

Transforms candidate research gaps (M4), synthesized claims (M3), and primary evidence chunks (M2/M3)
into structured, testable, and falsifiable Candidate Hypotheses with tripartite provenance anchoring.
"""

from app.hypotheses.generator import HypothesisGenerationError, HypothesisGenerator
from app.hypotheses.models import (
    CandidateHypothesis,
    CandidateHypothesisSet,
    HypothesisValidationReport,
)
from app.hypotheses.prompts import HYPOTHESIS_SYSTEM_PROMPT, build_hypothesis_prompt
from app.hypotheses.validators import validate_hypothesis_set

__all__ = [
    "CandidateHypothesis",
    "CandidateHypothesisSet",
    "HypothesisGenerationError",
    "HypothesisGenerator",
    "HypothesisValidationReport",
    "HYPOTHESIS_SYSTEM_PROMPT",
    "build_hypothesis_prompt",
    "validate_hypothesis_set",
]
