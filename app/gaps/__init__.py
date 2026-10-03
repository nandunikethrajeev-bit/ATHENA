"""ATHENA Candidate Research-Gap Analysis Subsystem (Milestone M4).

Transforms evidence-grounded scientific synthesis into structured,
categorized, and provenance-anchored candidate research gaps.
"""

from app.gaps.analyzer import GapAnalysisError, GapAnalyzer
from app.gaps.models import (
    CandidateGap,
    GapType,
    GapValidationReport,
    ResearchGapAnalysis,
)
from app.gaps.prompts import GAP_SYSTEM_PROMPT, build_gap_analysis_prompt
from app.gaps.validators import validate_gap_analysis

__all__ = [
    "CandidateGap",
    "GAP_SYSTEM_PROMPT",
    "GapAnalysisError",
    "GapAnalyzer",
    "GapType",
    "GapValidationReport",
    "ResearchGapAnalysis",
    "build_gap_analysis_prompt",
    "validate_gap_analysis",
]
