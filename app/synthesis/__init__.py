"""ATHENA Scientific Evidence Synthesis (Milestone M3).

Transforms retrieved and ranked scientific evidence into structured,
grounded claims, ensuring strict citation validation and complete provenance tracking.
"""

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
    LLMClient,
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

__all__ = [
    "ClaimStatus",
    "EvidenceContext",
    "EvidenceReference",
    "LLMClient",
    "LLMConfigurationError",
    "LLMProviderError",
    "MockLLMClient",
    "OpenAILikeClient",
    "ResearchSynthesis",
    "SYSTEM_PROMPT",
    "SynthesisParsingError",
    "SynthesizedClaim",
    "Synthesizer",
    "ValidationReport",
    "build_synthesis_prompt",
    "extract_json_payload",
    "format_evidence_block",
    "get_llm_client",
    "validate_synthesis",
]
