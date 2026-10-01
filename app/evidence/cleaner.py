"""Scientific text cleaning and normalization utilities for ATHENA (Milestone M2).

Normalizes whitespace, line breaks, encoding quirks, and typographical artifacts
while strictly preserving scientific notations, statistical formulas, chemical symbols,
and mathematical operators.
"""

import re
import unicodedata

# Unicode character normalization map
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u00a0": " ",      # non-breaking space
    "\u200b": "",       # zero-width space
    "\u200e": "",       # left-to-right mark
    "\u200f": "",       # right-to-left mark
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u2013": "-",      # en dash
    "\u2014": "-",      # em dash
    "\u2026": "...",    # ellipsis
    "\ufeff": "",       # BOM
    "\xad": "",         # soft hyphen
}

# Regex to collapse horizontal whitespace (tabs, consecutive spaces) without mangling newlines
_MULTI_SPACE_RE = re.compile(r"[^\S\r\n]+")
# Regex to collapse excessive vertical newlines (more than 2 newlines into 2)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_scientific_text(text: str | None) -> str:
    """Clean and normalize scientific text while preserving notation and scientific terms.

    Operations:
    1. Gracefully handles None or non-string inputs.
    2. Maps non-standard Unicode spaces and punctuation to ASCII equivalents.
    3. Normalizes unicode characters using standard NFKC form (preserves Greek letters, accents).
    4. Collapses erratic whitespace and linebreaks without erasing paragraph divisions.
    5. Preserves scientific terminology:
       - Statistical notations (p < 0.05, t = 2.14, 95% CI)
       - Chemical formulae (H2O, CO2, Ca2+)
       - Units of measurement (mg/kg, ug/mL, 37 deg C, 500 nm)
       - Mathematical inequalities and symbols (<, >, =, +, -, %, +/-)

    Args:
        text: Raw text string (abstract, section text, or title).

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Fast check for empty/whitespace-only string
    if not text.strip():
        return ""

    # 1. Replace known problematic Unicode characters
    for orig, repl in _UNICODE_REPLACEMENTS.items():
        if orig in text:
            text = text.replace(orig, repl)

    # 2. Apply NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Standardize linebreaks (\r\n -> \n, \r -> \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Collapse consecutive horizontal whitespace on each line
    lines = text.split("\n")
    cleaned_lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in lines]
    text = "\n".join(cleaned_lines)

    # 5. Collapse excessive linebreaks (max 2 consecutive newlines for paragraph breaks)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()
