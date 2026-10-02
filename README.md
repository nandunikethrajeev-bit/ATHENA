# ATHENA — Autonomous Scientific Intelligence

A human-supervised, AI-assisted scientific research system.

ATHENA is designed to help users investigate scientific questions by retrieving
scientific literature, organizing evidence, synthesizing findings across
studies, identifying candidate research gaps, generating candidate hypotheses,
and producing source-traceable research reports.

> **Status: active development (M0 — project foundation, M1 — scientific literature retrieval, M2 — document processing & evidence retrieval, and M3 — scientific evidence synthesis complete).**
> ATHENA can now retrieve real scientific literature from OpenAlex, clean scientific text, chunk and rank evidence using deterministic BM25 retrieval, and synthesize structured, evidence-grounded scientific claims with provenance validation.
> Subsequent stages (research-gap analysis, hypothesis generation, report generation) remain planned.

## What ATHENA is — and is not

ATHENA:

- Assists researchers; it does **not** replace scientists.
- Operates under human oversight at defined checkpoints.
- Grounds its outputs in retrieved scientific sources wherever possible.
- Clearly distinguishes between retrieved evidence, evidence-based synthesis,
  model inference, candidate research gaps, and candidate hypotheses.

ATHENA does **not**:

- Claim to autonomously discover scientifically validated knowledge.
- Present generated hypotheses or inferred gaps as established scientific facts.
- Fabricate results, citations, or scientific content.

## Evidence classes

Every important statement produced by ATHENA will carry one of these labels:

| Label | Meaning |
|---|---|
| **Source evidence** | Text retrieved from a scientific source, with provenance |
| **Synthesis** | Cross-study comparison derived from retrieved evidence |
| **Inference** | Model reasoning that goes beyond the retrieved evidence |
| **Candidate gap** | Hypothesized gap in the literature — *not* proof of novelty |
| **Candidate hypothesis** | Proposed, testable hypothesis — requires human evaluation |

## Planned workflow

```
Research Question
→ Research Planning
→ Literature Retrieval
→ Document Processing
→ Evidence Retrieval / RAG
→ Evidence Synthesis
→ Candidate Gap Analysis
→ Hypothesis Generation
→ Critic / Verification
→ Final Research Report
```

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| M0 | Environment & project foundation | ✅ complete |
| M1 | Scientific literature retrieval (free/public APIs) | ✅ complete |
| M2 | Document processing & evidence retrieval | ✅ complete |
| M3 | Scientific evidence synthesis & claim grounding | ✅ complete |
| M4 | Candidate research-gap analysis | planned |
| M5 | Candidate hypothesis generation | planned |
| M6 | Critic / evidence verification | planned |
| M7 | Research report generation | planned |
| M8 | Streamlit interface | planned |
| M9 | Evaluation, testing & demo preparation | planned |

## M1 — Scientific Literature Retrieval

Milestone M1 establishes ATHENA's literature retrieval layer, enabling researchers to search for relevant scientific papers directly from the command line using public scholarly infrastructure.

> **M1 retrieves scholarly metadata and available abstracts. It does not yet perform evidence synthesis, hypothesis generation, or research-gap discovery.**

### Why OpenAlex?
OpenAlex is chosen as the primary literature discovery engine because:
- It provides a comprehensive, fully open catalog of hundreds of millions of scholarly publications, authors, institutions, and venues.
- It requires no paid API subscription and offers open REST access.
- It supports structured metadata including DOIs, open-access status, inverted-index abstracts, and citation counts.
- It avoids scraping publisher websites or commercial search engines.

### What Information Is Retrieved?
Retrieved papers are mapped into an internal, canonical `Paper` data model preserving source provenance:
- **Title**: Full paper title (`str | None`)
- **Authors**: List of author names (`list[str]`)
- **Publication Year**: Integer publication year (`int | None`)
- **DOI**: Digital Object Identifier (`str | None`)
- **Abstract**: Human-readable text reconstructed from OpenAlex's inverted index (`str | None`)
- **Venue**: Journal, conference, or repository host (`str | None`)
- **OpenAlex ID**: Scholarly work identifier (`str`, e.g. `https://openalex.org/W...`)
- **Citations**: Total citation count (`int | None`)
- **Publication Type**: Work classification (`str | None`, e.g. `journal-article`)
- **Landing Page URL**: Primary access URL or DOI link (`str | None`)
- **Open Access**: Detailed OA status and fulltext PDF URL if available
- **Source**: Explicit provenance tag (`source = "OpenAlex"`)

Missing metadata fields are strictly represented as `None` without fabricating data.

### How to Run a Literature Search

```bash
# Basic search
python -m app.main "Can machine learning improve early detection of Alzheimer's disease?"

# Limit results and export metadata to JSON
python -m app.main "machine learning Alzheimer's disease" --max-results 5 --export data/results.json

# Using polite pool or API key
python -m app.main "CRISPR gene editing" --email user@example.com
```

### Running Tests

```bash
# Run deterministic, offline unit tests (no network required)
python -m pytest

# Run live integration test against OpenAlex API
python -m pytest -m integration
```

### Limitations
- **Abstract Availability**: Abstracts are subject to OpenAlex indexing coverage and publisher licensing; if an abstract is absent in OpenAlex, it is represented as `None`.
- **Search Rate Limits**: OpenAlex anonymous searches are subject to rate limiting during periods of elevated search cluster load. Users can set `OPENALEX_API_KEY` (a free key obtained from [openalex.org/settings/api](https://openalex.org/settings/api)) or provide `OPENALEX_EMAIL` for polite pool access.
- **Scope**: M1 only retrieves metadata and abstracts; evidence extraction, chunking, embeddings, and hypothesis generation are deferred to subsequent milestones.

## M2 — Document Processing & Evidence Retrieval

Milestone M2 converts retrieved literature into discrete, traceable evidence items and provides deterministic in-memory BM25 retrieval without requiring paid APIs or heavy vector databases.

### Key Capabilities:
- **Scientific Text Cleaning**: Normalizes typography and whitespace while strictly preserving chemical formulas ($H_2O$, $CO_2$), statistical values ($p < 0.05$, $95\% CI$), and scientific units ($mg/kg$, $\mu g/mL$).
- **Evidence Chunking**: Splits abstracts into cohesive sentence windows while respecting scientific abbreviations (`et al.`, `e.g.`, `Fig.`). Papers without indexed abstracts produce explicit `metadata` chunks, ensuring abstracts are never fabricated.
- **Evidence Classification**: Explicitly distinguishes between `metadata`, `abstract`, and `full_text` evidence classes.
- **Source Traceability**: Every chunk retains an immutable `EvidenceSource` containing its OpenAlex ID, DOI, landing page URL, title, authors, and year.
- **In-Memory BM25 Retrieval**: Fast, deterministic lexical retrieval using pure-Python standard library BM25Okapi ($k_1=1.5, b=0.75$).

### How to Run Literature Search with Evidence Extraction:

```bash
# Retrieve papers AND extract/rank evidence chunks
python -m app.main "Can machine learning improve early detection of Alzheimer's disease?" --extract-evidence

# Limit evidence chunks to top 3 and export both metadata and evidence
python -m app.main "Alzheimer biomarkers" --extract-evidence --top-k 3 --export data/evidence.json
```

## M3 — Scientific Evidence Synthesis

Milestone M3 transforms ranked evidence items into a structured, evidence-grounded scientific synthesis. ATHENA does **not** perform generic, ungrounded paper summarization. Instead, the model is strictly constrained to synthesize findings solely from controlled evidence chunks retrieved by M2, and every scientific claim is cross-validated against genuine chunk identifiers.

> **Scope Boundary**:
> M3 synthesizes evidence supplied by M2 and validates claim grounding.
> M3 does **not** perform candidate research-gap discovery (M4), hypothesis generation (M5), critic verification (M6), automated experimentation, or autonomous scientific discovery. Those capabilities belong to subsequent milestones.

### Key Capabilities:
- **Model-Agnostic LLM Layer**: Protocol-driven `LLMClient` supporting:
  - `mock`: 100% offline, zero-network, deterministic client for development and tests (no API key required).
  - `ollama`: Free local inference using OpenAI-compatible endpoints (`http://localhost:11434/v1`) without paid services.
  - `openai` / `openrouter`: Commercial or open-router endpoints using the existing `httpx` client.
- **Strict Grounding Prompts**: Prompts explicitly forbid hallucinating citations, authors, DOIs, or findings, and enforce structured JSON output.
- **Claim & Citation Validation Engine**:
  - Validates every cited evidence ID against input `EvidenceChunk` IDs.
  - Categorizes claims as `SUPPORTED`, `PARTIALLY_VALID`, or `UNSUPPORTED`.
  - Flags phantom/hallucinated citations in `invalid_evidence_ids`.
  - Computes an objective `grounding_score` ($\text{valid\_citations} / \max(1, \text{total\_citations})$).
- **Full Provenance Preservation**: Maps every verified citation directly back to its originating paper title, authors, publication year, venue, DOI, and OpenAlex ID.
- **Zero-Evidence Safe Fallback**: When no evidence is retrieved, ATHENA returns a structured notice of insufficient evidence without invoking the LLM.

### How to Run Evidence Synthesis:

```bash
# Offline synthesis using the mock provider (no API key or network required)
python -m app.main "perovskite solar cell efficiency" --extract-evidence --top-k 5 --synthesize --llm-provider mock

# Export synthesis and literature data to JSON
python -m app.main "CRISPR base editing" --synthesize --llm-provider mock --export data/synthesis.json

# Using a local Ollama server (e.g. llama3.1:8b)
python -m app.main "mRNA vaccine stability" --synthesize --llm-provider ollama --llm-model llama3.1:8b
```

## Project structure

```
ATHENA/
├── app/                  # Application package
│   ├── __init__.py       # Package metadata & milestone status
│   ├── main.py           # CLI entry point & demonstration
│   ├── retrieval/        # Literature retrieval subsystem (M1)
│   │   ├── __init__.py   # Retrieval API exports
│   │   ├── exceptions.py # Domain errors & rate limit types
│   │   ├── models.py     # Paper & OpenAccess data models
│   │   ├── openalex.py   # OpenAlex REST client & work parser
│   │   └── utils.py      # Abstract reconstructor & query validator
│   ├── evidence/         # Document processing & evidence retrieval (M2)
│   │   ├── __init__.py   # Evidence API exports
│   │   ├── cleaner.py    # Scientific text normalizer
│   │   ├── models.py     # NormalizedDocument, EvidenceChunk, EvidenceMatch
│   │   ├── normalizer.py # Paper to NormalizedDocument transformer
│   │   ├── chunker.py    # Scientific sentence splitter & chunker
│   │   ├── retriever.py  # In-memory BM25Okapi search engine
│   │   └── context.py    # Context assembly & citation formatting
│   └── synthesis/        # Scientific evidence synthesis (M3)
│       ├── __init__.py   # Synthesis API exports
│       ├── models.py     # ResearchSynthesis, SynthesizedClaim, EvidenceReference
│       ├── prompts.py    # Deterministic prompt construction & integrity clauses
│       ├── providers.py  # LLMClient protocol, MockLLMClient, OpenAILikeClient
│       ├── validators.py # Claim grounding & phantom citation validator
│       └── synthesizer.py# Pipeline orchestrator
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_foundation.py     # M0 foundation tests
│   ├── test_retrieval.py      # M1 unit tests (offline/mocked)
│   ├── test_openalex_live.py  # M1 live integration test
│   ├── test_evidence.py       # M2 evidence & retrieval unit tests
│   └── test_synthesis.py      # M3 synthesis & validation unit tests
├── data/                 # Research data (generated locally; not committed)
│   ├── raw/
│   └── processed/
├── docs/                 # Documentation
│   ├── README.md
│   └── M2_EVIDENCE_SYSTEM.md  # Detailed M2 architecture specification
├── .env.example          # Placeholder environment variables
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Getting started

Requirements: Python 3.12+ (developed on 3.14) and pip.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main      # verify the foundation starts
python -m pytest        # run the test suite
```

Windows (Git Bash) / Linux / macOS:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
python -m pytest
```

## Environment variables

Secrets and configuration are read from environment variables — never hard-coded
and never committed. Copy `.env.example` to `.env` (git-ignored) and fill values
in later milestones. M0 ships placeholders only; no API keys are required yet.

## Engineering principles

1. Every important scientific claim should be traceable to retrieved evidence.
2. Free-first: public scientific APIs, open-source models, local processing.
3. No fabricated results, fake citations, or placeholder outputs that look real.
4. Errors are surfaced, not hidden.
5. Minimal, deliberate dependencies, added milestone-by-milestone.
6. Components remain independently testable.
