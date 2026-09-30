# ATHENA — Autonomous Scientific Intelligence

A human-supervised, AI-assisted scientific research system.

ATHENA is designed to help users investigate scientific questions by retrieving
scientific literature, organizing evidence, synthesizing findings across
studies, identifying candidate research gaps, generating candidate hypotheses,
and producing source-traceable research reports.

> **Status: early development (M0 — project foundation & M1 — scientific literature retrieval complete).**
> ATHENA can now retrieve real scientific literature and abstracts from OpenAlex.
> Subsequent stages (evidence ingestion, RAG, synthesis, hypothesis generation) remain planned.

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
| M2 | Paper metadata & document processing | planned |
| M3 | Evidence retrieval / RAG (local embeddings & index) | planned |
| M4 | Evidence synthesis | planned |
| M5 | Candidate research-gap analysis | planned |
| M6 | Candidate hypothesis generation | planned |
| M7 | Critic / evidence verification | planned |
| M8 | Research report generation | planned |
| M9 | Streamlit interface | planned |
| M10 | Evaluation, testing & demo preparation | planned |

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

## Project structure

```
ATHENA/
├── app/                  # Application package
│   ├── __init__.py       # Package metadata & milestone status
│   ├── main.py           # CLI entry point & demonstration
│   └── retrieval/        # Literature retrieval subsystem (M1)
│       ├── __init__.py   # Retrieval API exports
│       ├── exceptions.py # Domain errors & rate limit types
│       ├── models.py     # Paper & OpenAccess data models
│       ├── openalex.py   # OpenAlex REST client & work parser
│       └── utils.py      # Abstract reconstructor & query validator
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_foundation.py     # M0 foundation tests
│   ├── test_retrieval.py      # M1 unit tests (offline/mocked)
│   └── test_openalex_live.py  # M1 live integration test
├── data/                 # Research data (generated locally; not committed)
│   ├── raw/
│   └── processed/
├── docs/                 # Documentation
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
