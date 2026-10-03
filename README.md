# ATHENA — Autonomous Scientific Intelligence

A human-supervised, AI-assisted scientific research system.

ATHENA is designed to help users investigate scientific questions by retrieving
scientific literature, organizing evidence, synthesizing findings across
studies, identifying candidate research gaps, generating candidate hypotheses,
and producing source-traceable research reports.

> **Status: active development (M0 — project foundation, M1 — scientific literature retrieval, M2 — document processing & evidence retrieval, M3 — scientific evidence synthesis, and M4 — candidate research-gap analysis complete).**
> ATHENA can now retrieve real scientific literature from OpenAlex, clean scientific text, chunk and rank evidence using deterministic BM25 / dense semantic / hybrid retrieval, synthesize structured evidence-grounded claims, and discover validated candidate research gaps anchored directly to scientific evidence.
> Subsequent stages (hypothesis generation, critic verification, report generation) remain planned.

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
| M4 | Candidate research-gap analysis & validation | ✅ complete |
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

## M3 — Persistent Storage, Dense Embeddings, Hybrid Retrieval & Synthesis

Milestone M3 advances ATHENA from a transient, lexical-only prototype into a persistent, multi-modal evidence retrieval and synthesis engine:

1. **Persistent Evidence Storage**:
   - Backed by pure Python standard-library `sqlite3` (no external database server needed, zero C-compiler dependencies).
   - Preserves complete document provenance: OpenAlex ID, DOI, source URL, title, authors, publication year, venue, evidence type, section, and text.
2. **Dense Vector Embeddings & Semantic Retrieval**:
   - Local, model-agnostic `EmbeddingClient` protocol.
   - `MockEmbeddingClient`: 100% deterministic, offline, zero-network unit-normalized vector generation.
   - `OllamaEmbeddingClient`: Integrates with local Ollama (`qwen2.5:7b`, `nomic-embed-text`) using standard HTTP without paid APIs.
   - Computes exact cosine similarities with pure-Python vector math.
3. **Hybrid Retrieval (BM25 + Dense Vector RRF)**:
   - Merges lexical BM25Okapi keyword scores with dense semantic vector similarities using **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF\_Score}(d) = \frac{w_{\text{lex}}}{60 + r_{\text{lex}}(d)} + \frac{w_{\text{sem}}}{60 + r_{\text{sem}}(d)}$$
   - Seamlessly returns ranked `EvidenceMatch` items compatible with downstream synthesis.
4. **Structured Scientific Synthesis & Claim Grounding**:
   - Strictly validates all generated claim citations against genuine chunk IDs.
   - Computes `grounding_score`, isolates phantom citations, and resolves full paper provenance.

> **Scope Boundary**:
> M3 retrieves, stores, indexes, and synthesizes evidence supplied by M1/M2.
> M3 does **not** perform candidate research-gap discovery (M4), hypothesis generation (M5), critic verification (M6), automated experimentation, or autonomous scientific discovery. Those belong to subsequent milestones.

### How to Run:

```bash
# 1. Hybrid retrieval (BM25 + Semantic RRF) with mock provider (100% offline)
python -m app.main "perovskite solar cell efficiency" --extract-evidence --retrieval-mode hybrid --top-k 5

# 2. Persist extracted evidence chunks to SQLite database
python -m app.main "CRISPR gene therapy" --extract-evidence --persist-evidence --db-path data/athena_evidence.db

# 3. Hybrid retrieval + Synthesis with mock LLM (offline demo)
python -m app.main "mRNA vaccine stability" --extract-evidence --retrieval-mode hybrid --synthesize --llm-provider mock

# 4. Hybrid retrieval + Synthesis with local Ollama
python -m app.main "Alzheimer biomarkers" --extract-evidence --retrieval-mode hybrid --embedding-provider ollama --synthesize --llm-provider ollama --llm-model qwen2.5:7b
```

## M4 — Candidate Research-Gap Analysis & Validation

Milestone M4 enables ATHENA to discover, structure, and validate candidate scientific research gaps from synthesized literature. Rather than generating ungrounded speculative ideas, M4 strictly grounds candidate research gaps in synthesized claims ($C_1$, $C_2$, etc.) and original evidence chunks ($W...-abs-...$).

### Key Capabilities:
1. **Taxonomy-Governed Gap Discovery**:
   - `CONTRADICTION`: Conflicting findings or unresolved empirical disputes across literature.
   - `METHODOLOGICAL`: Limitations in experimental protocols, assay designs, sample sizes, animal models, or lack of negative controls.
   - `COVERAGE_SCOPE`: Unexplored materials, untested conditions, operational regimes, or unstudied populations.
   - `UNVERIFIED_CLAIM`: Mechanistic claims, hypothesized pathways, or theoretical models lacking empirical validation.
2. **Deterministic Dual Anchoring**:
   - Every candidate gap must cite both its supporting synthesis claim IDs (`source_claim_ids`) and underlying evidence chunk IDs (`evidence_ids`).
3. **Rigorous Grounding Validation**:
   - `GapValidationReport` validates cited claim IDs and evidence chunk IDs, detects invalid/phantom IDs, and computes an automated `grounding_score`.
   - Resolves full bibliographic metadata (`DOI`, title, authors, publication year) for every cited evidence chunk.
4. **Epistemic Humility & Guardrails**:
   - Explicitly framed as candidate gaps within the retrieved literature subset, not proof of universal scientific absence or novelty.
   - Gaps lacking supporting citations are flagged as `UNSUPPORTED CITATION(S)`.

> **Scope Boundary**:
> M4 identifies candidate research gaps in the synthesized literature.
> M4 does **not** generate candidate hypotheses (M5), perform automated experimentation, or autonomously draft publication manuscripts. Those belong to subsequent milestones.

### How to Run Research-Gap Analysis:

```bash
# 1. Full pipeline: Retrieval -> Evidence -> Synthesis -> Research Gaps (Mock LLM, 100% offline)
python -m app.main "perovskite solar cells stability" --extract-evidence --synthesize --analyze-gaps --llm-provider mock

# 2. Filter gaps by taxonomy type (e.g. methodological gaps)
python -m app.main "mRNA vaccine delivery" --analyze-gaps --gap-type methodological --llm-provider mock

# 3. Export end-to-end results including papers, evidence, synthesis, and research gaps to JSON
python -m app.main "solid state lithium battery" --analyze-gaps --llm-provider mock --export data/gap_analysis.json

# 4. Live local analysis using Ollama (qwen2.5:7b)
python -m app.main "CRISPR off-target detection" --analyze-gaps --llm-provider ollama --llm-model qwen2.5:7b
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
│   ├── evidence/         # Document processing, storage & retrieval (M2 & M3)
│   │   ├── __init__.py   # Evidence API exports
│   │   ├── cleaner.py    # Scientific text normalizer
│   │   ├── models.py     # NormalizedDocument, EvidenceChunk, EvidenceMatch
│   │   ├── normalizer.py # Paper to NormalizedDocument transformer
│   │   ├── chunker.py    # Scientific sentence splitter & chunker
│   │   ├── store.py      # Persistent SQLite evidence storage (M3)
│   │   ├── embeddings.py # Local & mock dense embedding layer (M3)
│   │   ├── vector_index.py# Pure-Python dense vector index & cosine retrieval (M3)
│   │   ├── retriever.py  # In-memory BM25Okapi search engine (M2)
│   │   ├── hybrid.py     # Hybrid BM25 + Vector RRF retrieval engine (M3)
│   │   └── context.py    # Context assembly & citation formatting
│   ├── synthesis/        # Scientific evidence synthesis (M3)
│   │   ├── __init__.py   # Synthesis API exports
│   │   ├── models.py     # ResearchSynthesis, SynthesizedClaim, EvidenceReference
│   │   ├── prompts.py    # Deterministic prompt construction & integrity clauses
│   │   ├── providers.py  # LLMClient protocol, MockLLMClient, OpenAILikeClient
│   │   ├── validators.py # Claim grounding & phantom citation validator
│   │   └── synthesizer.py# Pipeline orchestrator
│   └── gaps/             # Candidate research-gap analysis (M4)
│       ├── __init__.py   # Gap API exports
│       ├── models.py     # CandidateGap, GapType, GapValidationReport, ResearchGapAnalysis
│       ├── prompts.py    # Deterministic gap prompt builder & integrity clauses
│       ├── validators.py # Provenance validator & grounding auditor
│       └── analyzer.py   # Gap discovery orchestrator
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_foundation.py     # M0 foundation tests
│   ├── test_retrieval.py      # M1 unit tests (offline/mocked)
│   ├── test_openalex_live.py  # M1 live integration test
│   ├── test_evidence.py       # M2 evidence processing tests
│   ├── test_storage.py        # M3 persistent SQLite storage tests
│   ├── test_retrieval_m3.py   # M3 embeddings, vector & hybrid retrieval tests
│   ├── test_synthesis.py      # M3 synthesis & validation unit tests
│   └── test_gaps.py           # M4 candidate research gap unit & integration tests
├── data/                 # Research data (generated locally; not committed)
│   ├── athena_evidence.db     # Local SQLite persistent evidence store
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
