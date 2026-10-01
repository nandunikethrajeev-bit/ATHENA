# ATHENA Milestone M2 — Evidence Processing & Retrieval Foundation

## 1. M2 Objective

Milestone M2 establishes ATHENA's document normalization, scientific text cleaning, provenance-preserving chunking, and deterministic evidence retrieval layer.

It bridges the gap between raw literature retrieval (M1) and evidence-grounded scientific synthesis (M3+). Given retrieved scientific papers from OpenAlex, M2 normalizes the metadata, cleans and standardizes the scientific text, splits the text into discrete evidence chunks with complete provenance attribution, indexes the chunks in a local in-memory store, and retrieves the most relevant evidence chunks for any scientific research question using standard BM25Okapi ranking.

---

## 2. Architecture & Subsystem Layout

All M2 capabilities are encapsulated in `app/evidence/` with zero third-party AI frameworks or paid external APIs:

```
app/evidence/
├── __init__.py      # Public exports for the evidence subsystem
├── models.py        # EvidenceType, EvidenceSource, NormalizedDocument, EvidenceChunk, EvidenceMatch
├── cleaner.py       # Scientific text cleaner (preserves formulas, units, statistics)
├── normalizer.py    # Converts Paper models or raw dictionaries to NormalizedDocument
├── chunker.py       # Scientific sentence splitter & evidence chunk extractor
├── retriever.py     # Pure-Python, in-memory BM25Okapi relevance ranking engine
└── context.py       # Context assembly and citation-ready prompt formatter
```

---

## 3. Data Flow

```
Research Question ("Can machine learning improve early detection of Alzheimer's disease?")
      │
      ▼
[M1: search_papers()]
      │
      ▼
List of canonical [Paper] models
      │
      ▼
[M2: Document Normalization (app/evidence/normalizer.py)]
      │ Applies clean_scientific_text()
      │ Validates identifiers, authors, publication years, DOI, and OA status
      ▼
List of [NormalizedDocument] instances
      │
      ▼
[M2: Evidence Chunking (app/evidence/chunker.py)]
      ├── Paper WITH abstract:
      │     ├── 1. Metadata Chunk (type="metadata", section="metadata")
      │     └── 2. Abstract Chunks (type="abstract", section="abstract", 1-3 sentences)
      └── Paper WITHOUT abstract:
            └── 1. Metadata Chunk (type="metadata", explicit note: "Abstract not indexed")
      │
      ▼
Collection of [EvidenceChunk] instances (each stamped with immutable EvidenceSource)
      │
      ▼
[M2: In-Memory BM25 Indexing & Ranking (app/evidence/retriever.py)]
      - Tokenizes query and chunk text (preserves alphanumeric & hyphenated terms)
      - Computes BM25Okapi term weights (k1=1.5, b=0.75) with smoothed IDF
      - Ranks chunks and filters to top-K highest-scoring items
      │
      ▼
List of [EvidenceMatch] items (chunk, relevance score, matched terms)
      │
      ▼
[M2: Context Assembly (app/evidence/context.py)]
      - Generates citation-numbered evidence blocks [1], [2]
      - Emits integrity warnings for abstract/metadata evidence
      │
      ▼
Formatted Evidence Context (Ready for terminal display, export, or future LLM synthesis)
```

---

## 4. Evidence Representation & Data Models

### 4.1. EvidenceType (Enum)
Strictly differentiates the evidentiary source:
- `METADATA`: Bibliographical metadata (title, authors, venue, publication year, DOI, citation count, open access status).
- `ABSTRACT`: Sentences extracted directly from the peer-reviewed abstract.
- `FULL_TEXT`: Body sections (e.g. methods, results, discussion) when full text is ingested in later milestones.

> **Integrity Rule**: An abstract must **never** be labeled or represented as full text.

### 4.2. EvidenceSource (Dataclass)
Immutable provenance record attached to every single chunk:
- `openalex_id`: Canonical OpenAlex URI (e.g., `https://openalex.org/W3018492956`)
- `paper_title`: Cleaned paper title
- `publication_year`: Integer publication year
- `doi`: Digital Object Identifier URI
- `landing_page_url`: Access URL or DOI link
- `venue`: Journal or conference name
- `authors`: Tuple of author display names
- `source_database`: Originating catalog (default: `"OpenAlex"`)

### 4.3. EvidenceChunk (Dataclass)
A discrete, verifiable unit of scientific evidence:
- `chunk_id`: Deterministic unique identifier (e.g. `W3018492956-abs-1`, `W3018492956-meta-0`)
- `document_id`: Source document URI
- `source`: Associated `EvidenceSource` provenance record
- `evidence_type`: `EvidenceType` classification
- `section`: Document section (`"metadata"`, `"abstract"`, etc.)
- `text`: Cleaned text excerpt
- `char_count`: Total character length
- `word_count`: Total word count
- `chunk_index`: Sequential integer index within the document

---

## 5. Text Processing & Cleaning Rules

Scientific text contains specialized terminology that standard NLP pipelines often corrupt. `app/evidence/cleaner.py` enforces:

1. **Whitespace & Typography**:
   - Collapses irregular horizontal whitespace without destroying paragraph structure.
   - Converts typographical curly quotes (`“`, `”`, `‘`, `’`) to standard ASCII quotes.
   - Standardizes dashes (`—`, `–`) and non-breaking spaces (`\u00a0`).
2. **Scientific Notation Preservation**:
   - Statistical notations: $p < 0.05$, $p = 0.001$, $t = 3.42$, $F(1, 38) = 4.12$, $95\% \text{ CI } [1.02, 1.45]$.
   - Chemical formulae: $\text{H}_2\text{O}$, $\text{CO}_2$, $\text{Ca}^{2+}$.
   - Units and measurements: $25\text{ mg/kg}$, $10\ \mu\text{g/mL}$, $37\ ^\circ\text{C}$, $500\text{ nm}$.
   - Mathematical operators: $<, >, =, \pm, \%, \le, \ge$.

---

## 6. Chunking Strategy

Implemented in `app/evidence/chunker.py`:

1. **Scientific Sentence Splitting**:
   - Standard sentence splitters break upon encountering abbreviations like `et al.` or `Fig. 1`.
   - ATHENA masks known scientific abbreviations (`et al.`, `e.g.`, `i.e.`, `Fig.`, `Tab.`, `vs.`, `approx.`, `Dr.`, `Prof.`) and decimal numbers (e.g. `0.05`, `3.14`) with temporary tokens before splitting, then cleanly restores them.
2. **Cohesive Sentence Windows**:
   - Chunks are formed by grouping 1 to 3 related sentences up to a target size (~50–120 words).
   - Prevents fragmenting coherent scientific claims into single isolated clauses.
3. **Handling Papers Without Abstracts**:
   - Papers without indexed abstracts produce an explicit `METADATA` chunk containing an explicit notice: `[Note: Abstract text is not indexed in the source repository for this work.]`
   - Under no circumstances does ATHENA fabricate an abstract or generate synthetic text.

---

## 7. Retrieval Strategy (Pure-Python BM25Okapi)

Implemented in `app/evidence/retriever.py`:

- **Zero Heavy Infrastructure**: Runs entirely in-memory using Python's standard library (`math`, `re`, `collections.Counter`). No vector database, no C-compiler requirements, and full native support for Python 3.14.
- **BM25Okapi Scoring**:
  $$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  with $k_1 = 1.5, b = 0.75$, and smoothed positive IDF:
  $$\text{IDF}(q) = \ln\left(1 + \frac{N - df(q) + 0.5}{df(q) + 0.5}\right)$$
- **Scientific Tokenization**:
  - Handles alphanumeric tokens and compound hyphenated scientific terms (e.g. `CRISPR-Cas9`, `cross-validation`), indexing both compound and component tokens.
  - Filters out functional English stop words while preserving domain keywords (`p53`, `covid-19`, `rna`, `dna`).
- **Evidence-Type Filtering**:
  - Allows callers to filter matches by evidence type (e.g. searching only `ABSTRACT` chunks).

---

## 8. Limitations & Constraints

1. **Abstract-Level Evidence**:
   - Many scholarly papers indexed by OpenAlex provide only abstracts (and some closed-access works provide only metadata).
   - In M2, evidence is primarily derived from abstracts and metadata. ATHENA explicitly flags this in terminal output and data exports so users are never misled.
2. **Lexical Matching vs. Dense Semantic Matching**:
   - BM25 relies on exact term overlap and stemming; it does not capture conceptual synonyms (e.g. "memory impairment" vs. "cognitive deficit") unless query terms match.
   - Dense embeddings (M3) will complement BM25 in a hybrid retrieval setup.
3. **No Automatic Full-Text PDF Fetching**:
   - M2 accepts full-text sections if provided in structured input, but does not autonomously download or OCR publisher PDFs.

---

## 9. What M2 Does NOT Yet Implement

In accordance with milestone design principles, M2 strictly avoids premature implementation of later stages:
- **No LLM Evidence Synthesis** (Milestone M4)
- **No Candidate Research-Gap Discovery** (Milestone M5)
- **No Candidate Hypothesis Generation** (Milestone M6)
- **No Automated Evidence Verification / Critic** (Milestone M7)
- **No Full Autonomous Research Report Authoring** (Milestone M8)
- **No Streamlit Graphical User Interface** (Milestone M9)
- **No Heavy Vector Databases / PyTorch / Transformers** (Deferred until environment wheel support is verified)
