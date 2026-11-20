# Project Architecture
## Claim-Level Hallucination Detection and Reduction in RAG Systems

## 1. Architecture Summary
The application follows a modular RAG pipeline with an additional claim-level verification and correction loop.

```text
                 ┌──────────────────────┐
                 │   Streamlit UI       │
                 │ Upload / Query /     │
                 │ Results / Experiments│
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Application /       │
                 │ Pipeline Controller │
                 └──────────┬───────────┘
                            │
       ┌────────────────────▼────────────────────┐
       │             Document Ingestion          │
       │ Parse → Clean → Chunk → Metadata        │
       └────────────────────┬────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
    ┌────────▼────────┐           ┌────────▼────────┐
    │ Embedding Model │           │ BM25 Index      │
    │ Sentence        │           │ Lexical Search  │
    │ Transformers    │           └────────┬────────┘
    └────────┬────────┘                    │
             │                             │
    ┌────────▼────────┐                    │
    │ FAISS Vector    │                    │
    │ Index           │                    │
    └────────┬────────┘                    │
             └──────────────┬──────────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Retrieval Selector   │
                 │ Dense / BM25 / Hybrid│
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Context + LLM       │
                 │ Initial Answer      │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Atomic Claim         │
                 │ Extraction           │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Claim Verification   │
                 │ NLI / optional LLM   │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Verdict + Evidence   │
                 │ Supported /          │
                 │ Contradicted /       │
                 │ Insufficient Evidence│
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Correction Engine   │
                 │ Re-retrieve /        │
                 │ Regenerate / Remove /│
                 │ Abstain              │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Final Verification   │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Final Answer +       │
                 │ Claim Evidence Report│
                 └──────────────────────┘
```

## 2. Main Components

### 2.1 User Interface
**Technology:** Streamlit  
Responsibilities:
- Document upload/selection.
- Query input and retrieval configuration.
- Display initial and corrected answers.
- Display claim verdicts and evidence.
- Display experiment metrics and comparisons.

### 2.2 Document Ingestion
Responsibilities:
- Read supported document formats.
- Extract text and preserve source/page metadata.
- Clean extracted text without removing meaningful content.
- Split text into chunks.
- Store chunk identifiers and metadata.

Chunk size and overlap must remain configurable because they are experimental variables.

### 2.3 Embedding and Indexing
**Technology:** Sentence Transformers + FAISS  
- Generate dense vectors for each chunk.
- Build a FAISS index for similarity search.
- Maintain a mapping from vector IDs to original chunks and metadata.

### 2.4 Lexical Retrieval
**Technology:** BM25 implementation  
- Tokenize and index document chunks.
- Rank chunks using lexical relevance.
- Return chunk text, source metadata, and scores.

### 2.5 Hybrid Retrieval
Combine lexical and dense retrieval signals. The fusion method and parameters must be recorded in experiment configuration. Use the same query set and evaluation protocol for fair comparison.

### 2.6 Answer Generation
A configurable LLM receives the user query and selected evidence. The prompt must instruct the model to use the supplied evidence and avoid inventing unsupported details.

### 2.7 Claim Extraction
Convert the initial answer into atomic factual claims. Keep extraction behavior documented and consistent across experiments because changing the claim set affects measured faithfulness.

### 2.8 Claim Verification
Input:
- One atomic claim.
- Relevant retrieved evidence.

The NLI model estimates entailment, contradiction, or neutrality. An optional LLM verifier may provide a complementary assessment. The system maps the assessment to:
- Supported
- Contradicted
- Insufficient Evidence

The verifier must be evaluated on labelled data; neither NLI nor an LLM judge should be assumed perfectly reliable.

### 2.9 Correction Engine
For Contradicted or Insufficient Evidence claims, configurable actions include:
1. Targeted re-retrieval.
2. Evidence-constrained regeneration.
3. Removal of unsupported claims.
4. Abstention when adequate evidence cannot be found.

Correction must not introduce claims without evidence.

### 2.10 Final Verification
Run the claim extraction/verification process again on the corrected answer. Compare before/after results to determine whether grounding improved and whether new errors were introduced.

### 2.11 Evaluation and Experiment Tracking
Record:
- Dataset and query identifier.
- Retrieval strategy and Top-K.
- Chunk size and overlap.
- Embedding and verifier model identifiers.
- Generation model/configuration.
- Correction strategy.
- Retrieval, grounding, verification, latency, and cost metrics.

## 3. Suggested Module Boundaries
```text
project/
├── app.py                       # Streamlit entry point
├── config/
│   └── settings.py              # Model and experiment configuration
├── ingestion/
│   ├── loaders.py               # Document parsing
│   ├── cleaning.py              # Text cleaning
│   └── chunking.py              # Chunk creation + metadata
├── retrieval/
│   ├── embeddings.py             # Sentence Transformer wrapper
│   ├── faiss_store.py            # Dense index
│   ├── bm25_store.py             # BM25 index
│   ├── hybrid.py                 # Fusion/ranking
│   └── search.py                 # Retrieval interface
├── generation/
│   └── answer_generator.py       # Initial answer generation
├── verification/
│   ├── claim_extractor.py        # Atomic claim extraction
│   ├── nli_verifier.py           # NLI inference
│   └── verdict_mapper.py         # Operational verdict mapping
├── correction/
│   ├── re_retrieval.py           # Targeted retrieval
│   ├── regeneration.py           # Evidence-constrained regeneration
│   └── correction_pipeline.py    # Remove / abstain / regenerate
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── claim_metrics.py
│   ├── rag_evaluation.py
│   └── experiment_tracker.py
├── data/
│   ├── documents/
│   ├── indexes/
│   └── results/
├── tests/
└── requirements.txt
```
This is a proposed logical structure; adapt it to the existing repository rather than duplicating existing modules.

## 4. Key Data Contracts

### Document Chunk
```json
{
  "chunk_id": "string",
  "document_id": "string",
  "source_name": "string",
  "page_number": 1,
  "text": "chunk content",
  "metadata": {}
}
```

### Claim Verification Result
```json
{
  "claim_id": "claim_001",
  "claim_text": "atomic factual claim",
  "verdict": "Supported | Contradicted | Insufficient Evidence",
  "evidence": [
    {
      "chunk_id": "string",
      "source_name": "string",
      "page_number": 1,
      "text": "evidence excerpt"
    }
  ],
  "verifier": "model identifier",
  "confidence": null
}
```
Confidence should only be populated if the selected verifier provides a meaningful, documented score.

### Experiment Record
```json
{
  "experiment_id": "string",
  "retrieval_method": "bm25 | dense | hybrid",
  "top_k": 5,
  "chunk_size": null,
  "chunk_overlap": null,
  "embedding_model": "string",
  "verifier_model": "string",
  "correction_strategy": "string",
  "metrics": {},
  "latency_ms": null,
  "inference_cost": null
}
```

## 5. Failure Handling
- No documents indexed: request ingestion or corpus selection.
- No relevant chunks: state that evidence was not found; do not fabricate an answer.
- Parsing failure: report the affected file and continue only where safe.
- Model/API failure: show a clear error and do not label an unverified claim as Supported.
- No evidence after re-retrieval: remove the claim or abstain according to configuration.
- Final verification failure: clearly mark the answer as not fully verified.

## 6. Architecture Decisions and Constraints
- FAISS is the dense vector search component specified by the project synopsis.
- BM25 is the lexical baseline.
- Hybrid retrieval combines lexical and semantic signals.
- Streamlit is the demonstration UI.
- Sentence Transformers provides embeddings.
- NLI and optional LLM verification are complementary approaches to evaluate.
- LangChain and LlamaIndex are optional integrations, not mandatory dependencies.
- RAGAs can be used as supplementary evaluation, not as a replacement for claim-level ground truth.
