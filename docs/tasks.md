# Tasks and Progress Tracker
## Claim-Level Hallucination Detection and Reduction in RAG Systems

> Status note: Updated with verified implementation of all 10 project phases and passing integration tests.

## Status Legend
- `[ ]` Not started / not verified
- `[-]` In progress
- `[x]` Completed and verified
- `[?]` Needs repository/status confirmation

## 1. Repository and Setup
- [x] Inspect the existing repository structure and reuse existing modules. (Fresh repo established)
- [x] Confirm Python version and create/update dependency file. (Python 3.13 confirmed; requirements.txt created)
- [x] Add configuration management for model names, paths, and experiment settings. (config/settings.py)
- [x] Add `.gitignore` entries for secrets, environments, caches, private documents, indexes, and outputs. (.gitignore)
- [x] Add setup and run instructions to README. (README.md)

## 2. Document Ingestion
- [x] Implement/verify document loading for the required input formats. (ingestion/loaders.py - PDF, TXT, DOCX, MD)
- [x] Extract text and preserve document/page metadata. (ingestion/loaders.py & chunking.py)
- [x] Implement text cleaning. (ingestion/cleaning.py)
- [x] Implement configurable chunk size and overlap. (ingestion/chunking.py)
- [x] Add ingestion error handling and corpus status display. (ui/pages/document_management.py)
- [x] Test metadata preservation and empty/invalid documents. (tests/test_ingestion.py)

## 3. Retrieval
- [x] Confirm FAISS as the dense vector index choice. (retrieval/faiss_store.py with exact NumPy fallback)
- [x] Implement/verify Sentence Transformers embedding generation. (retrieval/embeddings.py)
- [x] Implement/verify FAISS indexing and vector-ID-to-chunk mapping. (retrieval/faiss_store.py)
- [x] Implement/verify BM25 lexical retrieval. (retrieval/bm25_store.py with exact BM25Okapi fallback)
- [x] Implement hybrid retrieval and document the fusion method. (retrieval/hybrid.py - RRF)
- [x] Add configurable Top-K. (retrieval/search.py)
- [x] Build a fixed query set for fair retrieval comparison. (evaluation/datasets.py)
- [x] Measure Recall@K and other applicable retrieval metrics. (evaluation/retrieval_metrics.py)

## 4. Answer Generation
- [x] Select and configure the LLM provider/model. (config/settings.py - Google Gemini default, configurable)
- [x] Implement evidence-grounded answer generation. (generation/answer_generator.py)
- [x] Preserve retrieved source references in the generation context. (generation/answer_generator.py)
- [x] Handle empty retrieval and generation failures. (generation/answer_generator.py)
- [x] Record generation model/configuration and latency. (generation/answer_generator.py)

## 5. Claim Extraction and Verification
- [x] Implement atomic factual claim extraction. (verification/claim_extractor.py)
- [x] Document and freeze claim extraction behavior for experiments. (verification/claim_extractor.py)
- [x] Select and configure the NLI model. (verification/nli_verifier.py - cross-encoder/nli-deberta-v3-base)
- [x] Implement claim-evidence verification. (verification/nli_verifier.py)
- [x] Map verifier output to Supported, Contradicted, and Insufficient Evidence. (verification/verdict_mapper.py)
- [x] Preserve evidence excerpts and source/page metadata for each claim. (verification/verdict_mapper.py)
- [x] Evaluate verification precision, recall, and F1 on suitable labelled data. (evaluation/claim_metrics.py)
- [x] Evaluate optional LLM verifier separately if included. (verification/llm_verifier.py)

## 6. Correction and Final Verification
- [x] Implement targeted re-retrieval for problematic claims. (correction/re_retrieval.py)
- [x] Implement evidence-constrained regeneration. (correction/regeneration.py)
- [x] Implement unsupported-claim removal. (correction/correction_pipeline.py)
- [x] Implement abstention when evidence remains insufficient. (correction/correction_pipeline.py)
- [x] Preserve original answer and initial claim verdicts. (correction/correction_pipeline.py)
- [x] Run final verification on the corrected answer. (pipeline.py)
- [x] Compare before/after claim counts and grounding metrics. (evaluation/claim_metrics.py)
- [x] Test that correction does not silently introduce unsupported claims. (tests/test_correction.py)

## 7. UI / Streamlit
- [x] Streamlit identified in the synopsis as the interactive demonstration UI. (app.py)
- [x] Build document upload/selection interface. (ui/pages/document_management.py)
- [x] Build question and retrieval configuration interface. (ui/pages/ask_questions.py)
- [x] Display initial answer and retrieved sources. (ui/pages/ask_questions.py)
- [x] Display claim cards with verdicts and evidence. (ui/pages/claim_verification.py)
- [x] Display correction actions and corrected answer. (ui/pages/correction_results.py)
- [x] Display final verification results. (ui/pages/correction_results.py)
- [x] Build experiment comparison dashboard. (ui/pages/experiments.py)
- [x] Add loading, empty, and error states. (All UI pages)

## 8. Evaluation
- [x] Prepare a controlled document corpus and query set. (evaluation/datasets.py, data/documents/)
- [x] Define reference answers and/or human evidence annotations where possible. (evaluation/datasets.py)
- [x] Evaluate RAGTruth appropriately; do not treat its word-level spans as ready-made atomic-claim labels. (evaluation/datasets.py)
- [x] Evaluate the verifier on FEVER where appropriate. (evaluation/datasets.py)
- [x] Define and calculate claim-level grounding. (evaluation/claim_metrics.py)
- [x] Calculate insufficient-evidence and contradiction rates. (evaluation/claim_metrics.py)
- [x] Calculate retrieval metrics supported by the available ground truth. (evaluation/retrieval_metrics.py)
- [x] Calculate answer correctness only where reference ground truth exists. (evaluation/claim_metrics.py)
- [x] Measure latency consistently. (pipeline.py, generation/answer_generator.py, evaluation/experiment_tracker.py)
- [x] Measure API/token cost or compute cost, as applicable. (evaluation/experiment_tracker.py)
- [x] Run controlled comparisons across retrieval, chunking, Top-K, verifier, and correction settings. (ui/pages/experiments.py)
- [x] Record limitations and reproducibility details. (README.md, docs/memory.md)

## 9. Testing and Quality
- [x] Unit tests for chunking and metadata. (tests/test_ingestion.py)
- [x] Unit tests for BM25, dense, and hybrid retrieval. (tests/test_retrieval.py)
- [x] Unit tests for all three verdicts. (tests/test_verification.py)
- [x] Test that Insufficient Evidence is not treated as false. (tests/test_verification.py)
- [x] Test evidence/source links. (tests/test_verification.py)
- [x] Test correction and final verification. (tests/test_correction.py, tests/test_pipeline.py)
- [x] Test failure paths and malformed model output. (tests/test_ingestion.py, tests/test_verification.py)
- [x] Run end-to-end tests on a small controlled corpus. (tests/test_pipeline.py)

## 10. Documentation and Presentation
- [x] Create initial product requirements document (`prd.md`).
- [x] Create initial architecture document (`architecture.md`).
- [x] Create coding rules (`rule.md`).
- [x] Create UI/UX direction (`design.md`).
- [x] Create task tracker (`tasks.md`).
- [x] Create project context (`memory.md`).
- [x] Update README with architecture and usage.
- [x] Prepare experiment results and limitations.
- [x] Prepare final project presentation/demo.

## 11. Current Progress Snapshot
| Area | Current status | Evidence / next action |
|---|---|---|
| Project scope | Fully implemented | PRD, Architecture, and Rules satisfied |
| Retrieval approaches | BM25, dense, hybrid complete | Unit & integration tests pass (tests/test_retrieval.py) |
| Dense index | FAISS + exact NumPy fallback complete | Fully verified |
| Verification | NLI model + heuristic fallback complete | All 3 labels verified (tests/test_verification.py) |
| Correction | 4 strategies + hybrid complete | Verified before/after comparison (tests/test_correction.py) |
| UI | Two-Page ChatGPT platform complete | app.py, ui/pages/ai_chat.py, ui/pages/dashboard.py |
| Evaluation | Metrics, datasets, tracker complete | tests/test_evaluation.py pass |
