# Tasks and Progress Tracker
## Claim-Level Hallucination Detection and Reduction in RAG Systems

> Status note: This tracker is initialized from the project synopsis and the project discussions available at the time these files were prepared. A topic being discussed does not prove that its implementation or testing is complete. Confirm each item against the repository before marking it Done.

## Status Legend
- `[ ]` Not started / not verified
- `[-]` In progress
- `[x]` Completed and verified
- `[?]` Needs repository/status confirmation

## 1. Repository and Setup
- [?] Inspect the existing repository structure and reuse existing modules.
- [ ] Confirm Python version and create/update dependency file.
- [ ] Add configuration management for model names, paths, and experiment settings.
- [ ] Add `.gitignore` entries for secrets, environments, caches, private documents, indexes, and outputs.
- [ ] Add setup and run instructions to README.

## 2. Document Ingestion
- [ ] Implement/verify document loading for the required input formats.
- [ ] Extract text and preserve document/page metadata.
- [ ] Implement text cleaning.
- [ ] Implement configurable chunk size and overlap.
- [ ] Add ingestion error handling and corpus status display.
- [ ] Test metadata preservation and empty/invalid documents.

## 3. Retrieval
- [?] Confirm FAISS as the dense vector index choice.
- [ ] Implement/verify Sentence Transformers embedding generation.
- [ ] Implement/verify FAISS indexing and vector-ID-to-chunk mapping.
- [ ] Implement/verify BM25 lexical retrieval.
- [ ] Implement hybrid retrieval and document the fusion method.
- [ ] Add configurable Top-K.
- [ ] Build a fixed query set for fair retrieval comparison.
- [ ] Measure Recall@K and other applicable retrieval metrics.

## 4. Answer Generation
- [ ] Select and configure the LLM provider/model.
- [ ] Implement evidence-grounded answer generation.
- [ ] Preserve retrieved source references in the generation context.
- [ ] Handle empty retrieval and generation failures.
- [ ] Record generation model/configuration and latency.

## 5. Claim Extraction and Verification
- [ ] Implement atomic factual claim extraction.
- [ ] Document and freeze claim extraction behavior for experiments.
- [ ] Select and configure the NLI model.
- [ ] Implement claim-evidence verification.
- [ ] Map verifier output to Supported, Contradicted, and Insufficient Evidence.
- [ ] Preserve evidence excerpts and source/page metadata for each claim.
- [ ] Evaluate verification precision, recall, and F1 on suitable labelled data.
- [ ] Evaluate optional LLM verifier separately if included.

## 6. Correction and Final Verification
- [ ] Implement targeted re-retrieval for problematic claims.
- [ ] Implement evidence-constrained regeneration.
- [ ] Implement unsupported-claim removal.
- [ ] Implement abstention when evidence remains insufficient.
- [ ] Preserve original answer and initial claim verdicts.
- [ ] Run final verification on the corrected answer.
- [ ] Compare before/after claim counts and grounding metrics.
- [ ] Test that correction does not silently introduce unsupported claims.

## 7. UI / Streamlit
- [?] Streamlit identified in the synopsis as the interactive demonstration UI.
- [ ] Build document upload/selection interface.
- [ ] Build question and retrieval configuration interface.
- [ ] Display initial answer and retrieved sources.
- [ ] Display claim cards with verdicts and evidence.
- [ ] Display correction actions and corrected answer.
- [ ] Display final verification results.
- [ ] Build experiment comparison dashboard.
- [ ] Add loading, empty, and error states.

## 8. Evaluation
- [ ] Prepare a controlled document corpus and query set.
- [ ] Define reference answers and/or human evidence annotations where possible.
- [ ] Evaluate RAGTruth appropriately; do not treat its word-level spans as ready-made atomic-claim labels.
- [ ] Evaluate the verifier on FEVER where appropriate.
- [ ] Define and calculate claim-level grounding.
- [ ] Calculate insufficient-evidence and contradiction rates.
- [ ] Calculate retrieval metrics supported by the available ground truth.
- [ ] Calculate answer correctness only where reference ground truth exists.
- [ ] Measure latency consistently.
- [ ] Measure API/token cost or compute cost, as applicable.
- [ ] Run controlled comparisons across retrieval, chunking, Top-K, verifier, and correction settings.
- [ ] Record limitations and reproducibility details.

## 9. Testing and Quality
- [ ] Unit tests for chunking and metadata.
- [ ] Unit tests for BM25, dense, and hybrid retrieval.
- [ ] Unit tests for all three verdicts.
- [ ] Test that Insufficient Evidence is not treated as false.
- [ ] Test evidence/source links.
- [ ] Test correction and final verification.
- [ ] Test failure paths and malformed model output.
- [ ] Run end-to-end tests on a small controlled corpus.

## 10. Documentation and Presentation
- [x] Create initial product requirements document (`prd.md`).
- [x] Create initial architecture document (`architecture.md`).
- [x] Create coding rules (`rule.md`).
- [x] Create UI/UX direction (`design.md`).
- [x] Create task tracker (`tasks.md`).
- [x] Create project context (`memory.md`).
- [ ] Update README with architecture and usage.
- [ ] Prepare experiment results and limitations.
- [ ] Prepare final project presentation/demo.

## 11. Current Progress Snapshot
| Area | Current status | Evidence / next action |
|---|---|---|
| Project scope | Defined in synopsis | Use PRD as baseline |
| Retrieval approaches | BM25, dense, hybrid specified | Verify implementation and benchmark |
| Dense index | FAISS specified/discussed | Confirm working index in repository |
| Verification | NLI and optional LLM verifier specified | Confirm model and test labels |
| Correction | Strategies defined in synopsis | Implement and verify end-to-end |
| UI | Streamlit specified | Confirm current screens and workflow |
| Evaluation | Metrics and datasets identified | Build reproducible evaluation runs |

## 12. Progress Update Rules
- Mark `[x]` only after implementation and verification.
- Mark `[-]` when work has started but is incomplete.
- Use `[?]` when implementation status is unknown.
- Add a short evidence note or commit/PR reference for completed tasks.
- Update the snapshot whenever a major component changes.
