# Project Context / Working Memory
## Claim-Level Hallucination Detection and Reduction in RAG Systems

This file is the compact source of truth for future project work. Update it when a design decision, implementation detail, or experiment result changes.

## 1. Project Identity
- **Project title:** A Claim-Level Framework for Detecting and Reducing Hallucinations in Retrieval-Augmented Generation (RAG) Systems.
- **Project type:** Minor project.
- **Institution:** Acropolis Institute of Technology and Research, Department of Information Technology.
- **Repository listed in synopsis:** https://github.com/VineetDode/RAG-HALLUCINATION-DETECTION
- **Team listed in synopsis:** Shivam Patel, Utkarsh Singh, Rajkumar Makwana, Vineet Dode.
- **Guide listed in synopsis:** Prof. Kapil Sahu; external guide Harshal Jadhav; project incharge Prof. Monika Choudhary.

## 2. One-Line Description
A RAG application that compares retrieval strategies, verifies individual generated claims against retrieved evidence, applies targeted correction, and verifies the corrected answer again.

## 3. Core Research Focus
The project is not claiming that claim-level checking itself is new. Existing work such as RAGChecker and RAGVUE already explores claim-level RAG evaluation. The project's research focus is the experimental study of an end-to-end pipeline combining:
1. BM25, dense, and hybrid retrieval.
2. Claim extraction and verification.
3. Targeted correction.
4. Final post-correction verification.
5. Measurement of retrieval quality, grounding, verification performance, latency, and inference cost.

## 4. Required Claim Labels
- **Supported:** Retrieved evidence entails or adequately supports the claim.
- **Contradicted:** Retrieved evidence conflicts with the claim.
- **Insufficient Evidence:** Retrieved evidence does not provide enough information to support or refute the claim.

Important: Insufficient Evidence does not mean false. Unsupported claims and factual falsehoods must not be treated as interchangeable.

## 5. Pipeline
Document ingestion → text extraction/cleaning → chunking → embeddings/indexing → BM25/dense/hybrid retrieval → LLM answer generation → atomic claim extraction → claim verification → targeted correction → final verification → answer and evidence report.

## 6. Technology in the Synopsis
| Technology | Intended purpose |
|---|---|
| Python | Core development and experiment control |
| FAISS | Dense vector indexing and similarity search |
| Sentence Transformers | Embedding generation |
| BM25 | Lexical retrieval |
| NLI model | Claim-evidence entailment/contradiction/neutrality estimation |
| Configurable LLM | Answer generation and optional LLM verification |
| Streamlit | Interactive UI/demo |
| RAGAs | Supplementary RAG evaluation |
| LangChain | Optional orchestration/integration |
| LlamaIndex | Optional document ingestion/RAG integration |

Project discussions have also focused on FAISS, BM25, dense/hybrid retrieval, NLI, RAGTruth, FEVER, and Streamlit. Confirm exact package/model/provider choices from the current code before changing them.

## 7. Datasets and Their Proper Use
- **RAGTruth:** RAG-specific hallucination corpus with word-level hallucination annotations. It is not direct atomic-claim ground truth for every response.
- **FEVER:** Claim verification dataset with Supported, Refuted, and Not Enough Information labels and evidence; useful as an auxiliary verifier evaluation resource.
- **Controlled corpus:** Needed for end-to-end experiments with a query set and reference answers and/or human evidence annotations.

## 8. Evaluation Metrics
- Claim-level faithfulness/grounding.
- Insufficient-evidence rate.
- Contradiction rate.
- Claim-verification precision, recall, and F1 where labelled data exists.
- Retrieval metrics such as context precision, context recall, and Recall@K, depending on ground truth.
- Answer correctness only when suitable reference answers or human-validated ground truth exist.
- Response latency.
- Inference/API/token or compute cost.

## 9. Correction Strategies
- Targeted re-retrieval.
- Evidence-constrained regeneration.
- Removal of unsupported claims.
- Abstention when evidence remains insufficient.
- Final verification after correction is mandatory for measuring whether grounding improved.

## 10. Research and Design Guardrails
- Do not claim novelty for claim-level verification alone.
- Do not claim every unsupported statement is false.
- Do not present model confidence or similarity scores as guaranteed truth.
- Keep claim extraction consistent across experiments.
- Compare retrieval methods using the same corpus, queries, and evaluation protocol.
- Record chunk size, overlap, Top-K, model identifiers, correction strategy, latency, and cost.
- Treat high-stakes use cases as requiring domain validation and human oversight.

## 11. Literature Cited in the Synopsis
1. Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
2. Niu et al. (2024), RAGTruth.
3. Ru et al. (2024), RAGChecker.
4. Es et al. (2024), RAGAs.
5. Saad-Falcon et al. (2024), ARES.
6. Thorne et al. (2018), FEVER.
7. Murugaraj et al. (2026), RAGVUE.
8. Reimers & Gurevych (2019), Sentence-BERT.
9. Johnson, Douze & Jégou (2017), Billion-scale similarity search with GPUs.
10. Robertson & Zaragoza (2009), BM25 and Beyond.

## 12. Current Known Status
The synopsis defines the intended architecture, technology categories, research gap, datasets, metrics, and correction strategies. Prior project discussions have covered retrieval approaches, FAISS, NLI, RAGTruth/FEVER, Streamlit, and presentation preparation. Those discussions do not independently prove that components are implemented or passing tests.

Use `tasks.md` as the implementation status tracker. Confirm actual code, model identifiers, provider settings, and test results from the repository before treating them as facts.

## 13. How an AI Coding Assistant Should Work on This Project
1. Read this file and `prd.md`, `architecture.md`, and `rule.md` before making substantial changes.
2. Inspect the existing repository before proposing new files or replacing modules.
3. Preserve the research framing and three verdict definitions.
4. Make small, testable changes.
5. Do not invent benchmark results, citations, page numbers, or completed status.
6. Update tests and relevant documentation with each significant change.
7. Update `tasks.md` with verified progress.
