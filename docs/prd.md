# Product Requirements Document (PRD)
## Project: Claim-Level Hallucination Detection and Reduction in RAG Systems

**Project type:** Minor academic project  
**Institution:** Acropolis Institute of Technology and Research, Department of Information Technology  
**Repository noted in synopsis:** https://github.com/VineetDode/RAG-HALLUCINATION-DETECTION

## 1. Product Overview
The product is an interactive Retrieval-Augmented Generation (RAG) application that answers questions using a controlled document collection, decomposes generated answers into atomic factual claims, checks each claim against retrieved evidence, and attempts targeted correction where evidence is contradictory or insufficient.

The system must make claim-level decisions explainable by retaining links between claims and the evidence used to assess them.

## 2. Problem
RAG systems can generate answers containing claims that are not supported by the retrieved documents. A response-level faithfulness score may indicate a problem but does not identify the specific statements responsible. Retrieval alone does not guarantee that every generated claim is grounded.

## 3. Goals
- Compare BM25, dense, and hybrid retrieval under a consistent evaluation protocol.
- Generate answers grounded in retrieved documents.
- Split answers into atomic factual claims.
- Classify claims as **Supported**, **Contradicted**, or **Insufficient Evidence**.
- Correct problematic claims using targeted re-retrieval, evidence-constrained regeneration, claim removal, or abstention.
- Verify the corrected answer again.
- Measure retrieval quality, grounding, verifier performance, answer correctness where references exist, latency, and inference cost.
- Present claim-to-evidence explanations through a Streamlit interface.

## 4. Non-Goals
- Claiming to be the first claim-level hallucination detection framework.
- Guaranteeing that every generated answer is factually correct.
- Treating every unsupported claim as false.
- Replacing human review in high-stakes domains.
- Claiming production-grade deployment or validated domain performance without experiments.

## 5. Target Users
- Students and researchers evaluating RAG systems.
- Users asking questions over an uploaded or curated document collection.
- Project evaluators who need to inspect retrieval, verification, and correction results.

## 6. Core User Workflow
1. User uploads or selects a document collection.
2. System extracts and cleans document text.
3. Text is split into chunks; chunk size and overlap are configurable experimental variables.
4. The system creates embeddings and indexes chunks for dense retrieval; lexical indexing supports BM25.
5. User enters a question and selects a retrieval configuration.
6. System retrieves Top-K evidence and generates an initial answer.
7. System decomposes the answer into atomic factual claims.
8. Each claim is verified against relevant evidence.
9. The system displays verdicts and supporting/contradicting evidence.
10. Claims requiring correction trigger a configured correction strategy.
11. The corrected answer receives a final verification pass.
12. The interface displays the original answer, corrected answer, claim-level results, evidence, and evaluation measurements.

## 7. Functional Requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Accept a controlled document collection for ingestion. | Must |
| FR-02 | Extract, clean, chunk, and preserve document/page metadata. | Must |
| FR-03 | Generate embeddings using Sentence Transformers or an equivalent model. | Must |
| FR-04 | Support dense retrieval using FAISS. | Must |
| FR-05 | Support BM25 lexical retrieval. | Must |
| FR-06 | Support a hybrid retrieval configuration combining lexical and semantic signals. | Must |
| FR-07 | Allow configuration of Top-K and chunking parameters for experiments. | Must |
| FR-08 | Generate an initial answer using retrieved context and a configurable LLM. | Must |
| FR-09 | Extract atomic factual claims from the initial answer. | Must |
| FR-10 | Verify each claim using an NLI model and/or an LLM-based verifier. | Must |
| FR-11 | Return exactly one operational verdict per evaluated claim: Supported, Contradicted, or Insufficient Evidence. | Must |
| FR-12 | Preserve claim-to-evidence links and source metadata. | Must |
| FR-13 | Offer targeted re-retrieval, evidence-constrained regeneration, removal, or abstention. | Must |
| FR-14 | Run final verification after correction. | Must |
| FR-15 | Display initial and corrected answers with claim-level explanations. | Must |
| FR-16 | Record experiment configuration, metrics, latency, and cost. | Must |
| FR-17 | Support evaluation using RAGTruth and FEVER where appropriate. | Should |
| FR-18 | Export or save experiment results for comparison. | Should |

## 8. Verdict Definitions
- **Supported:** Retrieved evidence entails or adequately supports the claim.
- **Contradicted:** Retrieved evidence conflicts with the claim.
- **Insufficient Evidence:** Retrieved evidence does not provide enough information to support or refute the claim. This does not, by itself, mean the claim is false.

## 9. Non-Functional Requirements
- **Traceability:** Every verdict should be linked to the evidence considered.
- **Reproducibility:** Record retrieval method, Top-K, chunking configuration, model identifiers, and correction strategy.
- **Consistency:** Keep claim extraction and evaluation rules stable across experiments.
- **Reliability:** Handle empty retrieval results, parsing failures, and model errors without fabricating evidence.
- **Performance:** Measure latency consistently across configurations rather than assuming a target.
- **Privacy:** Do not expose uploaded document contents or API credentials in logs or UI output.
- **Maintainability:** Keep ingestion, retrieval, generation, verification, correction, evaluation, and UI responsibilities separated.

## 10. Evaluation Metrics
- Claim-level faithfulness/grounding: supported factual claims divided by total factual claims, under the project's operational definition.
- Insufficient-evidence rate.
- Contradiction rate.
- Claim-verification precision, recall, and F1 on datasets with ground-truth labels, such as FEVER where appropriate.
- Retrieval measures such as context precision, context recall, and Recall@K, depending on available ground truth.
- Answer correctness only when suitable reference answers or human-validated ground truth are available.
- Response latency and inference cost; report API/token cost when applicable, otherwise compute cost.

## 11. Dataset and Evaluation Notes
- **RAGTruth:** RAG-specific hallucination resource with word-level hallucination spans; do not describe it as direct atomic-claim ground truth for every response.
- **FEVER:** Auxiliary claim-verification benchmark with Supported, Refuted, and Not Enough Information labels and evidence.
- **Controlled corpus:** Prepare a document collection and query set with reference answers and/or human evidence annotations for end-to-end experiments.

## 12. Acceptance Criteria
- A user can query an indexed document collection.
- The system can run BM25, dense, and hybrid retrieval using the same query set.
- The generated answer is decomposed into atomic claims.
- Each claim receives one of the three defined verdicts.
- Evidence used for each verdict is inspectable.
- A correction strategy can be run and the resulting answer is verified again.
- Experiments record configuration and the defined metrics.
- The UI clearly distinguishes unsupported evidence from proven falsehood.
