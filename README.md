# Claim-Level Framework for Detecting and Reducing Hallucinations in RAG Systems

A Retrieval-Augmented Generation (RAG) application that compares retrieval strategies, decomposes generated answers into atomic factual claims, verifies each claim against retrieved evidence, applies targeted correction, and verifies the corrected answer again.

> **Academic Project** — Acropolis Institute of Technology and Research, Department of Information Technology.

---

## Overview

RAG systems can generate answers that contain claims unsupported by the retrieved documents. This project implements a **claim-level** pipeline that:

1. **Retrieves** relevant document chunks using BM25, dense (FAISS + Sentence Transformers), or hybrid retrieval.
2. **Generates** an evidence-grounded answer via a configurable LLM.
3. **Extracts** atomic factual claims from the answer.
4. **Verifies** each claim against evidence using an NLI model.
5. **Classifies** claims as **Supported**, **Contradicted**, or **Insufficient Evidence**.
6. **Corrects** problematic claims through targeted re-retrieval, regeneration, removal, or abstention.
7. **Re-verifies** the corrected answer.
8. **Evaluates** performance across retrieval methods, chunking strategies, and verification configurations.

## Architecture

```
Documents → Ingestion → Chunking → Indexing (FAISS + BM25)
                                        ↓
Query → Retrieval (BM25/Dense/Hybrid) → LLM Generation → Claim Extraction
                                                              ↓
                                                    Claim Verification (NLI)
                                                              ↓
                                                    Correction Engine
                                                              ↓
                                                    Final Verification
                                                              ↓
                                                    Answer + Evidence Report
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/VineetDode/RAG-HALLUCINATION-DETECTION.git
cd RAG-HALLUCINATION-DETECTION
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Streamlit entry point
├── config/
│   └── settings.py             # Centralized configuration
├── ingestion/                  # Document parsing, cleaning, chunking
├── retrieval/                  # BM25, FAISS dense, hybrid retrieval
├── generation/                 # LLM answer generation
├── verification/               # Claim extraction, NLI verification
├── correction/                 # Hallucination correction strategies
├── evaluation/                 # Metrics, experiment tracking
├── ui/pages/                   # Streamlit multi-page UI
├── tests/                      # Unit and integration tests
├── data/                       # Documents, indexes, results
└── docs/                       # Project documentation
```

## Verdict Definitions

| Verdict | Meaning |
|---|---|
| **Supported** | Retrieved evidence entails or adequately supports the claim. |
| **Contradicted** | Retrieved evidence conflicts with the claim. |
| **Insufficient Evidence** | Evidence does not provide enough information to support or refute the claim. This does **not** mean the claim is false. |

## Technologies

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| Dense Retrieval | FAISS + Sentence Transformers |
| Lexical Retrieval | BM25 (rank-bm25) |
| Answer Generation | Google Gemini (configurable) |
| Claim Verification | NLI (cross-encoder/nli-deberta-v3-base) |
| Evaluation | RAGAs + Custom Metrics |

## Documentation

- [Product Requirements](docs/prd.md)
- [Architecture](docs/architecture.md)
- [Coding Rules](docs/rule.md)
- [UI/UX Design](docs/design.md)
- [Task Tracker](docs/tasks.md)
- [Project Context](docs/memory.md)

## Team

- Shivam Patel, Utkarsh Singh, Rajkumar Makwana, Vineet Dode
- Guide: Prof. Kapil Sahu | External Guide: Harshal Jadhav

## License

Academic project — not licensed for production use.
