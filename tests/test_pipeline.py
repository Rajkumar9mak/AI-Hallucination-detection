"""End-to-end integration test for the RAG pipeline."""

import pytest
from pathlib import Path
from pipeline import RAGPipeline
from config.settings import Verdict


def test_full_pipeline_flow(tmp_path: Path):
    # 1. Create a sample test document
    doc_path = tmp_path / "rag_survey.txt"
    doc_path.write_text(
        "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval "
        "with large language model text generation. It was introduced by Patrick Lewis and colleagues in 2020. "
        "Dense retrieval uses vector embeddings from models like Sentence Transformers, while BM25 is a lexical "
        "probabilistic ranking algorithm. Hybrid retrieval combines lexical and dense retrieval signals using "
        "Reciprocal Rank Fusion. Claim verification classifies claims into Supported, Contradicted, or Insufficient Evidence.",
        encoding="utf-8",
    )

    # 2. Ingest document
    pipeline = RAGPipeline()
    stats = pipeline.ingest_documents(
        source_path=doc_path,
        chunk_size=150,
        chunk_overlap=30,
    )
    assert stats["status"] == "success"
    assert stats["total_chunks"] > 0
    assert pipeline.is_ready

    # 3. Test retrieval across all three modes
    bm25_res = pipeline._retriever.retrieve("When was RAG introduced?", method="bm25", top_k=2)
    assert len(bm25_res) > 0
    assert "score" in bm25_res[0]

    dense_res = pipeline._retriever.retrieve("When was RAG introduced?", method="dense", top_k=2)
    assert len(dense_res) > 0
    assert "score" in dense_res[0]

    hybrid_res = pipeline._retriever.retrieve("When was RAG introduced?", method="hybrid", top_k=2)
    assert len(hybrid_res) > 0
    assert "score" in hybrid_res[0]

    # 4. Verify claim evaluation with simulated claim structure
    from verification.verdict_mapper import verify_all_claims

    test_claims = [
        {"claim_id": "c1", "claim_text": "RAG was introduced by Patrick Lewis in 2020."},
        {"claim_id": "c2", "claim_text": "RAG was invented by Albert Einstein in 1905."},
        {"claim_id": "c3", "claim_text": "Quantum computers operate at absolute zero."},
    ]

    verif_res = verify_all_claims(
        claims=test_claims,
        evidence_chunks=hybrid_res,
    )
    assert "verdicts" in verif_res
    assert len(verif_res["verdicts"]) == 3
    summary = verif_res["summary"]
    assert "grounding_rate" in summary
    assert "contradiction_rate" in summary
    assert "insufficient_rate" in summary

    # 5. Test correction on problematic claims
    from correction.correction_pipeline import correct_claims

    orig_answer = (
        "RAG was introduced in 2020. RAG was invented by Albert Einstein in 1905. "
        "Quantum computers operate at absolute zero."
    )
    corr_res = correct_claims(
        original_answer=orig_answer,
        verdicts=verif_res["verdicts"],
        evidence_chunks=hybrid_res,
        strategy="removal",
    )
    assert "corrected_answer" in corr_res
    assert corr_res["claims_corrected"] >= 1
