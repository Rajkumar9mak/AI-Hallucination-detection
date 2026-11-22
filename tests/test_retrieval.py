"""Unit tests for retrieval modules: BM25, FAISS, and Hybrid fusion."""

import numpy as np
import pytest
from retrieval.bm25_store import BM25Store
from retrieval.faiss_store import FAISSStore
from retrieval.hybrid import reciprocal_rank_fusion


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_01",
            "source_name": "ai_overview.txt",
            "page_number": 1,
            "text": "Retrieval-Augmented Generation enhances language models with external knowledge search.",
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_01",
            "source_name": "ai_overview.txt",
            "page_number": 1,
            "text": "BM25 is a popular probabilistic lexical ranking function used in information retrieval.",
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "doc_02",
            "source_name": "faiss_guide.txt",
            "page_number": 2,
            "text": "FAISS provides efficient dense vector similarity search and clustering of dense vectors.",
        },
    ]


def test_bm25_store(sample_chunks):
    store = BM25Store()
    store.build_index(sample_chunks)
    assert store.is_built
    assert store.total_documents == 3

    results = store.search("BM25 lexical ranking", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "chunk_002"
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_faiss_store(sample_chunks):
    store = FAISSStore()
    # Mock normalized embeddings with dimension 4
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    store.build_index(sample_chunks, embeddings)
    assert store.is_built
    assert store.total_vectors == 3

    # Query vector close to chunk_003
    query_vec = np.array([0.05, 0.0, 0.95, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_003"
    assert "score" in results[0]


def test_reciprocal_rank_fusion(sample_chunks):
    bm25_res = [
        {"chunk_id": "chunk_001", "text": sample_chunks[0]["text"]},
        {"chunk_id": "chunk_002", "text": sample_chunks[1]["text"]},
    ]
    dense_res = [
        {"chunk_id": "chunk_002", "text": sample_chunks[1]["text"]},
        {"chunk_id": "chunk_003", "text": sample_chunks[2]["text"]},
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_res,
        dense_results=dense_res,
        alpha=0.5,
        k_constant=60,
        top_k=3,
    )
    assert len(fused) == 3
    # chunk_002 appears in both, so it should rank first with highest RRF score
    assert fused[0]["chunk_id"] == "chunk_002"
    assert "score" in fused[0]
    assert "bm25_rank" in fused[0]
    assert "dense_rank" in fused[0]
