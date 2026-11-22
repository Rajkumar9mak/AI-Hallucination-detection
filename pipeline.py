"""
Main pipeline orchestrator for the RAG Hallucination Detection system.

Provides the end-to-end pipeline that ties together ingestion,
retrieval, generation, verification, correction, and final verification.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from config.settings import get_config, Verdict, DOCUMENTS_DIR, INDEXES_DIR

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline with claim-level verification and correction.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_documents(Path("data/documents"))
        result = pipeline.run_query("What is RAG?")
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self._retriever = None
        self._chunks: list[dict] = []
        self._is_indexed = False

    # ------------------------------------------------------------------
    # Document Ingestion
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        source_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict:
        """
        Ingest documents: load, clean, chunk, and index.

        Args:
            source_path: Path to a document file or directory.
            chunk_size: Override chunk size (default from config).
            chunk_overlap: Override chunk overlap (default from config).

        Returns:
            Dict with ingestion statistics.
        """
        from ingestion.loaders import load_document, load_directory
        from ingestion.cleaning import clean_pages
        from ingestion.chunking import chunk_pages
        from retrieval.search import RetrieverManager

        start = time.time()

        source_path = Path(source_path)
        cs = chunk_size or self.config.retrieval.chunk_size
        co = chunk_overlap or self.config.retrieval.chunk_overlap

        # Load
        if source_path.is_dir():
            pages = load_directory(
                source_path,
                self.config.supported_formats,
                self.config.max_file_size_mb,
            )
        elif source_path.is_file():
            pages = load_document(source_path, self.config.max_file_size_mb)
        else:
            raise FileNotFoundError(f"Path not found: {source_path}")

        if not pages:
            return {"status": "error", "message": "No text extracted from documents."}

        # Clean
        cleaned = clean_pages(pages)

        # Chunk
        self._chunks = chunk_pages(cleaned, chunk_size=cs, chunk_overlap=co)

        if not self._chunks:
            return {"status": "error", "message": "No chunks created after processing."}

        # Index
        self._retriever = RetrieverManager()
        index_stats = self._retriever.index_chunks(
            self._chunks,
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
        )

        self._is_indexed = True
        elapsed = time.time() - start

        result = {
            "status": "success",
            "pages_extracted": len(pages),
            "pages_after_cleaning": len(cleaned),
            "total_chunks": len(self._chunks),
            "chunk_size": cs,
            "chunk_overlap": co,
            "indexing": index_stats,
            "total_time_seconds": round(elapsed, 2),
        }

        logger.info("Ingestion complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Query Pipeline
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        retrieval_method: Optional[str] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        correction_strategy: Optional[str] = None,
        run_correction: bool = True,
    ) -> dict:
        """
        Run the full query pipeline: retrieve → generate → verify → correct → re-verify.

        Args:
            query: The user's question.
            retrieval_method: Override retrieval method.
            top_k: Override top-K.
            alpha: Override hybrid alpha.
            correction_strategy: Override correction strategy.
            run_correction: Whether to run the correction + final verification loop.

        Returns:
            Comprehensive result dict with all pipeline outputs.
        """
        if not self._is_indexed or self._retriever is None:
            return {"error": "No documents indexed. Please ingest documents first."}

        pipeline_start = time.time()
        method = retrieval_method or self.config.retrieval.method
        k = top_k or self.config.retrieval.top_k
        a = alpha if alpha is not None else self.config.retrieval.hybrid_alpha
        strategy = correction_strategy or self.config.correction.strategy

        # 1. Retrieve
        evidence_chunks = self._retriever.retrieve(
            query=query, method=method, top_k=k, alpha=a,
        )

        # 2. Generate answer
        from generation.answer_generator import generate_answer

        gen_result = generate_answer(
            query=query,
            evidence_chunks=evidence_chunks,
            model_name=self.config.llm.model_name,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        # 3. Extract claims
        from verification.claim_extractor import extract_claims

        claim_result = extract_claims(
            answer=gen_result["answer"],
            method="llm",
            model_name=self.config.llm.model_name,
            api_key=self.config.llm.api_key,
        )

        # 4. Verify claims
        from verification.verdict_mapper import verify_all_claims

        verification_result = verify_all_claims(
            claims=claim_result["claims"],
            evidence_chunks=evidence_chunks,
            nli_model_name=self.config.verification.nli_model,
            nli_device=self.config.verification.device,
            entailment_threshold=self.config.verification.entailment_threshold,
            contradiction_threshold=self.config.verification.contradiction_threshold,
        )

        # 5. Correction + Final Verification
        correction_result = None
        final_verification = None

        if run_correction and verification_result["verdicts"]:
            has_problems = any(
                v["verdict"] != Verdict.SUPPORTED
                for v in verification_result["verdicts"]
            )

            if has_problems:
                from correction.correction_pipeline import correct_claims

                llm_config = {
                    "model_name": self.config.llm.model_name,
                    "api_key": self.config.llm.api_key,
                    "temperature": self.config.llm.temperature,
                    "max_tokens": self.config.llm.max_tokens,
                }

                correction_result = correct_claims(
                    original_answer=gen_result["answer"],
                    verdicts=verification_result["verdicts"],
                    evidence_chunks=evidence_chunks,
                    retriever_manager=self._retriever,
                    llm_config=llm_config,
                    strategy=strategy,
                )

                # 6. Final verification of corrected answer
                corrected_answer = correction_result["corrected_answer"]
                final_claims = extract_claims(
                    answer=corrected_answer,
                    method="llm",
                    model_name=self.config.llm.model_name,
                    api_key=self.config.llm.api_key,
                )

                final_verification = verify_all_claims(
                    claims=final_claims["claims"],
                    evidence_chunks=evidence_chunks,
                    nli_model_name=self.config.verification.nli_model,
                    nli_device=self.config.verification.device,
                    entailment_threshold=self.config.verification.entailment_threshold,
                    contradiction_threshold=self.config.verification.contradiction_threshold,
                )
                final_verification["claims"] = final_claims["claims"]

        pipeline_latency = round((time.time() - pipeline_start) * 1000, 1)

        # Compile result
        result = {
            "query": query,
            "retrieval": {
                "method": method,
                "top_k": k,
                "alpha": a if method == "hybrid" else None,
                "chunks_retrieved": len(evidence_chunks),
                "evidence": evidence_chunks,
            },
            "generation": gen_result,
            "claims": claim_result,
            "verification": verification_result,
            "correction": correction_result,
            "final_verification": final_verification,
            "pipeline_config": {
                "retrieval_method": method,
                "top_k": k,
                "chunk_size": self.config.retrieval.chunk_size,
                "chunk_overlap": self.config.retrieval.chunk_overlap,
                "embedding_model": self.config.embedding.model_name,
                "nli_model": self.config.verification.nli_model,
                "llm_model": self.config.llm.model_name,
                "correction_strategy": strategy,
            },
            "pipeline_latency_ms": pipeline_latency,
        }

        logger.info("Pipeline complete in %.1fms for query: %s", pipeline_latency, query[:80])
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._is_indexed

    @property
    def status(self) -> dict:
        return {
            "is_indexed": self._is_indexed,
            "total_chunks": len(self._chunks),
            "retriever_status": self._retriever.status if self._retriever else None,
            "config": {
                "embedding_model": self.config.embedding.model_name,
                "nli_model": self.config.verification.nli_model,
                "llm_model": self.config.llm.model_name,
                "retrieval_method": self.config.retrieval.method,
                "top_k": self.config.retrieval.top_k,
            },
        }

    @property
    def chunks(self) -> list[dict]:
        return list(self._chunks)
