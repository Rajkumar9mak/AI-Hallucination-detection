"""
Settings panel and dialog for ClaimGuard AI.
Allows configuring retrieval, chunking, LLM, verification, and correction hyperparameters.
"""

import streamlit as st  # type: ignore
from config.settings import get_config, DOCUMENTS_DIR


def render_settings_dialog():
    """Render the configuration settings panel."""
    config = get_config()

    st.markdown("### ⚙️ System Configuration")
    st.caption("Adjust hyperparameters for retrieval, indexing, generation, and verification.")

    tab_retrieval, tab_chunking, tab_model, tab_system = st.tabs([
        "🔍 Retrieval",
        "📄 Chunking & Index",
        "🧠 Models & Verification",
        "ℹ️ System Info",
    ])

    with tab_retrieval:
        st.markdown("**Retrieval Strategy & Depth**")
        method = st.selectbox(
            "Default Retrieval Method",
            ["hybrid", "dense", "bm25"],
            index=["hybrid", "dense", "bm25"].index(st.session_state.get("cfg_retrieval_method", config.retrieval.method)),
            key="cfg_retrieval_method_input",
        )
        st.session_state["cfg_retrieval_method"] = method

        top_k = st.slider(
            "Default Top-K Evidence Depth",
            min_value=1,
            max_value=20,
            value=st.session_state.get("cfg_top_k", config.retrieval.top_k),
            key="cfg_top_k_input",
        )
        st.session_state["cfg_top_k"] = top_k

        alpha = st.slider(
            "Hybrid Fusion Alpha (Dense Weight vs BM25)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("cfg_alpha", config.retrieval.hybrid_alpha),
            step=0.05,
            key="cfg_alpha_input",
        )
        st.session_state["cfg_alpha"] = alpha

    with tab_chunking:
        st.markdown("**Sliding-Window Chunking Parameters**")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=st.session_state.get("cfg_chunk_size", config.retrieval.chunk_size),
            step=50,
            key="cfg_chunk_size_input",
        )
        st.session_state["cfg_chunk_size"] = chunk_size

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=300,
            value=st.session_state.get("cfg_chunk_overlap", config.retrieval.chunk_overlap),
            step=10,
            key="cfg_chunk_overlap_input",
        )
        st.session_state["cfg_chunk_overlap"] = chunk_overlap

        st.info(f"Effective window step: {chunk_size - chunk_overlap} characters per chunk.")

        if st.button("🔄 Re-index Ingested Documents", use_container_width=True):
            pipeline = st.session_state.get("pipeline")
            if pipeline:
                with st.spinner("Rebuilding FAISS and BM25 indexes with updated chunking..."):
                    try:
                        stats = pipeline.ingest_documents(
                            DOCUMENTS_DIR,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.session_state.ingestion_stats = stats
                        st.success(f"Re-indexing complete! {stats.get('total_chunks', 0)} chunks created.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Re-indexing failed: {e}")
            else:
                st.warning("No pipeline loaded yet to re-index.")

    with tab_model:
        st.markdown("**Generation & Verification Models**")
        st.caption(f"**Embedding Model:** `{config.embedding.model_name}` ({config.embedding.device})")
        st.caption(f"**NLI Verifier:** `{config.verification.nli_model}`")

        llm_model = st.text_input(
            "LLM Generator Model",
            value=st.session_state.get("cfg_llm_model", config.llm.model_name),
            key="cfg_llm_model_input",
        )
        st.session_state["cfg_llm_model"] = llm_model

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("cfg_temperature", config.llm.temperature),
            step=0.05,
            key="cfg_temperature_input",
        )
        st.session_state["cfg_temperature"] = temperature

        st.markdown("**Mitigation Strategy**")
        strategy = st.selectbox(
            "Default Correction Strategy",
            ["hybrid", "removal", "abstention", "regeneration", "re_retrieval"],
            index=["hybrid", "removal", "abstention", "regeneration", "re_retrieval"].index(
                st.session_state.get("cfg_strategy", config.correction.strategy)
            ),
            key="cfg_strategy_input",
        )
        st.session_state["cfg_strategy"] = strategy

    with tab_system:
        st.markdown("**ClaimGuard AI Architecture**")
        pipeline = st.session_state.get("pipeline")
        chunks_count = len(pipeline.chunks) if pipeline and pipeline.is_ready else 0
        docs_count = len([f for f in DOCUMENTS_DIR.iterdir() if f.is_file()]) if DOCUMENTS_DIR.exists() else 0

        st.caption(f"**Indexed Documents:** {docs_count}")
        st.caption(f"**Indexed Chunks:** {chunks_count}")
        st.caption(f"**Vector Store:** FAISS (Exact Inner Product)")
        st.caption(f"**Lexical Engine:** BM25Okapi with Query Expansion")
        st.caption(f"**Fusion Method:** Reciprocal Rank Fusion (k=60)")
