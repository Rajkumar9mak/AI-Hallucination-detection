"""Knowledge Hub (Document Management) — upload, clean, chunk, and inspect documents."""

import streamlit as st  # type: ignore
from pathlib import Path
from config.settings import get_config, DOCUMENTS_DIR


def render():
    config = get_config()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: 800; color: #111827;">Knowledge Hub</div>
        <div style="font-size: 14px; color: #6B7280;">Upload documents, configure sliding-window chunking, and inspect indexed metadata chunks.</div>
    </div>
    """, unsafe_allow_html=True)

    # Chunking Configuration Bar
    with st.container():
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px 20px; margin-bottom: 20px;">
            <div style="font-weight: 700; font-size: 15px; color: #1F2937; margin-bottom: 12px;">⚙️ Chunking Engine Parameters</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1.5, 1.5, 2])
        with c1:
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=100, max_value=2000,
                value=config.retrieval.chunk_size,
                step=50,
                help="Sliding window chunk capacity in characters.",
            )
        with c2:
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0, max_value=300,
                value=config.retrieval.chunk_overlap,
                step=10,
                help="Overlapping characters between consecutive chunks to preserve boundary context.",
            )
        with c3:
            effective_step = chunk_size - chunk_overlap
            st.metric(
                "Step Size / Window",
                f"{effective_step} chars",
                delta=f"{chunk_overlap} char context buffer",
                delta_color="normal",
            )

    # Document Upload Section
    st.markdown("### 📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or Markdown files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help=f"Supported formats: .pdf, .txt, .docx, .md (Max size: {config.max_file_size_mb}MB)",
    )

    if uploaded_files:
        st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, f in enumerate(uploaded_files):
            col_target = cols[idx % len(cols)]
            size_kb = len(f.getvalue()) / 1024
            with col_target:
                st.markdown(f"""
                <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 14px; font-size: 13px; margin-bottom: 8px;">
                    📄 <b>{f.name}</b><br>
                    <span style="color: #6B7280; font-size: 11.5px;">{size_kb:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)

        if st.button("🚀 Process & Build Index", type="primary"):
            _process_uploaded_documents(uploaded_files, chunk_size, chunk_overlap)

    st.markdown("---")

    # Current Document Repository
    st.markdown("### 📚 Indexed Knowledge Repository")
    pipeline = st.session_state.get("pipeline")
    doc_files = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file()] if DOCUMENTS_DIR.exists() else []

    if doc_files and pipeline and pipeline.is_ready:
        st.success(f"✅ Active Index: {len(doc_files)} document(s) ingested into {len(pipeline.chunks)} vector & lexical chunks.")

        # Display Document Cards
        for doc in doc_files:
            size_kb = doc.stat().st_size / 1024
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 12px 18px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 24px;">📄</div>
                    <div>
                        <div style="font-weight: 700; font-size: 14.5px; color: #111827;">{doc.name}</div>
                        <div style="font-size: 12px; color: #6B7280;">Size: {size_kb:.1f} KB • Format: {doc.suffix.upper()}</div>
                    </div>
                </div>
                <div>
                    <span class="status-badge status-online">Indexed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chunk Explorer
        with st.expander("🔍 Interactive Chunk Explorer", expanded=True):
            search_filter = st.text_input("Filter chunks by keyword", placeholder="e.g. exam, timing, controller...")
            matching_chunks = pipeline.chunks
            if search_filter:
                matching_chunks = [c for c in matching_chunks if search_filter.lower() in c.get("text", "").lower()]

            st.caption(f"Showing {len(matching_chunks)} of {len(pipeline.chunks)} chunks")

            for i, chunk in enumerate(matching_chunks[:6]):
                st.markdown(f"""
                <div class="evidence-box" style="margin-bottom: 12px;">
                    <span class="evidence-citation">Chunk #{i+1} | Document: {chunk.get('source_name', '?')} | Page: {chunk.get('page_number', '1')}</span>
                    <div>{chunk.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("ℹ️ No documents are currently indexed. Upload a file above to build your knowledge base.")


def _process_uploaded_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Save and ingest uploaded files."""
    from pipeline import RAGPipeline

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous documents
    for existing in DOCUMENTS_DIR.iterdir():
        if existing.is_file():
            existing.unlink()

    prog = st.progress(0, text="Saving uploaded files...")

    for i, f in enumerate(uploaded_files):
        dest = DOCUMENTS_DIR / f.name
        dest.write_bytes(f.getvalue())
        prog.progress((i + 1) / (len(uploaded_files) + 2), text=f"Saved {f.name}")

    prog.progress(0.7, text="Extracting text, cleaning, and building FAISS + BM25 indexes...")

    try:
        pipeline = RAGPipeline()
        stats = pipeline.ingest_documents(
            source_path=DOCUMENTS_DIR,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        prog.progress(1.0, text="✅ Ingestion complete!")

        st.session_state.pipeline = pipeline
        st.session_state.ingestion_stats = stats
        st.success(f"Successfully indexed {stats.get('total_chunks', 0)} chunks across {stats.get('pages_extracted', 0)} page(s)!")
        st.rerun()
    except Exception as e:
        prog.empty()
        st.error(f"Ingestion failed: {e}")
