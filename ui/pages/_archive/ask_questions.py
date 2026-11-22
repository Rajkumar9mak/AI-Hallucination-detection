"""Query Studio (Ask Questions) — conversational search, retrieval config, and instant verification preview."""

import streamlit as st  # type: ignore
from config.settings import get_config, DOCUMENTS_DIR
from ui.components import button_outline


def render():
    pipeline = st.session_state.get("pipeline")
    config = get_config()

    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800; color: #FFFFFF;">Query Studio</div>
        <div style="font-size: 14px; color: #9CA3AF;">Ask questions across your document corpus with multi-modal retrieval and real-time claim verification.</div>
    </div>
    """, unsafe_allow_html=True)

    if not pipeline or not pipeline.is_ready:
        st.warning("⚠️ No documents are indexed. Please go to **Knowledge Hub** to upload and process documents first.")
        return

    if "query_input_text" not in st.session_state:
        st.session_state["query_input_text"] = ""
    if "show_attach_panel" not in st.session_state:
        st.session_state["show_attach_panel"] = False

    # Sample prompt shortcut pills
    st.markdown("<div style='font-size: 13px; font-weight: 600; color: #9CA3AF; margin-bottom: 8px;'>Sample Prompt Shortcuts:</div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        if st.button("🚂 Train Number & Name", use_container_width=True):
            st.session_state["query_input_text"] = "train number"
            st.rerun()
    with sc2:
        if st.button("🕒 Exam Reporting Timing", use_container_width=True):
            st.session_state["query_input_text"] = "what is the reporting timing for the exam?"
            st.rerun()
    with sc3:
        if st.button("👤 Passenger Name & Status", use_container_width=True):
            st.session_state["query_input_text"] = "passenger name"
            st.rerun()
    with sc4:
        if st.button("⚡ RAG Architecture Overview", use_container_width=True):
            st.session_state["query_input_text"] = "What is retrieval-augmented generation and who introduced it?"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # shadcn/ui PromptInputDemo Search Bar Component
    # -------------------------------------------------------------------------
    active_docs = [f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file()] if DOCUMENTS_DIR.exists() else []

    with st.container():
        st.markdown('<div class="shadcn-prompt-card">', unsafe_allow_html=True)

        # 1. Attachment Bar (Block-start: File chips with icons & names)
        if active_docs:
            chips = "".join([
                f'<span class="shadcn-attachment-chip">📄 <span>{doc_name}</span></span>'
                for doc_name in active_docs[:4]
            ])
            attachments_html = f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);">{chips}</div>'
            st.markdown(attachments_html, unsafe_allow_html=True)

        # File Attachment Drawer (Toggled by the Plus button)
        if st.session_state["show_attach_panel"]:
            st.file_uploader(
                "Attach supplemental document for this query session",
                type=["pdf", "txt", "docx", "md"],
                key="query_supplemental_upload",
                label_visibility="collapsed",
            )

        # 2. InputGroupTextarea (Border-free prompt textarea)
        st.markdown('<div class="shadcn-prompt-input">', unsafe_allow_html=True)
        query = st.text_area(
            "Prompt Query",
            value=st.session_state.get("query_input_text", ""),
            height=85,
            placeholder="Ask the agent to verify facts or query your documents... (e.g. train number, reporting timing)",
            label_visibility="collapsed",
            key="active_query_textarea",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. InputGroupAddon Toolbar (Block-end: Attach, Model, Effort, Method, Up-Arrow Send)
        tb_col_plus, tb_col_model, tb_col_effort, tb_col_method, tb_col_send = st.columns(
            [0.45, 1.35, 1.15, 1.25, 0.55],
            vertical_alignment="center",
        )

        with tb_col_plus:
            st.markdown('<div class="shadcn-plus-btn">', unsafe_allow_html=True)
            if st.button("＋", help="Attach or inspect document files", key="attach_toggle_btn"):
                st.session_state["show_attach_panel"] = not st.session_state["show_attach_panel"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with tb_col_model:
            selected_model = st.selectbox(
                "Model",
                ["Gemini 2.0 Flash", "Extractive Synthesizer", "Claude Sonnet 3.5", "GPT-4o"],
                label_visibility="collapsed",
                help="Generator model powering evidence synthesis",
                key="prompt_model_select",
            )

        with tb_col_effort:
            selected_effort = st.selectbox(
                "Effort",
                ["Effort: Medium", "Effort: High", "Effort: Low"],
                label_visibility="collapsed",
                help="Verification and retrieval depth: Low (fast), Medium (balanced), High (exhaustive)",
                key="prompt_effort_select",
            )

        with tb_col_method:
            selected_method_label = st.selectbox(
                "Method",
                ["Hybrid (RRF)", "Dense (FAISS)", "BM25 (Lexical)"],
                label_visibility="collapsed",
                help="Evidence retrieval index fusion strategy",
                key="prompt_method_select",
            )

        with tb_col_send:
            st.markdown('<div class="shadcn-send-btn">', unsafe_allow_html=True)
            send_clicked = st.button("↑", type="primary", help="Execute Grounded Query (Enter)", key="prompt_submit_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Map Effort to Hyperparameters
    if "High" in selected_effort:
        top_k = 8
        run_correction = True
        strategy = "hybrid"
    elif "Low" in selected_effort:
        top_k = 3
        run_correction = False
        strategy = "removal"
    else:  # Medium
        top_k = config.retrieval.top_k
        run_correction = True
        strategy = config.correction.strategy

    # Map Retrieval Method
    if "Dense" in selected_method_label:
        retrieval_method = "dense"
        alpha = 1.0
    elif "BM25" in selected_method_label:
        retrieval_method = "bm25"
        alpha = 0.0
    else:
        retrieval_method = "hybrid"
        alpha = config.retrieval.hybrid_alpha

    # Execute on Send Click
    if send_clicked:
        active_query = query.strip()
        if active_query:
            _execute_query(pipeline, active_query, retrieval_method, top_k, alpha, strategy, run_correction)
        else:
            st.warning("Please type a query or click a prompt shortcut above.")

    # -------------------------------------------------------------------------
    # Display Verification Results
    # -------------------------------------------------------------------------
    result = st.session_state.get("last_result")
    if result and "error" not in result:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        latency = result.get("pipeline_latency_ms", 0)
        claims_count = result.get("claims", {}).get("total_claims", 0)
        summary = result.get("verification", {}).get("summary", {})
        grounding_pct = summary.get("grounding_rate", 1.0) * 100

        # Quick Results Metric Header
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(18, 18, 24, 0.90); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; backdrop-filter: blur(12px);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span class="status-badge status-online">Verified Result</span>
                <span style="font-size: 13px; color: #9CA3AF;">Latency: <b style="color: #FFFFFF;">{latency:.0f}ms</b></span>
                <span style="font-size: 13px; color: #9CA3AF;">Claims: <b style="color: #FFFFFF;">{claims_count}</b></span>
            </div>
            <div>
                <span class="verdict-badge verdict-supported">Grounding: {grounding_pct:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Grounded Answer Card
        st.markdown("### 💬 Grounded Answer")
        answer_text = result["generation"]["answer"]
        st.markdown(f"""
        <div style="background: rgba(18, 18, 24, 0.92); border: 1px solid rgba(255, 255, 255, 0.14); border-left: 4px solid #E50000; border-radius: 12px; padding: 18px 22px; font-size: 15.5px; line-height: 1.6; color: #F8FAFC; box-shadow: 0 4px 20px rgba(0,0,0,0.4); margin-bottom: 20px; backdrop-filter: blur(14px);">
            {answer_text}
        </div>
        """, unsafe_allow_html=True)

        # Retrieved Evidence Accordion
        evidence = result["retrieval"]["evidence"]
        with st.expander(f"📚 Inspect Retrieved Evidence ({len(evidence)} chunks from {retrieval_method.upper()})", expanded=False):
            for i, chunk in enumerate(evidence):
                score_val = chunk.get("score", 0)
                st.markdown(f"""
                <div class="evidence-box">
                    <span class="evidence-citation">#{i+1} • {chunk.get('source_name', 'Document')} (Page {chunk.get('page_number', '1')}) • Relevance Score: {score_val:.4f}</span>
                    <div>{chunk.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)


def _execute_query(pipeline, query, method, top_k, alpha, strategy, run_correction):
    """Run the pipeline with status toast."""
    with st.spinner("Analyzing document evidence and decomposing claims..."):
        try:
            result = pipeline.run_query(
                query=query,
                retrieval_method=method,
                top_k=top_k,
                alpha=alpha,
                correction_strategy=strategy,
                run_correction=run_correction,
            )
            st.session_state.last_result = result
            st.session_state.experiment_history.append(result)
            st.success(f"Complete! {result['claims']['total_claims']} factual claims verified in {result['pipeline_latency_ms']:.0f}ms.")
            st.rerun()
        except Exception as e:
            st.error(f"Execution failed: {e}")
