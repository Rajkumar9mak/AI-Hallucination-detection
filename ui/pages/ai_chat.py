"""
AI Chat page for ClaimGuard AI — ChatGPT-inspired document intelligence interface.
Combines question-answering, atomic claim verification, self-healing mitigation,
and inline document upload into a single unified workspace.
"""

import streamlit as st  # type: ignore
from config.settings import get_config, DOCUMENTS_DIR
from ui.conversation_store import ConversationStore
from pipeline import RAGPipeline


def _format_answer_badge(text: str) -> str:
    """Format status tags into clean dark-themed pill badges."""
    badge_supported = (
        '<span class="verdict-badge verdict-supported" style="margin-left: 8px; font-size: 11.5px; padding: 2px 10px; vertical-align: middle; display: inline-flex; align-items: center; gap: 4px;">'
        '● Supported</span>'
    )
    badge_unsupported = (
        '<span class="verdict-badge verdict-insufficient" style="margin-left: 8px; font-size: 11.5px; padding: 2px 10px; vertical-align: middle; display: inline-flex; align-items: center; gap: 4px;">'
        '▲ Unsupported</span>'
    )
    badge_irrelevant = (
        '<span style="background: rgba(161, 161, 170, 0.15); color: #A1A1AA; border: 1px solid #303035; border-radius: 9999px; padding: 2px 10px; font-size: 11.5px; font-weight: 700; margin-left: 8px; vertical-align: middle; display: inline-flex; align-items: center; gap: 4px;">'
        '✕ Irrelevant</span>'
    )
    formatted = text.replace("[Supported]", badge_supported)
    formatted = formatted.replace("[Unsupported]", badge_unsupported)
    formatted = formatted.replace("[Irrelevant]", badge_irrelevant)
    return formatted


def render_ai_chat():
    """Render the main conversational interface."""
    config = get_config()
    pipeline: RAGPipeline | None = st.session_state.get("pipeline")
    store = ConversationStore()

    active_conv_id = st.session_state.get("active_conversation_id")
    if not active_conv_id:
        active_conv_id = store.create_conversation("New Chat")
        st.session_state["active_conversation_id"] = active_conv_id

    conv = store.get_conversation(active_conv_id)
    conv_title = conv["title"] if conv else "New Chat"

    # Header
    col_t, col_status = st.columns([3, 1], vertical_alignment="center")
    with col_t:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 20px; font-weight: 800; color: #FFFFFF;">💬 {conv_title}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_status:
        if pipeline and pipeline.is_ready:
            st.markdown(f"""
            <div style="text-align: right;">
                <span class="status-badge status-online">⚡ {len(pipeline.chunks)} Chunks Indexed</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right;">
                <span style="background: rgba(245, 158, 11, 0.2); color: #FBBF24; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">⚠️ No Index</span>
            </div>
            """, unsafe_allow_html=True)

    # Load messages
    messages = store.get_messages(active_conv_id)

    # Empty State
    if not messages:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px 20px 20px;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 18px; background: linear-gradient(135deg, #E63932, #FF554D); box-shadow: 0 8px 24px rgba(230, 57, 50, 0.4); font-size: 30px; margin-bottom: 16px;">
                ⚡
            </div>
            <div style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 6px;">
                ClaimGuard AI Workspace
            </div>
            <div style="font-size: 14px; color: #A1A1AA; max-width: 520px; margin: 0 auto 24px auto; line-height: 1.5;">
                Ask questions over your indexed documents. Answers are split into atomic statements, verified against evidence using Natural Language Inference, and self-healed if unsupported.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Render Chat History
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        meta = msg.get("metadata")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(f"**{content}**")
        elif role == "assistant":
            with st.chat_message("assistant"):
                # Clean, single human-sounding answer with styled tag badge
                st.markdown(f"""
                <div style="background: #111114; border: 1px solid #303035; border-left: 4px solid #E63932; border-radius: 14px; padding: 18px 22px; font-size: 15.5px; line-height: 1.6; color: #FFFFFF; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
                    {_format_answer_badge(content)}
                </div>
                """, unsafe_allow_html=True)
        elif role == "system":
            st.info(content)

    # -------------------------------------------------------------------------
    # Chat Input Bar (shadcn-prompt-card)
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if "show_chat_attach" not in st.session_state:
        st.session_state["show_chat_attach"] = False
    if "chat_query_buffer" not in st.session_state:
        st.session_state["chat_query_buffer"] = ""

    active_docs = [f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file()] if DOCUMENTS_DIR.exists() else []

    with st.container():
        st.markdown('<div class="shadcn-prompt-card">', unsafe_allow_html=True)

        # 1. Attachment chips with remove button
        if active_docs:
            col_chips, col_del = st.columns([4, 1.2], vertical_alignment="center")
            with col_chips:
                chips = "".join([
                    f'<span class="shadcn-attachment-chip">📄 <span>{doc_name}</span></span>'
                    for doc_name in active_docs[:4]
                ])
                st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 8px;">{chips}</div>', unsafe_allow_html=True)
            with col_del:
                if st.button("✕ Remove PDF", help="Delete uploaded documents and reset index", key="btn_clear_uploaded_docs"):
                    for f in DOCUMENTS_DIR.iterdir():
                        if f.is_file():
                            f.unlink()
                    st.session_state.pipeline = None
                    st.session_state.ingestion_stats = None
                    st.rerun()
            st.markdown('<div style="height: 6px; border-bottom: 1px solid #303035; margin-bottom: 10px;"></div>', unsafe_allow_html=True)

        # 2. File Attachment Drawer
        if st.session_state["show_chat_attach"]:
            st.markdown("<div style='font-size: 12px; font-weight: 600; color: #9CA3AF; margin-bottom: 4px;'>Upload Document(s) to Knowledge Base:</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload and index documents",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True,
                key="chat_uploader",
                label_visibility="collapsed",
            )
            if uploaded:
                col_u1, col_u2 = st.columns([3, 1])
                with col_u1:
                    st.caption(f"{len(uploaded)} file(s) selected for ingestion.")
                with col_u2:
                    if st.button("Index Now", type="primary", use_container_width=True):
                        _ingest_from_chat(uploaded, pipeline, active_conv_id, store)

        # 3. Text Input
        st.markdown('<div class="shadcn-prompt-input">', unsafe_allow_html=True)
        user_text = st.text_area(
            "Query Input",
            value=st.session_state.get("chat_query_buffer", ""),
            placeholder="Ask a question or request fact-checking across documents...",
            height=75,
            label_visibility="collapsed",
            key="chat_input_textarea",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. Toolbar Row
        t_plus, t_model, t_effort, t_method, t_send = st.columns(
            [0.45, 1.35, 1.15, 1.25, 0.55],
            vertical_alignment="center",
        )

        with t_plus:
            st.markdown('<div class="shadcn-plus-btn">', unsafe_allow_html=True)
            if st.button("＋", help="Toggle document upload drawer", key="chat_attach_toggle"):
                st.session_state["show_chat_attach"] = not st.session_state["show_chat_attach"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with t_model:
            selected_model = st.selectbox(
                "Model",
                ["Gemini 2.0 Flash", "Extractive Synthesizer", "Claude Sonnet 3.5", "GPT-4o"],
                index=0,
                label_visibility="collapsed",
                key="chat_model_select",
            )

        with t_effort:
            selected_effort = st.selectbox(
                "Effort",
                ["Effort: Medium", "Effort: High", "Effort: Low"],
                index=0,
                label_visibility="collapsed",
                key="chat_effort_select",
            )

        with t_method:
            selected_method = st.selectbox(
                "Method",
                ["Hybrid (RRF)", "Dense (FAISS)", "BM25 (Lexical)"],
                index=0,
                label_visibility="collapsed",
                key="chat_method_select",
            )

        with t_send:
            st.markdown('<div class="shadcn-send-btn">', unsafe_allow_html=True)
            send_clicked = st.button("↑", type="primary", help="Execute Grounded Query", key="chat_send_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Handle submission
    if send_clicked:
        q = user_text.strip()
        if q:
            st.session_state["chat_query_buffer"] = ""
            _run_chat_query(q, active_conv_id, pipeline, store, config, selected_method, selected_effort)
        else:
            st.warning("Please enter a query to run.")


def _ingest_from_chat(files, pipeline, conv_id, store):
    """Process uploaded files and add a system message to the chat."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner("Parsing and indexing uploaded documents..."):
        for f in files:
            dest = DOCUMENTS_DIR / f.name
            dest.write_bytes(f.getvalue())

        from pipeline import RAGPipeline
        pipe = pipeline or RAGPipeline()
        stats = pipe.ingest_documents(DOCUMENTS_DIR)
        st.session_state.pipeline = pipe
        st.session_state.ingestion_stats = stats
        st.session_state["show_chat_attach"] = False

        store.add_message(
            conv_id,
            "system",
            f"✅ Ingested {len(files)} document(s): {stats.get('total_chunks', 0)} chunks indexed into FAISS & BM25.",
        )
        st.success("Documents indexed successfully!")
        st.rerun()


def _run_chat_query(query: str, conv_id: str, pipeline, store, config, method_label, effort_label):
    """Execute pipeline query and persist messages to SQLite."""
    if not pipeline or not pipeline.is_ready:
        st.error("Cannot query: No documents are indexed. Please attach and index a document first.")
        return

    # Map method
    if "Dense" in method_label:
        retrieval_method = "dense"
        alpha = 1.0
    elif "BM25" in method_label:
        retrieval_method = "bm25"
        alpha = 0.0
    else:
        retrieval_method = "hybrid"
        alpha = config.retrieval.hybrid_alpha

    # Map effort
    if "High" in effort_label:
        top_k = 8
        run_correction = True
        strategy = "hybrid"
    elif "Low" in effort_label:
        top_k = 3
        run_correction = False
        strategy = "removal"
    else:
        top_k = config.retrieval.top_k
        run_correction = True
        strategy = config.correction.strategy

    # Record user message
    store.add_message(conv_id, "user", query)

    # Set title if first message
    conv = store.get_conversation(conv_id)
    if conv and conv["title"] == "New Chat":
        store.auto_title(conv_id)

    with st.spinner("Retrieving evidence & verifying claims..."):
        try:
            # For documents with small chunk collections, ensure high recall
            actual_top_k = top_k
            if pipeline and len(pipeline.chunks) <= 25:
                actual_top_k = min(len(pipeline.chunks), max(top_k, 8))

            result = pipeline.run_query(
                query=query,
                retrieval_method=retrieval_method,
                top_k=actual_top_k,
                alpha=alpha,
                correction_strategy=strategy,
                run_correction=run_correction,
            )
            st.session_state.last_result = result
            if "experiment_history" in st.session_state:
                st.session_state.experiment_history.append(result)

            # Pick best answer (prefer corrected answer if correction occurred and produced a result)
            if result.get("correction") and result["correction"].get("corrected_answer"):
                ans = result["correction"]["corrected_answer"].strip()
            else:
                ans = result.get("generation", {}).get("answer", "").strip()

            # Ensure the response ends with one of the required tags
            has_tag = any(t in ans for t in ["[Supported]", "[Unsupported]", "[Irrelevant]"])
            if not has_tag:
                verif = result.get("final_verification") or result.get("verification", {})
                summary = verif.get("summary", {})
                contra = summary.get("contradicted", 0)
                supp = summary.get("supported", 0)
                in_ev = summary.get("insufficient", 0)

                if "unrelated" in ans.lower() or "irrelevant" in ans.lower():
                    tag = "[Irrelevant]"
                elif "not provide enough information" in ans.lower() or contra > 0 or (supp == 0 and in_ev > 0):
                    tag = "[Unsupported]"
                else:
                    tag = "[Supported]"
                ans = f"{ans.rstrip('.')} {tag}"

            store.add_message(conv_id, "assistant", ans, metadata=result)
            st.rerun()
        except Exception as e:
            st.error(f"Query execution error: {e}")
