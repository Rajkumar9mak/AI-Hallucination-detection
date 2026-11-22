"""
ClaimGuard AI — Enterprise Claim-Level Hallucination Detection & Mitigation Platform.
ChatGPT-inspired two-page architecture: Mission Control & AI Chat, with SQLite conversation history.
"""

from pathlib import Path
import streamlit as st  # type: ignore

from ui.styles import apply_styles
from config.settings import get_config, DOCUMENTS_DIR
from ui.conversation_store import ConversationStore

# Page configuration
st.set_page_config(
    page_title="ClaimGuard AI | RAG Hallucination Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom dark theme & styles
apply_styles()

# ---------------------------------------------------------------------------
# State Initialization & Auto-Indexing
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "ingestion_stats" not in st.session_state:
    st.session_state.ingestion_stats = None
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "show_settings_modal" not in st.session_state:
    st.session_state.show_settings_modal = False
if "show_share_modal" not in st.session_state:
    st.session_state.show_share_modal = False

store = ConversationStore()

# Auto-sync pipeline with active documents on disk
doc_file_names = sorted([f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".docx", ".md")]) if DOCUMENTS_DIR.exists() else []
curr_pipeline = st.session_state.get("pipeline")
indexed_sources = sorted(list({c.get("source_name") for c in curr_pipeline.chunks})) if curr_pipeline and hasattr(curr_pipeline, "chunks") and curr_pipeline.chunks else []

if curr_pipeline is None or doc_file_names != indexed_sources:
    if doc_file_names:
        try:
            from pipeline import RAGPipeline
            pipe = RAGPipeline()
            stats = pipe.ingest_documents(DOCUMENTS_DIR)
            st.session_state.pipeline = pipe
            st.session_state.ingestion_stats = stats
        except Exception:
            pass
    else:
        st.session_state.pipeline = None
        st.session_state.ingestion_stats = None

# ---------------------------------------------------------------------------
# ChatGPT-Style Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Branding Header
    st.markdown("""
    <div style="padding: 6px 0 14px 0;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #E63932, #FF554D); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 19px; font-weight: 800; box-shadow: 0 4px 12px rgba(230, 57, 50, 0.45);">
                ⚡
            </div>
            <div>
                <div style="font-size: 18px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">ClaimGuard <span style="color: #FF554D;">AI</span></div>
                <div style="font-size: 10.5px; font-weight: 600; color: #A1A1AA; text-transform: uppercase; letter-spacing: 0.05em;">Hallucination Detection RAG</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Main Action Buttons
    col_new, col_mc = st.columns([1.1, 1.2])
    with col_new:
        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            new_id = store.create_conversation("New Chat")
            st.session_state.active_conversation_id = new_id
            st.session_state.current_page = "chat"
            st.rerun()

    with col_mc:
        is_mc = st.session_state.current_page == "mission_control"
        mc_label = "📊 Dashboard" if not is_mc else "💬 Go to Chat"
        if st.button(mc_label, use_container_width=True):
            st.session_state.current_page = "chat" if is_mc else "mission_control"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 3. Conversation History
    st.markdown("<div style='font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #A1A1AA; margin-bottom: 6px;'>Conversation History</div>", unsafe_allow_html=True)

    groups = store.get_conversations_grouped()
    active_id = st.session_state.get("active_conversation_id")

    for group_name, convs in groups.items():
        if convs:
            st.caption(f"**{group_name}**")
            for c in convs:
                cid = c["id"]
                title = c["title"]
                is_active = (cid == active_id and st.session_state.current_page == "chat")

                col_btn, col_act = st.columns([4, 1])
                with col_btn:
                    prefix = "▸ " if is_active else "  "
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(f"{prefix}{title}", key=f"conv_select_{cid}", use_container_width=True, type=btn_type):
                        st.session_state.active_conversation_id = cid
                        st.session_state.current_page = "chat"
                        st.rerun()

                with col_act:
                    with st.popover("⋯", help="Chat options"):
                        new_name = st.text_input("Rename", value=title, key=f"rename_input_{cid}")
                        if st.button("Save", key=f"save_rename_{cid}"):
                            if new_name.strip():
                                store.rename_conversation(cid, new_name.strip())
                                st.rerun()
                        if st.button("Delete", type="primary", key=f"delete_btn_{cid}"):
                            store.delete_conversation(cid)
                            if st.session_state.active_conversation_id == cid:
                                st.session_state.active_conversation_id = None
                            st.rerun()

    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # 4. Bottom Controls: Settings & Share
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings_modal = not st.session_state.show_settings_modal
            st.rerun()
    with b_col2:
        if st.button("📤 Export", use_container_width=True):
            st.session_state.show_share_modal = not st.session_state.show_share_modal
            st.rerun()

    # System Status Indicator
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None and pipeline.is_ready:
        chunks_count = len(pipeline.chunks)
        st.markdown(f"""
        <div class="status-badge status-online" style="margin-top: 10px; width: 100%; justify-content: center;">
            <span class="pulse-dot"></span>
            <span>READY ({chunks_count} CHUNKS)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: rgba(245, 158, 11, 0.15); color: #FBBF24; padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 600; margin-top: 10px;">
            <span>⚠️ INDEX EMPTY</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Acropolis Institute of Technology & Research")

# ---------------------------------------------------------------------------
# Modals / Panels
# ---------------------------------------------------------------------------
if st.session_state.get("show_settings_modal"):
    from ui.settings_modal import render_settings_dialog
    with st.expander("⚙️ System Configuration Settings", expanded=True):
        render_settings_dialog()
        if st.button("Close Settings", key="close_settings_btn"):
            st.session_state.show_settings_modal = False
            st.rerun()

if st.session_state.get("show_share_modal"):
    from ui.sharing import render_share_panel
    with st.expander("📤 Export & Share Conversation", expanded=True):
        active_id = st.session_state.get("active_conversation_id")
        conv = store.get_conversation(active_id) if active_id else None
        title = conv["title"] if conv else "ClaimGuard_Chat"
        msgs = store.get_messages(active_id) if active_id else []
        render_share_panel(title, msgs)
        if st.button("Close Export", key="close_share_btn"):
            st.session_state.show_share_modal = False
            st.rerun()

# ---------------------------------------------------------------------------
# Page Routing (Two-Page Platform)
# ---------------------------------------------------------------------------
if st.session_state.current_page == "mission_control":
    from ui.pages.dashboard import render
    render()
else:
    from ui.pages.ai_chat import render_ai_chat
    render_ai_chat()
