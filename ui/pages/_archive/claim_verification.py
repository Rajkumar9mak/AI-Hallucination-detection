"""Claim Audit (Claim Verification) — granular claim-level verification and evidence traceability graph."""

import streamlit as st  # type: ignore
from config.settings import Verdict


def render():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: 800; color: #111827;">Claim Audit & Verification</div>
        <div style="font-size: 14px; color: #6B7280;">Decomposed atomic factual statements verified against retrieved evidence using Natural Language Inference.</div>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    if not result or "error" in result:
        st.info("ℹ️ No active query result. Submit a query in **Query Studio** first to inspect claim verification.")
        return

    # Question summary
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748B;">Query Evaluated</div>
        <div style="font-size: 15px; font-weight: 700; color: #0F172A; margin-top: 2px;">"{result['query']}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Verification Summary Metrics
    summary = result.get("verification", {}).get("summary", {})
    total_claims = summary.get("total_claims", 0)
    supported = summary.get("supported", 0)
    contradicted = summary.get("contradicted", 0)
    insufficient = summary.get("insufficient", 0)
    grounding_rate = summary.get("grounding_rate", 1.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Claims</div>
            <div class="metric-value">{total_claims}</div>
            <div class="metric-sub">Atomic statements</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Supported</div>
            <div class="metric-value" style="color: #059669;">{supported}</div>
            <div class="metric-sub">{supported/total_claims*100 if total_claims else 0:.0f}% Entailed by evidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Contradicted</div>
            <div class="metric-value" style="color: #DC2626;">{contradicted}</div>
            <div class="metric-sub">Factual conflicts</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Insufficient Evidence</div>
            <div class="metric-value" style="color: #D97706;">{insufficient}</div>
            <div class="metric-sub">Unverified (not false)</div>
        </div>
        """, unsafe_allow_html=True)

    # Grounding Health Bar
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.progress(grounding_rate, text=f"Overall Answer Grounding Health: {grounding_rate*100:.1f}%")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Filterable Claim Cards
    st.subheader("📋 Granular Claim Audit Trail")

    filter_option = st.radio(
        "Filter Claims",
        ["All Claims", "Supported", "Contradicted", "Insufficient Evidence"],
        horizontal=True,
    )

    verdicts = result.get("verification", {}).get("verdicts", [])

    filtered_verdicts = verdicts
    if filter_option == "Supported":
        filtered_verdicts = [v for v in verdicts if v.get("verdict") == Verdict.SUPPORTED]
    elif filter_option == "Contradicted":
        filtered_verdicts = [v for v in verdicts if v.get("verdict") == Verdict.CONTRADICTED]
    elif filter_option == "Insufficient Evidence":
        filtered_verdicts = [v for v in verdicts if v.get("verdict") == Verdict.INSUFFICIENT]

    if not filtered_verdicts:
        st.info(f"No claims matching filter '{filter_option}'.")
    else:
        for idx, item in enumerate(filtered_verdicts):
            _render_startup_claim_card(item, idx + 1)


def _render_startup_claim_card(item: dict, display_num: int):
    verdict = item.get("verdict", Verdict.INSUFFICIENT)
    claim_text = item.get("claim_text", "")
    claim_id = item.get("claim_id", f"claim_{display_num:03d}")
    evidence = item.get("evidence", [])
    confidence = item.get("confidence")

    if verdict == Verdict.SUPPORTED:
        badge_html = '<span class="verdict-badge verdict-supported">✅ Supported</span>'
        border_color = "#10B981"
    elif verdict == Verdict.CONTRADICTED:
        badge_html = '<span class="verdict-badge verdict-contradicted">❌ Contradicted</span>'
        border_color = "#EF4444"
    else:
        badge_html = '<span class="verdict-badge verdict-insufficient">⚠️ Insufficient Evidence</span>'
        border_color = "#F59E0B"

    conf_str = f"Confidence: {confidence * 100:.1f}%" if confidence is not None else "Confidence: Evaluated"

    st.markdown(f"""
    <div class="claim-card-container" style="border-left: 4px solid {border_color};">
        <div class="claim-header">
            <div>
                <span class="claim-id-pill">{claim_id.upper()}</span>
                <span style="font-size: 12px; color: #6B7280; margin-left: 8px;">{conf_str}</span>
            </div>
            <div>
                {badge_html}
            </div>
        </div>
        <div class="claim-text">"{claim_text}"</div>
    </div>
    """, unsafe_allow_html=True)

    if evidence:
        with st.expander(f"🔍 Trace Evidence Matches for {claim_id.upper()} ({len(evidence)} passage(s))", expanded=False):
            for i, ev in enumerate(evidence):
                src = ev.get("source_name", "Document")
                page = ev.get("page_number", "1")
                nli_label = ev.get("nli_label", "neutral")
                scores = ev.get("nli_scores", {})
                entail = scores.get("entailment", 0.0) * 100 if scores else 0
                contrad = scores.get("contradiction", 0.0) * 100 if scores else 0

                st.markdown(f"""
                <div class="evidence-box">
                    <span class="evidence-citation">Source: {src} (Page {page}) • NLI Decision: {nli_label.upper()}</span>
                    <div style="font-size: 13px; color: #1E293B;">{ev.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                if scores:
                    st.caption(f"Entailment: {entail:.1f}% | Contradiction: {contrad:.1f}%")
