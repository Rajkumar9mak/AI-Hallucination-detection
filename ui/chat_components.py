"""
Chat message sub-components for ClaimGuard AI.
Renders claim cards, evidence accordions, verification badges, and correction deltas
matching the dark OLED theme (#080808 + #111114 + #303035 + #E63932).
"""

import streamlit as st  # type: ignore
from config.settings import Verdict


def render_verification_summary(summary: dict, latency_ms: float = 0):
    """Render a compact glassmorphic verification summary bar."""
    total_claims = summary.get("total_claims", 0)
    supported = summary.get("supported", 0)
    contradicted = summary.get("contradicted", 0)
    insufficient = summary.get("insufficient", 0)
    grounding_rate = summary.get("grounding_rate", 1.0) * 100

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; background: #111114; border: 1px solid #303035; border-radius: 14px; padding: 12px 18px; margin: 12px 0 16px 0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <span class="status-badge status-online">🛡️ Verified Result</span>
            <span style="font-size: 13px; color: #A1A1AA;">Latency: <b style="color: #FFFFFF;">{latency_ms:.0f}ms</b></span>
            <span style="font-size: 13px; color: #A1A1AA;">Claims: <b style="color: #FFFFFF;">{total_claims}</b></span>
            <span style="font-size: 12.5px; color: #55C991; font-weight: 600;">✅ {supported}</span>
            <span style="font-size: 12.5px; color: #F16B6B; font-weight: 600;">❌ {contradicted}</span>
            <span style="font-size: 12.5px; color: #F2B94B; font-weight: 600;">⚠️ {insufficient}</span>
        </div>
        <div>
            <span class="verdict-badge verdict-supported" style="font-size: 13px;">Grounding: {grounding_rate:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_claim_card(item: dict, display_num: int):
    """Render a single atomic claim card with expandable evidence."""
    verdict = item.get("verdict", Verdict.INSUFFICIENT)
    claim_text = item.get("claim_text", "")
    claim_id = item.get("claim_id", f"claim_{display_num:03d}")
    evidence = item.get("evidence", [])
    confidence = item.get("confidence")

    if verdict == Verdict.SUPPORTED:
        badge_html = '<span class="verdict-badge verdict-supported">✅ Supported</span>'
        border_color = "#55C991"
    elif verdict == Verdict.CONTRADICTED:
        badge_html = '<span class="verdict-badge verdict-contradicted">❌ Contradicted</span>'
        border_color = "#E63932"
    else:
        badge_html = '<span class="verdict-badge verdict-insufficient">⚠️ Insufficient Evidence</span>'
        border_color = "#F2B94B"

    conf_str = f"Confidence: {confidence * 100:.1f}%" if confidence is not None else "Confidence: Evaluated"

    st.markdown(f"""
    <div class="claim-card-container" style="border-left: 4px solid {border_color}; margin-bottom: 10px;">
        <div class="claim-header">
            <div>
                <span class="claim-id-pill">{claim_id.upper()}</span>
                <span style="font-size: 12px; color: #A1A1AA; margin-left: 8px;">{conf_str}</span>
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
            for ev in evidence:
                src = ev.get("source_name", "Document")
                page = ev.get("page_number", "1")
                nli_label = ev.get("nli_label", "neutral")
                scores = ev.get("nli_scores", {})
                entail = scores.get("entailment", 0.0) * 100 if scores else 0
                contrad = scores.get("contradiction", 0.0) * 100 if scores else 0

                st.markdown(f"""
                <div class="evidence-box">
                    <span class="evidence-citation">Source: {src} (Page {page}) • NLI Decision: {nli_label.upper()}</span>
                    <div style="font-size: 13px; color: #FFFFFF; line-height: 1.5;">{ev.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                if scores:
                    st.caption(f"Entailment: {entail:.1f}% | Contradiction: {contrad:.1f}%")


def render_correction_delta(
    correction: dict,
    final_verification: dict | None = None,
    initial_summary: dict | None = None,
):
    """Render correction results and delta metrics if mitigation was applied."""
    if not correction:
        return

    init_gr = (initial_summary.get("grounding_rate", 1.0) if initial_summary else 1.0) * 100
    final_summary = final_verification.get("summary", {}) if final_verification else {}
    final_gr = final_summary.get("grounding_rate", init_gr / 100) * 100 if final_summary else init_gr
    delta_gr = final_gr - init_gr

    st.markdown(f"""
    <div style="background: #111114; border: 1px solid #303035; border-radius: 14px; padding: 16px 20px; margin: 14px 0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 16px;">🔧</span>
                <span style="font-weight: 700; font-size: 14px; color: #FFFFFF;">Mitigation Applied: <span style="color: #FF554D;">{correction.get('strategy', 'hybrid').upper()}</span></span>
            </div>
            <span style="font-size: 12.5px; font-weight: 700; color: #55C991;">Grounding: {init_gr:.0f}% → {final_gr:.0f}% ({delta_gr:+.0f}%)</span>
        </div>
        <div style="font-size: 13px; color: #A1A1AA; margin-bottom: 8px;">
            Corrected Answer:
        </div>
        <div style="background: rgba(85, 201, 145, 0.08); border: 1px solid rgba(85, 201, 145, 0.35); border-radius: 10px; padding: 12px 16px; font-size: 14px; color: #FFFFFF; line-height: 1.5;">
            {correction.get('corrected_answer', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    actions = correction.get("correction_actions", [])
    if actions:
        with st.expander(f"🛠️ Inspect Mitigation Actions ({len(actions)})", expanded=False):
            for act in actions:
                action_type = act.get("action", "modified")
                verdict = act.get("verdict", "problematic")
                claim = act.get("claim_text", "")
                reason = act.get("reason", "")
                st.markdown(f"""
                <div style="background: #171215; border: 1px solid #303035; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                    <span style="background: rgba(230, 57, 50, 0.15); color: #FFAAA8; border: 1px solid rgba(230, 57, 50, 0.35); font-weight: 700; font-size: 11px; padding: 2px 6px; border-radius: 4px;">{action_type.upper()}</span>
                    <span style="font-size: 12px; color: #A1A1AA; margin-left: 6px;">Triggered by: <b style="color: #FFFFFF;">{verdict}</b></span>
                    <div style="font-size: 13px; font-weight: 600; color: #FFFFFF; margin-top: 6px;">"{claim}"</div>
                    <div style="font-size: 12px; color: #A1A1AA; margin-top: 2px;">Reason: {reason}</div>
                </div>
                """, unsafe_allow_html=True)
