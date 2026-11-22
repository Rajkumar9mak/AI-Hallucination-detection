"""Mitigation Lab (Correction Results) — side-by-side diff comparison and post-correction verification."""

import streamlit as st  # type: ignore
from config.settings import Verdict


def render():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: 800; color: #111827;">Mitigation Lab & Before/After Audit</div>
        <div style="font-size: 14px; color: #6B7280;">Automated hallucination reduction through evidence-constrained regeneration, targeted re-retrieval, or abstention.</div>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    if not result or "error" in result:
        st.info("ℹ️ No active query result. Execute a query in **Query Studio** first.")
        return

    correction = result.get("correction")
    final_verification = result.get("final_verification")

    # If all claims were already supported
    if correction is None:
        st.success("✅ **Zero Hallucinations Detected!** All extracted claims were verified as Supported by retrieved evidence. No correction required.")
        st.markdown("### Grounded Response:")
        st.info(result["generation"]["answer"])
        return

    # Before vs After Side-by-Side Comparison
    st.subheader("⚖️ Side-by-Side Answer Comparison")

    col_init, col_corr = st.columns(2)

    with col_init:
        st.markdown("""
        <div class="compare-card compare-initial">
            <div style="font-weight: 700; font-size: 15px; color: #334155; margin-bottom: 10px;">📝 Initial Generated Response</div>
        </div>
        """, unsafe_allow_html=True)
        st.write(correction.get("original_answer", result["generation"]["answer"]))

    with col_corr:
        st.markdown("""
        <div class="compare-card compare-corrected">
            <div style="font-weight: 700; font-size: 15px; color: #166534; margin-bottom: 10px;">🛡️ Corrected & Verified Response</div>
        </div>
        """, unsafe_allow_html=True)
        st.write(correction.get("corrected_answer", ""))

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Grounding Improvement Metrics
    initial_summary = result.get("verification", {}).get("summary", {})
    final_summary = final_verification.get("summary", {}) if final_verification else initial_summary

    init_gr = initial_summary.get("grounding_rate", 1.0) * 100
    final_gr = final_summary.get("grounding_rate", 1.0) * 100
    delta_gr = final_gr - init_gr

    st.subheader("📈 Mitigation Delta Metrics")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("Initial Grounding", f"{init_gr:.1f}%")
    with d2:
        st.metric("Final Grounding", f"{final_gr:.1f}%", delta=f"{delta_gr:+.1f}%")
    with d3:
        contrad_reduction = initial_summary.get("contradicted", 0) - final_summary.get("contradicted", 0)
        st.metric("Contradiction Reduction", f"{contrad_reduction} claim(s)")
    with d4:
        st.metric("Strategy Applied", correction.get("strategy", "hybrid").upper())

    # Detailed Mitigation Action Log
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.subheader("🛠️ Mitigation Actions Applied")

    actions = correction.get("correction_actions", [])
    if actions:
        for act in actions:
            action_type = act.get("action", "modified")
            verdict = act.get("verdict", "problematic")
            claim = act.get("claim_text", "")
            reason = act.get("reason", "")

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="background: #EEF2FF; color: #4338CA; font-weight: 700; font-size: 12px; padding: 3px 8px; border-radius: 4px;">{action_type.upper()}</span>
                        <span style="font-size: 13px; color: #64748B; margin-left: 8px;">Triggered by: <b>{verdict}</b></span>
                    </div>
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #1E293B;">"{claim}"</div>
                <div style="font-size: 12.5px; color: #64748B; margin-top: 4px;">Reason: {reason}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No claim-level modifications were required.")
