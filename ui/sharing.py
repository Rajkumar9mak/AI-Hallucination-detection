"""
Share and export utilities for ClaimGuard AI conversations and verification results.
"""

import io
import csv
import streamlit as st  # type: ignore


def export_conversation_markdown(conv_title: str, messages: list[dict]) -> str:
    """Generate Markdown export of a conversation with verification audit data."""
    lines = [
        f"# ClaimGuard AI — Conversation Export",
        f"**Topic:** {conv_title}",
        f"**Date:** {messages[0]['created_at'] if messages else 'N/A'}",
        "",
        "---",
        "",
    ]

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"### {role}:")
        lines.append(content)
        lines.append("")

        meta = msg.get("metadata")
        if meta and role == "Assistant":
            verif = meta.get("verification", {}).get("summary", {})
            if verif:
                lines.append(f"> **Verification:** {verif.get('total_claims', 0)} claims | "
                             f"Supported: {verif.get('supported', 0)} | "
                             f"Contradicted: {verif.get('contradicted', 0)} | "
                             f"Insufficient: {verif.get('insufficient', 0)} | "
                             f"Grounding: {verif.get('grounding_rate', 1.0)*100:.0f}%")
                lines.append("")

            # Itemize claims
            claims = meta.get("verification", {}).get("verdicts", [])
            if claims:
                lines.append("**Claim Audit Trail:**")
                for c in claims:
                    verdict = c.get("verdict", "")
                    claim_text = c.get("claim_text", "")
                    lines.append(f"- `[{verdict.upper()}]` {claim_text}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_claims_csv(messages: list[dict]) -> str:
    """Generate CSV of all verified claims across the conversation."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Message ID", "Claim ID", "Claim Text", "Verdict", "Confidence", "Evidence Count", "Primary Source"])

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        meta = msg.get("metadata")
        if not meta:
            continue
        verdicts = meta.get("verification", {}).get("verdicts", [])
        for c in verdicts:
            evidence = c.get("evidence", [])
            primary_src = evidence[0].get("source_name", "N/A") if evidence else "None"
            writer.writerow([
                msg.get("id", ""),
                c.get("claim_id", ""),
                c.get("claim_text", ""),
                c.get("verdict", ""),
                f"{c.get('confidence', 0):.3f}" if c.get("confidence") is not None else "N/A",
                len(evidence),
                primary_src,
            ])

    return output.getvalue()


def render_share_panel(conv_title: str, messages: list[dict]):
    """Render share / export action buttons."""
    st.markdown("### 📤 Share & Export")
    if not messages:
        st.info("No messages in the active conversation to export.")
        return

    md_content = export_conversation_markdown(conv_title, messages)
    csv_content = export_claims_csv(messages)

    st.download_button(
        "📄 Export Conversation (Markdown)",
        data=md_content,
        file_name=f"{conv_title.replace(' ', '_')[:30]}_claimguard.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        "📊 Export Claims Audit (CSV)",
        data=csv_content,
        file_name=f"{conv_title.replace(' ', '_')[:30]}_claims.csv",
        mime="text/csv",
        use_container_width=True,
    )
