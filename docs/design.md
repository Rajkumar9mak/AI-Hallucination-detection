# UI/UX Direction
## Claim-Level Hallucination Detection and Reduction in RAG Systems

## 1. Design Objective
Create a clear academic demonstration interface that helps users understand not only the generated answer, but also which individual claims are supported, contradicted, or unresolved by the retrieved documents.

The UI should prioritize evidence traceability and before/after comparison over decorative elements.

## 2. Design Principles
- **Clarity:** Make the workflow and system status easy to understand.
- **Evidence first:** Show the evidence behind each verdict.
- **Honest communication:** Distinguish unsupported claims from claims proven false.
- **Comparison:** Make retrieval strategies and correction outcomes comparable.
- **Consistency:** Use the same labels and terminology throughout the application.
- **Responsive layout:** Keep the interface usable on typical laptop screens.

## 3. Suggested Information Architecture

### Page 1: Overview
- Project name and short explanation.
- Pipeline status: documents loaded, chunks indexed, models ready.
- Short explanation of the three verdicts.
- Quick link or navigation to the query workspace and experiment results.

### Page 2: Document Management
- Upload/select documents.
- Show file name, processing status, chunk count, and ingestion errors.
- Show chunking configuration (size and overlap).
- Provide a clear indication when the corpus/index is ready.

### Page 3: Ask a Question
- Query input.
- Retrieval method selector: BM25, Dense, Hybrid.
- Top-K control.
- Optional advanced experiment controls for chunking and model selection.
- Run button and visible processing state.

### Page 4: Verification Results
Display results in this order:
1. User question.
2. Initial generated answer.
3. Claim-level verification summary.
4. Individual claim cards.
5. Correction action and corrected answer.
6. Final verification summary.

### Page 5: Experiments
- Compare BM25, dense, and hybrid retrieval.
- Show retrieval and grounding metrics.
- Show claim-verification precision, recall, and F1 where labelled data exists.
- Show latency and inference cost.
- Display experimental configuration beside each result.

## 4. Claim Card Design
Each claim card should contain:
- Claim number/ID.
- Exact claim text.
- Verdict badge.
- Evidence excerpt(s).
- Source/document name and page number, when available.
- Optional verifier/model details.
- Correction action or reason no correction was possible.

Suggested visual semantics:
- **Supported:** green status treatment.
- **Contradicted:** red status treatment.
- **Insufficient Evidence:** amber/yellow status treatment.

Do not rely on color alone; always include the written verdict label and, where possible, an icon or accessible text.

## 5. Answer Comparison
Use separate, clearly titled panels:
- **Initial Answer**
- **Corrected Answer**
- **Final Verification**

Show what changed and whether the corrected claims gained evidence. Avoid implying improvement merely because the wording changed.

## 6. Experiment Dashboard
Recommended visualizations:
- Bar chart: retrieval metric comparison by retrieval method.
- Bar chart: claim-verification precision/recall/F1.
- Before/after chart: supported, contradicted, and insufficient-evidence claim counts.
- Table: latency and inference cost by configuration.
- Configuration table: Top-K, chunk size, overlap, embedding model, verifier, correction strategy.

Only plot metrics that have actually been measured. Mark unavailable values as N/A rather than filling them with estimates.

## 7. Streamlit Layout Guidance
- Use a persistent sidebar for navigation and experiment controls.
- Use page headers that describe the user's current task.
- Use expandable sections for long evidence excerpts and advanced settings.
- Keep the main answer and claim verdict summary visible without excessive scrolling.
- Show progress indicators during ingestion, retrieval, generation, and verification.
- Use concise error messages with a recovery action.
- Provide a clear empty state when no documents have been indexed.

## 8. Tone and Microcopy
Prefer precise language:
- “Evidence found” rather than “Fact confirmed” unless ground truth supports that statement.
- “Insufficient evidence in retrieved documents” rather than “False”.
- “Correction attempted” rather than “Hallucination fixed” until final verification confirms an improvement.
- “Final verification incomplete” when the verifier fails.

## 9. Accessibility and Usability
- Maintain readable contrast.
- Use text labels in addition to color.
- Keep controls consistently named.
- Provide helpful labels and short explanations for Top-K and retrieval method.
- Avoid overly dense tables; allow horizontal scrolling or concise summaries where needed.
- Ensure errors are visible and not communicated by color alone.

## 10. Visual Direction
Use a restrained, professional academic/research dashboard style:
- Neutral background and clear content panels.
- One primary accent color for actions and navigation.
- Consistent status colors for the three verdicts.
- Simple typography with a clear heading/body hierarchy.
- Minimal decorative graphics; prioritize charts, evidence excerpts, and structured claim cards.
