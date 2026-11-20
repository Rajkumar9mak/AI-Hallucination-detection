# Coding Rules and Engineering Standards

## 1. General Principles
- Keep code modular: ingestion, retrieval, generation, verification, correction, evaluation, and UI must have clear responsibilities.
- Prefer readable, explicit code over clever abstractions.
- Avoid duplicating logic; create shared functions for repeated operations.
- Do not silently swallow exceptions.
- Do not hard-code API keys, credentials, machine-specific paths, or private data.
- Keep model names and tunable parameters in configuration.

## 2. Python Style
- Use Python 3.10+ compatible syntax unless the project environment specifies otherwise.
- Follow PEP 8 naming and formatting conventions.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants.
- Add type hints to public functions and important data structures.
- Add docstrings to modules, classes, and non-obvious public functions.
- Keep functions focused and reasonably small.
- Use `pathlib` for filesystem paths.
- Use structured logging rather than scattered debug `print()` statements.

## 3. Project Structure
- Keep UI code in the Streamlit entry point and presentation helpers.
- Keep retrieval backends behind a common interface so BM25, dense, and hybrid can be compared consistently.
- Keep model loading separate from per-query inference.
- Keep evaluation code separate from application response logic.
- Do not create a second implementation of a component before checking whether the repository already contains one.

## 4. Retrieval Rules
- Preserve source identifiers and page/chunk metadata throughout retrieval.
- Make Top-K configurable.
- Keep chunk size and overlap configurable for experiments.
- Use the same corpus and query set when comparing retrieval methods.
- Record the hybrid fusion strategy and its parameters.
- Do not present similarity scores as calibrated probabilities.
- Ensure dense vector IDs map reliably to the original chunks.
- Handle empty results explicitly.

## 5. Grounding and Verification Rules
- Treat claims as atomic factual statements wherever practical.
- Use only supplied/retrieved evidence to justify a claim verdict.
- Every verdict must retain the evidence considered.
- Use only these operational labels:
  - `Supported`
  - `Contradicted`
  - `Insufficient Evidence`
- `Insufficient Evidence` does not mean the claim is false.
- Do not classify a claim as Contradicted unless evidence conflicts with it.
- Do not mark a claim Supported merely because it sounds plausible or matches model knowledge.
- Do not assume the NLI model or LLM verifier is always correct.
- Keep claim extraction rules stable across experimental runs.

## 6. Correction Rules
- Prefer targeted correction over rewriting the entire answer without cause.
- Re-retrieval must be related to the claim being corrected.
- Regeneration must be constrained to the available evidence.
- Remove claims that cannot be supported when appropriate.
- Abstain when adequate evidence remains unavailable.
- Never add invented citations, page numbers, document names, or quotations.
- Run final verification after every correction.
- Preserve the original answer and verification results for before/after comparison.

## 7. Data, Logging, and Privacy
- Do not log API keys, access tokens, or secrets.
- Avoid logging full uploaded documents unless explicitly required for a controlled experiment.
- Store only the metadata needed to reproduce experiments.
- Validate file type and size before document processing.
- Treat uploaded documents as untrusted input.
- Do not execute content extracted from documents.
- Keep generated indexes and experiment outputs in designated data directories.
- Add sensitive local data and secrets to `.gitignore`.

## 8. Error Handling
- Raise or return meaningful errors with context.
- Do not convert an exception into an apparently successful verification result.
- If verification fails, mark the claim as unverified/error rather than Supported.
- Handle missing evidence, malformed model output, rate limits, and timeouts.
- Validate structured LLM output before using it.

## 9. Testing
- Add unit tests for chunking, metadata preservation, retrieval ranking, verdict mapping, and correction behavior.
- Test empty corpus and empty retrieval cases.
- Test all three claim verdicts.
- Test that Insufficient Evidence is not treated as Contradicted.
- Test that final verification runs after correction.
- Use fixed fixtures for reproducible tests.
- Add integration tests for the full query pipeline where practical.

## 10. Git and Collaboration
- Use descriptive branch names, such as `feature/hybrid-retrieval` or `fix/claim-mapping`.
- Use concise commit messages with an action verb.
- Do not commit virtual environments, cache files, API keys, private documents, or generated large indexes.
- Keep `requirements.txt` or the chosen dependency lock file updated when dependencies change.
- Update documentation when a module, configuration, or workflow changes.

## 11. Definition of Done
A task is complete only when:
- Implementation is integrated with the existing project.
- Relevant tests pass or limitations are documented.
- Errors and edge cases are handled.
- Evidence/source metadata is preserved where applicable.
- Documentation/configuration is updated.
- The task status in `tasks.md` is updated.
