# Agentic TDD Pipeline — 9.2 Upgrade

## Why the previous version was ~6.5/10

The original pipeline had the right high-level idea but several quality gaps:

1. Scanner mostly produced aggregate statistics rather than a rich deterministic evidence inventory.
2. Chunking operated on one combined source string, which could mix unrelated files.
3. Retrieval used a small fixed set of queries and then hard-limited results.
4. Architect output was JSON-structured, but claims were not individually tied to stable evidence IDs.
5. There was no independent quality gate between Architect and Manager.
6. ZIP extraction in the web layer used `extractall()` without a traversal check.
7. Web uploads shared global extraction paths, creating concurrency/race risks.
8. The web progress bar was client-side simulated rather than based on actual stage events.
9. The final document did not expose a machine-readable generation manifest.
10. Testing was not sufficient to demonstrate the pipeline's core invariants.

## What this version changes

### Evidence integrity
- Deterministic scanner extracts Python AST symbols and generic-language symbols.
- Every chunk has a stable `EV-*` evidence ID.
- Chunks preserve file path and line ranges.
- Important project files are always eligible for retrieval.
- Architect is instructed to cite evidence IDs for concrete claims.
- Critic rejects unknown evidence IDs.

### Retrieval
- Default lightweight lexical retrieval works on the free deployment without
  sentence-transformers/Chroma.
- Optional `RAG_MODE=hybrid` activates sentence-transformers if installed.
- Retrieval returns scores and metadata.

### Agentic quality control
- Scanner → Architect → Critic → Manager.
- Critic provides deterministic checks and optional LLM review.
- The generated TDD exposes PASS/REVIEW status and a quality score.
- A JSON manifest is generated for reproducibility.

### Security and reliability
- ZIP traversal is checked before extraction.
- Per-request temporary directories avoid shared upload collisions.
- Upload size limits are enforced.
- History JSON is written atomically.
- Errors are logged server-side without exposing a traceback to the browser.

## Important honesty rule

Do not claim "9.5/10 production-ready" merely because the code looks stronger.
The project earns a high portfolio score when the following are actually demonstrated:

- pipeline runs on representative projects
- tests pass
- generated TDD contains valid evidence IDs
- unsupported claims are rejected or marked as undetermined
- Render deployment succeeds
- at least one real generated TDD is reviewed for correctness
- performance numbers are measured rather than invented

The target assessment for this upgrade is **9.2/10 portfolio engineering quality**,
subject to those validation steps.
