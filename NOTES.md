# Notes

Working scratch pad. Edit freely. Claude reads this at the start of a session and
writes to it as we work. Keep **Now** short — if it has 5+ items, something's wrong.
ROADMAP.md is the full plan; this is just what's in front of us.

## Now — Phase 1 DONE ✅ (2026-05-30)
End-to-end Q&A works on MSFT 2023. 5/5 smoke questions pass with correct routing;
numbers exact (rev $211.915B, R&D $27.195B, ratio 12.83%). 21 tests green.
Full retrieval stack (Voyage→Chroma+bm25→RRF) + FMP tool + hand-written agent loop built.
**Langfuse tracing wired in** (2026-05-30): every run_agent call is a waterfall on
us.cloud.langfuse.com — agent → generation/turn (usage+USD cost auto-priced) →
retriever/tool spans. No-op when LANGFUSE_* unset. Notebook §1b deep-links the run.

Pick the next thing (don't need to decide both):
- ⬜ Streamlit explorer — corpus/chunk/retrieval/agent-trace viewer (the "click around" tool you wanted)
- ⬜ Phase 2 — scale corpus (pick ~18 companies × ~5y to maximize FinanceBench overlap)

## Things built this session (reference)
- src/sec_filings/retrieval/: embedding.py · vector_store.py · lexical.py · hybrid.py
- src/sec_filings/tools/: fmp.py (+FMP_TOOL_SCHEMA) · retrieve.py (+RETRIEVE_TOOL_SCHEMA)
- src/sec_filings/agent/loop.py (run_agent → AgentTrace)
- src/sec_filings/observability.py — Langfuse seam (no-op if unconfigured); loop.py traces through it
- scripts/build_index.py (writes data/chroma_manifest.json) · scripts/ask.py (smoke harness)
- notebooks/01_explore.ipynb — interactive: ask agent / inspect trace / raw retrieval / FMP / manifest
- Agent recovers from an unindexed-year request by retrying with an available year (loop.py catches LookupError → is_error → retry).

## Open questions
- Embedder: `voyage-4-large` (primary) vs `voyage-finance-2` (ablation). Decide via a small
  benchmark later; wire model name as a single config value so the swap is one line.
- Item 8 (Financial Statements) chunked into 95 sentence-based chunks — but it's mostly
  tables, which have no real sentences. Tabular chunking may need special handling later.
  Not a Phase 1 blocker.
- MSFT/GOOGL/AAPL have different fiscal-year-ends (Jun/Dec/Sep). Let "fiscal 2023" ambiguity
  surface naturally in eval rather than special-casing now.

## Decisions captured this session
- Token counter: tiktoken `cl100k_base`, injectable (tests use a whitespace counter).
- Chunking: within-section only, sentence-granular greedy packing, ~500 tok / ~50 overlap,
  exact char offsets, deterministic `chunk_id = accession::item::char_start`.
- Observability: build scripts will write a `data/chroma_manifest.json` so notebooks/Streamlit
  visualize corpus coverage by reading JSON, not by querying Chroma.
- UI: lightweight **Streamlit** for interactive exploration (NOT Electron/React), grown one
  page per component. Notebooks remain for reproducible analysis. (Supersedes PROJECT.md
  line 20's "no UI" — that line was loose AI-generated text.)

## Random
- First R&D corpus target: MSFT, GOOGL, AAPL × FY2021–2023 = 9 filings.
