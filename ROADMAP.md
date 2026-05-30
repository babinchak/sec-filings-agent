# Roadmap

Status legend: ⬜ not started · 🟡 in progress · ✅ done

See [DECISIONS.md](./DECISIONS.md) for the reasoning behind the choices baked into this roadmap, and [PROJECT.md](./PROJECT.md) for the project's overall design philosophy.

---

## Phase 0 — Foundations ✅

**Goal:** Repo is set up, one filing has been inspected, core data shapes are defined.

- `uv init`, `pyproject.toml` with pinned dependencies
- `src/` package skeleton: `corpus/`, `retrieval/`, `tools/`, `agent/`, `evaluation/`
- Core pydantic models: `Chunk`, `Filing`, `EvalRecord`, `AgentTrace`
- Ingest one 10-K with edgartools; eyeball the structure
- `.env.example` for `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, `FMP_API_KEY`, `COHERE_API_KEY`

**Done when:** Importing the corpus module and ingesting one filing returns a parsed `Filing` object whose sections look right.

---

## Phase 1 — Parallel MVP on a single filing ✅

> Done 2026-05-30. All 5 smoke questions pass end-to-end on MSFT 2023 with correct routing; numeric answers exact (revenue $211.915B, R&D $27.195B, ratio 12.83%). 21 unit tests green. Agent recovers from an unindexed-year request by retrying with an available year.

**Goal:** End-to-end question → answer works on one filing. Both tracks must hit this milestone before scaling.

### Retrieval track
- `src/corpus/chunker.py` — section-aware chunker (Item-level + sub-section, length-bounded)
- `src/retrieval/embedding.py` — Voyage embeddings, batched
- `src/retrieval/vector_store.py` — ChromaDB wrapper
- `src/retrieval/lexical.py` — bm25s index
- `src/retrieval/hybrid.py` — Reciprocal Rank Fusion over dense + lexical
- Smoke test: query → top-10 chunks, eyeball relevance

### Agent track
- `src/tools/fmp.py` — `get_financial_fact(ticker, year, concept) -> float`
- `src/tools/retrieve.py` — wraps hybrid retrieval as an Anthropic tool
- `src/agent/loop.py` — Anthropic SDK tool-use loop with router system prompt, supporting sub-query decomposition and multi-step tool chaining

**Done when:** 5 hand-written questions on one filing return reasonable answers end-to-end; agent traces show correct tool routing.

---

## Phase 2 — Scale corpus ⬜

**Goal:** Full corpus indexed; pipeline survives the scale jump.

- Inspect FinanceBench question coverage; select ~18 companies × ~5 years to maximize overlap (target ≥50 usable FB questions)
- Ingest, chunk, index all filings
- Latency budget: hybrid retrieval < 1s per query at this scale

**Done when:** Full corpus indexed; the 5 smoke-test questions still work; retrieval latency is acceptable.

---

## Phase 3 — Eval infrastructure ⬜

**Goal:** Eval runs are reproducible, results are persisted as JSON, notebooks load from JSON.

- `src/evaluation/records.py` — pydantic schemas for retrieval and end-to-end eval records
- `src/evaluation/loaders.py` — FinanceBench loader (filters to corpus overlap)
- `src/evaluation/judge.py` — LLM-as-judge for answer grading; calibrate against a small gold set
- `scripts/run_eval.py` — accepts a config, writes timestamped JSON to `results/`
- `notebooks/02_retrieval_eval.ipynb` and `03_agent_eval.ipynb` — load JSON, plot

**Done when:** A FinanceBench run end-to-end produces a results JSON that the notebook visualizes.

---

## Phase 4 — Hand-written failure-mode eval ⬜

**Goal:** ~60 hand-labeled questions exposing the four target failure modes.

- ~15 questions per mode: segment/terminology mismatch · multi-company comparison · numeric-vs-narrative routing · calculation chains
- Each question labeled with: target failure mode, gold answer, gold-relevant chunks (where applicable), source filing(s)
- Schema mirrors `EvalRecord` so the eval runner is reusable
- This is the human-bottleneck phase — don't rush it; quality of labels caps the ceiling of the project

**Done when:** `data/eval_sets/failure_modes.jsonl` contains ~60 labeled records; a baseline run produces interpretable per-mode metrics.

---

## Phase 5 — Ablations ⬜

**Goal:** Each component's contribution to each failure mode is quantified.

Ablations to run (each one = one eval run = one results JSON):
- **Chunking:** section-aware vs naive fixed-length
- **Retrieval:** BM25-only / dense-only / hybrid / hybrid + reranker
- **Router:** on / off (force-RAG baseline)
- **Decomposition:** on / off
- **Tool chaining:** on / off

`notebooks/04_ablations.ipynb` — comparison table, per-failure-mode breakdown.

**Done when:** Ablations table shows component contributions per failure mode; results JSONs are committed (or a pointer to where they live).

---

## Phase 6 — Failure analysis ⬜

**Goal:** Failures are categorized, taxonomy is visualized, deep dives illustrate each category.

- Hand-categorize misses into: retrieval miss / extraction error / reasoning error / hallucination
- `notebooks/05_failure_analysis.ipynb` — stacked-bar by failure mode × failure category, 3-5 deep-dive examples with full agent traces

**Done when:** Notebook shows the taxonomy chart and at least 3 detailed example walkthroughs.

---

## Phase 7 — README + headline results ⬜

**Goal:** Repo is interview-ready.

- README: architecture diagram, FinanceBench score, hand-eval scores per failure mode, ablation table, "if I had more time" section
- Architecture diagram (mermaid — no external tool dependency)
- Run-from-scratch instructions

**Done when:** A fresh clone + a single command (`make eval` or equivalent) reproduces the headline numbers.
