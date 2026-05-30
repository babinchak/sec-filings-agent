# sec-filings-agent

An agentic RAG system over SEC 10-K filings with rigorous retrieval and end-to-end evaluation.

**See also:** [ROADMAP.md](./ROADMAP.md) for phased build plan and current status · [DECISIONS.md](./DECISIONS.md) for the reasoning behind load-bearing choices.

## What this project is

A portfolio project for AI/ML engineering roles in NYC (fintech, AI-native research tools, hedge funds). The goal is to demonstrate competence in:

1. **Agentic system design** — multi-step agents with tool use, routing between retrieval and structured-data backends
2. **Retrieval engineering** — chunking strategies, hybrid retrieval (dense + BM25), reranking
3. **Rigorous evaluation** — retrieval metrics, end-to-end accuracy, ablations, failure analysis

The portfolio centerpiece is the **evaluation work**, not the application itself. The system exists primarily as a substrate to evaluate well.

## What this project is NOT

- Not a startup. Not trying to make money. Not a product.
- Not a UI-forward project. No Electron app, no Next.js, no web frontend (at least not initially).
- Not built with LangChain or LlamaIndex as the backbone. Those frameworks abstract away exactly what this project is trying to demonstrate understanding of.
- Not a tutorial follow-along. Every design choice should be defensible in an interview.

## Domain context

The corpus is SEC 10-K filings — annual reports that US public companies are legally required to file. They follow a standardized Item structure (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, etc.) but have wildly varying segment definitions, line item names, and narrative content across companies and years.

This asymmetry — rigid skeleton, varying flesh — is what makes the retrieval problem interesting. A 10-K from Comcast organizes its business into 5 segments; Microsoft uses 3 different ones; a bank would have totally different financial line items. Naive retrieval struggles with this; thoughtful retrieval handles it.

**Initial corpus scope:** ~15-20 companies across 2-3 industries (e.g., big tech, media/entertainment, telecom), ~5 years of 10-Ks each. Roughly 75-100 filings total. Bounded enough to evaluate carefully, large enough that retrieval is non-trivial.

## Architecture (initial sketch)

**Hybrid agent** that routes between two backends:

1. **Semantic retrieval** over chunked filing text — for narrative questions ("how does management discuss AI risk?")
2. **Structured-data tool** backed by FinancialModelingPrep — for numeric questions ("what was R&D in 2023?")

The agent operates at **Level 3**: it routes between tools, decomposes multi-entity questions into per-company / per-year sub-queries, and chains tool calls where step N's output informs step N+1 (e.g. fetch revenue → fetch R&D → compute ratio).

**Component breakdown:**
- Corpus ingestion (edgartools)
- Section-aware chunking (custom — write this from scratch, it's a portfolio talking point)
- Embeddings (Voyage AI, possibly voyage-finance-2)
- Vector store (ChromaDB, local)
- Lexical retrieval (bm25s) + Reciprocal Rank Fusion for hybrid
- Reranker (Cohere Rerank, optional in v1)
- Agent loop (hand-written against the Anthropic SDK's native tool use)
- Structured-data tool (wraps FinancialModelingPrep)
- Eval runner (custom)

## Tech stack

- **Python 3.12+**, managed with **uv**
- **pydantic** v2 for structured data everywhere
- **anthropic** SDK as primary LLM (Claude Sonnet 4.6 for the agent, possibly Haiku 4.5 for cheap subtasks)
- **openai** SDK as a comparison point for ablations
- **edgartools** for filing ingestion
- **chromadb** for vector store (local, file-backed)
- **bm25s** for lexical retrieval
- **voyageai** for embeddings
- **cohere** for reranking
- **pytest** for tests on `src/` modules
- **jupyter** + **plotly** for analysis notebooks
- **pandas** for eval result analysis

**Explicitly avoiding:** LangChain, LlamaIndex, LangGraph (too much abstraction for a project whose point is demonstrating understanding of the fundamentals).

## Repo structure

```
sec-filings-agent/
├── README.md                    # architecture + headline results + how to run
├── PROJECT.md                   # this file
├── pyproject.toml
├── src/sec_filings/            # single installable package (editable)
│   ├── config.py               # env-driven settings (pydantic-settings)
│   ├── corpus/                 # filing ingestion, chunking
│   ├── retrieval/              # embedding, indexing, search, reranking
│   ├── tools/                  # structured-data API wrappers
│   ├── agent/                  # planner, router, executor, agent loop
│   └── evaluation/             # metrics, eval runners
├── notebooks/
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_retrieval_eval.ipynb
│   ├── 03_agent_eval.ipynb
│   ├── 04_ablations.ipynb
│   └── 05_failure_analysis.ipynb
├── data/
│   ├── eval_sets/              # labeled questions (checked in — the artifact)
│   ├── filings/                # cached corpus (gitignored)
│   ├── chroma/                 # vector store (gitignored)
│   └── bm25/                   # lexical index (gitignored)
├── scripts/
│   ├── inspect_one_filing.py   # Phase 0 smoke test
│   ├── ingest.py               # one-time corpus build
│   └── run_eval.py             # reproducible eval runs
├── results/
│   └── *.json                   # saved eval outputs; notebooks load from here
└── tests/
```

## Coding principles

- **Python modules contain the system. Notebooks contain analysis.** Notebooks import from `src/`; they never define core logic.
- **Save eval outputs to JSON, then have notebooks load and visualize.** Re-running a notebook should be fast (just re-plot); the expensive eval runs are in scripts.
- **Type hints everywhere.** mypy- or pyright-clean.
- **pydantic models for any structured data** crossing module boundaries — chunks, eval records, agent state, tool inputs/outputs.
- **No silent fallbacks.** If an API call fails, raise. If a chunk is malformed, raise. Eval integrity depends on not papering over errors.
- **Reproducibility:** every eval run writes a JSON with model versions, prompt versions, retrieval config, and timestamps. The notebook reading it should be deterministic.

## Evaluation methodology (high level)

Two eval sets, serving complementary purposes:

1. **FinanceBench** (Patronus AI) — external credibility anchor. ~150 expert-written QA pairs over real 10-Ks. Corpus is selected to maximize FB overlap (target ≥50 usable questions). Metric: answer accuracy judged against gold answers (LLM-as-judge, calibrated).
2. **Hand-written failure-mode set** — the eval-design centerpiece. ~60 questions, each labeled with the failure mode it probes. The four target modes:
   - **Segment / terminology mismatch** — question uses one term, filing uses another. Pairs with retrieval ablation (BM25 / dense / hybrid / +reranker).
   - **Multi-company comparison** — requires sub-query decomposition + synthesis. Pairs with decomposition on/off ablation.
   - **Numeric vs narrative routing** — tests whether the router picks the right backend. Pairs with router on/off ablation.
   - **Calculation chains** — multiple facts + arithmetic. Pairs with tool-chaining on/off ablation.

Plus retrieval metrics (recall@k, MRR, nDCG) computed wherever gold-relevant chunks are labeled.

**Ablations** swap components one at a time (chunking strategy, retrieval mode, router on/off, decomposition on/off, tool-chaining on/off) so each component's contribution to each failure mode is quantified.

A **failure analysis notebook** hand-categorizes misses into a taxonomy (retrieval miss / extraction error / reasoning error / hallucination) with stacked-bar visualization and example deep-dives.

## Build plan

See [ROADMAP.md](./ROADMAP.md) for the phased build plan (Phase 0 foundations through Phase 7 README/headline results) and current status. The roadmap is the source of truth for sequencing; this document focuses on the *what* and *why*, not the *when*.

## Notes for Claude Code

- This project's value is in design decisions, not lines of code. When in doubt, write less and explain more.
- Don't pull in a framework to solve something a small amount of custom code would handle better. If you reach for LangChain/LlamaIndex/LangGraph, stop and ask.
- The agent loop, chunker, eval runner, and retrieval orchestration should all be hand-written. These are the components that need to be defensible in an interview.
- Eval design (what questions, what metrics, what ablations) is a human decision — don't auto-generate eval questions without flagging it for review.
- Prefer minimal dependencies. Each new package should justify itself.
