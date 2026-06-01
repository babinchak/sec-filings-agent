# Notes

Working scratch pad. Edit freely. Claude reads this at the start of a session and
writes to it as we work. Keep **Now** short — if it has 5+ items, something's wrong.
ROADMAP.md is the full plan; this is just what's in front of us.

## Now — AGENT 87% (26/30), STABLE @ temp 0 (2026-05-31)
QUICK-WIN ITERATION (on top of the hardened judge): pinned agent temperature=0 (config
`agent_temperature`) + 2 answer rules in ROUTER_SYSTEM ("commit to 'none' on absence Qs"; "report a
disclosed figure, don't collapse to $0"). **77% -> 87% (23/30 -> 26/30), and now STABLE run-to-run.**
Fixed 3: AMCR 00799 (quick-ratio formula aligned), AXP 00476 (commits to "none"), BA 00678 (settled
at temp 0). Still failing (4): AMD 00222 (compute — re-summed instead of using the printed "Total
current liabilities"; needs a reconcile self-check), AXP 01028 (RETRIEVAL miss — geo region table
unretrievable; needs table-aware chunking), BA 00585 (tax-rate sign), PEP 01328 (FLAWED QUESTION — it
literally says "state 0 if not on the income statement"; agent cites the $411M but obeys the question
-> drop/rewrite it in our own eval set, not an agent bug).
Snapshots: eval_run_v1.json (90%, lenient judge) / eval_run_v2.json (77%, hardened judge, temp default)
/ eval_run.json (87%, current). NOTE: judge_audit.json's 30/30 panel was on the v2 answers; the 87% is
graded by the validated judge but not yet re-audited by the panel on these new answers.

The judge-hardening run that set the 77% baseline (still-valid context):
77% was corroborated by an INDEPENDENT 3-grader blind panel (`scripts/judge_audit.workflow.js`):
judge-vs-panel agreement **30/30**, panel 29/30 unanimous. Artifacts: `results/judge_audit.json`.

What hardening did (`src/sec_filings/evaluation/judge.py`): verdict forced through a `record_verdict`
TOOL CALL (can't truncate/mis-parse — kills the v1 `max_tokens=300` crash), judge now sees the GOLD
EVIDENCE passage (verifies figures, not just the terse gold string), stricter rubric naming the
failure modes (hedge / "$0" / missing figure / wrong sign) with few-shot calibration, and
`JudgeError` instead of a silent wrong-default. Pinned by 12 parser unit tests + `scripts/validate_judge.py`
(12/12 labeled behavioural cases, incl. the exact $0-vs-$411M pattern).

**Why 90% → 77% (the portfolio story, sharpened):** TWO causes the lenient judge was masking.
(1) JUDGE LENIENCY — AXP 00476 was a hedged "see Exhibit 4.2" non-answer v1 passed; PEP 01328 ($0
vs gold $411M) only "failed" in v1 via the JSON crash. (2) AGENT NONDETERMINISM — AMD 00222 and
BA 00678 passed in v1's run but the FRESH agent sampled worse answers this run (wrong quick-ratio
denominator; opposite gross-margin conclusion). The blind panel rated all 7 fails 0/3 — so the
drop is REAL, not over-strict grading. Lesson: a lenient judge said 90%; a rigorous judge + a
visible agent that samples at temperature says 77%. Eval is the centerpiece *because* it moved the number.

The 7 fails (all 0/3 from the panel): AMCR 00799 (quick-ratio numerator 0.53→0.57 vs gold 0.67→0.69),
AMD 00222 (wrong current-liabilities denom → 1.65x vs 1.57x), AXP 00476 (hedged, never concluded
"none"), AXP 01028 (listed countries, not EMEA/APAC/LACC segments), BA 00678 (opposite gross-margin
conclusion), BA 00585 (tax-rate sign+direction flipped vs gold), PEP 01328 (said $0; gold $411M).

Next:
- ⬜ Stabilise the number: pin the AGENT to temperature=0 for eval (or report pass@k mean±std) —
  AMD 00222 / BA 00678 are agent-variance flips, so the headline wobbles run-to-run.
- ⬜ Consider judge_model = Opus (one-line `JUDGE_MODEL` swap) to cut self-enhancement bias (judge
  currently = agent model, sonnet-4-6); the panel already cross-checks it.
- ⬜ Scale the eval to more FinanceBench filings (bigger, more trustworthy n).
- ⬜ Use `scripts/regrade.py` for cheap judge iteration (re-grades stored FULL answers, no agent re-run).
- ⬜ Low priority: expand FMP concepts; the geo-segment (AXP 01028) retrieval miss.

## Error analysis — the 7 misses (2026-05-31)
Workflow `scripts/error_analysis.workflow.js` (analyst + independent verifier per failure) →
`results/error_analysis.json`. The 7 failures cluster into 4 buckets:
- RETRIEVAL MISS (2, HIGH): AXP 01028 geo-segments (Note 24 region table buried at rank 40, acronyms
  EMEA/APAC/LACC never surfaced), BA 00678 gross margin (income statement not retrieved). → table-aware
  chunking + index the primary financial statements / note tables as discrete, high-priority chunks.
- AGENT HAD THE DATA, MIS-HANDLED IT (4): AMD 00222 (re-summed line items, dropped $336M instead of using
  the printed "Total current liabilities" subtotal — add a reconcile-against-stated-total self-check),
  BA 00585 (effective-tax-rate sign on a loss — normalize benefit→negative), AXP 00476 (hedged instead
  of committing to "none" — add an absence/negative-evidence answer pattern), PEP 01328 (collapsed a
  disclosed $411M to "$0" — report the disclosed figure).
- EVAL-QUALITY (2, overlap w/ above): AMCR 00799 (quick-ratio DEFINITION mismatch — agent used strict
  cash+receivables = 0.53/0.57, gold used (curr assets−inv) = 0.67/0.69; both valid), PEP 01328 (question
  literally says "state 0 if not explicitly outlined"). → motivates curating our OWN cleaner evals.
Takeaway: 4/7 misses the agent had the evidence in hand (fixable with targeted prompt guidance + temp=0);
2/7 are true retrieval misses (fixable with better chunking); 2/7 are partly the eval's fault.

Fix roadmap (cheapest-highest-leverage first): (1) agent temperature=0 (stabilise + AMD 00222 is partly
variance); (2) 2 prompt fixes — "commit to none on absence Qs", "report disclosed figures, don't collapse
to $0"; (3) table-aware chunking / index financial statements as discrete chunks (fixes both HIGH-sev
retrieval misses, helps whole corpus); (4) reconcile-against-stated-subtotal self-check; (5) eval hygiene.

## Things built this session (reference)
- src/sec_filings/retrieval/: embedding.py · vector_store.py · lexical.py · hybrid.py
- src/sec_filings/tools/: fmp.py (+FMP_TOOL_SCHEMA) · retrieve.py (+RETRIEVE_TOOL_SCHEMA)
- src/sec_filings/agent/loop.py (run_agent → AgentTrace)
- src/sec_filings/observability.py — Langfuse seam (no-op if unconfigured); loop.py traces through it
- scripts/build_index.py (writes data/chroma_manifest.json) · scripts/ask.py (smoke harness)
- notebooks/01_explore.ipynb — interactive: ask agent / inspect trace / raw retrieval / FMP / manifest
- Agent recovers from an unindexed-year request by retrying with an available year (loop.py catches LookupError → is_error → retry).
- (2026-05-30 corpus session) scripts/build_starter_corpus.py · extract_financebench_evalset.py · preflight_filing.py · audit_answerability.py
- ingest._coalesce_sections — merges edgartools' split Items (AXP returns Item 8 as TWO sections) so chunk_ids stay unique; fixes a Chroma DuplicateIDError. tests/corpus/test_ingest.py pins it.
- build_index.py: `voyage_throttle` config flag (default off = paid full-speed; on = free-tier pacing) + incremental manifest merge (add filings without re-embedding prior ones; preserves MSFT).
- EvalRecord +ticker/fiscal_year/doc_name/question_type/gold_evidence (FinanceBench curation labels).
- src/sec_filings/inspection.py (frontend-agnostic) + app/explorer.py (Streamlit) — the eval-miss / chunk-boundary explorer. streamlit added as a dev dep.
- notebooks/02_eval_explorer.ipynb — interactive eval browser: scorecard, pass/fail-by-company plot, full table (fails in red), per-question deep-dive (gold vs agent vs judge vs panel), live "ask the agent" cell. Backed by inspection.eval_records() (joins eval_run.json + judge_audit.json + eval set).
- scripts/run_eval.py — the real scoring harness: run_agent per Q → LLM-judge vs gold → results/eval_run.json (now stores FULL answers + grade_issues). (UTF-8 stdout for Windows.)
- (2026-05-31 judge-hardening) src/sec_filings/evaluation/judge.py — hardened LLM judge: forced `record_verdict` tool call, gold-evidence grounding, strict rubric + few-shot, robust text fallback parser, `JudgeError` (no silent wrong-default). config `judge_model` / JUDGE_MODEL knob.
- tests/evaluation/test_judge.py (12 parser tests) · scripts/validate_judge.py (12 labeled behavioural cases, gates a judge change) · scripts/regrade.py (re-grade stored full answers, no agent re-run) · scripts/judge_audit.workflow.js (independent 3-grader blind panel via the Workflow tool → results/judge_audit.json).
- scripts/audit_answerability.py / the per-Q classifier — the FLAWED retrieval-only probe (raw query); kept as a diagnostic, superseded by run_eval.py.

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
