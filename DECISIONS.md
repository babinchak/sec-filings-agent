# Decisions

Lightweight ADRs for the load-bearing choices on this project. Each is short by design — if it can't be explained in under ~100 words, the decision isn't crisp enough yet.

---

## ADR-001 — FinancialModelingPrep for structured financial data

**Decision:** FMP as the numeric backend.

**Alternatives considered:** SEC XBRL directly, SimFin, hand-curated CSV.

**Why:** XBRL concept-name normalization (`us-gaap:ResearchAndDevelopmentExpense`, segment tags, etc.) is a tarpit that would steal weeks from the eval work that's actually the portfolio centerpiece. FMP's clean API is worth ~$15-30/mo. Two of the four target failure modes (routing, calculation chains) depend on this tool being clean — flakiness here would corrupt the eval signal.

---

## ADR-002 — Two eval sets: FinanceBench + hand-written failure-mode set

**Decision:** Use both, for different jobs.

**Why:** FinanceBench provides external credibility (anchor against published baselines). The hand-written set demonstrates *eval design skill* — the actual portfolio centerpiece. They're complementary, not competing. Corpus selection prioritizes FB overlap (target ≥50 usable FB questions) so we keep the external anchor.

---

## ADR-003 — Hand-eval probes specific failure modes

**Decision:** Hand-written set is built around pre-defined failure modes; each item is labeled with the mode it probes.

**Alternatives considered:** Retrieval-only (Q → relevant chunks), or FinanceBench-style end-to-end QA.

**Why:** The "eval engineer" narrative is strongest when each item targets a specific failure mode and each ablation can be shown to fix one or more of them. That turns the results table into a story ("the reranker fixes terminology-mismatch but not calculation chains") rather than a single accuracy number.

---

## ADR-004 — The four target failure modes

**Decision:** Segment/terminology mismatch · multi-company comparison · numeric-vs-narrative routing · calculation chains.

**Why:** Each pairs cleanly with an ablation that should improve it — hybrid retrieval, sub-query decomposition, router, tool chaining respectively. Together they cover retrieval quality, agent orchestration, routing, and tool use. Each is plausibly hard for a naive baseline, so the ablations have signal instead of saturating.

---

## ADR-005 — Level 3 agent (router + decomposition + tool chaining)

**Decision:** Agent loop supports routing between tools, sub-query decomposition (one sub-query per entity/year), and multi-step tool chaining where step N's output informs step N+1.

**Alternatives considered:** Router-only (Level 1-2), or add self-correction (Level 4).

**Why:** Calculation chains and multi-company comparison failure modes require at least Level 3. Self-correction (Level 4) was deferred — eval gets harder under non-determinism, and it can be added as an ablation later if Phase 6 failure analysis shows it would help.

---

## ADR-006 — Parallel-track build order

**Decision:** Build retrieval and agent in parallel from Phase 1, both on a single filing, before scaling corpus.

**Alternatives considered:** Retrieval-first (PROJECT.md's original framing), vertical-slice-then-harden.

**Why:** Retrieval problems that actually matter end-to-end are only visible once an agent is consuming the results. A parallel MVP on one filing gives fast feedback without over-investing in throwaway code. The substrate-first ordering risks polishing the wrong retrieval problems.

---

## ADR-007 — No issue tracker for now

**Decision:** Track work in `ROADMAP.md` and Claude Code plans/todos. No GitHub issues.

**Why:** Solo project, no coordination overhead to amortize. Issues add ceremony without benefit at this scale. Revisit if/when several parallel threads emerge, or for "looks real to interviewers" polish near Phase 7.

---

## ADR-008 — No LangChain / LlamaIndex / LangGraph

**Decision:** Hand-write the agent loop, retrieval orchestration, and eval runner against the Anthropic SDK and minimal libraries.

**Why:** The project's value is demonstrating understanding of the fundamentals. Framework abstractions hide exactly the design decisions an interviewer wants to discuss. Each new dependency must justify itself.
