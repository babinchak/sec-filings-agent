"""Phase-1 end-to-end smoke test: ask the agent and assert tool ROUTING.

This is the integration check that proves the whole stack wires up — the
hand-written tool-use loop (``agent/loop.py:run_agent``), the two native tools
(``retrieve`` over the persisted MSFT-2023 hybrid index, ``get_financial_fact``
over FMP), and Claude's routing between them. It is deliberately a *routing*
test, not an answer-grading test: for each question we assert the SET of tool
names the agent actually invoked equals the expected set (and, for the chained
ratio case, that ``get_financial_fact`` is called more than once). Asserting the
SET — not call order or exact count — keeps the check robust to the LLM's
freedom to reorder or batch tool calls while still catching genuine mis-routes
(e.g. the Q4 trap, where "invests in R&D" reads numeric but is narrative).

Spends real Anthropic + FMP tokens by design (see PROJECT.md — fine for the
smoke test). Run from project root:
    uv run python scripts/ask.py            # all questions
    uv run python scripts/ask.py 3          # only question #3 (1-based)
"""

from __future__ import annotations

import sys
from collections import Counter

# The agent's answers can contain Unicode the model likes for finance (×, ÷, →,
# even emoji). Windows' default console encoding is cp1252 and chokes on those,
# which would crash our *printing* of an otherwise-correct answer. Force UTF-8 on
# stdout/stderr so the smoke test never dies on output (Python 3.7+ reconfigure).
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from sec_filings.agent.loop import run_agent
from sec_filings.agent.models import AgentTrace

# Each question pairs with the routing we expect Claude to pick. ``expected`` is
# the SET of tool names that MUST appear; ``min_fact_calls`` (>1 only for the
# chained ratio) additionally requires get_financial_fact to fire that many times
# so we verify the model fetched BOTH operands rather than estimating the ratio.
QUESTIONS: list[dict] = [
    {
        # Pure narrative: AI risk factors live in the filing prose, not a number.
        "question": "How does Microsoft describe its principal risks related to artificial intelligence?",
        "expected": {"retrieve"},
    },
    {
        # Exact reported figure -> numeric tool (answer ~ $211.9B).
        "question": "What was Microsoft's total revenue in fiscal year 2023?",
        "expected": {"get_financial_fact"},
    },
    {
        # Exact reported figure -> numeric tool (answer ~ $27.2B).
        "question": "What were Microsoft's research and development expenses in fiscal 2023?",
        "expected": {"get_financial_fact"},
    },
    {
        # The router TRAP: "how does it invest in R&D" sounds numeric but asks for
        # narrative/strategy -> must route to retrieve, NOT get_financial_fact.
        "question": "How does Microsoft say it invests in research and development?",
        "expected": {"retrieve"},
    },
    {
        # Derived metric -> CHAIN get_financial_fact twice (R&D + revenue), then
        # compute the ratio. No retrieval at all (answer ~ 12.8%).
        "question": "What was Microsoft's R&D spend as a percentage of revenue in fiscal 2023?",
        "expected": {"get_financial_fact"},
        "min_fact_calls": 2,
    },
]


def tools_used(trace: AgentTrace) -> Counter[str]:
    """Tally tool_name across the trace's tool_call steps.

    A Counter (not a set) so callers can both check the SET of names and the
    per-tool call count (the chained-ratio case needs get_financial_fact >= 2).
    """
    return Counter(
        step.tool_name
        for step in trace.steps
        if step.type == "tool_call" and step.tool_name is not None
    )


def check_routing(spec: dict, used: Counter[str]) -> bool:
    """True iff the agent's tool usage matches this question's expected routing.

    Asserts the SET of invoked tools equals ``expected`` (catches a missing or a
    stray tool), plus — when ``min_fact_calls`` is set — that get_financial_fact
    fired at least that many times (proves the ratio was chained, not estimated).
    """
    if set(used) != spec["expected"]:
        return False
    min_fact = spec.get("min_fact_calls")
    if min_fact is not None and used.get("get_financial_fact", 0) < min_fact:
        return False
    return True


def _excerpt(text: str, limit: int = 280) -> str:
    """One-line, length-capped view of the final answer for terminal output.

    Collapses whitespace, caps length, and round-trips through the active stdout
    encoding (replacing any un-encodable glyph) so printing an answer with finance
    symbols/emoji can never crash the smoke test on a non-UTF-8 console.
    """
    flat = " ".join(text.split())
    capped = flat if len(flat) <= limit else flat[: limit - 1] + "..."
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return capped.encode(encoding, errors="replace").decode(encoding)


def ask_one(n: int, spec: dict) -> bool:
    """Run one question through the agent, print PASS/FAIL + answer, return ok.

    A tool can RAISE mid-loop (e.g. ``retrieve`` against an un-indexed
    (ticker, year) -> ``LookupError`` from the manifest, which ``loop.py`` does
    NOT convert to a recoverable is_error). We CATCH that here so one failing
    question still reports honestly and does not abort the whole smoke suite —
    we never swallow it into a fake PASS (see PROJECT.md: report misses honestly).
    """
    question = spec["question"]
    print(f"\n=== Q{n}: {question}")

    expected_str = ", ".join(sorted(spec["expected"]))
    if "min_fact_calls" in spec:
        expected_str += f" (get_financial_fact>={spec['min_fact_calls']})"

    try:
        trace = run_agent(question)
    except Exception as exc:  # noqa: BLE001 — surface ANY agent/tool failure honestly
        print(f"  expected   : {expected_str}")
        print(f"  routing    : ERROR ({type(exc).__name__}: {_excerpt(str(exc), 200)})")
        return False

    used = tools_used(trace)
    ok = check_routing(spec, used)

    used_str = ", ".join(f"{name}x{cnt}" for name, cnt in sorted(used.items())) or "(none)"
    print(f"  tools_used : {used_str}")
    print(f"  expected   : {expected_str}")
    print(f"  routing    : {'PASS' if ok else 'FAIL'}")
    print(f"  answer     : {_excerpt(trace.final_answer)}")
    return ok


def main(argv: list[str]) -> None:
    """CLI entry point. Optional 1-based question index selects a single question."""
    args = argv[1:]
    if args:
        idx = int(args[0])
        if not 1 <= idx <= len(QUESTIONS):
            raise SystemExit(
                f"Question index {idx} out of range 1..{len(QUESTIONS)}."
            )
        selected = [(idx, QUESTIONS[idx - 1])]
    else:
        selected = list(enumerate(QUESTIONS, start=1))

    results: list[tuple[int, bool]] = []
    for n, spec in selected:
        results.append((n, ask_one(n, spec)))

    print("\n=== SUMMARY ===")
    for n, ok in results:
        print(f"  Q{n}: {'PASS' if ok else 'FAIL'}")
    passed = sum(1 for _, ok in results if ok)
    all_passed = passed == len(results)
    print(f"\nrouting: {passed}/{len(results)} passed; all_routing_passed={all_passed}")
    # Non-zero exit on any mis-route so CI / callers can gate on the smoke test.
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main(sys.argv)
