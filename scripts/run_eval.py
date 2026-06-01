"""Eval v1 — run the REAL agent on the FinanceBench starter set and GRADE answers.

The first answerability audit measured the wrong thing twice over: it fed the RAW
question to retrieval (the agent reformulates), and it ignored that the agent has
TWO tools. The only honest "is it passing?" is end-to-end: run the actual
``run_agent`` on each question, then LLM-judge its final answer against
FinanceBench's gold answer. Alongside the grade we record diagnostics — which
tools the agent used, which FMP concepts it tried (most are unsupported today),
whether it routed to the right filing, and whether the gold numbers were in what
it retrieved (raw-question baseline vs. the agent's own reformulated retrieval).

Writes results/eval_run.json and prints a per-question line + an aggregate. Hits
Anthropic (agent + judge) and Voyage; ~30 agent runs, a few minutes — run it in
the background.

Run from the repo root:
    uv run python scripts/run_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import anthropic

from sec_filings.agent.loop import run_agent
from sec_filings.config import settings
from sec_filings.evaluation.judge import JudgeError, judge_answer
from sec_filings.inspection import eval_questions, numbers
from sec_filings.retrieval.hybrid import hybrid_search

_OUT = settings.results_dir / "eval_run.json"
_COVER = 0.5  # fraction of gold numbers that must appear to count as "covered"


def _norm(values: set[str]) -> set[str]:
    return {v.replace(",", "") for v in values}


def _covered(gold_norm: set[str], texts: list[str]) -> bool | None:
    """True/False if gold numbers are in the texts; None for prose (no numbers)."""
    if not gold_norm:
        return None
    union: set[str] = set()
    for text in texts:
        union |= _norm(numbers(text))
    return len(gold_norm & union) / len(gold_norm) >= _COVER


def _trace_tools(trace, ticker: str, fiscal_year: int):
    """Extract the agent's tool behaviour: retrieve queries, FMP attempts, routing."""
    retrieve_queries: list[dict] = []
    fmp_calls: list[dict] = []
    correct_filing_texts: list[str] = []
    routed_ok = False
    pending: tuple[str, int] | None = None
    for step in trace.steps:
        if step.type == "tool_call":
            ti = step.tool_input or {}
            if step.tool_name == "retrieve":
                tk = str(ti.get("ticker") or "MSFT").upper()
                fy = int(ti.get("fiscal_year") or 2023)
                retrieve_queries.append({"query": ti.get("query"), "ticker": tk, "fiscal_year": fy})
                pending = (tk, fy)
            elif step.tool_name == "get_financial_fact":
                fmp_calls.append(
                    {"concept": ti.get("concept"), "ticker": ti.get("ticker"), "year": ti.get("year")}
                )
        elif step.type == "tool_result" and step.tool_name == "retrieve" and pending is not None:
            tk, fy = pending
            passages = step.tool_output.get("passages", []) if isinstance(step.tool_output, dict) else []
            if (tk, fy) == (ticker.upper(), fiscal_year):
                routed_ok = True
                correct_filing_texts.extend(
                    p.get("text", "") for p in passages if isinstance(p, dict)
                )
            pending = None
    return retrieve_queries, fmp_calls, correct_filing_texts, routed_ok


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so non-ASCII in any printed
    # value can't crash a multi-minute run (results are written as UTF-8 anyway).
    # line_buffering=True flushes each per-question line immediately even when
    # stdout is redirected to a file (background run) — otherwise Python block-
    # buffers and the progress lines don't appear until the process exits, which
    # looks exactly like a hang.
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    questions = eval_questions()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    rows: list[dict] = []

    for i, q in enumerate(questions, start=1):
        gold_norm = _norm(numbers(q.gold_evidence))

        raw_hits = hybrid_search(q.question, ticker=q.ticker, fiscal_year=q.fiscal_year, top_k=8)
        raw_cov = _covered(gold_norm, [h.chunk.text for h in raw_hits])

        error = None
        try:
            trace = run_agent(q.question)
            retrieve_queries, fmp_calls, texts, routed_ok = _trace_tools(
                trace, q.ticker, q.fiscal_year
            )
            agent_cov = _covered(gold_norm, texts)
            answer = trace.final_answer
            try:
                grade = judge_answer(
                    client,
                    question=q.question,
                    gold_answer=q.gold_answer or "",
                    agent_answer=answer,
                    gold_evidence=q.gold_evidence,
                ).as_dict()
            except JudgeError as exc:
                # A judge that can't produce a verdict is a VISIBLE failure, not a
                # silent "incorrect" (the v1 bug). Flag it so it's distinguishable.
                grade = {"correct": False, "reason": f"JUDGE_ERROR: {exc}", "issues": []}
            n_steps = len(trace.steps)
        except Exception as exc:  # one bad question must not sink the sweep
            error = f"{type(exc).__name__}: {exc}"
            retrieve_queries, fmp_calls, texts, routed_ok = [], [], [], False
            agent_cov, answer, n_steps = None, "", 0
            grade = {"correct": False, "reason": "agent raised", "issues": []}

        rows.append(
            {
                "qid": q.question_id,
                "ticker": q.ticker,
                "fiscal_year": q.fiscal_year,
                "type": q.question_type,
                "numeric": bool(gold_norm),
                "grade_correct": grade["correct"],
                "grade_reason": grade["reason"],
                "grade_issues": grade.get("issues", []),
                "raw_retrieval_covered": raw_cov,
                "agent_retrieval_covered": agent_cov,
                "routed_ok": routed_ok,
                "n_retrieve_calls": len(retrieve_queries),
                "retrieve_queries": [r["query"] for r in retrieve_queries],
                "fmp_concepts_tried": [c["concept"] for c in fmp_calls],
                "n_steps": n_steps,
                "gold_answer": q.gold_answer,
                "agent_answer": answer,  # FULL answer — lets regrade.py re-judge without re-running the agent
                "error": error,
            }
        )

        if error:
            mark = "ERR"
        elif grade["correct"]:
            mark = "OK"
        elif grade["reason"].startswith("JUDGE_ERROR"):
            mark = "J?"
        else:
            mark = "no"
        print(
            f"[{i:>2}/{len(questions)}] {mark} {q.ticker} {q.question_id[-5:]} "
            f"{q.question_type:<16} retr(raw={raw_cov} agent={agent_cov}) "
            f"fmp={len(fmp_calls)} route={'Y' if routed_ok else '.'}"
        )

    n = len(rows)
    correct = sum(1 for r in rows if r["grade_correct"])
    numeric = [r for r in rows if r["numeric"]]
    raw_num = sum(1 for r in numeric if r["raw_retrieval_covered"])
    agent_num = sum(1 for r in numeric if r["agent_retrieval_covered"])
    routed = sum(1 for r in rows if r["routed_ok"])
    errors = sum(1 for r in rows if r["error"])
    fmp_tried = Counter(c for r in rows for c in r["fmp_concepts_tried"])

    print("\n" + "=" * 64)
    print(f"FinanceBench starter — ANSWER ACCURACY: {correct}/{n} = {correct / n:.0%}")
    print("By type:")
    for t in ("domain-relevant", "novel-generated", "metrics-generated"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            ok = sum(1 for r in sub if r["grade_correct"])
            print(f"    {t:<18} {ok}/{len(sub)}")
    print(
        f"Retrieval coverage (numeric Qs): raw-question {raw_num}/{len(numeric)} "
        f"-> real-agent {agent_num}/{len(numeric)}"
    )
    print(f"Routing correct (searched right filing): {routed}/{n}")
    print(f"FMP concepts the agent tried: {dict(fmp_tried)}")
    print(f"Agent errors: {errors}/{n}")

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {_OUT}")


if __name__ == "__main__":
    main()
