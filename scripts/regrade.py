"""Re-grade stored agent answers with the current judge — no agent re-run.

Running the agent on the eval set is the slow, stable half of the loop; only the
JUDGE changes when we tune the rubric. This reads the FULL agent answers already
captured in results/eval_run.json (run_eval.py stores them in full) and re-grades
them with the current sec_filings.evaluation.judge — so a judge iteration costs
~30 cheap judge calls instead of 30 agent runs. Writes results/regrade.json and
prints the verdict diff vs the stored grades.

Requires an eval_run.json produced AFTER answers were stored in full (older runs
truncated them to 600 chars, which would grade unfairly). Run from the repo root:
    uv run python scripts/regrade.py
"""

from __future__ import annotations

import json
import sys

import anthropic

from sec_filings.config import settings
from sec_filings.evaluation.judge import JudgeError, judge_answer
from sec_filings.inspection import eval_questions

_IN = settings.results_dir / "eval_run.json"
_OUT = settings.results_dir / "regrade.json"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    if not _IN.exists():
        raise SystemExit(f"{_IN} not found — run scripts/run_eval.py first.")
    rows = json.loads(_IN.read_text(encoding="utf-8"))
    by_id = {q.question_id: q for q in eval_questions()}
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    out: list[dict] = []
    flips = 0
    for i, row in enumerate(rows, start=1):
        qid = row["qid"]
        q = by_id.get(qid)
        answer = row.get("agent_answer") or ""
        if q is None:
            print(f"[{i:>2}/{len(rows)}] skip {qid} (not in eval set)")
            continue
        if len(answer) == 600:
            print(f"[{i:>2}/{len(rows)}] WARN {qid}: answer is exactly 600 chars — likely truncated by an old run.")
        try:
            verdict = judge_answer(
                client,
                question=q.question,
                gold_answer=q.gold_answer or "",
                agent_answer=answer,
                gold_evidence=q.gold_evidence,
            )
            new_correct, reason, issues = verdict.correct, verdict.reason, list(verdict.issues)
        except JudgeError as exc:
            new_correct, reason, issues = False, f"JUDGE_ERROR: {exc}", []

        old_correct = bool(row.get("grade_correct"))
        flipped = old_correct != new_correct
        flips += flipped
        out.append(
            {
                "qid": qid,
                "ticker": row.get("ticker"),
                "old_correct": old_correct,
                "new_correct": new_correct,
                "flipped": flipped,
                "reason": reason,
                "issues": issues,
            }
        )
        flag = "  <-- FLIP" if flipped else ""
        mark = "OK" if new_correct else "no"
        print(f"[{i:>2}/{len(rows)}] {mark} {row.get('ticker')} {qid[-5:]} {reason[:80]}{flag}")

    old_acc = sum(1 for r in out if r["old_correct"])
    new_acc = sum(1 for r in out if r["new_correct"])
    n = len(out)
    print("\n" + "=" * 64)
    print(f"Re-graded {n} stored answers with the current judge.")
    print(f"Accuracy: stored {old_acc}/{n} -> regraded {new_acc}/{n}  ({flips} verdict flips)")
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
