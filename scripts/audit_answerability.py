"""Emit per-question retrieval results for one filing, for an answerability audit.

For each FinanceBench starter question on the given (ticker, fiscal_year), this
runs the agent's own hybrid retrieval and prints — as one JSON object — the gold
answer/evidence alongside the passages our index surfaced. A judge (human or
agent) then decides whether the evidence the question needs is actually
retrievable from our corpus. This is retrieval-only: it never calls the LLM
agent, so it isolates "can we find the evidence?" from "can the model reason?".

Run from project root (after the filing is indexed):
    uv run python scripts/audit_answerability.py AMD 2022
"""

from __future__ import annotations

import json
import sys

from sec_filings.config import settings
from sec_filings.evaluation.models import EvalRecord
from sec_filings.retrieval.hybrid import hybrid_search

_EVAL_SET = settings.eval_sets_dir / "financebench_starter.jsonl"
_TOP_K = 6
_EVIDENCE_CAP = 1200  # chars of gold evidence to show the judge
_PASSAGE_CAP = 700  # chars per retrieved passage


def _load(ticker: str, fiscal_year: int) -> list[EvalRecord]:
    if not _EVAL_SET.exists():
        raise SystemExit(
            f"Eval set {_EVAL_SET} not found. Run scripts/extract_financebench_evalset.py first."
        )
    records: list[EvalRecord] = []
    with _EVAL_SET.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = EvalRecord.model_validate_json(line)
            if record.ticker == ticker and record.fiscal_year == fiscal_year:
                records.append(record)
    if not records:
        raise SystemExit(f"No starter questions for {ticker} {fiscal_year} in {_EVAL_SET}.")
    return records


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit("Usage: python scripts/audit_answerability.py <TICKER> <FISCAL_YEAR>")
    ticker, fiscal_year = argv[1].upper(), int(argv[2])
    records = _load(ticker, fiscal_year)

    questions = []
    for record in records:
        hits = hybrid_search(
            record.question, ticker=ticker, fiscal_year=fiscal_year, top_k=_TOP_K
        )
        questions.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "question_type": record.question_type,
                "gold_answer": record.gold_answer,
                "gold_evidence": (record.gold_evidence or "")[:_EVIDENCE_CAP],
                "retrieved": [
                    {
                        "item": hit.chunk.item,
                        "section_path": hit.chunk.section_path,
                        "score": round(hit.fused_score, 4),
                        "text": hit.chunk.text[:_PASSAGE_CAP],
                    }
                    for hit in hits
                ],
            }
        )

    print(
        json.dumps(
            {"ticker": ticker, "fiscal_year": fiscal_year, "questions": questions},
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv)
