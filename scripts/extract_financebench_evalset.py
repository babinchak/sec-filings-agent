"""Extract the FinanceBench questions for our starter corpus into an eval set.

Downloads the FinanceBench open-source question set and writes the subset whose
source document is one of our five starter 10-Ks as ``EvalRecord`` JSONL under
``data/eval_sets/``. This is the ground truth the eval harness grades against and
is deliberately checked into git (see .gitignore) — it is the project's eval
artifact, not regenerable scratch.

We match on the exact FinanceBench ``doc_name`` and raise if any starter filing
yields zero questions, so a renamed/absent document surfaces loudly rather than
silently shrinking the set (no silent fallbacks — see PROJECT.md).

Run from project root:
    uv run python scripts/extract_financebench_evalset.py
"""

from __future__ import annotations

import json
import urllib.request

from sec_filings.config import settings
from sec_filings.evaluation.models import EvalRecord

_FINANCEBENCH_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/"
    "data/financebench_open_source.jsonl"
)

# FinanceBench ``doc_name`` -> (ticker, fiscal_year) for the five starter 10-Ks.
# Keyed on the exact doc_name string; the Amcor 10-Q/earnings docs are excluded by
# this exact match (we only want the 10-K).
_STARTER_DOCS: dict[str, tuple[str, int]] = {
    "AMD_2022_10K": ("AMD", 2022),
    "AMERICANEXPRESS_2022_10K": ("AXP", 2022),
    "BOEING_2022_10K": ("BA", 2022),
    "PEPSICO_2022_10K": ("PEP", 2022),
    "AMCOR_2023_10K": ("AMCR", 2023),
}

_OUTPUT = settings.eval_sets_dir / "financebench_starter.jsonl"


def _download() -> list[dict]:
    print(f"Downloading FinanceBench from {_FINANCEBENCH_URL} ...")
    with urllib.request.urlopen(_FINANCEBENCH_URL, timeout=60) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    print(f"  {len(records)} total FinanceBench questions downloaded.")
    return records


def _evidence_text(rec: dict) -> str | None:
    """Join the benchmark's cited evidence passages (the retrievable gold text)."""
    evidence = rec.get("evidence") or []
    texts = [e.get("evidence_text", "").strip() for e in evidence]
    joined = "\n\n".join(t for t in texts if t)
    return joined or None


def main() -> None:
    records = _download()

    selected: list[EvalRecord] = []
    seen_docs: set[str] = set()
    for rec in records:
        doc_name = rec.get("doc_name")
        if doc_name not in _STARTER_DOCS:
            continue
        ticker, fiscal_year = _STARTER_DOCS[doc_name]
        seen_docs.add(doc_name)
        selected.append(
            EvalRecord(
                question_id=rec["financebench_id"],
                question=rec["question"],
                source="financebench",
                gold_answer=rec.get("answer"),
                ticker=ticker,
                fiscal_year=fiscal_year,
                doc_name=doc_name,
                question_type=rec.get("question_type"),
                gold_evidence=_evidence_text(rec),
            )
        )

    missing = set(_STARTER_DOCS) - seen_docs
    if missing:
        raise RuntimeError(
            f"FinanceBench returned no questions for: {sorted(missing)}. "
            "Check the doc_name spellings in _STARTER_DOCS against the dataset."
        )

    settings.eval_sets_dir.mkdir(parents=True, exist_ok=True)
    with _OUTPUT.open("w", encoding="utf-8") as fh:
        for record in selected:
            fh.write(record.model_dump_json() + "\n")

    print(f"\nWrote {len(selected)} questions -> {_OUTPUT}")

    by_doc: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in selected:
        by_doc[r.doc_name or "?"] = by_doc.get(r.doc_name or "?", 0) + 1
        by_type[r.question_type or "?"] = by_type.get(r.question_type or "?", 0) + 1
    print("By filing: ", dict(sorted(by_doc.items(), key=lambda kv: -kv[1])))
    print("By type:   ", dict(sorted(by_type.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
