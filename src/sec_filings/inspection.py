"""Read-only inspection helpers for the explorer (frontend-agnostic).

Everything the explorer shows is computed here as plain Python over our real
retrieval stack: hybrid retrieval with rank provenance, gold-evidence number
overlap, and per-filing chunk layout. Kept UI-free on purpose — the same
functions can sit behind a notebook, a CLI, or a future web frontend; the
Streamlit app (app/explorer.py) is just a thin shell over these. Read-only: it
never writes the index.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from sec_filings.config import settings
from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.ingest import ingest_filing
from sec_filings.corpus.models import Chunk, Filing
from sec_filings.evaluation.models import EvalRecord
from sec_filings.retrieval.hybrid import hybrid_search

_EVAL_SET = settings.eval_sets_dir / "financebench_starter.jsonl"

# A "salient number" = a run of 3+ digits/commas, e.g. 4,835 or 15019. Financial
# evidence is numeric, so overlap on these is a cheap proxy for "does this chunk
# actually contain the figures the gold answer needs". Tiny ordinals (1, 2) are
# excluded deliberately — they are noise.
_NUM_RE = re.compile(r"\d[\d,]{2,}\b")


def numbers(text: str | None) -> set[str]:
    """Salient numeric tokens in `text`, comma form preserved (e.g. '4,835')."""
    return {m.group() for m in _NUM_RE.finditer(text or "")}


def _norm(values: set[str]) -> set[str]:
    """Compare numbers ignoring thousands separators ('4,835' == '4835')."""
    return {v.replace(",", "") for v in values}


def eval_questions() -> list[EvalRecord]:
    """The FinanceBench starter questions (the eval set), in file order."""
    if not _EVAL_SET.exists():
        raise FileNotFoundError(
            f"{_EVAL_SET} not found — run scripts/extract_financebench_evalset.py."
        )
    return [
        EvalRecord.model_validate_json(line)
        for line in _EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def eval_records() -> list[dict]:
    """Merge the eval run, the independent-panel audit, and the question labels.

    Returns one dict per starter question combining: the question + gold evidence
    (from the eval set), the agent's full answer + the hardened judge's verdict
    (from results/eval_run.json), and the blind 3-grader panel's verdict (from
    results/judge_audit.json, when present). This is the join the eval notebook
    renders, kept here so the notebook stays display-only.
    """
    run_path = settings.results_dir / "eval_run.json"
    if not run_path.exists():
        raise FileNotFoundError(f"{run_path} not found — run scripts/run_eval.py first.")
    rows = json.loads(run_path.read_text(encoding="utf-8"))

    audit: dict[str, dict] = {}
    audit_path = settings.results_dir / "judge_audit.json"
    if audit_path.exists():
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        audit = {q["qid"]: q for q in data.get("per_question", [])}

    labels = {q.question_id: q for q in eval_questions()}

    merged: list[dict] = []
    for r in rows:
        a = audit.get(r["qid"], {})
        q = labels.get(r["qid"])
        merged.append(
            {
                **r,
                "question": q.question if q else "",
                "gold_evidence": (q.gold_evidence if q else "") or "",
                "panel_correct": a.get("panel_correct"),
                "panel_votes": a.get("panel_votes"),
                "grader_notes": a.get("grader_notes", []),
            }
        )
    return merged


class RetrievedChunk(BaseModel):
    """One retrieval hit with fusion provenance and gold-overlap, for display."""

    rank: int
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None
    chunk_id: str
    item: str
    section_path: list[str]
    text: str
    token_count: int
    char_start: int
    char_end: int
    gold_numbers_hit: list[str]
    gold_overlap: float


def retrieve_ranked(
    query: str,
    *,
    ticker: str,
    fiscal_year: int,
    top_k: int,
    gold_evidence: str | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval and annotate each hit with gold-number overlap."""
    gold = numbers(gold_evidence)
    gold_norm = _norm(gold)
    scored = hybrid_search(query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k)
    results: list[RetrievedChunk] = []
    for rank, sc in enumerate(scored, start=1):
        chunk_nums = numbers(sc.chunk.text)
        hit = sorted(gold & chunk_nums) if gold else []
        overlap = (
            len(gold_norm & _norm(chunk_nums)) / len(gold_norm) if gold_norm else 0.0
        )
        results.append(
            RetrievedChunk(
                rank=rank,
                fused_score=sc.fused_score,
                dense_rank=sc.dense_rank,
                lexical_rank=sc.lexical_rank,
                chunk_id=sc.chunk.chunk_id,
                item=sc.chunk.item,
                section_path=sc.chunk.section_path,
                text=sc.chunk.text,
                token_count=sc.chunk.token_count,
                char_start=sc.chunk.char_start,
                char_end=sc.chunk.char_end,
                gold_numbers_hit=hit,
                gold_overlap=overlap,
            )
        )
    return results


def best_gold_overlap(
    question: str,
    *,
    ticker: str,
    fiscal_year: int,
    gold_evidence: str | None,
    search_depth: int = 40,
) -> tuple[int | None, float]:
    """Rank (1-based) and overlap of the chunk best covering the gold numbers.

    Searches `search_depth` deep with the RAW question and reports where the best
    gold-number-overlap chunk lands. High overlap at a deep rank is the "evidence
    is chunked fine but the query doesn't retrieve it" signal; a low best overlap
    means a genuine chunking gap. Returns (None, 0.0) for prose (number-free)
    evidence, where this numeric proxy doesn't apply.
    """
    gold_norm = _norm(numbers(gold_evidence))
    if not gold_norm:
        return (None, 0.0)
    scored = hybrid_search(
        question, ticker=ticker, fiscal_year=fiscal_year, top_k=search_depth
    )
    best_rank: int | None = None
    best_ov = 0.0
    for rank, sc in enumerate(scored, start=1):
        ov = len(gold_norm & _norm(numbers(sc.chunk.text))) / len(gold_norm)
        if ov > best_ov:
            best_ov, best_rank = ov, rank
    return (best_rank, best_ov)


def filing_chunks(ticker: str, fiscal_year: int) -> tuple[Filing, list[Chunk]]:
    """Ingest + chunk a filing (deterministic; matches the index) for layout views."""
    filing = ingest_filing(ticker, fiscal_year)
    return filing, chunk_filing(filing)
