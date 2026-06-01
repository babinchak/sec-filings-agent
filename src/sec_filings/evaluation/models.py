"""Data shapes for evaluation records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

FailureMode = Literal["segment", "multi_company", "routing", "calc_chain"]
EvalSource = Literal["financebench", "handwritten"]


class EvalRecord(BaseModel):
    """One question's labels + (after running) the system's result.

    Labels are set at curation time and never change. Result fields are filled
    by the eval runner and reflect the configuration of one specific run.
    """

    question_id: str
    question: str
    source: EvalSource

    # Labels (curation-time).
    gold_answer: str | None = None
    gold_chunk_ids: list[str] = Field(default_factory=list)
    failure_mode: FailureMode | None = None
    source_filings: list[str] = Field(
        default_factory=list,
        description="Accession numbers of filings the gold answer depends on.",
    )

    # FinanceBench curation labels (set when source == "financebench"). ticker +
    # fiscal_year let the eval runner route the retrieve tool at the right filing
    # without resolving the accession first; question_type mirrors FinanceBench's
    # own taxonomy (metrics-generated / domain-relevant / novel-generated).
    ticker: str | None = None
    fiscal_year: int | None = None
    doc_name: str | None = None
    question_type: str | None = None
    gold_evidence: str | None = Field(
        default=None,
        description="The benchmark's cited evidence text supporting the gold answer.",
    )

    # Result (run-time).
    predicted_answer: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    judge_score: float | None = None
    judge_rationale: str | None = None
    agent_trace_id: str | None = None

    # Run metadata.
    timestamp: datetime | None = None
    run_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of the config that produced this result: model, retrieval mode, etc.",
    )
