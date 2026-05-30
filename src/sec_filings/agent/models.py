"""Data shapes for agent traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

StepType = Literal["reasoning", "tool_call", "tool_result", "answer"]


class AgentStep(BaseModel):
    """One step inside an agent trace."""

    step_index: int
    type: StepType
    content: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class AgentTrace(BaseModel):
    """The full record of one agent invocation."""

    trace_id: str
    question: str
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str = ""
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float | None = None
    elapsed_seconds: float | None = None
    model: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Set when Langfuse tracing is enabled — the id of the matching Langfuse
    # trace, so a notebook/CLI can deep-link to its waterfall view.
    langfuse_trace_id: str | None = None
