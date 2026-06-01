"""LLM-as-judge for grading agent answers against FinanceBench gold answers.

Hardened over the v1 inline judge that lived in ``scripts/run_eval.py``. That
version had three weaknesses, each of which produced a wrong grade on the
starter set:

1. **Fragile output.** It asked for a free-text JSON object and parsed it with a
   regex under ``max_tokens=300``. On PEP_2022 restructuring the verdict JSON was
   truncated mid-string, failed ``json.loads``, and the code *silently* defaulted
   to "incorrect" — which only by luck was the right answer (the judge had been
   about to wrongly PASS a "$0" answer against a $411M gold).
2. **Thin grounding.** It saw only the terse gold-answer string, never the cited
   evidence passage, so it could not check whether the agent's *figure* matched
   the source.
3. **A lenient rubric.** It waved away a sign-flipped effective tax rate as "just
   a convention" and passed a hedged "I couldn't find the table" answer because
   the conclusion happened to land right.

This module fixes all three:

- The verdict is collected via **forced tool use** (``record_verdict``), so the
  model must emit a schema-shaped object — there is no prose JSON to truncate or
  mis-parse. ``parse_verdict`` is a text-JSON fallback for the rare case where a
  model emits prose anyway, and is what the unit tests pin.
- The judge is handed the **gold evidence** passage, so it can verify the agent's
  number against the source, not just against the one-line gold answer.
- The rubric names the failure modes explicitly and is calibrated with few-shot
  examples: a missing/zero/"cannot determine" figure, a wrong sign, or a hedge
  that lacks the data all grade INCORRECT.

Nothing here fails silently. If no verdict can be recovered after one retry,
``judge_answer`` raises :class:`JudgeError` and the caller decides how to record
it — a visible judge failure, never a value masquerading as a real "incorrect".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from sec_filings.config import settings

_MAX_REASON = 280
_MAX_TOKENS = 1024

JUDGE_SYSTEM = (
    "You are a strict grader for an AI agent that answers questions about SEC "
    "10-K filings. You compare the agent's answer to an authoritative benchmark "
    "GOLD ANSWER (and, when provided, the GOLD EVIDENCE passage the gold figures "
    "are drawn from). Decide CORRECT or INCORRECT.\n\n"
    "Mark CORRECT only when BOTH hold:\n"
    "1. SAME CONCLUSION — the agent reaches the same bottom line as the gold "
    "(same yes/no, same direction of change, same named item, or the same "
    "'this metric is not meaningful for this company').\n"
    "2. SAME KEY FIGURES — every figure the gold answer asserts appears in the "
    "agent's answer and matches within ~1% (relative). Rounding, units, and "
    "wording may differ: $9,068M = $9.07B = 9068 all match.\n\n"
    "Mark INCORRECT if ANY of these is true:\n"
    "- A required figure is missing, wrong by more than ~1%, or has the WRONG "
    "SIGN. A sign flip on a rate or amount is a real error — it flips expense vs "
    "benefit, gain vs loss. Never excuse it as 'just a convention' unless the "
    "magnitude AND the explicitly stated direction both clearly agree with the "
    "gold.\n"
    "- The agent answers '$0', 'not disclosed', 'not a separate line item', "
    "'cannot determine', or otherwise hedges that it lacks the data, while the "
    "gold gives a specific value. Failing to find or compute the answer is "
    "INCORRECT even when the hedge is technically worded.\n"
    "- The conclusion differs from the gold, or the agent's own reasoning "
    "contradicts the gold conclusion.\n\n"
    "Do NOT penalize extra correct detail, added context, or a more precise "
    "figure that still rounds to the gold. Reward substance over format.\n\n"
    "First list each discrepancy in `issues` (empty list if none), then set "
    "`correct`, then give a one-line `reason`. Always answer by calling "
    "record_verdict.\n\n"
    "Calibration examples:\n"
    "- GOLD '16.5%'; AGENT computes EBITDA/revenue = 16.5% -> issues [], correct "
    "true.\n"
    "- GOLD '$411 million'; AGENT '$0 - restructuring is not a separate income-"
    "statement line item' -> issues ['gold is $411M; agent answered $0/none'], "
    "correct false.\n"
    "- GOLD '0.67 -> 0.69'; AGENT '0.63 -> 0.69' -> issues ['FY2022 quick ratio "
    "0.63 vs gold 0.67'], correct false.\n"
    "- GOLD 'United States, EMEA, APAC, LACC'; AGENT names only the United States "
    "and U.S. segments -> issues ['omits EMEA/APAC/LACC geographic segments'], "
    "correct false.\n"
    "- GOLD 'Performance is not measured through operating margin'; AGENT "
    "'operating margin is not a meaningful metric for a bank like AXP because...' "
    "-> issues [], correct true."
)

_VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the grade for the agent's answer against the gold answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Each material discrepancy vs the gold answer: a wrong, "
                    "missing, or wrong-sign figure; a hedge / 'cannot answer'; or "
                    "a different conclusion. Empty list if there are none."
                ),
            },
            "correct": {
                "type": "boolean",
                "description": (
                    "True only if the answer reaches the same conclusion AND the "
                    "same key figure(s) as the gold answer."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One line (<=280 chars) justifying the grade.",
            },
        },
        "required": ["issues", "correct", "reason"],
    },
}


class JudgeError(RuntimeError):
    """Raised when no verdict can be recovered from the judge (after a retry)."""


@dataclass(frozen=True)
class Verdict:
    """A grading decision: the boolean, a one-line reason, and the discrepancies."""

    correct: bool
    reason: str
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"correct": self.correct, "reason": self.reason, "issues": list(self.issues)}


def build_judge_prompt(
    question: str, gold_answer: str, gold_evidence: str | None, agent_answer: str
) -> str:
    """Assemble the user-turn the judge grades. Includes evidence when we have it."""
    evidence = (gold_evidence or "").strip()
    ev_block = (
        f"\n\nGOLD EVIDENCE (source passage the gold figures come from):\n{evidence}"
        if evidence
        else ""
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER (benchmark, authoritative):\n{gold_answer}"
        f"{ev_block}\n\n"
        f"AGENT ANSWER (grade this):\n{agent_answer}\n\n"
        "Grade the agent answer against the gold answer by calling record_verdict."
    )


def _verdict_from_input(data: dict) -> Verdict | None:
    if "correct" not in data:
        return None
    return Verdict(
        correct=bool(data["correct"]),
        reason=str(data.get("reason", ""))[:_MAX_REASON],
        issues=tuple(str(i) for i in (data.get("issues") or [])),
    )


def parse_verdict(text: str) -> Verdict | None:
    """Best-effort recovery of a Verdict from a free-text judge response.

    Tolerates markdown code fences, prose around the JSON, and a verdict that was
    truncated after the boolean (the exact way the v1 judge crashed). Returns
    None only when no correctness signal can be recovered at all — the caller
    then retries or raises, never silently grades "incorrect".
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)

    # Prefer a well-formed object spanning the first '{' to the last '}'.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            verdict = _verdict_from_input(json.loads(cleaned[start : end + 1]))
            if verdict is not None:
                return verdict
        except json.JSONDecodeError:
            pass

    # Truncated or malformed: pull the boolean (and reason, if any) directly.
    bool_match = re.search(r'"correct"\s*:\s*(true|false)', cleaned, re.I)
    if bool_match:
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', cleaned)
        reason = reason_match.group(1)[:_MAX_REASON] if reason_match else ""
        return Verdict(
            correct=bool_match.group(1).lower() == "true",
            reason=reason or "(recovered from truncated verdict)",
        )
    return None


def _response_text(resp: anthropic.types.Message) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _verdict_from_response(resp: anthropic.types.Message) -> Verdict | None:
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == _VERDICT_TOOL["name"]:
            verdict = _verdict_from_input(block.input or {})
            if verdict is not None:
                return verdict
    # Model emitted prose instead of calling the tool — recover from text.
    return parse_verdict(_response_text(resp))


def judge_answer(
    client: anthropic.Anthropic | None,
    *,
    question: str,
    gold_answer: str,
    agent_answer: str,
    gold_evidence: str | None = None,
    model: str | None = None,
) -> Verdict:
    """Grade ``agent_answer`` against ``gold_answer`` (and evidence) with the judge.

    Empty answers short-circuit to INCORRECT without an API call (so ``client``
    may be None in that case — used by tests). Otherwise the verdict is forced
    through the ``record_verdict`` tool. Raises :class:`JudgeError` if no verdict
    can be recovered after one retry — never a silent default.
    """
    if not (agent_answer or "").strip():
        return Verdict(False, "empty answer", ("agent returned no answer",))
    if client is None:
        raise ValueError("client is required to grade a non-empty answer")

    model = model or settings.judge_model
    user = build_judge_prompt(question, gold_answer, gold_evidence, agent_answer)
    last_text = ""
    for _ in range(2):
        resp = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            system=JUDGE_SYSTEM,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": _VERDICT_TOOL["name"]},
            messages=[{"role": "user", "content": user}],
        )
        verdict = _verdict_from_response(resp)
        if verdict is not None:
            return verdict
        last_text = _response_text(resp)[:200]
    raise JudgeError(f"no parseable verdict after retry; last response text: {last_text!r}")
