"""Unit tests for the verdict parser and the no-API paths of the judge.

These pin the robustness the v1 judge lacked: the live judge now forces its
verdict through a tool call, but if a model ever emits prose instead we must
still recover a grade — and we must never silently default a truncated verdict
to "incorrect" the way the v1 regex did. No network here; the live grading path
is exercised by scripts/validate_judge.py against the real API.
"""

from __future__ import annotations

import pytest

from sec_filings.evaluation.judge import (
    Verdict,
    build_judge_prompt,
    judge_answer,
    parse_verdict,
)


def test_parses_clean_json_object():
    v = parse_verdict('{"issues": [], "correct": true, "reason": "matches"}')
    assert v == Verdict(correct=True, reason="matches", issues=())


def test_parses_false_with_issues():
    v = parse_verdict('{"issues": ["FY22 0.63 vs gold 0.67"], "correct": false, "reason": "wrong figure"}')
    assert v is not None
    assert v.correct is False
    assert v.issues == ("FY22 0.63 vs gold 0.67",)


def test_strips_markdown_code_fences():
    text = '```json\n{"issues": [], "correct": true, "reason": "ok"}\n```'
    v = parse_verdict(text)
    assert v is not None and v.correct is True


def test_recovers_from_prose_around_json():
    text = 'Here is my grade:\n{"issues": [], "correct": false, "reason": "no"}\nThanks.'
    v = parse_verdict(text)
    assert v is not None and v.correct is False


def test_recovers_from_truncated_verdict():
    # The exact v1 failure: max_tokens cut the JSON mid-string. v1 defaulted to
    # "incorrect" silently; we must instead recover the boolean that WAS emitted.
    text = '{"correct": true, "reason": "The agent correctly identifies that restructuring costs are not explici'
    v = parse_verdict(text)
    assert v is not None
    assert v.correct is True
    assert "restructuring" in v.reason


def test_returns_none_when_no_signal():
    assert parse_verdict("I think the answer looks fine overall.") is None
    assert parse_verdict("") is None


def test_caps_reason_length():
    long_reason = "x" * 5000
    v = parse_verdict(f'{{"issues": [], "correct": true, "reason": "{long_reason}"}}')
    assert v is not None and len(v.reason) <= 280


def test_empty_answer_short_circuits_without_client():
    # No client needed: an empty answer is graded incorrect before any API call.
    v = judge_answer(None, question="q", gold_answer="g", agent_answer="   ")
    assert v.correct is False
    assert "empty" in v.reason.lower()


def test_non_empty_answer_requires_client():
    with pytest.raises(ValueError):
        judge_answer(None, question="q", gold_answer="g", agent_answer="a real answer")


def test_prompt_includes_evidence_when_present():
    prompt = build_judge_prompt("Q?", "GOLD", "the cited passage", "AGENT")
    assert "GOLD EVIDENCE" in prompt
    assert "the cited passage" in prompt


def test_prompt_omits_evidence_block_when_absent():
    prompt = build_judge_prompt("Q?", "GOLD", None, "AGENT")
    assert "GOLD EVIDENCE" not in prompt


def test_verdict_as_dict_roundtrip():
    v = Verdict(correct=False, reason="r", issues=("a", "b"))
    assert v.as_dict() == {"correct": False, "reason": "r", "issues": ["a", "b"]}
