"""Behavioural regression test for the LLM judge (sec_filings.evaluation.judge).

The judge is graded the same way it grades: against labeled cases. Each case is
an (answer, gold) pair with a known-correct verdict, chosen to pin the exact
failure modes that made the v1 judge unreliable on the starter set — a "$0 / not
a line item" answer against a real-number gold, a wrong figure, a missing
required item, a direction flip, and a pure "I can't find it" hedge — plus the
benign cases it must NOT over-penalize (reworded, unit-equivalent, within ~1%,
extra correct detail, "metric not meaningful"). Hits the real API (~12 judge
calls) and exits non-zero if any case is graded wrong, so it can gate a judge
change before a full eval re-run.

Run from the repo root:
    uv run python scripts/validate_judge.py
"""

from __future__ import annotations

import sys

import anthropic

from sec_filings.config import settings
from sec_filings.evaluation.judge import JudgeError, judge_answer

# (name, question, gold_answer, gold_evidence, agent_answer, expected_correct)
CASES = [
    (
        "reworded_correct",
        "Did one customer account for a large share of AMD's FY2022 revenue?",
        "Yes, one customer accounted for 16% of consolidated net revenue.",
        "One customer accounted for 16% of the Company's consolidated net revenue in 2022.",
        "Yes — AMD disclosed that a single customer (Customer A) represented 16% of consolidated net revenue in FY2022.",
        True,
    ),
    (
        "zero_vs_real_number",  # the PEP restructuring bug the v1 judge nearly passed
        "What was PepsiCo's FY2022 restructuring cost?",
        "Pepsico's restructuring costs in FY2022 amounted to $411 million.",
        "Restructuring and impairment charges were $411 million in 2022.",
        "$0 — restructuring costs are not explicitly outlined as a separate line item on PepsiCo's Consolidated Statement of Income for FY2022.",
        False,
    ),
    (
        "wrong_numeric_value",  # the AMCR quick-ratio miss
        "How has Amcor's quick ratio changed from FY2022 to FY2023?",
        "The quick ratio has slightly improved from 0.67 times to 0.69 times.",
        "",
        "Amcor's quick ratio improved from 0.63 in FY2022 to 0.69 in FY2023.",
        False,
    ),
    (
        "missing_required_items",  # the AXP geography miss
        "What geographies does American Express operate in?",
        "United States, EMEA, APAC, and LACC.",
        "The Company's geographic operating segments are the United States, EMEA, APAC and LACC.",
        "American Express operates primarily in the United States, served by its U.S. Consumer Services and Commercial Services segments.",
        False,
    ),
    (
        "direction_flip",
        "Did AMD's data center segment revenue rise or fall in FY2022?",
        "It rose, increasing roughly 64% year over year.",
        "Data Center net revenue increased 64% to $6.0 billion.",
        "AMD's Data Center segment revenue declined by about 64% in FY2022.",
        False,
    ),
    (
        "cant_find_no_answer",
        "What was Boeing's FY2022 effective tax rate?",
        "Effective tax rate in FY2022 was 0.62%.",
        "The effective tax rate was 0.6% in 2022.",
        "I was unable to locate the effective tax rate in the retrieved passages.",
        False,
    ),
    (
        "unit_equivalent",
        "What was PepsiCo's FY2022 unadjusted EBITDA less capex?",
        "$9068.00",
        "Operating profit $11,512M + D&A $2,763M - capex $5,207M.",
        "Unadjusted EBITDA less capex works out to about $9.07 billion for FY2022.",
        True,
    ),
    (
        "within_1pct",
        "What was PepsiCo's FY2022 unadjusted EBITDA margin?",
        "16.5%",
        "EBITDA $14,275M / net revenue $86,392M.",
        "The unadjusted EBITDA margin is approximately 16.4%.",
        True,
    ),
    (
        "off_by_10pct",
        "What was PepsiCo's FY2022 unadjusted EBITDA margin?",
        "16.5%",
        "EBITDA $14,275M / net revenue $86,392M.",
        "The unadjusted EBITDA margin is approximately 18.2%.",
        False,
    ),
    (
        "metric_not_meaningful",
        "What is American Express's gross margin and is it improving?",
        "Performance is not measured through gross margin.",
        "",
        "Gross margin is not a meaningful metric for American Express, a financial-services company with no cost of goods sold; performance is better assessed via net interest income and return on equity.",
        True,
    ),
    (
        "conclusion_yes_vs_no",
        "Is Boeing's business subject to cyclicality?",
        "Yes, Boeing's business is subject to cyclicality due to its exposure to the airline industry.",
        "The airline industry has historically been cyclical.",
        "No, Boeing's business is not meaningfully cyclical; its order backlog smooths demand across the cycle.",
        False,
    ),
    (
        "extra_detail_still_correct",
        "Who are Boeing's primary customers?",
        "Boeing's primary customers are a limited number of commercial airlines and the US government, which accounted for 40% of revenue.",
        "The U.S. government accounted for 40% of Boeing's total revenues in 2022.",
        "Boeing's primary customers fall into two groups: commercial airlines worldwide (via Commercial Airplanes) and the U.S. government (via Defense, Space & Security), with the U.S. government representing ~40% of FY2022 revenue. It also serves leasing companies and foreign governments.",
        True,
    ),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    print(f"Validating judge ({settings.judge_model}) against {len(CASES)} labeled cases\n")
    wrong = []
    for name, question, gold, evidence, answer, expected in CASES:
        try:
            v = judge_answer(
                client,
                question=question,
                gold_answer=gold,
                agent_answer=answer,
                gold_evidence=evidence or None,
            )
            got: bool | None = v.correct
            reason = v.reason
        except JudgeError as exc:
            got = None
            reason = f"JUDGE_ERROR: {exc}"
        ok = got is expected
        if not ok:
            wrong.append(name)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:<28} expected={str(expected):<5} got={got}  {reason[:80]}")

    print("\n" + "=" * 64)
    if wrong:
        print(f"JUDGE VALIDATION FAILED: {len(wrong)}/{len(CASES)} graded wrong: {wrong}")
        raise SystemExit(1)
    print(f"JUDGE VALIDATION PASSED: {len(CASES)}/{len(CASES)} cases graded as expected.")


if __name__ == "__main__":
    main()
