"""FinancialModelingPrep numeric backend for the agent's `get_financial_fact` tool.

The agent answers *qualitative* questions from retrieved 10-K text, but precise
*numbers* (revenue, R&D spend, ...) are better pulled from a structured financial
API than parsed out of prose — the filing's text often rounds or narrates figures.
This module is that numeric backend: a single income-statement lookup, with a
hard "no silent fallbacks" stance (see PROJECT.md) — every failure raises.

We hit FMP's `/stable/income-statement` endpoint (the legacy `/api/v3/` path is
dead with a 403, and `limit > 5` trips a 402 quota error). FMP's free tier is
capped at ~250 requests/day, so we cache the matched annual row per
(ticker, year) in-process: all concepts for a given year are served from ONE
network fetch.
"""

from __future__ import annotations

import httpx

from sec_filings.config import settings

# Only the concepts Phase 1's eval set / tests actually exercise. Adding more is a
# one-line change: drop the camelCase income-statement field name into this set.
SUPPORTED_CONCEPTS = frozenset({"revenue", "researchAndDevelopmentExpenses"})

# Anthropic ToolParam describing this tool to the model.
FMP_TOOL_SCHEMA: dict = {
    "name": "get_financial_fact",
    "description": (
        "Look up a single annual financial figure for a public company from its "
        "income statement (FinancialModelingPrep). Use this for precise numbers "
        "(e.g. total revenue, R&D expense) instead of estimating from filing text. "
        "Returns the value in absolute US dollars for the requested fiscal year."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'MSFT'.",
            },
            "year": {
                "type": "integer",
                "description": "Fiscal year, e.g. 2023.",
            },
            "concept": {
                "type": "string",
                "description": (
                    "Which income-statement figure to return. Allowed values: "
                    "'revenue' (total revenue), "
                    "'researchAndDevelopmentExpenses' (R&D expense)."
                ),
            },
        },
        "required": ["ticker", "year", "concept"],
    },
}

# In-memory cache: (ticker, year) -> the matched RAW annual row from FMP. Keeping
# the whole row (not a single value) means every concept for that year is served
# from one fetch, respecting the daily request budget. Process-lifetime only —
# deliberately not persisted (financials get restated; a stale on-disk file would
# silently serve wrong numbers).
_ROW_CACHE: dict[tuple[str, int], dict] = {}


def _fetch_row(ticker: str, year: int) -> dict:
    """Fetch (or return cached) the annual income-statement row for (ticker, year).

    Raises RuntimeError on any non-200 response — checked BEFORE `.json()` because
    FMP's error bodies (402 quota, 403 auth) are not valid JSON.
    """
    key = (ticker, year)
    if key in _ROW_CACHE:
        return _ROW_CACHE[key]

    resp = httpx.get(
        f"{settings.fmp_base_url}/income-statement",
        params={
            "symbol": ticker,
            "period": "annual",
            "limit": 5,  # >5 trips a 402 on the free tier.
            "apikey": settings.fmp_api_key,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"FMP income-statement request for {ticker} failed with HTTP "
            f"{resp.status_code}: {resp.text[:200]}"
        )

    rows = resp.json()  # newest fiscal year first.
    # 'fiscalYear' is a STRING in FMP's payload — compare as strings.
    for row in rows:
        if str(row.get("fiscalYear")) == str(year):
            _ROW_CACHE[key] = row
            return row

    available = [str(r.get("fiscalYear")) for r in rows]
    raise ValueError(
        f"No annual income statement for {ticker} fiscal year {year}. "
        f"Available years: {available}."
    )


def get_financial_fact(ticker: str, year: int, concept: str) -> float:
    """Return a single numeric financial fact as a float.

    Raises:
        ValueError: `concept` is not supported, or no row matches `year`.
        RuntimeError: FMP returned a non-200 status.
        KeyError: the matched row is missing `concept` (or it is null).
    """
    if concept not in SUPPORTED_CONCEPTS:
        raise ValueError(
            f"Unsupported concept {concept!r}. Supported: {sorted(SUPPORTED_CONCEPTS)}."
        )

    row = _fetch_row(ticker, year)
    if row.get(concept) is None:
        raise KeyError(
            f"Concept {concept!r} missing from FMP row for {ticker} fiscal year {year}."
        )
    return float(row[concept])


def handle_get_financial_fact(ticker: str, year: int, concept: str) -> dict:
    """Tool dispatch handler: return the structured result the agent consumes."""
    value = get_financial_fact(ticker, year, concept)
    return {"ticker": ticker, "year": year, "concept": concept, "value": value}
