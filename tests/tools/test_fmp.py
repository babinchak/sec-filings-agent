"""Tests for the FMP numeric backend.

All network is mocked: we monkeypatch `sec_filings.tools.fmp.httpx.get` to return
a fake response object exposing `.status_code`, `.json()` and `.text`. No live FMP
call is ever made (the survey already verified live values; quota is precious).

The module caches the matched row per (ticker, year) in a process-level dict, so we
clear `_ROW_CACHE` at the start of every test to keep them independent.
"""

from __future__ import annotations

import pytest

from sec_filings.tools import fmp


class FakeResponse:
    """Minimal stand-in for httpx.Response covering the fields fmp.py touches."""

    def __init__(self, status_code: int, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("error body is not JSON")  # mimic FMP error bodies
        return self._payload


def _msft_2023_row() -> dict:
    """A trimmed real-shape MSFT FY2023 row; `fiscalYear` is a STRING like FMP's."""
    return {
        "symbol": "MSFT",
        "fiscalYear": "2023",
        "revenue": 211915000000,
        "researchAndDevelopmentExpenses": 27195000000,
    }


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Ensure each test starts with an empty row cache."""
    fmp._ROW_CACHE.clear()


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> None:
    monkeypatch.setattr(fmp.httpx, "get", lambda *a, **k: response)


def test_revenue_extracted_as_float(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, FakeResponse(200, [_msft_2023_row()]))
    value = fmp.get_financial_fact("MSFT", 2023, "revenue")
    assert value == 211915000000.0
    assert isinstance(value, float)


def test_handler_returns_structured_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, FakeResponse(200, [_msft_2023_row()]))
    result = fmp.handle_get_financial_fact("MSFT", 2023, "researchAndDevelopmentExpenses")
    assert result == {
        "ticker": "MSFT",
        "year": 2023,
        "concept": "researchAndDevelopmentExpenses",
        "value": 27195000000.0,
    }


def test_non_200_raises_runtimeerror_before_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # payload=None => .json() would raise; the status check must fire first.
    _patch_get(monkeypatch, FakeResponse(402, payload=None, text="Limit Reached"))
    with pytest.raises(RuntimeError):
        fmp.get_financial_fact("MSFT", 2023, "revenue")


def test_unsupported_concept_raises_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    # Validation happens before any fetch, so even a 200 row must not be reached.
    _patch_get(monkeypatch, FakeResponse(200, [_msft_2023_row()]))
    with pytest.raises(ValueError):
        fmp.get_financial_fact("MSFT", 2023, "netIncome")


def test_absent_fiscal_year_raises_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, FakeResponse(200, [_msft_2023_row()]))
    with pytest.raises(ValueError):
        fmp.get_financial_fact("MSFT", 1999, "revenue")


def test_cache_serves_second_concept_from_one_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concepts for the same year must trigger only ONE network call."""
    calls = {"n": 0}

    def counting_get(*_a: object, **_k: object) -> FakeResponse:
        calls["n"] += 1
        return FakeResponse(200, [_msft_2023_row()])

    monkeypatch.setattr(fmp.httpx, "get", counting_get)

    assert fmp.get_financial_fact("MSFT", 2023, "revenue") == 211915000000.0
    assert fmp.get_financial_fact("MSFT", 2023, "researchAndDevelopmentExpenses") == 27195000000.0
    assert calls["n"] == 1
