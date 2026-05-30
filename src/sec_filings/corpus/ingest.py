"""Ingest a single 10-K filing into our `Filing` model via edgartools.

Higher-level callers should use `ingest_filing(ticker, fiscal_year)`. Raises on
any failure — see PROJECT.md (no silent fallbacks).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from edgar import Company, set_identity  # type: ignore[import-untyped]

from sec_filings.config import settings
from sec_filings.corpus.models import Filing, Section

if TYPE_CHECKING:
    from edgar import Filing as EdgarFiling  # type: ignore[import-untyped]


ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


def _item_sort_key(item_label: str) -> tuple[int, str]:
    """Sort key so 'Item 1', 'Item 1A', 'Item 2', ..., 'Item 10' order correctly."""
    suffix = item_label.removeprefix("Item ").strip()
    digits = "".join(c for c in suffix if c.isdigit())
    letters = "".join(c for c in suffix if c.isalpha())
    return (int(digits) if digits else 0, letters)


def _ensure_identity() -> None:
    if not settings.edgar_identity:
        raise RuntimeError(
            "EDGAR_IDENTITY is not set. SEC requires identifying your traffic. "
            "Set EDGAR_IDENTITY='Your Name <you@example.com>' in .env."
        )
    set_identity(settings.edgar_identity)


def _pick_10k_for_year(company: Company, fiscal_year: int) -> "EdgarFiling":
    """Find the 10-K whose period_of_report falls in `fiscal_year`.

    A 10-K for fiscal year N is typically filed in calendar year N or N+1, so we
    search both and filter by period_of_report.
    """
    candidates = company.get_filings(form="10-K", year=[fiscal_year, fiscal_year + 1])
    for filing in candidates:
        period = filing.period_of_report
        if period and str(period).startswith(str(fiscal_year)):
            return filing
    raise LookupError(
        f"No 10-K found for {company.ticker} with period_of_report in fiscal year {fiscal_year}."
    )


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def ingest_filing(ticker: str, fiscal_year: int) -> Filing:
    """Fetch a 10-K from EDGAR and return our `Filing` model."""
    _ensure_identity()
    company = Company(ticker)
    edgar_filing = _pick_10k_for_year(company, fiscal_year)
    tenk = edgar_filing.obj()
    if tenk is None:
        raise RuntimeError(f"edgartools could not parse 10-K {edgar_filing.accession_number}.")

    sections: list[Section] = []
    for _key, section_obj in (tenk.sections or {}).items():
        item_suffix = getattr(section_obj, "item", None)
        if not item_suffix:
            continue
        item_suffix = str(item_suffix).upper()
        item_label = f"Item {item_suffix}"
        title = ITEM_TITLES.get(item_suffix, item_label)
        text_attr = getattr(section_obj, "text", None)
        text = text_attr() if callable(text_attr) else str(section_obj)
        if not text.strip():
            continue
        sections.append(Section(item=item_label, title=title, text=text))

    # Stable order: by canonical item label.
    sections.sort(key=lambda s: _item_sort_key(s.item))

    if not sections:
        raise RuntimeError(
            f"Parsed 10-K {edgar_filing.accession_number} but found zero usable sections."
        )

    return Filing(
        ticker=ticker.upper(),
        cik=str(company.cik),
        accession_number=edgar_filing.accession_number,
        company_name=company.name,
        form_type=edgar_filing.form,
        filing_date=_parse_date(edgar_filing.filing_date),
        fiscal_year_end=_parse_date(edgar_filing.period_of_report),
        sections=sections,
    )
