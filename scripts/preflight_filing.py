"""Preflight one filing: ingest + chunk (no embedding) and print a JSON report.

Used to de-risk EDGAR parsing and chunking on a filing *before* spending any
Voyage embedding budget on it — the ingest/chunk pipeline had only ever run on
MSFT, so this proves it survives other issuers. Prints a single JSON object to
stdout and raises loudly on any parse failure (no silent fallbacks — PROJECT.md).

Run from project root:
    uv run python scripts/preflight_filing.py AMD 2022
"""

from __future__ import annotations

import json
import sys

from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.ingest import ingest_filing


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit("Usage: python scripts/preflight_filing.py <TICKER> <FISCAL_YEAR>")
    ticker, fiscal_year = argv[1].upper(), int(argv[2])

    filing = ingest_filing(ticker, fiscal_year)
    chunks = chunk_filing(filing)

    items = sorted({c.item for c in chunks})
    report = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "accession": filing.accession_number,
        "company_name": filing.company_name,
        "filing_date": str(filing.filing_date),
        "fiscal_year_end": str(filing.fiscal_year_end),
        "n_sections": len(filing.sections),
        "n_chunks": len(chunks),
        "total_tokens": sum(c.token_count for c in chunks),
        "items_present": items,
        "has_item7_mdna": any(c.item == "Item 7" for c in chunks),
        "has_item8_financials": any(c.item == "Item 8" for c in chunks),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv)
