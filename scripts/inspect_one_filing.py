"""Phase 0 smoke test: ingest one 10-K and print its structure.

Run from project root:
    uv run python scripts/inspect_one_filing.py MSFT 2023
"""

from __future__ import annotations

import sys

from sec_filings.corpus.ingest import ingest_filing


def main(ticker: str, fiscal_year: int) -> None:
    print(f"Ingesting {ticker} 10-K for fiscal year {fiscal_year}...")
    filing = ingest_filing(ticker, fiscal_year)

    print()
    print(f"Company:           {filing.company_name}")
    print(f"CIK:               {filing.cik}")
    print(f"Accession number:  {filing.accession_number}")
    print(f"Form type:         {filing.form_type}")
    print(f"Filing date:       {filing.filing_date}")
    print(f"Fiscal year end:   {filing.fiscal_year_end}")
    print(f"Section count:     {len(filing.sections)}")
    print()
    print(f"{'Item':<10} {'Chars':>10}  Title")
    print("-" * 80)
    total_chars = 0
    for section in filing.sections:
        n = len(section.text)
        total_chars += n
        print(f"{section.item:<10} {n:>10,}  {section.title}")
    print("-" * 80)
    print(f"{'TOTAL':<10} {total_chars:>10,}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/inspect_one_filing.py <TICKER> <FISCAL_YEAR>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
