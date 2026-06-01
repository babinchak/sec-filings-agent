"""Build the FinanceBench starter corpus: the five richest 10-K filings.

These are the five 10-Ks with the most FinanceBench questions (30 between them),
picked to give the eval harness a real multi-company, multi-sector body to grade
against while still indexing in minutes. ``build_index`` merges them into the
existing Chroma collection + manifest, so anything already indexed (e.g. MSFT
2023) is preserved without re-embedding.

Run from project root:
    uv run python scripts/build_starter_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ holds build_index.py as a sibling, not an installed package; put this
# directory on the path so the import resolves however the script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_index import build_index  # noqa: E402

from sec_filings.retrieval.hybrid import hybrid_search  # noqa: E402

# (ticker, fiscal_year) for each starter filing. Fiscal years match FinanceBench's
# doc_name year; ingest resolves each to the right EDGAR accession by period.
STARTER_TARGETS: list[tuple[str, int]] = [
    ("AMD", 2022),
    ("AXP", 2022),
    ("BA", 2022),
    ("PEP", 2022),
    ("AMCR", 2023),
]


def main() -> None:
    manifest = build_index(STARTER_TARGETS)

    print("\n=== Manifest after starter build ===")
    for entry in manifest["filings"]:
        print(
            f"  {entry['ticker']} FY{entry['fiscal_year']}: "
            f"{entry['chunk_count']} chunks, accession {entry['accession']}"
        )
    print(f"  total_chunks={manifest['total_chunks']}\n")

    print("=== Per-filing retrieval check ===")
    for ticker, fiscal_year in STARTER_TARGETS:
        hits = hybrid_search("revenue", ticker=ticker, fiscal_year=fiscal_year, top_k=3)
        if not hits or not hits[0].chunk.text:
            raise RuntimeError(
                f"{ticker} {fiscal_year} indexed but 'revenue' returned no usable "
                "hits; the index is built but not retrievable."
            )
        print(
            f"  {ticker} {fiscal_year}: 'revenue' -> {len(hits)} hits, "
            f"top item {hits[0].chunk.item}"
        )

    print("\nSTARTER BUILD OK.")


if __name__ == "__main__":
    main()
