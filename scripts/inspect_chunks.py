"""Phase 1 smoke test: ingest one 10-K, chunk it, print chunk stats.

Run from project root:
    uv run python scripts/inspect_chunks.py MSFT 2023
"""

from __future__ import annotations

import statistics
import sys

from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.ingest import ingest_filing


def main(ticker: str, fiscal_year: int) -> None:
    print(f"Ingesting {ticker} 10-K for fiscal year {fiscal_year}...")
    filing = ingest_filing(ticker, fiscal_year)
    print(f"Chunking {len(filing.sections)} sections...")
    chunks = chunk_filing(filing)

    counts = [c.token_count for c in chunks]
    print()
    print(f"Total chunks:      {len(chunks)}")
    print(f"Token count min/median/max: "
          f"{min(counts)} / {int(statistics.median(counts))} / {max(counts)}")
    print()

    # Chunks per section.
    per_item: dict[str, int] = {}
    for c in chunks:
        per_item[c.item] = per_item.get(c.item, 0) + 1
    print(f"{'Item':<10} {'Chunks':>7}  Title")
    print("-" * 70)
    for section in filing.sections:
        title = section.title
        print(f"{section.item:<10} {per_item.get(section.item, 0):>7}  {title}")
    print("-" * 70)

    # Spot-check the first chunk: id, offsets, verbatim-slice integrity.
    first = chunks[0]
    assert first.text == filing.sections[0].text[first.char_start : first.char_end]
    print()
    print("First chunk:")
    print(f"  id:         {first.chunk_id}")
    print(f"  item:       {first.item}")
    print(f"  chars:      [{first.char_start}, {first.char_end}]  tokens: {first.token_count}")
    print(f"  text head:  {first.text[:160]!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/inspect_chunks.py <TICKER> <FISCAL_YEAR>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
