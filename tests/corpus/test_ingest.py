"""Tests for ingest's section coalescing — the AXP duplicate-Item-8 fix.

edgartools can return one Item as several section objects (American Express's
Item 8 splits into the auditor's report and the financial statements). The
chunker keys ids on (item, in-section offset), so without coalescing the two
parts both emit a chunk at offset 0 and collide — which crashed the index build
with chromadb DuplicateIDError. These tests pin the fix with no network/edgar.
"""

from __future__ import annotations

from datetime import date

from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.ingest import _coalesce_sections
from sec_filings.corpus.models import Filing


def _ws_counter(text: str) -> int:
    """Whitespace token counter so the chunker needs neither tiktoken nor net."""
    return len(text.split())


def test_coalesces_split_item_into_one_section_in_document_order():
    raw = [
        ("1", "Business overview. We make chips."),
        ("8", "Auditor report part. Opinion follows."),
        ("8", "Consolidated statements part. Revenue grew."),
    ]
    sections = _coalesce_sections(raw)

    assert [s.item for s in sections] == ["Item 1", "Item 8"]  # one Item 8
    item8 = next(s for s in sections if s.item == "Item 8")
    assert "Auditor report part" in item8.text
    assert "Consolidated statements part" in item8.text
    # Concatenated in the order seen (auditor report precedes statements).
    assert item8.text.index("Auditor") < item8.text.index("Consolidated")
    # Canonical title is applied, not whatever a single part carried.
    assert item8.title == "Financial Statements and Supplementary Data"


def test_chunk_ids_unique_after_coalescing_split_item():
    # Regression for the AXP DuplicateIDError: two Item 8 parts must not produce
    # two chunks with id ...::item8::000000.
    raw = [("8", "First part sentence. " * 60), ("8", "Second part sentence. " * 60)]
    sections = _coalesce_sections(raw)
    assert len(sections) == 1  # merged, not two Item 8 sections

    filing = Filing(
        ticker="AXP",
        cik="4962",
        accession_number="0000004962-23-000006",
        company_name="AMERICAN EXPRESS CO",
        filing_date=date(2023, 2, 10),
        fiscal_year_end=date(2022, 12, 31),
        sections=sections,
    )
    chunk_ids = [c.chunk_id for c in chunk_filing(filing, count_tokens=_ws_counter)]

    assert len(chunk_ids) == len(set(chunk_ids)), "chunk ids must be unique per filing"


def test_drops_empty_and_unlabeled_sections():
    raw = [(None, "orphan text with no item"), ("2", "   "), ("3", "Real properties.")]
    sections = _coalesce_sections(raw)
    assert [s.item for s in sections] == ["Item 3"]
