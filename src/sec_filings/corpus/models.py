"""Data shapes for ingested filings and chunks."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A top-level Item-section of a 10-K (e.g. Item 1A Risk Factors)."""

    item: str = Field(description="Canonical item label, e.g. 'Item 1A'.")
    title: str = Field(description="Section title as written in the filing.")
    text: str


class Filing(BaseModel):
    """An ingested 10-K filing."""

    ticker: str
    cik: str
    accession_number: str = Field(description="SEC's unique ID for this filing.")
    company_name: str
    form_type: str = Field(default="10-K")
    filing_date: date
    fiscal_year_end: date
    sections: list[Section]

    @property
    def fiscal_year(self) -> int:
        return self.fiscal_year_end.year


class Chunk(BaseModel):
    """A retrieval unit produced by the chunker."""

    chunk_id: str = Field(description="Deterministic hash of (accession, item, char_start).")
    filing_accession: str
    ticker: str
    fiscal_year: int
    item: str = Field(description="Canonical item label, e.g. 'Item 1A'.")
    section_path: list[str] = Field(
        default_factory=list,
        description="Breadcrumb trail, e.g. ['Item 1A', 'Cybersecurity Risks'].",
    )
    text: str
    char_start: int
    char_end: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
