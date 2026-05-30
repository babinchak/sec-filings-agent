"""Tests for the section-aware chunker.

We inject a whitespace token counter (`len(text.split())`) so these tests need
neither tiktoken nor network, and the token budgets are easy to reason about.
"""

from __future__ import annotations

from datetime import date

import pytest

from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.models import Filing, Section


def words(text: str) -> int:
    """Trivial, deterministic token counter: one token per whitespace word."""
    return len(text.split())


def make_sentences(n: int, words_each: int = 10) -> str:
    """Build `n` sentences, each `words_each` words, ending in a period + space."""
    sentences = []
    for s in range(n):
        body = " ".join(f"w{s}_{w}" for w in range(words_each - 1))
        sentences.append(f"{body} end.")
    return " ".join(sentences)


def make_filing(sections: list[Section]) -> Filing:
    return Filing(
        ticker="TEST",
        cik="0000000000",
        accession_number="0000000000-00-000000",
        company_name="Test Corp",
        form_type="10-K",
        filing_date=date(2024, 1, 1),
        fiscal_year_end=date(2023, 12, 31),
        sections=sections,
    )


def test_empty_section_yields_no_chunks() -> None:
    filing = make_filing([Section(item="Item 1", title="Business", text="")])
    assert chunk_filing(filing, count_tokens=words) == []


def test_short_section_is_single_chunk() -> None:
    text = make_sentences(2, words_each=5)  # ~10 tokens, well under budget
    filing = make_filing([Section(item="Item 1", title="Business", text=text)])
    chunks = chunk_filing(filing, target_tokens=100, overlap_tokens=10, count_tokens=words)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_char_offsets_are_exact_slices() -> None:
    """Every chunk's text must equal the verbatim source slice."""
    text = make_sentences(40, words_each=10)
    section = Section(item="Item 7", title="MD&A", text=text)
    filing = make_filing([section])
    chunks = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]


def test_respects_token_budget() -> None:
    text = make_sentences(40, words_each=10)
    filing = make_filing([Section(item="Item 7", title="MD&A", text=text)])
    chunks = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    for chunk in chunks:
        assert chunk.token_count <= 50


def test_consecutive_chunks_overlap() -> None:
    text = make_sentences(40, words_each=10)
    filing = make_filing([Section(item="Item 7", title="MD&A", text=text)])
    chunks = chunk_filing(filing, target_tokens=50, overlap_tokens=15, count_tokens=words)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        # The next chunk starts before the previous one ends => shared context.
        assert nxt.char_start < prev.char_end


def test_no_cross_section_bleed() -> None:
    s1 = Section(item="Item 1", title="Business", text=make_sentences(30))
    s2 = Section(item="Item 1A", title="Risk Factors", text=make_sentences(30))
    filing = make_filing([s1, s2])
    chunks = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    by_item = {c.item for c in chunks}
    assert by_item == {"Item 1", "Item 1A"}
    # Each chunk's text lives entirely within its own section.
    for chunk in chunks:
        source = s1.text if chunk.item == "Item 1" else s2.text
        assert chunk.text == source[chunk.char_start : chunk.char_end]


def test_chunk_ids_are_unique_and_deterministic() -> None:
    text = make_sentences(40, words_each=10)
    filing = make_filing([Section(item="Item 7", title="MD&A", text=text)])
    first = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    second = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))  # unique
    assert ids == [c.chunk_id for c in second]  # deterministic across runs


def test_section_path_is_breadcrumb() -> None:
    filing = make_filing([Section(item="Item 1A", title="Risk Factors", text=make_sentences(5))])
    chunks = chunk_filing(filing, count_tokens=words)
    assert chunks[0].section_path == ["Item 1A", "Risk Factors"]


def test_oversized_sentence_is_hard_split() -> None:
    """A single sentence over budget must still be broken into in-budget chunks."""
    giant = " ".join(f"token{i}" for i in range(200)) + "."  # 201 tokens, no breaks
    filing = make_filing([Section(item="Item 1", title="Business", text=giant)])
    chunks = chunk_filing(filing, target_tokens=50, overlap_tokens=10, count_tokens=words)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 50
        assert chunk.text == giant[chunk.char_start : chunk.char_end]


def test_overlap_must_be_less_than_target() -> None:
    filing = make_filing([Section(item="Item 1", title="Business", text="a. b. c.")])
    with pytest.raises(ValueError):
        chunk_filing(filing, target_tokens=10, overlap_tokens=10, count_tokens=words)
