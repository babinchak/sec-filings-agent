"""Section-aware chunking of ingested 10-K filings.

This is hand-written on purpose (see PROJECT.md) — chunking is a load-bearing
retrieval decision, and the strategy here is a portfolio talking point.

Design:
  * **Within-section only.** A chunk never spans two Items. Item 1A (Risk
    Factors) and Item 7 (MD&A) are semantically different; merging them across a
    boundary would pollute retrieval. Section structure is real signal.
  * **Sentence-granular, greedy packing.** We split each section into sentences
    (with exact character spans into the original text), then greedily pack
    sentences into a chunk until the token budget is hit.
  * **Token-budgeted with overlap.** Target ~`target_tokens` per chunk with
    ~`overlap_tokens` of trailing context carried into the next chunk, so a fact
    that straddles a boundary still appears whole in one chunk.
  * **Exact char offsets.** Every chunk's `text` is a verbatim slice of the
    source section text — `section.text[char_start:char_end]`. This keeps chunks
    traceable back to the filing (useful for the inspector UI and for debugging
    retrieval).
  * **Deterministic ids.** `chunk_id` is a pure function of
    (accession, item, char_start), so re-chunking the same corpus yields the
    same ids and eval gold labels don't drift.

The token counter is injectable. The default uses tiktoken's ``cl100k_base``
(close enough to most embedders for *sizing* purposes; we are nowhere near the
embedder's context limit, so exact tokenizer parity doesn't matter here). Tests
inject a trivial whitespace counter so they need neither tiktoken nor network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from sec_filings.corpus.models import Chunk, Filing, Section

TokenCounter = Callable[[str], int]

DEFAULT_TARGET_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

# A sentence ends at . ! or ? (optionally followed by a closing quote/bracket),
# then whitespace. Imperfect for abbreviations ("Inc.", "U.S.") but boundaries
# only affect where chunks split; overlap covers the seams. Spans are exact.
_SENTENCE_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s+")


@lru_cache(maxsize=1)
def _default_token_counter() -> TokenCounter:
    """tiktoken cl100k_base counter, loaded lazily and cached.

    Raised eagerly with a clear message if tiktoken is missing — no silent
    fallback to a heuristic (eval integrity depends on consistent counts).
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "tiktoken is required for the default token counter. Install it "
            "(`uv sync`) or pass an explicit `count_tokens` callable."
        ) from exc

    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into (start, end) char spans, one per sentence.

    Spans are contiguous and cover the whole string; each span includes the
    trailing whitespace up to the next sentence so that re-slicing and joining
    reproduces the original exactly.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _split_oversized_span(
    text: str,
    span: tuple[int, int],
    target_tokens: int,
    count_tokens: TokenCounter,
) -> list[tuple[int, int]]:
    """Hard-split a single span that alone exceeds the token budget.

    Splits on whitespace boundaries into char windows sized to roughly the token
    budget. Rare in practice (a single >target-token sentence), but we refuse to
    silently emit a chunk that blows the budget.
    """
    start, end = span
    approx_chars = max(1, target_tokens * 4)  # ~4 chars/token heuristic for windowing
    sub_spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        window_end = min(end, cursor + approx_chars)
        # Back off to the last whitespace so we don't split mid-word.
        if window_end < end:
            whitespace = text.rfind(" ", cursor + 1, window_end)
            if whitespace > cursor:
                window_end = whitespace + 1
        # Shrink further while still over budget (e.g. no whitespace found).
        while (
            window_end - cursor > 1
            and count_tokens(text[cursor:window_end]) > target_tokens
        ):
            window_end = cursor + (window_end - cursor) // 2
        sub_spans.append((cursor, window_end))
        cursor = window_end
    return sub_spans


def _chunk_id(accession: str, item: str, char_start: int) -> str:
    """Deterministic, readable id from (accession, item, char_start)."""
    item_slug = item.replace(" ", "").lower()  # "Item 1A" -> "item1a"
    return f"{accession}::{item_slug}::{char_start:06d}"


def _chunk_section(
    filing: Filing,
    section: Section,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    text = section.text
    raw_spans = _sentence_spans(text)

    # Pre-split any single span that exceeds the budget on its own.
    spans: list[tuple[int, int]] = []
    for span in raw_spans:
        if count_tokens(text[span[0] : span[1]]) > target_tokens:
            spans.extend(_split_oversized_span(text, span, target_tokens, count_tokens))
        else:
            spans.append(span)

    if not spans:
        return []

    span_tokens = [count_tokens(text[s:e]) for s, e in spans]
    section_path = [section.item, section.title]

    chunks: list[Chunk] = []
    i = 0
    n = len(spans)
    while i < n:
        # Greedily extend the chunk from span i while we stay within budget.
        # Always take at least one span so we make progress.
        j = i
        running = 0
        while j < n and (j == i or running + span_tokens[j] <= target_tokens):
            running += span_tokens[j]
            j += 1

        char_start = spans[i][0]
        char_end = spans[j - 1][1]
        chunk_text = text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(filing.accession_number, section.item, char_start),
                filing_accession=filing.accession_number,
                ticker=filing.ticker,
                fiscal_year=filing.fiscal_year,
                item=section.item,
                section_path=section_path,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                token_count=running,
            )
        )

        if j >= n:
            break

        # Back up from j to carry ~overlap_tokens of trailing context into the
        # next chunk. `k > i + 1` guarantees forward progress (next start > i).
        overlap = 0
        k = j
        while k > i + 1 and overlap < overlap_tokens:
            k -= 1
            overlap += span_tokens[k]
        i = k

    return chunks


def chunk_filing(
    filing: Filing,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk every section of `filing` into retrieval units.

    Chunks never cross section boundaries. Returns them in document order
    (section order, then position within section).
    """
    if overlap_tokens >= target_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < target_tokens ({target_tokens})."
        )
    counter = count_tokens or _default_token_counter()

    chunks: list[Chunk] = []
    for section in filing.sections:
        chunks.extend(
            _chunk_section(filing, section, target_tokens, overlap_tokens, counter)
        )
    return chunks


def chunk_filings(
    filings: list[Filing],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk a list of filings (convenience over `chunk_filing`)."""
    counter = count_tokens or _default_token_counter()
    chunks: list[Chunk] = []
    for filing in filings:
        chunks.extend(
            chunk_filing(
                filing,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                count_tokens=counter,
            )
        )
    return chunks
