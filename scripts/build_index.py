"""Build the dense + lexical retrieval indexes for a set of 10-K filings.

This is the single offline step that turns raw filings into a queryable corpus:
for each (ticker, fiscal_year) we ingest the 10-K, chunk it, embed every chunk
with Voyage, upsert the vectors into the shared Chroma collection, and build a
per-accession bm25s lexical index. It then writes the Chroma manifest that
``retrieval/hybrid.py`` reads as the single source of truth for "what's indexed"
(its ``_accession_for`` resolves (ticker, fiscal_year) -> accession from here).

Why a manifest rather than inferring from disk: hybrid retrieval must fail loud
on an un-indexed filing (no silent fallbacks — see PROJECT.md), and the manifest
gives it an explicit, auditable list to check against. The embed model/dim are
recorded too so a later read can detect an index built with a stale embedder.

Run from project root:
    uv run python scripts/build_index.py MSFT 2023
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import voyageai.error as voyage_error

from sec_filings.config import settings
from sec_filings.corpus.chunker import chunk_filing
from sec_filings.corpus.ingest import ingest_filing
from sec_filings.corpus.models import Chunk
from sec_filings.retrieval.embedding import embed_documents
from sec_filings.retrieval.hybrid import hybrid_search
from sec_filings.retrieval.lexical import build_lexical_index
from sec_filings.retrieval.vector_store import VectorStore

_MANIFEST_FILENAME = "chroma_manifest.json"

# Voyage free tier (no payment method) is throttled to 3 RPM / 10K TPM. We size
# each embed request well under both caps and pace requests so the build still
# completes on a free key. These are only used to *schedule* requests around the
# limit — they are not retrieval policy and do not affect what gets indexed.
_FREE_TIER_TPM = 10_000
# The Voyage SDK itself retries a rate-limited request up to ~3 times, re-sending
# the WHOLE batch each time — so a batch's tokens count against the 10K/min window
# multiple times. We therefore keep each request small enough that several
# re-sends still fit under the cap (2.5K * 3 = 7.5K < 10K).
_TOKENS_PER_REQUEST = 2_500
_SECONDS_PER_REQUEST = 22.0  # 3 RPM => one request / 20s; +2s of slack/jitter
_MAX_RATELIMIT_RETRIES = 8


def _token_batched(chunks: list[Chunk], budget: int) -> list[list[Chunk]]:
    """Greedily pack chunks into batches whose summed token_count <= budget.

    Voyage's free-tier TPM cap is per-request-window, so we keep each request's
    token total under budget. A single chunk that alone exceeds budget still gets
    its own batch (we never drop it — embedding will just count it against the
    window); chunks are capped at 500 tokens here so this never triggers.
    """
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    running = 0
    for chunk in chunks:
        if current and running + chunk.token_count > budget:
            batches.append(current)
            current = []
            running = 0
        current.append(chunk)
        running += chunk.token_count
    if current:
        batches.append(current)
    return batches


def _embed_chunks_rate_limited(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunk texts, pacing/retrying around the Voyage free-tier limits.

    Wraps ``embed_documents`` per token-budgeted batch with a fixed inter-request
    delay (RPM) and exponential backoff on ``RateLimitError`` (TPM bursts). Order
    is preserved 1:1 with ``chunks``. A non-rate-limit error still propagates
    immediately — no silent fallback (see PROJECT.md).
    """
    batches = _token_batched(chunks, _TOKENS_PER_REQUEST)
    vectors: list[list[float]] = []
    for i, batch in enumerate(batches):
        texts = [c.text for c in batch]
        delay = _SECONDS_PER_REQUEST
        for attempt in range(_MAX_RATELIMIT_RETRIES):
            try:
                # batch_size large => one underlying request per token-budgeted batch.
                vectors.extend(embed_documents(texts, batch_size=len(texts)))
                break
            except voyage_error.RateLimitError:
                if attempt == _MAX_RATELIMIT_RETRIES - 1:
                    raise
                print(
                    f"    rate-limited on batch {i + 1}/{len(batches)}; "
                    f"backing off {delay:.0f}s (attempt {attempt + 1})..."
                )
                time.sleep(delay)
                delay *= 2
        print(f"    embedded batch {i + 1}/{len(batches)} ({len(texts)} chunks)")
        if i + 1 < len(batches):
            time.sleep(_SECONDS_PER_REQUEST)  # respect RPM before the next request
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedded {len(vectors)} vectors for {len(chunks)} chunks (batching bug)."
        )
    return vectors


def build_index(targets: list[tuple[str, int]]) -> dict:
    """Ingest, chunk, embed and index each (ticker, fiscal_year); write manifest.

    Returns the manifest dict (also persisted to ``data_dir/chroma_manifest.json``).
    Raises on any failure (missing filing, dim mismatch, empty chunks) — partial
    indexes must surface loudly rather than corrupt retrieval silently.
    """
    if not targets:
        raise ValueError("No build targets given.")

    vs = VectorStore.open()

    entries: list[dict] = []
    total_chunks = 0
    for ticker, fiscal_year in targets:
        print(f"[{ticker} {fiscal_year}] ingesting 10-K...")
        filing = ingest_filing(ticker, fiscal_year)

        print(f"[{ticker} {fiscal_year}] chunking {len(filing.sections)} sections...")
        chunks = chunk_filing(filing)
        if not chunks:
            raise RuntimeError(
                f"{ticker} {fiscal_year} ({filing.accession_number}) produced zero chunks."
            )

        print(f"[{ticker} {fiscal_year}] embedding {len(chunks)} chunks with {settings.embed_model}...")
        vecs = _embed_chunks_rate_limited(chunks)

        print(f"[{ticker} {fiscal_year}] upserting into Chroma collection {settings.chroma_collection!r}...")
        vs.upsert_chunks(chunks, vecs)

        bm25_subdir = settings.bm25_dir / filing.accession_number
        print(f"[{ticker} {fiscal_year}] building lexical index at {bm25_subdir}...")
        build_lexical_index(chunks, bm25_subdir)

        entries.append(
            {
                "ticker": filing.ticker,
                "fiscal_year": fiscal_year,
                "accession": filing.accession_number,
                "chunk_count": len(chunks),
                # Relative subdir name (the accession) so the manifest stays
                # portable; callers join it under settings.bm25_dir themselves.
                "bm25_subdir": filing.accession_number,
            }
        )
        total_chunks += len(chunks)
        print(f"[{ticker} {fiscal_year}] done: {len(chunks)} chunks, accession {filing.accession_number}")

    manifest = {
        "embed_model": settings.embed_model,
        "embed_dim": settings.embed_dim,
        "collection": settings.chroma_collection,
        "total_chunks": total_chunks,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "filings": entries,
    }

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.data_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")

    return manifest


def _verify(manifest: dict) -> int:
    """Post-build sanity check: count matches manifest and hybrid search works.

    Reopens the store (a fresh handle, not the build-time one) so we're checking
    persisted state, asserts the chunk count, then runs a real hybrid query to
    prove both the dense and lexical indexes are wired through the manifest.
    Returns the number of hits from the verify query. Raises on any mismatch.
    """
    vs = VectorStore.open()
    count = vs.count()
    total = manifest["total_chunks"]
    if count != total:
        raise RuntimeError(
            f"Post-build count mismatch: Chroma holds {count} chunks but the "
            f"manifest claims {total}."
        )
    print(f"Verify: Chroma count {count} == manifest total_chunks {total}.")

    hits = hybrid_search("revenue", ticker="MSFT", fiscal_year=2023, top_k=3)
    if not hits or not hits[0].chunk.text:
        raise RuntimeError(
            "Verify query 'revenue' (MSFT 2023) returned no usable hits; "
            "the index is built but not retrievable."
        )
    print(f"Verify: hybrid_search('revenue', MSFT 2023) -> {len(hits)} hits.")
    top = hits[0]
    print(
        f"  top hit: {top.chunk.chunk_id} (item {top.chunk.item}, "
        f"dense_rank={top.dense_rank}, lexical_rank={top.lexical_rank})"
    )
    print(f"  text head: {top.chunk.text[:160]!r}")
    return len(hits)


def main(argv: list[str]) -> None:
    """CLI entry point.

    Usage: python scripts/build_index.py [TICKER FISCAL_YEAR]
    Defaults to [('MSFT', 2023)] when no args are given.
    """
    args = argv[1:]
    if not args:
        targets = [("MSFT", 2023)]
    elif len(args) == 2:
        targets = [(args[0].upper(), int(args[1]))]
    else:
        raise SystemExit(
            "Usage: python scripts/build_index.py [<TICKER> <FISCAL_YEAR>]"
        )

    manifest = build_index(targets)

    print()
    print("=== Manifest summary ===")
    print(
        f"embed_model={manifest['embed_model']} embed_dim={manifest['embed_dim']} "
        f"collection={manifest['collection']} total_chunks={manifest['total_chunks']}"
    )
    for entry in manifest["filings"]:
        print(
            f"  {entry['ticker']} FY{entry['fiscal_year']}: "
            f"{entry['chunk_count']} chunks, accession {entry['accession']}"
        )
    print()

    hit_count = _verify(manifest)
    print()
    print(f"BUILD OK: {manifest['total_chunks']} chunks indexed, verify hits={hit_count}.")


if __name__ == "__main__":
    main(sys.argv)
