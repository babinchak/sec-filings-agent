"""Hybrid retrieval: Reciprocal Rank Fusion over dense + lexical rankers.

Dense (Chroma/Voyage) and lexical (bm25s) retrieval see different signal. Dense
vectors capture paraphrase and semantic similarity but blur exact terms; BM25
nails exact line-item names, tickers, and section identifiers but misses
synonyms. Fusing the two recovers the union of their strengths.

We fuse with **Reciprocal Rank Fusion** rather than score-blending because the
two rankers produce incomparable scores (cosine distance in [0, 2] vs. an
unbounded BM25 score). RRF throws the magnitudes away and keeps only *rank*,
which is the one thing both rankers agree on the meaning of. A document's fused
score is the sum over rankers of ``1 / (rrf_k + rank)`` (0-based rank); a
document that ranks well in *both* lists beats one that ranks well in only one.

``rrf_k`` damps the contribution of low ranks (the classic value is 60) so that
the very top results dominate but tail results still nudge the fusion. We
hard-code ``RRF_K`` and the per-ranker candidate depth ``CANDIDATE_K`` here:
they are fixed retrieval policy for Phase 1, not knobs the caller should turn.
Tuning them is a Phase-5 ablation concern, deliberately kept out of the public
surface so callers cannot accidentally couple to a not-yet-validated value.

No silent fallbacks (see PROJECT.md): an unknown (ticker, fiscal_year) with no
manifest entry RAISES rather than returning empty results.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from sec_filings.config import settings
from sec_filings.corpus.models import Chunk
from sec_filings.retrieval.embedding import embed_query
from sec_filings.retrieval.lexical import LexicalIndex
from sec_filings.retrieval.vector_store import VectorStore

# Per-ranker retrieval depth: how many candidates each of the dense and lexical
# rankers contributes to the fusion pool. Larger than top_k so a document the
# fusion will surface can still enter via its strong ranker even if its other
# ranker buried it. Fixed policy — not a public parameter (Phase-5 ablation).
CANDIDATE_K = 30

# RRF damping constant. The literature's default; raising it flattens the
# rank-weighting, lowering it sharpens the top's dominance. Fixed policy.
RRF_K = 60

_MANIFEST_FILENAME = "chroma_manifest.json"


def reciprocal_rank_fusion(
    dense_ids: list[str],
    lexical_ids: list[str],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float, int | None, int | None]]:
    """Fuse two ranked id-lists into one, by Reciprocal Rank Fusion.

    Each input is an *ordered* list of chunk ids (best first). A document's
    fused score is the sum over the rankers it appears in of ``1 / (rrf_k +
    rank)`` where ``rank`` is the document's 0-based position in that ranker's
    list. Returns ``(chunk_id, fused_score, dense_rank, lexical_rank)`` tuples
    sorted by fused score descending; ``dense_rank``/``lexical_rank`` are
    ``None`` when the id did not appear in that ranker.

    Pure and network-free: it operates only on the id-lists handed to it, which
    is what makes the fusion logic unit-testable in isolation.

    Ties (e.g. a document appearing at the same rank in both, or two documents
    with identical fused scores) are broken deterministically by chunk_id so the
    output ordering is stable across runs.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}.")

    dense_rank = {cid: rank for rank, cid in enumerate(dense_ids)}
    lexical_rank = {cid: rank for rank, cid in enumerate(lexical_ids)}

    scores: dict[str, float] = {}
    for cid, rank in dense_rank.items():
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for cid, rank in lexical_rank.items():
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    fused = [
        (cid, score, dense_rank.get(cid), lexical_rank.get(cid))
        for cid, score in scores.items()
    ]
    # Sort by score desc, then chunk_id asc for a stable, deterministic order.
    fused.sort(key=lambda row: (-row[1], row[0]))
    return fused


class ScoredChunk(BaseModel):
    """A hydrated chunk with its fusion provenance.

    Carries the per-ranker ranks (``None`` if absent from that ranker) alongside
    the fused score so the inspector UI and eval can explain *why* a chunk
    surfaced — dense-only, lexical-only, or agreed-on-by-both.
    """

    chunk: Chunk
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None


class HybridRetriever:
    """Dense + lexical retrieval over a single filing, fused with RRF.

    Bound to one filing (one accession) at ``open`` time: the dense store is
    additionally filtered by (ticker, fiscal_year) in :meth:`search`, and the
    lexical index is the per-accession bm25s index. Construct via :meth:`open`.
    """

    def __init__(self, vector_store: VectorStore, lexical_index: LexicalIndex) -> None:
        self._vector_store = vector_store
        self._lexical_index = lexical_index

    @classmethod
    def open(
        cls,
        *,
        accession: str,
        chroma_dir: Path = settings.chroma_dir,
        bm25_root: Path = settings.bm25_dir,
    ) -> "HybridRetriever":
        """Wire a VectorStore and the per-accession LexicalIndex.

        The vector store is the shared multi-filing Chroma collection (queries
        are filtered down to one filing in :meth:`search`); the lexical index is
        the accession-specific bm25s directory under ``bm25_root``.
        """
        vector_store = VectorStore.open(persist_dir=chroma_dir)
        lexical_index = LexicalIndex.load(bm25_root / accession)
        return cls(vector_store, lexical_index)

    def search(
        self,
        query: str,
        *,
        ticker: str,
        fiscal_year: int,
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        """Retrieve the top-``top_k`` chunks for ``query``, fused across rankers.

        Runs both rankers to ``CANDIDATE_K`` depth, fuses the two ordered
        id-lists with RRF, hydrates the top ``top_k`` fused ids back into full
        Chunks, and returns them in fused order with their rank provenance.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        qvec = embed_query(query)
        dense = self._vector_store.query(
            qvec, ticker=ticker, fiscal_year=fiscal_year, n_results=CANDIDATE_K
        )
        lexical = self._lexical_index.search(query, k=CANDIDATE_K)

        dense_ids = [cid for cid, _distance in dense]
        lexical_ids = [cid for cid, _score in lexical]

        fused = reciprocal_rank_fusion(dense_ids, lexical_ids, rrf_k=RRF_K)
        top = fused[:top_k]

        top_ids = [cid for cid, _s, _d, _l in top]
        chunks = self._vector_store.get_chunks(top_ids)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}

        scored: list[ScoredChunk] = []
        for cid, fused_score, dense_rank, lexical_rank in top:
            chunk = by_id.get(cid)
            if chunk is None:
                # A fused id with no row in Chroma means the dense and lexical
                # indexes have drifted out of sync — never silently drop it.
                raise RuntimeError(
                    f"Fused chunk_id {cid!r} has no row in the vector store; "
                    "the dense and lexical indexes are out of sync."
                )
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    fused_score=fused_score,
                    dense_rank=dense_rank,
                    lexical_rank=lexical_rank,
                )
            )
        return scored


# Cache of opened retrievers keyed by accession. Opening a retriever loads the
# bm25s index from disk and constructs a Chroma client, so we keep one per
# filing for the process lifetime rather than re-opening on every call.
_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}


def _accession_for(ticker: str, fiscal_year: int) -> str:
    """Resolve a filing's accession from the Chroma manifest.

    Reads ``data_dir/chroma_manifest.json`` (written by the build scripts as the
    single source of truth for "what's indexed") and finds the entry matching
    ``ticker`` + ``fiscal_year``. Raises rather than guessing if the manifest is
    missing or has no matching/ambiguous entry — no silent fallbacks.
    """
    manifest_path = settings.data_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path!s}; build the index before retrieving "
            "(scripts/build_index.py writes the manifest)."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The manifest is a list of filing entries; each carries at least ticker,
    # fiscal_year and accession. (Build scripts own the full schema.)
    entries = manifest["filings"] if isinstance(manifest, dict) else manifest

    ticker_upper = ticker.upper()
    matches = [
        entry["accession"]
        for entry in entries
        if str(entry["ticker"]).upper() == ticker_upper
        and int(entry["fiscal_year"]) == fiscal_year
    ]
    if not matches:
        available = sorted(
            f"{str(entry['ticker']).upper()} FY{entry['fiscal_year']}"
            for entry in entries
        )
        raise LookupError(
            f"No indexed filing for {ticker_upper} fiscal year {fiscal_year}. "
            f"Available indexed filings: {', '.join(available) or '(none)'}. "
            "Retry with one of the available fiscal years."
        )
    if len(set(matches)) > 1:
        raise LookupError(
            f"Ambiguous manifest: multiple accessions for {ticker_upper} "
            f"fiscal year {fiscal_year}: {sorted(set(matches))}."
        )
    return matches[0]


def hybrid_search(
    query: str,
    *,
    ticker: str,
    fiscal_year: int,
    top_k: int = 8,
) -> list[ScoredChunk]:
    """Hybrid retrieve over the filing identified by (ticker, fiscal_year).

    Resolves the filing's accession via the Chroma manifest, lazily opens (and
    caches) a :class:`HybridRetriever` for it, and runs the search. This is the
    module-level entry point the retrieval tool calls; the per-filing
    :class:`HybridRetriever` is the lower-level surface for callers that already
    know the accession.
    """
    accession = _accession_for(ticker, fiscal_year)
    retriever = _RETRIEVER_CACHE.get(accession)
    if retriever is None:
        retriever = HybridRetriever.open(accession=accession)
        _RETRIEVER_CACHE[accession] = retriever
    return retriever.search(
        query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k
    )
