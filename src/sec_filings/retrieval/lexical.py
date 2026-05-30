"""BM25 lexical index over chunks (the keyword half of hybrid retrieval).

This is the sparse counterpart to dense embedding retrieval. BM25 matches on
exact terms (tickers, line-item names like "deferred revenue", section
identifiers), which dense vectors blur — so we keep a lexical index alongside
the vector store and fuse their results upstream.

We use ``bm25s`` (a fast pure-numpy/scipy BM25) rather than a full search
engine: the corpus is small, persistence is a couple of files, and there is no
service to run. The index stores documents *positionally*; bm25s does not carry
our ``chunk_id`` through, so we persist a sidecar ``chunk_ids.json`` in the same
insertion order. ``retrieve`` returns positional indices, which we map back to
chunk ids via that list — this is why insertion order is load-bearing and must
not be reordered between build and load.

No silent fallbacks (see PROJECT.md): a missing sidecar, a corrupt index, or
``k`` larger than the corpus surfaces as an exception (we clamp ``k`` to the
corpus size because bm25s' ``retrieve`` itself raises when ``k`` exceeds the
number of documents).
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s

from sec_filings.corpus.models import Chunk

_CHUNK_IDS_FILENAME = "chunk_ids.json"


def build_lexical_index(chunks: list[Chunk], persist_dir: Path) -> None:
    """Build a BM25 index over ``chunks`` and persist it to ``persist_dir``.

    Writes the bm25s index files plus a ``chunk_ids.json`` sidecar that maps
    each document's positional index (the order in ``chunks``) back to its
    ``chunk_id``. The positional ordering is the contract between build and
    search, so the sidecar must be written from the same list passed to
    ``index`` and never reordered.
    """
    if not chunks:
        raise ValueError("Cannot build a lexical index from zero chunks.")

    tokens = bm25s.tokenize(
        [c.text for c in chunks], stopwords="en", show_progress=False
    )
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)

    persist_dir.mkdir(parents=True, exist_ok=True)
    # Save without corpus=: we carry ids out-of-band in the sidecar instead of
    # letting bm25s store/serialise the document texts.
    retriever.save(str(persist_dir), show_progress=False)

    chunk_ids = [c.chunk_id for c in chunks]
    (persist_dir / _CHUNK_IDS_FILENAME).write_text(
        json.dumps(chunk_ids), encoding="utf-8"
    )


class LexicalIndex:
    """A loaded BM25 index plus its positional chunk-id mapping.

    Construct via :meth:`load`; the index is immutable once loaded. ``search``
    returns ``(chunk_id, bm25_score)`` pairs in descending score order.
    """

    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str]) -> None:
        self._retriever = retriever
        self._chunk_ids = chunk_ids

    @classmethod
    def load(cls, persist_dir: Path) -> LexicalIndex:
        """Load a previously built index from ``persist_dir``.

        Raises if the directory lacks the bm25s index or the ``chunk_ids.json``
        sidecar — a partial/corrupt index must not silently degrade retrieval.
        """
        sidecar = persist_dir / _CHUNK_IDS_FILENAME
        if not sidecar.exists():
            raise FileNotFoundError(
                f"Missing {_CHUNK_IDS_FILENAME} sidecar in {persist_dir!s}; "
                "the lexical index is incomplete or was not built by "
                "build_lexical_index."
            )

        # mmap=False: corpus is small and we want a self-contained in-memory
        # index (no open file handles pinning the persist dir).
        retriever = bm25s.BM25.load(str(persist_dir), load_corpus=False, mmap=False)
        chunk_ids = json.loads(sidecar.read_text(encoding="utf-8"))
        return cls(retriever, chunk_ids)

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, bm25_score)`` pairs for ``query``.

        ``k`` is clamped to the corpus size: bm25s' ``retrieve`` raises if asked
        for more documents than exist, and asking for "all of a smaller corpus"
        is a legitimate caller intent rather than an error.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        num_docs = len(self._chunk_ids)
        k = min(k, num_docs)

        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        # retrieve returns arrays of shape (1, k): one row per query. We issue a
        # single query, so we read row 0.
        idx, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)

        results: list[tuple[str, float]] = []
        for position, score in zip(idx[0], scores[0]):
            results.append((self._chunk_ids[int(position)], float(score)))
        return results
