"""ChromaDB-backed vector store where WE own the embeddings.

We compute Voyage vectors ourselves (see retrieval/embedding.py) and hand them
to Chroma. Chroma is used purely as an HNSW index + metadata filter; it must
NOT embed anything itself. We therefore pin `embedding_function=None` on the
collection so Chroma never installs its DefaultEmbeddingFunction and re-embeds
our documents with the wrong model. A defensive assert enforces this.

No silent fallbacks (see PROJECT.md): a dim mismatch, a None embedding_function
that quietly became something else, or Chroma rejecting metadata will surface as
an exception rather than corrupting the index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from sec_filings.config import settings
from sec_filings.corpus.models import Chunk

# Chroma rejects None and empty-list metadata values, so we never emit them.
# These are the chunk fields we mirror into Chroma metadata for filtering and
# for rebuilding a Chunk in get_chunks(). `text` lives in `documents`, not here.
_METADATA_FIELDS = (
    "ticker",
    "fiscal_year",
    "filing_accession",
    "item",
    "section_path",
    "char_start",
    "char_end",
    "token_count",
)


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """Build a Chroma-safe metadata dict from a Chunk.

    Chroma rejects None values AND empty lists, so we drop any such key (an empty
    `section_path` is simply omitted and reconstructed as [] on read). We never
    emit an empty dict — every chunk has at least ticker/fiscal_year populated.
    """
    raw = {
        "ticker": chunk.ticker,
        "fiscal_year": chunk.fiscal_year,
        "filing_accession": chunk.filing_accession,
        "item": chunk.item,
        "section_path": chunk.section_path,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "token_count": chunk.token_count,
    }
    return {
        key: value
        for key, value in raw.items()
        if value is not None and value != []
    }


class VectorStore:
    """Thin wrapper over a single Chroma collection of pre-embedded chunks."""

    def __init__(self, client: chromadb.api.ClientAPI, collection: Any) -> None:
        # Constructed via `open()`; takes the already-opened client/collection so
        # the embedding_function guard runs exactly once at open time.
        self._client = client
        self._collection = collection

    @classmethod
    def open(
        cls,
        persist_dir: Path = settings.chroma_dir,
        collection: str = settings.chroma_collection,
    ) -> "VectorStore":
        """Open (or create) a persistent Chroma collection at `persist_dir`.

        Pins cosine space and disables Chroma-side embedding so our Voyage
        vectors are stored verbatim.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_or_create_collection(
            name=collection,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        # Defensive: if Chroma installed an embedding function despite our None,
        # our distances would be computed against the wrong vectors. Fail loud.
        # getattr() so a future internal rename degrades to a clear assert, not
        # an AttributeError.
        assert getattr(col, "_embedding_function", None) is None, (
            "Chroma installed an embedding function despite embedding_function=None; "
            "our Voyage vectors would be ignored."
        )
        return cls(client, col)

    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Insert/overwrite chunks with their precomputed embeddings.

        Idempotent: re-upserting the same chunk_id replaces its row, so re-runs
        of ingestion don't duplicate. Empty input is a no-op (Chroma rejects
        empty id lists).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "length mismatch."
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(c) for c in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        *,
        ticker: str,
        fiscal_year: int,
        n_results: int,
    ) -> list[tuple[str, float]]:
        """Nearest chunks within a single (ticker, fiscal_year) filing.

        Returns (chunk_id, cosine_distance) pairs; distance is in [0, 2] and the
        caller converts to similarity via 1 - distance.
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"fiscal_year": {"$eq": fiscal_year}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
        # Results are nested one list per query embedding; we sent exactly one.
        ids0 = result["ids"][0]
        distances0 = result["distances"][0]
        return list(zip(ids0, distances0))

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Rebuild Chunks for the given ids, preserving input order.

        Chroma's get() returns rows in its own order, so we index by id and
        re-emit in the caller's order. Ids missing from the collection are
        skipped (get() simply won't return them).
        """
        if not chunk_ids:
            return []
        result = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        by_id: dict[str, Chunk] = {}
        for cid, document, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            meta = metadata or {}
            by_id[cid] = Chunk(
                chunk_id=cid,
                filing_accession=meta["filing_accession"],
                ticker=meta["ticker"],
                fiscal_year=meta["fiscal_year"],
                item=meta["item"],
                section_path=list(meta.get("section_path", [])),
                text=document,
                char_start=meta["char_start"],
                char_end=meta["char_end"],
                token_count=meta["token_count"],
                metadata={},
            )
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()
