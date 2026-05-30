"""Voyage embedding wrapper for documents and queries.

Thin wrapper around the ``voyageai`` SDK so the rest of the retrieval stack
talks to one stable surface (``embed_documents`` / ``embed_query``) and the
model/dimension are pinned from settings (reproducible retrieval).

Why a lazy client singleton: ``voyageai.Client(...)`` raises ``AuthenticationError``
at *construction* when no key is present. Building it at import time would break
plain imports and test collection on key-less machines, so we defer construction
to the first embed call and cache it.

No silent fallbacks (see PROJECT.md): we pass ``truncation=False`` so over-long
input RAISES instead of being quietly clipped, and we assert every returned
vector has the expected dimension.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import voyageai

from sec_filings.config import settings

if TYPE_CHECKING:
    from voyageai import Client

EMBED_MODEL = settings.embed_model
EMBED_DIM = settings.embed_dim

# Voyage caps a single request at 1000 texts / 120K tokens; 128 is the safe
# convention here and stays well clear of both limits for filing-sized chunks.
DEFAULT_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _get_client() -> "Client":
    """Construct (and cache) the Voyage client lazily.

    Deferred to first use because the constructor authenticates eagerly; the
    key is passed explicitly from settings (pydantic-settings does not export
    into os.environ — see PROJECT.md).
    """
    return voyageai.Client(
        api_key=settings.voyage_api_key,
        max_retries=2,
        timeout=60.0,
    )


def _embed(texts: list[str], *, input_type: str) -> list[list[float]]:
    """Embed `texts` with the pinned model, validating each vector's dimension."""
    resp = _get_client().embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type=input_type,
        output_dimension=EMBED_DIM,
        truncation=False,  # over-long input must RAISE, never be silently clipped
    )
    vectors = resp.embeddings
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Voyage returned {len(vectors)} vectors for {len(texts)} inputs."
        )
    for vec in vectors:
        if len(vec) != EMBED_DIM:
            raise RuntimeError(
                f"Voyage returned a vector of dim {len(vec)}, expected {EMBED_DIM}."
            )
    return vectors


def embed_documents(
    texts: list[str], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[list[float]]:
    """Embed corpus documents/chunks, batching client calls.

    The SDK does not batch internally, so we slice into `batch_size` requests
    and concatenate. Returned vectors align 1:1 with `texts`.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed(batch, input_type="document"))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single search query (uses Voyage's asymmetric 'query' input type)."""
    return _embed([text], input_type="query")[0]
