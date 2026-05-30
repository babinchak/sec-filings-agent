"""Hybrid-retrieval backend for the agent's `retrieve` tool.

The agent answers *qualitative* questions from the filing's own prose, so it
needs a way to pull the most relevant passages of a 10-K on demand. This module
wraps :func:`sec_filings.retrieval.hybrid.hybrid_search` (dense + lexical, fused
with RRF) as an Anthropic native tool: a schema the model sees and a handler the
dispatch loop calls.

The handler returns only the fields the model needs to reason and cite —
``chunk_id`` (for provenance/citations), ``item`` and ``section_path`` (where in
the filing the passage lives), ``text``, and the fused relevance ``score`` — all
JSON-serializable. We keep the retrieval policy (CANDIDATE_K, RRF_K) hidden in
the hybrid module; the tool exposes only ``query`` plus the filing selectors.

No silent fallbacks (see PROJECT.md): ``hybrid_search`` raises on an unindexed
(ticker, fiscal_year) or out-of-sync indexes, and we let those propagate.
"""

from __future__ import annotations

from sec_filings.retrieval.hybrid import hybrid_search

# Anthropic ToolParam describing this tool to the model. Phrased for a
# financial-filings analyst: when to reach for filing text vs. a numeric lookup.
RETRIEVE_TOOL_SCHEMA: dict = {
    "name": "retrieve",
    "description": (
        "Search a company's 10-K annual report for the passages most relevant to "
        "a question, using hybrid (semantic + keyword) retrieval over the filing's "
        "text. Use this to ground qualitative analysis — business strategy, risk "
        "factors, MD&A commentary, competition, segments, legal proceedings — in "
        "the filing's own words. Each returned passage carries its filing section "
        "(e.g. 'Item 1A Risk Factors') and a chunk_id you can cite. For precise "
        "reported numbers, prefer the dedicated financial-fact tool instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language search query describing the information you "
                    "need, e.g. 'supply chain and manufacturing risks'."
                ),
            },
            "ticker": {
                "type": "string",
                "description": "Stock ticker of the company to search, e.g. 'MSFT'.",
            },
            "fiscal_year": {
                "type": "integer",
                "description": "Fiscal year of the 10-K to search, e.g. 2023.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to return (default 8).",
            },
        },
        "required": ["query"],
    },
}


def handle_retrieve(
    query: str,
    ticker: str = "MSFT",
    fiscal_year: int = 2023,
    top_k: int = 8,
) -> dict:
    """Tool dispatch handler: hybrid-retrieve passages for the agent to read.

    Returns a JSON-serializable dict of the query echoed back plus a list of
    passages (each with its citation id, section provenance, text, and fused
    relevance score). Defaults target the canonical MSFT FY2023 filing so the
    model can call this with just a ``query``.
    """
    results = hybrid_search(query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k)
    return {
        "query": query,
        "passages": [
            {
                "chunk_id": sc.chunk.chunk_id,
                "item": sc.chunk.item,
                "section_path": sc.chunk.section_path,
                "text": sc.chunk.text,
                "score": round(sc.fused_score, 5),
            }
            for sc in results
        ],
    }
