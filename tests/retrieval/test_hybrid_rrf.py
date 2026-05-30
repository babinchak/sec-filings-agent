"""Unit tests for Reciprocal Rank Fusion (the pure half of hybrid retrieval).

These call ``reciprocal_rank_fusion`` directly with fabricated id-lists, so they
need neither Chroma, bm25s, nor network — exactly why the fusion logic is a pure
function separated from the I/O in HybridRetriever.
"""

from __future__ import annotations

from sec_filings.retrieval.hybrid import reciprocal_rank_fusion


def _order(fused: list[tuple[str, float, int | None, int | None]]) -> list[str]:
    """Project the fused result down to just its id ordering."""
    return [row[0] for row in fused]


def _score(
    fused: list[tuple[str, float, int | None, int | None]], cid: str
) -> float:
    return next(row[1] for row in fused if row[0] == cid)


def test_rrf_doc_in_both_beats_doc_in_one() -> None:
    """A doc present in both rankers outscores one present in only a single one.

    'both' sits at rank 1 in each list (a worse rank than the rank-0 doc each
    list leads with), yet its two contributions sum above either single
    rank-0 contribution — that is the whole point of fusion.
    """
    dense = ["dense_only", "both"]
    lexical = ["lexical_only", "both"]

    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)

    assert _order(fused)[0] == "both"
    assert _score(fused, "both") > _score(fused, "dense_only")
    assert _score(fused, "both") > _score(fused, "lexical_only")


def test_rrf_combines_ranks() -> None:
    """A doc ranked #1 in both lists lands first overall."""
    dense = ["winner", "a", "b", "c"]
    lexical = ["winner", "x", "y", "z"]

    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)

    assert _order(fused)[0] == "winner"
    # 'winner' is at rank 0 in both rankers, so its provenance is (0, 0).
    winner_row = next(row for row in fused if row[0] == "winner")
    assert winner_row[2] == 0  # dense_rank
    assert winner_row[3] == 0  # lexical_rank


def test_rrf_handles_ids_in_only_one_list() -> None:
    """Ids unique to one ranker still appear, with None for the absent ranker."""
    dense = ["d1", "shared"]
    lexical = ["shared", "l1"]

    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)

    ids = _order(fused)
    assert set(ids) == {"d1", "shared", "l1"}

    by_id = {row[0]: row for row in fused}
    # d1 is dense-only: has a dense rank, no lexical rank.
    assert by_id["d1"][2] == 0
    assert by_id["d1"][3] is None
    # l1 is lexical-only: no dense rank, has a lexical rank.
    assert by_id["l1"][2] is None
    assert by_id["l1"][3] == 1
    # shared appears in both, so it outscores either single-list doc.
    assert by_id["shared"][2] == 1
    assert by_id["shared"][3] == 0
    assert _score(fused, "shared") > _score(fused, "d1")
    assert _score(fused, "shared") > _score(fused, "l1")


def test_rrf_sorted_descending_by_score() -> None:
    """Output is sorted by fused score, descending (with deterministic tiebreak)."""
    dense = ["a", "b", "c"]
    lexical = ["c", "b", "a"]

    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60)

    scores = [row[1] for row in fused]
    assert scores == sorted(scores, reverse=True)
    # b is at rank 1 in both -> symmetric; a and c each have one good + one bad
    # rank that sum to the same as b's two middling ranks here. The tiebreak by
    # chunk_id keeps the order stable and deterministic regardless.
    assert _order(fused) == sorted(_order(fused), key=lambda cid: (-_score(fused, cid), cid))


def test_rrf_empty_inputs_yield_empty_output() -> None:
    assert reciprocal_rank_fusion([], []) == []
