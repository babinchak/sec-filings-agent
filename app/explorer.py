"""Streamlit explorer — debug FinanceBench retrieval misses over the real corpus.

Run from the repo root:
    uv run streamlit run app/explorer.py

Tab 1 (Miss inspector): pick a FinanceBench question and see its gold answer +
gold evidence beside what our hybrid retrieval returns — rank, fused score, and
dense/lexical provenance — with the gold numbers highlighted inside each
retrieved chunk and a banner classifying the miss (chunk-gap vs query/ranking).
Edit the query and rerun to watch the ranking move.

Tab 2 (Chunk boundaries): see how a filing's Item is split into chunks, with
char offsets, token counts, and the overlap each chunk shares with the previous.

All compute lives in sec_filings.inspection; this file is only UI + caching.
"""

from __future__ import annotations

import html

import streamlit as st

from sec_filings import inspection as insp

st.set_page_config(page_title="SEC Filings — Eval Miss Explorer", layout="wide")

_HL = "#fde68a"  # gold-number highlight
_OVERLAP = "#dbeafe"  # chunk-overlap tint


@st.cache_data(show_spinner=False)
def _questions():
    return insp.eval_questions()


@st.cache_data(show_spinner="Retrieving…")
def _retrieve(query, ticker, fiscal_year, top_k, gold_evidence):
    return insp.retrieve_ranked(
        query,
        ticker=ticker,
        fiscal_year=fiscal_year,
        top_k=top_k,
        gold_evidence=gold_evidence,
    )


@st.cache_data(show_spinner=False)
def _best_overlap(question, ticker, fiscal_year, gold_evidence):
    return insp.best_gold_overlap(
        question, ticker=ticker, fiscal_year=fiscal_year, gold_evidence=gold_evidence
    )


@st.cache_data(show_spinner="Ingesting + chunking filing…")
def _filing_chunks(ticker, fiscal_year):
    return insp.filing_chunks(ticker, fiscal_year)


def _highlight(text: str, needles) -> str:
    """HTML-escape `text`, then wrap each needle (len>=4) in a highlight mark."""
    esc = html.escape(text)
    for needle in sorted({n for n in needles if len(n) >= 4}, key=len, reverse=True):
        esc = esc.replace(
            html.escape(needle),
            f"<mark style='background:{_HL};padding:0 2px'>{html.escape(needle)}</mark>",
        )
    return esc


def _box(body_html: str) -> str:
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;"
        "margin:4px 0 14px;max-height:340px;overflow:auto;font-size:0.82rem;"
        "white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,monospace'>"
        f"{body_html}</div>"
    )


questions = _questions()

st.sidebar.header("FinanceBench question")
labels = [
    f"[{q.ticker} {q.fiscal_year}] {q.question_type} — {q.question[:50]}"
    for q in questions
]
idx = st.sidebar.selectbox(
    "Question", range(len(questions)), format_func=lambda i: labels[i]
)
q = questions[idx]
top_k = st.sidebar.slider("top_k (results shown)", 3, 20, 8)
st.sidebar.caption(f"id: {q.question_id}")
st.sidebar.caption(f"{len(questions)} questions in the starter eval set")

tab_miss, tab_chunks = st.tabs(["🔎 Miss inspector", "🧱 Chunk boundaries"])

with tab_miss:
    st.subheader(f"{q.ticker} {q.fiscal_year} · {q.question_type}")
    st.markdown(f"**Q:** {q.question}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Gold answer** (FinanceBench)")
        st.info(q.gold_answer or "—")
    with col_b:
        st.markdown("**Gold evidence** (the passage a correct answer must use)")
        st.markdown(
            _box(_highlight(q.gold_evidence or "—", insp.numbers(q.gold_evidence))),
            unsafe_allow_html=True,
        )

    best_rank, best_ov = _best_overlap(
        q.question, q.ticker, q.fiscal_year, q.gold_evidence or ""
    )
    if best_rank is None:
        st.caption(
            "Prose question (no salient numbers) — judge by reading the passages below."
        )
    else:
        msg = (
            f"Best gold-overlap chunk holds **{best_ov:.0%}** of the gold numbers, "
            f"at **rank #{best_rank}** for the raw question."
        )
        if best_ov < 0.5:
            st.error(f"{msg}  →  **CHUNK-GAP**: the figures aren't together in any one chunk.")
        elif best_rank > top_k:
            st.warning(
                f"{msg}  →  **QUERY/RANKING**: the chunk exists but ranks below "
                f"top_k={top_k}. Reword the query below and watch it climb."
            )
        else:
            st.success(f"{msg}  →  retrievable within the current top_k.")

    query = st.text_area(
        "Search query (defaults to the question — edit and rerun to watch ranking move)",
        value=q.question,
        height=70,
    )
    results = _retrieve(query, q.ticker, q.fiscal_year, top_k, q.gold_evidence or "")
    has_gold = bool(insp.numbers(q.gold_evidence))
    st.markdown(f"**Top {len(results)} passages** · {q.ticker} FY{q.fiscal_year}")
    for r in results:
        dense = f"dense #{r.dense_rank + 1}" if r.dense_rank is not None else "dense —"
        lex = f"lexical #{r.lexical_rank + 1}" if r.lexical_rank is not None else "lexical —"
        gold = (
            f" · gold# {len(r.gold_numbers_hit)} ({r.gold_overlap:.0%})" if has_gold else ""
        )
        st.markdown(
            f"**#{r.rank}** · {r.item} · fused {r.fused_score:.4f} · {dense} / {lex} "
            f"· {r.token_count} tok{gold}"
        )
        st.markdown(
            _box(_highlight(r.text, set(r.gold_numbers_hit))), unsafe_allow_html=True
        )

with tab_chunks:
    st.subheader(f"Chunk layout · {q.ticker} {q.fiscal_year}")
    filing, chunks = _filing_chunks(q.ticker, q.fiscal_year)
    items = sorted({c.item for c in chunks}, key=lambda s: (len(s), s))
    default = items.index("Item 8") if "Item 8" in items else 0
    item = st.selectbox("Item", items, index=default)
    item_chunks = [c for c in chunks if c.item == item]
    st.caption(
        f"{len(item_chunks)} chunks · {sum(c.token_count for c in item_chunks)} tokens · "
        f"blue = text shared with the previous chunk (overlap)"
    )

    prev_end: int | None = None
    for n, c in enumerate(item_chunks, start=1):
        overlap_chars = (
            prev_end - c.char_start if prev_end is not None and c.char_start < prev_end else 0
        )
        if overlap_chars > 0:
            shared = html.escape(c.text[:overlap_chars])
            rest = html.escape(c.text[overlap_chars:])
            body = f"<span style='background:{_OVERLAP}'>{shared}</span>{rest}"
        else:
            body = html.escape(c.text)
        meta = f"**chunk {n}/{len(item_chunks)}** · chars [{c.char_start}:{c.char_end}] · {c.token_count} tok"
        if overlap_chars > 0:
            meta += f" · overlaps prev by {overlap_chars} chars"
        st.markdown(meta)
        st.markdown(_box(body), unsafe_allow_html=True)
        prev_end = c.char_end
