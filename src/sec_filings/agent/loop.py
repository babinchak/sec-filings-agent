"""Hand-written Anthropic tool-use loop (the Level-3 agentic core).

This is the agent itself: a bounded reasoning loop that lets Claude decompose a
question, call our two native tools (``retrieve`` for filing prose,
``get_financial_fact`` for exact reported numbers), read the results, and chain
calls (fetch numbers, then compute a ratio) before answering. We write the loop
by hand on purpose (see PROJECT.md — no LangChain/LlamaIndex) so every step is
visible and recordable into an :class:`AgentTrace` for the inspector UI and eval.

Verified against the Anthropic Python SDK 0.104.1:
  * ``client.messages.create(..., tools=..., tool_choice={'type': 'auto'})`` —
    the model decides whether to call a tool.
  * When ``resp.stop_reason == 'tool_use'`` the assistant turn's ``content`` is a
    list of blocks; tool calls are the ``type == 'tool_use'`` blocks, each with
    ``.id``, ``.name`` and ``.input`` (``.input`` is ALREADY a dict — never
    ``json.loads`` it). Several such blocks in one turn = parallel tool calls.
  * We echo the assistant ``content`` back VERBATIM, then reply with ONE user
    message bundling every ``tool_result`` block (each keyed by ``tool_use_id``,
    with a STRING ``content`` — we ``json.dumps`` our dict results).

Failure policy (see PROJECT.md — no silent fallbacks):
  * An EXPECTED runtime failure from a tool (``RuntimeError``/``ValueError``/
    ``LookupError`` — bad/absent fiscal year, quota, missing concept; note
    ``KeyError`` is a ``LookupError``) becomes a ``tool_result`` with
    ``is_error=True`` so the model can recover (retry with an available year,
    fall back to ``retrieve``). The agent stays robust without us hiding the
    problem.
  * An unknown tool name is a PROGRAMMER error (our schema and dispatch drifted),
    so ``dispatch_tool`` raises ``ValueError`` and we let it propagate.
  * If the loop hits ``max_steps`` without a final text answer, we ``raise
    RuntimeError`` rather than return a half-formed trace.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import anthropic

from sec_filings import observability as obs
from sec_filings.agent.models import AgentStep, AgentTrace
from sec_filings.config import settings
from sec_filings.tools.fmp import FMP_TOOL_SCHEMA, handle_get_financial_fact
from sec_filings.tools.retrieve import RETRIEVE_TOOL_SCHEMA, handle_retrieve

# System prompt cast as a financial-filings analyst. It encodes the routing
# policy the whole design hinges on: EXACT reported figures come from the
# numeric tool, narrative/qualitative content from retrieval, ratios are chained
# (fetch each number, then compute), and every number must be cited. "Answer
# only from tool results" keeps the model from inventing figures from memory.
ROUTER_SYSTEM: str = (
    "You are a meticulous financial-filings analyst answering questions about "
    "public companies' SEC 10-K annual reports. You have two tools:\n\n"
    "  - get_financial_fact: returns a single EXACT reported figure (e.g. total "
    "revenue, R&D expense) for a company and fiscal year from its income "
    "statement. Use this for any precise number a user asks for or that a "
    "calculation needs. Never quote a figure from memory.\n"
    "  - retrieve: hybrid search over the 10-K's own text. Use this for "
    "narrative or qualitative content — business strategy, risk factors, MD&A "
    "commentary, competition, segments, legal proceedings.\n\n"
    "Method:\n"
    "  1. Decompose the question into sub-queries. Decide for EACH whether it "
    "asks for an exact reported figure (route to get_financial_fact) or "
    "narrative/qualitative content (route to retrieve).\n"
    "  2. For ratios, growth rates, or any derived metric, CHAIN tools: fetch "
    "each underlying number with get_financial_fact first, then compute the "
    "result yourself from those returned values. Do not estimate.\n"
    "  3. You may issue several tool calls at once when sub-queries are "
    "independent.\n"
    "  4. The indexed corpus is limited to specific filings. If a tool reports "
    "that the requested fiscal year is not available, retry with one of the "
    "available fiscal years it lists rather than giving up.\n\n"
    "Answering rules:\n"
    "  - If a question asks which securities or items are registered, listed, or "
    "disclosed and the retrieved evidence shows only non-matching items (or "
    "states 'none'), answer plainly that there are none. Do not hedge or defer "
    "to an exhibit you have not actually read.\n"
    "  - If a specific figure for the requested item is disclosed ANYWHERE in "
    "the filing — on the face of a statement OR within a note — report that "
    "figure. Do not answer '0' or 'not present' merely because it is not a "
    "separate line item on the face of a statement.\n\n"
    "Answer ONLY from the tool results you receive — never from prior knowledge. "
    "If the tools cannot supply something, say so plainly. Cite the source of "
    "every number (the tool, company, fiscal year, and concept) and cite the "
    "chunk_id / filing section for every qualitative claim drawn from retrieved "
    "text."
)

# The native tools the model sees, in the order it should reach for them. Both
# schemas live next to their handlers in sec_filings.tools so schema and handler
# never drift; this loop just wires them together.
TOOLS: list[dict] = [RETRIEVE_TOOL_SCHEMA, FMP_TOOL_SCHEMA]

# Names the loop will dispatch; anything else is schema/dispatch drift (RAISE).
_KNOWN_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOLS)


def dispatch_tool(name: str, tool_input: dict) -> object:
    """Route a tool-use block to its handler and return the raw result object.

    ``tool_input`` is the SDK-parsed ``.input`` dict, passed straight through as
    kwargs. An unknown ``name`` means our TOOLS schema and this dispatch have
    drifted — a programmer error, so we RAISE (see PROJECT.md), never a silent
    no-op. Expected *runtime* failures inside a handler (bad year, quota) are
    NOT caught here; the loop catches those to build an is_error tool_result.
    """
    if name == "retrieve":
        return handle_retrieve(**tool_input)
    if name == "get_financial_fact":
        return handle_get_financial_fact(**tool_input)
    raise ValueError(f"Unknown tool {name!r}. Known tools: {[t['name'] for t in TOOLS]}.")


# --- Langfuse logging helpers -------------------------------------------------
# These render what we send to / get back from Claude into compact, JSON-safe
# shapes for the trace UI. They never feed the model — only the observability
# layer — so they truncate freely and must never raise on odd input.

def _safe_block(block: Any) -> dict[str, Any]:
    """One Anthropic content block (SDK object OR plain dict) -> JSON-safe dict."""
    def attr(key: str) -> Any:
        return block.get(key) if isinstance(block, dict) else getattr(block, key, None)

    btype = attr("type")
    if btype == "text":
        return {"type": "text", "text": attr("text")}
    if btype == "tool_use":
        return {"type": "tool_use", "name": attr("name"), "input": attr("input")}
    if btype == "tool_result":
        content = attr("content")
        text = content if isinstance(content, str) else str(content)
        return {"type": "tool_result", "content": text[:800]}
    return {"type": btype}


def _safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The running conversation rendered JSON-safe for a generation's input."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m["content"]
        safe = content if isinstance(content, str) else [_safe_block(b) for b in content]
        out.append({"role": m["role"], "content": safe})
    return out


def _assistant_summary(resp: Any) -> list[dict[str, Any]]:
    """An assistant turn's text + tool calls, as a generation's output."""
    return [_safe_block(b) for b in resp.content if b.type in ("text", "tool_use")]


def _tool_output_preview(name: str, result: Any) -> Any:
    """Compact a tool result for the trace: drop bulky retrieved passage TEXT
    (keep chunk_id/item/section_path/score) so spans stay small — we keep the
    full text in the in-memory AgentTrace, not in Langfuse."""
    if name == "retrieve" and isinstance(result, dict):
        passages = result.get("passages", [])
        return {
            "query": result.get("query"),
            "n_passages": len(passages),
            "passages": [
                {k: v for k, v in p.items() if k != "text"} if isinstance(p, dict) else p
                for p in passages
            ],
        }
    return result


def run_agent(
    question: str,
    *,
    model: str = settings.agent_model,
    temperature: float = settings.agent_temperature,
    max_steps: int = 12,
    on_event: Callable[[AgentStep], None] | None = None,
) -> AgentTrace:
    """Run the bounded tool-use loop for one question, recording a full trace.

    Loops up to ``max_steps`` times. Each turn either finishes (the model emits
    a text answer, ``stop_reason != 'tool_use'``) or requests tools, which we
    dispatch and feed back. Every reasoning blurb, tool call, and tool result is
    recorded as an :class:`AgentStep` so the run is fully replayable/inspectable.

    ``on_event`` (optional) is called with each :class:`AgentStep` the moment it
    is recorded — a live observer hook so callers (a notebook, a CLI, the future
    Streamlit UI) can show the agent's progress *while it runs* rather than
    staring at a blank cell for the whole multi-step round-trip. It must not raise.

    Raises:
        RuntimeError: the loop exhausted ``max_steps`` without a final answer.
        ValueError: a tool-use block named a tool not in our dispatch table.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    started = datetime.utcnow()
    trace = AgentTrace(
        trace_id=str(uuid.uuid4()),
        question=question,
        model=model,
        timestamp=started,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    step_index = 0

    def emit(step: AgentStep) -> None:
        """Record a step on the trace and surface it live to ``on_event``."""
        trace.steps.append(step)
        if on_event is not None:
            on_event(step)

    # The whole run is ONE Langfuse trace: a root ``agent`` observation, with a
    # child ``generation`` per Claude turn and a child ``retriever``/``tool``
    # span per dispatch, so the waterfall mirrors this loop exactly. Every call
    # here is a no-op when Langfuse is unconfigured (see sec_filings.observability),
    # so the loop's behaviour is identical with or without tracing. The outer
    # try/finally flushes AFTER the root span closes so a notebook or script sees
    # the completed trace immediately rather than waiting on the batch timer.
    try:
        with obs.observe_agent_run(question) as root:
            trace.langfuse_trace_id = obs.current_trace_id()

            for turn in range(max_steps):
                with obs.observe_generation(
                    name=f"claude_turn_{turn + 1}",
                    model=model,
                    input=_safe_messages(messages),
                    model_parameters={
                        "max_tokens": 2048,
                        "temperature": temperature,
                        "tool_choice": "auto",
                    },
                ) as gen:
                    resp = client.messages.create(
                        model=model,
                        max_tokens=2048,
                        temperature=temperature,
                        system=ROUTER_SYSTEM,
                        tools=TOOLS,
                        tool_choice={"type": "auto"},
                        messages=messages,
                    )
                    # Logging ``model`` + ``usage_details`` lets Langfuse price
                    # the call itself, so the trace carries USD cost per turn.
                    gen.update(
                        output=_assistant_summary(resp),
                        usage_details={
                            "input": resp.usage.input_tokens,
                            "output": resp.usage.output_tokens,
                        },
                    )

                # Accumulate usage from every model turn (input + output tokens).
                trace.total_tokens_in += resp.usage.input_tokens
                trace.total_tokens_out += resp.usage.output_tokens

                if resp.stop_reason != "tool_use":
                    # Terminal turn: stitch the text blocks into the final answer.
                    final_answer = "".join(b.text for b in resp.content if b.type == "text")
                    trace.final_answer = final_answer
                    emit(
                        AgentStep(
                            step_index=step_index,
                            type="answer",
                            content=final_answer,
                            tokens_in=resp.usage.input_tokens,
                            tokens_out=resp.usage.output_tokens,
                        )
                    )
                    root.update(output=final_answer)
                    obs.set_trace_output(final_answer)
                    trace.elapsed_seconds = (datetime.utcnow() - started).total_seconds()
                    return trace

                # Tool-use turn. Record any interleaved reasoning text the model
                # emitted alongside its tool calls (trace breadcrumb for the inspector).
                reasoning = "".join(b.text for b in resp.content if b.type == "text")
                if reasoning.strip():
                    emit(
                        AgentStep(
                            step_index=step_index,
                            type="reasoning",
                            content=reasoning,
                            tokens_in=resp.usage.input_tokens,
                            tokens_out=resp.usage.output_tokens,
                        )
                    )
                    step_index += 1

                # Echo the assistant turn back VERBATIM — the SDK requires the
                # original content blocks (with their tool_use ids) to thread it.
                messages.append({"role": "assistant", "content": resp.content})

                # Dispatch EVERY tool_use block (>1 = parallel calls), bundling
                # all the results into ONE follow-up user message.
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue

                    # block.input is ALREADY a dict from the SDK — do NOT json.loads it.
                    tool_input: dict = block.input  # type: ignore[assignment]
                    emit(
                        AgentStep(
                            step_index=step_index,
                            type="tool_call",
                            tool_name=block.name,
                            tool_input=tool_input,
                        )
                    )
                    step_index += 1

                    # Guard the unknown-tool case OUTSIDE the span + recover catch:
                    # an unknown tool is a programmer error (schema/dispatch drift)
                    # and MUST propagate, never become an is_error the model
                    # "recovers" from.
                    if block.name not in _KNOWN_TOOL_NAMES:
                        raise ValueError(
                            f"Unknown tool {block.name!r}. "
                            f"Known tools: {sorted(_KNOWN_TOOL_NAMES)}."
                        )

                    # ``retrieve`` is a RETRIEVER span (renders as documents in the
                    # waterfall); the numeric FMP call is a plain TOOL span.
                    span_kind = "retriever" if block.name == "retrieve" else "tool"
                    with obs.observe_tool(block.name, as_type=span_kind, input=tool_input) as tobs:
                        try:
                            result = dispatch_tool(block.name, tool_input)
                            is_error = False
                            # tool_result content MUST be a STRING — serialize our dict.
                            result_content = json.dumps(result)
                            tool_output: Any = result
                            tobs.update(output=_tool_output_preview(block.name, result))
                        except (RuntimeError, ValueError, LookupError) as exc:
                            # Expected runtime failure (e.g. an unindexed fiscal
                            # year -> LookupError, FMP quota, missing concept):
                            # hand the error back so the model can recover (retry
                            # with an available year, fall back to retrieve). We
                            # pre-checked the tool name above, so this can only be
                            # a genuine tool runtime error. (KeyError is a
                            # LookupError, so missing-concept errors stay covered.)
                            is_error = True
                            result_content = str(exc)
                            tool_output = result_content
                            tobs.update(
                                output=result_content,
                                level="ERROR",
                                status_message=str(exc),
                            )

                    emit(
                        AgentStep(
                            step_index=step_index,
                            type="tool_result",
                            tool_name=block.name,
                            tool_input=tool_input,
                            tool_output=tool_output,
                        )
                    )
                    step_index += 1

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_content,
                            "is_error": is_error,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

            raise RuntimeError(
                f"Agent did not converge: hit max_steps={max_steps} without a final answer "
                f"for question {question!r}."
            )
    finally:
        obs.flush()
