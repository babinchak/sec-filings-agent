"""Langfuse tracing — the one module that knows Langfuse exists.

Everything else (the agent loop, tools, scripts) talks to tracing through the
small context managers here, so Langfuse is a single, swappable seam rather than
imports sprinkled across the codebase. Each helper is a NO-OP when Langfuse is
not configured: if ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are blank we
yield a do-nothing observation and the agent runs exactly as before. That is a
deliberately optional feature, not a silent fallback hiding an error (see
PROJECT.md) — the smoke tests and unit tests must not require Langfuse keys.

Verified against the Langfuse Python SDK 4.7.1 (the OpenTelemetry-based v4 line):
  * ``Langfuse(public_key=, secret_key=, host=)`` builds the singleton; later
    ``get_client()`` returns it. We construct it explicitly from ``settings``
    because pydantic-settings loads .env into ``settings``, not ``os.environ``,
    so the SDK's own env auto-detection would not find the keys.
  * v4 unified span/generation creation into
    ``start_as_current_observation(name=, as_type=...)`` where ``as_type`` is one
    of span/generation/agent/tool/retriever/... — giving a semantically rich
    waterfall (a real ``retriever`` span for hybrid search, not a generic box).
  * Nesting is automatic via the OpenTelemetry active context: an observation
    opened inside another's ``with`` block becomes its child.
  * Token usage / cost ride on ``.update(usage_details=..., cost_details=...)``;
    passing ``model`` + ``usage_details`` lets Langfuse price the call itself.
  * Trace-level attributes: ``propagate_attributes(trace_name=, tags=)`` and
    ``set_current_trace_io(input=, output=)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from sec_filings.config import settings


@lru_cache(maxsize=1)
def _client_or_none() -> Any | None:
    """Return a configured Langfuse client, or None if tracing is unconfigured.

    Cached so the singleton (and its background exporter thread) is built once.
    Importing ``langfuse`` lazily keeps it off the import path of code that never
    traces (and out of test collection).
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )


def tracing_enabled() -> bool:
    """True when Langfuse keys are present and traces will be recorded."""
    return _client_or_none() is not None


class _NullObservation:
    """Stand-in yielded when tracing is off. Swallows every update call."""

    def update(self, **_kwargs: Any) -> "_NullObservation":
        return self


_NULL = _NullObservation()


@contextlib.contextmanager
def observe_agent_run(
    question: str,
    *,
    trace_name: str = "sec_filings_agent",
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Root span for one agent invocation — this IS the Langfuse trace.

    Names and tags the trace, records the question as the trace input, and yields
    the root ``agent`` observation. No-op (yields ``_NULL``) when tracing is off.
    """
    client = _client_or_none()
    if client is None:
        yield _NULL
        return

    from langfuse import propagate_attributes

    with propagate_attributes(trace_name=trace_name, tags=tags or ["sec-10k-agent"]):
        with client.start_as_current_observation(
            name=trace_name, as_type="agent", input=question
        ) as root:
            client.set_current_trace_io(input=question)
            yield root


@contextlib.contextmanager
def observe_generation(
    *,
    name: str,
    model: str,
    input: Any = None,
    model_parameters: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """A single Claude ``messages.create`` call, as a ``generation`` span.

    Set the call's output and token usage on the yielded object via
    ``.update(output=..., usage_details={"input": n, "output": m})`` — passing
    ``model`` here lets Langfuse compute the USD cost itself.
    """
    client = _client_or_none()
    if client is None:
        yield _NULL
        return
    with client.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
        model_parameters=model_parameters,
    ) as gen:
        yield gen


@contextlib.contextmanager
def observe_tool(name: str, *, as_type: str = "tool", input: Any = None) -> Iterator[Any]:
    """A single tool dispatch (``retriever`` for hybrid search, ``tool`` for FMP).

    Set the result with ``.update(output=...)``; on a recoverable tool error,
    ``.update(level="ERROR", status_message=..., output=...)``.
    """
    client = _client_or_none()
    if client is None:
        yield _NULL
        return
    with client.start_as_current_observation(name=name, as_type=as_type, input=input) as obs:
        yield obs


def set_trace_output(output: Any) -> None:
    """Record the final answer as the trace's top-level output (no-op if off)."""
    client = _client_or_none()
    if client is not None:
        client.set_current_trace_io(output=output)


def current_trace_id() -> str | None:
    """The active Langfuse trace id, if any — stash it on the AgentTrace."""
    client = _client_or_none()
    return client.get_current_trace_id() if client is not None else None


def trace_url(trace_id: str | None) -> str | None:
    """A clickable Langfuse URL for a stashed trace id (for notebooks/CLIs)."""
    client = _client_or_none()
    if client is None or trace_id is None:
        return None
    return client.get_trace_url(trace_id=trace_id)


def flush() -> None:
    """Block until buffered spans are sent — call after a run so a notebook or
    script sees the trace immediately rather than waiting for the batch timer."""
    client = _client_or_none()
    if client is not None:
        client.flush()
