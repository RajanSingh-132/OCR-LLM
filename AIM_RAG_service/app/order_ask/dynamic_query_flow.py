"""
Dynamic query flow — orchestrates the constrained query-builder chain for
`/api/v1/orders/ask`.

    question
      -> llm_query_builder.build_query_payload()      (1 Claude call)
      -> payload_validator.validate_payload()         (pure code)
      -> query_builder.execute_validated_query()      (SQL / procedure)
      -> shape_rows()                                 (strip internal cols)
      -> _phrase_answer()                             (1 Claude call)

On ANY failure to build a safe, runnable query the flow returns an honest
message — it never falls through to the Mongo pipeline and never guesses.

Wired in behind the AVAAL_DYNAMIC_QUERY_BUILDER env flag (off by default),
so with the flag unset `/api/v1/orders/ask` behaves exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.embedding_client import get_anthropic_llm
from app.order_ask.checkpoint import checkpoint
from app.order_ask.llm_query_builder import build_query_payload
from app.order_ask.payload_validator import ValidatedQuery, validate_payload
from app.order_ask.query_builder import (
    ProceduresNotLiveError,
    execute_validated_query,
    shape_rows,
)

logger = logging.getLogger("order_ask.dynamic_query_flow")

_UNSUPPORTED_MSG = (
    "I can answer questions about your order data — counts, freight totals, "
    "date ranges, lookups by order number, filtered lists. I can't help with "
    "that one."
)
_READONLY_MSG = (
    "I can only read and report on order data — I can't create, change, "
    "cancel, or delete anything. Ask me to look something up instead."
)

# Cheap pre-LLM guard: obvious write/modify intent never even reaches the
# query builder. The builder is already structurally read-only (allowlisted
# getter functions + SELECT-only dynamic path + READ ONLY DB session); this
# just makes the refusal immediate and explicit.
_WRITE_INTENT_RE = re.compile(
    r"\b(create|add|insert|new|update|edit|change|modif(?:y|ies)|set|"
    r"delete|remove|drop|truncate|cancel|assign|dispatch|approve|reject|"
    r"book|schedule|reschedule|mark\s+as|move\s+to|push\s+to|"
    r"upload|attach|send|email|generate|issue|post|save|overwrite)\b",
    re.IGNORECASE,
)
# ...but not when it's clearly a read ("show me new orders", "list added today")
_READ_CONTEXT_RE = re.compile(
    r"\b(how many|show|list|find|which|what|count|total|report|"
    r"give me|display|get|tell me|number of)\b",
    re.IGNORECASE,
)


def _looks_like_write(question: str) -> bool:
    q = question or ""
    if not _WRITE_INTENT_RE.search(q):
        return False
    # a leading read verb almost always means "new/added" is an adjective
    if _READ_CONTEXT_RE.search(q.split(".")[0][:60]):
        return False
    return True
_UNSAFE_MSG = (
    "I couldn't turn that into a safe query. It's been logged so we can add "
    "support for it. Try rephrasing, or ask a simpler version."
)
_PROCEDURES_INACTIVE_MSG = (
    "That needs a database capability that isn't switched on yet — the "
    "aggregate / lookup functions it relies on haven't been deployed. "
    "Simple created-date order counts already work; ask me one of those "
    "and I can answer now."
)


def _phrase_prompt(question: str, vq: ValidatedQuery, count: int, rows: List[Dict[str, Any]]) -> str:
    if vq.kind == "procedure":
        what = f"procedure {vq.procedure_kind} ({vq.procedure_name})"
        params = vq.procedure_params or {}
    else:
        what = f"dynamic query on {vq.table}"
        params = {
            "select": vq.select_fields,
            "filters": vq.filters,
            "group_by": vq.group_by,
            "aggregate": vq.aggregate,
            "aggregate_field": vq.aggregate_field,
        }
    sample = rows[:50]
    return (
        "You are answering a freight-order question from real query results. "
        "Be concise and factual. Use only the data given — do not invent numbers. "
        "If the result set is empty, say so plainly.\n\n"
        f"Question: {question}\n"
        f"Executed: {what}\n"
        f"Parameters: {json.dumps(params, default=str)}\n"
        f"Row count: {count}\n"
        f"Rows (up to 50 shown): {json.dumps(sample, default=str)}\n\n"
        "Answer:"
    )


def _phrase_sync(question: str, vq: ValidatedQuery, count: int, rows: List[Dict[str, Any]]) -> str:
    llm = get_anthropic_llm()
    raw = llm.invoke(_phrase_prompt(question, vq, count, rows))
    return (raw.content if hasattr(raw, "content") else str(raw)).strip()


async def run_dynamic_query_flow(
    corporate_id: str,
    question: str,
    session_id: Optional[str] = None,
    session_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Returns {"answer": str, "source": str, "session_id": ...}."""
    checkpoint("QUERYBUILD", "flow start", question=(question or "")[:120], corporate_id=corporate_id)

    if _looks_like_write(question):
        checkpoint("QUERYBUILD", "write intent refused (read-only)", question=(question or "")[:120])
        return {"answer": _READONLY_MSG, "source": "query_builder_readonly_refused", "session_id": session_id}

    payload = await build_query_payload(question, session_context)

    if str(payload.get("kind") or "").lower() == "unsupported":
        reason = str(payload.get("reason") or "")
        checkpoint("QUERYBUILD", "unsupported", reason=reason)
        if "write" in reason or "permit" in reason or "read-only" in reason:
            return {"answer": _READONLY_MSG, "source": "query_builder_readonly_refused", "session_id": session_id}
        return {"answer": _UNSUPPORTED_MSG, "source": "query_builder_unsupported", "session_id": session_id}

    vq = validate_payload(payload, question)
    if vq is None:
        return {"answer": _UNSAFE_MSG, "source": "query_builder_rejected", "session_id": session_id}

    try:
        count, rows = await execute_validated_query(corporate_id, vq)
    except ProceduresNotLiveError:
        checkpoint("QUERYBUILD", "procedures inactive", procedure=vq.procedure_kind)
        return {
            "answer": _PROCEDURES_INACTIVE_MSG,
            "source": "query_builder_procedures_inactive",
            "session_id": session_id,
        }
    except Exception as exc:
        msg = str(exc)
        if "statement timeout" in msg or "QueryCanceled" in type(exc).__name__:
            checkpoint("QUERYBUILD", "query timeout", vq_kind=vq.kind)
            return {
                "answer": (
                    "That query took too long to run. Try narrowing it — add a "
                    "date range or a tighter filter — and I'll try again."
                ),
                "source": "query_builder_timeout",
                "session_id": session_id,
            }
        logger.error("query execution failed: %s", exc, exc_info=True)
        checkpoint("QUERYBUILD", "execution error", error=msg[:200])
        return {"answer": _UNSAFE_MSG, "source": "query_builder_exec_error", "session_id": session_id}

    shaped = shape_rows(rows)
    try:
        answer = await asyncio.to_thread(_phrase_sync, question, vq, count, shaped)
    except Exception as exc:
        logger.warning("phrasing call failed, returning raw summary: %s", exc)
        answer = f"{count} row(s) returned: {json.dumps(shaped[:20], default=str)}"

    source = "query_builder_procedure" if vq.kind == "procedure" else "query_builder_dynamic"
    checkpoint("QUERYBUILD", "flow done", source=source, rows=count)
    return {"answer": answer, "source": source, "session_id": session_id}
