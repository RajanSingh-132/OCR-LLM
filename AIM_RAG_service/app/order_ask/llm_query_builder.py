"""
LLM query builder — the single Claude call in the dynamic query chain.

Input : the user's question (+ light session context for follow-ups)
Output: ONE JSON payload, nothing else. Never prose, never SQL.

The payload is NOT trusted — it always goes straight to
`payload_validator.validate_payload()` next. If Claude returns something
that isn't valid JSON, or a shape the validator rejects, the caller
returns a safe "couldn't process that" reply. There is no retry loop.

Payload schema is documented in full in `payload_validator.py`; the
system prompt below mirrors it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from app.embedding_client import get_anthropic_llm
from app.order_ask import query_allowlist as qa
from app.order_ask.checkpoint import checkpoint

logger = logging.getLogger("order_ask.llm_query_builder")

_TRNORDER_COLUMNS = ", ".join(qa.ALLOWED_COLUMNS["trnorder"])

_SYSTEM_PROMPT = f"""You convert a freight-order question into ONE JSON query payload.
Output JSON only — no prose, no markdown fence, no explanation.

This system is STRICTLY READ-ONLY. It can only look up and summarise existing
order data. If the question asks to create, add, update, edit, change, set,
delete, remove, cancel, assign, dispatch, approve, book, or otherwise MODIFY
anything, return {{"kind":"unsupported","reason":"write operation not permitted"}}.
Never emit a payload that implies a change.

Today's date is used for "last N days" style questions; you do not need it —
emit "last_n_days" and the server computes the window.

There are two execution paths. Choose exactly one via "kind".

============================================================
kind = "procedure"   (preferred whenever it fits)
============================================================
"procedure": one of
  - "regular"  : ONE order by id/number, or a full-detail lookup of a specific order
  - "summary"  : a short readable LIST of orders matching filters
                 (order no, status, customer, pickup/delivery, date, freight, notes)
  - "count"    : "how many orders ..."  (optionally grouped)
  - "freight"  : "total freight / revenue ..."  (optionally grouped)
  Prefer "summary" over "regular" for "show / list / which orders ..." questions;
  use "regular" only when the user names one specific order or wants every field.

Regular-only keys:
  "order_id":     <int>     direct lookup of one order by its numeric id
  "order_number": "<str>"   partial match on order number
  "customer_code":"<str>"
  "limit":        <int>     max rows (hard cap 500)

Summary-only keys:
  "limit": <int>            max rows returned (default 25, hard cap 500)

Count/Freight-only keys:
  "group_by": one of "day" | "week" | "status" | "delivery_city" | "pickup_city" | "country"
  "top_n":    <int>         keep only the N biggest groups

Shared semantic filters (procedure path only):
  "date_field": "created" | "order" | "pickup" | "delivery"   (default "created")
  "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"]                   inclusive
  "last_n_days": <int>                                        (mutually exclusive with date_range)
  "order_status": "<exact status word>"   e.g. "Delivered", "In-Transit", "Confirmed", "Cancelled"
  "pickup_country": "<2-3 letter code>"   e.g. "CA", "US"
  "delivery_country": "<2-3 letter code>"

============================================================
kind = "dynamic"   (only when no procedure fits)
============================================================
Single table only: "table": "trnorder".
Columns you may select / filter / group on (trnorder):
  {_TRNORDER_COLUMNS}
Keys:
  "select_fields": ["<col>", ...]            required unless "aggregate" is set
  "dynamic_filters": [{{"field":"<col>", "op":"<op>", "value": <v>}}]
       ops: = != > >= < <= between in ilike
       between value = [lo, hi]; in value = [..]; ilike value = "%pattern%"
       dates as "YYYY-MM-DD" strings
  "group_by_column": "<col>"                 with "aggregate"
  "aggregate": "count" | "sum" | "avg" | "min" | "max"
  "aggregate_field": "<numeric col>"         required for sum/avg/min/max
  "limit": <int>                             hard cap 500
Do NOT put date_range / country / order_status / last_n_days on a dynamic query —
spell those out as dynamic_filters on real trnorder columns
(createdon, orderdate for dates; there is no country/status column on trnorder —
if the question needs those, use a procedure instead).

============================================================
kind = "unsupported"
============================================================
Use for greetings, chit-chat, "why was order X late", document-text questions,
or anything the two paths above cannot express. Shape:
  {{"kind": "unsupported", "reason": "<short>"}}

============================================================
EXAMPLES
============================================================
Q: how many orders were created between Aug 1 and Aug 15 2026?
{{"kind":"procedure","procedure":"count","date_field":"created","date_range":["2026-08-01","2026-08-15"]}}

Q: total freight by delivery city for orders delivered in the last 30 days, top 10
{{"kind":"procedure","procedure":"freight","group_by":"delivery_city","date_field":"delivery","last_n_days":30,"top_n":10}}

Q: how many delivered orders went to Canada in July 2026?
{{"kind":"procedure","procedure":"count","order_status":"Delivered","delivery_country":"CA","date_field":"created","date_range":["2026-07-01","2026-07-31"]}}

Q: show me order 57060
{{"kind":"procedure","procedure":"regular","order_id":57060}}

Q: list the 20 most recent orders for customer ACME
{{"kind":"procedure","procedure":"regular","customer_code":"ACME","limit":20}}

Q: show me delivered orders that went to Canada last month
{{"kind":"procedure","procedure":"summary","order_status":"Delivered","delivery_country":"CA","date_field":"delivery","last_n_days":30,"limit":25}}

Q: which orders are in transit right now?
{{"kind":"procedure","procedure":"summary","order_status":"In-Transit"}}

Q: which orders have a total amount over 50000?
{{"kind":"dynamic","table":"trnorder","select_fields":["ordernumber","totalamount","orderdate"],"dynamic_filters":[{{"field":"totalamount","op":">","value":50000}}],"limit":50}}

Q: average offered amount on orders ordered in 2026
{{"kind":"dynamic","table":"trnorder","aggregate":"avg","aggregate_field":"offeredamount","dynamic_filters":[{{"field":"orderdate","op":"between","value":["2026-01-01","2026-12-31"]}}]}}

Q: hello there
{{"kind":"unsupported","reason":"greeting"}}

Q: mark order 57060 as delivered
{{"kind":"unsupported","reason":"write operation not permitted"}}

Q: add a note to order 12345
{{"kind":"unsupported","reason":"write operation not permitted"}}
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _call_llm(question: str, history: str) -> str:
    llm = get_anthropic_llm()
    prompt = (
        _SYSTEM_PROMPT
        + "\n\nConversation so far:\n"
        + (history or "(no prior turns)")
        + f"\n\nQuestion: {question}\nJSON:"
    )
    raw = llm.invoke(prompt)
    return raw.content if hasattr(raw, "content") else str(raw)


async def build_query_payload(
    question: str, session_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One Claude call. Returns the raw (untrusted) payload dict, or an
    {"kind": "unsupported", ...} payload on parse failure."""
    history = ""
    if session_context:
        history = str(session_context.get("history") or "")

    try:
        text = await asyncio.to_thread(_call_llm, question, history)
    except Exception as exc:  # LLM/network failure — safe fallback, no retry
        logger.warning("llm_query_builder call failed: %s", exc)
        return {"kind": "unsupported", "reason": f"llm_error: {exc}"}

    payload = _extract_json(text)
    if payload is None:
        checkpoint("QUERYBUILD", "LLM output not JSON", raw=text[:200])
        return {"kind": "unsupported", "reason": "unparseable_llm_output"}

    checkpoint("QUERYBUILD", "LLM payload", payload=payload)
    return payload
