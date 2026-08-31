"""
LLM query planner for Avaal orders — the primary routing path for
``/api/v1/orders/ask`` (the regex engine in ``app/domains/rules/orders.py`` +
``app/order_ask/analytics.py`` stays wired as an automatic fallback).

One Claude call reads the question + conversation history + a live sampled
schema of ``Avaal_order`` and returns a strict JSON ``QueryPlan``:

    task      lookup | list | aggregate | compare | conversation | greeting | unsupported
    filters   operator-DSL list  [{"field","op","value"}]
    aggregate {operation, group_by, metrics, distinct_field}   (task=aggregate)
    sort/limit/response_style

``_validate_plan`` checks every field against the sampled schema and every op
against the allow-list. ``_build_match`` turns the validated filters into a safe
Mongo ``$match`` (numeric ranges via ``$expr`` + ``$convert``; geo fields routed
through the tested address-regex helpers). ``execute_query_plan`` runs it and
returns the same payload shape as ``app/order_ask/tools.py`` ``execute_tools``.

Returns ``None`` (→ regex fallback) when disabled, on any error, on an invalid
plan, or for task greeting / conversation / unsupported.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_anthropic_llm
from app.order_ask.checkpoint import checkpoint
from app.order_ask.dynamic_analytics import (
    AGG_TIMEOUT_MS,
    _build_pipeline,
    _numeric_expr,
    _schema_for_prompt,
    _shape_result,
    _validate_spec,
    format_dynamic_analytics_for_context,
    get_orders_schema,
)
from app.order_ask.rag_retrieval import (
    _base_order_match,
    extract_order_token,
    find_order_by_id_or_number,
    format_order_doc_for_context,
    format_order_list_for_context,
    search_orders,
)
from app.tenants.router import get_orders_collection

PLANNER_ENABLED = os.environ.get(
    "AVAAL_QUERY_PLANNER", "1"
).strip().lower() not in ("0", "false", "no", "off")

MAX_FILTERS = 12

_TASKS = frozenset(
    {"lookup", "list", "aggregate", "compare", "conversation", "greeting", "unsupported"}
)

# op -> which field kinds it may target
_STRING_OPS = frozenset({"eq", "ne", "in", "nin", "contains", "starts_with", "exists"})
_NUMERIC_OPS = frozenset({"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "exists"})
_DATE_OPS = frozenset(
    {
        "eq",
        "ne",
        "date_eq",
        "date_gte",
        "date_lte",
        "date_gt",
        "date_lt",
        "last_days",
        "exists",
    }
)
_ALL_OPS = _STRING_OPS | _NUMERIC_OPS | _DATE_OPS

# Virtual fields resolved through the address-string regex helpers in
# rag_retrieval._base_order_match (not real top-level Mongo fields).
_VIRTUAL_GEO = frozenset(
    {
        "state",
        "city",
        "country",
        "pin",
        "address",
        "location",
        "pickup_location",
        "delivery_location",
        "location_side",
    }
)

_DATE_FIELD_RE = re.compile(r"date$", re.I)


# ------------------------------------------------------------------- plan
@dataclass
class QueryPlan:
    task: str
    record_tokens: List[str] = dc_field(default_factory=list)
    filters: List[Dict[str, Any]] = dc_field(default_factory=list)
    aggregate: Optional[Dict[str, Any]] = None
    sort: Optional[Dict[str, str]] = None
    limit: int = 15
    response_style: str = "medium"
    reason: str = ""

    def to_intent_info(self) -> Dict[str, Any]:
        intent = {
            "lookup": "order_lookup",
            "list": "list_filter",
            "aggregate": "analytics",
            "compare": "compare",
        }.get(self.task, "open_qa")
        max_tokens = {"short": 150, "medium": 500, "detailed": 1200}.get(
            self.response_style, 500
        )
        if self.task == "list":
            max_tokens = max(max_tokens, 700)
        return {
            "intent": intent,
            "response_style": self.response_style,
            "max_tokens_hint": max_tokens,
            "retrieve_k": 0,
            "needs_exact_order": self.task in ("lookup", "compare"),
            "needs_analytics": self.task == "aggregate",
            "reason": f"planner:{self.reason or self.task}",
        }

    def to_entities(self, **_ignored: Any) -> Dict[str, Any]:
        """Minimal entity dict for session stickiness + the response payload."""
        ent: Dict[str, Any] = {}
        if self.record_tokens:
            ent["order_token"] = self.record_tokens[0]
            ent["record_token"] = self.record_tokens[0]
        for f in self.filters:
            if f.get("op") == "eq" and isinstance(f.get("value"), (str, int, float)):
                ent.setdefault(f["field"], f["value"])
        if self.limit:
            ent["limit"] = self.limit
        if self.task == "aggregate":
            ent["analytics"] = "dynamic"
        if self.sort and self.sort.get("key"):
            ent["sort_by"] = self.sort["key"]
            ent["ascending"] = self.sort.get("dir") == "asc"
        return ent


# ------------------------------------------------------------------- LLM
_PLANNER_PROMPT = """You are a query planner for the Avaal_order MongoDB collection (freight orders).
Convert the user's question into a STRICT JSON plan. Output JSON only — no prose, no code fence.

Today (UTC): {today}

{schema}

Conversation so far:
{history}

Plan shape:
{{
  "task": "lookup" | "list" | "aggregate" | "compare" | "conversation" | "greeting" | "unsupported",
  "record_tokens": ["MRP12345", ...],
  "filters": [
    {{"field": "<field>", "op": "<op>", "value": <scalar|list>}}
  ],
  "aggregate": {{
    "operation": "group" | "metric" | "count" | "distinct_count",
    "group_by": ["<groupable field>", ...],
    "metrics": [{{"fn": "sum|avg|min|max|count", "field": "<numeric field>"}}],
    "distinct_field": "<field>"
  }},
  "sort": {{"key": "<field or metric key like avg_totalfreight>", "dir": "asc|desc"}},
  "limit": <int 1-100>,
  "response_style": "short" | "medium" | "detailed",
  "reason": "<short>"
}}

Tasks:
- lookup    : user wants ONE specific order (they gave a number/id) -> set record_tokens.
- compare   : user compares 2 specific orders -> record_tokens has 2.
- list      : user wants a filtered list of orders -> filters (+ optional sort/limit).
- aggregate : counts / totals / averages / group-by / distinct -> aggregate block.
- conversation : freeform question needing document context ("why was it delayed") -> planner cannot answer; leave rest empty.
- greeting  : hi / thanks / smalltalk.
- unsupported : cannot be mapped.

Filter ops:
  eq, ne, in, nin            any field
  contains, starts_with      text fields only (case-insensitive substring / prefix)
  gt, gte, lt, lte           numeric fields only
  date_eq, date_gte, date_lte, date_gt, date_lt   date fields (value = "YYYY-MM-DD", based on Today above)
  last_days                  date fields, value = integer N -> on/after (Today - N days). Use for "last 7 days", "past week", "last month" (30).
  exists                     value true/false

Date field for orders is "orderdate" unless the user says pickup / delivery.

Field notes:
- orderstatus = transport lifecycle (Quoted, Confirmed, Dispatched, In-Transit, Delivered, Cancelled...). Plain "order status" / "status" means THIS field.
- accountingstatus = invoicing / payment (Invoiced, PartiallyPaid, Paid, Restricted).
- outstatus = outsourcing (Open, Planned, Assigned, ...).

Rules:
- Use ONLY field names shown above, or these virtual location fields: state, city, country, pin, address, location, pickup_location, delivery_location, location_side (value pickup|delivery|both).
- Numeric ops only on fields tagged numeric. contains/starts_with only on text fields.
- group_by only on fields tagged groupable. metric key = fn + "_" + field, or "count".
- group_by may hold 2 fields for a two-dimensional breakdown
  (e.g. "order status and accounting status wise" -> group_by ["orderstatus","accountingstatus"]).
- Do not invent filters the user did not ask for.
- Prefer "aggregate" for "how many / total / average / per / by / wise / distinct".

Question: {question}
JSON:"""


def _plan_llm(question: str, history: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    llm = get_anthropic_llm()
    chain = PromptTemplate.from_template(_PLANNER_PROMPT) | llm
    raw = chain.invoke(
        {
            "question": question,
            "history": history or "(no prior turns)",
            "schema": _schema_for_prompt(schema),
            "today": datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)"),
        }
    )
    text = raw.content if hasattr(raw, "content") else str(raw)
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    match = re.search(r"\{.*\}", text.strip(), re.S)
    if not match:
        return None
    return json.loads(match.group(0))


# ------------------------------------------------------------------- validate
def _field_kind(name: str, schema: Dict[str, Any]) -> Optional[str]:
    if name in _VIRTUAL_GEO:
        return "geo"
    info = schema.get("fields", {}).get(name)
    if info is None:
        return None
    if info.get("numeric"):
        return "numeric"
    if _DATE_FIELD_RE.search(name):
        return "date"
    return "text"


def _valid_filter(f: Any, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(f, dict):
        return None
    name = f.get("field")
    op = str(f.get("op") or "").lower()
    value = f.get("value")
    if not isinstance(name, str) or op not in _ALL_OPS:
        return None

    kind = _field_kind(name, schema)
    if kind is None:
        return None
    if kind == "geo":
        if op not in ("eq", "contains", "in"):
            op = "eq"
        if name == "location_side":
            value = str(value).lower()
            if value not in ("pickup", "delivery", "both"):
                return None
        if value in (None, "", [], {}):
            return None
        return {"field": name, "op": op, "value": value}

    if op == "exists":
        return {"field": name, "op": "exists", "value": bool(value)}
    if value in (None, "", [], {}):
        return None
    if kind == "numeric" and op not in _NUMERIC_OPS:
        return None
    if kind == "date" and op not in _DATE_OPS:
        return None
    if kind == "text" and op not in _STRING_OPS:
        return None
    if op in ("in", "nin") and not isinstance(value, list):
        value = [value]
    if op in ("gt", "gte", "lt", "lte"):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    if op == "last_days":
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
    return {"field": name, "op": op, "value": value}


def _validate_plan(
    raw: Any, schema: Dict[str, Any], question: str
) -> Optional[QueryPlan]:
    if not isinstance(raw, dict):
        return None
    task = str(raw.get("task") or "").lower()
    if task not in _TASKS:
        return None

    style = str(raw.get("response_style") or "medium").lower()
    if style not in ("short", "medium", "detailed"):
        style = "medium"

    tokens = [
        str(t).strip()
        for t in (raw.get("record_tokens") or [])
        if isinstance(t, (str, int)) and str(t).strip()
    ]
    if task in ("lookup", "compare") and not tokens:
        tok = extract_order_token(question)
        if tok:
            tokens = [tok]

    filters: List[Dict[str, Any]] = []
    for f in (raw.get("filters") or [])[:MAX_FILTERS]:
        clean = _valid_filter(f, schema)
        if clean is not None:
            filters.append(clean)

    try:
        limit = max(1, min(100, int(raw.get("limit") or 15)))
    except (TypeError, ValueError):
        limit = 15

    sort = None
    if isinstance(raw.get("sort"), dict) and isinstance(raw["sort"].get("key"), str):
        sort = {
            "key": raw["sort"]["key"],
            "dir": "asc" if str(raw["sort"].get("dir")).lower() == "asc" else "desc",
        }

    aggregate = None
    if task == "aggregate":
        agg_raw = dict(raw.get("aggregate") or {})
        agg_raw.setdefault("limit", limit)
        agg_raw.setdefault("sort", sort)
        aggregate = _validate_spec(agg_raw, schema)
        if aggregate is None:
            # aggregate intent but unusable spec — let the regex engine try
            return None

    if task == "lookup" and not tokens:
        return None

    return QueryPlan(
        task=task,
        record_tokens=tokens[:2],
        filters=filters,
        aggregate=aggregate,
        sort=sort,
        limit=limit,
        response_style=style,
        reason=str(raw.get("reason") or "")[:80],
    )


def run_query_planner(
    question: str, *, history: str = "(no prior turns)"
) -> Optional[QueryPlan]:
    if not PLANNER_ENABLED:
        return None
    q = (question or "").strip()
    if not q:
        return None
    schema = get_orders_schema()
    if not schema.get("fields"):
        return None

    raw = _plan_llm(q, history, schema)
    plan = _validate_plan(raw, schema, q)
    checkpoint(
        "PLANNER",
        "plan",
        raw=raw,
        task=(plan.task if plan else None),
        filters=(len(plan.filters) if plan else None),
    )
    if plan is None or plan.task in ("greeting", "conversation", "unsupported"):
        return None
    return plan


# ------------------------------------------------------------------- match builder
def _geo_entity_filters(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in filters:
        if f["field"] in _VIRTUAL_GEO:
            out[f["field"]] = f["value"]
    return out


def _op_to_mongo(op: str, value: Any, field: str, kind: str):
    """Return a per-field condition dict, or {"__expr__": ...} for a numeric range."""
    if op == "exists":
        return (
            {"$nin": [None, ""]}
            if value
            else {"$in": [None, ""]}
        )
    if op == "eq":
        if isinstance(value, str):
            return {"$regex": f"^{re.escape(value)}$", "$options": "i"}
        return value
    if op == "ne":
        return {"$ne": value}
    if op == "in":
        return {"$in": value if isinstance(value, list) else [value]}
    if op == "nin":
        return {"$nin": value if isinstance(value, list) else [value]}
    if op == "contains":
        return {"$regex": re.escape(str(value)), "$options": "i"}
    if op == "starts_with":
        return {"$regex": f"^{re.escape(str(value))}", "$options": "i"}
    if op == "date_eq":
        return {"$regex": f"^{re.escape(str(value))}", "$options": "i"}
    if op in ("date_gte", "date_lte", "date_gt", "date_lt"):
        mop = {"date_gte": "$gte", "date_lte": "$lte", "date_gt": "$gt", "date_lt": "$lt"}[op]
        return {mop: str(value)}
    if op == "last_days":
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(value))
        ).strftime("%Y-%m-%d")
        return {"$gte": cutoff}
    if op in ("gt", "gte", "lt", "lte"):
        mop = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}[op]
        if kind == "numeric":
            return {"__expr__": {mop: [_numeric_expr(field), float(value)]}}
        return {mop: value}
    return None


def _build_match(
    filters: List[Dict[str, Any]], schema: Dict[str, Any]
) -> Dict[str, Any]:
    geo = _geo_entity_filters(filters)
    match: Dict[str, Any] = _base_order_match(geo) if geo else _base_order_match(None)

    and_parts: List[Dict[str, Any]] = list(match.pop("$and", []))
    field_conds: Dict[str, List[Any]] = {}

    for f in filters:
        name, op, value = f["field"], f["op"], f.get("value")
        if name in _VIRTUAL_GEO:
            continue
        kind = _field_kind(name, schema)
        if kind is None:
            continue
        cond = _op_to_mongo(op, value, name, kind)
        if cond is None:
            continue
        if isinstance(cond, dict) and "__expr__" in cond:
            and_parts.append({"$expr": cond["__expr__"]})
        else:
            field_conds.setdefault(name, []).append(cond)

    for name, conds in field_conds.items():
        if name in match and isinstance(match[name], dict):
            and_parts.append({name: match[name]})
            del match[name]
        if len(conds) == 1 and name not in match:
            match[name] = conds[0]
        else:
            and_parts.extend({name: c} for c in conds)

    if and_parts:
        match["$and"] = and_parts
    return match


def _filters_summary(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        f["field"]: (f"{f['op']} {f['value']}" if f["op"] != "eq" else f["value"])
        for f in filters
    }


def _list_orders(plan: QueryPlan, schema: Dict[str, Any]) -> Dict[str, Any]:
    match = _build_match(plan.filters, schema)
    sort_by = (plan.sort or {}).get("key") or "orderid"
    ascending = (plan.sort or {}).get("dir") == "asc"
    payload = search_orders(
        match=match, limit=plan.limit, sort_by=sort_by, ascending=ascending
    )
    payload["filters"] = _filters_summary(plan.filters)
    checkpoint(
        "PLANNER",
        "list executed",
        total=payload.get("total_matching"),
        returned=payload.get("returned"),
    )
    return payload


# ------------------------------------------------------------------- execute
def execute_query_plan(plan: QueryPlan, *, question: str) -> Dict[str, Any]:
    """Run a validated plan. Returns the same shape as tools.execute_tools."""
    context_blocks: List[str] = []
    matches: List[Dict[str, Any]] = []
    calc_payload = None
    list_payload = None
    analytics_payload = None
    tools_run: List[str] = []
    active_token = plan.record_tokens[0] if plan.record_tokens else ""

    schema = get_orders_schema()

    if plan.task == "lookup":
        token = plan.record_tokens[0] if plan.record_tokens else ""
        doc = find_order_by_id_or_number(token) if token else None
        if doc:
            context_blocks.append(
                "EXACT ORDER RECORD:\n" + format_order_doc_for_context(doc)
            )
            matches.append(
                {
                    "orderid": doc.get("orderid"),
                    "ordernumber": doc.get("ordernumber"),
                    "match_type": "exact",
                }
            )
        else:
            context_blocks.append(
                f"EXACT ORDER RECORD: not found for token={token}"
            )
        tools_run.append("get_record")

    elif plan.task == "compare":
        parts: List[str] = []
        for token in plan.record_tokens[:2]:
            doc = find_order_by_id_or_number(token)
            if doc:
                parts.append(
                    f"ORDER {token}:\n"
                    + format_order_doc_for_context(doc, max_fields=40)
                )
                matches.append(
                    {"ordernumber": doc.get("ordernumber"), "match_type": "compare"}
                )
            else:
                parts.append(f"ORDER {token}: not found")
        context_blocks.append("COMPARE ORDERS:\n" + "\n\n".join(parts))
        tools_run.append("compare_records")

    elif plan.task == "aggregate" and plan.aggregate is not None:
        match = _build_match(plan.filters, schema)
        pipeline = _build_pipeline(plan.aggregate, match)
        rows = list(
            get_orders_collection().aggregate(
                pipeline, maxTimeMS=AGG_TIMEOUT_MS, allowDiskUse=False
            )
        )
        analytics_payload = _shape_result(
            plan.aggregate, rows, _filters_summary(plan.filters), question
        )
        context_blocks.append(
            format_dynamic_analytics_for_context(analytics_payload)
        )
        checkpoint(
            "PLANNER",
            "aggregate executed",
            operation=plan.aggregate.get("operation"),
            rows=len(rows),
        )
        tools_run.append("run_analytics")

    else:  # list (default)
        list_payload = _list_orders(plan, schema)
        context_blocks.append(format_order_list_for_context(list_payload))
        for row in list_payload.get("orders") or []:
            matches.append(
                {
                    "orderid": row.get("orderid"),
                    "ordernumber": row.get("ordernumber"),
                    "match_type": "filter",
                }
            )
        tools_run.append("search_records")

    return {
        "context_blocks": context_blocks,
        "matches": matches,
        "calculation": calc_payload,
        "analytics": analytics_payload,
        "list_result": list_payload,
        "tools_run": tools_run,
        "active_order_token": active_token,
        "domain": "orders",
    }
