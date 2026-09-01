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
from datetime import date, datetime, timedelta, timezone
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
    date_format_for,
    format_dynamic_analytics_for_context,
    get_orders_schema,
    is_date_field,
    resolve_field,
)
from app.order_ask.invoice_query_planner import (
    _iso_date_expr,
    _iso_string_condition,
    _target_date_expr,
    _us_date_condition,
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

MAX_FILTERS = 14

_TASKS = frozenset(
    {
        "lookup", "list", "aggregate", "compare", "percentage",
        "conversation", "greeting", "unsupported",
    }
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
        "date_range",
        "last_days",
        "period",
        "exists",
    }
)
_ALL_OPS = _STRING_OPS | _NUMERIC_OPS | _DATE_OPS

_PERIODS = frozenset(
    {
        "today", "yesterday", "this_week", "last_week",
        "this_month", "last_month", "this_year", "last_year",
    }
)

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
    # task=compare (segments) / task=percentage
    segments: List[Dict[str, Any]] = dc_field(default_factory=list)
    numerator: List[Dict[str, Any]] = dc_field(default_factory=list)
    pct_of: str = "orders"
    metric: Optional[Dict[str, Any]] = None

    def to_intent_info(self) -> Dict[str, Any]:
        intent = {
            "lookup": "order_lookup",
            "list": "list_filter",
            "aggregate": "analytics",
            "compare": "compare" if self.record_tokens else "analytics",
            "percentage": "analytics",
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
            "needs_exact_order": self.task == "lookup"
            or (self.task == "compare" and bool(self.record_tokens)),
            "needs_analytics": self.task in ("aggregate", "percentage")
            or (self.task == "compare" and not self.record_tokens),
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
Convert the user's question into ONE strict JSON plan. Output JSON only — no prose, no code fence.

Today (UTC): {today}

{schema}

Conversation so far:
{history}

PLAN SHAPE (include only the keys the task needs):
{{
  "task": "lookup" | "list" | "aggregate" | "compare" | "percentage" | "conversation" | "greeting" | "unsupported",
  "record_tokens": ["MRP12345", ...],
  "filters": [ {{"field": "<field>", "op": "<op>", "value": <scalar|list>}} ],
  "aggregate": {{
    "operation": "count" | "metric" | "group" | "distinct_count",
    "group_by": ["<groupable field>", ...],
    "metrics": [{{"fn": "sum|avg|min|max|count", "field": "<numeric field>"}}],
    "distinct_field": "<field>",
    "date_bucket": {{"field": "orderdate", "unit": "day|week|month"}},
    "having": [{{"key": "orders|count|sum_<field>|avg_<field>", "op": "gt|gte|lt|lte", "value": <num>}}]
  }},
  "segments": [ {{"label": "August", "filters": [ ... ]}}, {{"label": "July", "filters": [ ... ]}} ],
  "metric": {{"fn": "count|sum|avg", "field": "<numeric field>"}},
  "numerator": [ {{"field": "...", "op": "...", "value": ...}} ],
  "pct_of": "orders" | "<numeric field>",
  "sort": {{"key": "<field / metric key like avg_totalfreight / count / orders / period>", "dir": "asc|desc"}},
  "limit": <int 1-100>,
  "response_style": "short|medium|detailed",
  "reason": "<short>"
}}

TASKS
- lookup     : user names ONE order (number/id) -> record_tokens.
- compare(records) : compares 2 named orders -> record_tokens has 2.
- compare(segments): compares metrics across time windows / places ("August vs July", "Canada vs US",
                     "Aug 1-15 vs Aug 16-31") -> segments[] each with its own filters, plus one metric.
- list       : wants the actual orders -> filters (+ sort/limit). "show / list / find / top N orders".
- aggregate  : how many / total / average / min / max / per / by / wise / distinct / daily / weekly.
- percentage : "what % of ..." -> numerator (the subset) + pct_of ("orders" or a numeric field).
                filters = the population; numerator = the extra condition.
- conversation: needs document text ("why delayed") - leave the rest empty.
- greeting / unsupported.

FILTER OPS
  eq, ne, in, nin           any field         contains, starts_with   text only
  gt, gte, lt, lte          numeric only      exists                  value true/false
  date_gte/date_lte/date_gt/date_lt/date_eq   date field, value "YYYY-MM-DD"
  date_range                date field, value ["YYYY-MM-DD","YYYY-MM-DD"] (inclusive) - use for "between Aug 1 and Aug 15"
  period                    date field, value one of: today, yesterday, this_week, last_week, this_month, last_month, this_year, last_year
  last_days                 date field, value integer N - "last 7 days", "past 30 days"
Date ops work on ANY field tagged date(...) in the schema above. Pick the one the user means:
"ordered / placed / order date" -> orderdate (this is the default when none is named);
"created / entered in the system" -> createdon; "last updated / modified" -> modifiedon;
"picked up / pickup" -> pickupdate; "delivered / drop / delivery" -> deliverydate;
"enquiry" -> enquirydate; "quoted / quotation" -> quotationdate.
A month name like "August" with no year -> date_range for that whole month in the current year.

VOCAB (map the user's word to the real field/value)
- status words: pending = orderstatus in ["Quoted","Confirmed","Dispatched","Started","In-Transit","Partially Delivered"]
  (use op "nin" value ["Delivered","Cancelled","Rejected"]); "in transit" = orderstatus eq "In-Transit";
  delivered / cancelled / confirmed / quoted / dispatched = orderstatus eq that exact word.
  "delayed" / "late" = deliverydate date_lt Today AND orderstatus ne "Delivered".
- accounting: "not invoiced" = accountingstatus in [null,"NotInvoiced"] OR accountingtypebreakdown.notInvoiced gt 0
  (prefer: field "accountingstatus" op "in" value [null]); invoiced / paid = accountingstatus eq that.
- flags: "active" = isactive eq true; "archived" = isarchived eq true; "FTL" = loadtypelucode eq "FTL"; "LTL" = "LTL".
- money: revenue/gross -> grosstotalfreight; amount/price/value/cost -> totalfreight; freight -> totalfreight;
  rate -> freightratevalue; tax -> totaltaxamount; fuel -> fuelcharges. "C$50,000" -> 50000.
- other: client/buyer -> customername; carrier -> outcarriername; trip -> tripno; miles/km -> distance;
  PO -> pono; reference -> referno; salesman/agent -> salesmanname; commodity/product -> commodityname;
  "route" -> group_by ["pickuplocationname","deliverylocationname"]; "freight per mile" -> not directly supported,
  use metric avg on totalfreight and note distance separately, or mark unsupported.
- geo virtual fields: city, state, country, pin, location, pickup_location, delivery_location, location_side
  ("from X" -> pickup side, "to / going to X" -> delivery side). "Canadian" -> country eq "Canada".

RULES
- Use ONLY fields shown above (dotted nested allowed) or the geo virtual fields. Never invent one.
- Numeric fn/ops only on numeric fields. group_by only on groupable fields (max 2).
- date_bucket: for "daily / weekly / monthly" series. operation stays "group"; group_by may be [] (pure time series)
  or hold ONE other field. "field" may be any field tagged date(...) above.
- having: threshold AFTER grouping ("customers with more than 5 orders AND freight above 5000" ->
  group_by ["customername"], metrics [count, sum totalfreight], having [{{count>5}},{{sum_totalfreight>5000}}]).
- "top N X by Y": task aggregate, group_by [X], metric sum/avg on Y, sort desc, limit N.
- "highest-value orders" (individual orders, not a group) -> task list, sort by grosstotalfreight desc.
- Do NOT invent filters the user didn't ask for. Put the population filters in "filters", the compared
  dimension in "segments"/"numerator".
- If truly not expressible, task "unsupported".

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
    if info.get("date") or is_date_field(name, schema) or _DATE_FIELD_RE.search(name):
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

    # Fuzzy / alias resolution for misspelled or business-term field names
    if name not in _VIRTUAL_GEO and name not in schema.get("fields", {}):
        resolved = resolve_field(name, schema)
        if resolved:
            name = resolved

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
    if op == "date_range":
        if not (isinstance(value, list) and len(value) == 2):
            return None
        value = [str(value[0]).strip()[:10], str(value[1]).strip()[:10]]
        if not all(re.match(r"^\d{4}-\d{2}-\d{2}$", v) for v in value):
            return None
    if op in ("date_eq", "date_gte", "date_lte", "date_gt", "date_lt"):
        value = str(value).strip()[:10]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return None
    if op == "period":
        value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        if value not in _PERIODS:
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

    # ---- metric shared by compare/percentage ----
    def _valid_metric(raw_m: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_m, dict):
            return {"fn": "count", "field": None}
        fn = str(raw_m.get("fn") or "count").lower()
        if fn not in ("count", "sum", "avg", "min", "max"):
            fn = "count"
        if fn == "count":
            return {"fn": "count", "field": None}
        fld = resolve_field(raw_m.get("field"), schema)
        if not fld or not schema["fields"].get(fld, {}).get("numeric"):
            return {"fn": "count", "field": None}
        return {"fn": fn, "field": fld}

    segments: List[Dict[str, Any]] = []
    numerator: List[Dict[str, Any]] = []
    pct_of = "orders"
    metric: Optional[Dict[str, Any]] = None

    if task == "compare" and isinstance(raw.get("segments"), list) and raw["segments"]:
        for seg in raw["segments"][:4]:
            if not isinstance(seg, dict):
                continue
            sf = [
                c
                for c in (_valid_filter(f, schema) for f in (seg.get("filters") or []))
                if c is not None
            ]
            segments.append(
                {
                    "label": str(seg.get("label") or f"segment {len(segments) + 1}")[:40],
                    "filters": sf,
                }
            )
        segments = [s for s in segments if s["filters"]]
        if len(segments) < 2:
            return None
        metric = _valid_metric(raw.get("metric"))

    if task == "compare" and not segments and not tokens:
        return None  # nothing concrete to compare — let the regex engine try

    if task == "percentage":
        numerator = [
            c
            for c in (_valid_filter(f, schema) for f in (raw.get("numerator") or []))
            if c is not None
        ]
        if not numerator:
            return None
        of = raw.get("pct_of") or raw.get("of") or "orders"
        if str(of).lower() in ("orders", "order", "count", "records"):
            pct_of = "orders"
        else:
            r = resolve_field(of, schema)
            pct_of = r if (r and schema["fields"].get(r, {}).get("numeric")) else "orders"

    return QueryPlan(
        task=task,
        record_tokens=tokens[:2],
        filters=filters,
        aggregate=aggregate,
        sort=sort,
        limit=limit,
        response_style=style,
        reason=str(raw.get("reason") or "")[:80],
        segments=segments,
        numerator=numerator,
        pct_of=pct_of,
        metric=metric,
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


# ------------------------------------------------------------------- dates
def _month_add(d: date, n: int) -> date:
    total = (d.year * 12 + (d.month - 1)) + n
    return date(total // 12, total % 12 + 1, 1)


def _period_bounds(name: str) -> Optional[tuple]:
    """(start_inclusive, end_exclusive) ISO dates for a named period."""
    today = datetime.now(timezone.utc).date()
    if name == "today":
        s, e = today, today + timedelta(days=1)
    elif name == "yesterday":
        s, e = today - timedelta(days=1), today
    elif name == "this_week":
        s = today - timedelta(days=today.weekday())
        e = s + timedelta(days=7)
    elif name == "last_week":
        e = today - timedelta(days=today.weekday())
        s = e - timedelta(days=7)
    elif name == "this_month":
        s = today.replace(day=1)
        e = _month_add(s, 1)
    elif name == "last_month":
        e = today.replace(day=1)
        s = _month_add(e, -1)
    elif name == "this_year":
        s, e = date(today.year, 1, 1), date(today.year + 1, 1, 1)
    elif name == "last_year":
        s, e = date(today.year - 1, 1, 1), date(today.year, 1, 1)
    else:
        return None
    return s.isoformat(), e.isoformat()


def _plus_one_day(iso: str) -> str:
    try:
        return (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    except ValueError:
        return iso + "~"  # lexical upper-bound fallback


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
    if op == "date_range":
        start, end = value[0], value[1]
        return {"$gte": start, "$lt": _plus_one_day(end)}
    if op == "period":
        bounds = _period_bounds(value)
        if not bounds:
            return None
        return {"$gte": bounds[0], "$lt": bounds[1]}
    if op in ("gt", "gte", "lt", "lte"):
        mop = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}[op]
        if kind == "numeric":
            return {"__expr__": {mop: [_numeric_expr(field), float(value)]}}
        return {mop: value}
    return None


def _native_date_condition(field: str, op: str, value: Any):
    """{"$expr": ...} clause for a column stored as a real BSON date."""
    src = f"${field}"
    if op in ("date_gte", "date_gt", "date_lte", "date_lt"):
        mop = {
            "date_gte": "$gte", "date_gt": "$gt",
            "date_lte": "$lte", "date_lt": "$lt",
        }[op]
        return {"$expr": {mop: [src, _target_date_expr(value)]}}
    if op == "date_eq":
        return {
            "$expr": {
                "$and": [
                    {"$gte": [src, _target_date_expr(value)]},
                    {"$lt": [src, _target_date_expr(_plus_one_day(value))]},
                ]
            }
        }
    if op == "date_range":
        return {
            "$expr": {
                "$and": [
                    {"$gte": [src, _target_date_expr(value[0])]},
                    {"$lt": [src, _target_date_expr(_plus_one_day(value[1]))]},
                ]
            }
        }
    if op == "last_days":
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(value))
        ).strftime("%Y-%m-%d")
        return {"$expr": {"$gte": [src, _target_date_expr(cutoff)]}}
    if op == "period":
        bounds = _period_bounds(value)
        if not bounds:
            return None
        return {
            "$expr": {
                "$and": [
                    {"$gte": [src, _target_date_expr(bounds[0])]},
                    {"$lt": [src, _target_date_expr(bounds[1])]},
                ]
            }
        }
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

        if kind == "date":
            # eq / ne / exists keep exact-string semantics (any format).
            if op in ("eq", "ne", "exists"):
                cond = _op_to_mongo(op, value, name, "text")
                if cond is not None:
                    field_conds.setdefault(name, []).append(cond)
                continue
            fmt = date_format_for(name, schema)
            if fmt == "us":
                clause = _us_date_condition(name, op, value)
                if clause:
                    and_parts.append(clause)
                continue
            if fmt == "native":
                clause = _native_date_condition(name, op, value)
                if clause:
                    and_parts.append(clause)
                continue
            cond = _iso_string_condition(op, value)
            if cond is not None:
                field_conds.setdefault(name, []).append(cond)
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


def _date_sort_expr(field: str, fmt: str) -> Dict[str, Any]:
    """Aggregation expr that yields a real date for sorting a date column."""
    if fmt == "us":
        return _iso_date_expr(field)
    return {
        "$convert": {
            "input": {"$substrBytes": [{"$toString": f"${field}"}, 0, 10]},
            "to": "date",
            "onError": None,
            "onNull": None,
        }
    }


def _list_orders(plan: QueryPlan, schema: Dict[str, Any]) -> Dict[str, Any]:
    match = _build_match(plan.filters, schema)
    sort_by = (plan.sort or {}).get("key") or "orderid"
    ascending = (plan.sort or {}).get("dir") == "asc"

    resolved_sort = resolve_field(sort_by, schema) or sort_by
    sort_expr = None
    if _field_kind(resolved_sort, schema) == "date":
        sort_expr = _date_sort_expr(
            resolved_sort, date_format_for(resolved_sort, schema)
        )
        sort_by = resolved_sort

    payload = search_orders(
        match=match, limit=plan.limit, sort_by=sort_by, ascending=ascending,
        sort_expr=sort_expr,
    )
    payload["filters"] = _filters_summary(plan.filters)
    checkpoint(
        "PLANNER",
        "list executed",
        total=payload.get("total_matching"),
        returned=payload.get("returned"),
    )
    return payload


def _metric_value(match: Dict[str, Any], metric: Dict[str, Any]) -> float:
    """One-shot count / sum / avg / min / max over a match."""
    collection = get_orders_collection()
    fn = metric.get("fn") or "count"
    if fn == "count":
        return float(collection.count_documents(match))
    field = metric.get("field")
    if not field:
        return float(collection.count_documents(match))
    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": None, "v": {f"${fn}": _numeric_expr(field)}}},
            ],
            maxTimeMS=AGG_TIMEOUT_MS,
        )
    )
    return float(rows[0]["v"]) if rows and rows[0].get("v") is not None else 0.0


def _metric_label(metric: Dict[str, Any]) -> str:
    fn = metric.get("fn") or "count"
    return "order_count" if fn == "count" else f"{fn}_{metric.get('field')}"


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

    elif plan.task == "compare" and plan.segments:
        metric = plan.metric or {"fn": "count", "field": None}
        label = _metric_label(metric)
        seg_rows = []
        for seg in plan.segments:
            m = _build_match(plan.filters + seg["filters"], schema)
            seg_rows.append(
                {"segment": seg["label"], label: round(_metric_value(m, metric), 4)}
            )
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "dynamic_planner",
            "operation": "compare",
            "metric": label,
            "filters": _filters_summary(plan.filters),
            "rows": seg_rows,
        }
        context_blocks.append(
            format_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

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

    elif plan.task == "percentage":
        metric = (
            {"fn": "count", "field": None}
            if plan.pct_of == "orders"
            else {"fn": "sum", "field": plan.pct_of}
        )
        den_match = _build_match(plan.filters, schema)
        num_match = _build_match(plan.filters + plan.numerator, schema)
        den = _metric_value(den_match, metric)
        num = _metric_value(num_match, metric)
        pct = round((num / den * 100.0), 2) if den else 0.0
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "dynamic_planner",
            "operation": "percentage",
            "of": ("orders" if plan.pct_of == "orders" else f"sum {plan.pct_of}"),
            "numerator": round(num, 4),
            "denominator": round(den, 4),
            "percentage": pct,
            "numerator_filters": _filters_summary(plan.numerator),
            "filters": _filters_summary(plan.filters),
        }
        context_blocks.append(
            format_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

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
