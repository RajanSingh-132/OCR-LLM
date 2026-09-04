"""
LLM query planner for Avaal invoices — the primary routing path for the
``invoices`` domain of ``/api/v1/orders/ask``. The regex engine
(``app/domains/rules/invoices.py`` + ``app/order_ask/invoice_analytics.py``)
stays wired as an automatic fallback.

One Claude call reads the question + conversation history + a live sampled
schema of ``Avaal_invoice`` and returns a strict JSON ``InvoiceQueryPlan``:

    task      lookup | list | aggregate | compare | percentage | conversation
              | greeting | unsupported
    filters   operator-DSL list  [{"field","op","value"}]
    aggregate {operation, group_by, metrics, distinct_field, date_bucket, having}
    segments / numerator / metric / pct_of   (compare / percentage)
    sort / limit / response_style

``_validate_plan`` checks every field against the sampled schema and every op
against the allow-list. ``_build_invoice_match`` turns the validated filters
into a safe Mongo ``$match`` (numeric ranges via ``$expr`` + ``$convert``;
US-format ``InvoiceDate`` / ``DueDate`` via ``$dateFromString`` on the date
part; ISO ``createdon`` via lexical compare; geo via address-string regex).
``execute_invoice_query_plan`` runs it and returns the same payload shape as
``app/order_ask/tools.py`` ``execute_tools``.

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

from app.embedding_client import get_planner_llm
from app.order_ask.checkpoint import checkpoint
from app.order_ask.dynamic_analytics import (
    AGG_TIMEOUT_MS,
    _build_pipeline,
    _numeric_expr,
    resolve_field,
)
from app.order_ask.invoice_dynamic_analytics import (
    _INVOICE_FIELD_ALIASES,
    _INVOICE_ISO_DATE_FIELDS,
    _shape_invoice_result,
    format_invoice_dynamic_analytics_for_context,
    get_invoices_schema,
    invoice_schema_for_prompt,
    validate_invoice_spec,
)
from app.order_ask.invoice_analytics import _base_match as _invoice_base_match
from app.domains.lookup.invoices.lookup import extract_token as extract_invoice_token
from app.domains.retrieval import (
    format_record_doc_for_context,
    format_record_list_for_context,
    find_record_by_token,
)
from app.tenants.router import get_domain_collection

INVOICE_PLANNER_ENABLED = os.environ.get(
    "AVAAL_INVOICE_QUERY_PLANNER", "1"
).strip().lower() not in ("0", "false", "no", "off")

MAX_FILTERS = 14

_TASKS = frozenset(
    {
        "lookup", "list", "aggregate", "compare", "percentage",
        "conversation", "greeting", "unsupported",
    }
)

_STRING_OPS = frozenset({"eq", "ne", "in", "nin", "contains", "starts_with", "exists"})
_NUMERIC_OPS = frozenset({"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "exists"})
_DATE_OPS = frozenset(
    {
        "eq", "ne", "date_eq", "date_gte", "date_lte", "date_gt", "date_lt",
        "date_range", "last_days", "period", "exists",
    }
)
_ALL_OPS = _STRING_OPS | _NUMERIC_OPS | _DATE_OPS

_PERIODS = frozenset(
    {
        "today", "yesterday", "this_week", "last_week",
        "this_month", "last_month", "this_year", "last_year",
    }
)

# Virtual fields resolved through address-string regex (not real Mongo fields).
_VIRTUAL_GEO = frozenset(
    {
        "country", "state", "city", "destination", "location",
        "pickup_location", "delivery_location", "location_side",
    }
)

# US-format date strings ("2/16/2026 2:30:00 AM") — parsed via $dateFromString.
_US_DATE_FIELDS = frozenset(
    {"InvoiceDate", "DueDate", "PickupDate", "DeliveryDate"}
)
_DATE_FIELD_RE = re.compile(r"date$", re.I)


# ------------------------------------------------------------------- plan
@dataclass
class InvoiceQueryPlan:
    task: str
    record_tokens: List[str] = dc_field(default_factory=list)
    filters: List[Dict[str, Any]] = dc_field(default_factory=list)
    aggregate: Optional[Dict[str, Any]] = None
    sort: Optional[Dict[str, str]] = None
    limit: int = 15
    response_style: str = "medium"
    reason: str = ""
    segments: List[Dict[str, Any]] = dc_field(default_factory=list)
    numerator: List[Dict[str, Any]] = dc_field(default_factory=list)
    pct_of: str = "invoices"
    metric: Optional[Dict[str, Any]] = None

    def to_intent_info(self) -> Dict[str, Any]:
        intent = {
            "lookup": "invoice_lookup",
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
            "reason": f"invoice_planner:{self.reason or self.task}",
        }

    def to_entities(self, **_ignored: Any) -> Dict[str, Any]:
        ent: Dict[str, Any] = {}
        if self.record_tokens:
            ent["record_token"] = self.record_tokens[0]
            ent["order_token"] = self.record_tokens[0]
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
_PLANNER_PROMPT = """You are a query planner for the Avaal_invoice MongoDB collection (freight invoices).
Convert the user's question into ONE strict JSON plan. Output JSON only — no prose, no code fence.

Today (UTC): {today}

{schema}

Conversation so far:
{history}

PLAN SHAPE (include only the keys the task needs):
{{
  "task": "lookup" | "list" | "aggregate" | "compare" | "percentage" | "conversation" | "greeting" | "unsupported",
  "record_tokens": ["MR3932", ...],
  "filters": [ {{"field": "<field>", "op": "<op>", "value": <scalar|list>}} ],
  "aggregate": {{
    "operation": "count" | "metric" | "group" | "distinct_count",
    "group_by": ["<groupable field>", ...],
    "metrics": [{{"fn": "sum|avg|min|max|count", "field": "<numeric field>"}}],
    "distinct_field": "<field>",
    "date_bucket": {{"field": "createdon", "unit": "day|week|month"}},
    "having": [{{"key": "invoices|count|sum_<field>|avg_<field>", "op": "gt|gte|lt|lte", "value": <num>}}]
  }},
  "segments": [ {{"label": "August", "filters": [ ... ]}}, {{"label": "July", "filters": [ ... ]}} ],
  "metric": {{"fn": "count|sum|avg", "field": "<numeric field>"}},
  "numerator": [ {{"field": "...", "op": "...", "value": ...}} ],
  "pct_of": "invoices" | "<numeric field>",
  "sort": {{"key": "<field / metric key like sum_totalamount / count / invoices / period>", "dir": "asc|desc"}},
  "limit": <int 1-100>,
  "response_style": "short|medium|detailed",
  "reason": "<short>"
}}

TASKS
- lookup     : user names ONE invoice (number/id) -> record_tokens.
- compare(records) : compares 2 named invoices -> record_tokens has 2.
- compare(segments): compares a metric across time windows / places / customers
                     ("August vs July", "Canada vs US", "CAD vs USD") ->
                     segments[] each with its own filters, plus one metric.
- list       : wants the actual invoices -> filters (+ sort/limit).
               "show / list / find / top N invoices", "invoices between $500 and $2000".
- aggregate  : how many / total / sum / average / min / max / per / by / wise / distinct
               / daily / weekly / monthly. "highest / lowest invoice" -> operation metric
               (min/max) OR list sorted; "top N customers by X" -> group + sort + limit.
- percentage : "what % of ..." -> numerator (the subset) + pct_of ("invoices" or a numeric field).
- conversation / greeting / unsupported.

FILTER OPS
  eq, ne, in, nin           any field         contains, starts_with   text only
  gt, gte, lt, lte          numeric only      exists                  value true/false
  date_gte/date_lte/date_gt/date_lt/date_eq   date field, value "YYYY-MM-DD"
  date_range                date field, value ["YYYY-MM-DD","YYYY-MM-DD"] (inclusive) — "between Feb 1 and Feb 15"
  period                    date field, value: today, yesterday, this_week, last_week, this_month, last_month, this_year, last_year
  last_days                 date field, value integer N — "last 7 days", "past 30 days"
Date fields: "created ..." -> "createdon"; "invoice date / billed ..." -> "InvoiceDate";
"due ..." -> "DueDate". A month name like "August" with no year -> date_range for that
whole month in the CURRENT year. date_bucket (daily/weekly/monthly series) uses "createdon".

VOCAB (map the user's word to the real field/value)
- status: InvoiceStatus is one of Paid, Open, PartiallyPaid, BadDebt, OverDue.
  "paid" = InvoiceStatus eq "Paid".  "unpaid" / "not paid" / "outstanding invoices" =
  InvoiceStatus op "nin" value ["Paid"].  "partially paid" = eq "PartiallyPaid".
  "bad debt" = eq "BadDebt".
  "overdue" = TWO filters: {{"field":"DueDate","op":"date_lt","value":"<today>"}} AND
  {{"field":"InvoiceStatus","op":"nin","value":["Paid"]}}.
  "overdue by more than 30 days" = DueDate date_lt <today-30d> AND InvoiceStatus nin ["Paid"].
- money: amount / total / value / "after tax" -> TotalAmount; "before tax" / pretax / subtotal
  -> PreTaxAmount; freight -> freightcharges; fuel / fuel surcharge -> fuelsurcharges;
  other charges -> othercharges; discount -> DiscountAmount; outstanding / balance / "still owed"
  -> outstandinamount; exchange rate -> ExchangeRate. "$1,000" / "C$1,000" -> 1000.
  "converted amount" / "amount using exchange rate" -> not a stored field; use metric sum on
  TotalAmount and note ExchangeRate separately, or mark unsupported if a per-row product is required.
- other: customer / client / buyer -> CustomerName; company / branch -> CompanyName;
  currency -> CurrencyCode; salesman / agent -> salesmanname; commodity / product -> commodityname;
  linked order -> InvoiceOrderNumbers; PO / customer order -> CustomerOrderNumbers;
  trip -> TripNumbers; carrier -> CarrierName; driver -> DriverName.
- geo virtual fields: country, state, city, destination, location, pickup_location,
  delivery_location, location_side ("from X" -> pickup, "to / delivered to / for deliveries in X"
  -> delivery). "US" / "United States" / "American" -> country eq "United States";
  "Canadian" -> country eq "Canada". You MUST emit these as filters, e.g.
  "invoices for deliveries in the United States" ->
  filters [{{"field":"country","op":"eq","value":"United States"}},
           {{"field":"location_side","op":"eq","value":"delivery"}}].

RULES
- Use ONLY fields shown in the schema above (dotted nested allowed) or the geo virtual fields. Never invent one.
- Numeric fn/ops only on numeric fields. group_by only on groupable fields (max 2).
- "total invoice amount" with no other qualifier -> task aggregate, operation metric, metrics [sum TotalAmount].
- "total invoice amount for each customer" -> operation group, group_by ["CustomerName"], metrics [sum TotalAmount].
- "how many invoices ..." -> operation count (+ filters).  "average invoice amount" -> metric avg TotalAmount.
- "which customers ..." / "list the customers ..." / "customers with ..." -> operation group,
  group_by ["CustomerName"] (so the names are returned), NOT distinct_count.
- put every aggregate key inside the "aggregate" object, never at the top level of the plan.
- "top N customers by invoice amount" -> group_by ["CustomerName"], metric sum TotalAmount, sort desc, limit N.
- "which invoice has the highest amount" -> task aggregate operation metric metrics [max TotalAmount]
  (or task list sort sum? prefer metric max). "show invoices below $500" -> task list, filter TotalAmount lt 500.
- having: threshold AFTER grouping ("customers with total invoices above $10,000" ->
  group_by ["CustomerName"], metrics [sum TotalAmount], having [{{sum_totalamount>10000}}]).
- Do NOT invent filters the user didn't ask for.
- If truly not expressible, task "unsupported".

Question: {question}
JSON:"""


def _plan_llm(question: str, history: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    llm = get_planner_llm()
    chain = PromptTemplate.from_template(_PLANNER_PROMPT) | llm
    raw = chain.invoke(
        {
            "question": question,
            "history": history or "(no prior turns)",
            "schema": invoice_schema_for_prompt(schema),
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
    if (
        name in _INVOICE_ISO_DATE_FIELDS
        or name in _US_DATE_FIELDS
        or _DATE_FIELD_RE.search(name)
    ):
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

    if name not in _VIRTUAL_GEO and name not in schema.get("fields", {}):
        resolved = resolve_field(name, schema, aliases=_INVOICE_FIELD_ALIASES)
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


def _valid_metric(raw_m: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_m, dict):
        return {"fn": "count", "field": None}
    fn = str(raw_m.get("fn") or "count").lower()
    if fn not in ("count", "sum", "avg", "min", "max"):
        fn = "count"
    if fn == "count":
        return {"fn": "count", "field": None}
    fld = resolve_field(raw_m.get("field"), schema, aliases=_INVOICE_FIELD_ALIASES)
    if not fld or not schema["fields"].get(fld, {}).get("numeric"):
        return {"fn": "count", "field": None}
    return {"fn": fn, "field": fld}


def _validate_plan(
    raw: Any, schema: Dict[str, Any], question: str
) -> Optional[InvoiceQueryPlan]:
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
        tok = extract_invoice_token(question)
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
        # The LLM sometimes hoists the aggregate keys to the top level.
        for k in ("operation", "metrics", "group_by", "distinct_field",
                  "date_bucket", "having"):
            if k not in agg_raw and raw.get(k) is not None:
                agg_raw[k] = raw[k]
        agg_raw.setdefault("limit", limit)
        agg_raw.setdefault("sort", sort)
        aggregate = validate_invoice_spec(agg_raw, schema)
        if aggregate is None:
            return None

    if task == "lookup" and not tokens:
        return None

    segments: List[Dict[str, Any]] = []
    numerator: List[Dict[str, Any]] = []
    pct_of = "invoices"
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
        metric = _valid_metric(raw.get("metric"), schema)

    if task == "compare" and not segments and not tokens:
        return None

    if task == "percentage":
        numerator = [
            c
            for c in (_valid_filter(f, schema) for f in (raw.get("numerator") or []))
            if c is not None
        ]
        if not numerator:
            return None
        of = raw.get("pct_of") or raw.get("of") or "invoices"
        if str(of).lower() in ("invoices", "invoice", "count", "records"):
            pct_of = "invoices"
        else:
            r = resolve_field(of, schema, aliases=_INVOICE_FIELD_ALIASES)
            pct_of = r if (r and schema["fields"].get(r, {}).get("numeric")) else "invoices"

    return InvoiceQueryPlan(
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


def run_invoice_query_planner(
    question: str, *, history: str = "(no prior turns)"
) -> Optional[InvoiceQueryPlan]:
    if not INVOICE_PLANNER_ENABLED:
        return None
    q = (question or "").strip()
    if not q:
        return None
    schema = get_invoices_schema()
    if not schema.get("fields"):
        return None

    raw = _plan_llm(q, history, schema)
    plan = _validate_plan(raw, schema, q)
    checkpoint(
        "INV_PLANNER",
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
        return (date.fromisoformat(iso[:10]) + timedelta(days=1)).isoformat()
    except ValueError:
        return iso + "~"


# ------------------------------------------------------------------- match builder
_COUNTRY_RX = {
    "united states": r"\b(?:United\s+States|USA|U\.S\.A\.?|U\.S\.?)\b",
    "usa": r"\b(?:United\s+States|USA|U\.S\.A\.?|U\.S\.?)\b",
    "us": r"\b(?:United\s+States|USA|U\.S\.A\.?|U\.S\.?)\b",
    "canada": r"\bCanada\b",
    "india": r"\bIndia\b",
}


def _side_geo_fields(side: str) -> List[str]:
    side = (side or "both").lower()
    if side == "pickup":
        return ["pickuplocation"]
    if side == "delivery":
        return ["deliverylocation", "destinationname"]
    return ["pickuplocation", "deliverylocation", "destinationname"]


def _iso_date_expr(field: str) -> Dict[str, Any]:
    """Parse a US-format ('2/16/2026 2:30:00 AM') date string to a real date
    using only the date part (Mongo has no %p specifier)."""
    return {
        "$dateFromString": {
            "dateString": {
                "$arrayElemAt": [
                    {"$split": [{"$toString": f"${field}"}, " "]},
                    0,
                ]
            },
            "format": "%m/%d/%Y",
            "onError": None,
            "onNull": None,
        }
    }


def _target_date_expr(iso: str) -> Dict[str, Any]:
    return {
        "$dateFromString": {
            "dateString": str(iso)[:10],
            "format": "%Y-%m-%d",
            "onError": None,
        }
    }


def _us_date_condition(field: str, op: str, value: Any) -> Optional[Dict[str, Any]]:
    """Return a {"$expr": ...} clause for a US-format date field."""
    src = _iso_date_expr(field)
    if op in ("date_gte", "date_gt", "date_lte", "date_lt"):
        mop = {"date_gte": "$gte", "date_gt": "$gt", "date_lte": "$lte", "date_lt": "$lt"}[op]
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
        start, end = value[0], value[1]
        return {
            "$expr": {
                "$and": [
                    {"$gte": [src, _target_date_expr(start)]},
                    {"$lt": [src, _target_date_expr(_plus_one_day(end))]},
                ]
            }
        }
    if op == "last_days":
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(value))).strftime("%Y-%m-%d")
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


def _iso_string_condition(op: str, value: Any) -> Optional[Dict[str, Any]]:
    """Per-field condition for an ISO-string date field (createdon/modifiedon)."""
    if op == "date_eq":
        return {"$regex": f"^{re.escape(str(value)[:10])}", "$options": "i"}
    if op in ("date_gte", "date_lte", "date_gt", "date_lt"):
        mop = {"date_gte": "$gte", "date_lte": "$lte", "date_gt": "$gt", "date_lt": "$lt"}[op]
        return {mop: str(value)[:10]}
    if op == "last_days":
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(value))).strftime("%Y-%m-%d")
        return {"$gte": cutoff}
    if op == "date_range":
        return {"$gte": str(value[0])[:10], "$lt": _plus_one_day(value[1])}
    if op == "period":
        bounds = _period_bounds(value)
        if not bounds:
            return None
        return {"$gte": bounds[0], "$lt": bounds[1]}
    return None


def _op_to_mongo(op: str, value: Any, field: str, kind: str):
    if op == "exists":
        return {"$nin": [None, ""]} if value else {"$in": [None, ""]}
    if op == "eq":
        if isinstance(value, str):
            return {"$regex": f"^{re.escape(value)}$", "$options": "i"}
        return value
    if op == "ne":
        if isinstance(value, str):
            return {"$not": {"$regex": f"^{re.escape(value)}$", "$options": "i"}}
        return {"$ne": value}
    if op == "in":
        return {"$in": value if isinstance(value, list) else [value]}
    if op == "nin":
        return {"$nin": value if isinstance(value, list) else [value]}
    if op == "contains":
        return {"$regex": re.escape(str(value)), "$options": "i"}
    if op == "starts_with":
        return {"$regex": f"^{re.escape(str(value))}", "$options": "i"}
    if op in ("gt", "gte", "lt", "lte"):
        mop = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}[op]
        if kind == "numeric":
            return {"__expr__": {mop: [_numeric_expr(field), float(value)]}}
        return {mop: value}
    return None


def _geo_clause(field: str, value: Any, side: str) -> Optional[Dict[str, Any]]:
    fields = _side_geo_fields(
        "pickup" if field == "pickup_location"
        else "delivery" if field in ("delivery_location", "destination")
        else side
    )
    val = str(value).strip()
    if field == "country":
        pattern = _COUNTRY_RX.get(val.lower(), rf"\b{re.escape(val)}\b")
    else:
        pattern = re.escape(val)
    ors = [{f: {"$regex": pattern, "$options": "i"}} for f in fields]
    return {"$or": ors} if ors else None


def _build_invoice_match(
    filters: List[Dict[str, Any]], schema: Dict[str, Any]
) -> Dict[str, Any]:
    match: Dict[str, Any] = dict(_invoice_base_match())
    and_parts: List[Dict[str, Any]] = []
    field_conds: Dict[str, List[Any]] = {}

    side = "both"
    for f in filters:
        if f["field"] == "location_side":
            side = str(f.get("value") or "both").lower()

    for f in filters:
        name, op, value = f["field"], f["op"], f.get("value")
        if name == "location_side":
            continue
        if name in _VIRTUAL_GEO:
            clause = _geo_clause(name, value, side)
            if clause:
                and_parts.append(clause)
            continue

        kind = _field_kind(name, schema)
        if kind is None:
            continue

        if kind == "date":
            if name in _US_DATE_FIELDS:
                if op in ("eq", "ne"):
                    # exact-string equality still works as a fallback
                    field_conds.setdefault(name, []).append(
                        _op_to_mongo(op, value, name, "text")
                    )
                    continue
                clause = _us_date_condition(name, op, value)
                if clause:
                    and_parts.append(clause)
                continue
            # ISO string date field
            if op in ("eq", "ne"):
                field_conds.setdefault(name, []).append(
                    _op_to_mongo(op, value, name, "text")
                )
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
        if len(conds) == 1:
            if name in match:
                and_parts.append({name: conds[0]})
            else:
                match[name] = conds[0]
        else:
            and_parts.extend({name: c} for c in conds)

    if and_parts:
        match["$and"] = and_parts
    return match


def _filters_summary(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """field -> readable condition; multiple conditions on one field are kept as a list."""
    out: Dict[str, Any] = {}
    for f in filters:
        val = f["value"] if f["op"] == "eq" else f"{f['op']} {f['value']}"
        if f["field"] in out:
            existing = out[f["field"]]
            out[f["field"]] = existing + [val] if isinstance(existing, list) else [existing, val]
        else:
            out[f["field"]] = val
    return out


# ------------------------------------------------------------------- list / metric
_LIST_FIELDS = (
    "InvoiceID", "InvoiceNumber", "CustomerName", "InvoiceStatus", "TotalAmount",
    "PreTaxAmount", "freightcharges", "othercharges", "outstandinamount",
    "CurrencyCode", "ExchangeRate", "InvoiceDate", "DueDate", "CompanyName",
    "InvoiceOrderNumbers", "commodityname", "pickuplocation", "deliverylocation",
)
_NUMERIC_SORT_KEYS = {
    "TotalAmount", "PreTaxAmount", "freightcharges", "othercharges",
    "outstandinamount", "ExchangeRate", "fuelsurcharges", "DiscountAmount",
}


def _list_invoices(plan: InvoiceQueryPlan, schema: Dict[str, Any]) -> Dict[str, Any]:
    match = _build_invoice_match(plan.filters, schema)
    collection = get_domain_collection("invoices")

    sort_key = (plan.sort or {}).get("key") or "InvoiceID"
    ascending = (plan.sort or {}).get("dir") == "asc"
    resolved = resolve_field(sort_key, schema, aliases=_INVOICE_FIELD_ALIASES) or "InvoiceID"
    direction = 1 if ascending else -1
    limit = max(1, min(int(plan.limit or 15), 50))

    pipeline: List[Dict[str, Any]] = [{"$match": match}]
    if resolved in _NUMERIC_SORT_KEYS or schema["fields"].get(resolved, {}).get("numeric"):
        pipeline.append({"$addFields": {"__sort": _numeric_expr(resolved)}})
        pipeline.append({"$sort": {"__sort": direction, "InvoiceID": -1}})
    else:
        pipeline.append({"$sort": {resolved: direction}})
    pipeline.append({"$limit": limit})
    projection: Dict[str, Any] = {"_id": 0}
    for _f in _LIST_FIELDS:
        projection[_f] = 1
    pipeline.append({"$project": projection})

    total = collection.count_documents(match)
    rows = list(collection.aggregate(pipeline, maxTimeMS=AGG_TIMEOUT_MS))
    records = [{k: r.get(k) for k in _LIST_FIELDS if r.get(k) not in (None, "")} for r in rows]

    checkpoint(
        "INV_PLANNER", "list executed", total=total, returned=len(records)
    )
    return {
        "domain": "invoices",
        "filters": _filters_summary(plan.filters),
        "sort_by": resolved,
        "ascending": ascending,
        "total_matching": total,
        "returned": len(records),
        "records": records,
    }


def _metric_value(match: Dict[str, Any], metric: Dict[str, Any]) -> float:
    collection = get_domain_collection("invoices")
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
    return "invoice_count" if fn == "count" else f"{fn}_{metric.get('field')}"


def _derive_paid_amount(doc: Dict[str, Any]) -> Dict[str, Any]:
    try:
        total = float(str(doc.get("TotalAmount") or "").replace(",", "") or 0)
        outstanding_raw = doc.get("outstandinamount")
        status = str(doc.get("InvoiceStatus") or "").strip().lower()
        outstanding = (
            float(outstanding_raw) if outstanding_raw not in (None, "") else None
        )
        doc = dict(doc)
        if status == "paid":
            doc["PaidAmount"] = total
        elif outstanding is not None:
            doc["PaidAmount"] = max(0.0, total - outstanding)
    except (TypeError, ValueError):
        pass
    return doc


# ------------------------------------------------------------------- execute
def execute_invoice_query_plan(
    plan: InvoiceQueryPlan, *, question: str
) -> Dict[str, Any]:
    """Run a validated plan. Returns the same shape as tools.execute_tools."""
    context_blocks: List[str] = []
    matches: List[Dict[str, Any]] = []
    analytics_payload = None
    list_payload = None
    tools_run: List[str] = []
    active_token = plan.record_tokens[0] if plan.record_tokens else ""

    schema = get_invoices_schema()

    if plan.task == "lookup":
        token = plan.record_tokens[0] if plan.record_tokens else ""
        doc = find_record_by_token(token) if token else None
        if doc:
            doc = _derive_paid_amount(doc)
            context_blocks.append(
                "EXACT INVOICE RECORD:\n" + format_record_doc_for_context(doc)
            )
            matches.append(
                {
                    "InvoiceID": doc.get("InvoiceID"),
                    "InvoiceNumber": doc.get("InvoiceNumber"),
                    "match_type": "exact",
                }
            )
        else:
            context_blocks.append(
                f"EXACT INVOICE RECORD: not found for token={token}"
            )
        tools_run.append("get_record")

    elif plan.task == "compare" and plan.segments:
        metric = plan.metric or {"fn": "count", "field": None}
        label = _metric_label(metric)
        seg_rows = []
        for seg in plan.segments:
            m = _build_invoice_match(plan.filters + seg["filters"], schema)
            seg_rows.append(
                {"segment": seg["label"], label: round(_metric_value(m, metric), 4)}
            )
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "invoice_dynamic_planner",
            "operation": "compare",
            "metric": label,
            "filters": _filters_summary(plan.filters),
            "rows": seg_rows,
        }
        context_blocks.append(
            format_invoice_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

    elif plan.task == "compare":
        parts: List[str] = []
        for token in plan.record_tokens[:2]:
            doc = find_record_by_token(token)
            if doc:
                doc = _derive_paid_amount(doc)
                parts.append(
                    f"INVOICE {token}:\n"
                    + format_record_doc_for_context(doc, max_fields=40)
                )
                matches.append(
                    {"InvoiceNumber": doc.get("InvoiceNumber"), "match_type": "compare"}
                )
            else:
                parts.append(f"INVOICE {token}: not found")
        context_blocks.append("COMPARE INVOICES:\n" + "\n\n".join(parts))
        tools_run.append("compare_records")

    elif plan.task == "percentage":
        metric = (
            {"fn": "count", "field": None}
            if plan.pct_of == "invoices"
            else {"fn": "sum", "field": plan.pct_of}
        )
        den_match = _build_invoice_match(plan.filters, schema)
        num_match = _build_invoice_match(plan.filters + plan.numerator, schema)
        den = _metric_value(den_match, metric)
        num = _metric_value(num_match, metric)
        pct = round((num / den * 100.0), 2) if den else 0.0
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "invoice_dynamic_planner",
            "operation": "percentage",
            "of": ("invoices" if plan.pct_of == "invoices" else f"sum {plan.pct_of}"),
            "numerator": round(num, 4),
            "denominator": round(den, 4),
            "percentage": pct,
            "numerator_filters": _filters_summary(plan.numerator),
            "filters": _filters_summary(plan.filters),
        }
        context_blocks.append(
            format_invoice_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

    elif plan.task == "aggregate" and plan.aggregate is not None:
        match = _build_invoice_match(plan.filters, schema)
        pipeline = _build_pipeline(plan.aggregate, match)
        rows = list(
            get_domain_collection("invoices").aggregate(
                pipeline, maxTimeMS=AGG_TIMEOUT_MS, allowDiskUse=False
            )
        )
        analytics_payload = _shape_invoice_result(
            plan.aggregate, rows, _filters_summary(plan.filters), question
        )
        context_blocks.append(
            format_invoice_dynamic_analytics_for_context(analytics_payload)
        )
        checkpoint(
            "INV_PLANNER",
            "aggregate executed",
            operation=plan.aggregate.get("operation"),
            rows=len(rows),
        )
        tools_run.append("run_analytics")

    else:  # list (default)
        list_payload = _list_invoices(plan, schema)
        context_blocks.append(format_record_list_for_context(list_payload))
        for row in list_payload.get("records") or []:
            matches.append(
                {
                    "InvoiceID": row.get("InvoiceID"),
                    "InvoiceNumber": row.get("InvoiceNumber"),
                    "match_type": "filter",
                }
            )
        tools_run.append("search_records")

    return {
        "context_blocks": context_blocks,
        "matches": matches,
        "calculation": None,
        "analytics": analytics_payload,
        "list_result": list_payload,
        "tools_run": tools_run,
        "active_order_token": active_token,
        "domain": "invoices",
    }
