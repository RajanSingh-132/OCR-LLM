"""
LLM query planner for Avaal trips — the primary routing path for the ``trips``
domain of ``/api/v1/orders/ask``. The regex engine
(``app/domains/rules/trips.py`` + ``app/order_ask/trip_analytics.py``) stays
wired as an automatic fallback.

One Claude call reads the question + conversation history + a live sampled
schema of ``Avaal_trip`` and returns a strict JSON ``TripQueryPlan``:

    task      lookup | list | aggregate | compare | percentage | conversation
              | greeting | unsupported
    filters   operator-DSL list  [{"field","op","value"}]
    aggregate {operation, group_by, metrics, distinct_field, date_bucket, having}
    segments / numerator / metric / pct_of   (compare / percentage)
    sort / limit / response_style

``_validate_plan`` checks every field against the sampled schema and every op
against the allow-list. ``_build_trip_match`` turns the validated filters into a
safe Mongo ``$match`` (numeric ranges via ``$expr`` + ``$convert``; ISO date
strings via lexical prefix compare; geo via real pickup/delivery field regex).
``execute_trip_query_plan`` runs it and returns the same payload shape as
``app/order_ask/tools.py`` ``execute_tools``.

Returns ``None`` (→ regex fallback) when disabled, on any error, on an invalid
plan, or for task greeting / conversation / unsupported.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
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
from app.order_ask.invoice_query_planner import (
    _iso_string_condition,
    _op_to_mongo,
    _period_bounds,
)
from app.order_ask.trip_dynamic_analytics import (
    _TRIP_FIELD_ALIASES,
    _TRIP_ISO_DATE_FIELDS,
    _shape_trip_result,
    format_trip_dynamic_analytics_for_context,
    get_trips_schema,
    trip_schema_for_prompt,
    validate_trip_spec,
)
from app.order_ask.trip_analytics import _base_match as _trip_base_match
from app.domains.lookup.trips.lookup import extract_token as extract_trip_token
from app.domains.registry import get_domain_profile
from app.domains.retrieval import (
    find_record_by_token,
    format_record_doc_for_context,
    format_record_list_for_context,
)
from app.tenants.router import get_domain_collection

TRIP_PLANNER_ENABLED = os.environ.get(
    "AVAAL_TRIP_QUERY_PLANNER", "1"
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

# Virtual geo fields — resolved to real Avaal_trip pickup/delivery fields
# (not passed to Mongo verbatim).
_VIRTUAL_GEO = frozenset(
    {
        "city", "state", "province", "country", "location",
        "pickup_city", "delivery_city", "pickup_state", "delivery_state",
        "pickup_country", "delivery_country", "pickup_location",
        "delivery_location", "location_side",
    }
)

_DATE_FIELD_RE = re.compile(r"(date|datetime)$", re.I)

# Canonical UI trip statuses.
KNOWN_TRIP_STATUSES = (
    "Planned", "Dispatched", "Started", "In-Transit", "Delivered", "Rejected",
)

_COUNTRY_RX = {
    "united states": r"^(?:United\s*States|USA|U\.?S\.?A?\.?)$",
    "usa": r"^(?:United\s*States|USA|U\.?S\.?A?\.?)$",
    "us": r"^(?:United\s*States|USA|U\.?S\.?A?\.?)$",
    "america": r"^(?:United\s*States|USA|U\.?S\.?A?\.?)$",
    "canada": r"^Canada$",
    "india": r"^India$",
}


# ------------------------------------------------------------------- plan
@dataclass
class TripQueryPlan:
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
    pct_of: str = "trips"
    metric: Optional[Dict[str, Any]] = None

    def to_intent_info(self) -> Dict[str, Any]:
        intent = {
            "lookup": "trip_lookup",
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
            "reason": f"trip_planner:{self.reason or self.task}",
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
_PLANNER_PROMPT = """You are a query planner for the Avaal_trip MongoDB collection (fleet dispatch trips).
Convert the user's question into ONE strict JSON plan. Output JSON only — no prose, no code fence.

Today (UTC): {today}

{schema}

Conversation so far:
{history}

PLAN SHAPE (include only the keys the task needs):
{{
  "task": "lookup" | "list" | "aggregate" | "compare" | "percentage" | "conversation" | "greeting" | "unsupported",
  "record_tokens": ["ETP4455", ...],
  "filters": [ {{"field": "<field>", "op": "<op>", "value": <scalar|list>}} ],
  "aggregate": {{
    "operation": "count" | "metric" | "group" | "distinct_count",
    "group_by": ["<groupable field>", ...],
    "metrics": [{{"fn": "sum|avg|min|max|count", "field": "<numeric field>"}}],
    "distinct_field": "<field>",
    "date_bucket": {{"field": "createdon", "unit": "day|week|month"}},
    "having": [{{"key": "trips|count|sum_<field>|avg_<field>", "op": "gt|gte|lt|lte", "value": <num>}}]
  }},
  "segments": [ {{"label": "Dispatched", "filters": [ ... ]}}, {{"label": "Delivered", "filters": [ ... ]}} ],
  "metric": {{"fn": "count|sum|avg", "field": "<numeric field>"}},
  "numerator": [ {{"field": "...", "op": "...", "value": ...}} ],
  "pct_of": "trips" | "<numeric field>",
  "sort": {{"key": "<field / metric key like sum_triptotaldistance / count / trips / period>", "dir": "asc|desc"}},
  "limit": <int 1-100>,
  "response_style": "short|medium|detailed",
  "reason": "<short>"
}}

TASKS
- lookup     : user names ONE trip (trip number like ETP4455 / trip id like 29797) -> record_tokens.
               Also lookup for "who are the drivers for ETP4455", "order ids for ETP4455",
               "is ETP4455 dispatched", "what is the total distance of this trip",
               "give me details of trip ETP4455". Any single-trip attribute question = lookup.
- compare(records) : compares 2 named trips -> record_tokens has 2.
- compare(segments): compares a metric across statuses / places / drivers / companies / time windows
                     ("planned vs dispatched", "Canada vs US", "August vs July") ->
                     segments[] each with its own filters, plus one metric.
- list       : wants the actual trip rows -> filters (+ sort/limit).
               "show / list / find / which trips ...", "trips from Montreal",
               "trips delivered in Ontario", "trips with distance greater than 500 miles",
               "trips where settlement status is Paid", "trips assigned to driver DRV00895".
- aggregate  : how many / count / total / sum / average / min / max / per / by / wise / distinct
               / "count city wise" / "count by driver" / "count by trip status" / daily / weekly
               / monthly. "which pickup city has the most trips" / "which driver traveled the
               highest total distance" -> group + sort desc + limit 1.
               "which trip has the highest offered amount" -> task list sort by totalofferedamount desc limit 1.
- percentage : "what % of trips ..." -> numerator (the subset) + pct_of ("trips" or a numeric field).
- conversation / greeting / unsupported.

FILTER OPS
  eq, ne, in, nin           any field         contains, starts_with   text only
  gt, gte, lt, lte          numeric only      exists                  value true/false
  date_gte/date_lte/date_gt/date_lt/date_eq   date field, value "YYYY-MM-DD"
  date_range                date field, value ["YYYY-MM-DD","YYYY-MM-DD"] (inclusive) — "between Aug 1 and Aug 15"
  period                    date field, value: today, yesterday, this_week, last_week, this_month, last_month, this_year, last_year
  last_days                 date field, value integer N — "last 7 days", "past 30 days"
Date fields: "created ..." -> "createdon"; "modified / updated ..." -> "modifiedon";
"picked up / pickup date" -> "firstpickupdate"; "delivered / delivery date" -> "lastdeliverydate".
A month name like "August" with no year -> date_range for that whole month in the CURRENT year.
date_bucket (daily/weekly/monthly series) uses "createdon".

VOCAB (map the user's word to the real field/value)
- status: tripstatus is one of Planned, Dispatched, Started, In-Transit, Delivered, Rejected
  (DB may also store "In Transit" / "Enroute" / "Cancelled").
  "dispatched" = tripstatus eq "Dispatched"; same for planned / started / delivered / rejected.
  "in transit" / "en route" -> tripstatus eq "In-Transit".
  "active" / "on road" / "running" / "not delivered yet" / "open" / "pending" =
  tripstatus op "nin" value ["Delivered","Rejected","Cancelled"].
  "completed" / "closed" -> tripstatus eq "Delivered".
  "planned and dispatched" / "planned or dispatched" -> ONE filter
  {{"field":"tripstatus","op":"in","value":["Planned","Dispatched"]}}.
  If the user asks a per-status breakdown ("status wise", "count by trip status",
  "status wise summary") -> task aggregate, operation group, group_by ["tripstatus"], metrics [count].
- distance: "distance" / "trip distance" / "total distance" / "miles" -> triptotaldistance;
  "loaded distance" -> totalloaddistance; "empty distance" / "deadhead" -> totalemptydistance.
  "distance unit" -> distanceunit (text: Miles / Km). "500 miles" -> value 500.
- cargo: weight -> totalweight; quantity -> totalquantity; "items" / "item count" -> itemscount;
  "trip items" -> tripitemscount; commodity / goods / product -> commodity (comma-joined text — use contains).
- money: amount / offered amount / revenue / pay / linehaul -> totalofferedamount;
  rate -> rate; tax -> totaltaxamount. "$1,000" / "C$1,000" -> 1000.
- people/equipment: driver -> firstdrivername (a trip has firstdrivername + seconddrivername;
  a plain "driver X" question means EITHER — use op "contains" on firstdrivername, the planner
  also checks seconddrivername automatically); driver code like DRV00895 -> firstdrivercode
  (checked against seconddrivercode too); truck -> trucknumber; trailer -> firsttrailernumber;
  salesman / agent (EMP codes) -> salesmancodes, salesman name -> salesmannames.
- customer / client -> customername (comma-joined — use contains). company / branch -> companyname
  ("belongs to Avaal Group" -> companyname eq "Avaal Group"). carrier -> carriername.
- linked docs: order number (MRP...) -> ordernumber (contains); order id (digits) -> orderids (contains).
- type: triptype ("REGULAR - Loaded" ...) ; triptypemain ("REGULAR" / "OUTSOURCING"); variant -> tripvariant.
- settlement: "settlement status is Paid" -> settlementstatus eq "Paid".
- geo VIRTUAL fields (map the user's place to these; the planner expands each to the real
  pickup*/delivery* Avaal_trip fields): city, state, country, pickup_city, delivery_city,
  pickup_state, delivery_state, pickup_country, delivery_country, pickup_location,
  delivery_location, location_side.
  "from X" / "picked up in X" / "out of X" -> pickup side; "to X" / "delivered in/to X" /
  "for deliveries in X" -> delivery side; a bare place -> use city/state/country (both sides).
  "Montreal" is a city, "Quebec" / "Ontario" a state/province, "Canada" / "US" a country.
  Examples: "trips from Montreal" -> [{{"field":"pickup_city","op":"eq","value":"Montreal"}}];
  "trips delivered in Ontario" -> [{{"field":"delivery_state","op":"eq","value":"Ontario"}}];
  "how many trips in Quebec" -> [{{"field":"state","op":"eq","value":"Quebec"}}].
  "US" / "United States" / "American" -> country/…_country eq "United States"; "Canadian" -> "Canada".

RULES
- Use ONLY fields shown in the schema above (dotted nested allowed) or the geo virtual fields. Never invent one.
- Numeric fn/ops only on numeric fields. group_by only on groupable fields (max 2).
- "trip count city wise" / "give me trip count by driver" / "count by company" ->
  operation group, group_by [that field], metrics [count], sort desc.
  city wise -> group_by ["pickupcity"]; state wise -> ["pickupstate"]; driver -> ["firstdrivername"];
  company -> ["companyname"]; trip status -> ["tripstatus"].
- "trip count by pickup city and trip status" -> group_by ["pickupcity","tripstatus"].
- "top 10 drivers by number of trips" -> group_by ["firstdrivername"], metrics [count], sort desc, limit 10.
- "total distance by driver" -> group_by ["firstdrivername"], metrics [sum triptotaldistance].
- "average trip distance" -> operation metric, metrics [avg triptotaldistance] (NO group_by).
- "which pickup city has the highest number of trips" / "which driver traveled the highest total
  distance" -> operation group, group_by [that field], sort desc, limit 1.
- "how many trips are dispatched" -> operation count + filter tripstatus eq "Dispatched".
- "give me status wise summary of the trips" -> operation group, group_by ["tripstatus"], metrics [count].
- having: threshold AFTER grouping ("drivers with more than 5 trips" ->
  group_by ["firstdrivername"], metrics [count], having [{{"key":"trips","op":"gt","value":5}}]).
- put every aggregate key inside the "aggregate" object, never at the top level of the plan.
- Do NOT invent filters the user didn't ask for.
- Follow-ups: if the user already fixed a trip / filter earlier in the conversation and now asks a
  bare attribute or "and the drivers?" / "what about delivered ones", carry that context forward.
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
            "schema": trip_schema_for_prompt(schema),
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
        name in _TRIP_ISO_DATE_FIELDS
        or name.split(".")[-1] in _TRIP_ISO_DATE_FIELDS
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
        resolved = resolve_field(name, schema, aliases=_TRIP_FIELD_ALIASES)
        if resolved:
            name = resolved

    kind = _field_kind(name, schema)
    if kind is None:
        return None

    if kind == "geo":
        if name == "location_side":
            value = str(value).lower()
            if value not in ("pickup", "delivery", "both"):
                return None
            return {"field": name, "op": "eq", "value": value}
        if op not in ("eq", "contains", "in"):
            op = "eq"
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
    fld = resolve_field(raw_m.get("field"), schema, aliases=_TRIP_FIELD_ALIASES)
    if not fld or not schema["fields"].get(fld, {}).get("numeric"):
        return {"fn": "count", "field": None}
    return {"fn": fn, "field": fld}


def _validate_plan(
    raw: Any, schema: Dict[str, Any], question: str
) -> Optional[TripQueryPlan]:
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
        tok = extract_trip_token(question)
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
        for k in ("operation", "metrics", "group_by", "distinct_field",
                  "date_bucket", "having"):
            if k not in agg_raw and raw.get(k) is not None:
                agg_raw[k] = raw[k]
        agg_raw.setdefault("limit", limit)
        agg_raw.setdefault("sort", sort)
        aggregate = validate_trip_spec(agg_raw, schema)
        if aggregate is None:
            return None

    if task == "lookup" and not tokens:
        return None

    segments: List[Dict[str, Any]] = []
    numerator: List[Dict[str, Any]] = []
    pct_of = "trips"
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
        of = raw.get("pct_of") or raw.get("of") or "trips"
        if str(of).lower() in ("trips", "trip", "count", "records"):
            pct_of = "trips"
        else:
            r = resolve_field(of, schema, aliases=_TRIP_FIELD_ALIASES)
            pct_of = r if (r and schema["fields"].get(r, {}).get("numeric")) else "trips"

    return TripQueryPlan(
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


def run_trip_query_planner(
    question: str, *, history: str = "(no prior turns)"
) -> Optional[TripQueryPlan]:
    if not TRIP_PLANNER_ENABLED:
        return None
    q = (question or "").strip()
    if not q:
        return None
    schema = get_trips_schema()
    if not schema.get("fields"):
        return None

    raw = _plan_llm(q, history, schema)
    plan = _validate_plan(raw, schema, q)
    checkpoint(
        "TRIP_PLANNER",
        "plan",
        raw=raw,
        task=(plan.task if plan else None),
        filters=(len(plan.filters) if plan else None),
    )
    if plan is None or plan.task in ("greeting", "conversation", "unsupported"):
        return None
    return plan


# ------------------------------------------------------------------- match builder
def _side_geo_fields(kind: str, side: str) -> List[str]:
    """kind: city | state | country | location. side: pickup | delivery | both."""
    field_by_kind = {
        "city": ("pickupcity", "deliverycity"),
        "state": ("pickupstate", "deliverystate"),
        "country": ("pickupcountry", "deliverycountry"),
        "location": ("pickuplocationname", "deliverylocationname"),
    }
    pu, de = field_by_kind.get(kind, ("pickupcity", "deliverycity"))
    if side == "pickup":
        return [pu]
    if side == "delivery":
        return [de]
    return [pu, de]


def _geo_clause(field: str, value: Any, side: str) -> Optional[Dict[str, Any]]:
    kind = "city"
    resolved_side = side
    if field in ("city", "pickup_city", "delivery_city"):
        kind = "city"
    elif field in ("state", "province", "pickup_state", "delivery_state"):
        kind = "state"
    elif field in ("country", "pickup_country", "delivery_country"):
        kind = "country"
    elif field in ("location", "pickup_location", "delivery_location"):
        kind = "location"
    if field.startswith("pickup_"):
        resolved_side = "pickup"
    elif field.startswith("delivery_"):
        resolved_side = "delivery"

    fields = _side_geo_fields(kind, resolved_side)
    val = str(value).strip()
    if not val:
        return None
    if kind == "country":
        pattern = _COUNTRY_RX.get(val.lower(), rf"^{re.escape(val)}$")
    else:
        pattern = re.escape(val)
    ors = [{f: {"$regex": pattern, "$options": "i"}} for f in fields]
    return {"$or": ors} if ors else None


def _driver_clause(op: str, value: Any) -> Dict[str, Any]:
    """A plain driver filter matches first OR second driver (name or code)."""
    v = str(value).strip()
    rx = {"$regex": re.escape(v), "$options": "i"} if op == "contains" else {
        "$regex": f"^{re.escape(v)}$", "$options": "i"
    }
    return {
        "$or": [
            {"firstdrivername": rx},
            {"seconddrivername": rx},
            {"firstdrivercode": rx},
            {"seconddrivercode": rx},
        ]
    }


def _build_trip_match(
    filters: List[Dict[str, Any]], schema: Dict[str, Any]
) -> Dict[str, Any]:
    match: Dict[str, Any] = dict(_trip_base_match())
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

        # A bare driver-name / driver-code filter fans out to both drivers.
        if name in ("firstdrivername", "firstdrivercode") and op in ("eq", "contains"):
            and_parts.append(_driver_clause(op, value))
            continue

        if kind == "date":
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
_NUMERIC_SORT_HINTS = frozenset(
    {
        "triptotaldistance", "totalloaddistance", "totalemptydistance",
        "totalofferedamount", "offeredamount", "totalweight", "totalquantity",
        "totaltaxamount", "rate", "ratetypevalue", "itemscount", "tripitemscount",
        "totaldistance", "ebdistance", "eedistance",
    }
)


def _list_trips(plan: TripQueryPlan, schema: Dict[str, Any]) -> Dict[str, Any]:
    match = _build_trip_match(plan.filters, schema)
    collection = get_domain_collection("trips")
    list_fields = list(get_domain_profile("trips").list_fields)

    sort_key = (plan.sort or {}).get("key") or "tripid"
    ascending = (plan.sort or {}).get("dir") == "asc"
    resolved = (
        resolve_field(sort_key, schema, aliases=_TRIP_FIELD_ALIASES) or "tripid"
    )
    direction = 1 if ascending else -1
    limit = max(1, min(int(plan.limit or 15), 50))

    numeric_sort = (
        resolved in _NUMERIC_SORT_HINTS
        or schema["fields"].get(resolved, {}).get("numeric")
    )
    pipeline: List[Dict[str, Any]] = [{"$match": match}]
    if numeric_sort:
        pipeline.append({"$addFields": {"__sort": _numeric_expr(resolved)}})
        pipeline.append({"$sort": {"__sort": direction, "tripid": -1}})
    else:
        pipeline.append({"$sort": {resolved: direction}})
    pipeline.append({"$limit": limit})
    projection: Dict[str, Any] = {"_id": 0}
    for _f in list_fields:
        projection[_f] = 1
    pipeline.append({"$project": projection})

    total = collection.count_documents(match)
    rows = list(
        collection.aggregate(pipeline, maxTimeMS=AGG_TIMEOUT_MS, allowDiskUse=False)
    )
    records = [
        {k: r.get(k) for k in list_fields if r.get(k) not in (None, "")}
        for r in rows
    ]

    checkpoint(
        "TRIP_PLANNER", "list executed", total=total, returned=len(records)
    )
    return {
        "domain": "trips",
        "filters": _filters_summary(plan.filters),
        "sort_by": resolved,
        "ascending": ascending,
        "total_matching": total,
        "returned": len(records),
        "records": records,
    }


def _metric_value(match: Dict[str, Any], metric: Dict[str, Any]) -> float:
    collection = get_domain_collection("trips")
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
    return "trip_count" if fn == "count" else f"{fn}_{metric.get('field')}"


# ------------------------------------------------------------------- execute
def execute_trip_query_plan(
    plan: TripQueryPlan, *, question: str
) -> Dict[str, Any]:
    """Run a validated plan. Returns the same shape as tools.execute_tools."""
    context_blocks: List[str] = []
    matches: List[Dict[str, Any]] = []
    analytics_payload = None
    list_payload = None
    tools_run: List[str] = []
    active_token = plan.record_tokens[0] if plan.record_tokens else ""

    schema = get_trips_schema()

    if plan.task == "lookup":
        token = plan.record_tokens[0] if plan.record_tokens else ""
        doc = find_record_by_token(token) if token else None
        if doc:
            context_blocks.append(
                "EXACT TRIP RECORD:\n" + format_record_doc_for_context(doc)
            )
            matches.append(
                {
                    "tripid": doc.get("tripid"),
                    "tripnumber": doc.get("tripnumber"),
                    "match_type": "exact",
                }
            )
        else:
            context_blocks.append(
                f"EXACT TRIP RECORD: not found for token={token}"
            )
        tools_run.append("get_record")

    elif plan.task == "compare" and plan.segments:
        metric = plan.metric or {"fn": "count", "field": None}
        label = _metric_label(metric)
        seg_rows = []
        for seg in plan.segments:
            m = _build_trip_match(plan.filters + seg["filters"], schema)
            seg_rows.append(
                {"segment": seg["label"], label: round(_metric_value(m, metric), 4)}
            )
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "trip_dynamic_planner",
            "operation": "compare",
            "metric": label,
            "filters": _filters_summary(plan.filters),
            "rows": seg_rows,
        }
        context_blocks.append(
            format_trip_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

    elif plan.task == "compare":
        parts: List[str] = []
        for token in plan.record_tokens[:2]:
            doc = find_record_by_token(token)
            if doc:
                parts.append(
                    f"TRIP {token}:\n"
                    + format_record_doc_for_context(doc, max_fields=50)
                )
                matches.append(
                    {"tripnumber": doc.get("tripnumber"), "match_type": "compare"}
                )
            else:
                parts.append(f"TRIP {token}: not found")
        context_blocks.append("COMPARE TRIPS:\n" + "\n\n".join(parts))
        tools_run.append("compare_records")

    elif plan.task == "percentage":
        metric = (
            {"fn": "count", "field": None}
            if plan.pct_of == "trips"
            else {"fn": "sum", "field": plan.pct_of}
        )
        den_match = _build_trip_match(plan.filters, schema)
        num_match = _build_trip_match(plan.filters + plan.numerator, schema)
        den = _metric_value(den_match, metric)
        num = _metric_value(num_match, metric)
        pct = round((num / den * 100.0), 2) if den else 0.0
        analytics_payload = {
            "analytics_type": "dynamic",
            "engine": "trip_dynamic_planner",
            "operation": "percentage",
            "of": ("trips" if plan.pct_of == "trips" else f"sum {plan.pct_of}"),
            "numerator": round(num, 4),
            "denominator": round(den, 4),
            "percentage": pct,
            "numerator_filters": _filters_summary(plan.numerator),
            "filters": _filters_summary(plan.filters),
        }
        context_blocks.append(
            format_trip_dynamic_analytics_for_context(analytics_payload)
        )
        tools_run.append("run_analytics")

    elif plan.task == "aggregate" and plan.aggregate is not None:
        match = _build_trip_match(plan.filters, schema)
        pipeline = _build_pipeline(plan.aggregate, match)
        rows = list(
            get_domain_collection("trips").aggregate(
                pipeline, maxTimeMS=AGG_TIMEOUT_MS, allowDiskUse=False
            )
        )
        analytics_payload = _shape_trip_result(
            plan.aggregate, rows, _filters_summary(plan.filters), question
        )
        context_blocks.append(
            format_trip_dynamic_analytics_for_context(analytics_payload)
        )
        checkpoint(
            "TRIP_PLANNER",
            "aggregate executed",
            operation=plan.aggregate.get("operation"),
            rows=len(rows),
        )
        tools_run.append("run_analytics")

    else:  # list (default)
        list_payload = _list_trips(plan, schema)
        context_blocks.append(format_record_list_for_context(list_payload))
        for row in list_payload.get("records") or []:
            matches.append(
                {
                    "tripid": row.get("tripid"),
                    "tripnumber": row.get("tripnumber"),
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
        "domain": "trips",
    }
