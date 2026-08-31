"""
Dynamic analytics for Avaal orders.

Instead of a fixed formula/analytics catalog, an LLM *planner* reads a sampled
field schema of the Avaal_order collection and emits a constrained aggregation
spec (operation + group_by + metrics + sort + limit). A deterministic builder
turns that spec into a safe Mongo aggregation pipeline.

Filters (status / customer / date / place) are NOT chosen by the planner — they
still come from the tested entity-extraction layer via ``_base_order_match``.

``run_dynamic_analytics`` returns ``None`` (caller falls back to the hardcoded
``app.order_ask.analytics`` engine) whenever the planner is disabled, errors,
produces an invalid/degenerate spec, or the question is one the specialised
catalog engine handles better (geo string parsing, date-activity, period,
trip-distance, customer ranking).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_anthropic_llm
from app.order_ask.checkpoint import checkpoint
from app.order_ask.entities import entities_to_mongo_filters
from app.order_ask.rag_retrieval import _base_order_match
from app.tenants.context import require_tenant
from app.tenants.router import get_orders_collection

# --------------------------------------------------------------------- config
DYNAMIC_ANALYTICS_ENABLED = os.environ.get(
    "AVAAL_DYNAMIC_ANALYTICS", "1"
).strip().lower() not in ("0", "false", "no", "off")
SCHEMA_SAMPLE_SIZE = int(os.environ.get("AVAAL_ANALYTICS_SCHEMA_SAMPLE", "500"))
SCHEMA_TTL_SECONDS = int(os.environ.get("AVAAL_ANALYTICS_SCHEMA_TTL", "3600"))
AGG_TIMEOUT_MS = int(os.environ.get("AVAAL_ANALYTICS_TIMEOUT_MS", "8000"))

MAX_GROUP_BY = 2
MAX_LIMIT = 100
MAX_METRICS = 6
GROUPABLE_MAX_DISTINCT = 200

_ALLOWED_FNS = frozenset({"sum", "avg", "min", "max", "count"})
_ALLOWED_OPS = frozenset({"group", "metric", "count", "distinct_count"})
_HAVING_OPS = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte", "eq": "$eq", "ne": "$ne"}
_BUCKET_UNITS = frozenset({"day", "week", "month"})
# Date fields stored ISO-8601 (safe for $toDate / string prefix compare).
_ISO_DATE_FIELDS = frozenset(
    {"orderdate", "createdon", "modifiedon", "enquirydate", "quotationdate"}
)

_SKIP_FIELDS = frozenset(
    {"_id", "embedding", "page_content", "metadata", "namespace"}
)
_ID_LIKE_FIELDS = frozenset(
    {"orderid", "ordernumber", "tempordernumber", "tripno"}
)

# Questions the specialised catalog engine does better — defer to it.
_DEFER_RE = re.compile(
    r"\b(state|province|city|town|country)[\s-]*wise\b|"
    r"\bby\s+(state|province|city|town|country)\b|"
    r"\b(today|aaj|yesterday|kal)\b|"
    r"\blast\s+\d+\s+(day|days|week|weeks|month|months)\b|"
    r"\b(last|past|previous)\s+(one\s+)?(day|week|month)\b|"
    r"\bpast\s+\d+\s+days?\b|"
    r"\b(best|worst|top|low(est)?)\s+customer|customer\s+(with|by)\s+(most|least|highest|lowest)\b|"
    r"\btrip\s*distance\b|"
    r"\borders?\s+on\s+20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
    re.I,
)

_NUM_STR_RE = re.compile(r"^-?\$?\s?[\d,]+(\.\d+)?%?$")

# Nested-object walk depth (0 = top level). commoditydetails[]/outsourcedetails{}/
# compliancedetails{}/accountingtypebreakdown{} all live one level down.
_MAX_DEPTH = int(os.environ.get("AVAAL_ANALYTICS_SCHEMA_DEPTH", "2"))
_MAX_FIELDS = int(os.environ.get("AVAAL_ANALYTICS_MAX_FIELDS", "260"))

# Common business phrasings → real field (keys are _norm()'d). Only used when a
# planner-named field is not found exactly / fuzzily.
_FIELD_ALIASES: Dict[str, str] = {
    "revenue": "grosstotalfreight", "grossrevenue": "grosstotalfreight",
    "grossfreight": "grosstotalfreight", "totalrevenue": "grosstotalfreight",
    "amount": "totalfreight", "price": "totalfreight", "cost": "totalfreight",
    "freight": "totalfreight", "freightamount": "totalfreight",
    "rate": "freightratevalue", "freightrate": "freightratevalue",
    "client": "customername", "clientname": "customername", "customer": "customername",
    "buyer": "customername",
    "status": "orderstatus", "orderstate": "orderstatus", "state": "orderstatus",
    "accountstatus": "accountingstatus", "billingstatus": "accountingstatus",
    "invoicestatus": "accountingstatus",
    "carrier": "outcarriername", "carriername": "outcarriername",
    "trip": "tripno", "tripnumber": "tripno", "tripid": "tripno",
    "miles": "distance", "km": "distance", "kilometers": "distance", "mileage": "distance",
    "po": "pono", "ponumber": "pono", "purchaseorder": "pono", "ordernoexternal": "customerorderno",
    "reference": "referno", "ref": "referno", "referenceno": "referno",
    "salesman": "salesmanname", "salesrep": "salesmanname", "agent": "salesmanname",
    "commodity": "commodityname", "product": "commodityname", "goods": "commodityname",
    "material": "commodityname",
    "company": "companycode", "branch": "companycode",
    "origin": "pickupfulladdress", "shipper": "pickuplocationname",
    "pickup": "pickuplocationname", "pickupcity": "pickupfulladdress",
    "destination": "deliveryfulladdress", "consignee": "deliverylocationname",
    "delivery": "deliverylocationname", "deliverycity": "deliveryfulladdress",
    "drop": "deliverylocationname",
    "tax": "totaltaxamount", "taxamount": "totaltaxamount",
    "fuel": "fuelcharges", "fuelamount": "fuelcharges",
    "currency": "currencycode",
    "orderno": "ordernumber", "ordernum": "ordernumber", "ordernо": "ordernumber",
    "createddate": "createdon", "created": "createdon", "createddatetime": "createdon",
    "modifieddate": "modifiedon", "updatedon": "modifiedon",
    "loadtype": "loadtypelucode", "equipmentkind": "equipmenttype",
    "weightkg": "weight", "qty": "quantity",
}


# ----------------------------------------------------------------- schema
_schema_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_numeric_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and bool(_NUM_STR_RE.match(value.strip()))
    )


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _record_field(fields: Dict[str, Dict[str, Any]], path: str, value: Any) -> None:
    info = fields.setdefault(
        path, {"n": 0, "num": 0, "numstr": 0, "values": set(), "examples": []}
    )
    if value in (None, ""):
        return
    info["n"] += 1
    if _is_number(value):
        info["num"] += 1
    elif _looks_numeric_string(value):
        info["numstr"] += 1
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if len(info["values"]) < 2000:
            info["values"].add(str(value))
        if len(info["examples"]) < 3 and str(value) not in info["examples"]:
            info["examples"].append(str(value)[:40])


def _walk(
    fields: Dict[str, Dict[str, Any]],
    array_prefixes: set,
    prefix: str,
    value: Any,
    depth: int,
) -> None:
    if isinstance(value, dict):
        if depth > _MAX_DEPTH or len(fields) > _MAX_FIELDS:
            return
        for k, v in value.items():
            if k in _SKIP_FIELDS:
                continue
            _walk(fields, array_prefixes, f"{prefix}.{k}" if prefix else k, v, depth + 1)
    elif isinstance(value, list):
        if not value or depth > _MAX_DEPTH:
            return
        first = value[0]
        if isinstance(first, dict):
            if prefix:
                array_prefixes.add(prefix)
            _walk(fields, array_prefixes, prefix, first, depth)
        elif isinstance(first, (str, int, float)) and not isinstance(first, bool):
            _record_field(fields, prefix, first)
    else:
        _record_field(fields, prefix, value)


def _sample_schema(sample_size: int) -> Dict[str, Any]:
    collection = get_orders_collection()
    base = _base_order_match(None)
    size = max(50, int(sample_size or 500))
    try:
        docs = list(
            collection.aggregate(
                [{"$match": base}, {"$sample": {"size": size}}],
                maxTimeMS=AGG_TIMEOUT_MS,
            )
        )
    except Exception as exc:  # $sample unsupported / timeout — plain scan
        checkpoint("DYN_ANALYTICS", "schema $sample failed, using find()", error=str(exc))
        docs = list(collection.find(base).limit(size))

    fields: Dict[str, Dict[str, Any]] = {}
    array_prefixes: set = set()
    for doc in docs:
        _walk(fields, array_prefixes, "", doc, 0)

    schema: Dict[str, Any] = {
        "sample_size": len(docs),
        "fields": {},
        "array_prefixes": sorted(array_prefixes),
    }
    for key, info in fields.items():
        n = info["n"] or 1
        numeric_hits = info["num"] + info["numstr"]
        numeric = numeric_hits > 0 and numeric_hits >= 0.8 * n
        string_numeric = numeric and info["numstr"] >= info["num"]
        distinct = len(info["values"])
        seg = key.split(".")[-1]
        id_like = key in _ID_LIKE_FIELDS or seg in _ID_LIKE_FIELDS or (
            n > 20 and distinct >= 0.9 * n
        )
        in_array = any(key == p or key.startswith(p + ".") for p in array_prefixes)
        groupable = (not id_like) and 1 <= distinct <= GROUPABLE_MAX_DISTINCT
        schema["fields"][key] = {
            "numeric": bool(numeric),
            "string_numeric": bool(string_numeric),
            "distinct_in_sample": distinct,
            "groupable": bool(groupable),
            "id_like": bool(id_like),
            "array": bool(in_array),
            "examples": info["examples"],
        }

    # Normalized-name index for fuzzy resolution
    index: Dict[str, str] = {}
    for real in schema["fields"]:
        index.setdefault(_norm(real), real)
        index.setdefault(_norm(real.split(".")[-1]), real)
    schema["_index"] = index
    return schema


def resolve_field(
    name: Any,
    schema: Dict[str, Any],
    *,
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Map a possibly-misspelled / aliased field name to a real schema field."""
    if not isinstance(name, str) or not name.strip():
        return None
    aliases = aliases if aliases is not None else _FIELD_ALIASES
    fields = schema.get("fields", {})
    if name in fields:
        return name
    index = schema.get("_index") or {}
    n = _norm(name)
    if n in index:
        return index[n]
    alias = aliases.get(n)
    if alias and alias in fields:
        return alias
    # last path segment alias (e.g. "outStatus" -> outstatus)
    if "." in name:
        seg_alias = aliases.get(_norm(name.split(".")[-1]))
        if seg_alias and seg_alias in fields:
            return seg_alias
    import difflib

    close = difflib.get_close_matches(n, list(index.keys()), n=1, cutoff=0.82)
    if close:
        return index[close[0]]
    return None


def _array_prefix_for(field: str, schema: Dict[str, Any]) -> Optional[str]:
    for p in schema.get("array_prefixes", []):
        if field == p or field.startswith(p + "."):
            return p
    return None


def get_orders_schema(*, force: bool = False) -> Dict[str, Any]:
    """Sampled Avaal_order field schema for the active tenant (cached, TTL)."""
    tenant = require_tenant()
    key = (tenant.database, tenant.collection_for("orders"))
    now = time.time()
    cached = _schema_cache.get(key)
    if (
        cached
        and not force
        and "_index" in cached[1]
        and now - cached[0] < SCHEMA_TTL_SECONDS
    ):
        return cached[1]
    schema = _sample_schema(SCHEMA_SAMPLE_SIZE)
    _schema_cache[key] = (now, schema)
    checkpoint(
        "DYN_ANALYTICS",
        "schema built",
        fields=len(schema["fields"]),
        sample=schema["sample_size"],
    )
    return schema


def _schema_for_prompt(
    schema: Dict[str, Any],
    *,
    max_fields: int = 180,
    collection_label: str = "Avaal_order",
) -> str:
    lines = [
        f"{collection_label} fields (sampled {schema.get('sample_size')} docs; "
        "dotted names are nested — you may use them):"
    ]
    items = list(schema.get("fields", {}).items())
    # numeric first, then groupable, then top-level, then the rest
    items.sort(
        key=lambda kv: (
            not kv[1]["numeric"],
            not kv[1]["groupable"],
            "." in kv[0],
            kv[0],
        )
    )
    for name, info in items[:max_fields]:
        tags: List[str] = []
        if info["numeric"]:
            tags.append("numeric")
        if info["groupable"]:
            tags.append(f"groupable(~{info['distinct_in_sample']} distinct)")
        if info["id_like"]:
            tags.append("id-not-groupable")
        if info.get("array"):
            tags.append("array")
        ex = ", ".join(info["examples"][:3])
        lines.append(
            f"- {name}: {', '.join(tags) or 'text'}" + (f" | e.g. {ex}" if ex else "")
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- planner
_PLANNER_PROMPT = """You convert a logistics analytics question into a STRICT JSON aggregation plan over the Avaal_order MongoDB collection.

{schema}

Return ONLY JSON (no prose, no code fence):
{{
  "operation": "group" | "metric" | "count" | "distinct_count" | "unsupported",
  "group_by": ["field", ...],
  "metrics": [{{"fn": "sum|avg|min|max|count", "field": "<numeric field>"}}],
  "distinct_field": "<field>",
  "date_bucket": {{"field": "orderdate", "unit": "day|week|month"}},
  "having": [{{"key": "orders|count|sum_<field>|avg_<field>", "op": "gt|gte|lt|lte", "value": <num>}}],
  "sort": {{"key": "<metric key e.g. avg_weight / sum_totalfreight / count / orders / period, or a group field>", "dir": "asc|desc"}},
  "limit": <int 1-100>
}}
- date_bucket -> daily/weekly/monthly time series (operation "group", group_by may be []).
- having -> keep only groups meeting a threshold ("customers with > 5 orders").

Field notes:
- orderstatus = transport lifecycle (Quoted, Confirmed, Dispatched, In-Transit, Delivered, Cancelled...). Plain "order status" / "status" means THIS field.
- accountingstatus = invoicing / payment (Invoiced, PartiallyPaid, Paid, Restricted).
- outstatus = outsourcing (Open, Planned, Assigned, ...).

Common phrasings (map to the real field): revenue/gross -> grosstotalfreight; amount/price/cost/freight -> totalfreight;
rate -> freightratevalue; client/buyer -> customername; carrier -> outcarriername; trip -> tripno;
miles/km/mileage -> distance; PO -> pono / customerorderno; reference -> referno; salesman/agent -> salesmanname;
commodity/product/goods -> commodityname; tax -> totaltaxamount; fuel -> fuelcharges; created -> createdon.
If the user's word is a near-miss for a field name, the closest real field is used automatically — still, prefer exact names.

Rules:
- Use ONLY field names listed above (dotted nested names allowed). Never invent a field.
- sum/avg/min/max only on fields tagged numeric. group_by only on fields tagged groupable, never id-not-groupable.
- 1-2 group_by fields. Use 2 when the user asks to break down by two dimensions
  (e.g. "order status and accounting status wise" -> group_by ["orderstatus","accountingstatus"]).
- Metric key = fn + "_" + field (e.g. avg_weight), or "count".
- operation "count" = number of matching orders. "metric" = totals/averages with NO grouping.
  "group" = per-group metrics. "distinct_count" = number of distinct values of distinct_field.
- Do NOT include filters (status / customer / date / city / state / country) — those are applied separately.
- Return {{"operation": "unsupported"}} when the question is about: grouping by state/city/country/pincode,
  activity on a date or "today"/"last month"/"last N days", best/worst customer ranking, or fleet trip-distance.

Question: {question}
JSON:"""


def _plan(question: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    llm = get_anthropic_llm()
    chain = PromptTemplate.from_template(_PLANNER_PROMPT) | llm
    raw = chain.invoke(
        {"question": question, "schema": _schema_for_prompt(schema)}
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


# ----------------------------------------------------------------- validate
def _clamp_limit(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 20
    return max(1, min(MAX_LIMIT, n))


def _clean_sort(
    sort: Any, valid_keys: Optional[set] = None
) -> Optional[Dict[str, str]]:
    if not isinstance(sort, dict) or not isinstance(sort.get("key"), str):
        return None
    key = sort["key"]
    if valid_keys is not None and key not in valid_keys:
        return None
    direction = "asc" if str(sort.get("dir") or "desc").lower() == "asc" else "desc"
    return {"key": key, "dir": direction}


def _validate_spec(
    spec: Any,
    schema: Dict[str, Any],
    *,
    iso_date_fields: Optional[frozenset] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict):
        return None
    op = str(spec.get("operation") or "").lower()
    if op not in _ALLOWED_OPS:
        return None

    iso_date_fields = iso_date_fields if iso_date_fields is not None else _ISO_DATE_FIELDS
    fields = schema.get("fields", {})
    unwind: set = set()

    def R(name: Any) -> Optional[str]:
        return resolve_field(name, schema, aliases=aliases)

    out: Dict[str, Any] = {"operation": op}

    if op == "count":
        return out

    if op == "distinct_count":
        target = R(spec.get("distinct_field"))
        if not target:
            return None
        out["distinct_field"] = target
        pfx = _array_prefix_for(target, schema)
        if pfx:
            unwind.add(pfx)
        group_by = []
        for g in (spec.get("group_by") or []):
            rg = R(g)
            if rg and fields[rg]["groupable"]:
                group_by.append(rg)
                gp = _array_prefix_for(rg, schema)
                if gp:
                    unwind.add(gp)
        group_by = group_by[:MAX_GROUP_BY]
        out["group_by"] = group_by
        out["limit"] = _clamp_limit(spec.get("limit"))
        out["sort"] = _clean_sort(spec.get("sort"), {"distinct_count", *group_by})
        out["_unwind"] = sorted(unwind)
        return out

    # metric / group
    metrics: List[Dict[str, Any]] = []
    for raw in (spec.get("metrics") or [])[:MAX_METRICS]:
        if not isinstance(raw, dict):
            continue
        fn = str(raw.get("fn") or "").lower()
        if fn not in _ALLOWED_FNS:
            continue
        if fn == "count":
            metrics.append({"fn": "count", "field": None, "key": "count"})
            continue
        field = R(raw.get("field"))
        if not field or not fields[field]["numeric"]:
            continue
        pfx = _array_prefix_for(field, schema)
        if pfx:
            unwind.add(pfx)
        metrics.append(
            {"fn": fn, "field": field, "key": f"{fn}_{_norm(field)}"}
        )

    seen: set = set()
    metrics = [m for m in metrics if not (m["key"] in seen or seen.add(m["key"]))]
    if not metrics:
        metrics = [{"fn": "count", "field": None, "key": "count"}]
    out["metrics"] = metrics

    has_bucket = isinstance(spec.get("date_bucket"), dict) and bool(
        spec["date_bucket"].get("field")
    )
    if op == "group":
        group_by = []
        for g in (spec.get("group_by") or []):
            rg = R(g)
            if rg and fields[rg]["groupable"]:
                group_by.append(rg)
                gp = _array_prefix_for(rg, schema)
                if gp:
                    unwind.add(gp)
        group_by = group_by[:MAX_GROUP_BY]
        if not group_by and not has_bucket:
            return None
        out["group_by"] = group_by

    # date_bucket = a synthetic daily/weekly/monthly time dimension
    db = spec.get("date_bucket")
    if isinstance(db, dict):
        fld = R(db.get("field"))
        unit = str(db.get("unit") or "day").lower()
        if fld and unit in _BUCKET_UNITS and (
            fld in iso_date_fields or fld.split(".")[-1] in iso_date_fields
        ):
            out["date_bucket"] = {"field": fld, "unit": unit}
            if op == "metric":
                out["operation"] = op = "group"
                out.setdefault("group_by", [])

    out["limit"] = _clamp_limit(spec.get("limit"))
    metric_keys = {m["key"] for m in metrics}

    # HAVING — post-group threshold on a metric / count / orders
    having: List[Dict[str, Any]] = []
    for h in (spec.get("having") or [])[:4]:
        if not isinstance(h, dict):
            continue
        raw_key = str(h.get("key") or "").lower()
        mop = _HAVING_OPS.get(str(h.get("op") or "").lower())
        try:
            hval = float(h.get("value"))
        except (TypeError, ValueError):
            continue
        if raw_key in ("orders", "order_count", "count_orders"):
            pkey = "_matched"
        elif raw_key in ("count", "distinct_count"):
            pkey = raw_key
        elif raw_key in metric_keys:
            pkey = raw_key
        elif R(raw_key) and f"sum_{_norm(R(raw_key))}" in metric_keys:
            pkey = f"sum_{_norm(R(raw_key))}"
        else:
            continue
        if mop:
            having.append({pkey: {mop: hval}})
    if having:
        out["having"] = having

    valid_keys = metric_keys | set(out.get("group_by", [])) | {"count", "orders"}
    if out.get("date_bucket"):
        valid_keys.add("period")
    out["sort"] = _clean_sort(spec.get("sort"), valid_keys)
    out["_unwind"] = sorted(unwind)
    return out


# ----------------------------------------------------------------- builder
def _numeric_expr(field: str) -> Dict[str, Any]:
    """Best-effort numeric coercion; non-numeric values are ignored by $sum/$avg."""
    return {
        "$convert": {
            "input": f"${field}",
            "to": "double",
            "onError": None,
            "onNull": None,
        }
    }


def _group_key_expr(field: str) -> Dict[str, Any]:
    """Collapse null / empty group-key values to 'Unknown' so rows read cleanly."""
    ref = f"${field}"
    return {
        "$let": {
            "vars": {"v": ref},
            "in": {"$cond": [{"$in": ["$$v", [None, ""]]}, "Unknown", "$$v"]},
        }
    }


def _group_sort_doc(
    sort: Optional[Dict[str, str]], default_key: str, alias: Dict[str, str]
) -> Dict[str, int]:
    """Sort a grouped result; a group field maps to its _id.<alias> path."""
    if isinstance(sort, dict) and isinstance(sort.get("key"), str):
        key = sort["key"]
        direction = 1 if sort.get("dir") == "asc" else -1
        if key in alias:
            return {f"_id.{alias[key]}": direction}
        if key == "period":
            return {"_id.__bucket": direction}
        if key in ("orders", "order_count"):
            return {"_matched": direction}
        return {key: direction}
    return {default_key: -1}


def _bucket_expr(field: str, unit: str) -> Dict[str, Any]:
    """Truncate an ISO date-string field to a day / week / month label."""
    src = {"$toString": f"${field}"}
    if unit == "day":
        return {"$substrBytes": [src, 0, 10]}          # 2026-08-28
    if unit == "month":
        return {"$substrBytes": [src, 0, 7]}           # 2026-08
    # week — ISO year-week (needs $toDate; ok on ISO strings)
    return {
        "$dateToString": {
            "format": "%G-W%V",
            "date": {"$toDate": f"${field}"},
        }
    }


def _build_pipeline(
    spec: Dict[str, Any], match: Dict[str, Any]
) -> List[Dict[str, Any]]:
    op = spec["operation"]
    pipeline: List[Dict[str, Any]] = [{"$match": match}]

    if op == "count":
        pipeline.append({"$count": "count"})
        return pipeline

    # Flatten array-of-object paths (commoditydetails[], ...) before grouping.
    for prefix in spec.get("_unwind") or []:
        pipeline.append({"$unwind": f"${prefix}"})

    if op == "distinct_count":
        target = spec["distinct_field"]
        group_by = spec.get("group_by") or []
        alias = {g: _norm(g) for g in group_by}
        if group_by:
            pipeline += [
                {
                    "$group": {
                        "_id": {
                            **{alias[g]: _group_key_expr(g) for g in group_by},
                            "__v": f"${target}",
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {alias[g]: f"$_id.{alias[g]}" for g in group_by},
                        "distinct_count": {"$sum": 1},
                    }
                },
                {"$sort": _group_sort_doc(spec.get("sort"), "distinct_count", alias)},
                {"$limit": spec["limit"]},
            ]
        else:
            pipeline += [
                {"$group": {"_id": f"${target}"}},
                {"$count": "distinct_count"},
            ]
        return pipeline

    # metric / group
    metrics = spec["metrics"]
    coerce = {m["field"] for m in metrics if m["field"]}

    add_fields: Dict[str, Any] = {
        f"__num_{_norm(f)}": _numeric_expr(f) for f in coerce
    }
    bucket = spec.get("date_bucket")
    if bucket:
        add_fields["__bucket"] = _bucket_expr(bucket["field"], bucket["unit"])
    if add_fields:
        pipeline.append({"$addFields": add_fields})

    group_by = spec.get("group_by") or []
    alias = {g: _norm(g) for g in group_by}
    group_id: Optional[Dict[str, Any]] = None
    if op == "group":
        group_id = {alias[g]: _group_key_expr(g) for g in group_by}
        if bucket:
            group_id["__bucket"] = "$__bucket"

    group_stage: Dict[str, Any] = {"_id": group_id, "_matched": {"$sum": 1}}
    for m in metrics:
        if m["fn"] == "count":
            group_stage["count"] = {"$sum": 1}
        else:
            group_stage[m["key"]] = {f"${m['fn']}": f"$__num_{_norm(m['field'])}"}
    pipeline.append({"$group": group_stage})

    if spec.get("having"):
        pipeline.append({"$match": {"$and": spec["having"]}})

    if op == "group":
        default_sort = "_id.__bucket" if bucket else metrics[0]["key"]
        if bucket and not spec.get("sort"):
            pipeline.append({"$sort": {"_id.__bucket": 1}})
        else:
            pipeline.append(
                {"$sort": _group_sort_doc(spec.get("sort"), default_sort, alias)}
            )
        pipeline.append({"$limit": spec["limit"]})

    return pipeline


def _round(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        rounded = round(float(value), 4)
        return int(rounded) if rounded.is_integer() else rounded
    return value


def _shape_result(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    filters: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    op = spec["operation"]
    payload: Dict[str, Any] = {
        "analytics_type": "dynamic",
        "engine": "dynamic_planner",
        "operation": op,
        "filters": filters or {},
        "question": question,
    }

    if op == "count":
        payload["matching_orders"] = int(rows[0]["count"]) if rows else 0
        return payload

    if op == "distinct_count":
        payload["distinct_field"] = spec["distinct_field"]
        if spec.get("group_by"):
            gb = spec["group_by"]
            alias = {g: _norm(g) for g in gb}
            payload["group_by"] = gb
            payload["rows"] = [
                {
                    **{g: (r.get("_id") or {}).get(alias[g]) for g in gb},
                    "distinct_count": int(r.get("distinct_count") or 0),
                }
                for r in rows
            ]
        else:
            payload["distinct_count"] = (
                int(rows[0]["distinct_count"]) if rows else 0
            )
        return payload

    metric_keys = [m["key"] for m in spec["metrics"]]
    payload["metrics"] = metric_keys

    if op == "metric":
        row = rows[0] if rows else {}
        payload["values"] = {k: _round(row.get(k)) for k in metric_keys}
        payload["matching_orders"] = int(row.get("_matched") or 0)
        return payload

    gb = spec.get("group_by") or []
    alias = {g: _norm(g) for g in gb}
    bucket = spec.get("date_bucket")
    payload["group_by"] = (["period"] if bucket else []) + gb
    if bucket:
        payload["date_bucket"] = bucket
    if spec.get("having"):
        payload["having"] = spec["having"]
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        rid = r.get("_id") or {}
        row: Dict[str, Any] = {}
        if bucket:
            row["period"] = rid.get("__bucket")
        for g in gb:
            row[g] = rid.get(alias[g])
        for k in metric_keys:
            row[k] = _round(r.get(k))
        row["orders"] = int(r.get("_matched") or 0)
        out_rows.append(row)
    payload["rows"] = out_rows
    return payload


# ----------------------------------------------------------------- entry
def run_dynamic_analytics(
    question: str,
    *,
    entities: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Plan + run a dynamic aggregation. Returns None to fall back to the catalog."""
    if not DYNAMIC_ANALYTICS_ENABLED:
        return None
    q = (question or "").strip()
    if not q:
        return None
    if _DEFER_RE.search(q):
        checkpoint("DYN_ANALYTICS", "defer to catalog engine", reason="pattern_match")
        return None

    entities = entities or {}
    try:
        schema = get_orders_schema()
        if not schema.get("fields"):
            return None

        spec_raw = _plan(q, schema)
        spec = _validate_spec(spec_raw, schema)
        checkpoint("DYN_ANALYTICS", "planned", raw=spec_raw, valid=spec)
        if spec is None:
            return None

        filters = entities_to_mongo_filters(entities, domain="orders")
        match = _base_order_match(filters)
        pipeline = _build_pipeline(spec, match)

        collection = get_orders_collection()
        rows = list(
            collection.aggregate(
                pipeline, maxTimeMS=AGG_TIMEOUT_MS, allowDiskUse=False
            )
        )
        payload = _shape_result(spec, rows, filters, q)
        checkpoint(
            "DYN_ANALYTICS",
            "done",
            operation=spec["operation"],
            rows=len(payload.get("rows") or []),
            matching=payload.get("matching_orders"),
        )
        return payload
    except Exception as exc:
        checkpoint("DYN_ANALYTICS", "FAILED — fallback to catalog", error=str(exc))
        return None


def format_dynamic_analytics_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text ground-truth block for the answer LLM."""
    lines = [
        "ANALYTICS RESULT (dynamic aggregation engine — exact Mongo output, "
        "do not recalculate or invent):",
        f"operation: {payload.get('operation')}",
    ]
    if payload.get("filters"):
        lines.append(f"filters: {payload['filters']}")

    op = payload.get("operation")
    if op == "count":
        lines.append(f"matching_orders: {payload.get('matching_orders')}")

    elif op == "distinct_count":
        if payload.get("rows") is not None:
            lines.append(
                f"distinct {payload.get('distinct_field')} count "
                f"by {payload.get('group_by')}:"
            )
            for row in payload["rows"]:
                lines.append(f"- {row}")
            if not payload["rows"]:
                lines.append("(no rows matched)")
        else:
            lines.append(
                f"distinct {payload.get('distinct_field')} count: "
                f"{payload.get('distinct_count')}"
            )

    elif op == "metric":
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        for key, value in (payload.get("values") or {}).items():
            lines.append(f"- {key}: {value}")

    elif op == "group":
        if payload.get("having"):
            lines.append(f"having (post-group filter): {payload['having']}")
        lines.append(
            f"grouped by {payload.get('group_by')}, "
            f"metrics {payload.get('metrics')} (orders = row count):"
        )
        for row in payload.get("rows") or []:
            lines.append(
                "- " + " | ".join(f"{k}={v}" for k, v in row.items())
            )
        if not payload.get("rows"):
            lines.append("(no rows matched these filters)")

    elif op == "percentage":
        lines.append(f"basis: {payload.get('of')}")
        if payload.get("numerator_filters"):
            lines.append(f"numerator condition: {payload['numerator_filters']}")
        lines.append(f"numerator: {payload.get('numerator')}")
        lines.append(f"denominator (total): {payload.get('denominator')}")
        lines.append(f"percentage: {payload.get('percentage')}%")

    elif op == "compare":
        lines.append(f"metric: {payload.get('metric')}")
        for row in payload.get("rows") or []:
            lines.append(
                "- " + " | ".join(f"{k}={v}" for k, v in row.items())
            )
        if not payload.get("rows"):
            lines.append("(no segments matched)")

    lines.append("Use these numbers as ground truth. Write numbers without commas.")
    return "\n".join(lines)
