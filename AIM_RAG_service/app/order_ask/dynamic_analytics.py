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
    for doc in docs:
        for key, value in doc.items():
            if key in _SKIP_FIELDS or "." in key:
                continue
            info = fields.setdefault(
                key,
                {"n": 0, "num": 0, "numstr": 0, "values": set(), "examples": []},
            )
            if value in (None, ""):
                continue
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

    schema: Dict[str, Any] = {"sample_size": len(docs), "fields": {}}
    for key, info in fields.items():
        n = info["n"] or 1
        numeric_hits = info["num"] + info["numstr"]
        numeric = numeric_hits > 0 and numeric_hits >= 0.8 * n
        string_numeric = numeric and info["numstr"] >= info["num"]
        distinct = len(info["values"])
        id_like = key in _ID_LIKE_FIELDS or (n > 20 and distinct >= 0.9 * n)
        groupable = (not id_like) and 1 <= distinct <= GROUPABLE_MAX_DISTINCT
        schema["fields"][key] = {
            "numeric": bool(numeric),
            "string_numeric": bool(string_numeric),
            "distinct_in_sample": distinct,
            "groupable": bool(groupable),
            "id_like": bool(id_like),
            "examples": info["examples"],
        }
    return schema


def get_orders_schema(*, force: bool = False) -> Dict[str, Any]:
    """Sampled Avaal_order field schema for the active tenant (cached, TTL)."""
    tenant = require_tenant()
    key = (tenant.database, tenant.collection_for("orders"))
    now = time.time()
    cached = _schema_cache.get(key)
    if cached and not force and now - cached[0] < SCHEMA_TTL_SECONDS:
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


def _schema_for_prompt(schema: Dict[str, Any], *, max_fields: int = 90) -> str:
    lines = [f"Avaal_order fields (sampled {schema.get('sample_size')} docs):"]
    items = list(schema.get("fields", {}).items())
    # numeric first, then groupable, then the rest — most useful to the planner
    items.sort(key=lambda kv: (not kv[1]["numeric"], not kv[1]["groupable"], kv[0]))
    for name, info in items[:max_fields]:
        tags: List[str] = []
        if info["numeric"]:
            tags.append("numeric")
        if info["groupable"]:
            tags.append(f"groupable(~{info['distinct_in_sample']} distinct)")
        if info["id_like"]:
            tags.append("id-not-groupable")
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
  "sort": {{"key": "<metric key e.g. avg_weight / sum_totalfreight / count, or a group field>", "dir": "asc|desc"}},
  "limit": <int 1-100>
}}

Field notes:
- orderstatus = transport lifecycle (Quoted, Confirmed, Dispatched, In-Transit, Delivered, Cancelled...). Plain "order status" / "status" means THIS field.
- accountingstatus = invoicing / payment (Invoiced, PartiallyPaid, Paid, Restricted).
- outstatus = outsourcing (Open, Planned, Assigned, ...).

Rules:
- Use ONLY field names listed above. Never invent a field.
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
    spec: Any, schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict):
        return None
    op = str(spec.get("operation") or "").lower()
    if op not in _ALLOWED_OPS:
        return None

    fields = schema.get("fields", {})

    def known(name: Any) -> bool:
        return isinstance(name, str) and name in fields

    out: Dict[str, Any] = {"operation": op}

    if op == "count":
        return out

    if op == "distinct_count":
        target = spec.get("distinct_field")
        if not known(target):
            return None
        out["distinct_field"] = target
        group_by = [
            g
            for g in (spec.get("group_by") or [])
            if known(g) and fields[g]["groupable"]
        ][:MAX_GROUP_BY]
        out["group_by"] = group_by
        out["limit"] = _clamp_limit(spec.get("limit"))
        out["sort"] = _clean_sort(
            spec.get("sort"), {"distinct_count", *group_by}
        )
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
        field = raw.get("field")
        if not known(field) or not fields[field]["numeric"]:
            continue
        metrics.append({"fn": fn, "field": field, "key": f"{fn}_{field}"})

    # de-dupe metric keys, keep order
    seen: set = set()
    metrics = [m for m in metrics if not (m["key"] in seen or seen.add(m["key"]))]
    if not metrics:
        metrics = [{"fn": "count", "field": None, "key": "count"}]
    out["metrics"] = metrics

    if op == "group":
        group_by = [
            g
            for g in (spec.get("group_by") or [])
            if known(g) and fields[g]["groupable"]
        ][:MAX_GROUP_BY]
        if not group_by:
            return None
        out["group_by"] = group_by

    out["limit"] = _clamp_limit(spec.get("limit"))
    valid_keys = {m["key"] for m in metrics} | set(out.get("group_by", []))
    out["sort"] = _clean_sort(spec.get("sort"), valid_keys)
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


def _sort_doc(sort: Optional[Dict[str, str]], default_key: str) -> Dict[str, int]:
    if isinstance(sort, dict) and isinstance(sort.get("key"), str):
        return {sort["key"]: 1 if sort.get("dir") == "asc" else -1}
    return {default_key: -1}


def _group_key_expr(field: str) -> Dict[str, Any]:
    """Collapse null / empty group-key values to 'Unknown' so rows read cleanly."""
    ref = f"${field}"
    return {
        "$let": {
            "vars": {"v": ref},
            "in": {"$cond": [{"$in": ["$$v", [None, ""]]}, "Unknown", "$$v"]},
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

    if op == "distinct_count":
        target = spec["distinct_field"]
        group_by = spec.get("group_by") or []
        if group_by:
            pipeline += [
                {
                    "$group": {
                        "_id": {
                            **{g: _group_key_expr(g) for g in group_by},
                            "__v": f"${target}",
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {g: f"$_id.{g}" for g in group_by},
                        "distinct_count": {"$sum": 1},
                    }
                },
                {"$sort": _sort_doc(spec.get("sort"), "distinct_count")},
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
    if coerce:
        pipeline.append(
            {"$addFields": {f"__num_{f}": _numeric_expr(f) for f in coerce}}
        )

    group_id = (
        {g: _group_key_expr(g) for g in spec["group_by"]} if op == "group" else None
    )
    group_stage: Dict[str, Any] = {"_id": group_id, "_matched": {"$sum": 1}}
    for m in metrics:
        if m["fn"] == "count":
            group_stage["count"] = {"$sum": 1}
        else:
            group_stage[m["key"]] = {f"${m['fn']}": f"$__num_{m['field']}"}
    pipeline.append({"$group": group_stage})

    if op == "group":
        pipeline.append(
            {"$sort": _sort_doc(spec.get("sort"), metrics[0]["key"])}
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
            payload["group_by"] = spec["group_by"]
            payload["rows"] = [
                {
                    **{g: (r.get("_id") or {}).get(g) for g in spec["group_by"]},
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

    payload["group_by"] = spec["group_by"]
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        rid = r.get("_id") or {}
        row = {g: rid.get(g) for g in spec["group_by"]}
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

    lines.append("Use these numbers as ground truth. Write numbers without commas.")
    return "\n".join(lines)
