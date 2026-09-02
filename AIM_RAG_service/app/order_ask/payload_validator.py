"""
Payload validator — the gate every LLM-produced query payload passes
through before anything touches the database.

Pure code, no LLM, no DB. `validate_payload()` returns a `ValidatedQuery`
only if the payload is completely safe; otherwise it returns `None` and
logs the rejected payload. It never partial-passes and never "best
efforts" a broken payload into something runnable.

Payload schema produced by `llm_query_builder.build_query_payload()`
(this file is the single source of truth for what is accepted):

    {
      "kind": "procedure" | "dynamic" | "unsupported",

      # ---- kind == "procedure" ----
      "procedure": "regular" | "count" | "freight" | "summary",
      "order_id":     <int>,      # regular only — direct PK lookup
      "order_number": "<str>",    # regular only — partial match
      "customer_code": "<str>",   # regular only
      "group_by": "day"|"week"|"status"|"delivery_city"|"pickup_city"|"country",
                                  # count / freight only
      "top_n": <int>,             # count / freight only — app slices first N
      "limit": <int>,             # regular (page size) or summary (row cap);
                                  #   capped at 500

      # ---- kind == "dynamic" ----
      "table": "trnorder",
      "select_fields": ["ordernumber", ...],
      "dynamic_filters": [{"field": "<col>", "op": "<op>", "value": <v>}],
      "group_by_column": "<col>",
      "aggregate": "count"|"sum"|"avg"|"min"|"max",
      "aggregate_field": "<numeric col>",
      "limit": <int>,

      # ---- shared semantic filters (procedure path only) ----
      "date_field": "created"|"order"|"pickup"|"delivery",
      "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "last_n_days": <int>,
      "order_status": "<str>",
      "pickup_country": "<code>",
      "delivery_country": "<code>",
      "customer_code": "<str>",    # regular only

      "reason": "<short string>"   # ignored by the validator, kept for logs
    }

`kind == "unsupported"` is handled by the caller before the validator is
reached (it never produces a `ValidatedQuery`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.order_ask import query_allowlist as qa
from app.order_ask.checkpoint import checkpoint
from app.postgres_client import _ORDER_FN_PARAM_ORDER

logger = logging.getLogger("order_ask.payload_validator")

# procedure short name -> real Postgres function name (see query_allowlist)
_PROC_ALIAS = {
    "regular": "fn_getorders_regular_ai",
    "count": "fn_getorders_aggregate_count_ai",
    "freight": "fn_getorders_aggregate_freight_ai",
    "summary": "fn_getorders_summary_ai",
}
_AGG_KINDS = {"count", "freight"}

_DATE_FIELD_TO_DATETYPE = {
    "created": "CD",
    "order": "OD",
    "pickup": "PD",   # NEW in the _ai function (patch B) — not in the old one
    "delivery": "DD",  # NEW
}
_AGG_GROUP_BY = {"day", "week", "status", "delivery_city", "pickup_city", "country"}

# Real p_* params the two aggregate _ai functions accept (per the contract).
_AGG_FN_PARAMS = {
    "p_companycode", "p_isdate", "p_datetype", "p_fromdate", "p_todate",
    "p_orderstatus", "p_pickupcountrycode", "p_deliverycountrycode",
    "p_status", "p_groupby",
}
# fn_getorders_summary_ai — same filter set as the aggregates, but no
# p_groupby; instead p_limit and a fixed useful column set in the result.
_SUMMARY_FN_PARAMS = {
    "p_companycode", "p_isdate", "p_datetype", "p_fromdate", "p_todate",
    "p_orderstatus", "p_pickupcountrycode", "p_deliverycountrycode",
    "p_status", "p_limit",
}
_REGULAR_FN_PARAMS = set(_ORDER_FN_PARAM_ORDER)

# For the dynamic-SQL path we currently only assemble single-table queries
# against trnorder (no cross-table join graph exists in the allowlist).
_DYNAMIC_TABLES = {"trnorder"}

_ALLOWED_TOP_LEVEL_KEYS = {
    "kind", "procedure", "order_id", "order_number", "group_by", "top_n",
    "table", "select_fields", "dynamic_filters", "group_by_column",
    "aggregate", "aggregate_field", "limit",
    "date_field", "date_range", "last_n_days", "order_status",
    "pickup_country", "delivery_country", "customer_code", "reason",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_COUNTRY_RE = re.compile(r"^[A-Za-z]{2,3}$")
_NUMERIC_TYPES = {"numeric", "integer", "bigint", "double precision", "real"}


@dataclass
class ValidatedQuery:
    kind: str  # "procedure" | "dynamic"
    # procedure path
    procedure_name: Optional[str] = None
    procedure_kind: Optional[str] = None  # regular | count | freight
    procedure_params: Optional[Dict[str, Any]] = None
    top_n: Optional[int] = None
    # dynamic path
    table: Optional[str] = None
    select_fields: Optional[List[str]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    group_by: Optional[str] = None
    aggregate: Optional[str] = None
    aggregate_field: Optional[str] = None
    limit: int = qa.MAX_ROW_LIMIT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _reject(question: str, payload: Any, reason: str) -> None:
    checkpoint(
        "QUERYBUILD",
        "payload REJECTED",
        reason=reason,
        question=(question or "")[:120],
    )
    logger.warning(
        "payload rejected — reason=%s | question=%s | payload=%s",
        reason, question, payload,
    )
    return None


def _valid_date(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip()[:10]
    if not _DATE_RE.match(v):
        return None
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        return None
    return v


def _clamp_limit(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return qa.MAX_ROW_LIMIT
    return max(1, min(qa.MAX_ROW_LIMIT, n))


def _scalar_matches_type(pg_type: str, value: Any) -> bool:
    if pg_type in ("numeric", "double precision", "real"):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    if pg_type in ("integer", "bigint"):
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False
    if pg_type == "boolean":
        return isinstance(value, bool) or str(value).lower() in ("true", "false")
    if "timestamp" in pg_type or pg_type == "date":
        return _valid_date(str(value)) is not None
    # character varying / text / character
    return isinstance(value, (str, int, float))


def _shared_filter_params(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Translate the semantic filters (date_range / last_n_days / countries /
    status) into real p_* params. Returns {} when none are set, or None on
    any malformed value (caller then rejects the whole payload).
    """
    params: Dict[str, Any] = {}

    date_range = payload.get("date_range")
    last_n = payload.get("last_n_days")
    if date_range is not None and last_n is not None:
        return None  # ambiguous — one or the other, never both

    if date_range is not None:
        if not (isinstance(date_range, list) and len(date_range) == 2):
            return None
        d1, d2 = _valid_date(date_range[0]), _valid_date(date_range[1])
        if not d1 or not d2:
            return None
        lo, hi = sorted([d1, d2])
        df = str(payload.get("date_field") or "created").lower()
        dt = _DATE_FIELD_TO_DATETYPE.get(df)
        if not dt:
            return None
        params["p_isdate"] = True
        params["p_datetype"] = dt
        params["p_fromdate"] = lo
        params["p_todate"] = hi
    elif last_n is not None:
        try:
            n = int(last_n)
        except (TypeError, ValueError):
            return None
        if n <= 0 or n > 3660:
            return None
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=n - 1)
        df = str(payload.get("date_field") or "created").lower()
        dt = _DATE_FIELD_TO_DATETYPE.get(df)
        if not dt:
            return None
        params["p_isdate"] = True
        params["p_datetype"] = dt
        params["p_fromdate"] = start.strftime("%Y-%m-%d")
        params["p_todate"] = today.strftime("%Y-%m-%d")
    elif payload.get("date_field") is not None:
        # date_field with no range/last_n is meaningless
        return None

    for key, param in (
        ("pickup_country", "p_pickupcountrycode"),
        ("delivery_country", "p_deliverycountrycode"),
    ):
        val = payload.get(key)
        if val is not None:
            if not isinstance(val, str) or not _COUNTRY_RE.match(val.strip()):
                return None
            params[param] = val.strip().upper()

    status = payload.get("order_status")
    if status is not None:
        if not isinstance(status, str) or not status.strip():
            return None
        params["p_orderstatus"] = status.strip()

    return params


# ---------------------------------------------------------------------------
# procedure path
# ---------------------------------------------------------------------------
def _validate_procedure(payload: Dict[str, Any], question: str) -> Optional[ValidatedQuery]:
    proc = str(payload.get("procedure") or "").lower()
    fn_name = _PROC_ALIAS.get(proc)
    if not fn_name or fn_name not in qa.ALLOWED_PROCEDURES:
        return _reject(question, payload, f"unknown procedure {proc!r}")

    shared = _shared_filter_params(payload)
    if shared is None:
        return _reject(question, payload, "malformed semantic filter(s)")

    params: Dict[str, Any] = dict(shared)
    top_n: Optional[int] = None

    if proc == "regular":
        if payload.get("group_by") is not None or payload.get("top_n") is not None:
            return _reject(question, payload, "group_by/top_n not valid for 'regular'")

        oid = payload.get("order_id")
        if oid is not None:
            try:
                params["p_orderid"] = int(oid)
            except (TypeError, ValueError):
                return _reject(question, payload, "order_id not an int")

        onum = payload.get("order_number")
        if onum is not None:
            if not isinstance(onum, str) or not onum.strip():
                return _reject(question, payload, "order_number not a string")
            params["p_ordernumber"] = onum.strip()

        cust = payload.get("customer_code")
        if cust is not None:
            if not isinstance(cust, str) or not cust.strip():
                return _reject(question, payload, "customer_code not a string")
            params["p_customercode"] = cust.strip()

        if payload.get("limit") is not None:
            params["p_pagesize"] = _clamp_limit(payload.get("limit"))

        bad = set(params) - _REGULAR_FN_PARAMS
        if bad:
            return _reject(question, payload, f"unknown regular params: {sorted(bad)}")

    elif proc == "summary":
        for banned in ("order_id", "order_number", "customer_code", "group_by", "top_n"):
            if payload.get(banned) is not None:
                return _reject(question, payload, f"{banned} not valid for 'summary'")
        if payload.get("limit") is not None:
            params["p_limit"] = _clamp_limit(payload.get("limit"))

        bad = set(params) - _SUMMARY_FN_PARAMS
        if bad:
            return _reject(question, payload, f"unknown summary params: {sorted(bad)}")

    else:  # count | freight
        for banned in ("order_id", "order_number", "customer_code", "limit"):
            if payload.get(banned) is not None:
                return _reject(question, payload, f"{banned} not valid for aggregate procedure")

        gb = payload.get("group_by")
        if gb is not None:
            if gb not in _AGG_GROUP_BY:
                return _reject(question, payload, f"bad group_by {gb!r}")
            params["p_groupby"] = gb

        tn = payload.get("top_n")
        if tn is not None:
            try:
                top_n = int(tn)
            except (TypeError, ValueError):
                return _reject(question, payload, "top_n not an int")
            if top_n <= 0:
                return _reject(question, payload, "top_n must be positive")

        bad = set(params) - _AGG_FN_PARAMS
        if bad:
            return _reject(question, payload, f"unknown aggregate params: {sorted(bad)}")

    return ValidatedQuery(
        kind="procedure",
        procedure_name=fn_name,
        procedure_kind=proc,
        procedure_params=params,
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# dynamic path
# ---------------------------------------------------------------------------
def _validate_dynamic(payload: Dict[str, Any], question: str) -> Optional[ValidatedQuery]:
    table = payload.get("table")
    if table not in _DYNAMIC_TABLES:
        return _reject(
            question, payload,
            f"dynamic queries currently limited to {sorted(_DYNAMIC_TABLES)}, got {table!r}",
        )

    # semantic filters are procedure-only — dynamic must spell everything out
    for k in ("date_range", "last_n_days", "date_field", "pickup_country",
              "delivery_country", "order_status", "customer_code",
              "procedure", "order_id", "order_number", "group_by", "top_n"):
        if payload.get(k) is not None:
            return _reject(question, payload, f"{k} not valid on a dynamic query")

    # SELECT / GROUP BY may only name real output columns — join keys such
    # as `orderid` pass is_column_allowed() (they're needed for filtering)
    # but must never be selectable, per query_allowlist's own contract.
    real_columns = set(qa.ALLOWED_COLUMNS.get(table, []))

    select_fields: List[str] = []
    for col in payload.get("select_fields") or []:
        if not isinstance(col, str) or col not in real_columns:
            return _reject(question, payload, f"disallowed select column {col!r}")
        select_fields.append(col)

    filters: List[Dict[str, Any]] = []
    for f in payload.get("dynamic_filters") or []:
        if not isinstance(f, dict):
            return _reject(question, payload, "filter is not an object")
        field = f.get("field")
        op = str(f.get("op") or "").lower()
        value = f.get("value")
        if not isinstance(field, str) or not qa.is_column_allowed(table, field):
            return _reject(question, payload, f"disallowed filter field {field!r}")
        if not qa.is_operator_allowed(op):
            return _reject(question, payload, f"disallowed operator {op!r}")
        pg_type = qa.get_column_type(table, field)
        if pg_type is None:
            return _reject(question, payload, f"no confirmed type for {field!r}")
        if op == "between":
            if not (isinstance(value, list) and len(value) == 2
                    and all(_scalar_matches_type(pg_type, v) for v in value)):
                return _reject(question, payload, f"bad BETWEEN value for {field!r}")
        elif op == "in":
            if not (isinstance(value, list) and value
                    and all(_scalar_matches_type(pg_type, v) for v in value)):
                return _reject(question, payload, f"bad IN value for {field!r}")
        elif op == "ilike":
            if not isinstance(value, str):
                return _reject(question, payload, f"ILIKE value must be a string for {field!r}")
        else:
            if not _scalar_matches_type(pg_type, value):
                return _reject(question, payload, f"value type mismatch for {field!r} ({pg_type})")
        filters.append({"field": field, "op": op, "value": value, "type": pg_type})

    group_by = payload.get("group_by_column")
    if group_by is not None:
        if not isinstance(group_by, str) or group_by not in real_columns:
            return _reject(question, payload, f"disallowed group_by column {group_by!r}")

    aggregate = payload.get("aggregate")
    aggregate_field = payload.get("aggregate_field")
    if aggregate is not None:
        aggregate = str(aggregate).lower()
        if not qa.is_aggregate_allowed(aggregate):
            return _reject(question, payload, f"disallowed aggregate {aggregate!r}")
        if aggregate == "count":
            aggregate_field = None
        else:
            if not isinstance(aggregate_field, str) or aggregate_field not in real_columns:
                return _reject(question, payload, f"aggregate {aggregate} needs a valid field")
            if qa.get_column_type(table, aggregate_field) not in _NUMERIC_TYPES:
                return _reject(question, payload, f"{aggregate_field!r} is not numeric")
    elif aggregate_field is not None or group_by is not None:
        return _reject(question, payload, "group_by/aggregate_field set without aggregate")

    if not select_fields and aggregate is None:
        return _reject(question, payload, "dynamic query has neither select_fields nor an aggregate")

    return ValidatedQuery(
        kind="dynamic",
        table=table,
        select_fields=select_fields or None,
        filters=filters or None,
        group_by=group_by,
        aggregate=aggregate,
        aggregate_field=aggregate_field,
        limit=_clamp_limit(payload.get("limit")),
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def validate_payload(payload: Any, question: str = "") -> Optional[ValidatedQuery]:
    """Return a ValidatedQuery if the payload is completely safe, else None."""
    if not isinstance(payload, dict):
        return _reject(question, payload, "payload is not an object")

    unknown = set(payload) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        return _reject(question, payload, f"unknown payload keys: {sorted(unknown)}")

    kind = str(payload.get("kind") or "").lower()
    if kind == "procedure":
        return _validate_procedure(payload, question)
    if kind == "dynamic":
        return _validate_dynamic(payload, question)
    return _reject(question, payload, f"unhandled kind {kind!r}")
