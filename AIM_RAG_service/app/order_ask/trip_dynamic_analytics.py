"""
Dynamic analytics for Avaal trips — the aggregation half of the trip LLM
query planner (``app/order_ask/trip_query_planner.py``).

Mirrors ``app/order_ask/invoice_dynamic_analytics.py`` but targets the
``Avaal_trip`` collection. The generic, domain-agnostic primitives (schema
walk, spec validation, pipeline builder, result rounding) are reused from
``dynamic_analytics``; only the trip-specific glue lives here:

- ``get_trips_schema``        sampled field schema (cached, TTL)
- ``_TRIP_FIELD_ALIASES``     business term -> real Avaal_trip field
- ``validate_trip_spec``      -> ``dynamic_analytics._validate_spec`` with the
                                trip ISO-date fields + aliases
- ``_shape_trip_result`` / ``format_trip_dynamic_analytics_for_context``
                              copies of the order shaper/formatter with the
                              noun "orders" -> "trips" and the
                              percentage/compare branches the planner builds.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.order_ask.checkpoint import checkpoint
from app.order_ask.dynamic_analytics import (
    AGG_TIMEOUT_MS,
    GROUPABLE_MAX_DISTINCT,
    SCHEMA_SAMPLE_SIZE,
    SCHEMA_TTL_SECONDS,
    _norm,
    _round,
    _schema_for_prompt,
    _validate_spec,
    _walk,
)
from app.order_ask.trip_analytics import _base_match as _trip_base_match
from app.tenants.context import require_tenant
from app.tenants.router import get_domain_collection

# Every Avaal_trip date field is an ISO-8601 string (lexically sortable, safe
# for $substrBytes date buckets and string-prefix range compares). A few carry
# a timezone offset ("2026-08-14T06:47:06.905237+05:30"); the first 10 chars
# are still a valid YYYY-MM-DD.
_TRIP_ISO_DATE_FIELDS = frozenset(
    {
        "createdon",
        "modifiedon",
        "firstpickupdate",
        "firstpickupdatetime",
        "lastdeliverydate",
        "lastdeliverydatetime",
    }
)

_TRIP_ID_LIKE = frozenset(
    {
        "tripid",
        "tripnumber",
        "tripno",
        "orderids",
        "ordernumber",
        "ordernumbers",
        "customerorderrefno",
        "truckcode",
        "trucknumber",
        "platenumber",
        "trlfplatenumber",
        "trlsplatenumber",
        "companycode",
        "corporateid",
        "customercodes",
        "salesmancodes",
        "firstdrivercode",
        "seconddrivercode",
        "firsttrailercode",
        "secondtrailercode",
        "firstpickupcode",
        "lastdeliverycode",
        "sealnumber",
        "containernumber",
        "ordercontainernumber",
        "vinnos",
        "bolpodnumber",
        "ebtripnumber",
        "eetripnumber",
        "pickuppostalcode",
        "deliverypostalcode",
        "pickupreferencenos",
        "deliveryreferencenos",
    }
)

_SKIP_FIELDS = frozenset(
    {"_id", "embedding", "page_content", "metadata", "namespace", "DomainEvents"}
)

# Business phrasing -> real Avaal_trip field (keys are _norm()'d). Only used
# when a planner-named field is not found exactly / fuzzily.
_TRIP_FIELD_ALIASES: Dict[str, str] = {
    # distance
    "distance": "triptotaldistance", "totaldistance": "triptotaldistance",
    "triptotaldistance": "triptotaldistance", "miles": "triptotaldistance",
    "mileage": "triptotaldistance", "km": "triptotaldistance",
    "kilometers": "triptotaldistance", "kilometres": "triptotaldistance",
    "loaddistance": "totalloaddistance", "loadeddistance": "totalloaddistance",
    "totalloaddistance": "totalloaddistance", "totalloadeddistance": "totalloaddistance",
    "emptydistance": "totalemptydistance", "deadhead": "totalemptydistance",
    "totalemptydistance": "totalemptydistance",
    "loadedmiles": "totalloaddistance", "emptymiles": "totalemptydistance",
    "ebdistance": "ebdistance", "eedistance": "eedistance",
    "odometer": "endodometer", "beginodometer": "beginodometer",
    "endodometer": "endodometer",
    # money
    "amount": "totalofferedamount", "offeredamount": "totalofferedamount",
    "totalofferedamount": "totalofferedamount", "offered": "totalofferedamount",
    "revenue": "totalofferedamount", "price": "totalofferedamount",
    "value": "totalofferedamount", "pay": "totalofferedamount",
    "payout": "totalofferedamount", "linehaul": "totalofferedamount",
    "rate": "rate", "ratevalue": "ratetypevalue", "ratetypevalue": "ratetypevalue",
    "tax": "totaltaxamount", "taxamount": "totaltaxamount",
    "totaltaxamount": "totaltaxamount",
    "addition": "totaladdition", "additions": "totaladdition",
    "deduction": "totaldeduction", "deductions": "totaldeduction",
    "accessorial": "pendingaccessorial", "pendingaccessorial": "pendingaccessorial",
    # cargo
    "weight": "totalweight", "totalweight": "totalweight",
    "quantity": "totalquantity", "qty": "totalquantity",
    "totalquantity": "totalquantity",
    "items": "itemscount", "itemcount": "itemscount", "itemscount": "itemscount",
    "tripitems": "tripitemscount", "tripitemscount": "tripitemscount",
    "commodity": "commodity", "goods": "commodity", "product": "commodity",
    "cargo": "commodity", "material": "commodity",
    "temperature": "reefertemp", "temp": "reefertemp", "reefertemp": "reefertemp",
    "reefertemperature": "reefertemp",
    # identity / status / type
    "status": "tripstatus", "tripstatus": "tripstatus", "state": "tripstatus",
    "type": "triptype", "triptype": "triptype", "triptypemain": "triptypemain",
    "variant": "tripvariant", "tripvariant": "tripvariant",
    "tripnumber": "tripnumber", "tripno": "tripnumber", "trip": "tripnumber",
    "tripid": "tripid",
    # people / equipment
    "driver": "firstdrivername", "drivername": "firstdrivername",
    "firstdriver": "firstdrivername", "firstdrivername": "firstdrivername",
    "seconddriver": "seconddrivername", "seconddrivername": "seconddrivername",
    "codriver": "seconddrivername",
    "drivercode": "firstdrivercode", "firstdrivercode": "firstdrivercode",
    "truck": "trucknumber", "trucknumber": "trucknumber", "truckcode": "truckcode",
    "unit": "trucknumber", "tractor": "trucknumber",
    "trailer": "firsttrailernumber", "trailernumber": "firsttrailernumber",
    "firsttrailernumber": "firsttrailernumber",
    "secondtrailernumber": "secondtrailernumber",
    "plate": "platenumber", "platenumber": "platenumber",
    "carrier": "carriername", "carriername": "carriername",
    "carriercode": "carriercode",
    "salesman": "salesmannames", "salesrep": "salesmannames",
    "agent": "salesmannames", "salesmannames": "salesmannames",
    "salesmancodes": "salesmancodes",
    # customer / company
    "customer": "customername", "client": "customername", "buyer": "customername",
    "customername": "customername", "account": "customername",
    "customercode": "customercodes", "customercodes": "customercodes",
    "company": "companyname", "companyname": "companyname", "branch": "companyname",
    "companycode": "companycode",
    # linked docs
    "order": "ordernumber", "orders": "ordernumber", "ordernumber": "ordernumber",
    "ordernumbers": "ordernumber", "orderid": "orderids", "orderids": "orderids",
    "po": "customerorderrefno", "customerref": "customerorderrefno",
    "customerorderrefno": "customerorderrefno", "reference": "customerorderrefno",
    # geo (real fields — direct)
    "pickupcity": "pickupcity", "origincity": "pickupcity",
    "pickupstate": "pickupstate", "pickupprovince": "pickupstate",
    "pickupcountry": "pickupcountry",
    "deliverycity": "deliverycity", "dropcity": "deliverycity",
    "destinationcity": "deliverycity",
    "deliverystate": "deliverystate", "deliveryprovince": "deliverystate",
    "deliverycountry": "deliverycountry",
    "pickup": "pickuplocationname", "pickuplocation": "pickuplocationname",
    "pickuplocationname": "pickuplocationname", "shipper": "pickuplocationname",
    "origin": "pickuplocationname",
    "delivery": "deliverylocationname", "deliverylocation": "deliverylocationname",
    "deliverylocationname": "deliverylocationname", "consignee": "deliverylocationname",
    "destination": "deliverylocationname", "drop": "deliverylocationname",
    # dates
    "created": "createdon", "createddate": "createdon", "createdon": "createdon",
    "createddatetime": "createdon",
    "modified": "modifiedon", "modifieddate": "modifiedon", "updatedon": "modifiedon",
    "lastmodified": "modifiedon",
    "pickupdate": "firstpickupdate", "pickedup": "firstpickupdate",
    "firstpickupdate": "firstpickupdate", "firstpickupdatetime": "firstpickupdatetime",
    "deliverydate": "lastdeliverydate", "delivered": "lastdeliverydate",
    "lastdeliverydate": "lastdeliverydate", "lastdeliverydatetime": "lastdeliverydatetime",
    "date": "createdon",
    # counts
    "documents": "documentcount", "documentcount": "documentcount",
    "settlements": "settlementcount", "settlementcount": "settlementcount",
    "notifications": "tripnotificationcount",
}


# ------------------------------------------------------------------- schema
_schema_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}


def _sample_trip_schema(sample_size: int) -> Dict[str, Any]:
    collection = get_domain_collection("trips")
    base = _trip_base_match()
    size = max(50, int(sample_size or 500))
    try:
        docs = list(
            collection.aggregate(
                [{"$match": base}, {"$sample": {"size": size}}],
                maxTimeMS=AGG_TIMEOUT_MS,
            )
        )
    except Exception as exc:  # $sample unsupported / timeout — plain scan
        checkpoint(
            "TRIP_DYN_ANALYTICS",
            "schema $sample failed, using find()",
            error=str(exc),
        )
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
        if key in _SKIP_FIELDS:
            continue
        n = info["n"] or 1
        numeric_hits = info["num"] + info["numstr"]
        numeric = numeric_hits > 0 and numeric_hits >= 0.8 * n
        string_numeric = numeric and info["numstr"] >= info["num"]
        distinct = len(info["values"])
        seg = key.split(".")[-1]
        id_like = key in _TRIP_ID_LIKE or seg in _TRIP_ID_LIKE or (
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

    index: Dict[str, str] = {}
    for real in schema["fields"]:
        index.setdefault(_norm(real), real)
        index.setdefault(_norm(real.split(".")[-1]), real)
    schema["_index"] = index
    return schema


def get_trips_schema(*, force: bool = False) -> Dict[str, Any]:
    """Sampled Avaal_trip field schema for the active tenant (cached, TTL)."""
    tenant = require_tenant()
    key = (tenant.database, tenant.collection_for("trips"))
    now = time.time()
    cached = _schema_cache.get(key)
    if (
        cached
        and not force
        and "_index" in cached[1]
        and now - cached[0] < SCHEMA_TTL_SECONDS
    ):
        return cached[1]
    schema = _sample_trip_schema(SCHEMA_SAMPLE_SIZE)
    _schema_cache[key] = (now, schema)
    checkpoint(
        "TRIP_DYN_ANALYTICS",
        "schema built",
        fields=len(schema["fields"]),
        sample=schema["sample_size"],
    )
    return schema


def trip_schema_for_prompt(schema: Dict[str, Any], *, max_fields: int = 190) -> str:
    return _schema_for_prompt(
        schema, max_fields=max_fields, collection_label="Avaal_trip"
    )


def validate_trip_spec(spec: Any, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _validate_spec(
        spec,
        schema,
        iso_date_fields=_TRIP_ISO_DATE_FIELDS,
        aliases=_TRIP_FIELD_ALIASES,
    )


# ----------------------------------------------------------------- shape
def _shape_trip_result(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    filters: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    op = spec["operation"]
    payload: Dict[str, Any] = {
        "analytics_type": "dynamic",
        "engine": "trip_dynamic_planner",
        "operation": op,
        "filters": filters or {},
        "question": question,
    }

    if op == "count":
        payload["matching_trips"] = int(rows[0]["count"]) if rows else 0
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
        payload["matching_trips"] = int(row.get("_matched") or 0)
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
        row["trips"] = int(r.get("_matched") or 0)
        out_rows.append(row)
    payload["rows"] = out_rows
    return payload


def format_trip_dynamic_analytics_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text ground-truth block for the answer LLM."""
    lines = [
        "TRIP ANALYTICS RESULT (dynamic aggregation engine — exact Mongo "
        "output, do not recalculate or invent):",
        f"operation: {payload.get('operation')}",
    ]
    if payload.get("filters"):
        lines.append(f"filters: {payload['filters']}")

    op = payload.get("operation")
    if op == "count":
        lines.append(f"matching_trips: {payload.get('matching_trips')}")

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
        lines.append(f"matching_trips: {payload.get('matching_trips')}")
        for key, value in (payload.get("values") or {}).items():
            lines.append(f"- {key}: {value}")

    elif op == "group":
        if payload.get("having"):
            lines.append(f"having (post-group filter): {payload['having']}")
        lines.append(
            f"grouped by {payload.get('group_by')}, "
            f"metrics {payload.get('metrics')} (trips = row count):"
        )
        for row in payload.get("rows") or []:
            lines.append("- " + " | ".join(f"{k}={v}" for k, v in row.items()))
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
            lines.append("- " + " | ".join(f"{k}={v}" for k, v in row.items()))
        if not payload.get("rows"):
            lines.append("(no segments matched)")

    lines.append("Use these numbers as ground truth. Write numbers without commas.")
    return "\n".join(lines)
