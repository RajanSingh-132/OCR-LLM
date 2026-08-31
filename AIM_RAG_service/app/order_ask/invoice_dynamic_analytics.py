"""
Dynamic analytics for Avaal invoices — the aggregation half of the invoice
LLM query planner (``app/order_ask/invoice_query_planner.py``).

Mirrors ``app/order_ask/dynamic_analytics.py`` (orders) but targets the
``Avaal_invoice`` collection. The generic, domain-agnostic primitives
(schema walk, spec validation, pipeline builder, result rounding) are reused
from ``dynamic_analytics``; only the invoice-specific glue lives here:

- ``get_invoices_schema``      sampled field schema (cached, TTL)
- ``_INVOICE_FIELD_ALIASES``   business term -> real Avaal_invoice field
- ``validate_invoice_spec``    -> ``dynamic_analytics._validate_spec`` with
                                  invoice ISO-date fields + aliases
- ``_shape_invoice_result`` / ``format_invoice_dynamic_analytics_for_context``
                                copies of the order shaper/formatter with the
                                noun "orders" -> "invoices" and the
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
    resolve_field,
)
from app.order_ask.invoice_analytics import _base_match as _invoice_base_match
from app.tenants.context import require_tenant
from app.tenants.router import get_domain_collection

# createdon / modifiedon are ISO-8601 strings (lexically sortable, safe for
# $substrBytes date buckets). InvoiceDate / DueDate are US-format
# ("2/16/2026 2:30:00 AM") and are handled separately in the planner.
_INVOICE_ISO_DATE_FIELDS = frozenset({"createdon", "modifiedon"})

_INVOICE_ID_LIKE = frozenset(
    {
        "InvoiceID",
        "invoiceid",
        "InvoiceNumber",
        "invoicenumber",
        "QBId",
        "InvoiceOrderIds",
        "InvoiceOrderNumbers",
        "CustomerCode",
        "companycode",
        "corporateid",
        "VerificationToken",
    }
)

_SKIP_FIELDS = frozenset(
    {"_id", "embedding", "page_content", "metadata", "namespace", "DomainEvents"}
)

# Business phrasing -> real Avaal_invoice field (keys are _norm()'d).
_INVOICE_FIELD_ALIASES: Dict[str, str] = {
    # money
    "amount": "TotalAmount", "total": "TotalAmount", "totalamount": "TotalAmount",
    "invoiceamount": "TotalAmount", "value": "TotalAmount", "price": "TotalAmount",
    "aftertax": "TotalAmount", "grandtotal": "TotalAmount", "billed": "TotalAmount",
    "billedamount": "TotalAmount", "amountaftertax": "TotalAmount",
    "pretax": "PreTaxAmount", "beforetax": "PreTaxAmount",
    "pretaxamount": "PreTaxAmount", "amountbeforetax": "PreTaxAmount",
    "subtotal": "PreTaxAmount",
    "freight": "freightcharges", "freightcharge": "freightcharges",
    "freightamount": "freightcharges",
    "fuel": "fuelsurcharges", "fuelsurcharge": "fuelsurcharges",
    "fuelcharges": "fuelsurcharges", "fuelcharge": "fuelsurcharges",
    "other": "othercharges", "othercharge": "othercharges",
    "otherchargesamount": "othercharges",
    "discount": "DiscountAmount", "discountamount": "DiscountAmount",
    "outstanding": "outstandinamount", "outstandingamount": "outstandinamount",
    "outstanding amount": "outstandinamount", "balance": "outstandinamount",
    "due": "outstandinamount", "dueamount": "outstandinamount",
    "unpaidamount": "outstandinamount", "remaining": "outstandinamount",
    "tax": "PreTaxAmount",  # no explicit tax total; pretax vs total is the delta
    "exchangerate": "ExchangeRate", "rate": "ExchangeRate", "fxrate": "ExchangeRate",
    "maxinvoiceamount": "MaxInvoiceAmount",
    # identity / status
    "status": "InvoiceStatus", "invoicestatus": "InvoiceStatus",
    "state": "InvoiceStatus", "paymentstatus": "InvoiceStatus",
    "customer": "CustomerName", "customername": "CustomerName",
    "client": "CustomerName", "buyer": "CustomerName", "account": "CustomerName",
    "customercode": "CustomerCode",
    "company": "CompanyName", "companyname": "CompanyName", "branch": "CompanyName",
    "currency": "CurrencyCode", "currencycode": "CurrencyCode",
    "invoicenumber": "InvoiceNumber", "invoiceno": "InvoiceNumber",
    "invoiceno.": "InvoiceNumber", "invoice": "InvoiceNumber",
    "invoiceid": "InvoiceID",
    "salesman": "salesmanname", "salesrep": "salesmanname", "agent": "salesmanname",
    "commodity": "commodityname", "product": "commodityname", "goods": "commodityname",
    "paymentterm": "PaymentTermName", "term": "PaymentTermName",
    "factoringcompany": "FactorycompanyName",
    # linked docs
    "order": "InvoiceOrderNumbers", "orders": "InvoiceOrderNumbers",
    "ordernumber": "InvoiceOrderNumbers", "ordernumbers": "InvoiceOrderNumbers",
    "customerordernumber": "CustomerOrderNumbers", "po": "CustomerOrderNumbers",
    "ponumber": "CustomerOrderNumbers", "reference": "CustomerOrderNumbers",
    "trip": "TripNumbers", "tripnumber": "TripNumbers", "tripno": "TripNumbers",
    "truck": "TruckNumber", "trucknumber": "TruckNumber",
    "trailer": "TrailerNumber", "driver": "DriverName", "drivername": "DriverName",
    "carrier": "CarrierName", "carriername": "CarrierName",
    # geo
    "destination": "destinationname", "destinationname": "destinationname",
    "drop": "deliverylocation", "consignee": "deliverylocationcode",
    "pickup": "pickuplocation", "pickuplocation": "pickuplocation",
    "pickuplocationcode": "pickuplocationcode",
    "delivery": "deliverylocation", "deliverylocation": "deliverylocation",
    "deliverylocationcode": "deliverylocationcode",
    "origin": "pickuplocation", "shipper": "pickuplocationcode",
    # dates
    "created": "createdon", "createddate": "createdon", "createdon": "createdon",
    "createddatetime": "createdon",
    "modified": "modifiedon", "modifieddate": "modifiedon", "updatedon": "modifiedon",
    "invoicedate": "InvoiceDate", "billdate": "InvoiceDate", "date": "InvoiceDate",
    "duedate": "DueDate", "due date": "DueDate",
    "pickupdate": "PickupDate", "deliverydate": "DeliveryDate",
}


# ------------------------------------------------------------------- schema
_schema_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}


def _sample_invoice_schema(sample_size: int) -> Dict[str, Any]:
    collection = get_domain_collection("invoices")
    base = _invoice_base_match()
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
            "INV_DYN_ANALYTICS",
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
        id_like = key in _INVOICE_ID_LIKE or seg in _INVOICE_ID_LIKE or (
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


def get_invoices_schema(*, force: bool = False) -> Dict[str, Any]:
    """Sampled Avaal_invoice field schema for the active tenant (cached, TTL)."""
    tenant = require_tenant()
    key = (tenant.database, tenant.collection_for("invoices"))
    now = time.time()
    cached = _schema_cache.get(key)
    if (
        cached
        and not force
        and "_index" in cached[1]
        and now - cached[0] < SCHEMA_TTL_SECONDS
    ):
        return cached[1]
    schema = _sample_invoice_schema(SCHEMA_SAMPLE_SIZE)
    _schema_cache[key] = (now, schema)
    checkpoint(
        "INV_DYN_ANALYTICS",
        "schema built",
        fields=len(schema["fields"]),
        sample=schema["sample_size"],
    )
    return schema


def invoice_schema_for_prompt(schema: Dict[str, Any], *, max_fields: int = 180) -> str:
    return _schema_for_prompt(
        schema, max_fields=max_fields, collection_label="Avaal_invoice"
    )


def validate_invoice_spec(
    spec: Any, schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    return _validate_spec(
        spec,
        schema,
        iso_date_fields=_INVOICE_ISO_DATE_FIELDS,
        aliases=_INVOICE_FIELD_ALIASES,
    )


# ----------------------------------------------------------------- shape
def _shape_invoice_result(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    filters: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    op = spec["operation"]
    payload: Dict[str, Any] = {
        "analytics_type": "dynamic",
        "engine": "invoice_dynamic_planner",
        "operation": op,
        "filters": filters or {},
        "question": question,
    }

    if op == "count":
        payload["matching_invoices"] = int(rows[0]["count"]) if rows else 0
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
        payload["matching_invoices"] = int(row.get("_matched") or 0)
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
        row["invoices"] = int(r.get("_matched") or 0)
        out_rows.append(row)
    payload["rows"] = out_rows
    return payload


def format_invoice_dynamic_analytics_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text ground-truth block for the answer LLM."""
    lines = [
        "INVOICE ANALYTICS RESULT (dynamic aggregation engine — exact Mongo "
        "output, do not recalculate or invent):",
        f"operation: {payload.get('operation')}",
    ]
    if payload.get("filters"):
        lines.append(f"filters: {payload['filters']}")

    op = payload.get("operation")
    if op == "count":
        lines.append(f"matching_invoices: {payload.get('matching_invoices')}")

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
        lines.append(f"matching_invoices: {payload.get('matching_invoices')}")
        for key, value in (payload.get("values") or {}).items():
            lines.append(f"- {key}: {value}")

    elif op == "group":
        if payload.get("having"):
            lines.append(f"having (post-group filter): {payload['having']}")
        lines.append(
            f"grouped by {payload.get('group_by')}, "
            f"metrics {payload.get('metrics')} (invoices = row count):"
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
