"""
Cross-domain "today's operations" summary for managers — the one answer that
needs orders, trips AND invoices together ("summary of today's cargo
operations: pending orders, active trips, delayed deliveries, payment
exceptions"). Each query planner handles a single collection, so a planner
returns task "ops_summary" and rag_engine calls ``build_ops_summary()``.

Every count reuses the planners' own match builders (same status / date
semantics as the per-domain answers). A section that fails (e.g. a tenant
with no invoices yet) is reported as unavailable instead of failing the reply.
There is no payment / receipt data — overdue open invoices are the closest
signal to "payment exceptions".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from app.Ai_Assistant.config import checkpoint
from app.Ai_Assistant.Orders.order_dynamic_analytics import (
    AGG_TIMEOUT_MS,
    _numeric_expr,
    get_orders_schema,
)
from app.tenants.router import get_domain_collection

_ORDER_OPEN = {"field": "orderstatus", "op": "nin", "value": ["Delivered", "Cancelled", "Rejected"]}
_TRIP_OPEN = {"field": "tripstatus", "op": "nin", "value": ["Delivered", "Rejected", "Cancelled"]}
_TRIP_ACTIVE = {
    "field": "tripstatus", "op": "in",
    "value": ["Dispatched", "Started", "In-Transit", "In Transit"],
}
_INVOICE_UNPAID = {"field": "InvoiceStatus", "op": "nin", "value": ["Paid"]}


def _order_section(today: str) -> Dict[str, Any]:
    from app.Ai_Assistant.Orders.order_query_planner import _build_match

    schema = get_orders_schema()
    coll = get_domain_collection("orders")

    def count(*filters: Dict[str, Any]) -> int:
        return coll.count_documents(_build_match(list(filters), schema))

    by_status = {
        str(r["_id"]): r["n"]
        for r in coll.aggregate(
            [
                {"$match": _build_match([_ORDER_OPEN], schema)},
                {"$group": {"_id": "$orderstatus", "n": {"$sum": 1}}},
            ],
            maxTimeMS=AGG_TIMEOUT_MS,
        )
    }
    return {
        "open_orders": count(_ORDER_OPEN),
        "open_orders_by_status": by_status,
        "orders_past_delivery_date_not_delivered": count(
            _ORDER_OPEN, {"field": "deliverydate", "op": "date_lt", "value": today}
        ),
        "orders_due_for_delivery_today": count(
            _ORDER_OPEN, {"field": "deliverydate", "op": "date_eq", "value": today}
        ),
        "open_orders_without_driver": count(
            _ORDER_OPEN, {"field": "drivercount", "op": "eq", "value": 0}
        ),
        "orders_created_today": count(
            {"field": "createdon", "op": "period", "value": "today"}
        ),
    }


def _trip_section(today: str) -> Dict[str, Any]:
    from app.Ai_Assistant.Trips.trip_dynamic_analytics import get_trips_schema
    from app.Ai_Assistant.Trips.trip_query_planner import _build_trip_match

    schema = get_trips_schema()
    coll = get_domain_collection("trips")

    def count(*filters: Dict[str, Any]) -> int:
        return coll.count_documents(_build_trip_match(list(filters), schema))

    return {
        "active_trips_dispatched_or_moving": count(_TRIP_ACTIVE),
        "planned_trips_not_dispatched": count(
            {"field": "tripstatus", "op": "eq", "value": "Planned"}
        ),
        "trips_past_delivery_date_not_delivered": count(
            _TRIP_OPEN, {"field": "lastdeliverydate", "op": "date_lt", "value": today}
        ),
        "open_trips_without_driver": count(
            _TRIP_OPEN, {"field": "firstdrivername", "op": "exists", "value": False}
        ),
    }


def _invoice_section(today: str) -> Dict[str, Any]:
    from app.Ai_Assistant.Invoices.invoice_dynamic_analytics import get_invoices_schema
    from app.Ai_Assistant.Invoices.invoice_query_planner import _build_invoice_match

    schema = get_invoices_schema()
    coll = get_domain_collection("invoices")
    overdue = [_INVOICE_UNPAID, {"field": "DueDate", "op": "date_lt", "value": today}]

    def outstanding(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(
            coll.aggregate(
                [
                    {"$match": _build_invoice_match(filters, schema)},
                    {"$group": {
                        "_id": "$CurrencyCode",
                        "invoices": {"$sum": 1},
                        "outstanding": {"$sum": _numeric_expr("outstandinamount")},
                    }},
                ],
                maxTimeMS=AGG_TIMEOUT_MS,
            )
        )
        return {
            str(r["_id"]): {"invoices": r["invoices"], "outstanding": round(r["outstanding"] or 0, 2)}
            for r in rows
        }

    return {
        "unpaid_invoices_by_currency": outstanding([_INVOICE_UNPAID]),
        "overdue_unpaid_invoices_by_currency": outstanding(overdue),
    }


_SECTIONS: List[tuple[str, Callable[[str], Dict[str, Any]]]] = [
    ("orders", _order_section),
    ("trips", _trip_section),
    ("invoices", _invoice_section),
]


def build_ops_summary() -> Dict[str, Any]:
    """execute_tools()-shaped result with one OPERATIONS SUMMARY context block."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary: Dict[str, Any] = {"today": today}
    for name, fn in _SECTIONS:
        try:
            summary[name] = fn(today)
        except Exception as exc:  # noqa: BLE001 — one domain must not sink the summary
            checkpoint("OPS_SUMMARY", f"{name} section failed", error=str(exc))
            summary[name] = "not available"

    lines = [
        f"OPERATIONS SUMMARY (today={today}; exact counts — do not invent others):",
        "Delayed = past its scheduled delivery date and not marked Delivered.",
        "No payment/receipt records exist — overdue unpaid invoices are the payment exceptions.",
    ]
    for name, _fn in _SECTIONS:
        section = summary[name]
        if isinstance(section, dict):
            lines.append(f"{name}:")
            lines.extend(f"  {k}={v}" for k, v in section.items())
        else:
            lines.append(f"{name}: {section}")
    checkpoint("OPS_SUMMARY", "built", **{k: v for k, v in summary.items() if k != "today"})
    return {
        "context_blocks": ["\n".join(lines)],
        "matches": [],
        "calculation": None,
        "analytics": {"analytics_type": "ops_summary", **summary},
        "list_result": None,
        "tools_run": ["ops_summary"],
        "active_order_token": "",
    }
