"""
Fleet-wide analytics over ALL Avaal_db order records.

Supports:
- order status summary / counts (Quoted, Cancelled, Confirmed, Dispatched, Delivered, Invoiced, …)
- best / worst / low customer by order count (default) or revenue
- customer counts by country using pickup / delivery (drop) addresses
- activity on a date (customers + orders)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.mongo_client import get_mongo_collection
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import AVAAL_COLLECTION_NAME, AVAAL_NAMESPACE

KNOWN_STATUSES = [
    "Quoted",
    "Cancelled",
    "Confirmed",
    "Dispatched",
    "Delivered",
    "Invoiced",
]

COUNTRY_PATTERNS = {
    "Canada": [r"\bCanada\b", r"\bCAN\b"],
    "United States": [
        r"\bUnited States\b",
        r"\bUSA\b",
        r"\bU\.S\.A\.?\b",
        r"\bU\.S\.?\b",
    ],
}


def _base_match() -> Dict[str, Any]:
    return {
        "namespace": AVAAL_NAMESPACE,
        "metadata.type": "avaal_order",
    }


def _country_regex(country: str) -> str:
    pats = COUNTRY_PATTERNS.get(country) or [rf"\b{re.escape(country)}\b"]
    return "(?:" + "|".join(pats) + ")"


def detect_country(question: str) -> Optional[str]:
    q = (question or "").lower()
    if re.search(r"\b(canada|canadian)\b", q):
        return "Canada"
    if re.search(
        r"\b(united states|u\.s\.a\.?|usa|america|american)\b|\bin the us\b|\bin us\b|\bus\b",
        q,
    ):
        return "United States"
    return None


def detect_location_side(question: str) -> str:
    """pickup | delivery | both — default both for country/customer geo questions."""
    q = (question or "").lower()
    wants_pickup = bool(re.search(r"\b(pickup|pick\s*up|origin|shipper)\b", q))
    wants_delivery = bool(
        re.search(r"\b(delivery|deliver|drop|consignee|destination)\b", q)
    )
    if wants_pickup and not wants_delivery:
        return "pickup"
    if wants_delivery and not wants_pickup:
        return "delivery"
    return "both"


def is_status_summary_question(question: str) -> bool:
    q = (question or "").lower()
    # "list/show confirmed orders" is a filtered LIST, not a status summary/count
    if re.search(r"\b(list|show|display|find|search|filter|get)\b", q) and not re.search(
        r"\b(how many|count|summary|break\s*down|breakdown|distribution|total)\b", q
    ):
        return False
    if re.search(r"\bstatus\b.*\b(summary|break\s*down|breakdown|count|how many|distribution)\b", q):
        return True
    if re.search(r"\b(summary|break\s*down|breakdown)\b.*\bstatus", q):
        return True
    if re.search(r"\bhow many\b.*\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b", q):
        return True
    if re.search(
        r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b.*\b(count|how many)\b",
        q,
    ):
        return True
    # "confirmed orders count" style — not plain "confirmed orders" list phrasing
    if re.search(
        r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b.*\borders?\b",
        q,
    ) and re.search(r"\b(how many|count|summary|total|breakdown)\b", q):
        return True
    if re.search(r"\border\s*status(es)?\b", q) and re.search(
        r"\b(summary|how many|count|break|breakdown|distribution)\b", q
    ):
        return True
    return False


_BEST_WORDS = r"best|top|biggest|largest|highest|most|maximum|max|premium"
_WORST_WORDS = (
    r"worst|lowest|low|least|fewest|smallest|minimum|min|bottom|poorest|weakest"
)


def detect_customer_direction(question: str) -> str:
    """best (highest) vs worst (lowest) customer ranking direction."""
    q = (question or "").lower()
    if re.search(rf"\b({_WORST_WORDS})\b", q):
        return "worst"
    return "best"


def is_best_customer_question(question: str) -> bool:
    """
    Any customer-ranking question — best/top AND worst/lowest/low customer.
    e.g. best customer, top customer by revenue, worst customer, low customer,
    customer with least orders, smallest customer.
    """
    q = (question or "").lower()
    if not re.search(r"\bcustomers?\b|\bclients?\b", q):
        return False
    if re.search(
        rf"\b({_BEST_WORDS}|{_WORST_WORDS})\b.*\b(customer|client)\b", q
    ):
        return True
    if re.search(
        rf"\b(customer|client)\b.*\b({_BEST_WORDS}|{_WORST_WORDS}|orders?|revenue|freight|amount|sales)\b",
        q,
    ) and re.search(rf"\b({_BEST_WORDS}|{_WORST_WORDS})\b", q):
        return True
    return False


def is_country_customer_question(question: str) -> bool:
    q = (question or "").lower()
    if not detect_country(q):
        return False
    return bool(
        re.search(
            r"\b(how many|count|number of|customers?|client|in canada|in us|in usa|in the us|in united)\b",
            q,
        )
    )


def normalize_date_prefix(raw: str) -> Optional[str]:
    """
    Normalize user date text to YYYY-MM-DD prefix for matching ISO orderdate fields.
    Supports: 2026-08-06, 2026/08/06, 07/13/2026, 7-13-2026.
    """
    if not raw:
        return None
    text = raw.strip()

    m = re.match(r"^(20\d{2})[-/](\d{1,2})[-/](\d{1,2})$", text)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{y}-{mo:02d}-{d:02d}"

    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](20\d{2})$", text)
    if m:
        # Assume MM/DD/YYYY (common in Avaal Order Sheets)
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if mo > 12 and d <= 12:
            mo, d = d, mo  # swap if clearly DD/MM
        return f"{y}-{mo:02d}-{d:02d}"

    return None


def extract_date_from_question(question: str) -> Optional[str]:
    q = question or ""
    m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", q)
    if m:
        return normalize_date_prefix(m.group(1))
    m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b", q)
    if m:
        return normalize_date_prefix(m.group(1))
    return None


def detect_date_field(question: str) -> str:
    """Which date field to filter: orderdate | pickupdate | deliverydate."""
    q = (question or "").lower()
    if re.search(r"\b(pickup|pick\s*up)\b", q):
        return "pickupdate"
    if re.search(r"\b(delivery|deliver|drop)\b", q):
        return "deliverydate"
    return "orderdate"


def is_date_activity_question(question: str) -> bool:
    """
    e.g. how many customers ordered on 2026-08-06 /
    is date ko kitne customers ne order diya /
    orders on 07/13/2026
    """
    q = (question or "").lower()
    has_date = bool(extract_date_from_question(question))
    if not has_date:
        return False
    return bool(
        re.search(
            r"\b(how many|count|kitne|kitna|number of|customers?|orders?|ordered|order diya|ne order)\b",
            q,
        )
    ) or bool(re.search(r"\b(on|for|dated|date)\b", q))


def is_analytics_question(question: str) -> bool:
    return (
        is_status_summary_question(question)
        or is_best_customer_question(question)
        or is_country_customer_question(question)
        or is_date_activity_question(question)
    )


def status_summary() -> Dict[str, Any]:
    """Count orders per orderstatus across ALL records."""
    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": {"_id": "$orderstatus", "order_count": {"$sum": 1}}},
                {"$sort": {"order_count": -1}},
            ]
        )
    )
    by_status = {
        (r["_id"] if r["_id"] not in (None, "") else "Unknown"): int(r["order_count"])
        for r in rows
    }
    total = sum(by_status.values())
    # Include known statuses with 0 for completeness
    ordered = []
    for name in KNOWN_STATUSES:
        ordered.append({"status": name, "order_count": by_status.get(name, 0)})
    for name, count in by_status.items():
        if name not in KNOWN_STATUSES:
            ordered.append({"status": name, "order_count": count})

    checkpoint("ANALYTICS", "status_summary", total_orders=total, statuses=len(ordered))
    return {
        "analytics_type": "status_summary",
        "total_orders": total,
        "by_status": ordered,
        "definition": "Counts from all Avaal order records grouped by orderstatus.",
    }


def best_customers(
    *,
    metric: str = "orders",
    limit: int = 5,
    direction: str = "best",
) -> Dict[str, Any]:
    """
    Rank customers across ALL records.
    direction="best"  -> highest (most orders / most revenue)
    direction="worst" -> lowest  (fewest orders / least revenue)
    metric="revenue" uses SUM(grosstotalfreight), else order count.
    """
    limit = max(1, min(int(limit or 5), 20))
    direction = "worst" if str(direction).lower() == "worst" else "best"
    order = 1 if direction == "worst" else -1  # asc for worst, desc for best
    label = "worst" if direction == "worst" else "best"
    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)

    group = {
        "_id": "$customername",
        "order_count": {"$sum": 1},
        "total_revenue": {"$sum": "$grosstotalfreight"},
        "total_freight": {"$sum": "$totalfreight"},
    }

    if metric == "revenue":
        sort = {"total_revenue": order, "order_count": order}
        definition = (
            f"{label.capitalize()} customer by "
            f"{'lowest' if direction == 'worst' else 'highest'} total revenue "
            "(sum of grosstotalfreight) across all orders."
        )
    else:
        sort = {"order_count": order, "total_revenue": order}
        definition = (
            f"{label.capitalize()} customer = customer with the "
            f"{'minimum' if direction == 'worst' else 'maximum'} number of orders "
            "across all records. Ties broken by revenue."
        )

    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": group},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": sort},
                {"$limit": limit},
            ]
        )
    )
    customers = []
    for r in rows:
        customers.append(
            {
                "customername": r.get("_id"),
                "order_count": int(r.get("order_count") or 0),
                "total_revenue": round(float(r.get("total_revenue") or 0), 4),
                "total_freight": round(float(r.get("total_freight") or 0), 4),
            }
        )

    checkpoint(
        "ANALYTICS",
        "best_customers",
        direction=direction,
        metric=metric,
        top=customers[0]["customername"] if customers else None,
        limit=limit,
    )
    return {
        "analytics_type": "best_customer",
        "direction": direction,
        "metric": metric,
        "definition": definition,
        "customers": customers,
        "best_customer": customers[0] if customers else None,
    }


def customers_by_country(
    country: str,
    *,
    location_side: str = "both",
) -> Dict[str, Any]:
    """
    Count distinct customers whose pickup and/or delivery address mentions country.
    Uses pickupfulladdress + deliveryfulladdress (and location name fallbacks).
    """
    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    rx = _country_regex(country)
    address_or: List[Dict[str, Any]] = []
    if location_side in ("pickup", "both"):
        address_or.extend(
            [
                {"pickupfulladdress": {"$regex": rx, "$options": "i"}},
                {"pickuplocationname": {"$regex": rx, "$options": "i"}},
            ]
        )
    if location_side in ("delivery", "both"):
        address_or.extend(
            [
                {"deliveryfulladdress": {"$regex": rx, "$options": "i"}},
                {"deliverylocationname": {"$regex": rx, "$options": "i"}},
            ]
        )

    match = {**_base_match(), "$or": address_or}
    order_count = collection.count_documents(match)

    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": "$customername",
                        "order_count": {"$sum": 1},
                    }
                },
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": {"order_count": -1}},
                {"$limit": 25},
            ]
        )
    )
    customers = [
        {"customername": r["_id"], "order_count": int(r["order_count"])} for r in rows
    ]
    distinct_rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": "$customername"}},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$count": "n"},
            ]
        )
    )
    distinct_customers = int(distinct_rows[0]["n"]) if distinct_rows else 0

    side_label = {
        "pickup": "pickup address only",
        "delivery": "delivery/drop address only",
        "both": "pickup OR delivery/drop address",
    }.get(location_side, "pickup OR delivery/drop address")

    checkpoint(
        "ANALYTICS",
        "customers_by_country",
        country=country,
        side=location_side,
        customers=distinct_customers,
        orders=order_count,
    )
    return {
        "analytics_type": "customers_by_country",
        "country": country,
        "location_side": location_side,
        "location_rule": side_label,
        "definition": (
            f"Distinct customers with {country} found in {side_label} "
            f"(matched on pickupfulladdress/deliveryfulladdress and location names)."
        ),
        "distinct_customers": distinct_customers,
        "matching_orders": order_count,
        "sample_customers": customers[:10],
    }


def activity_on_date(
    date_prefix: str,
    *,
    date_field: str = "orderdate",
) -> Dict[str, Any]:
    """
    On a given calendar day: order count + distinct customers (from ALL matching records).
    Dates in DB look like 2026-08-06T05:20:03.043+05:30 — match by YYYY-MM-DD prefix.
    """
    if date_field not in ("orderdate", "pickupdate", "deliverydate"):
        date_field = "orderdate"
    date_prefix = normalize_date_prefix(date_prefix) or date_prefix

    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    match = {
        **_base_match(),
        date_field: {"$regex": f"^{re.escape(date_prefix)}", "$options": "i"},
    }
    order_count = collection.count_documents(match)

    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": "$customername",
                        "order_count": {"$sum": 1},
                    }
                },
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": {"order_count": -1}},
                {"$limit": 25},
            ]
        )
    )
    customers = [
        {"customername": r["_id"], "order_count": int(r["order_count"])} for r in rows
    ]
    distinct_rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": "$customername"}},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$count": "n"},
            ]
        )
    )
    distinct_customers = int(distinct_rows[0]["n"]) if distinct_rows else 0

    field_label = {
        "orderdate": "order date",
        "pickupdate": "pickup date",
        "deliverydate": "delivery/drop date",
    }.get(date_field, "order date")

    checkpoint(
        "ANALYTICS",
        "activity_on_date",
        date=date_prefix,
        field=date_field,
        customers=distinct_customers,
        orders=order_count,
    )
    return {
        "analytics_type": "activity_on_date",
        "date": date_prefix,
        "date_field": date_field,
        "date_field_label": field_label,
        "definition": (
            f"Orders and distinct customers on {date_prefix} using {field_label} "
            f"across all Avaal order records."
        ),
        "distinct_customers": distinct_customers,
        "matching_orders": order_count,
        "sample_customers": customers[:10],
    }


def run_analytics(
    question: str,
    *,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Route analytics question → Mongo aggregation payload."""
    entities = entities or {}
    q = question or ""

    # Date-based activity (customers/orders on a day) — dynamic from full DB
    date_val = (
        entities.get("analytics_date")
        or entities.get("orderdate")
        or entities.get("pickupdate")
        or entities.get("deliverydate")
        or extract_date_from_question(q)
    )
    if date_val and (
        is_date_activity_question(q)
        or entities.get("analytics") == "activity_on_date"
        or (
            date_val
            and re.search(r"\b(customer|order|kitne|kitna|how many|count)\b", q, re.I)
        )
    ):
        date_field = entities.get("date_field") or detect_date_field(q)
        if entities.get("pickupdate") and not entities.get("orderdate"):
            date_field = "pickupdate"
        if entities.get("deliverydate") and not entities.get("orderdate"):
            date_field = "deliverydate"
        return activity_on_date(str(date_val), date_field=date_field)

    if is_best_customer_question(q) or entities.get("analytics") in (
        "best_customer",
        "worst_customer",
    ):
        metric = entities.get("best_customer_metric") or "orders"
        if re.search(r"\b(revenue|freight|amount|sales|money|value)\b", q, re.I):
            metric = "revenue"
        direction = detect_customer_direction(q)
        if entities.get("analytics") == "worst_customer":
            direction = "worst"
        if entities.get("customer_direction") in ("best", "worst"):
            direction = entities["customer_direction"]
        limit = int(entities.get("limit") or 5)
        return best_customers(metric=metric, limit=limit, direction=direction)

    if is_country_customer_question(q) or (
        entities.get("country") and not date_val
    ):
        country = entities.get("country") or detect_country(q) or "Canada"
        side = entities.get("location_side") or detect_location_side(q)
        return customers_by_country(country, location_side=side)

    if is_status_summary_question(q) or entities.get("analytics") == "status_summary":
        return status_summary()

    # Fallback: if analytics tool was forced, prefer status summary
    return status_summary()


def format_analytics_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text block for LLM — ground truth, do not invent."""
    lines = ["ANALYTICS RESULT (exact engine — do not recalculate or invent):"]
    atype = payload.get("analytics_type")
    if payload.get("definition"):
        lines.append(f"Definition: {payload['definition']}")

    if atype == "status_summary":
        lines.append(f"total_orders: {payload.get('total_orders')}")
        lines.append("by_status:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('order_count')}")

    elif atype == "best_customer":
        direction = payload.get("direction") or "best"
        lines.append(f"direction: {direction} (best=highest, worst=lowest)")
        lines.append(f"metric: {payload.get('metric')}")
        best = payload.get("best_customer") or {}
        pick_label = "worst_customer" if direction == "worst" else "best_customer"
        if best:
            lines.append(
                f"{pick_label}: {best.get('customername')} | "
                f"order_count={best.get('order_count')} | "
                f"total_revenue={best.get('total_revenue')} | "
                f"total_freight={best.get('total_freight')}"
            )
        list_label = "bottom_customers" if direction == "worst" else "top_customers"
        lines.append(f"{list_label}:")
        for i, row in enumerate(payload.get("customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')} | "
                f"revenue={row.get('total_revenue')}"
            )

    elif atype == "customers_by_country":
        lines.append(f"country: {payload.get('country')}")
        lines.append(f"location_rule: {payload.get('location_rule')}")
        lines.append(f"distinct_customers: {payload.get('distinct_customers')}")
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        lines.append("sample_customers:")
        for i, row in enumerate(payload.get("sample_customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')}"
            )

    elif atype == "activity_on_date":
        lines.append(f"date: {payload.get('date')}")
        lines.append(f"date_field: {payload.get('date_field')} ({payload.get('date_field_label')})")
        lines.append(f"distinct_customers: {payload.get('distinct_customers')}")
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        lines.append("sample_customers:")
        for i, row in enumerate(payload.get("sample_customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')}"
            )
        if not payload.get("sample_customers"):
            lines.append("(no customers/orders found on this date)")

    else:
        lines.append(str(payload))

    lines.append("Use these values as ground truth. Explain clearly. Numbers without commas.")
    return "\n".join(lines)
