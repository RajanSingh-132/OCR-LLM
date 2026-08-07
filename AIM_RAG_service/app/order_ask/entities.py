"""
Entity extraction for accurate Avaal order lists / follow-ups.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.order_ask.checkpoint import checkpoint
from app.order_ask.rag_retrieval import extract_order_token

STATUS_MAP = {
    "confirmed": "Confirmed",
    "delivered": "Delivered",
    "dispatched": "Dispatched",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "quoted": "Quoted",
}


def extract_entities(
    question: str,
    *,
    session_order_token: Optional[str] = None,
    session_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract entities from the current question and merge sticky session values
    when the user refers to 'that order' / 'same customer' etc.
    """
    q = question or ""
    ql = q.lower()
    entities: Dict[str, Any] = {}

    token = extract_order_token(q)
    if token:
        entities["order_token"] = token

    # Status
    for key, canonical in STATUS_MAP.items():
        if re.search(rf"\b{key}\b", ql):
            entities["orderstatus"] = canonical
            break

    # Currency
    m = re.search(r"\bcurrency\s*[:=]?\s*([A-Za-z]{3})\b", q, re.I)
    if m:
        entities["currencycode"] = m.group(1).upper()
    else:
        m = re.search(r"\b(CAD|USD)\b", q)
        if m:
            entities["currencycode"] = m.group(1).upper()

    # Customer code only when explicit
    m = re.search(r"\bcustomer\s*code\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["customercode"] = m.group(1).upper()

    # Customer name
    m = re.search(
        r"\b(?:for\s+customer|customer\s*name|orders?\s+for\s+customer|orders?\s+of\s+customer|"
        r"orders?\s+for|orders?\s+of|customer)\s+['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60}?)['\"]?(?=\s+(?:with|status|in|and|currency|$)|$)",
        q,
        re.I,
    )
    if m and not re.search(r"\bbest\b.*\bcustomer\b|\bcustomer\b.*\bbest\b|\bhow many\b.*\bcustomer", ql):
        name = m.group(1).strip(" .,")
        name = re.sub(r"^(?:customer|code)\s+", "", name, flags=re.I).strip()
        if name.lower() not in STATUS_MAP and len(name) >= 2:
            entities["customername"] = name

    # Company code
    m = re.search(r"\bcompany(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["companycode"] = m.group(1).upper()

    # Locations
    m = re.search(
        r"\b(?:pickup|pick\s*up)\s*(?:location|from)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
        q,
        re.I,
    )
    if m:
        entities["pickup_location"] = m.group(1).strip(" .,")
    # Avoid matching status word "delivered" as a delivery location
    if not re.search(r"\bdelivered\b", ql):
        m = re.search(
            r"\b(?:delivery|deliver(?:y)?|drop)\s*(?:location|to)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
            q,
            re.I,
        )
        if m:
            loc = m.group(1).strip(" .,")
            if loc.lower() not in STATUS_MAP and not re.match(r"^(ed|y)\b", loc, re.I):
                entities["delivery_location"] = loc

    # Dates: YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY (Order Sheet style)
    from app.order_ask.analytics import (
        detect_date_field,
        extract_date_from_question,
        is_date_activity_question,
        normalize_date_prefix,
    )

    date_val = extract_date_from_question(q)
    if not date_val:
        m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", q)
        if m:
            date_val = normalize_date_prefix(m.group(1))
        else:
            m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b", q)
            if m:
                date_val = normalize_date_prefix(m.group(1))

    if date_val:
        date_field = detect_date_field(q)
        entities["analytics_date"] = date_val
        entities["date_field"] = date_field
        if date_field == "pickupdate":
            entities["pickupdate"] = date_val
        elif date_field == "deliverydate":
            entities["deliverydate"] = date_val
        else:
            entities["orderdate"] = date_val
        if is_date_activity_question(q) or re.search(
            r"\b(customer|kitne|kitna|how many|count|ordered)\b", ql
        ):
            entities["analytics"] = "activity_on_date"

    # Sort / best / highest amount
    if re.search(r"\b(best|highest|max|top|largest|most)\b.*\b(amount|freight|revenue|tax|distance)\b", ql) or re.search(
        r"\b(amount|freight|revenue|tax|distance)\b.*\b(best|highest|max|top|largest|most)\b",
        ql,
    ):
        entities["limit"] = entities.get("limit") or 5
        if "tax" in ql:
            entities["sort_by"] = "taxes"
        elif "distance" in ql:
            entities["sort_by"] = "distance"
        elif "revenue" in ql or "gross" in ql:
            entities["sort_by"] = "grosstotalfreight"
        else:
            entities["sort_by"] = "totalfreight"
        entities["ascending"] = False

    if re.search(r"\b(lowest|smallest|least|cheapest|min)\b.*\b(amount|freight|tax|distance)\b", ql):
        entities["limit"] = entities.get("limit") or 5
        if "tax" in ql:
            entities["sort_by"] = "taxes"
        elif "distance" in ql:
            entities["sort_by"] = "distance"
        else:
            entities["sort_by"] = "totalfreight"
        entities["ascending"] = True

    # Limit / top N
    m = re.search(r"\b(?:top|last|recent|show|list)\s+(\d{1,3})\b", ql)
    if m:
        entities["limit"] = min(50, max(1, int(m.group(1))))
    elif re.search(r"\b(all|every)\b.*\border", ql):
        entities["limit"] = 25

    # Country + pickup/delivery side (for customer geo analytics)
    from app.order_ask.analytics import detect_country, detect_location_side

    country = detect_country(q)
    if country:
        entities["country"] = country
        entities["location_side"] = detect_location_side(q)

    if re.search(r"\bbest\b.*\bcustomer\b|\bcustomer\b.*\bbest\b", ql):
        entities["analytics"] = "best_customer"
        if re.search(r"\b(revenue|freight|amount|sales)\b", ql):
            entities["best_customer_metric"] = "revenue"
        else:
            entities["best_customer_metric"] = "orders"

    if re.search(r"\bstatus\b.*\b(summary|break|count|how many)\b|\b(summary)\b.*\bstatus\b", ql):
        entities["analytics"] = "status_summary"
    elif re.search(
        r"\b(how many|count)\b.*\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b",
        ql,
    ) or re.search(
        r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b.*\b(how many|count)\b",
        ql,
    ):
        entities["analytics"] = "status_summary"

    # Follow-up pronouns → reuse session
    sticky = session_entities or {}
    follow_up = bool(
        re.search(
            r"\b(that|this|same|it|its|uska|uski|uske|wo|woh|previous|again)\b",
            ql,
        )
    )
    if follow_up or (not entities.get("order_token") and session_order_token):
        if not entities.get("order_token") and session_order_token:
            if follow_up or re.search(
                r"\b(status|tax|freight|detail|details|info|amount|customer|delivery|pickup|distance|location)\b",
                ql,
            ):
                entities["order_token"] = session_order_token
                entities["from_session"] = True

    if follow_up:
        for key in (
            "customername",
            "customercode",
            "orderstatus",
            "currencycode",
            "companycode",
            "pickup_location",
            "delivery_location",
        ):
            if key not in entities and sticky.get(key):
                entities[key] = sticky[key]
                entities["from_session"] = True

    checkpoint("ENTITIES", "extracted", **{k: v for k, v in entities.items()})
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Map entities → Mongo filter fields (excluding order_token / limit / sort)."""
    filters: Dict[str, Any] = {}
    for key in (
        "orderstatus",
        "currencycode",
        "customercode",
        "companycode",
        "customername",
        "pickup_location",
        "delivery_location",
        "orderdate",
        "pickupdate",
        "deliverydate",
    ):
        if entities.get(key):
            filters[key] = entities[key]
    return filters
