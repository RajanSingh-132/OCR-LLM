"""
Entity extraction for accurate Avaal order lists / follow-ups.
Supports pin / state / city / address / location / customer filtering.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.order_ask.checkpoint import checkpoint
from app.order_ask.field_catalog import resolve_state_token
from app.order_ask.rag_retrieval import extract_order_token

STATUS_MAP = {
    "confirmed": "Confirmed",
    "delivered": "Delivered",
    "dispatched": "Dispatched",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "quoted": "Quoted",
}

# Geo / address filter entity keys (used by Mongo + tool planner)
GEO_FILTER_KEYS = (
    "pin",
    "state",
    "city",
    "address",
    "location",
    "pickup_location",
    "delivery_location",
)


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

    from app.order_ask.trip_retrieval import extract_trip_token

    trip_token = extract_trip_token(q)
    if trip_token:
        entities["trip_token"] = trip_token

    token = extract_order_token(q)
    # Don't treat ETP as order_token
    if token and re.match(r"^ETP\d+$", str(token), re.I):
        entities["trip_token"] = str(token).upper()
        token = None
    if token:
        entities["order_token"] = token

    if re.search(r"\btrips?\b", ql) and entities.get("order_token") and not entities.get("trip_token"):
        entities["want_trip_for_order"] = True
    if entities.get("trip_token") and re.search(
        r"\b(order|orders|ordernumber|which order|kis order)\b", ql
    ):
        entities["want_orders_for_trip"] = True

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
    # Skip name capture on ranking/count questions (best/worst/low/how many customers)
    ranking_customer_q = bool(
        re.search(
            r"\b(best|top|worst|lowest|low|least|fewest|smallest|minimum|min|bottom|"
            r"biggest|largest|highest|most|maximum|max)\b.*\bcustomers?\b|"
            r"\bcustomers?\b.*\b(best|top|worst|lowest|low|least|fewest|smallest|"
            r"minimum|min|bottom|most|maximum|max)\b|"
            r"\bhow many\b.*\bcustomers?\b",
            ql,
        )
    )
    if m and not ranking_customer_q:
        name = m.group(1).strip(" .,")
        name = re.sub(r"^(?:customer|code)\s+", "", name, flags=re.I).strip()
        # Ignore junk fragments like "with least orders", "by revenue"
        if re.match(
            r"^(with|by|for|of|the|a|an|in|on|from|to)\b", name, re.I
        ):
            name = ""
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
            # Skip "zip 79927" / "pin 92881" — those are pin filters, not location names
            if re.match(r"^(zip|pin|postal|pincode|code)\b", loc, re.I):
                loc = ""
            if loc.lower() not in STATUS_MAP and not re.match(r"^(ed|y)\b", loc, re.I):
                if loc:
                    entities["delivery_location"] = loc

    # Generic location (not already captured as pickup/delivery)
    m = re.search(
        r"\b(?:at\s+)?(?:location|place|facility|warehouse)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
        q,
        re.I,
    )
    if m and not entities.get("pickup_location") and not entities.get("delivery_location"):
        loc = m.group(1).strip(" .,")
        if loc.lower() not in STATUS_MAP and len(loc) >= 2:
            entities["location"] = loc

    # Pin / zip / postal (US 5-digit or Canadian A1A 1A1)
    m = re.search(
        r"\b(?:pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\s*[:=#]?\s*"
        r"([A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d|\d{5}(?:-\d{4})?)\b",
        q,
        re.I,
    )
    if m:
        entities["pin"] = re.sub(r"\s+", " ", m.group(1).strip().upper())
    else:
        # "orders in 92881" / bare zip (not a year)
        m = re.search(
            r"\b(?:in|at|for|with)\s+(\d{5}(?:-\d{4})?)\b",
            q,
            re.I,
        )
        if m and not re.fullmatch(r"20\d{2}", m.group(1)[:4]):
            entities["pin"] = m.group(1)

    # State / province — labeled or known name/code
    m = re.search(
        r"\b(?:state|province|region)\s*[:=#]?\s*['\"]?([A-Za-z][A-Za-z\s.]{1,30})",
        q,
        re.I,
    )
    if m:
        raw_state = m.group(1).strip(" .,")
        code = resolve_state_token(raw_state) or resolve_state_token(raw_state.split()[0])
        if code:
            entities["state"] = code
        elif len(raw_state) <= 3:
            entities["state"] = raw_state.upper()
        else:
            entities["state"] = raw_state.title()
    else:
        # "in California" / "to TX" / "from Ontario" (skip country-only US/Canada)
        m = re.search(
            r"\b(?:in|to|from|at|for)\s+([A-Za-z]{2}|[A-Za-z][A-Za-z\s]{2,24})\b",
            q,
            re.I,
        )
        if m:
            cand = m.group(1).strip()
            # Avoid swallowing "in the us" / customer phrases handled elsewhere
            if cand.lower() not in (
                "the",
                "us",
                "usa",
                "canada",
                "customer",
                "orders",
                "order",
                "status",
                "confirmed",
                "delivered",
                "dispatched",
                "cancelled",
                "quoted",
                "invoiced",
            ):
                code = resolve_state_token(cand)
                if code:
                    entities["state"] = code

    # City
    m = re.search(
        r"\b(?:city|town)\s*[:=#]?\s*['\"]?([A-Za-z][A-Za-z\s\-.]{1,40})",
        q,
        re.I,
    )
    if m:
        city = m.group(1).strip(" .,")
        if len(city) >= 2:
            entities["city"] = city

    # Address / street fragment
    m = re.search(
        r"\b(?:full\s+)?(?:address|street)\s*[:=#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,#]{2,60})",
        q,
        re.I,
    )
    if m:
        addr = m.group(1).strip(" .,")
        if len(addr) >= 3:
            entities["address"] = addr

    # Salesman / commodity filters
    m = re.search(
        r"\b(?:salesman|sales\s*person|sales\s*rep)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,40})",
        q,
        re.I,
    )
    if m:
        entities["salesmanname"] = m.group(1).strip(" .,")

    m = re.search(
        r"\b(?:commodity|product)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,40})",
        q,
        re.I,
    )
    if m:
        entities["commodityname"] = m.group(1).strip(" .,")

    # Which side of the address to search (pickup / delivery / both)
    from app.order_ask.analytics import detect_location_side

    if any(entities.get(k) for k in ("pin", "state", "city", "address", "location")):
        entities["location_side"] = detect_location_side(q)

    # Pin digits must not stay as order_token (false order lookup)
    if entities.get("pin") and entities.get("order_token"):
        pin_norm = re.sub(r"\s+", "", str(entities["pin"])).upper()
        tok_norm = re.sub(r"\s+", "", str(entities["order_token"])).upper()
        if pin_norm == tok_norm or tok_norm in pin_norm:
            entities.pop("order_token", None)

    # Dates: YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY (Order Sheet style)
    from app.order_ask.analytics import (
        detect_date_field,
        detect_period_days,
        extract_any_date_from_question,
        extract_date_from_question,
        is_best_city_question,
        is_city_wise_question,
        is_date_activity_question,
        is_period_orders_question,
        is_state_wise_question,
        is_trip_distance_question,
        normalize_date_prefix,
    )
    from app.order_ask.trip_analytics import (
        detect_distance_direction,
        detect_trip_rank_direction,
        is_best_worst_trip_question,
        is_longest_shortest_trip_question,
        is_trip_status_summary_question,
    )

    date_val = extract_any_date_from_question(q)
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
            r"\b(customer|kitne|kitna|how many|count|ordered|created|total)\b", ql
        ):
            entities["analytics"] = "activity_on_date"

    period_days = detect_period_days(q)
    if period_days:
        entities["period_days"] = period_days
        if is_period_orders_question(q):
            entities["analytics"] = "orders_in_period"

    if is_state_wise_question(q):
        entities["analytics"] = "orders_by_state"
    if is_city_wise_question(q):
        entities["analytics"] = "orders_by_city"
    if is_best_city_question(q):
        entities["analytics"] = "best_city"
    if is_trip_distance_question(q):
        entities["analytics"] = "trip_distance"
    if is_best_worst_trip_question(q):
        direction = detect_trip_rank_direction(q)
        entities["analytics"] = "worst_trip" if direction == "worst" else "best_trip"
        entities["trip_direction"] = direction
        entities["limit"] = entities.get("limit") or 5
    if is_longest_shortest_trip_question(q):
        direction = detect_distance_direction(q)
        entities["analytics"] = (
            "shortest_trip" if direction == "shortest" else "longest_trip"
        )
        entities["distance_direction"] = direction
        entities["limit"] = entities.get("limit") or 5
    if is_trip_status_summary_question(q):
        entities["analytics"] = "trip_status_summary"

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

    from app.order_ask.analytics import (
        detect_customer_direction,
        is_best_customer_question,
    )

    if is_best_customer_question(q):
        direction = detect_customer_direction(q)
        entities["customer_direction"] = direction
        entities["analytics"] = (
            "worst_customer" if direction == "worst" else "best_customer"
        )
        if re.search(r"\b(revenue|freight|amount|sales|money|value)\b", ql):
            entities["best_customer_metric"] = "revenue"
        else:
            entities["best_customer_metric"] = "orders"

    if re.search(r"\bstatus\b.*\b(summary|break|count|how many)\b|\b(summary)\b.*\bstatus\b", ql):
        if re.search(r"\btrips?\b", ql):
            entities["analytics"] = "trip_status_summary"
        else:
            entities["analytics"] = "status_summary"
    elif re.search(
        r"\b(how many|count)\b.*\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b",
        ql,
    ) or re.search(
        r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b.*\b(how many|count)\b",
        ql,
    ):
        if re.search(r"\btrips?\b", ql):
            entities["analytics"] = "trip_status_summary"
        else:
            entities["analytics"] = "status_summary"

    # Follow-up pronouns → reuse session
    sticky = session_entities or {}
    session_trip_token = sticky.get("trip_token")
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
                # Prefer trip sticky if follow-up is trip-flavored
                if session_trip_token and re.search(
                    r"\b(trip|driver|truck|etp)\b", ql
                ):
                    entities["trip_token"] = session_trip_token
                    entities["from_session"] = True
                else:
                    entities["order_token"] = session_order_token
                    entities["from_session"] = True

    if not entities.get("trip_token") and session_trip_token:
        if follow_up or re.search(
            r"\b(trip|driver|truck|status|distance|customer|country)\b",
            ql,
        ):
            if follow_up or re.search(r"\btrips?\b", ql):
                entities["trip_token"] = session_trip_token
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
            "pin",
            "state",
            "city",
            "address",
            "location",
            "location_side",
            "salesmanname",
            "commodityname",
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
        "pin",
        "state",
        "city",
        "address",
        "location",
        "location_side",
        "salesmanname",
        "commodityname",
    ):
        if entities.get(key):
            filters[key] = entities[key]
    return filters


def has_geo_or_list_filters(entities: Dict[str, Any]) -> bool:
    """True when structured list search should run from entities."""
    return any(
        entities.get(k)
        for k in (
            "orderstatus",
            "currencycode",
            "customercode",
            "companycode",
            "customername",
            "orderdate",
            "pickupdate",
            "deliverydate",
            "salesmanname",
            "commodityname",
            *GEO_FILTER_KEYS,
        )
    )
