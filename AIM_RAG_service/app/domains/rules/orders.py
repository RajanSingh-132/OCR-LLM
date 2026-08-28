"""Order domain rules — Avaal_order fields (existing order_ask logic)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import DomainRules, extract_limit
from app.domains.lookup.orders.lookup import extract_token as extract_order_token
from app.order_ask.calculation_engine import is_calculation_question

DOMAIN = "orders"

STATUS_MAP = {
    "confirmed": "Confirmed",
    "delivered": "Delivered",
    "dispatched": "Dispatched",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "quoted": "Quoted",
}

GEO_FILTER_KEYS = (
    "pin",
    "state",
    "city",
    "address",
    "location",
    "pickup_location",
    "delivery_location",
)

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all)\b.*\border",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|latest|last\s+\d+|top\s+\d+)\b.*\border|\border.*\b(recent|latest)\b",
    re.I,
)
COMPARE_RE = re.compile(r"\b(compare|difference|vs\.?|versus)\b", re.I)

STICKY_KEYS = (
    "order_token",
    "customername",
    "customercode",
    "orderstatus",
    "currencycode",
    "companycode",
    "pickup_location",
    "delivery_location",
    *GEO_FILTER_KEYS,
    "location_side",
    "salesmanname",
    "commodityname",
)


def extract_record_token(question: str) -> Optional[str]:
    return extract_order_token(question)


def extract_entities(
    question: str,
    *,
    session_order_token: Optional[str] = None,
    session_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Order entity extraction (Avaal_order document fields)."""
    q = question or ""
    ql = q.lower()
    entities: Dict[str, Any] = {}

    token = extract_order_token(q)
    if token:
        entities["order_token"] = token

    for key, canonical in STATUS_MAP.items():
        if re.search(rf"\b{key}\b", ql):
            entities["orderstatus"] = canonical
            break

    m = re.search(r"\bcurrency\s*[:=]?\s*([A-Za-z]{3})\b", q, re.I)
    if m:
        entities["currencycode"] = m.group(1).upper()
    else:
        m = re.search(r"\b(CAD|USD)\b", q)
        if m:
            entities["currencycode"] = m.group(1).upper()

    m = re.search(r"\bcustomer\s*code\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["customercode"] = m.group(1).upper()

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
    m = re.search(
        r"\b(?:for\s+customer|customer\s*name|orders?\s+for\s+customer|orders?\s+of\s+customer|"
        r"orders?\s+for|orders?\s+of|customer)\s+['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60}?)['\"]?(?=\s+(?:with|status|in|and|currency|$)|$)",
        q,
        re.I,
    )
    if m and not ranking_customer_q:
        name = m.group(1).strip(" .,")
        name = re.sub(r"^(?:customer|code)\s+", "", name, flags=re.I).strip()
        if re.match(r"^(with|by|for|of|the|a|an|in|on|from|to)\b", name, re.I):
            name = ""
        if name.lower() not in STATUS_MAP and len(name) >= 2:
            entities["customername"] = name

    m = re.search(r"\bcompany(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["companycode"] = m.group(1).upper()

    m = re.search(
        r"\b(?:pickup|pick\s*up)\s*(?:location|from)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
        q,
        re.I,
    )
    if m:
        entities["pickup_location"] = m.group(1).strip(" .,")
    if not re.search(r"\bdelivered\b", ql):
        m = re.search(
            r"\b(?:delivery|deliver(?:y)?|drop)\s*(?:location|to)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
            q,
            re.I,
        )
        if m:
            loc = m.group(1).strip(" .,")
            if re.match(r"^(zip|pin|postal|pincode|code)\b", loc, re.I):
                loc = ""
            if loc.lower() not in STATUS_MAP and not re.match(r"^(ed|y)\b", loc, re.I):
                if loc:
                    entities["delivery_location"] = loc

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

    m = re.search(
        r"\b(?:pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\s*[:=#]?\s*"
        r"([A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d|\d{5}(?:-\d{4})?)\b",
        q,
        re.I,
    )
    if m:
        entities["pin"] = re.sub(r"\s+", " ", m.group(1).strip().upper())
    else:
        m = re.search(r"\b(?:in|at|for|with)\s+(\d{5}(?:-\d{4})?)\b", q, re.I)
        if m and not re.fullmatch(r"20\d{2}", m.group(1)[:4]):
            entities["pin"] = m.group(1)

    from app.order_ask.field_catalog import resolve_state_token

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
        m = re.search(
            r"\b(?:in|to|from|at|for)\s+([A-Za-z]{2}|[A-Za-z][A-Za-z\s]{2,24})\b",
            q,
            re.I,
        )
        if m:
            cand = m.group(1).strip()
            if cand.lower() not in (
                "the", "us", "usa", "canada", "customer", "orders", "order",
                "status", "confirmed", "delivered", "dispatched", "cancelled",
                "quoted", "invoiced",
            ):
                code = resolve_state_token(cand)
                if code:
                    entities["state"] = code

    m = re.search(
        r"\b(?:city|town)\s*[:=#]?\s*['\"]?([A-Za-z][A-Za-z\s\-.]{1,40})",
        q,
        re.I,
    )
    if m:
        city = m.group(1).strip(" .,")
        if len(city) >= 2:
            entities["city"] = city

    m = re.search(
        r"\b(?:full\s+)?(?:address|street)\s*[:=#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,#]{2,60})",
        q,
        re.I,
    )
    if m:
        addr = m.group(1).strip(" .,")
        if len(addr) >= 3:
            entities["address"] = addr

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

    from app.order_ask.analytics import detect_location_side

    if any(entities.get(k) for k in ("pin", "state", "city", "address", "location")):
        entities["location_side"] = detect_location_side(q)

    if entities.get("pin") and entities.get("order_token"):
        pin_norm = re.sub(r"\s+", "", str(entities["pin"])).upper()
        tok_norm = re.sub(r"\s+", "", str(entities["order_token"])).upper()
        if pin_norm == tok_norm or tok_norm in pin_norm:
            entities.pop("order_token", None)

    from app.order_ask.analytics import (
        detect_date_field,
        detect_period_days,
        extract_any_date_from_question,
        is_best_city_question,
        is_city_wise_question,
        is_date_activity_question,
        is_period_orders_question,
        is_state_wise_question,
        is_trip_distance_question,
        normalize_date_prefix,
        detect_country,
        detect_customer_direction,
        is_best_customer_question,
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

    if re.search(
        r"\b(best|highest|max|top|largest|most)\b.*\b(amount|freight|revenue|tax|distance)\b", ql
    ) or re.search(
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

    limit = extract_limit(q, default_all=25)
    if limit:
        entities["limit"] = limit
    elif re.search(r"\b(all|every)\b.*\border", ql):
        entities["limit"] = 25

    country = detect_country(q)
    if country:
        entities["country"] = country
        entities["location_side"] = detect_location_side(q)

    if is_best_customer_question(q):
        direction = detect_customer_direction(q)
        entities["customer_direction"] = direction
        entities["analytics"] = "worst_customer" if direction == "worst" else "best_customer"
        if re.search(r"\b(revenue|freight|amount|sales|money|value)\b", ql):
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

    from app.domains.rules.base import is_follow_up, merge_session_entities

    sticky = session_entities or {}
    follow_up = is_follow_up(q)
    if follow_up or (not entities.get("order_token") and session_order_token):
        if not entities.get("order_token") and session_order_token:
            if follow_up or re.search(
                r"\b(status|tax|freight|detail|details|info|amount|customer|delivery|pickup|distance|location)\b",
                ql,
            ):
                entities["order_token"] = session_order_token
                entities["from_session"] = True

    if follow_up:
        for key in STICKY_KEYS:
            if key not in entities and sticky.get(key):
                entities[key] = sticky[key]
                entities["from_session"] = True

    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
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


def has_list_filters(entities: Dict[str, Any]) -> bool:
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


def classify_intent_local(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()

    from app.order_ask.analytics import (
        is_analytics_question,
        is_best_customer_question,
        is_best_city_question,
        is_city_wise_question,
        is_country_customer_question,
        is_date_activity_question,
        is_period_orders_question,
        is_state_wise_question,
        is_status_summary_question,
        is_trip_distance_question,
    )

    if is_date_activity_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "activity_on_date",
        }

    token = extract_order_token(q)
    geo_filter_q = bool(
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal|"
            r"state|province|city|town|address|street)\b",
            q,
            re.I,
        )
    )
    if (
        token
        and not geo_filter_q
        and (
            re.search(
                r"\b(give|get|show|find|lookup|look\s*up|detail|details|info|information|fetch|pull|order\s*number|ordernumber)\b",
                q,
                re.I,
            )
            or re.match(r"^(MRP\d+|TORD\d+|\d{4,})\s*$", q, re.I)
            or len(q.split()) <= 8
        )
    ):
        if not re.search(r"\b(all|list|filter|compare|vs)\b", q, re.I) or re.search(
            r"\b(detail|details|give|get|show)\b", q, re.I
        ):
            if not LIST_RE.search(q) or token:
                return {
                    "intent": "order_lookup",
                    "needs_rag": False,
                    "needs_calculation": False,
                    "needs_exact_order": True,
                    "response_style": "detailed",
                    "max_tokens_hint": 1200,
                    "retrieve_k": 0,
                    "order_token": token,
                    "reason": "explicit_order_token",
                }

    if COMPARE_RE.search(q) and re.search(r"\b(MRP\d+|TORD\d+|\d{4,})\b", q, re.I):
        return {
            "intent": "compare",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 700,
            "retrieve_k": 0,
            "reason": "compare_orders",
        }

    if re.search(
        r"\b(best|highest|top|largest|most|lowest|smallest)\b.*\b(order|amount|freight|tax|distance|revenue)\b",
        q,
        re.I,
    ) and not re.search(r"\bcustomer\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "ranked_list",
        }

    if (
        re.search(r"\b(list|show|display|find|search|filter)\b", q, re.I)
        and re.search(
            r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b",
            q,
            re.I,
        )
        and not re.search(r"\b(how many|count|summary|breakdown|break\s*down|total)\b", q, re.I)
    ):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "status_filtered_list",
        }

    if is_best_customer_question(q):
        from app.order_ask.analytics import detect_customer_direction

        direction = detect_customer_direction(q)
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": f"{direction}_customer",
        }

    if is_best_city_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 300,
            "retrieve_k": 0,
            "reason": "best_city",
        }

    if is_state_wise_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "state_wise_orders",
        }

    if is_city_wise_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "city_wise_orders",
        }

    if is_period_orders_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "period_orders",
        }

    if is_trip_distance_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 300,
            "retrieve_k": 0,
            "reason": "trip_distance",
        }

    if is_status_summary_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "status_summary",
        }

    if is_country_customer_question(q) or is_analytics_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "country_or_analytics",
        }

    if RECENT_RE.search(q) and not is_calculation_question(q):
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "list_recent",
        }

    if (
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\b",
            q,
            re.I,
        )
        or re.search(r"\b(state|province|city|town|address|street)\b", q, re.I)
        or re.search(
            r"\b(location|warehouse|facility)\b.*\b(order|orders|pickup|delivery|drop)\b|"
            r"\b(order|orders|pickup|delivery|drop)\b.*\b(location|warehouse|facility)\b",
            q,
            re.I,
        )
        or re.search(r"\borders?\b.*\b(in|from|to|at)\b\s+[A-Za-z]{2,}", q, re.I)
    ) and not re.search(r"\b(how many\s+customers?|customer\s+count|best|worst|low)\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "geo_or_address_filter",
        }

    if (
        LIST_RE.search(q)
        or re.search(
            r"\borderstatus\b|\bstatus\s+(confirmed|delivered|dispatched|cancelled|quoted)\b",
            q,
            re.I,
        )
        or re.search(r"\borders?\s+(with|for|by|in)\b", q, re.I)
        or re.search(r"\b(on|dated|date|pickup|delivery)\b.*\b20\d{2}\b", q, re.I)
    ) and not re.search(r"\b(total|sum|average|avg|how many|count)\b", q, re.I):
        token2 = extract_order_token(q)
        if not (
            token2
            and len(q.split()) <= 5
            and not re.search(
                r"\b(status|customer|currency|confirmed|delivered|date|location)\b",
                q,
                re.I,
            )
        ):
            return {
                "intent": "list_filter",
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": False,
                "response_style": "medium",
                "max_tokens_hint": 450,
                "retrieve_k": 0,
                "reason": "filtered_list",
            }

    if token and re.search(
        r"\b(detail|details|show|get|find|lookup|look\s*up|info|information|give)\b",
        q,
        re.I,
    ):
        return {
            "intent": "order_lookup",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "reason": "explicit_order_lookup",
        }

    if token and len(q.split()) <= 4:
        return {
            "intent": "order_lookup",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "reason": "short_order_token",
        }

    if history_hint and re.search(
        r"\b(that|this|same|it|its|uska|uski|uske|wo|woh|status|tax|freight)\b",
        q,
        re.I,
    ):
        if not is_calculation_question(q) or re.search(r"\b(uska|that|this|it)\b", q, re.I):
            if re.search(r"\b(status|tax|freight|detail|amount|customer|delivery)\b", q, re.I):
                return {
                    "intent": "order_lookup",
                    "needs_rag": False,
                    "needs_calculation": False,
                    "needs_exact_order": True,
                    "response_style": "medium",
                    "max_tokens_hint": 400,
                    "retrieve_k": 0,
                    "reason": "follow_up_with_session",
                }

    if is_calculation_question(q):
        return {
            "intent": "calculation",
            "needs_rag": False,
            "needs_calculation": True,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 250,
            "retrieve_k": 0,
            "reason": "calculation_keywords",
        }

    return None


def plan_tools(
    intent: str,
    entities: Dict[str, Any],
    intent_info: Dict[str, Any],
) -> List[str]:
    from app.order_ask.tools import (
        TOOL_COMPARE,
        TOOL_GET_RECORD,
        TOOL_LIST_RECENT,
        TOOL_RUN_ANALYTICS,
        TOOL_RUN_CALCULATION,
        TOOL_SEARCH,
        TOOL_SEMANTIC_RAG,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    if (
        intent == "analytics"
        or intent_info.get("needs_analytics")
        or entities.get("analytics")
        or entities.get("country")
    ):
        tools.append(TOOL_RUN_ANALYTICS)

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if TOOL_RUN_ANALYTICS not in tools:
            tools.append(TOOL_RUN_CALCULATION)

    token = entities.get("order_token") or intent_info.get("order_token")
    if intent in ("order_lookup", "record_lookup") or intent_info.get("needs_exact_order") or token:
        pin = str(entities.get("pin") or "")
        fake_pin_token = bool(
            token
            and pin
            and re.sub(r"\s+", "", str(token)).upper()
            == re.sub(r"\s+", "", pin).upper()
        )
        if token and not fake_pin_token and not (
            TOOL_RUN_ANALYTICS in tools and re.fullmatch(r"20\d{2}", str(token))
        ):
            if not entities.get("order_token") and intent_info.get("order_token"):
                entities["order_token"] = intent_info["order_token"]
            tools.append(TOOL_GET_RECORD)

    if intent in ("list_filter", "list_orders", "filter") or (
        entities.get("sort_by") and intent not in ("analytics", "calculation")
    ):
        tools.append(TOOL_SEARCH)

    if has_list_filters(entities) and TOOL_RUN_ANALYTICS not in tools and TOOL_SEARCH not in tools and TOOL_GET_RECORD not in tools:
        tools.append(TOOL_SEARCH)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent == "compare":
        tools.append(TOOL_COMPARE)

    if intent_info.get("needs_rag") or intent == "open_qa":
        filters = entities_to_mongo_filters(entities)
        if filters and TOOL_SEARCH not in tools:
            tools.append(TOOL_SEARCH)
        elif (
            TOOL_GET_RECORD not in tools
            and TOOL_SEARCH not in tools
            and TOOL_RUN_ANALYTICS not in tools
        ):
            tools.append(TOOL_SEMANTIC_RAG)

    seen: set[str] = set()
    ordered: List[str] = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


RULES = DomainRules(
    name=DOMAIN,
    extract_entities=extract_entities,
    classify_intent_local=classify_intent_local,
    entities_to_mongo_filters=entities_to_mongo_filters,
    has_list_filters=has_list_filters,
    plan_tools=plan_tools,
    extract_record_token=extract_record_token,
    sticky_entity_keys=STICKY_KEYS,
    compare_token_pattern=re.compile(r"\b(MRP\d+|TORD\d+|\d{4,})\b", re.I),
)
