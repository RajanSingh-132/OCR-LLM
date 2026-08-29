"""Trip domain rules — Avaal_trip fields + analytics."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import (
    DomainRules,
    extract_limit,
    merge_session_entities,
    _text_filter,
)
from app.domains.lookup.trips.lookup import (
    INTENT_NAME as TRIP_LOOKUP_INTENT,
    extract_token as extract_trip_token,
    is_list_or_status_question,
    try_lookup_intent,
)
from app.domains.lookup.base import is_ask_for_record_id_question
from app.order_ask.calculation_engine import is_calculation_question
from app.order_ask.trip_analytics import (
    detect_location_side,
    is_best_trip_question,
    is_status_summary_question,
    is_trip_analytics_question,
    is_trips_by_country_question,
    is_worst_trip_question,
)

DOMAIN = "trips"

# Match Avaal_trip document field names (lowercase in Mongo).
FIELD_ID = "tripid"
FIELD_NUMBER = "tripnumber"
FIELD_DRIVER = "firstdrivername"
FIELD_DRIVER_ALT = "seconddrivername"
FIELD_TRUCK = "trucknumber"
FIELD_TRAILER = "firsttrailernumber"
FIELD_STATUS = "tripstatus"
FIELD_TYPE = "triptype"
FIELD_CUSTOMER = "customername"
FIELD_COMMODITY = "commodity"
FIELD_SALESMAN = "salesmannames"
FIELD_PICKUP_COUNTRY = "pickupcountry"
FIELD_DELIVERY_COUNTRY = "deliverycountry"
FIELD_PICKUP_CITY = "pickupcity"
FIELD_DELIVERY_CITY = "deliverycity"
FIELD_PICKUP_LOC = "pickuplocationname"
FIELD_DELIVERY_LOC = "deliverylocationname"

_DRIVER_VALUE_STOPWORDS = {
    "name",
    "names",
    "details",
    "detail",
    "status",
    "info",
    "information",
    "also",
    "the",
    "and",
    "with",
    "please",
    "trip",
    "trips",
    "recent",
    "recently",
    "latest",
    "phone",
    "number",
    "first",
    "second",
}

# UI trip statuses (+ DB aliases). Prefer longer phrases first when matching.
STATUS_WORDS = (
    "partially delivered",
    "in-transit",
    "in transit",
    "intransit",
    "en route",
    "enroute",
    "cancelled",
    "canceled",
    "dispatched",
    "delivered",
    "rejected",
    "planned",
    "started",
    "stated",  # common typo for Started
    "active",
    "completed",
    "complete",
    "open",
    "closed",
)

# Canonical UI values (prompts + analytics known list)
KNOWN_TRIP_STATUSES = (
    "Planned",
    "Dispatched",
    "Started",
    "In-Transit",
    "Delivered",
    "Rejected",
)

TYPE_WORDS = (
    "regular - loaded",
    "regular - empty",
    "outsourcing - loaded",
    "outsourcing - empty",
    "regular",
    "outsourcing",
    "loaded",
    "empty",
)

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all|give|get)\b.*\btrips?\b",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|recently|latest|last\s+\d+|top\s+\d+|top)\b.*\btrips?\b|"
    r"\btrips?\b.*\b(recent|recently|latest|top)\b",
    re.I,
)

STICKY_KEYS = (
    "record_token",
    FIELD_DRIVER,
    FIELD_TRUCK,
    FIELD_TRAILER,
    FIELD_STATUS,
    FIELD_TYPE,
    FIELD_CUSTOMER,
    FIELD_COMMODITY,
    FIELD_SALESMAN,
    FIELD_PICKUP_COUNTRY,
    FIELD_DELIVERY_COUNTRY,
    "location_side",
    "analytics",
    "sort_by",
    "ascending",
    "limit",
    "focus_fields",
)


def extract_record_token(question: str) -> Optional[str]:
    return extract_trip_token(question)


def _canonical_status(raw: str) -> str:
    """Map user words → canonical tripstatus (UI: Planned…Rejected)."""
    s = re.sub(r"\s+", " ", (raw or "").strip().lower().replace("_", " "))
    s = s.replace("trasit", "transit")  # typo
    mapping = {
        "canceled": "Cancelled",
        "cancelled": "Cancelled",
        "dispatched": "Dispatched",
        "delivered": "Delivered",
        "deliverd": "Delivered",
        "enroute": "In-Transit",
        "en route": "In-Transit",
        "in transit": "In-Transit",
        "in-transit": "In-Transit",
        "intransit": "In-Transit",
        "planned": "Planned",
        "started": "Started",
        "stated": "Started",
        "rejected": "Rejected",
        "reject": "Rejected",
        "completed": "Delivered",
        "complete": "Delivered",
        "active": "Started",
        "open": "Planned",
        "closed": "Delivered",
    }
    return mapping.get(s, raw.strip().title() if raw else raw)


def _status_mongo_filter(canonical: str) -> Dict[str, Any]:
    """
    Case-insensitive match; In-Transit also matches DB 'In Transit' / Enroute.
    """
    c = (canonical or "").strip()
    cl = c.lower().replace(" ", "").replace("-", "")
    if cl in ("intransit", "enroute"):
        return {
            "$regex": r"^(In[-\s]?Transit|En\s*Route|Enroute)$",
            "$options": "i",
        }
    if cl == "dispatched":
        return {"$regex": r"^Dispatched$", "$options": "i"}
    if cl == "delivered":
        return {"$regex": r"^Delivered$", "$options": "i"}
    if cl == "cancelled":
        return {"$regex": r"^Cancell?ed$", "$options": "i"}
    return {"$regex": f"^{re.escape(c)}$", "$options": "i"}


def _detect_focus_fields(ql: str) -> List[str]:
    """Which trip attributes the user is asking about (for LLM guidance)."""
    focus: List[str] = []
    checks = [
        (r"\bpick\s*up\b.*\bcountr|\bpickupcountry\b", ["pickupcountry"]),
        (r"\bdeliver(?:y|y)?\b.*\bcountr|\bdeliverycountry\b", ["deliverycountry"]),
        (r"\bpick\s*up\b.*\blocation\s*name|\bpickuplocationname\b", ["pickuplocationname"]),
        (r"\bdeliver(?:y|y)?\b.*\blocation\s*name|\bdeliverylocationname\b", ["deliverylocationname"]),
        (r"\bpick\s*up\b.*\b(address|location|city)|\bpickup(city|fulladdress)\b", ["pickupcity", "pickupfulladdress", "pickuplocationname"]),
        (r"\bdeliver(?:y|y)?\b.*\b(address|location|city)|\bdelivery(city|fulladdress)\b", ["deliverycity", "deliveryfulladdress", "deliverylocationname"]),
        (r"\bpick\s*up\b.*\bdate|\bfirstpickupdate\b", ["firstpickupdate", "firstpickupdatetime"]),
        (r"\bdeliver(?:y|y)?\b.*\bdate|\blastdeliverydate\b", ["lastdeliverydate", "lastdeliverydatetime"]),
        (r"\b(first\s+)?driver\b.*\bphone|\bfirstdriver(phone|cell)", ["firstdrivername", "firstdriverphone", "firstdrivercell1"]),
        (r"\bsecond\s+driver\b.*\bphone|\bseconddriver(phone|cell)", ["seconddrivername", "seconddriverphone", "seconddrivercell1"]),
        (r"\b(both|two|2)\s+drivers?\b|\bfirst\s+and\s+second\s+driver|\bdrivers?\b", ["firstdrivername", "firstdriverphone", "seconddrivername", "seconddriverphone"]),
        (r"\bsecond\s+driver\b", ["seconddrivername", "seconddriverphone", "seconddrivercell1"]),
        (r"\b(first\s+)?driver\b", ["firstdrivername", "firstdriverphone", "firstdrivercell1"]),
        (r"\btrip\s*status\b|\bstatus\b", ["tripstatus"]),
        (r"\btrip\s*type\b|\btriptype\b", ["triptype", "triptypemain"]),
        (r"\bcustomer\b", ["customername", "customercodes"]),
        (r"\bcommodit", ["commodity"]),
        (r"\bsales\s*man|\bsalesman", ["salesmannames", "salesmancodes"]),
        (r"\btotalloaddistance\b|\bload\s*distance\b", ["totalloaddistance", "distanceunit"]),
        (r"\bdistance\b", ["totalloaddistance", "triptotaldistance", "totaldistance", "distanceunit"]),
        (r"\btruck\b", ["trucknumber"]),
        (r"\btrailer\b", ["firsttrailernumber", "secondtrailernumber"]),
    ]
    for pattern, fields in checks:
        if re.search(pattern, ql, re.I):
            for f in fields:
                if f not in focus:
                    focus.append(f)
    return focus


def extract_entities(
    question: str,
    *,
    session_order_token: Optional[str] = None,
    session_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    q = question or ""
    ql = q.lower()
    entities: Dict[str, Any] = {}

    token = extract_record_token(q)
    if token:
        entities["record_token"] = token

    for status in STATUS_WORDS:
        if re.search(rf"\b{re.escape(status)}\b", ql):
            entities[FIELD_STATUS] = _canonical_status(status)
            break

    for tword in TYPE_WORDS:
        if re.search(rf"\b{re.escape(tword)}\b", ql):
            # Keep phrase for regex filter (e.g. "REGULAR - Loaded")
            entities[FIELD_TYPE] = tword
            break

    # Prefer "driver name is X" / "driver: X" — ignore bare "driver name" requests.
    m = re.search(
        r"\b(?:first\s+|second\s+)?driver(?:\s*name)?\s*(?:is|=|:|#)\s*['\"]?"
        r"([A-Za-z][A-Za-z0-9 &\-./]{1,40})",
        q,
        re.I,
    )
    if not m:
        m = re.search(
            r"\b(?:first\s+|second\s+)?driver\s+([A-Za-z][A-Za-z0-9 &\-./]{1,40})",
            q,
            re.I,
        )
    if m:
        driver = m.group(1).strip(" .,")
        first = driver.split()[0].lower() if driver else ""
        if (
            first
            and first not in _DRIVER_VALUE_STOPWORDS
            and driver.lower() not in _DRIVER_VALUE_STOPWORDS
        ):
            entities[FIELD_DRIVER] = driver

    m = re.search(
        r"\b(?:truck|truck\s*number|truck\s*no)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./]{1,30})",
        q,
        re.I,
    )
    if m:
        truck = m.group(1).strip(" .,")
        if truck.lower() not in {"number", "no", "num", "details", "status", "name"}:
            entities[FIELD_TRUCK] = truck

    m = re.search(
        r"\b(?:trailer|trailer\s*number|trailer\s*no)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./]{1,30})",
        q,
        re.I,
    )
    if m:
        trailer = m.group(1).strip(" .,")
        if trailer.lower() not in {"number", "no", "num", "details", "status", "name"}:
            entities[FIELD_TRAILER] = trailer

    m = re.search(
        r"\b(?:customer|customer\s*name)\s*(?:is|=|:|#)\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,@]{1,50})",
        q,
        re.I,
    )
    if m:
        cust = m.group(1).strip(" .,")
        if cust.lower() not in {
            "name", "names", "on", "for", "the", "who", "which",
            "commodity", "salesman", "status", "type", "driver",
        }:
            entities[FIELD_CUSTOMER] = cust

    m = re.search(
        r"\b(?:commodity|product)\s*(?:is|=|:|#)\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,40})",
        q,
        re.I,
    )
    if m:
        commodity = m.group(1).strip(" .,")
        if commodity.lower() not in {"name", "is", "what", "the", "salesman", "customer"}:
            entities[FIELD_COMMODITY] = commodity

    m = re.search(
        r"\b(?:salesman|sales\s*man|sales\s*person)\s*(?:name\s*)?(?:is|=|:|#)\s*['\"]?"
        r"([A-Za-z][A-Za-z0-9 &\-./]{1,40})",
        q,
        re.I,
    )
    if m:
        sales = m.group(1).strip(" .,")
        if sales.lower() not in {"name", "names", "who", "the", "is"}:
            entities[FIELD_SALESMAN] = sales

    # Country filters
    side = detect_location_side(q)
    entities["location_side"] = side
    m = re.search(
        r"\b(?:country)\s*(?:is|=|:|#)?\s*['\"]?(canada|united states|usa|india|us)\b",
        q,
        re.I,
    )
    if not m:
        m = re.search(r"\b(canada|united states|usa|india)\b", q, re.I)
    if m:
        country = m.group(1).strip()
        cl = country.lower()
        if cl in {"usa", "us"}:
            country = "United States"
        elif cl == "canada":
            country = "Canada"
        elif cl == "india":
            country = "India"
        if side == "delivery":
            entities[FIELD_DELIVERY_COUNTRY] = country
        elif side == "pickup":
            entities[FIELD_PICKUP_COUNTRY] = country
        else:
            entities[FIELD_PICKUP_COUNTRY] = country

    # Analytics markers
    if is_worst_trip_question(q):
        entities["analytics"] = "worst_trip"
    elif is_best_trip_question(q):
        entities["analytics"] = "best_trip"
    elif is_trips_by_country_question(q):
        entities["analytics"] = "trips_by_country"
    elif is_status_summary_question(q) or (
        entities.get(FIELD_STATUS)
        and re.search(r"\b(how many|count|kitne|kitna|total)\b", ql)
    ):
        entities["analytics"] = "trip_status_summary"

    focus = _detect_focus_fields(ql)
    if focus:
        entities["focus_fields"] = focus

    if re.search(r"\b(recent|recently|latest|newest)\b", ql):
        entities["sort_by"] = FIELD_ID
        entities["ascending"] = False

    if re.search(r"\b(longest|highest|maximum|max)\b.*\bdistance\b|"
                 r"\bdistance\b.*\b(longest|highest|maximum|max)\b|"
                 r"\bbest\s+trip\b", ql):
        entities["sort_by"] = "totalloaddistance"
        entities["ascending"] = False
    if re.search(r"\b(shortest|lowest|minimum|min)\b.*\bdistance\b|"
                 r"\bdistance\b.*\b(shortest|lowest|minimum|min)\b|"
                 r"\bworst\s+trip\b", ql):
        entities["sort_by"] = "totalloaddistance"
        entities["ascending"] = True

    limit = extract_limit(q, default_all=25, some_default=10)
    if limit:
        entities["limit"] = limit
    elif re.search(r"\b(some|any|few)\b", ql):
        entities["limit"] = entities.get("limit") or 10

    merge_session_entities(
        entities,
        question=q,
        session_token=session_order_token,
        session_entities=session_entities,
        sticky_keys=STICKY_KEYS,
        token_key="record_token",
        follow_up_token_triggers=(
            "status",
            "detail",
            "details",
            "driver",
            "truck",
            "trailer",
            "pickup",
            "delivery",
            "customer",
            "commodity",
            "salesman",
            "distance",
            "country",
            "phone",
            "type",
            "location",
            "date",
        ),
    )
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    status = entities.get(FIELD_STATUS)
    if status:
        filters[FIELD_STATUS] = _status_mongo_filter(str(status))

    if entities.get(FIELD_TYPE):
        filters.update(_text_filter(FIELD_TYPE, str(entities[FIELD_TYPE])))

    if entities.get(FIELD_DRIVER):
        driver = str(entities[FIELD_DRIVER])
        filters["$or"] = [
            _text_filter(FIELD_DRIVER, driver),
            _text_filter(FIELD_DRIVER_ALT, driver),
        ]

    if entities.get(FIELD_TRUCK):
        filters.update(_text_filter(FIELD_TRUCK, str(entities[FIELD_TRUCK])))
    if entities.get(FIELD_TRAILER):
        filters.update(_text_filter(FIELD_TRAILER, str(entities[FIELD_TRAILER])))
    if entities.get(FIELD_CUSTOMER):
        filters.update(_text_filter(FIELD_CUSTOMER, str(entities[FIELD_CUSTOMER])))
    if entities.get(FIELD_COMMODITY):
        filters.update(_text_filter(FIELD_COMMODITY, str(entities[FIELD_COMMODITY])))
    if entities.get(FIELD_SALESMAN):
        filters.update(_text_filter(FIELD_SALESMAN, str(entities[FIELD_SALESMAN])))
    if entities.get(FIELD_PICKUP_COUNTRY):
        filters.update(
            _text_filter(FIELD_PICKUP_COUNTRY, str(entities[FIELD_PICKUP_COUNTRY]))
        )
    if entities.get(FIELD_DELIVERY_COUNTRY):
        filters.update(
            _text_filter(FIELD_DELIVERY_COUNTRY, str(entities[FIELD_DELIVERY_COUNTRY]))
        )
    if entities.get(FIELD_PICKUP_CITY):
        filters.update(_text_filter(FIELD_PICKUP_CITY, str(entities[FIELD_PICKUP_CITY])))
    if entities.get(FIELD_DELIVERY_CITY):
        filters.update(
            _text_filter(FIELD_DELIVERY_CITY, str(entities[FIELD_DELIVERY_CITY]))
        )
    if entities.get(FIELD_PICKUP_LOC):
        filters.update(_text_filter(FIELD_PICKUP_LOC, str(entities[FIELD_PICKUP_LOC])))
    if entities.get(FIELD_DELIVERY_LOC):
        filters.update(
            _text_filter(FIELD_DELIVERY_LOC, str(entities[FIELD_DELIVERY_LOC]))
        )
    return filters


def has_list_filters(entities: Dict[str, Any]) -> bool:
    return bool(entities_to_mongo_filters(entities))


def classify_intent_local(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()
    if not q:
        return None

    if is_ask_for_record_id_question(
        q, domain_noun="trips?", has_token=bool(extract_trip_token(q))
    ):
        return {
            "intent": "ask_for_record_id",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "short",
            "max_tokens_hint": 80,
            "retrieve_k": 0,
            "reason": "trip_details_without_token",
        }

    if is_trip_analytics_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_analytics": True,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "trip_analytics",
        }

    lookup_hit = try_lookup_intent(q, history_hint=history_hint)
    if lookup_hit:
        return lookup_hit

    peek = extract_entities(q)
    has_hard_filters = bool(entities_to_mongo_filters(peek))
    wants_recent = bool(
        RECENT_RE.search(q) or re.search(r"\b(recent|recently|latest|newest)\b", q, re.I)
    )

    # Field questions with sticky/session trip token → lookup
    if peek.get("record_token") and peek.get("focus_fields"):
        return {
            "intent": TRIP_LOOKUP_INTENT,
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 700,
            "retrieve_k": 0,
            "record_token": peek["record_token"],
            "order_token": peek["record_token"],
            "reason": "trip_field_lookup",
        }

    if is_list_or_status_question(q) and not is_calculation_question(q):
        if wants_recent and not has_hard_filters:
            return {
                "intent": "list_recent",
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": False,
                "response_style": "medium",
                "max_tokens_hint": 400,
                "retrieve_k": 0,
                "reason": "recent_trips",
            }
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "trip_status_or_list",
        }

    if wants_recent and not is_calculation_question(q):
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "recent_trips",
        }

    detailish = bool(
        re.search(
            r"\b(detail|details|info|information|pickup|delivery|driver|distance|"
            r"commodity|customer|salesman|country|phone|type|status|location|date)\b",
            q,
            re.I,
        )
        and re.search(r"\btrips?\b", q, re.I)
    )
    if detailish and not has_hard_filters and not peek.get("record_token"):
        # Open Q&A / recent context — use recent list so LLM has rows, or RAG
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "trip_details_without_id",
        }

    if (
        LIST_RE.search(q)
        or re.search(
            r"\btrips?\b.*\b(driver|truck|trailer|customer|status|type|country)\b",
            q,
            re.I,
        )
        or re.search(
            r"\b(driver|truck|trailer|customer|status|type|country)\b.*\btrips?\b",
            q,
            re.I,
        )
    ) and not re.search(r"\b(how many|count|total)\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "trip_filtered_list",
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
            "reason": "trip_count",
        }

    return None


def plan_tools(
    intent: str,
    entities: Dict[str, Any],
    intent_info: Dict[str, Any],
) -> List[str]:
    from app.order_ask.tools import (
        TOOL_COUNT,
        TOOL_GET_RECORD,
        TOOL_LIST_RECENT,
        TOOL_RUN_ANALYTICS,
        TOOL_SEARCH,
        TOOL_SEMANTIC_RAG,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    if intent == "ask_for_record_id":
        return []

    if intent == "analytics" or intent_info.get("needs_analytics") or entities.get("analytics"):
        tools.append(TOOL_RUN_ANALYTICS)
        return tools

    if intent == "calculation" or intent_info.get("needs_calculation"):
        tools.append(TOOL_COUNT)

    token = entities.get("record_token") or intent_info.get("record_token")
    if (
        intent in (TRIP_LOOKUP_INTENT, "record_lookup", "order_lookup")
        or intent_info.get("needs_exact_order")
    ):
        if token:
            entities["record_token"] = token
            tools.append(TOOL_GET_RECORD)

    if intent in ("list_filter", "filter") or (
        has_list_filters(entities) and TOOL_GET_RECORD not in tools
    ):
        tools.append(TOOL_SEARCH)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent_info.get("needs_rag") or intent == "open_qa":
        if (
            TOOL_SEARCH not in tools
            and TOOL_GET_RECORD not in tools
            and TOOL_COUNT not in tools
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
)
