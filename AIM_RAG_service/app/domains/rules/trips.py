"""Trip domain rules — fields from registry (Avaal_trip)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import (
    DomainRules,
    extract_limit,
    merge_session_entities,
    _status_filter,
    _text_filter,
)
from app.domains.lookup.trips.lookup import (
    INTENT_NAME as TRIP_LOOKUP_INTENT,
    extract_token as extract_trip_token,
    is_list_or_status_question,
    try_lookup_intent,
)
from app.order_ask.calculation_engine import is_calculation_question

DOMAIN = "trips"

FIELD_ID = "TripID"
FIELD_NUMBER = "TripNumber"
FIELD_ALT_NUMBER = "tripnumber"
FIELD_DRIVER = "DriverName"
FIELD_TRUCK = "TruckNumber"
FIELD_TRAILER = "TrailerNumber"
FIELD_STATUS = "TripStatus"
FIELD_ALT_STATUS = "status"

STATUS_WORDS = (
    "active",
    "completed",
    "complete",
    "cancelled",
    "canceled",
    "dispatched",
    "in transit",
    "intransit",
    "open",
    "closed",
)

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all)\b.*\btrips?\b",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|latest|last\s+\d+|top\s+\d+|top)\b.*\btrips?\b|"
    r"\btrips?\b.*\b(recent|latest|top)\b",
    re.I,
)

STICKY_KEYS = (
    "record_token",
    FIELD_DRIVER,
    FIELD_TRUCK,
    FIELD_TRAILER,
    FIELD_STATUS,
    FIELD_ALT_STATUS,
    "sort_by",
    "ascending",
    "limit",
)


def extract_record_token(question: str) -> Optional[str]:
    return extract_trip_token(question)


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
            canonical = status.title() if " " not in status else status.title()
            entities[FIELD_STATUS] = canonical
            entities[FIELD_ALT_STATUS] = canonical
            break

    m = re.search(
        r"\b(?:driver|driver\s*name)\s*[:=#]?\s*['\"]?([A-Za-z][A-Za-z0-9 &\-./]{1,40})",
        q,
        re.I,
    )
    if m:
        entities[FIELD_DRIVER] = m.group(1).strip(" .,")

    m = re.search(
        r"\b(?:truck|truck\s*number|truck\s*no)\s*[:=#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./]{1,30})",
        q,
        re.I,
    )
    if m:
        entities[FIELD_TRUCK] = m.group(1).strip(" .,")

    m = re.search(
        r"\b(?:trailer|trailer\s*number|trailer\s*no)\s*[:=#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./]{1,30})",
        q,
        re.I,
    )
    if m:
        entities[FIELD_TRAILER] = m.group(1).strip(" .,")

    if re.search(r"\b(recent|latest|newest)\b", ql):
        entities["sort_by"] = FIELD_ID
        entities["ascending"] = False

    limit = extract_limit(q, default_all=25)
    if limit:
        entities["limit"] = limit

    merge_session_entities(
        entities,
        question=q,
        session_token=session_order_token,
        session_entities=session_entities,
        sticky_keys=STICKY_KEYS,
        token_key="record_token",
        follow_up_token_triggers=("status", "detail", "driver", "truck", "trailer"),
    )
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    status = entities.get(FIELD_STATUS) or entities.get(FIELD_ALT_STATUS)
    if status:
        filters["$or"] = [
            _status_filter(FIELD_STATUS, str(status)),
            _status_filter(FIELD_ALT_STATUS, str(status)),
        ]
    if entities.get(FIELD_DRIVER):
        filters.update(_text_filter(FIELD_DRIVER, str(entities[FIELD_DRIVER])))
    if entities.get(FIELD_TRUCK):
        filters.update(_text_filter(FIELD_TRUCK, str(entities[FIELD_TRUCK])))
    if entities.get(FIELD_TRAILER):
        filters.update(_text_filter(FIELD_TRAILER, str(entities[FIELD_TRAILER])))
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

    lookup_hit = try_lookup_intent(q, history_hint=history_hint)
    if lookup_hit:
        return lookup_hit

    if is_list_or_status_question(q) and not is_calculation_question(q):
        if RECENT_RE.search(q):
            return {
                "intent": "list_recent",
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": False,
                "response_style": "medium",
                "max_tokens_hint": 350,
                "retrieve_k": 0,
                "reason": "recent_trips",
            }
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "trip_status_or_list",
        }

    if RECENT_RE.search(q) and not is_calculation_question(q):
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "recent_trips",
        }

    if (
        LIST_RE.search(q)
        or re.search(r"\btrips?\b.*\b(driver|truck|trailer)\b", q, re.I)
        or re.search(r"\b(driver|truck|trailer)\b.*\btrips?\b", q, re.I)
    ) and not re.search(r"\b(how many|count|total)\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
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
        TOOL_SEARCH,
        TOOL_SEMANTIC_RAG,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    if intent == "calculation" or intent_info.get("needs_calculation"):
        tools.append(TOOL_COUNT)
    elif intent == "analytics" or intent_info.get("needs_analytics"):
        tools.append(TOOL_COUNT)

    token = entities.get("record_token") or intent_info.get("record_token")
    if intent in (TRIP_LOOKUP_INTENT, "record_lookup", "order_lookup") or intent_info.get("needs_exact_order"):
        if token:
            entities["record_token"] = token
            tools.append(TOOL_GET_RECORD)

    if intent in ("list_filter", "filter") or has_list_filters(entities):
        tools.append(TOOL_SEARCH)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent_info.get("needs_rag") or intent == "open_qa":
        if TOOL_SEARCH not in tools and TOOL_GET_RECORD not in tools and TOOL_COUNT not in tools:
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
