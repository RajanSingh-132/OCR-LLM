"""Invoice domain rules — fields from registry (Avaal_invoice)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import (
    DomainRules,
    build_calc_result,
    extract_limit,
    merge_session_entities,
    _status_filter,
    _text_filter,
)
from app.domains.lookup.invoices.lookup import (
    INTENT_NAME as INVOICE_LOOKUP_INTENT,
    extract_token as extract_invoice_token,
    is_list_or_status_question,
    try_lookup_intent,
)
from app.order_ask.calculation_engine import is_calculation_question

DOMAIN = "invoices"

# Mongo fields (registry list_fields / sort_fields only)
FIELD_ID = "InvoiceID"
FIELD_NUMBER = "InvoiceNumber"
FIELD_STATUS = "InvoiceStatus"
FIELD_CUSTOMER = "CustomerName"
FIELD_AMOUNT = "TotalAmount"
FIELD_CURRENCY = "CurrencyCode"
FIELD_DATE = "InvoiceDate"
FIELD_DUE = "DueDate"

STATUS_MAP = {
    "paid": "Paid",
    "open": "Open",
    "overdue": "Overdue",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
}

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all)\b.*\binvoices?\b",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|latest|last\s+\d+|top\s+\d+|top)\b.*\binvoices?\b|"
    r"\binvoices?\b.*\b(recent|latest|top)\b",
    re.I,
)
SUM_AMOUNT_RE = re.compile(
    r"\b(total|sum|amount|revenue|billing)\b.*\b(invoices?|amount)\b|"
    r"\binvoices?\b.*\b(total|sum|amount|revenue)\b",
    re.I,
)

STICKY_KEYS = (
    "record_token",
    FIELD_STATUS,
    FIELD_CUSTOMER,
    FIELD_CURRENCY,
    FIELD_DATE,
    FIELD_DUE,
    "sort_by",
    "ascending",
    "limit",
)


def extract_record_token(question: str) -> Optional[str]:
    return extract_invoice_token(question)


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

    for key, canonical in STATUS_MAP.items():
        if re.search(rf"\b{key}\b", ql):
            entities[FIELD_STATUS] = canonical
            break

    m = re.search(r"\bcurrency\s*[:=]?\s*([A-Za-z]{3})\b", q, re.I)
    if m:
        entities[FIELD_CURRENCY] = m.group(1).upper()
    else:
        m = re.search(r"\b(CAD|USD)\b", q)
        if m:
            entities[FIELD_CURRENCY] = m.group(1).upper()

    m = re.search(
        r"\b(?:for\s+customer|customer\s*name|customer|invoices?\s+for)\s+['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60}?)['\"]?(?=\s+(?:with|status|and|$)|$)",
        q,
        re.I,
    )
    if m and not re.search(r"\b(how many|total|count)\b", ql):
        name = m.group(1).strip(" .,")
        if len(name) >= 2:
            entities[FIELD_CUSTOMER] = name

    m = re.search(r"\b(?:due\s*date|duedate)\s*[:=]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", q, re.I)
    if m:
        entities[FIELD_DUE] = m.group(1)
    m = re.search(r"\b(?:invoice\s*date|invoicedate|date)\s*[:=]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", q, re.I)
    if m:
        entities[FIELD_DATE] = m.group(1)

    if re.search(r"\b(highest|top|largest|max|most)\b.*\b(amount|total)\b", ql):
        entities["sort_by"] = FIELD_AMOUNT
        entities["ascending"] = False
        entities["limit"] = entities.get("limit") or 5
    elif re.search(r"\b(lowest|smallest|min|least)\b.*\b(amount|total)\b", ql):
        entities["sort_by"] = FIELD_AMOUNT
        entities["ascending"] = True
        entities["limit"] = entities.get("limit") or 5

    limit = extract_limit(q, default_all=25)
    if limit:
        entities["limit"] = limit

    if SUM_AMOUNT_RE.search(q):
        entities["needs_sum"] = FIELD_AMOUNT

    merge_session_entities(
        entities,
        question=q,
        session_token=session_order_token,
        session_entities=session_entities,
        sticky_keys=STICKY_KEYS,
        token_key="record_token",
        follow_up_token_triggers=("status", "detail", "amount", "customer", "due", "date"),
    )
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if entities.get(FIELD_STATUS):
        filters.update(_status_filter(FIELD_STATUS, str(entities[FIELD_STATUS])))
    if entities.get(FIELD_CUSTOMER):
        filters.update(_text_filter(FIELD_CUSTOMER, str(entities[FIELD_CUSTOMER])))
    if entities.get(FIELD_CURRENCY):
        filters[FIELD_CURRENCY] = str(entities[FIELD_CURRENCY]).upper()
    if entities.get(FIELD_DATE):
        filters[FIELD_DATE] = entities[FIELD_DATE]
    if entities.get(FIELD_DUE):
        filters[FIELD_DUE] = entities[FIELD_DUE]
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
                "reason": "recent_invoices",
            }
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "invoice_status_or_list",
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
            "reason": "recent_invoices",
        }

    if (
        LIST_RE.search(q)
        or re.search(r"\binvoices?\b.*\b(paid|open|overdue|cancelled)\b", q, re.I)
        or re.search(r"\b(paid|open|overdue|cancelled)\b.*\binvoices?\b", q, re.I)
    ) and not re.search(r"\b(how many|count|total|sum)\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "invoice_filtered_list",
        }

    if is_calculation_question(q) or SUM_AMOUNT_RE.search(q):
        return {
            "intent": "calculation",
            "needs_rag": False,
            "needs_calculation": True,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 250,
            "retrieve_k": 0,
            "reason": "invoice_calculation",
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
        TOOL_SUM_FIELD,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if entities.get("needs_sum"):
            tools.append(TOOL_SUM_FIELD)
        else:
            tools.append(TOOL_COUNT)
    elif intent == "analytics" or intent_info.get("needs_analytics"):
        tools.append(TOOL_COUNT)

    token = entities.get("record_token") or intent_info.get("record_token")
    if intent in (INVOICE_LOOKUP_INTENT, "record_lookup", "order_lookup") or intent_info.get("needs_exact_order"):
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


def needs_sum(entities: Dict[str, Any], question: str) -> Optional[str]:
    if entities.get("needs_sum"):
        return str(entities["needs_sum"])
    if SUM_AMOUNT_RE.search(question or ""):
        return FIELD_AMOUNT
    return None


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
