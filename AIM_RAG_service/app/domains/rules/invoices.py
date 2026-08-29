"""Invoice domain rules — Avaal_invoice fields + analytics."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import (
    DomainRules,
    extract_limit,
    merge_session_entities,
    _text_filter,
)
from app.domains.lookup.invoices.lookup import (
    INTENT_NAME as INVOICE_LOOKUP_INTENT,
    extract_token as extract_invoice_token,
    is_list_or_status_question,
    try_lookup_intent,
)
from app.domains.lookup.base import is_ask_for_record_id_question
from app.order_ask.calculation_engine import is_calculation_question
from app.order_ask.invoice_analytics import (
    detect_period_days,
    detect_status_filter,
    is_best_invoice_customer_question,
    is_best_invoice_question,
    is_due_next_week_question,
    is_invoice_analytics_question,
    is_invoices_by_country_question,
    is_period_invoices_question,
    is_status_summary_question,
    is_worst_invoice_question,
)

DOMAIN = "invoices"

FIELD_ID = "InvoiceID"
FIELD_NUMBER = "InvoiceNumber"
FIELD_STATUS = "InvoiceStatus"
FIELD_CUSTOMER = "CustomerName"
FIELD_AMOUNT = "TotalAmount"
FIELD_PRETAX = "PreTaxAmount"
FIELD_CURRENCY = "CurrencyCode"
FIELD_DATE = "InvoiceDate"
FIELD_DUE = "DueDate"
FIELD_COMPANY = "CompanyName"
FIELD_ORDERS = "InvoiceOrderNumbers"
FIELD_FREIGHT = "freightcharges"
FIELD_OTHER = "othercharges"
FIELD_OUTSTANDING = "outstandinamount"
FIELD_EXCHANGE = "ExchangeRate"
FIELD_COMMODITY = "commodityname"
FIELD_PICKUP = "pickuplocation"
FIELD_DELIVERY = "deliverylocation"

STATUS_MAP = {
    "partiallypaid": "PartiallyPaid",
    "partially paid": "PartiallyPaid",
    "partial paid": "PartiallyPaid",
    "baddebt": "BadDebt",
    "bad debt": "BadDebt",
    "overdue": "OverDue",
    "over due": "OverDue",
    "paid": "Paid",
    "open": "Open",
}

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all|some|any|give|get)\b.*\binvoices?\b",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|recently|latest|last\s+\d+|top\s+\d+|some|any)\b.*\binvoices?\b|"
    r"\binvoices?\b.*\b(recent|recently|latest|top)\b",
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
    FIELD_COMPANY,
    "analytics",
    "period_days",
    "sort_by",
    "ascending",
    "limit",
    "focus_fields",
)


def extract_record_token(question: str) -> Optional[str]:
    return extract_invoice_token(question)


def _detect_focus_fields(ql: str) -> List[str]:
    focus: List[str] = []
    checks = [
        (r"\bstatus\b", [FIELD_STATUS]),
        (r"\binvoice\s*id\b|\binvoiceid\b", [FIELD_ID]),
        (r"\bfreight\b", [FIELD_FREIGHT]),
        (r"\bother\s*charges?\b", [FIELD_OTHER]),
        (r"\bpaid\s*amount\b|\bamount\s*paid\b", ["PaidAmount", FIELD_AMOUNT, FIELD_OUTSTANDING]),
        (r"\boutstand", [FIELD_OUTSTANDING]),
        (r"\bcommodit", [FIELD_COMMODITY]),
        (r"\binvoice\s*date\b|\binvoicedate\b", [FIELD_DATE]),
        (r"\bdue\s*date\b|\bduedate\b", [FIELD_DUE]),
        (r"\bpick\s*up\b.*\blocation|\bpickuplocation\b", [FIELD_PICKUP]),
        (r"\bdeliver(?:y|y)?\b.*\blocation|\bdeliverylocation\b", [FIELD_DELIVERY]),
        (r"\bcustomer\b", [FIELD_CUSTOMER, "CustomerCode"]),
        (r"\btotal\s*amount\b|\btotalamount\b", [FIELD_AMOUNT]),
        (r"\bpretax\b|\bpre\s*tax\b", [FIELD_PRETAX]),
        (r"\border\s*number|\binvoiceordernumbers\b", [FIELD_ORDERS, "InvoiceOrderIds"]),
        (r"\bcompany\b", [FIELD_COMPANY, "companycode"]),
        (r"\bexchange\s*rate\b", [FIELD_EXCHANGE]),
        (r"\bcurrency\b", [FIELD_CURRENCY]),
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

    status = detect_status_filter(q)
    if status:
        entities[FIELD_STATUS] = status
    else:
        ql_status = re.sub(r"\b(?:paid\s+amount|amount\s+paid)\b", " ", ql)
        for key, canonical in STATUS_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", ql_status):
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
        r"\b(?:for\s+customer|customer\s*name|customer)\s*(?:is|=|:|#)\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,@]{1,60})",
        q,
        re.I,
    )
    if m:
        name = m.group(1).strip(" .,")
        if name.lower() not in {
            "name", "names", "who", "the", "status", "invoice", "invoices",
        }:
            entities[FIELD_CUSTOMER] = name

    m = re.search(
        r"\b(?:company|company\s*name)\s*(?:is|=|:|#)?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60})",
        q,
        re.I,
    )
    if m:
        company = m.group(1).strip(" .,")
        if company.lower() not in {"name", "wise", "the", "for"}:
            entities[FIELD_COMPANY] = company

    period_days = detect_period_days(q)
    if period_days:
        entities["period_days"] = period_days

    # Analytics markers
    if is_due_next_week_question(q):
        entities["analytics"] = "invoices_due_next_week"
    elif is_worst_invoice_question(q):
        entities["analytics"] = "worst_invoice"
    elif is_best_invoice_question(q):
        entities["analytics"] = "best_invoice"
    elif is_invoices_by_country_question(q):
        entities["analytics"] = "invoices_by_country"
    elif is_best_invoice_customer_question(q):
        if re.search(r"\b(worst|least|lowest|fewest|kam|km|minimum|min)\b", ql):
            entities["analytics"] = "worst_invoice_customer"
        else:
            entities["analytics"] = "best_invoice_customer"
    elif is_period_invoices_question(q):
        entities["analytics"] = "invoices_in_period"
    elif is_status_summary_question(q) or (
        entities.get(FIELD_STATUS)
        and re.search(r"\b(how many|count|kitne|kitna|total)\b", ql)
    ):
        entities["analytics"] = "invoice_status_summary"

    focus = _detect_focus_fields(ql)
    if focus:
        entities["focus_fields"] = focus

    if re.search(r"\b(highest|top|largest|max|most)\b.*\b(amount|total)\b", ql):
        entities["sort_by"] = FIELD_AMOUNT
        entities["ascending"] = False
        entities["limit"] = entities.get("limit") or 5
    elif re.search(r"\b(lowest|smallest|min|least)\b.*\b(amount|total)\b", ql):
        entities["sort_by"] = FIELD_AMOUNT
        entities["ascending"] = True
        entities["limit"] = entities.get("limit") or 5

    if re.search(r"\b(recent|recently|latest|newest)\b", ql):
        entities["sort_by"] = FIELD_ID
        entities["ascending"] = False

    limit = extract_limit(q, default_all=25, some_default=10)
    if limit:
        entities["limit"] = limit
    elif re.search(r"\b(some|any|few)\b", ql):
        entities["limit"] = entities.get("limit") or 10

    if SUM_AMOUNT_RE.search(q) and not entities.get("analytics"):
        entities["needs_sum"] = FIELD_AMOUNT

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
            "amount",
            "customer",
            "due",
            "date",
            "freight",
            "outstanding",
            "paid",
            "commodity",
            "company",
            "exchange",
            "pickup",
            "delivery",
            "order",
        ),
    )
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if entities.get(FIELD_STATUS):
        status = str(entities[FIELD_STATUS])
        # Open / OPEN both
        filters[FIELD_STATUS] = {
            "$regex": f"^{re.escape(status)}$",
            "$options": "i",
        }
    if entities.get(FIELD_CUSTOMER):
        filters.update(_text_filter(FIELD_CUSTOMER, str(entities[FIELD_CUSTOMER])))
    if entities.get(FIELD_CURRENCY):
        filters[FIELD_CURRENCY] = str(entities[FIELD_CURRENCY]).upper()
    if entities.get(FIELD_COMPANY):
        filters.update(_text_filter(FIELD_COMPANY, str(entities[FIELD_COMPANY])))
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
        q, domain_noun="invoices?", has_token=bool(extract_invoice_token(q))
    ):
        return {
            "intent": "ask_for_record_id",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "short",
            "max_tokens_hint": 80,
            "retrieve_k": 0,
            "reason": "invoice_details_without_token",
        }

    if is_invoice_analytics_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_analytics": True,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "invoice_analytics",
        }

    lookup_hit = try_lookup_intent(q, history_hint=history_hint)
    if lookup_hit:
        return lookup_hit

    peek = extract_entities(q)
    if peek.get("record_token") and peek.get("focus_fields"):
        return {
            "intent": INVOICE_LOOKUP_INTENT,
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 900,
            "retrieve_k": 0,
            "record_token": peek["record_token"],
            "order_token": peek["record_token"],
            "reason": "invoice_field_lookup",
        }

    wants_recent = bool(
        RECENT_RE.search(q) or re.search(r"\b(recent|recently|latest|newest|some|any)\b", q, re.I)
    )
    has_filters = bool(entities_to_mongo_filters(peek))

    if is_list_or_status_question(q) and not is_calculation_question(q):
        if wants_recent and not has_filters:
            return {
                "intent": "list_recent",
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": False,
                "response_style": "medium",
                "max_tokens_hint": 400,
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

    if wants_recent and not is_calculation_question(q):
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "recent_invoices",
        }

    if (
        LIST_RE.search(q)
        or re.search(
            r"\binvoices?\b.*\b(paid|open|overdue|bad\s*debt|partially\s*paid|customer|company)\b",
            q,
            re.I,
        )
        or re.search(
            r"\b(paid|open|overdue|bad\s*debt|partially\s*paid)\b.*\binvoices?\b",
            q,
            re.I,
        )
    ) and not re.search(r"\b(how many|count|total|sum|kitne|kitna)\b", q, re.I):
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
        TOOL_RUN_ANALYTICS,
        TOOL_SEARCH,
        TOOL_SEMANTIC_RAG,
        TOOL_SUM_FIELD,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    if intent == "ask_for_record_id":
        return []

    if intent == "analytics" or intent_info.get("needs_analytics") or entities.get("analytics"):
        tools.append(TOOL_RUN_ANALYTICS)
        return tools

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if entities.get("needs_sum"):
            tools.append(TOOL_SUM_FIELD)
        else:
            tools.append(TOOL_COUNT)

    token = entities.get("record_token") or intent_info.get("record_token")
    if (
        intent in (INVOICE_LOOKUP_INTENT, "record_lookup", "order_lookup")
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
