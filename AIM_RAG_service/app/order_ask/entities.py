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

    # Customer name (prefer phrases with "customer" / "for" / "of")
    m = re.search(
        r"\b(?:for\s+customer|customer\s*name|orders?\s+for\s+customer|orders?\s+of\s+customer|"
        r"orders?\s+for|orders?\s+of|customer)\s+['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60}?)['\"]?(?=\s+(?:with|status|in|and|currency|$)|$)",
        q,
        re.I,
    )
    if m:
        name = m.group(1).strip(" .,")
        # If matcher ate leading "customer ", strip it
        name = re.sub(r"^(?:customer|code)\s+", "", name, flags=re.I).strip()
        if name.lower().startswith("code "):
            pass
        elif name.lower() in STATUS_MAP:
            pass
        elif len(name) >= 2:
            entities["customername"] = name

    # Company code
    m = re.search(r"\bcompany(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["companycode"] = m.group(1).upper()

    # Limit / top N
    m = re.search(r"\b(?:top|last|recent|show|list)\s+(\d{1,3})\b", ql)
    if m:
        entities["limit"] = min(50, max(1, int(m.group(1))))
    elif re.search(r"\b(all|every)\b.*\border", ql):
        entities["limit"] = 25

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
                r"\b(status|tax|freight|detail|details|info|amount|customer|delivery|pickup)\b",
                ql,
            ):
                entities["order_token"] = session_order_token
                entities["from_session"] = True

    if follow_up:
        for key in ("customername", "customercode", "orderstatus", "currencycode", "companycode"):
            if key not in entities and sticky.get(key):
                entities[key] = sticky[key]
                entities["from_session"] = True

    checkpoint("ENTITIES", "extracted", **{k: v for k, v in entities.items()})
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Map entities → Mongo filter fields (excluding order_token / limit)."""
    filters: Dict[str, Any] = {}
    if entities.get("orderstatus"):
        filters["orderstatus"] = entities["orderstatus"]
    if entities.get("currencycode"):
        filters["currencycode"] = entities["currencycode"]
    if entities.get("customercode"):
        filters["customercode"] = entities["customercode"]
    if entities.get("companycode"):
        filters["companycode"] = entities["companycode"]
    if entities.get("customername"):
        filters["customername"] = entities["customername"]
    return filters
