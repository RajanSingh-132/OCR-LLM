"""Order lookup — MRP/TORD / orderid / ordernumber."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.order_ask.rag_retrieval import extract_order_token

INTENT_NAME = "order_lookup"
DOMAIN = "orders"

LOOKUP_VERB_RE = re.compile(
    r"\b(give|get|show|find|lookup|look\s*up|detail|details|info|information|fetch|pull)\b",
    re.I,
)
COMPARE_RE = re.compile(r"\b(compare|difference|vs\.?|versus)\b", re.I)
COMPARE_TOKEN_RE = re.compile(r"\b(MRP\d+|TORD\d+|\d{4,})\b", re.I)


def extract_token(question: str) -> Optional[str]:
    return extract_order_token(question or "")


def is_list_or_status_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\bstatus\b.*\b(summary|break|count|how many|breakdown)\b", q):
        return True
    if re.search(r"\b(summary|breakdown)\b.*\bstatus\b", q):
        return True
    if re.search(r"\b(list|show|display|all|some)\b.*\borders?\b", q):
        return True
    if re.search(r"\borders?\b.*\b(list|show|status|recent|confirmed|quoted|delivered)\b", q):
        if not extract_token(question or ""):
            return True
    return False


def try_lookup_intent(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()
    if not q or is_list_or_status_question(q):
        return None

    token = extract_token(q)
    geo_filter_q = bool(
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal|"
            r"state|province|city|town|address|street)\b",
            q,
            re.I,
        )
    )

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

    if (
        token
        and not geo_filter_q
        and (
            LOOKUP_VERB_RE.search(q)
            or re.match(r"^(MRP\d+|TORD\d+|\d{4,})\s*$", q, re.I)
            or len(q.split()) <= 8
        )
    ):
        if not re.search(r"\b(all|list|filter|compare|vs)\b", q, re.I) or LOOKUP_VERB_RE.search(q):
            return {
                "intent": INTENT_NAME,
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": True,
                "response_style": "detailed",
                "max_tokens_hint": 1200,
                "retrieve_k": 0,
                "order_token": token,
                "record_token": token,
                "reason": "order_lookup",
            }

    if token and LOOKUP_VERB_RE.search(q):
        return {
            "intent": INTENT_NAME,
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "record_token": token,
            "reason": "explicit_order_lookup",
        }

    if token and len(q.split()) <= 4:
        return {
            "intent": INTENT_NAME,
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "record_token": token,
            "reason": "short_order_token",
        }

    if history_hint and re.search(
        r"\b(that|this|same|it|its|uska|uski|uske|wo|woh|status|tax|freight)\b",
        q,
        re.I,
    ):
        if re.search(r"\b(status|tax|freight|detail|amount|customer|delivery)\b", q, re.I):
            return {
                "intent": INTENT_NAME,
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": True,
                "response_style": "medium",
                "max_tokens_hint": 400,
                "retrieve_k": 0,
                "reason": "follow_up_order_lookup",
            }

    return None
