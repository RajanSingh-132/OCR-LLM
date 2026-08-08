"""
Fast intent understanding for /api/v1/orders/ask.

Anthropic is used to understand ambiguous questions.
Obvious greetings / calc / order-id / list intents are resolved locally for speed.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_anthropic_llm
from app.order_ask.calculation_engine import is_calculation_question
from app.order_ask.checkpoint import checkpoint
from app.order_ask.prompts import INTENT_CLASSIFY_PROMPT
from app.order_ask.rag_retrieval import extract_order_token

logger = logging.getLogger("order_ask.intent")

GREETING_RE = re.compile(
    r"^\s*(hi|hii|hiii|hello|hey|heyya|yo|hola|namaste|good\s*(morning|afternoon|evening)|sup|hiya)\b[\s!?.]*$",
    re.IGNORECASE,
)
THANKS_RE = re.compile(
    r"^\s*(thanks|thank\s*you|thx|ty|ok|okay|cool|great|nice)\b[\s!?.]*$",
    re.IGNORECASE,
)
LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all)\b.*\border",
    re.IGNORECASE,
)
RECENT_RE = re.compile(
    r"\b(recent|latest|last\s+\d+|top\s+\d+)\b.*\border|\border.*\b(recent|latest)\b",
    re.IGNORECASE,
)
COMPARE_RE = re.compile(
    r"\b(compare|difference|vs\.?|versus)\b",
    re.IGNORECASE,
)


def classify_intent_local(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    """Return intent dict without LLM when pattern is obvious. Else None."""
    q = (question or "").strip()
    if not q:
        return {
            "intent": "empty",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "short",
            "max_tokens_hint": 60,
            "retrieve_k": 0,
        }

    if GREETING_RE.match(q) or THANKS_RE.match(q):
        return {
            "intent": "greeting",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "short",
            "max_tokens_hint": 80,
            "retrieve_k": 0,
            "reason": "greeting_or_thanks",
        }

    # Date-based customer/order counts before order-token heuristics
    # (years like 2026 inside dates must not become order ids)
    from app.order_ask.analytics import (
        is_analytics_question,
        is_best_customer_question,
        is_country_customer_question,
        is_date_activity_question,
        is_status_summary_question,
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
    # Pin/zip/state/city/address filters are NOT order lookups
    geo_filter_q = bool(
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal|"
            r"state|province|city|town|address|street)\b",
            q,
            re.I,
        )
    )
    # Any explicit MRP / TORD / order number request → full order details
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
        # Prefer lookup over list when a specific token is present (unless clear multi-list)
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

    # Best / highest amount / top by freight etc. (orders, not customers)
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

    # List/show orders by status (NOT counts) — before status-summary analytics
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

    # Analytics: best/worst/low customer / status summary / customers by country
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

    # Geo / address / pin / state / city / customer / location filters → list from full DB
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
        or re.search(
            r"\borders?\b.*\b(in|from|to|at)\b\s+[A-Za-z]{2,}",
            q,
            re.I,
        )
    ) and not re.search(
        r"\b(how many\s+customers?|customer\s+count|best|worst|low)\b", q, re.I
    ):
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

    # Accurate filtered lists: status / customer / currency / date / location + list words
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
        if token2 and re.search(r"\b(detail|details|info|information)\b", q, re.I) and not LIST_RE.search(q):
            pass
        else:
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

    # (token already handled above)
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

    # Follow-up like "uska status?" with history → order_lookup
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


def classify_intent_with_anthropic(
    question: str,
    *,
    history: str = "(no prior turns)",
) -> Dict[str, Any]:
    """Ask Anthropic to classify intent for ambiguous questions."""
    checkpoint("INTENT", "Anthropic classify (ambiguous)", question=question[:80])
    llm = get_anthropic_llm()
    chain = PromptTemplate.from_template(INTENT_CLASSIFY_PROMPT) | llm
    raw = chain.invoke({"question": question, "history": history})
    text = raw.content if hasattr(raw, "content") else str(raw)
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Intent JSON parse failed; defaulting to open_qa. raw=%s", text[:300])
        data = {}

    intent = str(data.get("intent") or "open_qa").lower()
    style = str(data.get("response_style") or "medium").lower()
    if style not in ("short", "medium", "detailed"):
        style = "medium"

    retrieve_k = int(data.get("retrieve_k") or 0)
    if intent == "open_qa" and retrieve_k <= 0:
        retrieve_k = 5
    if intent in (
        "greeting",
        "chitchat",
        "thanks",
        "list_filter",
        "list_recent",
        "calculation",
        "analytics",
    ):
        retrieve_k = 0

    max_tokens = {
        "short": 80,
        "medium": 280,
        "detailed": 900,
    }.get(style, 280)

    needs_analytics = bool(data.get("needs_analytics")) or intent == "analytics"

    return {
        "intent": intent,
        "needs_rag": (bool(data.get("needs_rag")) or intent == "open_qa") and not needs_analytics,
        "needs_calculation": bool(data.get("needs_calculation")) or intent == "calculation",
        "needs_exact_order": bool(data.get("needs_exact_order")) or intent == "order_lookup",
        "needs_analytics": needs_analytics,
        "response_style": style,
        "max_tokens_hint": int(data.get("max_tokens_hint") or max_tokens),
        "retrieve_k": retrieve_k,
        "reason": data.get("reason") or "anthropic_classify",
    }


def understand_question(
    question: str,
    *,
    history: str = "(no prior turns)",
) -> Dict[str, Any]:
    """
    Understand user question first.
    Local fast path for clear intents; Anthropic for ambiguous ones.
    """
    checkpoint("INTENT", "understand start", question=question[:100])
    local = classify_intent_local(question, history_hint=history)
    if local is not None:
        checkpoint(
            "INTENT",
            "local fast-path",
            intent=local.get("intent"),
            reason=local.get("reason"),
            style=local.get("response_style"),
        )
        return local
    try:
        result = classify_intent_with_anthropic(question, history=history)
        checkpoint(
            "INTENT",
            "anthropic result",
            intent=result.get("intent"),
            reason=result.get("reason"),
            style=result.get("response_style"),
        )
        return result
    except Exception as exc:
        logger.error("Anthropic intent failed: %s", exc, exc_info=True)
        checkpoint("INTENT", "fallback after error", error=str(exc))
        return {
            "intent": "open_qa",
            "needs_rag": True,
            "needs_calculation": is_calculation_question(question),
            "needs_exact_order": bool(extract_order_token(question)),
            "response_style": "medium",
            "max_tokens_hint": 280,
            "retrieve_k": 5,
            "reason": f"fallback_after_error:{exc}",
        }
