"""
Intent understanding — common greetings + per-domain rules + Claude fallback.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate

from app.domains.lookup import get_lookup_module
from app.domains.rules import get_domain_rules
from app.domains.rules.prompts import DOMAIN_INTENT_SUFFIX
from app.embedding_client import get_anthropic_llm
from app.order_ask.calculation_engine import is_calculation_question
from app.order_ask.checkpoint import checkpoint
from app.order_ask.prompts import INTENT_CLASSIFY_PROMPT
from app.tenants.context import get_active_domain

logger = logging.getLogger("order_ask.intent")

GREETING_RE = re.compile(
    r"^\s*(hi|hii|hiii|hello|helo|hlo|hey|heyya|yo|hola|namaste|"
    r"good\s*(morning|afternoon|evening|night)|sup|hiya|"
    r"howdy|greetings)\b[\s!?.]*$",
    re.IGNORECASE,
)
THANKS_RE = re.compile(
    r"^\s*(thanks|thank\s*you|thx|ty|ok|okay|cool|great|nice)\b[\s!?.]*$",
    re.IGNORECASE,
)

_LOOKUP_INTENTS = frozenset({"order_lookup", "invoice_lookup", "trip_lookup", "record_lookup"})


def _normalize_intent_result(result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Ensure lookup intents use domain-specific names and token fields."""
    intent = str(result.get("intent") or "").lower()
    lookup = get_lookup_module(domain)

    if intent == "record_lookup":
        result["intent"] = lookup.intent_name

    if result.get("record_token") and not result.get("order_token"):
        result["order_token"] = result["record_token"]
    if result.get("order_token") and not result.get("record_token"):
        result["record_token"] = result["order_token"]

    if result["intent"] in _LOOKUP_INTENTS:
        result["needs_exact_order"] = True

    return result


def classify_intent_common(question: str) -> Optional[Dict[str, Any]]:
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
    return None


def classify_intent_local(
    question: str,
    *,
    history_hint: str = "",
    domain: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    common = classify_intent_common(question)
    if common is not None:
        return common

    active = (domain or get_active_domain()).lower()
    rules = get_domain_rules(active)
    local = rules.classify_intent_local(question, history_hint=history_hint)
    if local is not None:
        return _normalize_intent_result(local, active)
    return None


def classify_intent_with_anthropic(
    question: str,
    *,
    history: str = "(no prior turns)",
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    active = (domain or get_active_domain()).lower()
    lookup = get_lookup_module(active)
    domain_hint = DOMAIN_INTENT_SUFFIX.get(active, "")
    domain_hint = domain_hint.replace("record_lookup", lookup.intent_name)
    checkpoint("INTENT", "Anthropic classify (ambiguous)", domain=active, question=question[:80])

    prompt = INTENT_CLASSIFY_PROMPT + "\n" + domain_hint
    llm = get_anthropic_llm()
    chain = PromptTemplate.from_template(prompt) | llm
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

    max_tokens = {"short": 80, "medium": 280, "detailed": 900}.get(style, 280)
    needs_analytics = bool(data.get("needs_analytics")) or intent == "analytics"
    lookup_intents = _LOOKUP_INTENTS | {lookup.intent_name}

    result = {
        "intent": intent,
        "needs_rag": (bool(data.get("needs_rag")) or intent == "open_qa") and not needs_analytics,
        "needs_calculation": bool(data.get("needs_calculation")) or intent == "calculation",
        "needs_exact_order": bool(data.get("needs_exact_order")) or intent in lookup_intents,
        "needs_analytics": needs_analytics,
        "response_style": style,
        "max_tokens_hint": int(data.get("max_tokens_hint") or max_tokens),
        "retrieve_k": retrieve_k,
        "reason": data.get("reason") or "anthropic_classify",
    }
    if data.get("order_token"):
        result["order_token"] = data["order_token"]
    return _normalize_intent_result(result, active)


def understand_question(
    question: str,
    *,
    history: str = "(no prior turns)",
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    active = (domain or get_active_domain()).lower()
    checkpoint("INTENT", "understand start", domain=active, question=question[:100])

    local = classify_intent_local(question, history_hint=history, domain=active)
    if local is not None:
        checkpoint(
            "INTENT",
            "local fast-path",
            domain=active,
            intent=local.get("intent"),
            reason=local.get("reason"),
        )
        return local

    try:
        result = classify_intent_with_anthropic(question, history=history, domain=active)
        checkpoint(
            "INTENT",
            "anthropic result",
            domain=active,
            intent=result.get("intent"),
            reason=result.get("reason"),
        )
        return result
    except Exception as exc:
        logger.error("Anthropic intent failed: %s", exc, exc_info=True)
        lookup = get_lookup_module(active)
        token = lookup.extract_token(question)
        return {
            "intent": "open_qa",
            "needs_rag": True,
            "needs_calculation": is_calculation_question(question),
            "needs_exact_order": bool(token),
            "response_style": "medium",
            "max_tokens_hint": 280,
            "retrieve_k": 5,
            "reason": f"fallback_after_error:{exc}",
        }
