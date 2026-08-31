"""
Ask AI system prompts — single place for order / invoice / trip LLM instructions.

Usage:
  from app.System_prompt import get_domain_answer_prompts, get_system_prompt
"""

from app.System_prompt.common import (
    AVAAL_GREETING_PROMPT,
    NATURAL_LIST_FORMAT_POLICY,
    NUMBER_REQUEST_POLICY,
)
from app.System_prompt.intent_prompt import (
    FILTER_CONTEXT_ANSWER_POLICY,
    INTENT_CLASSIFY_PROMPT,
)
from app.System_prompt.invoice_prompt import (
    INVOICE_ASK_PROMPT,
    INVOICE_CONVERSATION_PROMPT,
    INVOICE_GREETING_PROMPT,
    INVOICE_LOOKUP_PROMPT,
    INVOICE_SYSTEM_PROMPT,
)
from app.System_prompt.order_prompt import (
    ORDER_ASK_PROMPT,
    ORDER_CONVERSATION_PROMPT,
    ORDER_FORMULA_PROMPT,
    ORDER_GREETING_PROMPT,
    ORDER_LOOKUP_PROMPT,
    ORDER_SYSTEM_PROMPT,
)
from app.System_prompt.trip_prompt import (
    TRIP_ASK_PROMPT,
    TRIP_CONVERSATION_PROMPT,
    TRIP_GREETING_PROMPT,
    TRIP_LOOKUP_PROMPT,
    TRIP_SYSTEM_PROMPT,
)

_SYSTEM_BY_DOMAIN = {
    "orders": ORDER_SYSTEM_PROMPT,
    "invoices": INVOICE_SYSTEM_PROMPT,
    "trips": TRIP_SYSTEM_PROMPT,
}

_ANSWER_BY_DOMAIN = {
    "orders": {
        "system": ORDER_SYSTEM_PROMPT,
        "conversation": ORDER_CONVERSATION_PROMPT,
        "ask": ORDER_ASK_PROMPT,
        "lookup": ORDER_LOOKUP_PROMPT,
        "greeting": ORDER_GREETING_PROMPT,
        "formula": ORDER_FORMULA_PROMPT,
    },
    "invoices": {
        "system": INVOICE_SYSTEM_PROMPT,
        "conversation": INVOICE_CONVERSATION_PROMPT,
        "ask": INVOICE_ASK_PROMPT,
        "lookup": INVOICE_LOOKUP_PROMPT,
        "greeting": INVOICE_GREETING_PROMPT,
        "formula": "",
    },
    "trips": {
        "system": TRIP_SYSTEM_PROMPT,
        "conversation": TRIP_CONVERSATION_PROMPT,
        "ask": TRIP_ASK_PROMPT,
        "lookup": TRIP_LOOKUP_PROMPT,
        "greeting": TRIP_GREETING_PROMPT,
        "formula": "",
    },
}


def get_system_prompt(domain: str = "orders") -> str:
    key = (domain or "orders").lower().strip()
    return _SYSTEM_BY_DOMAIN.get(key, ORDER_SYSTEM_PROMPT)


def get_domain_answer_prompts(domain: str = "orders") -> dict:
    key = (domain or "orders").lower().strip()
    return dict(_ANSWER_BY_DOMAIN.get(key, _ANSWER_BY_DOMAIN["orders"]))


__all__ = [
    "NUMBER_REQUEST_POLICY",
    "NATURAL_LIST_FORMAT_POLICY",
    "AVAAL_GREETING_PROMPT",
    "INTENT_CLASSIFY_PROMPT",
    "FILTER_CONTEXT_ANSWER_POLICY",
    "ORDER_SYSTEM_PROMPT",
    "ORDER_CONVERSATION_PROMPT",
    "ORDER_ASK_PROMPT",
    "ORDER_LOOKUP_PROMPT",
    "ORDER_FORMULA_PROMPT",
    "ORDER_GREETING_PROMPT",
    "INVOICE_SYSTEM_PROMPT",
    "INVOICE_CONVERSATION_PROMPT",
    "INVOICE_ASK_PROMPT",
    "INVOICE_LOOKUP_PROMPT",
    "INVOICE_GREETING_PROMPT",
    "TRIP_SYSTEM_PROMPT",
    "TRIP_CONVERSATION_PROMPT",
    "TRIP_ASK_PROMPT",
    "TRIP_LOOKUP_PROMPT",
    "TRIP_GREETING_PROMPT",
    "get_system_prompt",
    "get_domain_answer_prompts",
]
