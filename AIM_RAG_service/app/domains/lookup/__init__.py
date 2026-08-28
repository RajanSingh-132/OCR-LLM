"""Domain lookup modules and prompts."""

from __future__ import annotations

from app.domains.lookup.base import DomainPrompts, LookupModule
from app.domains.lookup.invoices import lookup as invoice_lookup
from app.domains.lookup.invoices import prompts as invoice_prompts
from app.domains.lookup.orders import lookup as order_lookup
from app.domains.lookup.orders import prompts as order_prompts
from app.domains.lookup.trips import lookup as trip_lookup
from app.domains.lookup.trips import prompts as trip_prompts

_MODULES = {
    "orders": LookupModule(
        domain="orders",
        intent_name=order_lookup.INTENT_NAME,
        extract_token=order_lookup.extract_token,
        is_list_or_status_question=order_lookup.is_list_or_status_question,
        try_lookup_intent=order_lookup.try_lookup_intent,
        compare_token_pattern=order_lookup.COMPARE_TOKEN_RE,
    ),
    "invoices": LookupModule(
        domain="invoices",
        intent_name=invoice_lookup.INTENT_NAME,
        extract_token=invoice_lookup.extract_token,
        is_list_or_status_question=invoice_lookup.is_list_or_status_question,
        try_lookup_intent=invoice_lookup.try_lookup_intent,
    ),
    "trips": LookupModule(
        domain="trips",
        intent_name=trip_lookup.INTENT_NAME,
        extract_token=trip_lookup.extract_token,
        is_list_or_status_question=trip_lookup.is_list_or_status_question,
        try_lookup_intent=trip_lookup.try_lookup_intent,
    ),
}

_PROMPTS = {
    "orders": DomainPrompts(
        conversation=order_prompts.ORDER_CONVERSATION_PROMPT,
        ask=order_prompts.ORDER_ASK_PROMPT,
        greeting=order_prompts.ORDER_GREETING_PROMPT,
        formula=order_prompts.ORDER_FORMULA_PROMPT,
    ),
    "invoices": DomainPrompts(
        conversation=invoice_prompts.INVOICE_CONVERSATION_PROMPT,
        ask=invoice_prompts.INVOICE_ASK_PROMPT,
        greeting=invoice_prompts.INVOICE_GREETING_PROMPT,
    ),
    "trips": DomainPrompts(
        conversation=trip_prompts.TRIP_CONVERSATION_PROMPT,
        ask=trip_prompts.TRIP_ASK_PROMPT,
        greeting=trip_prompts.TRIP_GREETING_PROMPT,
    ),
}


def get_lookup_module(domain: str) -> LookupModule:
    key = (domain or "orders").lower()
    return _MODULES.get(key, _MODULES["orders"])


def get_domain_prompts(domain: str) -> DomainPrompts:
    key = (domain or "orders").lower()
    return _PROMPTS.get(key, _PROMPTS["orders"])


def extract_record_token(domain: str, question: str):
    return get_lookup_module(domain).extract_token(question or "")
