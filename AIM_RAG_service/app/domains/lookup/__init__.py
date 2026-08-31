"""Domain lookup modules — answer prompts come from app.System_prompt."""

from __future__ import annotations

from app.domains.lookup.base import DomainPrompts, LookupModule
from app.domains.lookup.invoices import lookup as invoice_lookup
from app.domains.lookup.orders import lookup as order_lookup
from app.domains.lookup.trips import lookup as trip_lookup
from app.System_prompt import get_domain_answer_prompts

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


def get_lookup_module(domain: str) -> LookupModule:
    key = (domain or "orders").lower()
    return _MODULES.get(key, _MODULES["orders"])


def get_domain_prompts(domain: str) -> DomainPrompts:
    """Load conversation/ask/greeting/lookup/formula from System_prompt."""
    key = (domain or "orders").lower()
    p = get_domain_answer_prompts(key)
    return DomainPrompts(
        conversation=p["conversation"],
        ask=p["ask"],
        greeting=p["greeting"],
        formula=p.get("formula") or "",
        lookup=p.get("lookup") or "",
    )


def extract_record_token(domain: str, question: str):
    return get_lookup_module(domain).extract_token(question or "")
