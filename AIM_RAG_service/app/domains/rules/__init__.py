"""Per-domain rules: intent, entities, filters, tool planning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domains.rules import invoices, orders, trips

if TYPE_CHECKING:
    from app.domains.rules.base import DomainRules

_RULES = {
    "orders": orders.RULES,
    "invoices": invoices.RULES,
    "trips": trips.RULES,
}


def get_domain_rules(domain: str) -> "DomainRules":
    key = (domain or "orders").lower()
    return _RULES.get(key, orders.RULES)


def list_rule_domains() -> list[str]:
    return list(_RULES.keys())
