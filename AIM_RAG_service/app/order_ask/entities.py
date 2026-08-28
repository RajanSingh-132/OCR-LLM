"""
Entity extraction — dispatches to domain rules (orders / invoices / trips).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.domains.rules import get_domain_rules
from app.order_ask.checkpoint import checkpoint
from app.tenants.context import get_active_domain

# Backward-compatible re-exports for order-specific callers
GEO_FILTER_KEYS = (
    "pin",
    "state",
    "city",
    "address",
    "location",
    "pickup_location",
    "delivery_location",
)


def extract_entities(
    question: str,
    *,
    session_order_token: Optional[str] = None,
    session_entities: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract entities using rules for the active domain."""
    active = (domain or get_active_domain()).lower()
    rules = get_domain_rules(active)
    entities = rules.extract_entities(
        question,
        session_order_token=session_order_token,
        session_entities=session_entities,
    )
    # Unify token key for tools layer
    if entities.get("record_token") and not entities.get("order_token"):
        entities["order_token"] = entities["record_token"]
    checkpoint("ENTITIES", "extracted", domain=active, **{k: v for k, v in entities.items()})
    return entities


def entities_to_mongo_filters(entities: Dict[str, Any], domain: Optional[str] = None) -> Dict[str, Any]:
    rules = get_domain_rules(domain or get_active_domain())
    return rules.entities_to_mongo_filters(entities)


def has_geo_or_list_filters(entities: Dict[str, Any], domain: Optional[str] = None) -> bool:
    rules = get_domain_rules(domain or get_active_domain())
    return rules.has_list_filters(entities)
