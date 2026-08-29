"""Shared helpers for all domain rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

IntentResult = Optional[Dict[str, Any]]


@dataclass
class DomainRules:
    """Hooks for one domain (orders / invoices / trips)."""

    name: str
    extract_entities: Callable[..., Dict[str, Any]]
    classify_intent_local: Callable[..., IntentResult]
    entities_to_mongo_filters: Callable[[Dict[str, Any]], Dict[str, Any]]
    has_list_filters: Callable[[Dict[str, Any]], bool]
    plan_tools: Callable[..., List[str]]
    extract_record_token: Callable[[str], Optional[str]]
    sticky_entity_keys: Tuple[str, ...] = ()
    compare_token_pattern: re.Pattern = re.compile(r"$^")


def extract_limit(question: str, *, default_all: int = 25, some_default: int = 10) -> Optional[int]:
    """
    Parse how many records the user wants.
    - "give me 20 orders" / "only 2" / "top 5" / "last 3"
    - "some" / "any" / "few" → some_default (10)
    - "all" / "every" → default_all
    """
    ql = (question or "").lower()

    # Explicit count near domain nouns or list verbs
    patterns = (
        r"\b(?:top|last|recent|show|list|get|give|fetch|only|just|bas|kewal)\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s+(?:orders?|invoices?|trips?|records?|rows?|items?|details?)\b",
        r"\b(?:orders?|invoices?|trips?)\s*[:=]?\s*(\d{1,3})\b",
    )
    for pat in patterns:
        m = re.search(pat, ql)
        if m:
            return min(50, max(1, int(m.group(1))))

    if re.search(r"\b(some|any|few|kuch|kai)\b", ql):
        return max(1, min(50, int(some_default)))
    if re.search(r"\b(all|every|saare|saari)\b", ql):
        return default_all
    return None


def is_follow_up(question: str) -> bool:
    return bool(
        re.search(
            r"\b(that|this|same|it|its|uska|uski|uske|wo|woh|previous|again)\b",
            (question or "").lower(),
        )
    )


def merge_session_entities(
    entities: Dict[str, Any],
    *,
    question: str,
    session_token: Optional[str],
    session_entities: Optional[Dict[str, Any]],
    sticky_keys: Tuple[str, ...],
    token_key: str = "record_token",
    follow_up_token_triggers: Tuple[str, ...] = (
        "status",
        "detail",
        "details",
        "info",
        "amount",
        "customer",
        "date",
    ),
) -> Dict[str, Any]:
    """Reuse session token/entities on follow-up questions."""
    ql = (question or "").lower()
    sticky = session_entities or {}
    follow_up = is_follow_up(question)

    if follow_up or (not entities.get(token_key) and session_token):
        if not entities.get(token_key) and session_token:
            if follow_up or any(t in ql for t in follow_up_token_triggers):
                entities[token_key] = session_token
                entities["from_session"] = True

    if follow_up:
        for key in sticky_keys:
            if key not in entities and sticky.get(key):
                entities[key] = sticky[key]
                entities["from_session"] = True

    return entities


def _status_filter(field: str, canonical: str) -> Dict[str, Any]:
    return {field: {"$regex": f"^{re.escape(canonical)}$", "$options": "i"}}


def _text_filter(field: str, value: str) -> Dict[str, Any]:
    return {field: {"$regex": re.escape(value.strip()), "$options": "i"}}


def build_calc_result(
    *,
    domain: str,
    question: str,
    total_count: Optional[int] = None,
    sum_field: Optional[str] = None,
    sum_total: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "domain": domain,
        "question": question,
        "filters": filters or {},
    }
    if total_count is not None:
        payload["total_count"] = total_count
    if sum_field and sum_total is not None:
        payload["sum_field"] = sum_field
        payload["sum_total"] = sum_total
    return payload


def format_domain_calc_context(payload: Dict[str, Any]) -> str:
    domain = payload.get("domain") or "records"
    lines = [f"CALCULATION RESULT ({domain}) — use these numbers only:"]
    if payload.get("total_count") is not None:
        lines.append(f"total_count={payload['total_count']}")
    if payload.get("sum_field") and payload.get("sum_total") is not None:
        lines.append(
            f"sum_{payload['sum_field']}={payload['sum_total']}"
        )
    if payload.get("filters"):
        lines.append(f"filters={payload['filters']}")
    return "\n".join(lines)
