"""Shared lookup types for orders / invoices / trips."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

IntentResult = Optional[Dict[str, Any]]

# Words that must never be treated as record IDs after invoice/trip nouns
GENERIC_LOOKUP_STOPWORDS = frozenset(
    {
        "status",
        "details",
        "detail",
        "info",
        "information",
        "summary",
        "breakdown",
        "list",
        "recent",
        "latest",
        "some",
        "all",
        "any",
        "paid",
        "open",
        "overdue",
        "cancelled",
        "canceled",
        "total",
        "count",
        "number",
        "numbers",
        "data",
        "record",
        "records",
    }
)

# Shared policies — canonical source is app.System_prompt.common
from app.System_prompt.common import (  # noqa: F401
    AVAAL_GREETING_PROMPT,
    NATURAL_LIST_FORMAT_POLICY,
    NUMBER_REQUEST_POLICY,
)

LABELED_FIELDS_POLICY = NATURAL_LIST_FORMAT_POLICY


def is_ask_for_record_id_question(
    question: str,
    *,
    domain_noun: str,
    has_token: bool,
) -> bool:
    """
    True when user wants a specific record's details but did not give a number/id.
    domain_noun: regex fragment e.g. 'orders?', 'invoices?', 'trips?'
    """
    if has_token:
        return False
    q = question or ""
    # List / recent / count questions are never "ask for id"
    if re.search(
        r"\b(list|all|some|any|few|recent|recently|latest|how many|count|"
        r"kitne|kitna|total|summary|breakdown|best|worst|filter|"
        r"status\s+wise|create[d]?\s+recently|newly\s+create[d]?|"
        r"only\s+\d+|just\s+\d+|top\s+\d+|last\s+\d+)\b",
        q,
        re.I,
    ):
        return False
    # Explicit count near domain noun ("2 orders", "give me 5 order")
    if re.search(
        rf"\b\d{{1,3}}\s+({domain_noun})\b|\b({domain_noun})\s+\d{{1,3}}\b",
        q,
        re.I,
    ):
        return False
    return bool(
        re.search(
            rf"\b(detail|details|info|information)\b.*\b({domain_noun})\b|"
            rf"\b({domain_noun})\b.*\b(detail|details|info|information|number|id)\b|"
            rf"\b(give|get|show|fetch|pull|lookup|look\s*up)\b.*\b({domain_noun})\b.*"
            rf"\b(detail|details|info|information|number|id)\b|"
            rf"\b({domain_noun})\b.*\b(chahiye|dikhao|batao)\b|"
            rf"\b(order|invoice|trip)\s*(number|id|no\.?)\b",
            q,
            re.I,
        )
    )


@dataclass(frozen=True)
class DomainPrompts:
    """Answer prompts for one domain."""

    conversation: str
    ask: str
    greeting: str
    formula: str = ""
    lookup: str = ""


@dataclass(frozen=True)
class LookupModule:
    domain: str
    intent_name: str
    extract_token: Callable[[str], Optional[str]]
    is_list_or_status_question: Callable[[str], bool]
    try_lookup_intent: Callable[..., IntentResult]
    compare_token_pattern: Any = None
