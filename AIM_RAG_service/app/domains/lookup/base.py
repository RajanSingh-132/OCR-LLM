"""Shared lookup types for orders / invoices / trips."""

from __future__ import annotations

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


@dataclass(frozen=True)
class DomainPrompts:
    """Answer prompts for one domain."""

    conversation: str
    ask: str
    greeting: str
    formula: str = ""


@dataclass(frozen=True)
class LookupModule:
    domain: str
    intent_name: str
    extract_token: Callable[[str], Optional[str]]
    is_list_or_status_question: Callable[[str], bool]
    try_lookup_intent: Callable[..., IntentResult]
    compare_token_pattern: Any = None
