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

# Injected into every domain answer prompt — never leak ID format examples to users.
NUMBER_REQUEST_POLICY = """
NUMBER / ID REQUEST (strict — user-facing):
- NEVER show format examples or prefixes. Do NOT say MRP, TORD, ETP, TRO, AIN, MR,
  "usually starts with", sample numbers like ####, or any example ID pattern.
- When you need an identifier, ask in ONE short sweet sentence only:
  - Orders: "Please provide the order number or order id and I’ll look it up for you."
  - Invoices: "Please provide the invoice number or invoice id and I’ll look it up for you."
  - Trips: "Please provide the trip number or trip id and I’ll look it up for you."
- Do not list options, prefixes, or “for example”.
- Not found: sweetly say it was not found, then ask again for the correct number/id
  the same way (still no examples).
- When the user later sends only a number/id, treat it as that lookup and answer from context.
""".strip()

# Every list/detail row must show Key: value so users know what each value means.
LABELED_FIELDS_POLICY = """
LABELED FIELDS (strict — lists and details):
- NEVER write bare values chained with dashes only (bad: "MR4067 - Open - CAD 260").
- ALWAYS write clear labels before each value (good:
  "InvoiceNumber: MR4067, CustomerName: …, Status: Open, Currency: CAD, Amount: 260, DueDate: …").
- Use readable labels matching the fields (InvoiceNumber, Status, CustomerName, Amount,
  OrderNumber, TripNumber, Driver, etc.).
- One list item per line; keep labels short but present on every field shown.
""".strip()


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
    if re.search(
        r"\b(list|all|some|any|recent|latest|how many|count|kitne|kitna|total|"
        r"summary|breakdown|best|worst|filter|status\s+wise)\b",
        q,
        re.I,
    ):
        return False
    return bool(
        re.search(
            rf"\b(detail|details|info|information)\b.*\b({domain_noun})\b|"
            rf"\b({domain_noun})\b.*\b(detail|details|info|information|number|id)\b|"
            rf"\b(give|get|show|fetch|pull|lookup|look\s*up)\b.*\b({domain_noun})\b|"
            rf"\b({domain_noun})\b.*\b(chahiye|do|dikhao|batao|chahiye)\b|"
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


@dataclass(frozen=True)
class LookupModule:
    domain: str
    intent_name: str
    extract_token: Callable[[str], Optional[str]]
    is_list_or_status_question: Callable[[str], bool]
    try_lookup_intent: Callable[..., IntentResult]
    compare_token_pattern: Any = None
