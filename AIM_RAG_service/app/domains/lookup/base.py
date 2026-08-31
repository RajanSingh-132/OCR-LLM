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

# Natural user-facing lists/details — no OrderNumber:/Status: style labels.
NATURAL_LIST_FORMAT_POLICY = """
RESPONSE FORMAT (strict — lists and details):
- Do NOT use field labels like OrderNumber:, Status:, CustomerName:, InvoiceNumber:, TripNumber:.
- Do NOT dump raw Mongo style (ordernumber=… | orderstatus=…).
- Write a short friendly intro, then a clean numbered list (one item per line).
- Each line: natural values only, e.g. "1. ORO21 — Quoted — Customer Name — CAD 1200".
- Use en-dash or commas between values; keep it readable for humans.
- Mention how many you are showing vs total_matching when the context has it.
- Include only useful fields from context (number, status, customer, amount/currency).
""".strip()

# Backward-compatible alias (older imports).
LABELED_FIELDS_POLICY = NATURAL_LIST_FORMAT_POLICY

# Shared greeting / thanks / chitchat — short prompt, no DB, LLM varies wording.
AVAAL_GREETING_PROMPT = """
You are Avaal AI assistant.

User message: {question}

Task: Reply to a greeting, thanks, or light chitchat ONLY.
- 2 to 4 short friendly sentences. Plain text. No markdown. No bullet dump of APIs.
- Introduce yourself as Avaal AI assistant (never OrderBot, ChatGPT, or Claude).
- Clearly offer help with these areas (mention most of them, weave naturally — do not
  always use the same sentence order or exact same wording):
  orders, trips, invoices, driver availability, maintenance plans.
- You may also briefly mention related help like order/trip status, lists, or lookups.
- Vary the phrasing every time (different greeting, different order of topics).
- Do NOT invent order/invoice/trip numbers or any business data.
- Do NOT ask for database details. Do NOT mention MongoDB, tools, or embeddings.
- If the user said thanks/ok: acknowledge warmly, then still offer the same kinds of help.

Write the reply now.
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
