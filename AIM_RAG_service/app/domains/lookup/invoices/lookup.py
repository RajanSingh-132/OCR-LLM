"""Invoice lookup — InvoiceID / InvoiceNumber (flexible formats)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.domains.lookup.base import GENERIC_LOOKUP_STOPWORDS

INTENT_NAME = "invoice_lookup"
DOMAIN = "invoices"

LOOKUP_VERB_RE = re.compile(
    r"\b(give|get|show|find|lookup|look\s*up|detail|details|info|information|fetch|pull)\b",
    re.I,
)
LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all|some|any|give|get)\b.*\binvoices?\b",
    re.I,
)
INVOICE_STATUS_RE = re.compile(
    r"\binvoices?\b.*\b(status|statuses|summary|breakdown)\b|"
    r"\b(status|statuses|summary|breakdown)\b.*\binvoices?\b",
    re.I,
)


def _valid_token(raw: str) -> bool:
    if not raw:
        return False
    tok = raw.strip().lower()
    if tok in GENERIC_LOOKUP_STOPWORDS:
        return False
    if tok in {
        "status", "paid", "open", "overdue", "baddebt", "details", "detail",
        "freight", "amount", "customer", "company", "invoice", "invoices",
    }:
        return False
    # Must look like an ID (digits required) — never bare words like "invoice".
    if tok.isdigit() and len(tok) >= 3:
        return True
    if re.search(r"\d", tok) and len(tok) >= 2:
        return True
    return False


def extract_token(question: str) -> Optional[str]:
    q = question or ""

    m = re.search(
        r"\binvoicenumber\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
        q,
        re.I,
    )
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    m = re.search(r"\binvoiceid\s*[:=]?\s*(\d+)\b", q, re.I)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"\binvoice\s*(?:no|number|num|#)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
        q,
        re.I,
    )
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    # Common invoice number shapes: MR3932, INO4, AIN12225, UGY10960, INV12
    # Require a digit so "invoice"/"INV" alone never match.
    m = re.search(
        r"\b((?:MR(?!P)|INO|AIN|UGY|INV)[A-Za-z0-9-]*\d[A-Za-z0-9-]*)\b",
        q,
        re.I,
    )
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    m = re.search(r"\b([A-Za-z]{2,5}\d{2,})\b", q)
    if m and _valid_token(m.group(1)):
        tok = m.group(1)
        if not re.match(r"^(MRP|TORD|ETP|TRO|TRIP)", tok, re.I):
            return tok.strip()

    return None


def is_list_or_status_question(question: str) -> bool:
    q = question or ""
    if INVOICE_STATUS_RE.search(q):
        return True
    if LIST_RE.search(q):
        return True
    if re.search(
        r"\binvoices?\b.*\b(paid|open|overdue|bad\s*debt|partially\s*paid|customer|company)\b",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"\b(paid|open|overdue|bad\s*debt|partially\s*paid)\b.*\binvoices?\b",
        q,
        re.I,
    ):
        return True
    if re.search(r"\b(recent|recently|latest|top|some|any)\b.*\binvoices?\b", q, re.I):
        return True
    return False


def try_lookup_intent(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()
    if not q:
        return None

    token = extract_token(q)
    fieldish = bool(
        token
        and re.search(
            r"\b(status|detail|details|freight|paid|outstanding|commodity|due|"
            r"pickup|delivery|customer|amount|pretax|company|exchange|order)\b",
            q,
            re.I,
        )
    )
    if token and (
        LOOKUP_VERB_RE.search(q)
        or len(q.split()) <= 12
        or fieldish
    ):
        return {
            "intent": INTENT_NAME,
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 900,
            "retrieve_k": 0,
            "record_token": token,
            "order_token": token,
            "reason": "invoice_lookup",
        }

    if is_list_or_status_question(q):
        return None

    return None
