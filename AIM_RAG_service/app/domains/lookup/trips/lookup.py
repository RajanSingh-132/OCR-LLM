"""Trip lookup — TripID / TripNumber (not generic words)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.domains.lookup.base import GENERIC_LOOKUP_STOPWORDS

INTENT_NAME = "trip_lookup"
DOMAIN = "trips"

LOOKUP_VERB_RE = re.compile(
    r"\b(give|get|show|find|lookup|look\s*up|detail|details|info|information|fetch|pull)\b",
    re.I,
)
LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all|some|any)\b.*\btrips?\b",
    re.I,
)
TRIP_STATUS_RE = re.compile(
    r"\btrips?\b.*\b(status|statuses|summary|breakdown)\b|"
    r"\b(status|statuses|summary|breakdown)\b.*\btrips?\b|"
    r"\b(planned|dispatched|started|stated|in[- ]?transit|enroute|"
    r"delivered|deliverd|rejected)\b.*\btrips?\b|"
    r"\btrips?\b.*\b(planned|dispatched|started|stated|in[- ]?transit|enroute|"
    r"delivered|deliverd|rejected)\b",
    re.I,
)


def _valid_token(raw: str) -> bool:
    if not raw:
        return False
    tok = raw.strip().lower()
    if tok in GENERIC_LOOKUP_STOPWORDS:
        return False
    if tok.isdigit() and len(tok) >= 2:
        return True
    if re.search(r"\d", tok):
        return True
    return False


def extract_token(question: str) -> Optional[str]:
    q = question or ""

    m = re.search(r"\btripnumber\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,30})\b", q, re.I)
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    m = re.search(r"\btripid\s*[:=]?\s*(\d+)\b", q, re.I)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"\btrip\s*(?:no|number|num|#)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
        q,
        re.I,
    )
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    # Bare trip numbers common in Avaal_trip: ETP4455, TRO4725, etc.
    m = re.search(r"\b((?:ETP|TRO|TRIP)[A-Za-z0-9-]{2,20})\b", q, re.I)
    if m and _valid_token(m.group(1)):
        return m.group(1).strip()

    return None


def is_list_or_status_question(question: str) -> bool:
    q = question or ""
    if TRIP_STATUS_RE.search(q):
        return True
    if LIST_RE.search(q):
        return True
    if re.search(
        r"\btrips?\b.*\b(driver|truck|trailer|customer|commodity|salesman|"
        r"pickup|delivery|country|distance|type|planned|dispatched|started|"
        r"in[- ]?transit|delivered|rejected)\b",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"\b(driver|truck|trailer|customer|commodity|salesman|"
        r"pickup|delivery|country|distance|type|planned|dispatched|started|"
        r"in[- ]?transit|delivered|rejected)\b.*\btrips?\b",
        q,
        re.I,
    ):
        return True
    if re.search(r"\b(recent|recently|latest|top)\b.*\btrips?\b", q, re.I):
        return True
    if re.search(r"\btrips?\b.*\b(recent|recently|latest|top)\b", q, re.I):
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
    # Specific trip id/number always wins over generic list phrasing.
    if token and (
        LOOKUP_VERB_RE.search(q)
        or len(q.split()) <= 14
        or re.search(
            r"\b(status|type|pickup|delivery|driver|phone|customer|commodity|"
            r"salesman|distance|country|location|date|detail|details)\b",
            q,
            re.I,
        )
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
            "reason": "trip_lookup",
        }

    if is_list_or_status_question(q):
        return None

    return None
