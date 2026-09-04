"""
Deterministic fast-path for simple "how many <records> were created in
<period>" questions — skips the LLM query planner entirely.

Why: the LLM query planner call (schema + question -> JSON plan) is the
single biggest latency chunk of a typical /api/v1/orders/ask request
(observed ~5.3s out of ~7.5s total for a plain "how many orders today"
question). That question needs no reasoning at all — it is a single
Mongo count_documents() with a date range. This module recognizes that
narrow pattern with regex and answers it directly from Mongo, producing
the exact same result shape as query_planner.execute_query_plan() /
tools.execute_tools() so the rest of rag_engine.py (final LLM answer,
memory save, response shape) needs no changes.

SCOPE IS DELIBERATELY NARROW (first optimization step): only a plain
"count of records created in a period" with NO other filter mentioned
(status, city, customer, amount, etc.). Anything else returns None and
the caller falls through to the normal LLM query planner, unchanged.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from app.order_ask.checkpoint import checkpoint
from app.tenants.router import (
    get_domain_collection,
    get_domain_metadata_type,
    get_domain_namespace,
)

# All three ingest pipelines (scripts/ingest/*, app/sync/*_live_api.py) store
# this as an ISO-8601 string (lexically sortable), so plain $gte/$lt string
# comparison is safe — same assumption the rest of the codebase relies on.
DATE_FIELD = "createdon"

_COUNT_WORDS_RE = re.compile(
    r"\b(how\s+many|count\s+of|total(?:\s+number\s+of)?|number\s+of|kitne|kitni)\b",
    re.IGNORECASE,
)

# If the question mentions any of these, it wants more than a plain date
# count (a filter, a metric, a comparison) — bail out to the LLM planner
# rather than risk answering the wrong question.
_HAS_OTHER_FILTERS_RE = re.compile(
    r"\b(status|city|state|country|customer|carrier|driver|truck|trailer|"
    r"salesman|company|currency|equipment|amount|greater|less|more\s+than|"
    r"less\s+than|above|below|between|percent|percentage|average|avg|"
    r"\bsum\b|revenue|freight|compare|versus|\bvs\b)\b",
    re.IGNORECASE,
)

_PERIOD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("today", re.compile(r"\btoday\b|\baaj\b", re.IGNORECASE)),
    ("yesterday", re.compile(r"\byesterday\b", re.IGNORECASE)),
    ("this_week", re.compile(r"\bthis\s+week\b|\bcurrent\s+week\b", re.IGNORECASE)),
    ("this_month", re.compile(r"\bthis\s+month\b|\bcurrent\s+month\b", re.IGNORECASE)),
]

# domain -> (count key name matching each domain's own
# format_*_dynamic_analytics_for_context() convention)
_COUNT_KEY_BY_DOMAIN = {
    "orders": "matching_orders",
    "trips": "matching_trips",
    "invoices": "matching_invoices",
}


def _detect_period(question: str) -> Optional[str]:
    for period, pattern in _PERIOD_PATTERNS:
        if pattern.search(question):
            return period
    return None


def _period_range(period: str, now: datetime) -> Tuple[str, str]:
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "today":
        start, end = today_start, today_start + timedelta(days=1)
    elif period == "yesterday":
        start, end = today_start - timedelta(days=1), today_start
    elif period == "this_week":
        start = today_start - timedelta(days=today_start.weekday())  # Monday
        end = start + timedelta(days=7)
    elif period == "this_month":
        start = today_start.replace(day=1)
        end = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else start.replace(month=start.month + 1)
        )
    else:  # pragma: no cover — _detect_period only returns known periods
        raise ValueError(f"unknown period: {period}")
    return start.isoformat(), end.isoformat()


def _format_context(domain: str, payload: Dict[str, Any]) -> str:
    if domain == "orders":
        from app.order_ask.dynamic_analytics import (
            format_dynamic_analytics_for_context as fmt,
        )
    elif domain == "trips":
        from app.order_ask.trip_dynamic_analytics import (
            format_trip_dynamic_analytics_for_context as fmt,
        )
    else:
        from app.order_ask.invoice_dynamic_analytics import (
            format_invoice_dynamic_analytics_for_context as fmt,
        )
    return fmt(payload)


def try_count_fast_path(question: str, domain: str) -> Optional[Dict[str, Any]]:
    """Return an execute_tools()-shaped result for a simple date-count
    question, or None when this isn't one — caller falls through to the
    normal LLM query planner in that case."""
    if domain not in _COUNT_KEY_BY_DOMAIN:
        return None

    q = (question or "").strip()
    if not q or not _COUNT_WORDS_RE.search(q):
        return None
    if _HAS_OTHER_FILTERS_RE.search(q):
        return None

    period = _detect_period(q)
    if period is None:
        return None

    now = datetime.now(timezone.utc)
    from_iso, to_iso = _period_range(period, now)

    collection = get_domain_collection(domain)
    match = {
        "namespace": get_domain_namespace(domain),
        "metadata.type": get_domain_metadata_type(domain),
        DATE_FIELD: {"$gte": from_iso, "$lt": to_iso},
    }
    count = collection.count_documents(match)

    count_key = _COUNT_KEY_BY_DOMAIN[domain]
    analytics_payload: Dict[str, Any] = {
        "analytics_type": "dynamic",
        "engine": "fast_path",
        "operation": "count",
        "filters": {DATE_FIELD: {"period": period}},
        "question": question,
        count_key: count,
    }
    context_block = _format_context(domain, analytics_payload)

    checkpoint(
        "FAST_PATH",
        "deterministic count — LLM query planner skipped",
        domain=domain,
        period=period,
        count=count,
    )

    return {
        "context_blocks": [context_block],
        "matches": [],
        "calculation": None,
        "analytics": analytics_payload,
        "list_result": None,
        "tools_run": ["run_analytics"],
        "active_order_token": "",
    }
