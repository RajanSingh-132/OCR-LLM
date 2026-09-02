"""
Bucket 1 — live-Postgres, no-LLM fast path.

Handles the subset of order questions that fn_getorders_regular can
answer directly via its `total_count` (no aggregation needed):

    - counts within a date range (created date only — p_datetype='CD')
    - compare counts between two date ranges
    - counts in the last N days

NOT handled here (see BUCKET1_PLAN.md Bucket 2 / Bucket 3):
    - pickup-date / delivery-date range filters (fn_getorders_regular
      has no p_datetype option for these — needs a backend change)
    - freight totals, daily/weekly breakdowns, top-N groupings (needs
      new aggregate SQL functions — fn_getorders_regular only returns
      a plain count + a page of rows, it cannot SUM/GROUP BY)
    - country filters (pickup/delivery). fn_getorders_regular DECLARES
      p_pickupcountrycode / p_deliverycountrycode but its body ignores
      them entirely (verified against the live function — zero
      references in the body; passing "CA" and passing garbage both
      return the unfiltered total). Answering a country question here
      would return a wrong number that looks right, so those questions
      are handed back to the normal pipeline instead (return None).

If a question doesn't clearly match one of the supported shapes, this
returns None and the caller should fall through to the normal
(entity-extraction -> LLM) pipeline rather than guess.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app import postgres_client
from app.order_ask.checkpoint import checkpoint
from app.order_ask.date_range import (
    detect_last_n_days,
    extract_date_range,
    extract_two_date_ranges,
    last_n_days_range,
)


def _is_count_question(ql: str) -> bool:
    return bool(re.search(r"\b(how many|count|number of|total\s+orders)\b", ql))


def _is_freight_question(ql: str) -> bool:
    # "total freight", "freight total", "revenue" — NOT bare "amount"
    # ("total amount over 1000" is a value filter, not a freight aggregation).
    return bool(re.search(r"\b(freight|revenue)\b", ql))


def _wants_country_filter(ql: str) -> bool:
    """
    Any mention of a country to filter on. fn_getorders_regular cannot
    filter by country (see module docstring), so Bucket 1 must NOT try —
    it hands these back to the normal pipeline (return None) rather than
    return an unfiltered count dressed up as a country count.
    """
    return bool(
        re.search(
            r"\b(canada|canadian|united states|u\.s\.a\.?|usa|america|american)\b",
            ql,
        )
    )


async def _count_for_params(corporate_id: str, params: Dict[str, Any]) -> int:
    """We only need total_count — pagesize=1 keeps the call cheap."""
    call_params = dict(params)
    call_params["p_pagesize"] = 1
    total, _details = await postgres_client.fetch_orders(corporate_id, call_params)
    return total


def _wants_pickup_or_delivery_date(ql: str) -> bool:
    """
    'picked up between X and Y' / 'delivered between X and Y' — these need
    pickup/delivery date filtering, which fn_getorders_regular does NOT
    support (only p_datetype='CD'/'OD' — created/order date). Silently
    applying created-date instead would give a WRONG answer, not just an
    incomplete one, so this must be caught and refused explicitly.
    """
    return bool(
        re.search(r"\bpick(?:ed)?\s*up\b.{0,25}\b(between|from|on)\b", ql)
        or re.search(r"\bdeliver(?:ed|y)?\b.{0,25}\b(between|from|on)\b", ql)
    )


async def try_answer(corporate_id: str, question: str) -> Optional[Dict[str, Any]]:
    """
    Returns {"answer": str, "source": "bucket1_live_pg"} if this question
    matches a supported Bucket 1 shape, a {"source": "bucket1_unsupported_*"}
    refusal for the explicitly-out-of-scope shapes, or None if the question
    doesn't match any Bucket 1 shape at all (caller should fall through to
    the normal pipeline).
    """
    q = question or ""
    ql = q.lower()

    # Pickup/delivery date filtering is a real gap in fn_getorders_regular
    # (Bucket 2) — must refuse explicitly rather than silently filter on
    # created-date instead, which would produce a wrong count.
    if _wants_pickup_or_delivery_date(ql) and _is_count_question(ql):
        checkpoint("BUCKET1", "pickup/delivery date question — not supported", question=q)
        return {
            "answer": (
                "I can't filter by pickup or delivery date yet — the current "
                "database function only supports filtering by created date. "
                "This needs a backend change we're planning next. I can answer "
                "this using created date instead if that works, or tell you "
                "order counts for a date range right now."
            ),
            "source": "bucket1_unsupported_pickup_delivery_date",
        }

    # Freight/total questions are Bucket 3 (aggregation) — explicitly
    # not handled here. Let a clear "not supported yet" message go out
    # rather than silently returning a count when freight was asked for.
    if _is_freight_question(ql) and not _is_count_question(ql):
        checkpoint("BUCKET1", "freight question detected, not yet supported", question=q)
        return {
            "answer": (
                "I can't calculate freight totals yet — that needs a new "
                "aggregation capability we're adding next. I can tell you "
                "order counts for a date range right now if that helps."
            ),
            "source": "bucket1_unsupported_freight",
        }

    if not _is_count_question(ql):
        return None

    # Country filtering is impossible with fn_getorders_regular (it ignores
    # the country params). Don't answer here — hand the whole question,
    # date range and all, to the normal pipeline which does its own country
    # handling. Answering the date part while dropping the country filter
    # would be a silently-wrong count.
    if _wants_country_filter(ql):
        checkpoint(
            "BUCKET1",
            "country filter requested — fn_getorders_regular can't filter by country, deferring to normal pipeline",
            question=q,
        )
        return None

    # Two-range compare: "compare order count between Aug 1-15 and Aug 16-31"
    two_ranges = extract_two_date_ranges(q)
    if two_ranges:
        (s1, e1), (s2, e2) = two_ranges
        checkpoint("BUCKET1", "compare two date ranges", range1=(s1, e1), range2=(s2, e2))
        base_params: Dict[str, Any] = {"p_isdate": True, "p_datetype": "CD"}

        c1 = await _count_for_params(
            corporate_id, {**base_params, "p_fromdate": s1, "p_todate": e1}
        )
        c2 = await _count_for_params(
            corporate_id, {**base_params, "p_fromdate": s2, "p_todate": e2}
        )
        diff = c1 - c2
        trend = "more" if diff > 0 else ("fewer" if diff < 0 else "the same number of")
        answer = (
            f"{s1} to {e1}: {c1} orders. "
            f"{s2} to {e2}: {c2} orders. "
            f"The first period had {abs(diff)} {trend} orders than the second."
        )
        return {"answer": answer, "source": "bucket1_live_pg"}

    # Last N days
    n_days = detect_last_n_days(q)
    if n_days:
        start, end = last_n_days_range(n_days)
        params: Dict[str, Any] = {
            "p_isdate": True,
            "p_datetype": "CD",
            "p_fromdate": start,
            "p_todate": end,
        }
        total = await _count_for_params(corporate_id, params)
        checkpoint("BUCKET1", "last N days count", n_days=n_days, total=total)
        return {
            "answer": f"{total} orders were created in the last {n_days} days ({start} to {end}).",
            "source": "bucket1_live_pg",
        }

    # Single date range
    date_range = extract_date_range(q)
    if date_range:
        start, end = date_range
        params = {
            "p_isdate": True,
            "p_datetype": "CD",
            "p_fromdate": start,
            "p_todate": end,
        }
        total = await _count_for_params(corporate_id, params)
        checkpoint("BUCKET1", "date range count", start=start, end=end, total=total)
        return {
            "answer": f"{total} orders were created between {start} and {end}.",
            "source": "bucket1_live_pg",
        }

    return None
