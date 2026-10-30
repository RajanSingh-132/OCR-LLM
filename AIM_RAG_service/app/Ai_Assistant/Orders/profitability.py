"""
Order profitability — by lane (origin city -> destination city), origin,
destination or customer. Deterministic (no LLM): rag_engine routes every
profit / margin / lane / "top revenue customers" question here directly.

Business rule (agreed 2026-10-30):
    Revenue      = grosstotalfreight                      (what the customer pays)
    Freight Cost = carrier cost = outorderamount           (what we pay the
                   outsourced carrier, in outcurrency)
    Profit       = Revenue - Freight Cost
    Margin %     = Profit / Revenue * 100

totalfreight / freightcharges are billed TO the customer (they equal revenue on
most orders), so they are never used as the cost.

Rules that keep the numbers honest:
- Amounts are never added across currencies: every row and total is per
  currencycode.
- Profit / margin are computed only over orders that HAVE a recorded carrier
  cost in the order's own currency. Orders without one are counted as
  "cost not recorded" (never shown as 100% margin); a carrier cost in a
  different currency is not converted and is counted as a currency mismatch.
- Revenue / freight / cost are summed first, then profit / margin computed
  from the totals — never averaged from per-order margins.
- A period in the question is applied in the query; if it can't be resolved
  the payload says so and the answer must say the figures cover all dates.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.Ai_Assistant.config import checkpoint
from app.Ai_Assistant.guards import period_bounds, resolve_period, time_phrase
from app.Ai_Assistant.Orders.order_analytics import (
    _base_match,
    _detect_customer_name,
    is_profitability_question,
)
from app.Ai_Assistant.Orders.order_dynamic_analytics import AGG_TIMEOUT_MS
from app.tenants.router import get_orders_collection

REVENUE_FIELD = "grosstotalfreight"
COST_FIELD = "outorderamount"
COST_CURRENCY_FIELD = "outcurrency"
CURRENCY_FIELD = "currencycode"
DATE_FIELD = "orderdate"

_LANE_RE = re.compile(
    r"\b(lanes?|routes?|corridors?|o-?d\s+pairs?|origin[- ]?(?:wise|and|to|&|/)|"
    r"destination[- ]wise|(?:origin|pickup)\s+(?:city\s+)?(?:and|to|&|/|-)\s+"
    r"(?:destination|delivery|drop))",
    re.I,
)
_MONEY_INTENT_RE = re.compile(
    r"\b(profit\w*|margins?|revenue|earn\w*|loss\w*|losing|costs?|costly|freight|"
    r"lucrative|perform\w*|focus|money|value|best|worst|top|rank\w*|highest|lowest|"
    r"most|least)\b",
    re.I,
)
# Delay / status questions about routes belong to the order planner, not here.
_NOT_MONEY_RE = re.compile(r"\b(delay\w*|late|overdue|on[- ]time|status|pending)\b", re.I)


def is_lane_question(question: str) -> bool:
    q = question or ""
    return bool(
        _LANE_RE.search(q) and _MONEY_INTENT_RE.search(q) and not _NOT_MONEY_RE.search(q)
    )


def is_profitability_or_lane_question(question: str) -> bool:
    return is_profitability_question(question) or is_lane_question(question)


# ------------------------------------------------------------------ parsing
def _dimensions(q: str) -> List[str]:
    origin = re.search(r"\borigin[- ]?wise\b|\bby\s+origin\b|\borigin\s+cit", q, re.I)
    dest = re.search(r"\bdestination[- ]?wise\b|\bby\s+destination\b|\bdestination\s+cit", q, re.I)
    if origin and dest and not re.search(r"\bcombination|\bpairs?\b|\bto\b", q, re.I):
        return ["origin", "destination"]
    if _LANE_RE.search(q):
        return ["lane"]
    return ["customer"]


def _metric(q: str) -> str:
    ql = q.lower()
    if re.search(r"\bcosts?\b|\bcostly\b|\bexpens", ql) and not re.search(r"\bprofit", ql):
        return "cost"
    if re.search(r"\bmargins?\b", ql) and not re.search(r"\bprofit\b", ql):
        return "margin"
    if re.search(r"\brevenue\b|\bsales\b|\bearn", ql) and not re.search(
        r"\bprofit|\bmargin", ql
    ):
        return "revenue"
    return "profit"


def _losing(q: str) -> bool:
    return bool(
        re.search(
            r"\blos(?:e|es|ing|s)\b[^.?!]{0,20}\bmoney\b|\bloss(?:es)?\b|"
            r"\bnegative\s+(?:profit|margin)\b|\bloss[- ]making\b",
            q,
            re.I,
        )
    )


def _direction(q: str, metric: str) -> str:
    if re.search(r"\b(worst|lowest|least|bottom|smallest|minimum|cheapest)\b", q, re.I):
        return "asc"
    return "desc"


def _limit(q: str, default: int) -> int:
    m = re.search(r"\b(?:top|bottom|first|best|worst)\s+(\d{1,3})\b", q, re.I)
    if m:
        return max(1, min(int(m.group(1)), 100))
    if re.search(r"\b(compare|between|each|every|all|wise)\b", q, re.I):
        return 50
    return default


_NAMED_PERIOD_PATTERNS = (
    (r"\b(?:this|current)\s+month\b", "this_month", "this month"),
    (r"\b(?:last|previous|prior)\s+month\b", "last_month", "last month"),
    (r"\b(?:this|current)\s+quarter\b", "this_quarter", "this quarter"),
    (r"\b(?:last|previous|prior)\s+quarter\b", "last_quarter", "last quarter"),
    (r"\b(?:this|current)\s+week\b", "this_week", "this week"),
    (r"\b(?:last|previous|prior)\s+week\b", "last_week", "last week"),
    (r"\b(?:this|current)\s+year\b", "this_year", "this year"),
    (r"\b(?:last|previous|prior)\s+year\b", "last_year", "last year"),
)


def _periods(q: str) -> List[Optional[Dict[str, str]]]:
    """One period, two for "this month vs last month", or [None] for all dates."""
    found = []
    for pattern, name, label in _NAMED_PERIOD_PATTERNS:
        m = re.search(pattern, q, re.I)
        if m:
            s, e = period_bounds(name)  # type: ignore[misc]
            found.append((m.start(), {"start": s, "end": e, "label": label}))
    if len(found) >= 2:
        return [p for _, p in sorted(found, key=lambda x: x[0])][:2]
    one = resolve_period(q)
    return [one] if one else [None]


# ------------------------------------------------------------------ query
def _num(field: str) -> Dict[str, Any]:
    return {"$convert": {"input": f"${field}", "to": "double", "onError": None, "onNull": None}}


def _city(city_field: str, state_field: str) -> Dict[str, Any]:
    city = {"$trim": {"input": {"$toUpper": {"$ifNull": [f"${city_field}", ""]}}}}
    state = {"$trim": {"input": {"$ifNull": [f"${state_field}", ""]}}}
    return {
        "$cond": [
            {"$eq": [city, ""]},
            "UNKNOWN",
            {"$cond": [{"$eq": [state, ""]}, city, {"$concat": [city, ", ", state]}]},
        ]
    }


def _dimension_key(dim: str) -> Dict[str, Any]:
    origin = _city("pickupcity", "pickupstatecode")
    dest = _city("deliverycity", "deliverystatecode")
    if dim == "lane":
        return {"$concat": [origin, " -> ", dest]}
    if dim == "origin":
        return origin
    if dim == "destination":
        return dest
    return {"$trim": {"input": {"$ifNull": ["$customername", "UNKNOWN"]}}}


def _aggregate(match: Dict[str, Any], dim: str) -> List[Dict[str, Any]]:
    cur = {"$toUpper": {"$trim": {"input": {"$ifNull": [f"${CURRENCY_FIELD}", ""]}}}}
    cost_cur = {"$toUpper": {"$trim": {"input": {"$ifNull": [f"${COST_CURRENCY_FIELD}", ""]}}}}
    pipeline = [
        {"$match": match},
        {
            "$addFields": {
                "__rev": _num(REVENUE_FIELD),
                "__cost": {"$ifNull": [_num(COST_FIELD), 0]},
                "__cur": {"$cond": [{"$eq": [cur, ""]}, "UNKNOWN", cur]},
                "__ccur": cost_cur,
                "__key": _dimension_key(dim),
            }
        },
        {"$match": {"__rev": {"$ne": None}}},
        {
            "$addFields": {
                "__has_cost": {
                    "$and": [
                        {"$gt": ["$__cost", 0]},
                        {"$in": ["$__ccur", ["", "$__cur"]]},
                    ]
                },
                "__mismatch": {
                    "$and": [
                        {"$gt": ["$__cost", 0]},
                        {"$not": [{"$in": ["$__ccur", ["", "$__cur"]]}]},
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": {"key": "$__key", "currency": "$__cur"},
                "orders": {"$sum": 1},
                "revenue": {"$sum": "$__rev"},
                "orders_with_cost": {"$sum": {"$cond": ["$__has_cost", 1, 0]}},
                "costed_revenue": {"$sum": {"$cond": ["$__has_cost", "$__rev", 0]}},
                "freight_cost": {"$sum": {"$cond": ["$__has_cost", "$__cost", 0]}},
                "cost_currency_mismatch": {"$sum": {"$cond": ["$__mismatch", 1, 0]}},
            }
        },
    ]
    return list(get_orders_collection().aggregate(pipeline, maxTimeMS=AGG_TIMEOUT_MS))


def _row(r: Dict[str, Any], dim: str) -> Dict[str, Any]:
    with_cost = int(r["orders_with_cost"])
    costed_rev = float(r["costed_revenue"])
    cost = float(r["freight_cost"])
    profit = (costed_rev - cost) if with_cost else None
    margin = (profit / costed_rev * 100.0) if (profit is not None and costed_rev) else None
    return {
        dim: r["_id"]["key"],
        "currency": r["_id"]["currency"],
        "orders": int(r["orders"]),
        "revenue": round(float(r["revenue"]), 2),
        "orders_with_carrier_cost": with_cost,
        "revenue_with_cost": round(costed_rev, 2),
        "freight_cost": round(cost, 2),
        "profit": round(profit, 2) if profit is not None else None,
        "margin_pct": round(margin, 2) if margin is not None else None,
        "orders_cost_not_recorded": int(r["orders"]) - with_cost - int(r["cost_currency_mismatch"]),
        "orders_cost_other_currency": int(r["cost_currency_mismatch"]),
    }


def _rank(rows: List[Dict[str, Any]], metric: str, direction: str, losing: bool):
    key = {"profit": "profit", "margin": "margin_pct", "revenue": "revenue", "cost": "freight_cost"}[metric]
    if losing:
        rows = [r for r in rows if r["profit"] is not None and r["profit"] < 0]
        key, direction = "profit", "asc"
    known = [r for r in rows if r[key] is not None]
    unknown = [r for r in rows if r[key] is None]
    known.sort(key=lambda r: (r[key], r["revenue"]), reverse=(direction == "desc"))
    # rows whose profit can't be computed go last, never ranked as top / worst
    return known + sorted(unknown, key=lambda r: r["revenue"], reverse=True)


def profitability(question: str, *, entities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entities = entities or {}
    q = question or ""
    dims = _dimensions(q)
    metric = _metric(q)
    losing = _losing(q)
    direction = _direction(q, metric)
    limit = _limit(q, 10)

    base = dict(_base_match())
    customer = None
    if dims == ["customer"]:
        customer = entities.get("customername") or _detect_customer_name(q)
        if customer:
            base["customername"] = {"$regex": re.escape(str(customer)), "$options": "i"}

    periods = _periods(q)
    notes: List[str] = []
    if periods == [None] and time_phrase(q):
        notes.append(
            f"PERIOD NOT APPLIED: the question says '{time_phrase(q)}' but it could not "
            "be turned into a date range — these figures cover ALL order dates. Say so "
            "plainly; never say the result is for that period."
        )

    results = []
    for period in periods:
        match = dict(base)
        if period:
            match[DATE_FIELD] = {"$gte": period["start"], "$lt": period["end"]}
        section: Dict[str, Any] = {
            "period": (
                f"{period['label']} ({period['start']} to {period['end']}, end exclusive, on {DATE_FIELD})"
                if period else "ALL order dates (no period in the question)"
            ),
            "by": {},
        }
        for dim in dims:
            rows = [_row(r, dim) for r in _aggregate(match, dim)]
            by_currency: Dict[str, Any] = {}
            for cur in sorted({r["currency"] for r in rows}):
                cur_rows = [r for r in rows if r["currency"] == cur]
                ranked = _rank(cur_rows, metric, direction, losing)
                by_currency[cur] = {
                    "totals": _totals(cur_rows),
                    "rows": ranked[:limit],
                    "groups_total": len(cur_rows),
                }
            section["by"][dim] = by_currency
        results.append(section)

    checkpoint(
        "ANALYTICS",
        "profitability",
        dimensions=dims,
        metric=metric,
        direction=direction,
        losing=losing,
        periods=[p["label"] if p else None for p in periods],
        customer=customer,
    )
    return {
        "analytics_type": "profitability",
        "dimensions": dims,
        "metric": metric,
        "direction": direction,
        "losing_money_mode": losing,
        "customer_filter": customer,
        "notes": notes,
        "results": results,
    }


def _totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    costed = sum(r["revenue_with_cost"] for r in rows)
    cost = sum(r["freight_cost"] for r in rows)
    with_cost = sum(r["orders_with_carrier_cost"] for r in rows)
    profit = (costed - cost) if with_cost else None
    return {
        "orders": sum(r["orders"] for r in rows),
        "revenue": round(sum(r["revenue"] for r in rows), 2),
        "orders_with_carrier_cost": with_cost,
        "freight_cost": round(cost, 2),
        "profit": round(profit, 2) if profit is not None else None,
        "margin_pct": round(profit / costed * 100.0, 2) if (profit is not None and costed) else None,
    }


def format_profitability_for_context(payload: Dict[str, Any]) -> str:
    dim_label = {
        "lane": "LANE (origin city -> destination city)",
        "origin": "ORIGIN CITY",
        "destination": "DESTINATION CITY",
        "customer": "CUSTOMER",
    }
    lines = [
        "PROFITABILITY RESULT (exact engine output — do not recalculate, re-rank or invent):",
        "Formula: Revenue = grosstotalfreight; Freight Cost = carrier cost (outorderamount, "
        "what we pay the outsourced carrier); Profit = Revenue - Freight Cost; "
        "Margin % = Profit / Revenue x 100. Profit and margin use ONLY orders that have a "
        "carrier cost recorded (revenue_with_cost); profit=None means no cost is recorded "
        "for that group — say 'cost not recorded', never 100% margin. When you show a "
        "profit or margin, show revenue_with_cost next to it (not the total revenue), and "
        "mention total revenue separately only if it differs. Driver pay and fuel "
        "cost are not in the order data, so they are not deducted — mention this once.",
        f"grouped by: {', '.join(dim_label[d] for d in payload.get('dimensions') or [])}",
        f"ranked by: {payload.get('metric')} ({payload.get('direction')})"
        + (" — losing-money mode: only negative-profit groups" if payload.get("losing_money_mode") else ""),
        "CURRENCY: every amount below belongs to the currency of its block. Show the "
        "currency code on every amount and NEVER add CAD and USD together.",
    ]
    if payload.get("customer_filter"):
        lines.append(f"customer_filter_applied: {payload['customer_filter']}")
    for note in payload.get("notes") or []:
        lines.append(note)
    for section in payload.get("results") or []:
        lines.append(f"=== PERIOD: {section['period']} ===")
        for dim, by_cur in section["by"].items():
            if not by_cur:
                lines.append(f"{dim_label[dim]}: no orders with revenue in this period — say so plainly.")
            for cur, block in by_cur.items():
                t = block["totals"]
                lines.append(
                    f"--- {dim_label[dim]} | currency {cur} | {block['groups_total']} groups | "
                    f"totals: orders={t['orders']} revenue={t['revenue']} {cur} "
                    f"freight_cost={t['freight_cost']} {cur} profit={t['profit']} {cur} "
                    f"margin_pct={t['margin_pct']} (orders_with_carrier_cost={t['orders_with_carrier_cost']})"
                )
                for i, r in enumerate(block["rows"], start=1):
                    lines.append(
                        f"[{i}] {r[dim]} | orders={r['orders']} | revenue={r['revenue']} {cur} | "
                        f"revenue_with_cost={r['revenue_with_cost']} {cur} | "
                        f"freight_cost={r['freight_cost']} {cur} | profit={r['profit']} | "
                        f"margin_pct={r['margin_pct']} | with_cost={r['orders_with_carrier_cost']} "
                        f"| cost_not_recorded={r['orders_cost_not_recorded']} "
                        f"| cost_in_other_currency={r['orders_cost_other_currency']}"
                    )
    lines.append("Write numbers without commas.")
    return "\n".join(lines)
