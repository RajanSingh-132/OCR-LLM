"""
Answer-accuracy guards shared by the order / invoice / trip query planners.

Periods
    ``period_bounds`` — named periods (today ... last_year, this_quarter,
    last_quarter) -> (start, end_exclusive) ISO dates.
    ``resolve_period`` — a time phrase in the question -> {"start", "end",
    "label"} (used by the deterministic engines and the period guard).
    ``apply_period_guard`` — when the question names a period but the LLM plan
    carries no date filter, inject one on the domain's default date field; if
    the phrase can't be resolved, return a note so the answer says the figures
    cover all dates. Never let an answer claim a filter that was not applied.

Currency
    ``split_spec_by_currency`` — a sum/avg/min/max over a money field is grouped
    by the currency field as well, so CAD and USD are never added together.
    ``currency_totals`` — one grand total per currency for the same match.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

PERIOD_NAMES = frozenset(
    {
        "today", "yesterday", "this_week", "last_week",
        "this_month", "last_month", "this_quarter", "last_quarter",
        "this_year", "last_year",
    }
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _month_add(d: date, n: int) -> date:
    total = (d.year * 12 + (d.month - 1)) + n
    return date(total // 12, total % 12 + 1, 1)


def _quarter_start(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def period_bounds(name: str) -> Optional[Tuple[str, str]]:
    """(start_inclusive, end_exclusive) ISO dates for a named period."""
    today = _today()
    if name == "today":
        s, e = today, today + timedelta(days=1)
    elif name == "yesterday":
        s, e = today - timedelta(days=1), today
    elif name == "this_week":
        s = today - timedelta(days=today.weekday())
        e = s + timedelta(days=7)
    elif name == "last_week":
        e = today - timedelta(days=today.weekday())
        s = e - timedelta(days=7)
    elif name == "this_month":
        s = today.replace(day=1)
        e = _month_add(s, 1)
    elif name == "last_month":
        e = today.replace(day=1)
        s = _month_add(e, -1)
    elif name == "this_quarter":
        s = _quarter_start(today)
        e = _month_add(s, 3)
    elif name == "last_quarter":
        e = _quarter_start(today)
        s = _month_add(e, -3)
    elif name == "this_year":
        s, e = date(today.year, 1, 1), date(today.year + 1, 1, 1)
    elif name == "last_year":
        s, e = date(today.year - 1, 1, 1), date(today.year, 1, 1)
    else:
        return None
    return s.isoformat(), e.isoformat()


def _named(name: str, label: str) -> Dict[str, str]:
    s, e = period_bounds(name)  # type: ignore[misc]
    return {"start": s, "end": e, "label": label}


def _parse_month_day(text: str, default_year: int) -> Optional[date]:
    m = re.match(
        r"([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", text.strip(), re.I
    )
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if not month:
        return None
    year = int(m.group(3)) if m.group(3) else default_year
    try:
        return date(year, month, int(m.group(2)))
    except ValueError:
        return None


def resolve_period(question: str) -> Optional[Dict[str, str]]:
    """Time phrase in the question -> {"start", "end" (exclusive), "label"}.

    Relative phrases are derived from today's UTC date. Returns None when the
    question names no period this function can resolve (see time_phrase()).
    """
    q = (question or "").lower()
    today = _today()

    for pattern, name, label in (
        (r"\btoday\b", "today", "today"),
        (r"\byesterday\b", "yesterday", "yesterday"),
        (r"\b(?:last|previous|prior)\s+week\b", "last_week", "last week"),
        (r"\bthis\s+week\b", "this_week", "this week"),
        (r"\b(?:last|previous|prior)\s+month\b", "last_month", "last month"),
        (r"\bthis\s+month\b|\bmonth[- ]to[- ]date\b|\bmtd\b", "this_month", "this month"),
        (r"\b(?:last|previous|prior)\s+quarter\b", "last_quarter", "last quarter"),
        (r"\b(?:this|current)\s+quarter\b|\bquarter[- ]to[- ]date\b|\bqtd\b",
         "this_quarter", "this quarter"),
        (r"\b(?:last|previous|prior)\s+year\b", "last_year", "last year"),
        (r"\b(?:this|current)\s+year\b|\byear[- ]to[- ]date\b|\bytd\b",
         "this_year", "this year"),
    ):
        if re.search(pattern, q):
            return _named(name, label)

    # "last / past N days|weeks|months"
    m = re.search(r"\b(?:last|past|previous)\s+(\d{1,3})\s+(day|week|month)s?\b", q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        start = today - timedelta(days=days)
        return {
            "start": start.isoformat(),
            "end": (today + timedelta(days=1)).isoformat(),
            "label": f"last {n} {unit}s",
        }

    # "Q3" / "Q3 2026"
    m = re.search(r"\bq([1-4])(?:\s*(?:of\s+)?(20\d{2}))?\b", q)
    if m:
        year = int(m.group(2)) if m.group(2) else today.year
        start = date(year, 3 * (int(m.group(1)) - 1) + 1, 1)
        return {
            "start": start.isoformat(),
            "end": _month_add(start, 3).isoformat(),
            "label": f"Q{m.group(1)} {year}",
        }

    # "between August 1 and August 31[, 2026]"
    m = re.search(
        rf"\bbetween\s+((?:{_MONTH_RE})\.?\s+\d{{1,2}}(?:,?\s*\d{{4}})?)\s+and\s+"
        rf"((?:{_MONTH_RE})\.?\s+\d{{1,2}}(?:,?\s*\d{{4}})?)\b",
        q,
    )
    if m:
        d1 = _parse_month_day(m.group(1), today.year)
        d2 = _parse_month_day(m.group(2), today.year)
        if d1 and d2:
            if d2 < d1:
                d1, d2 = d2, d1
            return {
                "start": d1.isoformat(),
                "end": (d2 + timedelta(days=1)).isoformat(),
                "label": f"{m.group(1)} to {m.group(2)}",
            }

    # "on August 15[, 2026]"
    m = re.search(rf"\b((?:{_MONTH_RE})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?)\b", q)
    if m:
        d = _parse_month_day(m.group(1), today.year)
        if d:
            return {
                "start": d.isoformat(),
                "end": (d + timedelta(days=1)).isoformat(),
                "label": m.group(1),
            }

    # "in August [2026]" — the whole month
    m = re.search(rf"\b(?:in|for|during|of)\s+({_MONTH_RE})(?:\s+(20\d{{2}}))?\b", q)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            year = int(m.group(2)) if m.group(2) else today.year
            start = date(year, month, 1)
            return {
                "start": start.isoformat(),
                "end": _month_add(start, 1).isoformat(),
                "label": f"{m.group(1).title()} {year}",
            }

    # ISO date 2026-08-15
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return {
                "start": d.isoformat(),
                "end": (d + timedelta(days=1)).isoformat(),
                "label": d.isoformat(),
            }
        except ValueError:
            pass
    return None


# Any wording that names a time window — resolvable or not.
_TIME_PHRASE_RE = re.compile(
    r"\b(?:today|yesterday|tonight|"
    r"(?:this|last|previous|prior|current|next|past)\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|quarters?|years?|fiscal\s+year|season)|"
    r"(?:month|quarter|year)[- ]to[- ]date|mtd|qtd|ytd|q[1-4]|"
    r"since\s+\w+|between\s+\w+|"
    rf"(?:in|for|during|of|on)\s+(?:{_MONTH_RE})\b)",
    re.I,
)


def time_phrase(question: str) -> Optional[str]:
    m = _TIME_PHRASE_RE.search(question or "")
    return m.group(0) if m else None


def _is_date_filter(f: Dict[str, Any]) -> bool:
    return str(f.get("op") or "").startswith("date_") or f.get("op") in (
        "period", "last_days",
    )


def has_date_filter(
    filters: List[Dict[str, Any]], segments: Optional[List[Dict[str, Any]]] = None
) -> bool:
    if any(_is_date_filter(f) for f in filters or []):
        return True
    for seg in segments or []:
        if any(_is_date_filter(f) for f in seg.get("filters") or []):
            return True
    return False


def apply_period_guard(
    question: str,
    filters: List[Dict[str, Any]],
    segments: Optional[List[Dict[str, Any]]],
    date_field: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Make sure a period named in the question is really in the DB query.

    Returns (filters, notes). When the question names a period and the plan has
    no date filter, a date_range filter on ``date_field`` is added. When the
    phrase can't be resolved, the filters are unchanged and a note tells the
    answer model to say the figures cover all dates.
    """
    phrase = time_phrase(question)
    if not phrase or has_date_filter(filters, segments):
        return filters, []
    period = resolve_period(question)
    if period:
        end_incl = (date.fromisoformat(period["end"]) - timedelta(days=1)).isoformat()
        added = {"field": date_field, "op": "date_range", "value": [period["start"], end_incl]}
        return list(filters) + [added], [
            f"period_applied: {period['label']} ({period['start']} to {end_incl}) "
            f"on {date_field}"
        ]
    return filters, [
        f"PERIOD NOT APPLIED: the question says '{phrase}' but no date filter "
        "could be applied — these figures cover ALL dates. Tell the user this "
        "plainly; never say the result is for that period."
    ]


def period_guard_context(notes: List[str]) -> str:
    """Context block for the answer model; empty when there is nothing to say."""
    if not notes:
        return ""
    return "QUERY NOTES (follow exactly):\n" + "\n".join(f"- {n}" for n in notes)


# ------------------------------------------------------------------ currency
def split_spec_by_currency(
    spec: Optional[Dict[str, Any]],
    currency_field: str,
    is_money_field,
) -> bool:
    """Group a money aggregate by currency too. Mutates ``spec``; True if split.

    A metric (one grand total) becomes a group by currency; a group gets the
    currency field appended. Count-only specs are left alone.
    """
    if not spec or spec.get("operation") not in ("metric", "group"):
        return False
    money = [
        m for m in spec.get("metrics") or []
        if m.get("field") and is_money_field(m["field"])
    ]
    if not money:
        return False
    group_by = list(spec.get("group_by") or [])
    if currency_field in group_by:
        return True
    group_by.append(currency_field)
    spec["group_by"] = group_by
    if spec["operation"] == "metric":
        spec["operation"] = "group"
        spec.setdefault("limit", 20)
        spec.setdefault("sort", None)
    return True


def currency_totals(
    collection,
    match: Dict[str, Any],
    spec: Dict[str, Any],
    currency_field: str,
    numeric_expr,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    """One row per currency: record count + every money metric over the match."""
    group: Dict[str, Any] = {"_id": f"${currency_field}", "records": {"$sum": 1}}
    for m in spec.get("metrics") or []:
        if m.get("field"):
            group[m["key"]] = {f"${m['fn']}": numeric_expr(m["field"])}
    rows = list(
        collection.aggregate(
            [{"$match": match}, {"$group": group}, {"$sort": {"records": -1}}],
            maxTimeMS=timeout_ms,
        )
    )
    out = []
    for r in rows:
        row = {"currency": r.get("_id") or "UNKNOWN", "records": r.get("records")}
        for k, v in r.items():
            if k not in ("_id", "records"):
                row[k] = round(float(v), 2) if isinstance(v, (int, float)) else v
        out.append(row)
    return out


def currency_totals_context(rows: List[Dict[str, Any]], noun: str) -> str:
    if not rows:
        return ""
    lines = [
        f"TOTALS BY CURRENCY (all matching {noun}, not just the rows shown — "
        "report one total per currency, NEVER add currencies together):"
    ]
    for r in rows:
        lines.append("- " + " | ".join(f"{k}={v}" for k, v in r.items()))
    return "\n".join(lines)
