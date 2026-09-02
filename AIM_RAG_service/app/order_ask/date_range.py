"""
Date range extraction — NEW capability, not present before this change.

The existing app/order_ask/analytics.py only extracts a SINGLE date
("10 august", "2026-08-10"). Bucket 1 questions ("orders created between
August 1 and August 15", "compare Aug 1-15 vs Aug 16-31") need actual
date RANGES, including two ranges in one question for comparisons.

Reuses the month-name map and day/month parsing patterns already proven
in app/order_ask/analytics.py rather than duplicating that logic blind.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from app.order_ask.analytics import _MONTH_MAP  # reuse existing month map

DateRange = Tuple[str, str]  # (YYYY-MM-DD, YYYY-MM-DD)

_DEFAULT_YEAR = 2026  # matches the fallback already used in analytics.py

_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)


def _month_num(name: str) -> Optional[int]:
    key3 = name[:3].lower()
    return _MONTH_MAP.get(key3) or _MONTH_MAP.get(name.lower())


def _mk_date(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _find_single_dates(text: str, year: int) -> List[str]:
    """Find every 'DD Month' or 'Month DD' occurrence in text, in order."""
    found: List[Tuple[int, str]] = []  # (position, date_str)

    for m in re.finditer(
        rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\b", text, re.I
    ):
        day = int(m.group(1))
        month = _month_num(m.group(2))
        if month:
            d = _mk_date(year, month, day)
            if d:
                found.append((m.start(), d))

    for m in re.finditer(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.I
    ):
        month = _month_num(m.group(1))
        day = int(m.group(2))
        if month:
            d = _mk_date(year, month, day)
            if d:
                found.append((m.start(), d))

    found.sort(key=lambda x: x[0])
    return [d for _, d in found]


def extract_date_range(question: str) -> Optional[DateRange]:
    """
    'between August 1 and August 15' -> ('2026-08-01', '2026-08-15')
    'August 1 to August 15'          -> same
    'August 1-15' / 'Aug 1 - 15'     -> same (shared month, two day numbers)
    """
    q = question or ""
    ql = q.lower()

    year_m = re.search(r"\b(20\d{2})\b", ql)
    year = int(year_m.group(1)) if year_m else _DEFAULT_YEAR

    # Pattern: "<Month> D1-D2" or "<Month> D1 to D2" (single month, day range)
    m = re.search(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*"
        rf"(?:-|to|through|till|until)\s*(\d{{1,2}})\s*(?:st|nd|rd|th)?\b",
        ql,
    )
    if m:
        month = _month_num(m.group(1))
        d1, d2 = int(m.group(2)), int(m.group(3))
        if month:
            start = _mk_date(year, month, d1)
            end = _mk_date(year, month, d2)
            if start and end:
                return (start, end) if start <= end else (end, start)

    # Generic: find every date-like mention in order, take first two
    dates = _find_single_dates(q, year)
    if len(dates) >= 2:
        start, end = dates[0], dates[1]
        return (start, end) if start <= end else (end, start)

    return None


def extract_two_date_ranges(question: str) -> Optional[Tuple[DateRange, DateRange]]:
    """
    For compare questions: 'compare order count between August 1-15 and
    August 16-31' -> (('2026-08-01','2026-08-15'), ('2026-08-16','2026-08-31'))

    Splits on the connecting 'and'/'vs'/'versus' between two "Month D-D"
    chunks, then parses each side independently.
    """
    q = question or ""
    ql = q.lower()
    year_m = re.search(r"\b(20\d{2})\b", ql)
    year = int(year_m.group(1)) if year_m else _DEFAULT_YEAR

    range_pat = rf"({_MONTH_PATTERN})\s+(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*(?:-|to|through|till|until)\s*(\d{{1,2}})\s*(?:st|nd|rd|th)?"
    matches = list(re.finditer(range_pat, ql))
    if len(matches) >= 2:
        ranges = []
        for m in matches[:2]:
            month = _month_num(m.group(1))
            d1, d2 = int(m.group(2)), int(m.group(3))
            if not month:
                return None
            start = _mk_date(year, month, d1)
            end = _mk_date(year, month, d2)
            if not start or not end:
                return None
            ranges.append((start, end) if start <= end else (end, start))
        return ranges[0], ranges[1]

    return None


def last_n_days_range(n: int) -> DateRange:
    """'last 7 days' -> (today-6, today) inclusive of today."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=n - 1)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def detect_last_n_days(question: str) -> Optional[int]:
    m = re.search(r"\blast\s+(\d{1,3})\s+days?\b", question or "", re.I)
    if m:
        return int(m.group(1))
    return None
