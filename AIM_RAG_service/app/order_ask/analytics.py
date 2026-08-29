"""
Fleet-wide analytics over ALL Avaal_order order records.

Supports:
- order status summary / counts (Quoted, Confirmed, Dispatched, Started, In-Transit,
  Partially Delivered, Delivered, Cancelled, Rejected)
- accounting status (Invoiced, PartiallyPaid, Paid, Restricted)
- outsource status (Open, Planned, Assigned, Quoted, Delivered)
- best / worst / low customer by order count (default) or revenue
- customer counts by country using pickup / delivery (drop) addresses
- activity on a date (customers + orders)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.order_ask.checkpoint import checkpoint
from app.tenants.router import (
    get_orders_collection,
    get_orders_metadata_type,
    get_orders_namespace,
)

KNOWN_STATUSES = [
    "Quoted",
    "Confirmed",
    "Dispatched",
    "Started",
    "In-Transit",
    "Partially Delivered",
    "Delivered",
    "Cancelled",
    "Rejected",
]

KNOWN_ACCOUNTING_STATUSES = [
    "Invoiced",
    "PartiallyPaid",
    "Paid",
    "Restricted",
]

KNOWN_OUTSOURCE_STATUSES = [
    "Open",
    "Planned",
    "Assigned",
    "Quoted",
    "Delivered",
]

COUNTRY_PATTERNS = {
    "Canada": [r"\bCanada\b", r"\bCAN\b"],
    "United States": [
        r"\bUnited States\b",
        r"\bUSA\b",
        r"\bU\.S\.A\.?\b",
        r"\bU\.S\.?\b",
    ],
}


def _base_match() -> Dict[str, Any]:
    return {
        "namespace": get_orders_namespace(),
        "metadata.type": get_orders_metadata_type(),
    }


def _country_regex(country: str) -> str:
    pats = COUNTRY_PATTERNS.get(country) or [rf"\b{re.escape(country)}\b"]
    return "(?:" + "|".join(pats) + ")"


def detect_country(question: str) -> Optional[str]:
    q = (question or "").lower()
    if re.search(r"\b(canada|canadian)\b", q):
        return "Canada"
    if re.search(
        r"\b(united states|u\.s\.a\.?|usa|america|american)\b|\bin the us\b|\bin us\b|\bus\b",
        q,
    ):
        return "United States"
    return None


def detect_location_side(question: str) -> str:
    """pickup | delivery | both — default both for country/customer geo questions."""
    q = (question or "").lower()
    wants_pickup = bool(re.search(r"\b(pickup|pick\s*up|origin|shipper)\b", q))
    wants_delivery = bool(
        re.search(r"\b(delivery|deliver|drop|consignee|destination)\b", q)
    )
    if wants_pickup and not wants_delivery:
        return "pickup"
    if wants_delivery and not wants_pickup:
        return "delivery"
    return "both"


def is_status_summary_question(question: str) -> bool:
    q = (question or "").lower()
    # "list/show confirmed orders" is a filtered LIST, not a status summary/count
    if re.search(r"\b(list|show|display|find|search|filter|get)\b", q) and not re.search(
        r"\b(how many|count|summary|break\s*down|breakdown|distribution|total|kitne|kitna)\b",
        q,
    ):
        return False
    if re.search(r"\bstatus\b.*\b(summary|break\s*down|breakdown|count|how many|distribution)\b", q):
        return True
    if re.search(r"\b(summary|break\s*down|breakdown)\b.*\bstatus", q):
        return True
    status_words = (
        r"quoted|cancelled|canceled|confirmed|confirm|dispatched|started|"
        r"in[- ]?transit|partially\s*delivered|delivered|rejected|"
        r"invoiced|partially\s*paid|paid|restricted|"
        r"open|planned|assigned"
    )
    if re.search(rf"\b(how many|count|kitne|kitna)\b.*\b({status_words})\b", q):
        return True
    if re.search(rf"\b({status_words})\b.*\b(count|how many|kitne|kitna|huye|hua|hue)\b", q):
        return True
    if re.search(rf"\b({status_words})\b.*\borders?\b", q) and re.search(
        r"\b(how many|count|summary|total|breakdown|kitne|kitna)\b", q
    ):
        return True
    if re.search(
        r"\b(order|accounting|out\s*source|outsource)\s*status(es)?\b", q
    ) and re.search(r"\b(summary|how many|count|break|breakdown|distribution)\b", q):
        return True
    return False


_BEST_WORDS = r"best|top|biggest|largest|highest|most|maximum|max|premium"
_WORST_WORDS = (
    r"worst|lowest|low|least|fewest|smallest|minimum|min|bottom|poorest|weakest"
)


def detect_customer_direction(question: str) -> str:
    """best (highest) vs worst (lowest) customer ranking direction."""
    q = (question or "").lower()
    if re.search(rf"\b({_WORST_WORDS})\b", q):
        return "worst"
    return "best"


def is_best_customer_question(question: str) -> bool:
    """
    Any customer-ranking question — best/top AND worst/lowest/low customer.
    e.g. best customer, top customer by revenue, worst customer, low customer,
    customer with least orders, smallest customer.
    """
    q = (question or "").lower()
    if not re.search(r"\bcustomers?\b|\bclients?\b", q):
        return False
    if re.search(
        rf"\b({_BEST_WORDS}|{_WORST_WORDS})\b.*\b(customer|client)\b", q
    ):
        return True
    if re.search(
        rf"\b(customer|client)\b.*\b({_BEST_WORDS}|{_WORST_WORDS}|orders?|revenue|freight|amount|sales)\b",
        q,
    ) and re.search(rf"\b({_BEST_WORDS}|{_WORST_WORDS})\b", q):
        return True
    return False


def is_country_customer_question(question: str) -> bool:
    q = (question or "").lower()
    if not detect_country(q):
        return False
    return bool(
        re.search(
            r"\b(how many|count|number of|customers?|client|in canada|in us|in usa|in the us|in united)\b",
            q,
        )
    )


def normalize_date_prefix(raw: str) -> Optional[str]:
    """
    Normalize user date text to YYYY-MM-DD prefix for matching ISO orderdate fields.
    Supports: 2026-08-06, 2026/08/06, 07/13/2026, 7-13-2026.
    """
    if not raw:
        return None
    text = raw.strip()

    m = re.match(r"^(20\d{2})[-/](\d{1,2})[-/](\d{1,2})$", text)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{y}-{mo:02d}-{d:02d}"

    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](20\d{2})$", text)
    if m:
        # Assume MM/DD/YYYY (common in Avaal Order Sheets)
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if mo > 12 and d <= 12:
            mo, d = d, mo  # swap if clearly DD/MM
        return f"{y}-{mo:02d}-{d:02d}"

    return None


def extract_date_from_question(question: str) -> Optional[str]:
    q = question or ""
    m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", q)
    if m:
        return normalize_date_prefix(m.group(1))
    m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b", q)
    if m:
        return normalize_date_prefix(m.group(1))
    return None


def detect_date_field(question: str) -> str:
    """Which date field to filter: orderdate | pickupdate | deliverydate."""
    q = (question or "").lower()
    if re.search(r"\b(pickup|pick\s*up)\b", q):
        return "pickupdate"
    if re.search(r"\b(delivery|deliver|drop)\b", q):
        return "deliverydate"
    return "orderdate"


def is_date_activity_question(question: str) -> bool:
    """
    e.g. how many customers ordered on 2026-08-06 /
    is date ko kitne customers ne order diya /
    orders on 07/13/2026 /
    10 august how many total orders /
    aaj kitne order create / confirm / dispatched
    """
    q = (question or "").lower()
    if is_today_orders_question(question):
        return True
    has_date = bool(extract_any_date_from_question(question))
    if not has_date:
        return False
    return bool(
        re.search(
            r"\b(how many|count|kitne|kitna|number of|customers?|orders?|ordered|order diya|ne order|created|total|confirm|confirmed|dispatched)\b",
            q,
        )
    ) or bool(re.search(r"\b(on|for|dated|date)\b", q))


def is_analytics_question(question: str) -> bool:
    return (
        is_status_summary_question(question)
        or is_best_customer_question(question)
        or is_country_customer_question(question)
        or is_date_activity_question(question)
        or is_state_wise_question(question)
        or is_city_wise_question(question)
        or is_best_city_question(question)
        or is_period_orders_question(question)
        or is_trip_distance_question(question)
        or is_orders_by_country_question(question)
        or is_best_order_question(question)
        or is_worst_order_question(question)
        or is_today_orders_question(question)
    )


_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_natural_date_from_question(question: str) -> Optional[str]:
    """
    Parse dates like '10 august', 'august 10', '10 aug 2026', 'Aug 10, 2026'.
    Returns YYYY-MM-DD prefix or None.
    """
    q = question or ""
    ql = q.lower()
    year = None
    ym = re.search(r"\b(20\d{2})\b", ql)
    if ym:
        year = int(ym.group(1))

    day = None
    month = None
    m = re.search(
        r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\b",
        ql,
    )
    if m:
        day = int(m.group(1))
        month = _MONTH_MAP.get(m.group(2)[:3] if m.group(2)[:3] in _MONTH_MAP else m.group(2))
        if month is None:
            month = _MONTH_MAP.get(m.group(2))
    else:
        m = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+"
            r"(\d{1,2})(?:st|nd|rd|th)?\b",
            ql,
        )
        if m:
            month = _MONTH_MAP.get(m.group(1)[:3] if m.group(1)[:3] in _MONTH_MAP else m.group(1))
            if month is None:
                month = _MONTH_MAP.get(m.group(1))
            day = int(m.group(2))

    if day is None or month is None:
        return None
    if day < 1 or day > 31:
        return None
    if year is None:
        # Prefer year from latest orderdate in DB-ish default used in dataset
        year = 2026
    return f"{year}-{month:02d}-{day:02d}"


def extract_any_date_from_question(question: str) -> Optional[str]:
    q = question or ""
    ql = q.lower()
    # today / aaj
    if re.search(r"\b(today|aaj|aj)\b", ql):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return extract_date_from_question(question) or extract_natural_date_from_question(question)


def is_today_orders_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\b(today|aaj|aj)\b", q):
        return False
    return bool(
        re.search(
            r"\b(how many|count|kitne|kitna|orders?|created|confirm|confirmed|"
            r"dispatched|quoted|delivered|invoiced|huye|hua)\b",
            q,
        )
    )


def is_orders_by_country_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\borders?\b", q):
        return False
    return bool(
        re.search(
            r"\bcountry[\s\-]*wise\b|"
            r"\b(by|per|across)\s+countr(?:y|ies)\b|"
            r"\borders?\b.*\bby\s+countr|"
            r"\bcountr(?:y|ies)\b.*\b(total|count|how many)\s+orders?\b|"
            r"\b(total|count|how many)\s+orders?\b.*\bcountr",
            q,
        )
    )


def is_best_order_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\bcustomers?\b|\bcit(?:y|ies)\b", q):
        return False
    return bool(
        re.search(
            r"\bbest\s+orders?\b|"
            r"\b(best|top|highest|maximum|max|largest)\b.*\borders?\b|"
            r"\borders?\b.*\b(best|highest|maximum|max)\b.*\b(freight|amount|value|revenue)?",
            q,
        )
    )


def is_worst_order_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\bcustomers?\b|\bcit(?:y|ies)\b", q):
        return False
    return bool(
        re.search(
            r"\bworst\s+orders?\b|"
            r"\b(worst|lowest|minimum|min|smallest|cheapest)\b.*\borders?\b|"
            r"\borders?\b.*\b(worst|lowest|minimum|min)\b",
            q,
        )
    )


def detect_period_days(question: str) -> Optional[int]:
    """last 1 month / last month / past 7 days / last 2 weeks → day count."""
    q = (question or "").lower()
    m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s*(day|days|week|weeks|month|months)\b", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            return max(1, n)
        if unit.startswith("week"):
            return max(1, n * 7)
        return max(1, n * 30)
    if re.search(r"\b(?:last|past|previous)\s+(one\s+)?month\b", q):
        return 30
    if re.search(r"\b(?:last|past|previous)\s+(one\s+)?week\b", q):
        return 7
    if re.search(r"\b(?:last|past)\s+30\s*days?\b", q):
        return 30
    return None


def detect_status_filter(question: str) -> Optional[str]:
    """Detect orderstatus value from question (lifecycle only)."""
    q = (question or "").lower()
    for key, canonical in (
        ("partially delivered", "Partially Delivered"),
        ("partiallydelivered", "Partially Delivered"),
        ("in-transit", "In-Transit"),
        ("in transit", "In-Transit"),
        ("intransit", "In-Transit"),
        ("quoted", "Quoted"),
        ("quotes", "Quoted"),
        ("quote", "Quoted"),
        ("cancelled", "Cancelled"),
        ("canceled", "Cancelled"),
        ("confirmed", "Confirmed"),
        ("confirmation", "Confirmed"),
        ("confirmations", "Confirmed"),
        ("confirm", "Confirmed"),
        ("dispatched", "Dispatched"),
        ("started", "Started"),
        ("delivered", "Delivered"),
        ("rejected", "Rejected"),
    ):
        if re.search(rf"\b{re.escape(key)}\b", q):
            return canonical
    return None


def detect_accounting_status_filter(question: str) -> Optional[str]:
    q = (question or "").lower()
    for key, canonical in (
        ("invoice restricted", "Restricted"),
        ("invoiced restricted", "Restricted"),
        ("restricted", "Restricted"),
        ("partially paid", "PartiallyPaid"),
        ("partiallypaid", "PartiallyPaid"),
        ("partial paid", "PartiallyPaid"),
        ("invoiced", "Invoiced"),
        ("paid", "Paid"),
    ):
        if re.search(rf"\b{re.escape(key)}\b", q):
            if canonical == "Paid" and not re.search(
                r"\b(paid\s+orders?|orders?\s+.*paid|accounting|how many\s+paid|kitne\s+paid)\b",
                q,
            ):
                continue
            return canonical
    return None


def detect_outsource_status_filter(question: str) -> Optional[str]:
    q = (question or "").lower()
    wants_out = bool(re.search(r"\b(out\s*source|outsource|out\s*status|outstatus)\b", q))
    if wants_out:
        if re.search(r"\bdelivered\b", q):
            return "Delivered"
        if re.search(r"\bquoted\b", q):
            return "Quoted"
    for key, canonical in (
        ("assigned", "Assigned"),
        ("planned", "Planned"),
        ("open", "Open"),
    ):
        if re.search(rf"\b{key}\b", q):
            return canonical
    return None


def detect_status_field(question: str) -> str:
    """Which status column to summarize: orderstatus | accountingstatus | outstatus."""
    q = (question or "").lower()
    if re.search(r"\b(accounting|account\s*status|invoiced|partially\s*paid|restricted)\b", q):
        return "accountingstatus"
    if re.search(r"\b(out\s*source|outsource|out\s*status|outstatus|assigned|planned)\b", q):
        if re.search(r"\b(quoted|confirmed|dispatched|started|cancelled|canceled|rejected)\b", q) and not re.search(
            r"\b(out\s*source|outsource)\b", q
        ):
            return "orderstatus"
        return "outstatus"
    if detect_accounting_status_filter(q):
        return "accountingstatus"
    if detect_outsource_status_filter(q) and not detect_status_filter(q):
        return "outstatus"
    return "orderstatus"


def is_state_wise_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\bstate[- ]?wise\b|\bby\s+state\b|\borders?\s+per\s+state\b|\bstate\s+wise\b", q)
        or (
            re.search(r"\b(states?)\b", q)
            and re.search(r"\b(orders?|count|how many|break\s*down|summary)\b", q)
            and not re.search(r"\b(customer|best|worst)\b", q)
        )
    )


def is_city_wise_question(question: str) -> bool:
    q = (question or "").lower()
    if is_best_city_question(q):
        return False
    return bool(
        re.search(r"\bcity[- ]?wise\b|\bby\s+city\b|\borders?\s+per\s+city\b|\bcity\s+wise\b", q)
        or (
            re.search(r"\bcities?\b", q)
            and re.search(r"\b(orders?|count|how many|break\s*down|summary)\b", q)
            and not re.search(r"\b(customer|best|worst)\b", q)
        )
    )


def is_best_city_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b(best|top|biggest|largest|highest|most)\b.*\bcity\b|"
            r"\bcity\b.*\b(best|top|most\s+orders?|maximum|max)\b",
            q,
        )
    )


def is_period_orders_question(question: str) -> bool:
    """last 1 month orders / last month quoted how many / created in last month."""
    q = (question or "").lower()
    if detect_period_days(q) is None:
        return False
    return bool(
        re.search(
            r"\b(how many|count|kitne|kitna|orders?|created|quoted|confirmed|confirmation|status)\b",
            q,
        )
    )


def is_trip_distance_question(question: str) -> bool:
    q = (question or "").lower()
    # Fleet-level trip/distance analytics (not a single-order lookup)
    if re.search(r"\b(MRP\d+|TORD\d+)\b", q, re.I):
        return False
    return bool(
        re.search(r"\b(trip|trips|tripno|trip\s*no)\b", q)
        or (
            re.search(r"\bdistance\b", q)
            and re.search(r"\b(total|sum|how many|average|avg|all|fleet)\b", q)
        )
    )


def parse_address_geo(addr: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Address shape: STREET, CITY, STATE, PIN, Country, email, phone
    e.g. 1241 OLD TEMESCAL ROAD #103, CORONA, CA, 92881, United States, ...
    """
    if not addr or not str(addr).strip():
        return None
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    if len(parts) < 3:
        return None
    city = parts[1] if len(parts) > 1 else ""
    state = parts[2] if len(parts) > 2 else ""
    country = ""
    if len(parts) >= 5:
        country = parts[4]
    elif len(parts) == 4:
        # STREET, CITY, STATE, Country (no pin)
        country = parts[3] if not re.match(r"^\d{5}", parts[3]) else ""
    # Drop email-like country mistakes
    if "@" in country:
        country = ""
    if not city and not state:
        return None
    return {
        "city": city.title() if city else "",
        "state": state.upper() if len(state) <= 3 else state.title(),
        "country": country.title() if country else "Unknown",
    }


def _iter_geo_from_orders(location_side: str = "both") -> List[Dict[str, str]]:
    collection = get_orders_collection()
    projection = {
        "pickupfulladdress": 1,
        "deliveryfulladdress": 1,
        "_id": 0,
    }
    geos: List[Dict[str, str]] = []
    for doc in collection.find(_base_match(), projection):
        addrs = []
        if location_side in ("pickup", "both"):
            addrs.append(doc.get("pickupfulladdress"))
        if location_side in ("delivery", "both"):
            addrs.append(doc.get("deliveryfulladdress"))
        seen = set()
        for addr in addrs:
            geo = parse_address_geo(addr)
            if not geo:
                continue
            key = (geo.get("country"), geo.get("state"), geo.get("city"))
            # Count each order once per side match — for "both", prefer first non-empty
            if key in seen:
                continue
            seen.add(key)
            geos.append(geo)
            if location_side == "both":
                break  # one geo per order when scanning both (pickup first)
    return geos


def orders_by_state(*, location_side: str = "both", limit: int = 50) -> Dict[str, Any]:
    """Country → state → order count only."""
    limit = max(1, min(int(limit or 50), 100))
    from collections import Counter

    counter: Counter = Counter()
    for geo in _iter_geo_from_orders(location_side):
        country = geo.get("country") or "Unknown"
        state = geo.get("state") or "Unknown"
        counter[(country, state)] += 1

    rows = [
        {"country": c, "state": s, "order_count": n}
        for (c, s), n in counter.most_common(limit)
    ]
    checkpoint("ANALYTICS", "orders_by_state", rows=len(rows), side=location_side)
    return {
        "analytics_type": "orders_by_state",
        "location_side": location_side,
        "definition": (
            "Order counts grouped by country then state, parsed from "
            "pickup/delivery full addresses. Answer format: country, then state, then count only."
        ),
        "response_format": "country_name → state_name → order_count (counts only)",
        "total_groups": len(rows),
        "rows": rows,
    }


def orders_by_city(*, location_side: str = "both", limit: int = 50) -> Dict[str, Any]:
    """City → order count only."""
    limit = max(1, min(int(limit or 50), 100))
    from collections import Counter

    counter: Counter = Counter()
    for geo in _iter_geo_from_orders(location_side):
        city = geo.get("city") or "Unknown"
        country = geo.get("country") or ""
        state = geo.get("state") or ""
        counter[(city, state, country)] += 1

    rows = [
        {"city": city, "state": st, "country": co, "order_count": n}
        for (city, st, co), n in counter.most_common(limit)
    ]
    checkpoint("ANALYTICS", "orders_by_city", rows=len(rows), side=location_side)
    return {
        "analytics_type": "orders_by_city",
        "location_side": location_side,
        "definition": (
            "Order counts grouped by city (from pickup/delivery addresses). "
            "Answer with city name and total order count only."
        ),
        "response_format": "city_name → order_count (counts only)",
        "total_groups": len(rows),
        "rows": rows,
    }


def best_cities(*, location_side: str = "both", limit: int = 5) -> Dict[str, Any]:
    """Best city = most orders."""
    payload = orders_by_city(location_side=location_side, limit=limit)
    rows = payload.get("rows") or []
    best = rows[0] if rows else None
    checkpoint(
        "ANALYTICS",
        "best_cities",
        top=(best or {}).get("city"),
        limit=limit,
    )
    return {
        "analytics_type": "best_city",
        "location_side": location_side,
        "definition": "Best city = city with the maximum number of orders (from addresses).",
        "response_format": "best city name + order_count",
        "best_city": best,
        "top_cities": rows,
    }


def orders_by_country(*, location_side: str = "both", limit: int = 50) -> Dict[str, Any]:
    """Country → order count from pickup/delivery addresses."""
    limit = max(1, min(int(limit or 50), 100))
    from collections import Counter

    counter: Counter = Counter()
    for geo in _iter_geo_from_orders(location_side):
        country = geo.get("country") or "Unknown"
        counter[country] += 1

    rows = [{"country": c, "order_count": n} for c, n in counter.most_common(limit)]
    checkpoint("ANALYTICS", "orders_by_country", rows=len(rows), side=location_side)
    return {
        "analytics_type": "orders_by_country",
        "location_side": location_side,
        "definition": (
            "Order counts grouped by country parsed from pickup/delivery full addresses."
        ),
        "response_format": "country_name → order_count (counts only)",
        "total_groups": len(rows),
        "rows": rows,
    }


def rank_orders_by_amount(
    *,
    direction: str = "best",
    limit: int = 5,
) -> Dict[str, Any]:
    """Best order = highest freight; worst = lowest freight."""
    limit = max(1, min(int(limit or 5), 25))
    direction = "worst" if str(direction).lower() == "worst" else "best"
    collection = get_orders_collection()
    projection = {
        "orderid": 1,
        "ordernumber": 1,
        "orderstatus": 1,
        "customername": 1,
        "totalfreight": 1,
        "grosstotalfreight": 1,
        "pickuplocationname": 1,
        "deliverylocationname": 1,
        "_id": 0,
    }

    def _amt(doc: Dict[str, Any]) -> Optional[float]:
        for field in ("grosstotalfreight", "totalfreight"):
            try:
                if doc.get(field) in (None, ""):
                    continue
                return float(doc.get(field))
            except (TypeError, ValueError):
                continue
        return None

    scored = []
    for doc in collection.find(_base_match(), projection):
        amt = _amt(doc)
        if amt is None:
            continue
        scored.append(
            {
                "orderid": doc.get("orderid"),
                "ordernumber": doc.get("ordernumber"),
                "orderstatus": doc.get("orderstatus"),
                "customername": doc.get("customername"),
                "totalfreight": amt,
                "pickuplocationname": doc.get("pickuplocationname"),
                "deliverylocationname": doc.get("deliverylocationname"),
            }
        )

    scored.sort(key=lambda r: r["totalfreight"], reverse=(direction == "best"))
    rows = scored[:limit]
    top = rows[0] if rows else None
    checkpoint("ANALYTICS", f"orders_by_amount_{direction}", rows=len(rows))
    return {
        "analytics_type": "best_order" if direction == "best" else "worst_order",
        "rank_by": "totalfreight",
        "direction": direction,
        "definition": (
            "Best order = highest freight amount. Worst order = lowest freight amount."
        ),
        "response_format": "ordernumber, amount, status, customer (from rows only)",
        "total_scored": len(scored),
        "rows": rows,
        "top": top,
    }


def _period_start_iso(days: int) -> str:
    from datetime import datetime, timedelta, timezone

    start = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    return start.strftime("%Y-%m-%d")


def orders_in_period(
    *,
    days: int = 30,
    status: Optional[str] = None,
    date_field: str = "orderdate",
) -> Dict[str, Any]:
    """Orders created in last N days, optional status filter (Quoted/Confirmed/…)."""
    if date_field not in ("orderdate", "pickupdate", "deliverydate"):
        date_field = "orderdate"
    start = _period_start_iso(days)
    collection = get_orders_collection()
    match: Dict[str, Any] = {
        **_base_match(),
        date_field: {"$gte": start},
    }
    if status:
        match["orderstatus"] = status

    total = collection.count_documents(match)
    status_rows = list(
        collection.aggregate(
            [
                {"$match": {**_base_match(), date_field: {"$gte": start}}},
                {"$group": {"_id": "$orderstatus", "order_count": {"$sum": 1}}},
                {"$sort": {"order_count": -1}},
            ]
        )
    )
    by_status = [
        {
            "status": (r["_id"] if r["_id"] not in (None, "") else "Unknown"),
            "order_count": int(r["order_count"]),
        }
        for r in status_rows
    ]
    checkpoint(
        "ANALYTICS",
        "orders_in_period",
        days=days,
        start=start,
        status=status,
        total=total,
    )
    return {
        "analytics_type": "orders_in_period",
        "days": days,
        "period_start": start,
        "date_field": date_field,
        "status_filter": status,
        "matching_orders": total,
        "by_status": by_status,
        "definition": (
            f"Orders with {date_field} on/after {start} (last ~{days} days)"
            + (f", status={status}" if status else "")
            + ". Use matching_orders as the total count."
        ),
        "response_format": "total order count for the period; if status asked, that status count; optional status breakdown",
    }


def trip_distance_summary(*, days: Optional[int] = None) -> Dict[str, Any]:
    """Fleet trip / distance snapshot from order fields."""
    collection = get_orders_collection()
    match: Dict[str, Any] = {**_base_match()}
    if days:
        match["orderdate"] = {"$gte": _period_start_iso(days)}

    total_orders = collection.count_documents(match)
    with_trip = collection.count_documents(
        {**match, "tripno": {"$exists": True, "$nin": [None, ""]}}
    )
    # Sum numeric distance where present
    dist_rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": None,
                        "total_distance": {
                            "$sum": {
                                "$cond": [
                                    {"$isNumber": "$distance"},
                                    "$distance",
                                    0,
                                ]
                            }
                        },
                        "orders_with_distance": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$ne": ["$distance", None]},
                                            {"$ne": ["$distance", ""]},
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                    }
                },
            ]
        )
    )
    total_distance = float((dist_rows[0] or {}).get("total_distance") or 0) if dist_rows else 0.0
    orders_with_distance = (
        int((dist_rows[0] or {}).get("orders_with_distance") or 0) if dist_rows else 0
    )
    checkpoint(
        "ANALYTICS",
        "trip_distance",
        trips=with_trip,
        distance=total_distance,
        days=days,
    )
    return {
        "analytics_type": "trip_distance",
        "days": days,
        "total_orders": total_orders,
        "orders_with_tripno": with_trip,
        "orders_with_distance": orders_with_distance,
        "total_distance": round(total_distance, 4),
        "definition": (
            "Trip count = orders with non-empty tripno. "
            "Distance = sum of numeric distance fields"
            + (f" in last ~{days} days." if days else " across all orders.")
        ),
        "response_format": "trip count and/or total distance numbers only from this result",
    }


def status_summary(
    *,
    status_field: str = "orderstatus",
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Count orders per status field across ALL records."""
    field = status_field if status_field in (
        "orderstatus",
        "accountingstatus",
        "outstatus",
    ) else "orderstatus"
    known = {
        "orderstatus": KNOWN_STATUSES,
        "accountingstatus": KNOWN_ACCOUNTING_STATUSES,
        "outsource": KNOWN_OUTSOURCE_STATUSES,
        "outstatus": KNOWN_OUTSOURCE_STATUSES,
    }.get(field, KNOWN_STATUSES)

    collection = get_orders_collection()
    group_field = f"${field}"
    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": {"_id": group_field, "order_count": {"$sum": 1}}},
                {"$sort": {"order_count": -1}},
            ]
        )
    )
    by_status = {
        (r["_id"] if r["_id"] not in (None, "") else "Unknown"): int(r["order_count"])
        for r in rows
    }
    total = sum(by_status.values())
    ordered = []
    for name in known:
        ordered.append({"status": name, "order_count": by_status.get(name, 0)})
    for name, count in by_status.items():
        if name not in known:
            ordered.append({"status": name, "order_count": count})

    matching = total
    if status:
        matching = 0
        for k, v in by_status.items():
            if str(k).lower() == status.lower():
                matching = v
                break

    checkpoint(
        "ANALYTICS",
        "status_summary",
        total_orders=total,
        statuses=len(ordered),
        status_field=field,
        status=status,
    )
    return {
        "analytics_type": "status_summary",
        "status_field": field,
        "status_filter": status,
        "matching_orders": matching if status else total,
        "total_orders": total,
        "by_status": ordered,
        "definition": (
            f"Counts from all Avaal order records grouped by {field}. "
            "Order status: Quoted/Confirmed/Dispatched/Started/In-Transit/"
            "Partially Delivered/Delivered/Cancelled/Rejected. "
            "Outsource: Open/Planned/Assigned/Quoted/Delivered. "
            "Accounting: Invoiced/PartiallyPaid/Paid/Restricted."
        ),
    }


def best_customers(
    *,
    metric: str = "orders",
    limit: int = 5,
    direction: str = "best",
) -> Dict[str, Any]:
    """
    Rank customers across ALL records.
    direction="best"  -> highest (most orders / most revenue)
    direction="worst" -> lowest  (fewest orders / least revenue)
    metric="revenue" uses SUM(grosstotalfreight), else order count.
    """
    limit = max(1, min(int(limit or 5), 20))
    direction = "worst" if str(direction).lower() == "worst" else "best"
    order = 1 if direction == "worst" else -1  # asc for worst, desc for best
    label = "worst" if direction == "worst" else "best"
    collection = get_orders_collection()

    group = {
        "_id": "$customername",
        "order_count": {"$sum": 1},
        "total_revenue": {"$sum": "$grosstotalfreight"},
        "total_freight": {"$sum": "$totalfreight"},
    }

    if metric == "revenue":
        sort = {"total_revenue": order, "order_count": order}
        definition = (
            f"{label.capitalize()} customer by "
            f"{'lowest' if direction == 'worst' else 'highest'} total revenue "
            "(sum of grosstotalfreight) across all orders."
        )
    else:
        sort = {"order_count": order, "total_revenue": order}
        definition = (
            f"{label.capitalize()} customer = customer with the "
            f"{'minimum' if direction == 'worst' else 'maximum'} number of orders "
            "across all records. Ties broken by revenue."
        )

    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": group},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": sort},
                {"$limit": limit},
            ]
        )
    )
    customers = []
    for r in rows:
        customers.append(
            {
                "customername": r.get("_id"),
                "order_count": int(r.get("order_count") or 0),
                "total_revenue": round(float(r.get("total_revenue") or 0), 4),
                "total_freight": round(float(r.get("total_freight") or 0), 4),
            }
        )

    checkpoint(
        "ANALYTICS",
        "best_customers",
        direction=direction,
        metric=metric,
        top=customers[0]["customername"] if customers else None,
        limit=limit,
    )
    return {
        "analytics_type": "best_customer",
        "direction": direction,
        "metric": metric,
        "definition": definition,
        "customers": customers,
        "best_customer": customers[0] if customers else None,
    }


def customers_by_country(
    country: str,
    *,
    location_side: str = "both",
) -> Dict[str, Any]:
    """
    Count distinct customers whose pickup and/or delivery address mentions country.
    Uses pickupfulladdress + deliveryfulladdress (and location name fallbacks).
    """
    collection = get_orders_collection()
    rx = _country_regex(country)
    address_or: List[Dict[str, Any]] = []
    if location_side in ("pickup", "both"):
        address_or.extend(
            [
                {"pickupfulladdress": {"$regex": rx, "$options": "i"}},
                {"pickuplocationname": {"$regex": rx, "$options": "i"}},
            ]
        )
    if location_side in ("delivery", "both"):
        address_or.extend(
            [
                {"deliveryfulladdress": {"$regex": rx, "$options": "i"}},
                {"deliverylocationname": {"$regex": rx, "$options": "i"}},
            ]
        )

    match = {**_base_match(), "$or": address_or}
    order_count = collection.count_documents(match)

    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": "$customername",
                        "order_count": {"$sum": 1},
                    }
                },
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": {"order_count": -1}},
                {"$limit": 25},
            ]
        )
    )
    customers = [
        {"customername": r["_id"], "order_count": int(r["order_count"])} for r in rows
    ]
    distinct_rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": "$customername"}},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$count": "n"},
            ]
        )
    )
    distinct_customers = int(distinct_rows[0]["n"]) if distinct_rows else 0

    side_label = {
        "pickup": "pickup address only",
        "delivery": "delivery/drop address only",
        "both": "pickup OR delivery/drop address",
    }.get(location_side, "pickup OR delivery/drop address")

    checkpoint(
        "ANALYTICS",
        "customers_by_country",
        country=country,
        side=location_side,
        customers=distinct_customers,
        orders=order_count,
    )
    return {
        "analytics_type": "customers_by_country",
        "country": country,
        "location_side": location_side,
        "location_rule": side_label,
        "definition": (
            f"Distinct customers with {country} found in {side_label} "
            f"(matched on pickupfulladdress/deliveryfulladdress and location names)."
        ),
        "distinct_customers": distinct_customers,
        "matching_orders": order_count,
        "sample_customers": customers[:10],
    }


def activity_on_date(
    date_prefix: str,
    *,
    date_field: str = "orderdate",
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    On a given calendar day: order count + distinct customers (from ALL matching records).
    Dates in DB look like 2026-08-06T05:20:03.043+05:30 — match by YYYY-MM-DD prefix.
    Optional status filter (Confirmed / Dispatched / …).
    """
    if date_field not in ("orderdate", "pickupdate", "deliverydate"):
        date_field = "orderdate"
    date_prefix = normalize_date_prefix(date_prefix) or date_prefix

    collection = get_orders_collection()
    match = {
        **_base_match(),
        date_field: {"$regex": f"^{re.escape(date_prefix)}", "$options": "i"},
    }
    if status:
        match["orderstatus"] = {
            "$regex": f"^{re.escape(status)}$",
            "$options": "i",
        }
    order_count = collection.count_documents(match)

    rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": "$customername",
                        "order_count": {"$sum": 1},
                    }
                },
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": {"order_count": -1}},
                {"$limit": 25},
            ]
        )
    )
    customers = [
        {"customername": r["_id"], "order_count": int(r["order_count"])} for r in rows
    ]
    distinct_rows = list(
        collection.aggregate(
            [
                {"$match": match},
                {"$group": {"_id": "$customername"}},
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$count": "n"},
            ]
        )
    )
    distinct_customers = int(distinct_rows[0]["n"]) if distinct_rows else 0

    # Status breakdown for the day (always useful for "today" questions)
    status_rows = list(
        collection.aggregate(
            [
                {
                    "$match": {
                        **_base_match(),
                        date_field: {
                            "$regex": f"^{re.escape(date_prefix)}",
                            "$options": "i",
                        },
                    }
                },
                {"$group": {"_id": "$orderstatus", "order_count": {"$sum": 1}}},
                {"$sort": {"order_count": -1}},
            ]
        )
    )
    by_status = [
        {
            "status": (r["_id"] if r["_id"] not in (None, "") else "Unknown"),
            "order_count": int(r["order_count"]),
        }
        for r in status_rows
    ]

    field_label = {
        "orderdate": "order date",
        "pickupdate": "pickup date",
        "deliverydate": "delivery/drop date",
    }.get(date_field, "order date")

    checkpoint(
        "ANALYTICS",
        "activity_on_date",
        date=date_prefix,
        field=date_field,
        status=status,
        customers=distinct_customers,
        orders=order_count,
    )
    return {
        "analytics_type": "activity_on_date",
        "date": date_prefix,
        "date_field": date_field,
        "date_field_label": field_label,
        "status_filter": status,
        "definition": (
            f"Orders and distinct customers on {date_prefix} using {field_label}"
            + (f", status={status}" if status else "")
            + " across all Avaal order records."
        ),
        "response_format": (
            "Report matching_orders for the asked day"
            + (" and status" if status else "")
            + "; optional by_status breakdown."
        ),
        "distinct_customers": distinct_customers,
        "matching_orders": order_count,
        "by_status": by_status,
        "sample_customers": customers[:10],
    }


def run_analytics(
    question: str,
    *,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Route analytics question → Mongo aggregation payload."""
    entities = entities or {}
    q = question or ""
    side = entities.get("location_side") or detect_location_side(q)

    # Date-based activity (customers/orders on a day) — dynamic from full DB
    date_val = (
        entities.get("analytics_date")
        or entities.get("orderdate")
        or entities.get("pickupdate")
        or entities.get("deliverydate")
        or extract_any_date_from_question(q)
    )
    if date_val and (
        is_date_activity_question(q)
        or is_today_orders_question(q)
        or entities.get("analytics") == "activity_on_date"
        or (
            date_val
            and re.search(
                r"\b(customer|order|kitne|kitna|how many|count|total|created|confirm|confirmed|dispatched)\b",
                q,
                re.I,
            )
        )
    ):
        date_field = entities.get("date_field") or detect_date_field(q)
        if entities.get("pickupdate") and not entities.get("orderdate"):
            date_field = "pickupdate"
        if entities.get("deliverydate") and not entities.get("orderdate"):
            date_field = "deliverydate"
        status = entities.get("orderstatus") or detect_status_filter(q)
        # "today created" / "aaj create" = all statuses that day
        if status and re.search(r"\b(created|create|new)\b", q, re.I) and not re.search(
            r"\b(quoted|cancelled|canceled|confirmed|confirm|dispatched|started|"
            r"in[- ]?transit|partially\s*delivered|delivered|rejected|invoiced|"
            r"paid|restricted|open|planned|assigned)\b",
            q,
            re.I,
        ):
            status = None
        return activity_on_date(str(date_val), date_field=date_field, status=status)

    # Last N days / last month (+ optional status like Quoted / Confirmed)
    period_days = entities.get("period_days") or detect_period_days(q)
    if period_days or entities.get("analytics") == "orders_in_period":
        days = int(period_days or 30)
        status = entities.get("orderstatus") or detect_status_filter(q)
        # Only apply status filter when user clearly asked status/quoted/confirmed count
        if status and not re.search(
            r"\b(quoted|quotes|quote|cancelled|canceled|confirmed|confirmation|confirmations|"
            r"confirm|dispatched|started|in[- ]?transit|partially\s*delivered|delivered|"
            r"rejected|invoiced|paid|restricted|open|planned|assigned|status)\b",
            q,
            re.I,
        ):
            status = None
        date_field = entities.get("date_field") or detect_date_field(q)
        return orders_in_period(days=days, status=status, date_field=date_field)

    if is_worst_order_question(q) or entities.get("analytics") == "worst_order":
        limit = int(entities.get("limit") or 5)
        return rank_orders_by_amount(direction="worst", limit=limit)

    if is_best_order_question(q) or entities.get("analytics") == "best_order":
        limit = int(entities.get("limit") or 5)
        return rank_orders_by_amount(direction="best", limit=limit)

    if is_best_city_question(q) or entities.get("analytics") == "best_city":
        limit = int(entities.get("limit") or 5)
        return best_cities(location_side=side, limit=limit)

    if is_state_wise_question(q) or entities.get("analytics") == "orders_by_state":
        limit = int(entities.get("limit") or 50)
        return orders_by_state(location_side=side, limit=limit)

    if is_city_wise_question(q) or entities.get("analytics") == "orders_by_city":
        limit = int(entities.get("limit") or 50)
        return orders_by_city(location_side=side, limit=limit)

    if is_orders_by_country_question(q) or entities.get("analytics") == "orders_by_country":
        limit = int(entities.get("limit") or 50)
        return orders_by_country(location_side=side, limit=limit)

    if is_trip_distance_question(q) or entities.get("analytics") == "trip_distance":
        days = detect_period_days(q)
        return trip_distance_summary(days=days)

    if is_best_customer_question(q) or entities.get("analytics") in (
        "best_customer",
        "worst_customer",
    ):
        metric = entities.get("best_customer_metric") or "orders"
        if re.search(r"\b(revenue|freight|amount|sales|money|value)\b", q, re.I):
            metric = "revenue"
        direction = detect_customer_direction(q)
        if entities.get("analytics") == "worst_customer":
            direction = "worst"
        if entities.get("customer_direction") in ("best", "worst"):
            direction = entities["customer_direction"]
        limit = int(entities.get("limit") or 5)
        return best_customers(metric=metric, limit=limit, direction=direction)

    if is_country_customer_question(q) or (
        entities.get("country") and not date_val and not period_days
    ):
        country = entities.get("country") or detect_country(q) or "Canada"
        return customers_by_country(country, location_side=side)

    if is_status_summary_question(q) or entities.get("analytics") == "status_summary":
        field = (
            "accountingstatus"
            if entities.get("accountingstatus")
            else "outstatus"
            if entities.get("outstatus")
            else detect_status_field(q)
        )
        status = (
            entities.get("accountingstatus")
            or entities.get("outstatus")
            or entities.get("orderstatus")
            or detect_accounting_status_filter(q)
            or detect_outsource_status_filter(q)
            or detect_status_filter(q)
        )
        # If user asked a specific status type, prefer that field
        if entities.get("accountingstatus"):
            field = "accountingstatus"
        elif entities.get("outstatus") and not entities.get("orderstatus"):
            field = "outstatus"
        elif entities.get("orderstatus"):
            field = "orderstatus"
        return status_summary(status_field=field, status=status)

    # Fallback: if analytics tool was forced, prefer status summary
    return status_summary(status_field=detect_status_field(q))


def format_analytics_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text block for LLM — ground truth, do not invent."""
    lines = ["ANALYTICS RESULT (exact engine — do not recalculate or invent):"]
    atype = payload.get("analytics_type")
    if payload.get("definition"):
        lines.append(f"Definition: {payload['definition']}")
    if payload.get("response_format"):
        lines.append(f"Response format required: {payload['response_format']}")

    if atype == "status_summary":
        lines.append(f"status_field: {payload.get('status_field') or 'orderstatus'}")
        lines.append(f"total_orders: {payload.get('total_orders')}")
        if payload.get("status_filter"):
            lines.append(f"status_filter: {payload.get('status_filter')}")
            lines.append(f"matching_orders: {payload.get('matching_orders')}")
        lines.append("by_status:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('order_count')}")

    elif atype == "best_customer":
        direction = payload.get("direction") or "best"
        lines.append(f"direction: {direction} (best=highest, worst=lowest)")
        lines.append(f"metric: {payload.get('metric')}")
        best = payload.get("best_customer") or {}
        pick_label = "worst_customer" if direction == "worst" else "best_customer"
        if best:
            lines.append(
                f"{pick_label}: {best.get('customername')} | "
                f"order_count={best.get('order_count')} | "
                f"total_revenue={best.get('total_revenue')} | "
                f"total_freight={best.get('total_freight')}"
            )
        list_label = "bottom_customers" if direction == "worst" else "top_customers"
        lines.append(f"{list_label}:")
        for i, row in enumerate(payload.get("customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')} | "
                f"revenue={row.get('total_revenue')}"
            )

    elif atype == "customers_by_country":
        lines.append(f"country: {payload.get('country')}")
        lines.append(f"location_rule: {payload.get('location_rule')}")
        lines.append(f"distinct_customers: {payload.get('distinct_customers')}")
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        lines.append("sample_customers:")
        for i, row in enumerate(payload.get("sample_customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')}"
            )

    elif atype == "activity_on_date":
        lines.append(f"date: {payload.get('date')}")
        lines.append(f"date_field: {payload.get('date_field')} ({payload.get('date_field_label')})")
        lines.append(f"status_filter: {payload.get('status_filter')}")
        lines.append(f"distinct_customers: {payload.get('distinct_customers')}")
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        if payload.get("by_status"):
            lines.append("by_status_on_date:")
            for row in payload.get("by_status") or []:
                lines.append(f"- {row.get('status')}: {row.get('order_count')}")
        lines.append("sample_customers:")
        for i, row in enumerate(payload.get("sample_customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('customername')} | orders={row.get('order_count')}"
            )
        if not payload.get("sample_customers") and not payload.get("matching_orders"):
            lines.append("(no customers/orders found on this date)")

    elif atype == "orders_by_state":
        lines.append("state_wise_counts (country then state then count ONLY):")
        for row in payload.get("rows") or []:
            lines.append(
                f"- country={row.get('country')} | state={row.get('state')} | "
                f"order_count={row.get('order_count')}"
            )
        if not payload.get("rows"):
            lines.append("(no state address data found)")

    elif atype == "orders_by_city":
        lines.append("city_wise_counts (city then count ONLY):")
        for row in payload.get("rows") or []:
            lines.append(
                f"- city={row.get('city')} | state={row.get('state')} | "
                f"country={row.get('country')} | order_count={row.get('order_count')}"
            )
        if not payload.get("rows"):
            lines.append("(no city address data found)")

    elif atype == "orders_by_country":
        lines.append("country_wise_counts (country then count ONLY):")
        for row in payload.get("rows") or []:
            lines.append(f"- country={row.get('country')} | order_count={row.get('order_count')}")
        if not payload.get("rows"):
            lines.append("(no country address data found)")

    elif atype in ("best_order", "worst_order"):
        lines.append(f"direction: {payload.get('direction')}")
        lines.append(f"total_scored: {payload.get('total_scored')}")
        top = payload.get("top") or {}
        if top:
            lines.append(
                f"top_order: ordernumber={top.get('ordernumber')} "
                f"amount={top.get('totalfreight')} status={top.get('orderstatus')} "
                f"customer={top.get('customername')}"
            )
        lines.append("ranked_orders:")
        for i, row in enumerate(payload.get("rows") or [], start=1):
            lines.append(
                f"[{i}] ordernumber={row.get('ordernumber')} "
                f"amount={row.get('totalfreight')} status={row.get('orderstatus')} "
                f"customer={row.get('customername')}"
            )

    elif atype == "best_city":
        best = payload.get("best_city") or {}
        if best:
            lines.append(
                f"best_city: {best.get('city')} | state={best.get('state')} | "
                f"country={best.get('country')} | order_count={best.get('order_count')}"
            )
        lines.append("top_cities:")
        for i, row in enumerate(payload.get("top_cities") or [], start=1):
            lines.append(
                f"[{i}] city={row.get('city')} | orders={row.get('order_count')}"
            )

    elif atype == "orders_in_period":
        lines.append(f"period_days: {payload.get('days')}")
        lines.append(f"period_start: {payload.get('period_start')}")
        lines.append(f"date_field: {payload.get('date_field')}")
        lines.append(f"status_filter: {payload.get('status_filter')}")
        lines.append(f"matching_orders: {payload.get('matching_orders')}")
        lines.append("by_status_in_period:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('order_count')}")

    elif atype == "trip_distance":
        lines.append(f"days: {payload.get('days')}")
        lines.append(f"total_orders: {payload.get('total_orders')}")
        lines.append(f"orders_with_tripno: {payload.get('orders_with_tripno')}")
        lines.append(f"orders_with_distance: {payload.get('orders_with_distance')}")
        lines.append(f"total_distance: {payload.get('total_distance')}")

    else:
        lines.append(str(payload))

    lines.append("Use these values as ground truth. Explain clearly. Numbers without commas.")
    return "\n".join(lines)
