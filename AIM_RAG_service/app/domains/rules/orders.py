"""Order domain rules — Avaal_order fields (existing order_ask logic)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.domains.rules.base import DomainRules, extract_limit
from app.domains.lookup.orders.lookup import extract_token as extract_order_token
from app.domains.lookup.base import is_ask_for_record_id_question
from app.order_ask.calculation_engine import is_calculation_question

DOMAIN = "orders"

STATUS_MAP = {
    # Order status (orderstatus)
    "confirmed": "Confirmed",
    "confirm": "Confirmed",
    "delivered": "Delivered",
    "dispatched": "Dispatched",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "quoted": "Quoted",
    "quated": "Quoted",  # common typo
    "started": "Started",
    "in-transit": "In-Transit",
    "in transit": "In-Transit",
    "intransit": "In-Transit",
    "partially delivered": "Partially Delivered",
    "partiallydelivered": "Partially Delivered",
    "partial delivered": "Partially Delivered",
    "rejected": "Rejected",
    "reject": "Rejected",
}

# Accounting status (accountingstatus) — NOT orderstatus
ACCOUNTING_STATUS_MAP = {
    "invoice restricted": "Restricted",
    "invoiced restricted": "Restricted",
    "restricted": "Restricted",
    "partially paid": "PartiallyPaid",
    "partiallypaid": "PartiallyPaid",
    "partial paid": "PartiallyPaid",
    "invoiced": "Invoiced",
    "paid": "Paid",
}

# Order outsource status (outstatus / outsourcedetails.outStatus)
OUTSOURCE_STATUS_MAP = {
    "planned": "Planned",
    "assigned": "Assigned",
    "open": "Open",
}

_ALL_STATUS_STOPWORDS = frozenset(
    STATUS_MAP.keys()
    | ACCOUNTING_STATUS_MAP.keys()
    | OUTSOURCE_STATUS_MAP.keys()
    | {
        "invoiced",
        "paid",
        "restricted",
        "started",
        "rejected",
        "assigned",
        "planned",
        "open",
    }
)

GEO_FILTER_KEYS = (
    "pin",
    "state",
    "city",
    "address",
    "location",
    "pickup_location",
    "delivery_location",
)

LIST_RE = re.compile(
    r"\b(list|show|display|find|search|filter|which|all|some|any|give|get)\b.*\border",
    re.I,
)
RECENT_RE = re.compile(
    r"\b(recent|recently|latest|last\s+\d+|top\s+\d+|some|any|few|"
    r"only\s+\d+|just\s+\d+|newly\s+create[d]?|create[d]?\s+recently)\b.*\border|"
    r"\border.*\b(recent|recently|latest|newly\s+create[d]?|create[d]?\s+recently|"
    r"only\s+\d+|just\s+\d+)\b",
    re.I,
)
COMPARE_RE = re.compile(r"\b(compare|difference|vs\.?|versus)\b", re.I)

STICKY_KEYS = (
    "order_token",
    "customername",
    "customercode",
    "orderstatus",
    "accountingstatus",
    "outstatus",
    "statuscode",
    "currencycode",
    "companycode",
    "pickup_location",
    "delivery_location",
    *GEO_FILTER_KEYS,
    "location_side",
    "salesmanname",
    "commodityname",
    "analytics",
    "limit",
    "focus_fields",
)


def _extract_status_entities(ql: str) -> Dict[str, str]:
    """
    Map question text → orderstatus / accountingstatus / outstatus.
    UI statuses:
      Order: Quoted, Confirmed, Dispatched, Started, In-Transit,
             Partially Delivered, Delivered, Cancelled, Rejected
      Outsource: Open, Planned, Assigned, Delivered, Quoted
      Accounting: Invoiced, PartiallyPaid, Paid, Restricted
    """
    out: Dict[str, str] = {}
    wants_outsource = bool(
        re.search(r"\b(out\s*source|outsource|out\s*status|outstatus)\b", ql)
    )
    wants_accounting = bool(
        re.search(r"\b(accounting|account\s*status)\b", ql)
    )

    # Accounting (Invoiced lives here — not on orderstatus)
    for key, canonical in sorted(
        ACCOUNTING_STATUS_MAP.items(), key=lambda kv: -len(kv[0])
    ):
        if not re.search(rf"\b{re.escape(key)}\b", ql):
            continue
        if canonical in ("Invoiced", "PartiallyPaid", "Restricted"):
            out["accountingstatus"] = canonical
            break
        if canonical == "Paid" and (
            wants_accounting
            or re.search(
                r"\b(paid\s+orders?|orders?\s+(?:that\s+are\s+)?paid|"
                r"how many\s+paid|kitne\s+paid|paid\s+kitne|"
                r"accounting\s+paid|paid\s+accounting)\b",
                ql,
            )
        ):
            out["accountingstatus"] = "Paid"
            break

    # Outsource Open / Planned / Assigned (or Delivered/Quoted with outsource cue)
    if wants_outsource:
        if re.search(r"\bpartially\s*delivered\b", ql):
            pass  # not an outsource label
        elif re.search(r"\bdelivered\b", ql):
            out["outstatus"] = "Delivered"
        elif re.search(r"\bquoted\b", ql):
            out["outstatus"] = "Quoted"
        elif re.search(r"\bassigned\b", ql):
            out["outstatus"] = "Assigned"
        elif re.search(r"\bplanned\b", ql):
            out["outstatus"] = "Planned"
        elif re.search(r"\bopen\b", ql):
            out["outstatus"] = "Open"
    else:
        for key, canonical in OUTSOURCE_STATUS_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", ql):
                out["outstatus"] = canonical
                break

    # Order lifecycle status (skip when only accounting/outsource was asked)
    if "accountingstatus" in out and not re.search(
        r"\b(quoted|confirmed|dispatched|started|in[- ]?transit|"
        r"partially\s*delivered|delivered|cancelled|canceled|rejected)\b",
        ql,
    ):
        return out
    if wants_outsource and "outstatus" in out:
        return out

    for key, canonical in sorted(STATUS_MAP.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", ql):
            out["orderstatus"] = canonical
            break
    return out


def extract_record_token(question: str) -> Optional[str]:
    return extract_order_token(question)


def extract_entities(
    question: str,
    *,
    session_order_token: Optional[str] = None,
    session_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Order entity extraction (Avaal_order document fields)."""
    q = question or ""
    ql = q.lower()
    entities: Dict[str, Any] = {}

    token = extract_order_token(q)
    if token:
        entities["order_token"] = token

    entities.update(_extract_status_entities(ql))

    m = re.search(r"\bcurrency\s*[:=]?\s*([A-Za-z]{3})\b", q, re.I)
    if m:
        entities["currencycode"] = m.group(1).upper()
    else:
        m = re.search(r"\b(CAD|USD)\b", q)
        if m:
            entities["currencycode"] = m.group(1).upper()

    m = re.search(r"\bcustomer\s*code\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["customercode"] = m.group(1).upper()

    ranking_customer_q = bool(
        re.search(
            r"\b(best|top|worst|lowest|low|least|fewest|smallest|minimum|min|bottom|"
            r"biggest|largest|highest|most|maximum|max)\b.*\bcustomers?\b|"
            r"\bcustomers?\b.*\b(best|top|worst|lowest|low|least|fewest|smallest|"
            r"minimum|min|bottom|most|maximum|max)\b|"
            r"\bhow many\b.*\bcustomers?\b",
            ql,
        )
    )
    m = re.search(
        r"\b(?:for\s+customer|customer\s*name|orders?\s+for\s+customer|orders?\s+of\s+customer|"
        r"orders?\s+for|orders?\s+of|customer)\s+['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,60}?)['\"]?(?=\s+(?:with|status|in|and|currency|$)|$)",
        q,
        re.I,
    )
    if m and not ranking_customer_q:
        name = m.group(1).strip(" .,")
        name = re.sub(r"^(?:customer|code)\s+", "", name, flags=re.I).strip()
        if re.match(r"^(with|by|for|of|the|a|an|in|on|from|to)\b", name, re.I):
            name = ""
        if name.lower() not in _ALL_STATUS_STOPWORDS and len(name) >= 2:
            entities["customername"] = name

    m = re.search(r"\bcompany(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)\b", q, re.I)
    if m:
        entities["companycode"] = m.group(1).upper()

    m = re.search(
        r"\b(?:pickup|pick\s*up)\s*(?:location|from)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
        q,
        re.I,
    )
    if m:
        entities["pickup_location"] = m.group(1).strip(" .,")
    if not re.search(r"\bdelivered\b", ql):
        m = re.search(
            r"\b(?:delivery|deliver(?:y)?|drop)\s*(?:location|to)?\s*[:#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
            q,
            re.I,
        )
        if m:
            loc = m.group(1).strip(" .,")
            if re.match(r"^(zip|pin|postal|pincode|code)\b", loc, re.I):
                loc = ""
            if loc.lower() not in _ALL_STATUS_STOPWORDS and not re.match(r"^(ed|y)\b", loc, re.I):
                if loc:
                    entities["delivery_location"] = loc

    is_loc_wise = bool(
        re.search(
            r"\b(?:location|place|facility|warehouse|site|depot)[\s-]*wise\b",
            q,
            re.I,
        )
    )
    if not is_loc_wise:
        m = re.search(
            r"\b(?:at\s+)?(?:location|place|facility|warehouse)\s*[:=#]?\s*['\"]?"
            r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,50})",
            q,
            re.I,
        )
        if m and not entities.get("pickup_location") and not entities.get("delivery_location"):
            loc = m.group(1).strip(" .,")
            # Drop trailing filler ("wise orders", "list", "summary")
            loc = re.sub(
                r"\s+(?:wise|orders?|list|summary|status|breakdown)\b.*$",
                "",
                loc,
                flags=re.I,
            ).strip()
            if (
                loc.lower() not in _ALL_STATUS_STOPWORDS
                and loc.lower() not in {"wise", "order", "orders", "list", "summary"}
                and len(loc) >= 2
            ):
                entities["location"] = loc

    m = re.search(
        r"\b(?:pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\s*[:=#]?\s*"
        r"([A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d|\d{5}(?:-\d{4})?)\b",
        q,
        re.I,
    )
    if m:
        entities["pin"] = re.sub(r"\s+", " ", m.group(1).strip().upper())
    else:
        m = re.search(r"\b(?:in|at|for|with)\s+(\d{5}(?:-\d{4})?)\b", q, re.I)
        if m and not re.fullmatch(r"20\d{2}", m.group(1)[:4]):
            entities["pin"] = m.group(1)

    from app.order_ask.field_catalog import find_state_in_text, resolve_state_token

    is_state_wise = bool(
        re.search(
            r"\b(state|province|region)[\s-]*wise\b|\bby\s+(state|province|region)\b",
            q,
            re.I,
        )
    )

    m = None
    if not is_state_wise:
        m = re.search(
            r"\b(?:state|province|region)\s*(?:is|=|:|#)?\s*['\"]?([A-Za-z][A-Za-z\s.]{1,30})",
            q,
            re.I,
        )
    if m:
        raw_state = m.group(1).strip(" .,")
        # Drop leading articles: "the Ontario" / "province the ON"
        raw_state = re.sub(r"^(?:the|a|an)\s+", "", raw_state, flags=re.I).strip()
        # Drop trailing filler: "Ontario wise", "CA confirmed orders"
        raw_state = re.sub(
            r"\s+(?:wise|orders?|list|summary|status|confirmed|delivered|"
            r"dispatched|quoted|cancelled|pending)\b.*$",
            "",
            raw_state,
            flags=re.I,
        ).strip()
        code = resolve_state_token(raw_state) or (
            resolve_state_token(raw_state.split()[0]) if raw_state.split() else None
        )
        if code:
            entities["state"] = code
        elif raw_state.lower() in {"wise", "order", "orders", "list", "summary", ""}:
            pass
        elif len(raw_state) <= 3:
            entities["state"] = raw_state.upper()
        else:
            entities["state"] = raw_state.title()
    else:
        # "in the Ontario" / "from CA" — skip articles after preposition
        m = re.search(
            r"\b(?:in|to|from|at|for)\s+(?:the\s+|a\s+|an\s+)?"
            r"([A-Za-z]{2}|[A-Za-z][A-Za-z\s]{2,24})\b",
            q,
            re.I,
        )
        if m:
            cand = m.group(1).strip()
            if cand.lower() not in (
                "the", "us", "usa", "canada", "customer", "orders", "order",
                "status", "confirmed", "delivered", "dispatched", "cancelled",
                "quoted", "invoiced", "summary", "list",
            ):
                code = resolve_state_token(cand)
                if code:
                    entities["state"] = code

    # Fallback: scan full question for known state/province names (incl. typos)
    if not entities.get("state"):
        code = find_state_in_text(q)
        if code:
            entities["state"] = code

    is_city_wise = bool(
        re.search(r"\b(city|town)[\s\-]*wise\b|\bby\s+(city|town)\b", q, re.I)
    )
    m = None
    # "city is Toronto" / "city Toronto" — skip when "city wise" (would eat "wise")
    if not is_city_wise:
        m = re.search(
            r"\b(?:city|town)\s*(?:is|=|:|#)\s*['\"]?([A-Za-z][A-Za-z\s\-.]{1,40})",
            q,
            re.I,
        )
        if not m:
            m = re.search(
                r"\b(?:city|town)\s+([A-Za-z][A-Za-z\s\-.]{1,40})",
                q,
                re.I,
            )
    # Always allow particular city: "in Toronto", "for Toronto", "of Chicago"
    # (even with "city wise status for Toronto")
    if not m:
        m = re.search(
            r"\b(?:in|from|at|for|of)\s+(?:the\s+|a\s+|an\s+)?"
            r"([A-Za-z][A-Za-z\-.]{2,40})(?:\s+city)?\b",
            q,
            re.I,
        )
    # Leading city: "Toronto status", "Chicago orders"
    if not m:
        m = re.search(
            r"^([A-Za-z][A-Za-z\-.]{2,40})\s+(?:status|summary|orders?|list)\b",
            q.strip(),
            re.I,
        )
    if m:
        city = m.group(1).strip(" .,")
        # Prefer state resolution first — don't treat Ontario as a city
        from app.order_ask.field_catalog import resolve_state_token

        if resolve_state_token(city):
            pass
        elif city.lower() not in {
            "wise", "wise order", "wise orders", "order", "orders",
            "the", "status", "summary", "list", "canada", "us", "usa",
            "confirmed", "quoted", "delivered", "dispatched", "ontario",
            "particular", "one", "this", "that", "each", "every",
        } and len(city) >= 2:
            entities["city"] = city.title() if city.islower() or city.isupper() else city

    m = re.search(
        r"\b(?:full\s+)?(?:address|street)\s*[:=#]?\s*['\"]?([A-Za-z0-9][A-Za-z0-9 &\-./,#]{2,60})",
        q,
        re.I,
    )
    if m:
        addr = m.group(1).strip(" .,")
        if len(addr) >= 3:
            entities["address"] = addr

    m = re.search(
        r"\b(?:salesman|sales\s*person|sales\s*rep)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,40})",
        q,
        re.I,
    )
    if m:
        entities["salesmanname"] = m.group(1).strip(" .,")

    m = re.search(
        r"\b(?:commodity|product)\s*[:=#]?\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9 &\-./,]{1,40})",
        q,
        re.I,
    )
    if m:
        entities["commodityname"] = m.group(1).strip(" .,")

    from app.order_ask.analytics import detect_location_side

    if any(entities.get(k) for k in ("pin", "state", "city", "address", "location")):
        entities["location_side"] = detect_location_side(q)

    if entities.get("pin") and entities.get("order_token"):
        pin_norm = re.sub(r"\s+", "", str(entities["pin"])).upper()
        tok_norm = re.sub(r"\s+", "", str(entities["order_token"])).upper()
        if pin_norm == tok_norm or tok_norm in pin_norm:
            entities.pop("order_token", None)

    from app.order_ask.analytics import (
        detect_date_field,
        detect_period_days,
        extract_any_date_from_question,
        is_best_city_question,
        is_city_wise_question,
        is_location_wise_question,
        is_date_activity_question,
        is_period_orders_question,
        is_state_wise_question,
        is_trip_distance_question,
        normalize_date_prefix,
        detect_country,
        detect_customer_direction,
        is_best_customer_question,
    )

    date_val = extract_any_date_from_question(q)
    if not date_val:
        m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", q)
        if m:
            date_val = normalize_date_prefix(m.group(1))
        else:
            m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b", q)
            if m:
                date_val = normalize_date_prefix(m.group(1))

    if date_val:
        date_field = detect_date_field(q)
        entities["analytics_date"] = date_val
        entities["date_field"] = date_field
        if date_field == "pickupdate":
            entities["pickupdate"] = date_val
        elif date_field == "deliverydate":
            entities["deliverydate"] = date_val
        else:
            entities["orderdate"] = date_val
        if is_date_activity_question(q) or re.search(
            r"\b(customer|kitne|kitna|how many|count|ordered|created|total)\b", ql
        ):
            entities["analytics"] = "activity_on_date"

    period_days = detect_period_days(q)
    if period_days:
        entities["period_days"] = period_days
        if is_period_orders_question(q):
            entities["analytics"] = "orders_in_period"

    if is_state_wise_question(q):
        entities["analytics"] = "orders_by_state"
    if is_city_wise_question(q):
        entities["analytics"] = "orders_by_city"
    if is_location_wise_question(q):
        entities["analytics"] = "orders_by_location"
    # Particular place + status/summary → filtered status_summary (not city-wise list)
    if (entities.get("city") or entities.get("state") or entities.get("country")) and re.search(
        r"\b(status|summary|how\s+many|count|breakdown)\b", ql
    ):
        entities["analytics"] = "status_summary"
    if is_best_city_question(q):
        entities["analytics"] = "best_city"
    if is_trip_distance_question(q):
        entities["analytics"] = "trip_distance"

    from app.order_ask.analytics import (
        is_best_order_question,
        is_orders_by_country_question,
        is_today_orders_question,
        is_worst_order_question,
    )

    if is_orders_by_country_question(q):
        entities["analytics"] = "orders_by_country"
    if is_worst_order_question(q):
        entities["analytics"] = "worst_order"
    elif is_best_order_question(q):
        entities["analytics"] = "best_order"
    if is_today_orders_question(q):
        entities["analytics"] = "activity_on_date"
        if not entities.get("analytics_date"):
            entities["analytics_date"] = extract_any_date_from_question(q)
            entities["date_field"] = entities.get("date_field") or "orderdate"
            entities["orderdate"] = entities.get("analytics_date")

    # Focus fields for exact-order follow-ups
    focus: List[str] = []
    if re.search(r"\bpick\s*up\b.*\b(address|location)|\bpickup(fulladdress|locationname)\b", ql):
        focus.extend(["pickupfulladdress", "pickuplocationname", "pickupdate"])
    if re.search(r"\bdeliver(?:y|y)?\b.*\b(address|location)|\bdelivery(fulladdress|locationname)\b", ql):
        focus.extend(["deliveryfulladdress", "deliverylocationname", "deliverydate"])
    if re.search(r"\bstatus\s*code\b|\bstatuscode\b", ql):
        focus.append("statuscode")
    if re.search(r"\baccounting\s*status\b|\baccountingstatus\b", ql):
        focus.append("accountingstatus")
    if re.search(r"\bout\s*source\s*status\b|\boutstatus\b|\bout\s*status\b", ql):
        focus.append("outstatus")
    if re.search(r"\b(order\s*)?status\b", ql):
        focus.append("orderstatus")
        if "accountingstatus" not in focus:
            focus.append("accountingstatus")
        if "outstatus" not in focus:
            focus.append("outstatus")
    if focus:
        entities["focus_fields"] = list(dict.fromkeys(focus))

    if re.search(
        r"\b(best|highest|max|top|largest|most)\b.*\b(amount|freight|revenue|tax|distance)\b", ql
    ) or re.search(
        r"\b(amount|freight|revenue|tax|distance)\b.*\b(best|highest|max|top|largest|most)\b",
        ql,
    ):
        entities["limit"] = entities.get("limit") or 5
        if "tax" in ql:
            entities["sort_by"] = "taxes"
        elif "distance" in ql:
            entities["sort_by"] = "distance"
        elif "revenue" in ql or "gross" in ql:
            entities["sort_by"] = "grosstotalfreight"
        else:
            entities["sort_by"] = "totalfreight"
        entities["ascending"] = False

    if re.search(r"\b(lowest|smallest|least|cheapest|min)\b.*\b(amount|freight|tax|distance)\b", ql):
        entities["limit"] = entities.get("limit") or 5
        if "tax" in ql:
            entities["sort_by"] = "taxes"
        elif "distance" in ql:
            entities["sort_by"] = "distance"
        else:
            entities["sort_by"] = "totalfreight"
        entities["ascending"] = True

    limit = extract_limit(q, default_all=25, some_default=10)
    if limit:
        entities["limit"] = limit
    elif re.search(r"\b(all|every)\b.*\border", ql):
        entities["limit"] = 25
    elif re.search(r"\b(some|any)\b.*\border|\border.*\b(some|any)\b", ql):
        entities["limit"] = entities.get("limit") or 10

    country = detect_country(q)
    if country:
        entities["country"] = country
        entities["location_side"] = detect_location_side(q)

    if is_best_customer_question(q):
        direction = detect_customer_direction(q)
        entities["customer_direction"] = direction
        entities["analytics"] = "worst_customer" if direction == "worst" else "best_customer"
        if re.search(r"\b(revenue|freight|amount|sales|money|value)\b", ql):
            entities["best_customer_metric"] = "revenue"
        else:
            entities["best_customer_metric"] = "orders"

    if re.search(r"\bstatus\b.*\b(summary|break|count|how many)\b|\b(summary)\b.*\bstatus\b", ql):
        entities["analytics"] = "status_summary"
    elif re.search(
        r"\b(how many|count)\b.*\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b",
        ql,
    ) or re.search(
        r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b.*\b(how many|count)\b",
        ql,
    ):
        entities["analytics"] = "status_summary"

    from app.domains.rules.base import is_follow_up, merge_session_entities

    sticky = session_entities or {}
    follow_up = is_follow_up(q)
    # Aggregate / "X wise" questions are fleet-wide — never inherit a single order token.
    aggregate_q = bool(entities.get("analytics")) or bool(
        re.search(r"\bwise\b|\ball\s+orders?\b|\bevery\s+order\b", ql)
    )
    if not aggregate_q and (
        follow_up or (not entities.get("order_token") and session_order_token)
    ):
        if not entities.get("order_token") and session_order_token:
            if follow_up or re.search(
                r"\b(status|tax|freight|detail|details|info|amount|customer|delivery|pickup|distance|location)\b",
                ql,
            ):
                entities["order_token"] = session_order_token
                entities["from_session"] = True

    if follow_up:
        for key in STICKY_KEYS:
            if key not in entities and sticky.get(key):
                entities[key] = sticky[key]
                entities["from_session"] = True

    return entities


def entities_to_mongo_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for key in (
        "orderstatus",
        "accountingstatus",
        "outstatus",
        "currencycode",
        "customercode",
        "companycode",
        "customername",
        "pickup_location",
        "delivery_location",
        "orderdate",
        "pickupdate",
        "deliverydate",
        "pin",
        "state",
        "city",
        "country",
        "address",
        "location",
        "location_side",
        "salesmanname",
        "commodityname",
    ):
        if entities.get(key):
            filters[key] = entities[key]
    return filters


def has_list_filters(entities: Dict[str, Any]) -> bool:
    return any(
        entities.get(k)
        for k in (
            "orderstatus",
            "accountingstatus",
            "outstatus",
            "currencycode",
            "customercode",
            "companycode",
            "customername",
            "orderdate",
            "pickupdate",
            "deliverydate",
            "salesmanname",
            "commodityname",
            *GEO_FILTER_KEYS,
        )
    )


def classify_intent_local(
    question: str,
    *,
    history_hint: str = "",
) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()

    # Recent / only-N list BEFORE ask-for-id (avoids "give me 2 orders recently" false positive)
    if RECENT_RE.search(q) and not is_calculation_question(q):
        return {
            "intent": "list_recent",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "list_recent",
        }

    # Specific order details without an id → ask sweetly for order number/id (no list dump).
    if is_ask_for_record_id_question(
        q, domain_noun="orders?", has_token=bool(extract_order_token(q))
    ):
        return {
            "intent": "ask_for_record_id",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "short",
            "max_tokens_hint": 80,
            "retrieve_k": 0,
            "reason": "order_details_without_token",
        }

    from app.order_ask.analytics import (
        is_analytics_question,
        is_best_customer_question,
        is_best_city_question,
        is_best_order_question,
        is_city_wise_question,
        is_location_wise_question,
        is_country_customer_question,
        is_date_activity_question,
        is_orders_by_country_question,
        is_period_orders_question,
        is_state_wise_question,
        is_status_summary_question,
        is_today_orders_question,
        is_trip_distance_question,
        is_worst_order_question,
    )

    if is_today_orders_question(q) or is_date_activity_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "activity_on_date",
        }

    if is_worst_order_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "worst_order",
        }

    if is_best_order_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "best_order",
        }

    if is_orders_by_country_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "orders_by_country",
        }

    token = extract_order_token(q)
    # Asking pickup/delivery/status OF a specific order → lookup, not geo list
    field_of_order_q = bool(
        token
        and re.search(
            r"\b(pickup|delivery|address|status|statuscode|detail|details|"
            r"customer|freight|date)\b",
            q,
            re.I,
        )
    )
    geo_filter_q = bool(
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal|"
            r"state|province|city|town|address|street)\b",
            q,
            re.I,
        )
    ) and not field_of_order_q
    if (
        token
        and not geo_filter_q
        and (
            re.search(
                r"\b(give|get|show|find|lookup|look\s*up|detail|details|info|information|fetch|pull|order\s*number|ordernumber)\b",
                q,
                re.I,
            )
            or re.match(r"^(MRP\d+|TORD\d+|[A-Za-z]{2,6}\d{2,}|\d{4,})\s*$", q, re.I)
            or len(q.split()) <= 10
            or field_of_order_q
        )
    ):
        if not re.search(r"\b(all|list|filter|compare|vs)\b", q, re.I) or re.search(
            r"\b(detail|details|give|get|show)\b", q, re.I
        ):
            if not LIST_RE.search(q) or token:
                return {
                    "intent": "order_lookup",
                    "needs_rag": False,
                    "needs_calculation": False,
                    "needs_exact_order": True,
                    "response_style": "detailed",
                    "max_tokens_hint": 1200,
                    "retrieve_k": 0,
                    "order_token": token,
                    "reason": "explicit_order_token",
                }

    if COMPARE_RE.search(q) and re.search(r"\b(MRP\d+|TORD\d+|\d{4,})\b", q, re.I):
        return {
            "intent": "compare",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 700,
            "retrieve_k": 0,
            "reason": "compare_orders",
        }

    if re.search(
        r"\b(best|highest|top|largest|most|lowest|smallest)\b.*\b(order|amount|freight|tax|distance|revenue)\b",
        q,
        re.I,
    ) and not re.search(r"\bcustomer\b", q, re.I):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "ranked_list",
        }

    if (
        re.search(r"\b(list|show|display|find|search|filter)\b", q, re.I)
        and re.search(
            r"\b(quoted|cancelled|canceled|confirmed|dispatched|delivered|invoiced)\b",
            q,
            re.I,
        )
        and not re.search(r"\b(how many|count|summary|breakdown|break\s*down|total)\b", q, re.I)
    ):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "status_filtered_list",
        }

    if is_best_customer_question(q):
        from app.order_ask.analytics import detect_customer_direction

        direction = detect_customer_direction(q)
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": f"{direction}_customer",
        }

    if is_best_city_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 300,
            "retrieve_k": 0,
            "reason": "best_city",
        }

    if is_state_wise_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "state_wise_orders",
        }

    if is_city_wise_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "city_wise_orders",
        }

    if is_location_wise_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 450,
            "retrieve_k": 0,
            "reason": "location_wise_orders",
        }

    if is_period_orders_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "period_orders",
        }

    if is_trip_distance_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 300,
            "retrieve_k": 0,
            "reason": "trip_distance",
        }

    if is_status_summary_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 400,
            "retrieve_k": 0,
            "reason": "status_summary",
        }

    if is_country_customer_question(q) or is_analytics_question(q):
        return {
            "intent": "analytics",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "needs_analytics": True,
            "response_style": "medium",
            "max_tokens_hint": 350,
            "retrieve_k": 0,
            "reason": "country_or_analytics",
        }

    if (
        re.search(
            r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\b",
            q,
            re.I,
        )
        or re.search(r"\b(state|province|city|town|address|street)\b", q, re.I)
        or re.search(
            r"\b(location|warehouse|facility)\b.*\b(order|orders|pickup|delivery|drop)\b|"
            r"\b(order|orders|pickup|delivery|drop)\b.*\b(location|warehouse|facility)\b",
            q,
            re.I,
        )
        or re.search(r"\borders?\b.*\b(in|from|to|at)\b\s+[A-Za-z]{2,}", q, re.I)
    ) and not re.search(
        r"\b(how many\s+customers?|customer\s+count|best|worst|low|wise)\b", q, re.I
    ):
        return {
            "intent": "list_filter",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 500,
            "retrieve_k": 0,
            "reason": "geo_or_address_filter",
        }

    if (
        LIST_RE.search(q)
        or re.search(
            r"\borderstatus\b|\baccountingstatus\b|\boutstatus\b|"
            r"\bstatus\s+(confirmed|delivered|dispatched|cancelled|canceled|quoted|"
            r"started|rejected|in[- ]?transit|partially\s*delivered|"
            r"invoiced|paid|partially\s*paid|restricted|open|planned|assigned)\b",
            q,
            re.I,
        )
        or re.search(r"\borders?\s+(with|for|by|in)\b", q, re.I)
        or re.search(r"\b(on|dated|date|pickup|delivery)\b.*\b20\d{2}\b", q, re.I)
    ) and not re.search(r"\b(total|sum|average|avg|how many|count)\b", q, re.I):
        token2 = extract_order_token(q)
        if not (
            token2
            and len(q.split()) <= 5
            and not re.search(
                r"\b(status|customer|currency|confirmed|delivered|date|location)\b",
                q,
                re.I,
            )
        ):
            return {
                "intent": "list_filter",
                "needs_rag": False,
                "needs_calculation": False,
                "needs_exact_order": False,
                "response_style": "medium",
                "max_tokens_hint": 450,
                "retrieve_k": 0,
                "reason": "filtered_list",
            }

    if token and re.search(
        r"\b(detail|details|show|get|find|lookup|look\s*up|info|information|give)\b",
        q,
        re.I,
    ):
        return {
            "intent": "order_lookup",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "reason": "explicit_order_lookup",
        }

    if token and len(q.split()) <= 4:
        return {
            "intent": "order_lookup",
            "needs_rag": False,
            "needs_calculation": False,
            "needs_exact_order": True,
            "response_style": "detailed",
            "max_tokens_hint": 1200,
            "retrieve_k": 0,
            "order_token": token,
            "reason": "short_order_token",
        }

    if history_hint and re.search(
        r"\b(that|this|same|it|its|uska|uski|uske|wo|woh|status|tax|freight)\b",
        q,
        re.I,
    ):
        if not is_calculation_question(q) or re.search(r"\b(uska|that|this|it)\b", q, re.I):
            if re.search(r"\b(status|tax|freight|detail|amount|customer|delivery)\b", q, re.I):
                return {
                    "intent": "order_lookup",
                    "needs_rag": False,
                    "needs_calculation": False,
                    "needs_exact_order": True,
                    "response_style": "medium",
                    "max_tokens_hint": 400,
                    "retrieve_k": 0,
                    "reason": "follow_up_with_session",
                }

    if is_calculation_question(q):
        return {
            "intent": "calculation",
            "needs_rag": False,
            "needs_calculation": True,
            "needs_exact_order": False,
            "response_style": "medium",
            "max_tokens_hint": 250,
            "retrieve_k": 0,
            "reason": "calculation_keywords",
        }

    return None


def plan_tools(
    intent: str,
    entities: Dict[str, Any],
    intent_info: Dict[str, Any],
) -> List[str]:
    from app.order_ask.tools import (
        TOOL_COMPARE,
        TOOL_GET_RECORD,
        TOOL_LIST_RECENT,
        TOOL_RUN_ANALYTICS,
        TOOL_RUN_CALCULATION,
        TOOL_SEARCH,
        TOOL_SEMANTIC_RAG,
    )

    tools: List[str] = []
    intent = (intent or "").lower()

    # User asked for a specific record but gave no id — LLM asks sweetly; no tools.
    if intent == "ask_for_record_id":
        return []

    if (
        intent == "analytics"
        or intent_info.get("needs_analytics")
        or entities.get("analytics")
        or entities.get("country")
    ):
        tools.append(TOOL_RUN_ANALYTICS)
        # Analytics answers are complete — don't also chase fake order tokens.
        if intent == "analytics" or intent_info.get("needs_analytics") or entities.get("analytics"):
            return tools

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if TOOL_RUN_ANALYTICS not in tools:
            tools.append(TOOL_RUN_CALCULATION)

    token = entities.get("order_token") or intent_info.get("order_token")
    if intent in ("order_lookup", "record_lookup") or intent_info.get("needs_exact_order"):
        pin = str(entities.get("pin") or "")
        fake_pin_token = bool(
            token
            and pin
            and re.sub(r"\s+", "", str(token)).upper()
            == re.sub(r"\s+", "", pin).upper()
        )
        if token and not fake_pin_token and not re.fullmatch(r"20\d{2}", str(token or "")):
            if not entities.get("order_token") and intent_info.get("order_token"):
                entities["order_token"] = intent_info["order_token"]
            tools.append(TOOL_GET_RECORD)

    if intent in ("list_filter", "list_orders", "filter") or (
        entities.get("sort_by") and intent not in ("analytics", "calculation")
    ):
        tools.append(TOOL_SEARCH)

    if has_list_filters(entities) and TOOL_RUN_ANALYTICS not in tools and TOOL_SEARCH not in tools and TOOL_GET_RECORD not in tools:
        tools.append(TOOL_SEARCH)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent == "compare":
        tools.append(TOOL_COMPARE)

    if intent_info.get("needs_rag") or intent == "open_qa":
        filters = entities_to_mongo_filters(entities)
        if filters and TOOL_SEARCH not in tools:
            tools.append(TOOL_SEARCH)
        elif (
            TOOL_GET_RECORD not in tools
            and TOOL_SEARCH not in tools
            and TOOL_RUN_ANALYTICS not in tools
        ):
            tools.append(TOOL_SEMANTIC_RAG)

    seen: set[str] = set()
    ordered: List[str] = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


RULES = DomainRules(
    name=DOMAIN,
    extract_entities=extract_entities,
    classify_intent_local=classify_intent_local,
    entities_to_mongo_filters=entities_to_mongo_filters,
    has_list_filters=has_list_filters,
    plan_tools=plan_tools,
    extract_record_token=extract_record_token,
    sticky_entity_keys=STICKY_KEYS,
    compare_token_pattern=re.compile(r"\b(MRP\d+|TORD\d+|\d{4,})\b", re.I),
)
