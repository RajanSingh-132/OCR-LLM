"""
Fleet-wide analytics for Avaal_invoice collection.

Statuses: Paid, Open, PartiallyPaid, BadDebt, OverDue (and OPEN variants).
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.order_ask.checkpoint import checkpoint
from app.tenants.router import (
    get_domain_collection,
    get_domain_metadata_type,
    get_domain_namespace,
)

KNOWN_STATUSES = [
    "Paid",
    "Open",
    "PartiallyPaid",
    "BadDebt",
    "OverDue",
]

AMOUNT_FIELDS = ("TotalAmount", "totalamount")
OUTSTANDING_FIELDS = ("outstandinamount", "outstandingamount", "OutstandingAmount")


def _base_match() -> Dict[str, Any]:
    return {
        "namespace": get_domain_namespace("invoices"),
        "metadata.type": get_domain_metadata_type("invoices"),
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _amount_of(doc: Dict[str, Any]) -> Optional[float]:
    for field in AMOUNT_FIELDS:
        amt = _to_float(doc.get(field))
        if amt is not None:
            return amt
    return None


def _outstanding_of(doc: Dict[str, Any]) -> Optional[float]:
    for field in OUTSTANDING_FIELDS:
        amt = _to_float(doc.get(field))
        if amt is not None:
            return amt
    return None


def _paid_of(doc: Dict[str, Any]) -> Optional[float]:
    total = _amount_of(doc)
    if total is None:
        return None
    status = str(doc.get("InvoiceStatus") or "").lower()
    if status == "paid":
        return total
    outstanding = _outstanding_of(doc)
    if outstanding is None:
        return None
    return max(0.0, total - outstanding)


def _parse_us_datetime(raw: Any) -> Optional[datetime]:
    """Parse dates like '2/16/2026 2:30:00 AM' or ISO."""
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    for fmt in (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text.split("+")[0].split(".")[0].strip(), fmt)
        except ValueError:
            continue
    m = re.match(r"^(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](20\d{2})", text)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:
            mo, d = d, mo
        try:
            return datetime(y, mo, d)
        except ValueError:
            return None
    return None


def _country_from_location(loc: Any) -> Optional[str]:
    if not loc:
        return None
    text = str(loc).replace("\r", " ").replace("\n", " ")
    # Prefer last comma segment that looks like a country
    parts = [p.strip(" ,|") for p in re.split(r"[|,]", text) if p.strip(" ,|")]
    if not parts:
        return None
    for candidate in reversed(parts):
        cl = candidate.lower()
        if "united states" in cl or cl in {"usa", "us"}:
            return "United States"
        if "canada" in cl or cl == "ca":
            return "Canada"
        if "india" in cl or cl == "in":
            return "India"
        if re.search(r"[A-Za-z]{3,}", candidate) and not re.search(r"\d{5}", candidate):
            if "@" in candidate or "phone" in cl:
                continue
            if len(candidate) <= 40:
                return candidate.title()
    return None


def detect_status_filter(question: str) -> Optional[str]:
    q = (question or "").lower()
    # Ignore "paid amount" / "amount paid" — those are fields, not InvoiceStatus.
    q_norm = re.sub(r"\b(?:paid\s+amount|amount\s+paid)\b", " ", q)
    q_norm = q_norm.replace("-", " ")
    for key, canonical in (
        ("partiallypaid", "PartiallyPaid"),
        ("partially paid", "PartiallyPaid"),
        ("partial paid", "PartiallyPaid"),
        ("baddebt", "BadDebt"),
        ("bad debt", "BadDebt"),
        ("overdue", "OverDue"),
        ("over due", "OverDue"),
        ("paid", "Paid"),
        ("open", "Open"),
    ):
        if re.search(rf"\b{re.escape(key)}\b", q_norm):
            return canonical
    return None


def detect_period_days(question: str) -> Optional[int]:
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
    return None


def is_status_summary_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\b(list|show|display|find|get)\b", q) and not re.search(
        r"\b(how many|count|kitne|kitna|summary|total|breakdown)\b", q
    ):
        return False
    if re.search(
        r"\b(how many|count|kitne|kitna|total)\b.*\b(paid|open|partially\s*paid|bad\s*debt|overdue)\b",
        q,
    ):
        return True
    if re.search(
        r"\b(paid|open|partially\s*paid|bad\s*debt|overdue)\b.*\b(how many|count|kitne|kitna|total|huye|hua)\b",
        q,
    ):
        return True
    if re.search(r"\binvoice\s*status\b", q) and re.search(
        r"\b(summary|how many|count|breakdown|distribution)\b", q
    ):
        return True
    return False


def is_best_invoice_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\bcustomers?\b|\bcountr", q):
        return False
    return bool(
        re.search(
            r"\bbest\s+invoices?\b|"
            r"\b(best|top|highest|maximum|max|largest)\b.*\binvoices?\b|"
            r"\binvoices?\b.*\b(best|highest|maximum|max)\b",
            q,
        )
    )


def is_worst_invoice_question(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"\bcustomers?\b|\bcountr", q):
        return False
    return bool(
        re.search(
            r"\bworst\s+invoices?\b|"
            r"\b(worst|lowest|minimum|min|smallest)\b.*\binvoices?\b|"
            r"\binvoices?\b.*\b(worst|lowest|minimum|min)\b",
            q,
        )
    )


def is_invoices_by_country_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\binvoices?\b", q):
        return False
    return bool(
        re.search(
            r"\bcountry[\s\-]*wise\b|"
            r"\b(by|per|across)\s+countr(?:y|ies)\b|"
            r"\bcountr(?:y|ies)\b.*\b(most|max|jyada|highest|total|count)\b|"
            r"\b(most|max|jyada|highest)\b.*\bcountr",
            q,
        )
    )


def is_best_invoice_customer_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\bcustomers?\b", q):
        return False
    if not re.search(r"\binvoices?\b", q):
        return False
    return bool(
        re.search(
            r"\b(best|top|most|highest|maximum|max|jyada|sabse\s+jyada)\b|"
            r"\b(worst|least|lowest|minimum|min|fewest|sabse\s+kam|km)\b",
            q,
        )
    )


def is_period_invoices_question(question: str) -> bool:
    q = (question or "").lower()
    if detect_period_days(q) is None:
        return False
    return bool(
        re.search(
            r"\b(how many|count|kitne|kitna|invoices?|created|status|paid|open|bad\s*debt)\b",
            q,
        )
    )


def is_due_next_week_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\b(due|duedate|due\s*date)\b", q)
        and re.search(r"\b(next\s+week|coming\s+week|is\s+hafte|agle\s+hafte)\b", q)
    )


def is_invoice_analytics_question(question: str) -> bool:
    return (
        is_status_summary_question(question)
        or is_best_invoice_question(question)
        or is_worst_invoice_question(question)
        or is_invoices_by_country_question(question)
        or is_best_invoice_customer_question(question)
        or is_period_invoices_question(question)
        or is_due_next_week_question(question)
    )


def status_summary(*, status: Optional[str] = None) -> Dict[str, Any]:
    collection = get_domain_collection("invoices")
    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": {"_id": "$InvoiceStatus", "invoice_count": {"$sum": 1}}},
                {"$sort": {"invoice_count": -1}},
            ]
        )
    )
    by_status: Dict[str, int] = {}
    for r in rows:
        key = r["_id"] if r["_id"] not in (None, "") else "Unknown"
        # Normalize OPEN -> Open
        if str(key).upper() == "OPEN":
            key = "Open"
        by_status[str(key)] = by_status.get(str(key), 0) + int(r["invoice_count"])

    ordered = []
    for name in KNOWN_STATUSES:
        ordered.append({"status": name, "invoice_count": by_status.get(name, 0)})
    for name, count in by_status.items():
        if name not in KNOWN_STATUSES:
            ordered.append({"status": name, "invoice_count": count})

    total = sum(by_status.values())
    filtered = None
    if status:
        # case-insensitive match
        filtered = 0
        for k, v in by_status.items():
            if k.lower() == status.lower():
                filtered += v
    checkpoint("ANALYTICS", "invoice_status_summary", total=total, status=status)
    return {
        "analytics_type": "invoice_status_summary",
        "total_invoices": total,
        "by_status": ordered,
        "status_filter": status,
        "matching_invoices": filtered if status else total,
        "definition": (
            "Counts from all Avaal invoice records grouped by InvoiceStatus "
            "(Paid, Open, PartiallyPaid, BadDebt, OverDue)."
        ),
        "response_format": "status → count; if status asked, report that count clearly",
    }


def rank_invoices_by_amount(
    *,
    direction: str = "best",
    limit: int = 5,
    prefer_paid_for_best: bool = True,
) -> Dict[str, Any]:
    """
    best = highest TotalAmount (prefer Paid when prefer_paid_for_best)
    worst = lowest TotalAmount
    """
    limit = max(1, min(int(limit or 5), 25))
    direction = "worst" if str(direction).lower() == "worst" else "best"
    collection = get_domain_collection("invoices")
    projection = {
        "InvoiceID": 1,
        "InvoiceNumber": 1,
        "InvoiceStatus": 1,
        "CustomerName": 1,
        "TotalAmount": 1,
        "PreTaxAmount": 1,
        "outstandinamount": 1,
        "CurrencyCode": 1,
        "CompanyName": 1,
        "_id": 0,
    }

    scored = []
    for doc in collection.find(_base_match(), projection):
        amt = _amount_of(doc)
        if amt is None:
            continue
        status = str(doc.get("InvoiceStatus") or "")
        paid = _paid_of(doc)
        scored.append(
            {
                "InvoiceID": doc.get("InvoiceID"),
                "InvoiceNumber": doc.get("InvoiceNumber"),
                "InvoiceStatus": status,
                "CustomerName": doc.get("CustomerName"),
                "TotalAmount": amt,
                "PaidAmount": paid,
                "OutstandingAmount": _outstanding_of(doc),
                "CurrencyCode": doc.get("CurrencyCode"),
                "CompanyName": doc.get("CompanyName"),
                "_paid_flag": 1 if status.lower() == "paid" else 0,
            }
        )

    if direction == "best" and prefer_paid_for_best:
        scored.sort(
            key=lambda r: (r["_paid_flag"], r["TotalAmount"], r.get("PaidAmount") or 0),
            reverse=True,
        )
    elif direction == "best":
        scored.sort(key=lambda r: r["TotalAmount"], reverse=True)
    else:
        scored.sort(key=lambda r: r["TotalAmount"])

    rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in scored[:limit]]
    top = rows[0] if rows else None
    checkpoint("ANALYTICS", f"invoices_by_amount_{direction}", rows=len(rows))
    return {
        "analytics_type": "best_invoice" if direction == "best" else "worst_invoice",
        "direction": direction,
        "definition": (
            "Best invoice = highest TotalAmount (Paid preferred). "
            "Worst invoice = lowest TotalAmount."
        ),
        "response_format": "InvoiceNumber, amount, status, customer, paid/outstanding",
        "total_scored": len(scored),
        "rows": rows,
        "top": top,
    }


def invoices_by_country(*, location_side: str = "both", limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 50), 100))
    collection = get_domain_collection("invoices")
    projection = {"pickuplocation": 1, "deliverylocation": 1, "_id": 0}
    counter: Counter = Counter()
    for doc in collection.find(_base_match(), projection):
        countries = []
        if location_side in ("pickup", "both"):
            countries.append(_country_from_location(doc.get("pickuplocation")))
        if location_side in ("delivery", "both"):
            countries.append(_country_from_location(doc.get("deliverylocation")))
        if location_side == "both":
            country = (
                _country_from_location(doc.get("pickuplocation"))
                or _country_from_location(doc.get("deliverylocation"))
                or "Unknown"
            )
            counter[country] += 1
        else:
            for c in countries:
                counter[(c or "Unknown")] += 1

    rows = [{"country": c, "invoice_count": n} for c, n in counter.most_common(limit)]
    checkpoint("ANALYTICS", "invoices_by_country", rows=len(rows), side=location_side)
    return {
        "analytics_type": "invoices_by_country",
        "location_side": location_side,
        "definition": "Invoice counts by country parsed from pickuplocation/deliverylocation.",
        "response_format": "country_name → invoice_count",
        "total_groups": len(rows),
        "rows": rows,
        "top": rows[0] if rows else None,
    }


def rank_customers_by_invoices(
    *,
    direction: str = "best",
    limit: int = 5,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 5), 20))
    direction = "worst" if str(direction).lower() == "worst" else "best"
    order = 1 if direction == "worst" else -1
    collection = get_domain_collection("invoices")
    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {
                    "$group": {
                        "_id": "$CustomerName",
                        "invoice_count": {"$sum": 1},
                        "total_amount": {
                            "$sum": {
                                "$convert": {
                                    "input": "$TotalAmount",
                                    "to": "double",
                                    "onError": 0,
                                    "onNull": 0,
                                }
                            }
                        },
                    }
                },
                {"$match": {"_id": {"$nin": [None, ""]}}},
                {"$sort": {"invoice_count": order, "total_amount": order}},
                {"$limit": limit},
            ]
        )
    )
    customers = [
        {
            "CustomerName": r["_id"],
            "invoice_count": int(r["invoice_count"]),
            "total_amount": float(r.get("total_amount") or 0),
        }
        for r in rows
    ]
    top = customers[0] if customers else None
    checkpoint("ANALYTICS", f"invoice_customers_{direction}", rows=len(customers))
    return {
        "analytics_type": (
            "best_invoice_customer" if direction == "best" else "worst_invoice_customer"
        ),
        "direction": direction,
        "definition": (
            f"{'Most' if direction == 'best' else 'Fewest'} invoices per customer."
        ),
        "response_format": "customer → invoice_count (+ total_amount)",
        "customers": customers,
        "top": top,
    }


def invoices_in_period(
    *,
    days: int = 30,
    status: Optional[str] = None,
    date_field: str = "InvoiceDate",
) -> Dict[str, Any]:
    """Count invoices whose date field falls in the last N days (parsed client-side)."""
    days = max(1, int(days))
    start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    collection = get_domain_collection("invoices")
    projection = {
        "InvoiceDate": 1,
        "DueDate": 1,
        "InvoiceStatus": 1,
        "InvoiceNumber": 1,
        "CustomerName": 1,
        "TotalAmount": 1,
        "_id": 0,
    }
    matched = []
    status_counter: Counter = Counter()
    for doc in collection.find(_base_match(), projection):
        raw = doc.get(date_field) if date_field in doc else doc.get("InvoiceDate")
        dt = _parse_us_datetime(raw)
        if not dt or dt < start:
            continue
        st = str(doc.get("InvoiceStatus") or "Unknown")
        if st.upper() == "OPEN":
            st = "Open"
        if status and st.lower() != status.lower():
            continue
        status_counter[st] += 1
        matched.append(
            {
                "InvoiceNumber": doc.get("InvoiceNumber"),
                "InvoiceStatus": st,
                "CustomerName": doc.get("CustomerName"),
                "TotalAmount": doc.get("TotalAmount"),
                "InvoiceDate": doc.get("InvoiceDate"),
            }
        )

    by_status = [
        {"status": s, "invoice_count": n} for s, n in status_counter.most_common()
    ]
    checkpoint(
        "ANALYTICS",
        "invoices_in_period",
        days=days,
        status=status,
        total=len(matched),
    )
    return {
        "analytics_type": "invoices_in_period",
        "days": days,
        "period_start": start.strftime("%Y-%m-%d"),
        "date_field": date_field,
        "status_filter": status,
        "matching_invoices": len(matched),
        "by_status": by_status,
        "sample": matched[:15],
        "definition": (
            f"Invoices with {date_field} in last ~{days} days"
            + (f", status={status}" if status else "")
        ),
        "response_format": "matching_invoices count + optional status breakdown",
    }


def due_in_next_week(*, limit: int = 25) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end = now + timedelta(days=7)
    collection = get_domain_collection("invoices")
    projection = {
        "InvoiceNumber": 1,
        "InvoiceID": 1,
        "InvoiceStatus": 1,
        "CustomerName": 1,
        "DueDate": 1,
        "TotalAmount": 1,
        "outstandinamount": 1,
        "CurrencyCode": 1,
        "_id": 0,
    }
    rows = []
    for doc in collection.find(_base_match(), projection):
        due = _parse_us_datetime(doc.get("DueDate"))
        if not due:
            continue
        if now.date() <= due.date() <= end.date():
            rows.append(
                {
                    "InvoiceNumber": doc.get("InvoiceNumber"),
                    "InvoiceID": doc.get("InvoiceID"),
                    "InvoiceStatus": doc.get("InvoiceStatus"),
                    "CustomerName": doc.get("CustomerName"),
                    "DueDate": doc.get("DueDate"),
                    "TotalAmount": doc.get("TotalAmount"),
                    "OutstandingAmount": _outstanding_of(doc),
                    "CurrencyCode": doc.get("CurrencyCode"),
                }
            )
    rows.sort(key=lambda r: str(r.get("DueDate") or ""))
    checkpoint("ANALYTICS", "invoices_due_next_week", rows=len(rows))
    return {
        "analytics_type": "invoices_due_next_week",
        "window_start": now.strftime("%Y-%m-%d"),
        "window_end": end.strftime("%Y-%m-%d"),
        "matching_invoices": len(rows),
        "rows": rows[: max(1, min(int(limit or 25), 50))],
        "definition": "Invoices whose DueDate falls within the next 7 days.",
        "response_format": "count + InvoiceNumber, DueDate, status, customer, amount",
    }


def run_invoice_analytics(
    question: str,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entities = entities or {}
    q = question or ""
    limit = int(entities.get("limit") or 10)
    atype = entities.get("analytics")
    status = entities.get("InvoiceStatus") or detect_status_filter(q)

    if atype == "invoices_due_next_week" or is_due_next_week_question(q):
        return due_in_next_week(limit=limit)

    if atype == "worst_invoice" or is_worst_invoice_question(q):
        return rank_invoices_by_amount(direction="worst", limit=limit)

    if atype == "best_invoice" or is_best_invoice_question(q):
        return rank_invoices_by_amount(direction="best", limit=limit)

    if atype == "invoices_by_country" or is_invoices_by_country_question(q):
        return invoices_by_country(limit=max(limit, 20))

    if atype in ("best_invoice_customer", "worst_invoice_customer") or is_best_invoice_customer_question(q):
        direction = "worst" if (
            atype == "worst_invoice_customer"
            or re.search(r"\b(worst|least|lowest|fewest|kam|km|minimum|min)\b", q, re.I)
        ) else "best"
        return rank_customers_by_invoices(direction=direction, limit=limit)

    period_days = entities.get("period_days") or detect_period_days(q)
    if atype == "invoices_in_period" or period_days:
        return invoices_in_period(
            days=int(period_days or 30),
            status=status,
            date_field=str(entities.get("date_field") or "InvoiceDate"),
        )

    if (
        atype == "invoice_status_summary"
        or is_status_summary_question(q)
        or status
    ):
        return status_summary(status=status)

    return status_summary()


def format_invoice_analytics_for_context(payload: Dict[str, Any]) -> str:
    atype = payload.get("analytics_type")
    lines = [
        "INVOICE ANALYTICS RESULT (use these exact numbers/rows only):",
        f"analytics_type={atype}",
    ]
    if payload.get("definition"):
        lines.append(f"definition={payload['definition']}")
    if payload.get("response_format"):
        lines.append(f"response_format={payload['response_format']}")

    if atype == "invoice_status_summary":
        lines.append(f"total_invoices={payload.get('total_invoices')}")
        lines.append(f"status_filter={payload.get('status_filter')}")
        lines.append(f"matching_invoices={payload.get('matching_invoices')}")
        lines.append("by_status:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('invoice_count')}")

    elif atype in ("best_invoice", "worst_invoice"):
        lines.append(f"total_scored={payload.get('total_scored')}")
        top = payload.get("top") or {}
        if top:
            lines.append(
                "top_invoice: "
                f"InvoiceNumber={top.get('InvoiceNumber')} "
                f"TotalAmount={top.get('TotalAmount')} "
                f"PaidAmount={top.get('PaidAmount')} "
                f"status={top.get('InvoiceStatus')} "
                f"customer={top.get('CustomerName')}"
            )
        lines.append("ranked_invoices:")
        for i, row in enumerate(payload.get("rows") or [], start=1):
            lines.append(
                f"[{i}] InvoiceNumber={row.get('InvoiceNumber')} "
                f"TotalAmount={row.get('TotalAmount')} "
                f"PaidAmount={row.get('PaidAmount')} "
                f"status={row.get('InvoiceStatus')} "
                f"customer={row.get('CustomerName')}"
            )

    elif atype == "invoices_by_country":
        lines.append(f"total_groups={payload.get('total_groups')}")
        lines.append("country_wise_counts:")
        for row in payload.get("rows") or []:
            lines.append(f"- {row.get('country')}: {row.get('invoice_count')}")

    elif atype in ("best_invoice_customer", "worst_invoice_customer"):
        lines.append(f"direction={payload.get('direction')}")
        top = payload.get("top") or {}
        if top:
            lines.append(
                f"top_customer={top.get('CustomerName')} "
                f"invoice_count={top.get('invoice_count')} "
                f"total_amount={top.get('total_amount')}"
            )
        lines.append("customers:")
        for i, row in enumerate(payload.get("customers") or [], start=1):
            lines.append(
                f"[{i}] {row.get('CustomerName')} | "
                f"invoices={row.get('invoice_count')} | "
                f"amount={row.get('total_amount')}"
            )

    elif atype == "invoices_in_period":
        lines.append(f"days={payload.get('days')}")
        lines.append(f"period_start={payload.get('period_start')}")
        lines.append(f"status_filter={payload.get('status_filter')}")
        lines.append(f"matching_invoices={payload.get('matching_invoices')}")
        lines.append("by_status:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('invoice_count')}")

    elif atype == "invoices_due_next_week":
        lines.append(f"window={payload.get('window_start')} → {payload.get('window_end')}")
        lines.append(f"matching_invoices={payload.get('matching_invoices')}")
        for i, row in enumerate(payload.get("rows") or [], start=1):
            lines.append(
                f"[{i}] InvoiceNumber={row.get('InvoiceNumber')} "
                f"DueDate={row.get('DueDate')} status={row.get('InvoiceStatus')} "
                f"customer={row.get('CustomerName')} amount={row.get('TotalAmount')}"
            )
    else:
        lines.append(str(payload))

    if not payload.get("rows") and not payload.get("by_status") and not payload.get("customers"):
        if payload.get("matching_invoices") in (0, None) and not payload.get("total_invoices"):
            lines.append("(no invoice analytics rows)")
    return "\n".join(lines)
