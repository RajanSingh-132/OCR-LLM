"""
Fleet-wide analytics for Avaal_trip collection.

- best / worst trip by loaded distance
- country-wise trip counts (pickup / delivery / both)
- trip status summary / counts (Planned, Dispatched, Started, In-Transit, Delivered, Rejected)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from app.order_ask.checkpoint import checkpoint
from app.tenants.router import (
    get_domain_collection,
    get_domain_metadata_type,
    get_domain_namespace,
)

DISTANCE_FIELDS = ("totalloaddistance", "triptotaldistance", "totaldistance")

KNOWN_TRIP_STATUSES = [
    "Planned",
    "Dispatched",
    "Started",
    "In-Transit",
    "Delivered",
    "Rejected",
]

_STATUS_ALIASES = {
    "planned": "Planned",
    "dispatched": "Dispatched",
    "started": "Started",
    "stated": "Started",
    "in-transit": "In-Transit",
    "in transit": "In-Transit",
    "intransit": "In-Transit",
    "enroute": "In-Transit",
    "en route": "In-Transit",
    "delivered": "Delivered",
    "deliverd": "Delivered",
    "rejected": "Rejected",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
}


def _base_match() -> Dict[str, Any]:
    return {
        "namespace": get_domain_namespace("trips"),
        "metadata.type": get_domain_metadata_type("trips"),
    }


def detect_location_side(question: str) -> str:
    q = (question or "").lower()
    wants_pickup = bool(re.search(r"\b(pickup|pick\s*up|origin)\b", q))
    wants_delivery = bool(
        re.search(r"\b(delivery|deliver|drop|destination)\b", q)
    )
    if wants_pickup and not wants_delivery:
        return "pickup"
    if wants_delivery and not wants_pickup:
        return "delivery"
    return "both"


def detect_status_filter(question: str) -> Optional[str]:
    q = (question or "").lower().replace("trasit", "transit")
    for key, canonical in sorted(_STATUS_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", q):
            return canonical
    return None


def _normalize_status_label(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "Unknown"
    key = re.sub(r"\s+", " ", s.lower().replace("_", " "))
    key = key.replace("trasit", "transit")
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    if key.replace("-", " ") in ("in transit",):
        return "In-Transit"
    if key == "enroute":
        return "In-Transit"
    # Title-case Dispatched / DISPATCHED → Dispatched
    for known in KNOWN_TRIP_STATUSES + ["Cancelled", "Enroute"]:
        if s.lower() == known.lower():
            return known if known != "Enroute" else "In-Transit"
    return s


def is_best_trip_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\btrips?\b", q):
        return False
    return bool(
        re.search(
            r"\b(best|top|longest|maximum|max|highest|most)\b.*\b(trip|distance)\b|"
            r"\b(trip|distance)\b.*\b(best|top|longest|maximum|max|highest|most)\b|"
            r"\bbest\s+trip\b",
            q,
        )
    )


def is_worst_trip_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\btrips?\b", q):
        return False
    return bool(
        re.search(
            r"\b(worst|shortest|minimum|min|lowest|least|fewest)\b.*\b(trip|distance)\b|"
            r"\b(trip|distance)\b.*\b(worst|shortest|minimum|min|lowest|least)\b|"
            r"\bworst\s+trip\b",
            q,
        )
    )


def is_trips_by_country_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\btrips?\b", q):
        return False
    return bool(
        re.search(
            r"\bcountry[\s\-]*wise\b|"
            r"\b(by|per|across)\s+countr(?:y|ies)\b|"
            r"\btrips?\b.*\bby\s+countr|"
            r"\bcountr(?:y|ies)\b.*\b(total|count|how many)\s+trips?\b|"
            r"\b(total|count|how many)\s+trips?\b.*\bcountr",
            q,
        )
    )


def is_status_summary_question(question: str) -> bool:
    q = (question or "").lower().replace("trasit", "transit")
    if re.search(r"\b(list|show|display|find|search|filter|get)\b", q) and not re.search(
        r"\b(how many|count|summary|break\s*down|breakdown|distribution|total|kitne|kitna)\b",
        q,
    ):
        return False
    status_words = (
        r"planned|dispatched|started|stated|in[- ]?transit|enroute|en\s*route|"
        r"delivered|deliverd|rejected|cancelled|canceled"
    )
    if re.search(r"\btrip\s*status(es)?\b", q) and re.search(
        r"\b(summary|how many|count|break|breakdown|distribution|kitne|kitna)\b", q
    ):
        return True
    if re.search(r"\bstatus\b.*\b(summary|break\s*down|breakdown|count|how many)\b", q):
        return True
    if re.search(rf"\b(how many|count|kitne|kitna)\b.*\b({status_words})\b", q):
        return True
    if re.search(rf"\b({status_words})\b.*\b(count|how many|kitne|kitna|huye|hua|hue)\b", q):
        return True
    if re.search(rf"\b({status_words})\b.*\btrips?\b", q) and re.search(
        r"\b(how many|count|summary|total|breakdown|kitne|kitna)\b", q
    ):
        return True
    return False


def is_trip_analytics_question(question: str) -> bool:
    return (
        is_best_trip_question(question)
        or is_worst_trip_question(question)
        or is_trips_by_country_question(question)
        or is_status_summary_question(question)
    )


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _distance_of(doc: Dict[str, Any]) -> Optional[float]:
    for field in DISTANCE_FIELDS:
        dist = _to_float(doc.get(field))
        if dist is not None:
            return dist
    return None


def _trip_summary(doc: Dict[str, Any], distance: float) -> Dict[str, Any]:
    return {
        "tripid": doc.get("tripid"),
        "tripnumber": doc.get("tripnumber"),
        "tripstatus": doc.get("tripstatus"),
        "triptype": doc.get("triptype"),
        "customername": doc.get("customername"),
        "firstdrivername": doc.get("firstdrivername"),
        "seconddrivername": doc.get("seconddrivername"),
        "pickupcountry": doc.get("pickupcountry"),
        "deliverycountry": doc.get("deliverycountry"),
        "totalloaddistance": distance,
        "distanceunit": doc.get("distanceunit") or "Miles",
    }


def rank_trips_by_distance(
    *,
    direction: str = "best",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    best = highest totalloaddistance / triptotaldistance
    worst = lowest positive (or zero) distance
    """
    limit = max(1, min(int(limit or 5), 25))
    collection = get_domain_collection("trips")
    projection = {
        "tripid": 1,
        "tripnumber": 1,
        "tripstatus": 1,
        "triptype": 1,
        "customername": 1,
        "firstdrivername": 1,
        "seconddrivername": 1,
        "pickupcountry": 1,
        "deliverycountry": 1,
        "totalloaddistance": 1,
        "triptotaldistance": 1,
        "totaldistance": 1,
        "distanceunit": 1,
        "_id": 0,
    }

    scored: List[Dict[str, Any]] = []
    for doc in collection.find(_base_match(), projection):
        dist = _distance_of(doc)
        if dist is None:
            continue
        scored.append(_trip_summary(doc, dist))

    reverse = direction != "worst"
    scored.sort(key=lambda row: row["totalloaddistance"], reverse=reverse)
    rows = scored[:limit]
    top = rows[0] if rows else None
    checkpoint(
        "ANALYTICS",
        f"trips_by_distance_{direction}",
        rows=len(rows),
    )
    return {
        "analytics_type": "best_trip" if direction == "best" else "worst_trip",
        "rank_by": "totalloaddistance",
        "direction": direction,
        "definition": (
            "Best trip = highest loaded/total distance. "
            "Worst trip = lowest loaded/total distance."
        ),
        "total_scored": len(scored),
        "rows": rows,
        "top": top,
        "response_format": (
            "tripnumber, distance, unit, status, drivers, customer "
            "(from rows only)"
        ),
    }


def trips_by_country(
    *,
    location_side: str = "both",
    limit: int = 50,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 50), 100))
    collection = get_domain_collection("trips")
    projection = {
        "pickupcountry": 1,
        "deliverycountry": 1,
        "_id": 0,
    }
    counter: Counter = Counter()
    for doc in collection.find(_base_match(), projection):
        countries = []
        if location_side in ("pickup", "both"):
            countries.append(doc.get("pickupcountry") or "Unknown")
        if location_side in ("delivery", "both"):
            countries.append(doc.get("deliverycountry") or "Unknown")
        if location_side == "both":
            # count trip once using pickup country when present else delivery
            country = doc.get("pickupcountry") or doc.get("deliverycountry") or "Unknown"
            counter[str(country).strip() or "Unknown"] += 1
        else:
            for c in countries:
                counter[str(c).strip() or "Unknown"] += 1

    rows = [
        {"country": country, "trip_count": n}
        for country, n in counter.most_common(limit)
    ]
    checkpoint(
        "ANALYTICS",
        "trips_by_country",
        rows=len(rows),
        side=location_side,
    )
    return {
        "analytics_type": "trips_by_country",
        "location_side": location_side,
        "definition": (
            "Trip counts grouped by country from pickupcountry / deliverycountry."
        ),
        "response_format": "country_name → trip_count (counts only)",
        "total_groups": len(rows),
        "rows": rows,
    }


def status_summary(*, status: Optional[str] = None) -> Dict[str, Any]:
    """Count trips per tripstatus (normalize In Transit/Enroute → In-Transit)."""
    collection = get_domain_collection("trips")
    rows = list(
        collection.aggregate(
            [
                {"$match": _base_match()},
                {"$group": {"_id": "$tripstatus", "trip_count": {"$sum": 1}}},
                {"$sort": {"trip_count": -1}},
            ]
        )
    )
    by_status: Dict[str, int] = {}
    for r in rows:
        label = _normalize_status_label(r["_id"])
        by_status[label] = by_status.get(label, 0) + int(r["trip_count"])

    total = sum(by_status.values())
    ordered = []
    for name in KNOWN_TRIP_STATUSES:
        ordered.append({"status": name, "trip_count": by_status.get(name, 0)})
    for name, count in by_status.items():
        if name not in KNOWN_TRIP_STATUSES:
            ordered.append({"status": name, "trip_count": count})

    matching = total
    if status:
        matching = 0
        want = _normalize_status_label(status)
        for k, v in by_status.items():
            if k.lower() == want.lower():
                matching = v
                break

    checkpoint(
        "ANALYTICS",
        "trip_status_summary",
        total_trips=total,
        status=status,
        statuses=len(ordered),
    )
    return {
        "analytics_type": "trip_status_summary",
        "status_filter": status,
        "matching_trips": matching if status else total,
        "total_trips": total,
        "by_status": ordered,
        "definition": (
            "Counts from all Avaal_trip records by tripstatus. "
            "Valid: Planned, Dispatched, Started, In-Transit, Delivered, Rejected. "
            "(DB 'In Transit' / Enroute counted under In-Transit.)"
        ),
        "response_format": "status → count; if status asked, report that count clearly",
    }


def run_trip_analytics(
    question: str,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entities = entities or {}
    q = question or ""
    side = entities.get("location_side") or detect_location_side(q)
    limit = int(entities.get("limit") or 10)
    atype = entities.get("analytics")
    status = entities.get("tripstatus") or detect_status_filter(q)

    if atype == "worst_trip" or is_worst_trip_question(q):
        return rank_trips_by_distance(direction="worst", limit=limit)
    if atype == "best_trip" or is_best_trip_question(q):
        return rank_trips_by_distance(direction="best", limit=limit)
    if atype == "trips_by_country" or is_trips_by_country_question(q):
        return trips_by_country(location_side=side, limit=max(limit, 20))
    if (
        atype == "trip_status_summary"
        or is_status_summary_question(q)
        or (status and re.search(r"\b(how many|count|kitne|kitna|total)\b", q, re.I))
    ):
        return status_summary(status=status)

    # Default fleet snapshot: best trip
    return rank_trips_by_distance(direction="best", limit=limit)


def format_trip_analytics_for_context(payload: Dict[str, Any]) -> str:
    atype = payload.get("analytics_type")
    lines = [
        "TRIP ANALYTICS RESULT (use these exact numbers/rows only):",
        f"analytics_type={atype}",
    ]
    if payload.get("definition"):
        lines.append(f"definition={payload['definition']}")
    if payload.get("response_format"):
        lines.append(f"response_format={payload['response_format']}")

    if atype in ("best_trip", "worst_trip"):
        lines.append(f"total_scored={payload.get('total_scored')}")
        top = payload.get("top") or {}
        if top:
            lines.append(
                "top_trip: "
                f"tripnumber={top.get('tripnumber')} "
                f"distance={top.get('totalloaddistance')} "
                f"unit={top.get('distanceunit')} "
                f"status={top.get('tripstatus')} "
                f"firstdriver={top.get('firstdrivername')} "
                f"seconddriver={top.get('seconddrivername')} "
                f"customer={top.get('customername')}"
            )
        lines.append("ranked_trips:")
        for i, row in enumerate(payload.get("rows") or [], start=1):
            lines.append(
                f"[{i}] tripnumber={row.get('tripnumber')} "
                f"distance={row.get('totalloaddistance')} "
                f"{row.get('distanceunit')} "
                f"status={row.get('tripstatus')} "
                f"drivers={row.get('firstdrivername')} / {row.get('seconddrivername')} "
                f"customer={row.get('customername')}"
            )
    elif atype == "trips_by_country":
        lines.append(f"location_side={payload.get('location_side')}")
        lines.append(f"total_groups={payload.get('total_groups')}")
        lines.append("country_wise_counts:")
        for row in payload.get("rows") or []:
            lines.append(f"- {row.get('country')}: {row.get('trip_count')}")
    elif atype == "trip_status_summary":
        lines.append(f"total_trips={payload.get('total_trips')}")
        if payload.get("status_filter"):
            lines.append(f"status_filter={payload.get('status_filter')}")
            lines.append(f"matching_trips={payload.get('matching_trips')}")
        lines.append("by_status:")
        for row in payload.get("by_status") or []:
            lines.append(f"- {row.get('status')}: {row.get('trip_count')}")
    else:
        lines.append(str(payload))

    if (
        not payload.get("rows")
        and not payload.get("top")
        and not payload.get("by_status")
    ):
        lines.append("(no trip analytics rows)")
    return "\n".join(lines)
