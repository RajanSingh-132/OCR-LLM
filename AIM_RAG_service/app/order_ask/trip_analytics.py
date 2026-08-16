"""
Trip analytics over Avaal_trip_db for /api/v1/orders/ask.

Best trip  = most linked orders (orderids / itemscount)
Worst trip = fewest linked orders
Longest / shortest by totaldistance (or triptotaldistance)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from app.mongo_client import get_mongo_collection
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import AVAAL_TRIP_COLLECTION_NAME, AVAAL_TRIP_NAMESPACE
from app.order_ask.trip_retrieval import (
    trip_distance_value,
    trip_order_count,
)

_PROJECTION = {"embedding": 0, "page_content": 0}


def _base() -> Dict[str, Any]:
    return {"namespace": AVAAL_TRIP_NAMESPACE, "metadata.type": "avaal_trip"}


def is_trip_db_question(question: str) -> bool:
    """True when the user is asking about trip records (not order-fleet distance)."""
    q = (question or "").lower()
    if not re.search(r"\btrips?\b|\betp\d+\b|\btrip\s*(id|number|no|status)\b", q, re.I):
        # driver/truck/country on trip context without word trip still possible via ETP
        if re.search(r"\b(ETP\d+)\b", question or "", re.I):
            return True
        return False
    return True


def is_best_worst_trip_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\btrips?\b", q):
        return False
    return bool(
        re.search(
            r"\b(best|worst|top|bottom|most|least|fewest|highest|lowest)\b",
            q,
        )
        and re.search(r"\b(trip|trips|order|orders)\b", q)
    )


def is_longest_shortest_trip_question(question: str) -> bool:
    q = (question or "").lower()
    if not re.search(r"\btrips?\b", q):
        return False
    return bool(
        re.search(
            r"\b(longest|shortest|long|short|maximum\s+distance|minimum\s+distance|"
            r"sabse\s+(zyada|jyada|kam|lm|km)|max\s+distance|min\s+distance)\b",
            q,
        )
        or (
            re.search(r"\b(longest|shortest|max|min|maximum|minimum)\b", q)
            and re.search(r"\bdistance\b", q)
        )
    )


def is_trip_status_summary_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\btrips?\b", q)
        and re.search(r"\b(status|statuses|kitne|how many|count|summary)\b", q)
        and not re.search(r"\b(ETP\d+|trip\s*(id|number|no)\s*[:#]?\s*\w+)\b", q, re.I)
    )


def is_trip_lookup_question(question: str) -> bool:
    q = question or ""
    if re.search(r"\b(ETP\d+)\b", q, re.I):
        return True
    if re.search(
        r"\btrip(?:\s*(?:id|number|no\.?|#))?\s*[:#]?\s*[A-Za-z]*\d+",
        q,
        re.I,
    ):
        return True
    # Field ask about a trip without analytics words
    if re.search(r"\btrips?\b", q, re.I) and re.search(
        r"\b(driver|truck|customer|distance|status|country|phone|number|"
        r"pickup|delivery|order|details|detail|info|kaun|koun|kis)\b",
        q,
        re.I,
    ):
        if is_best_worst_trip_question(q) or is_longest_shortest_trip_question(q):
            return False
        if is_trip_status_summary_question(q):
            return False
        return True
    return False


def is_trip_analytics_question(question: str) -> bool:
    return (
        is_best_worst_trip_question(question)
        or is_longest_shortest_trip_question(question)
        or is_trip_status_summary_question(question)
    )


def detect_trip_rank_direction(question: str) -> str:
    q = (question or "").lower()
    if re.search(
        r"\b(worst|least|fewest|lowest|bottom|kam|minimum|min)\b",
        q,
    ):
        return "worst"
    return "best"


def detect_distance_direction(question: str) -> str:
    q = (question or "").lower()
    if re.search(
        r"\b(shortest|short|least\s+distance|minimum|min|kam|smallest)\b",
        q,
    ):
        return "shortest"
    return "longest"


def _trip_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tripid": doc.get("tripid"),
        "tripnumber": doc.get("tripnumber"),
        "tripstatus": doc.get("tripstatus"),
        "customername": doc.get("customername"),
        "firstdrivername": doc.get("firstdrivername"),
        "firstdriverphone": doc.get("firstdriverphone") or doc.get("firstdrivercell1"),
        "trucknumber": doc.get("trucknumber") or doc.get("customtrucknumber"),
        "truckcode": doc.get("truckcode"),
        "totaldistance": doc.get("totaldistance"),
        "triptotaldistance": doc.get("triptotaldistance"),
        "totalloaddistance": doc.get("totalloaddistance"),
        "totalemptydistance": doc.get("totalemptydistance"),
        "distanceunit": doc.get("distanceunit"),
        "ordernumber": doc.get("ordernumber"),
        "orderids": doc.get("orderids"),
        "pickupcountry": doc.get("pickupcountry"),
        "deliverycountry": doc.get("deliverycountry"),
        "order_count": trip_order_count(doc),
        "parsed_distance": trip_distance_value(doc),
    }


def best_worst_trips(
    *,
    direction: str = "best",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Best = highest order_count; worst = lowest order_count (>0 preferred for worst ties).
    """
    limit = max(1, min(int(limit or 5), 20))
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    rows: List[Dict[str, Any]] = []
    for doc in collection.find(_base(), _PROJECTION):
        row = _trip_row(doc)
        rows.append(row)

    reverse = direction != "worst"
    rows.sort(key=lambda r: (r.get("order_count") or 0, r.get("parsed_distance") or 0), reverse=reverse)
    top = rows[:limit]
    label = "best_trips" if direction == "best" else "worst_trips"
    checkpoint("TRIP_ANALYTICS", label, n=len(top))
    return {
        "analytics_type": "best_worst_trips",
        "direction": direction,
        "metric": "order_count",
        "definition": (
            "best trip = most linked orders; worst trip = fewest linked orders"
        ),
        label: top,
        "top": top,
        "total_trips_scanned": len(rows),
    }


def longest_shortest_trips(
    *,
    direction: str = "longest",
    limit: int = 5,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 5), 20))
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    rows: List[Dict[str, Any]] = []
    for doc in collection.find(_base(), _PROJECTION):
        row = _trip_row(doc)
        if row.get("parsed_distance") is None:
            continue
        rows.append(row)

    reverse = direction != "shortest"
    rows.sort(key=lambda r: r.get("parsed_distance") or 0, reverse=reverse)
    top = rows[:limit]
    label = "longest_trips" if direction == "longest" else "shortest_trips"
    checkpoint("TRIP_ANALYTICS", label, n=len(top))
    return {
        "analytics_type": "longest_shortest_trips",
        "direction": direction,
        "metric": "totaldistance",
        "definition": "Uses totaldistance / triptotaldistance / totalloaddistance",
        label: top,
        "top": top,
        "total_trips_with_distance": len(rows),
    }


def trip_status_summary() -> Dict[str, Any]:
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    counter: Counter = Counter()
    total = 0
    for doc in collection.find(_base(), {"tripstatus": 1, "_id": 0}):
        status = (doc.get("tripstatus") or "Unknown").strip() or "Unknown"
        counter[status] += 1
        total += 1
    by_status = [{"status": s, "count": n} for s, n in counter.most_common()]
    checkpoint("TRIP_ANALYTICS", "status_summary", total=total)
    return {
        "analytics_type": "trip_status_summary",
        "total_trips": total,
        "by_status": by_status,
    }


def run_trip_analytics(
    question: str,
    *,
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entities = entities or {}
    q = question or ""
    limit = int(entities.get("limit") or 5)

    if (
        is_best_worst_trip_question(q)
        or entities.get("analytics") in ("best_trip", "worst_trip", "best_worst_trips")
    ):
        direction = entities.get("trip_direction") or detect_trip_rank_direction(q)
        if entities.get("analytics") == "worst_trip":
            direction = "worst"
        if entities.get("analytics") == "best_trip":
            direction = "best"
        return best_worst_trips(direction=direction, limit=limit)

    if (
        is_longest_shortest_trip_question(q)
        or entities.get("analytics") in ("longest_trip", "shortest_trip")
    ):
        direction = entities.get("distance_direction") or detect_distance_direction(q)
        if entities.get("analytics") == "shortest_trip":
            direction = "shortest"
        if entities.get("analytics") == "longest_trip":
            direction = "longest"
        return longest_shortest_trips(direction=direction, limit=limit)

    if is_trip_status_summary_question(q) or entities.get("analytics") == "trip_status_summary":
        return trip_status_summary()

    # Default fleet overview when trip analytics forced
    return trip_status_summary()


def _clip_csv(value: Any, max_items: int = 5) -> str:
    """Shorten comma-separated lists for LLM context without losing counts."""
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
    else:
        parts = [p.strip() for p in re.split(r"\s*,\s*", str(value)) if p.strip()]
    if not parts:
        return ""
    if len(parts) <= max_items:
        return ", ".join(parts)
    return ", ".join(parts[:max_items]) + f" (+{len(parts) - max_items} more)"


def format_trip_analytics_for_context(payload: Dict[str, Any]) -> str:
    lines = [
        "TRIP ANALYTICS RESULT:",
        f"analytics_type: {payload.get('analytics_type')}",
    ]
    if payload.get("definition"):
        lines.append(f"definition: {payload.get('definition')}")
    if payload.get("direction"):
        lines.append(f"direction: {payload.get('direction')}")
    if payload.get("metric"):
        lines.append(f"metric: {payload.get('metric')}")
    if payload.get("total_trips") is not None:
        lines.append(f"total_trips: {payload.get('total_trips')}")
    if payload.get("total_trips_scanned") is not None:
        lines.append(f"total_trips_scanned: {payload.get('total_trips_scanned')}")
    if payload.get("total_trips_with_distance") is not None:
        lines.append(
            f"total_trips_with_distance: {payload.get('total_trips_with_distance')}"
        )
    if payload.get("by_status"):
        lines.append("by_status:")
        for row in payload["by_status"]:
            lines.append(f"  - {row.get('status')}: {row.get('count')}")
    top = payload.get("top") or []
    if top:
        lines.append("ranked_trips:")
        for i, row in enumerate(top, 1):
            # Compact: metrics + short samples (full lists stay in API analytics payload)
            lines.append(
                f"  {i}. tripnumber={row.get('tripnumber')} status={row.get('tripstatus')} "
                f"order_count={row.get('order_count')} distance={row.get('parsed_distance')} "
                f"unit={row.get('distanceunit')} "
                f"sample_customers={_clip_csv(row.get('customername'), 4)} "
                f"driver={row.get('firstdrivername') or 'n/a'} "
                f"truck={row.get('trucknumber') or 'n/a'} "
                f"sample_orders={_clip_csv(row.get('ordernumber') or row.get('orderids'), 6)} "
                f"pickup_country={row.get('pickupcountry')} "
                f"delivery_country={row.get('deliverycountry')}"
            )
    return "\n".join(lines)
