"""
Fleet & driver productivity from Avaal_trip — the engine behind the trip
planner's ``productivity`` task ("most productive truck and driver", "top 10
drivers by completed trips", "truck-wise total mileage", "which driver-truck
combination has the highest mileage", ...).

Per driver / truck / driver+truck pair it returns:
    trips             trips assigned (any status) in the filtered scope
    completed_trips   trips with tripstatus Delivered
    miles             SUM(triptotaldistance) over all assigned trips
    completed_miles   SUM(triptotaldistance) over completed trips
    avg_miles_per_trip
    distance_units    units seen (Miles / Km) — never silently mixed

A trip has two driver slots; both count. Trips with no driver (driver view) or
no truck (truck view) are left out instead of forming an "Unknown" group.
There are no engine-hour / capacity fields, so "utilization" is reported with
trips and miles as the proxy (the payload says so).
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.Ai_Assistant.Orders.order_dynamic_analytics import AGG_TIMEOUT_MS
from app.tenants.router import get_domain_collection

DIMENSIONS = ("driver", "truck", "driver_truck")
RANK_KEYS = ("completed_trips", "trips", "miles", "completed_miles", "avg_miles_per_trip")
COMPLETED_STATUSES = ["Delivered", "Completed"]


def _trimmed(field: str) -> Dict[str, Any]:
    return {"$trim": {"input": {"$toString": {"$ifNull": [f"${field}", ""]}}}}


def _miles_expr() -> Dict[str, Any]:
    return {
        "$ifNull": [
            {
                "$convert": {
                    "input": "$triptotaldistance",
                    "to": "double",
                    "onError": 0,
                    "onNull": 0,
                }
            },
            0,
        ]
    }


def _rank(
    match: Dict[str, Any], dimension: str, rank_by: str, direction: str, limit: int
) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = [
        {"$match": match},
        {
            "$addFields": {
                "__miles": _miles_expr(),
                "__done": {
                    "$cond": [{"$in": ["$tripstatus", COMPLETED_STATUSES]}, 1, 0]
                },
                "__drivers": {
                    "$filter": {
                        "input": [_trimmed("firstdrivername"), _trimmed("seconddrivername")],
                        "cond": {"$ne": ["$$this", ""]},
                    }
                },
                "__truck": _trimmed("trucknumber"),
            }
        },
    ]
    if dimension in ("driver", "driver_truck"):
        stages.append({"$unwind": "$__drivers"})
    if dimension in ("truck", "driver_truck"):
        stages.append({"$match": {"__truck": {"$ne": ""}}})

    key = {
        "driver": {"driver": "$__drivers"},
        "truck": {"truck": "$__truck"},
        "driver_truck": {"driver": "$__drivers", "truck": "$__truck"},
    }[dimension]

    sort_dir = 1 if direction == "asc" else -1
    tie = "miles" if rank_by != "miles" else "completed_trips"
    stages += [
        {
            "$group": {
                "_id": key,
                "trips": {"$sum": 1},
                "completed_trips": {"$sum": "$__done"},
                "miles": {"$sum": "$__miles"},
                "completed_miles": {"$sum": {"$multiply": ["$__done", "$__miles"]}},
                "distance_units": {"$addToSet": "$distanceunit"},
                "trip_numbers": {"$push": "$tripnumber"},
            }
        },
        {
            "$addFields": {
                "avg_miles_per_trip": {
                    "$cond": [
                        {"$gt": ["$trips", 0]},
                        {"$divide": ["$miles", "$trips"]},
                        0,
                    ]
                }
            }
        },
        {"$sort": {rank_by: sort_dir, tie: sort_dir, "_id": 1}},
        {"$limit": limit},
    ]
    rows = get_domain_collection("trips").aggregate(stages, maxTimeMS=AGG_TIMEOUT_MS)
    out = []
    for r in rows:
        row = dict(r["_id"])
        row.update(
            {
                "trips": int(r["trips"]),
                "completed_trips": int(r["completed_trips"]),
                "miles": round(float(r["miles"]), 2),
                "completed_miles": round(float(r["completed_miles"]), 2),
                "avg_miles_per_trip": round(float(r["avg_miles_per_trip"]), 2),
                "distance_units": sorted(u for u in r["distance_units"] if u),
            }
        )
        if row["trips"] <= 10:
            row["trip_numbers"] = [t for t in r["trip_numbers"] if t]
        out.append(row)
    return out


def fleet_productivity(
    match: Dict[str, Any],
    *,
    dimensions: List[str],
    rank_by: str = "completed_trips",
    direction: str = "desc",
    limit: int = 10,
    filters_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    dims = [d for d in dimensions if d in DIMENSIONS] or ["driver", "truck"]
    rank_by = rank_by if rank_by in RANK_KEYS else "completed_trips"
    direction = "asc" if direction == "asc" else "desc"
    limit = max(1, min(int(limit or 10), 50))
    return {
        "analytics_type": "fleet_productivity",
        "rank_by": rank_by,
        "direction": direction,
        "filters": filters_summary or {},
        "rankings": {d: _rank(match, d, rank_by, direction, limit) for d in dims},
    }


def format_productivity_for_context(payload: Dict[str, Any]) -> str:
    lines = [
        "FLEET PRODUCTIVITY RESULT (exact Mongo output — do not recalculate or invent):",
        "definitions: completed_trips = trips with status Delivered; trips = all "
        "assigned trips in scope; miles = SUM(triptotaldistance) of assigned trips; "
        "completed_miles = miles of completed trips. Both driver slots count. "
        "There are no engine-hour or capacity fields, so 'utilization' = trips and "
        "miles (say so if the user asked about utilization).",
        f"ranked by: {payload.get('rank_by')} ({payload.get('direction')}), ties broken by "
        + ("completed_trips" if payload.get("rank_by") == "miles" else "miles"),
    ]
    if payload.get("filters"):
        lines.append(f"filters_applied: {payload['filters']}")
    for dim, rows in (payload.get("rankings") or {}).items():
        label = {"driver": "DRIVERS", "truck": "TRUCKS", "driver_truck": "DRIVER + TRUCK PAIRS"}[dim]
        lines.append(f"{label} (already sorted):")
        for i, row in enumerate(rows, start=1):
            lines.append(f"[{i}] " + " | ".join(f"{k}={v}" for k, v in row.items()))
        if not rows:
            lines.append("(none matched — say so plainly)")
    lines.append("Show distances with their unit. Write numbers without commas.")
    return "\n".join(lines)
