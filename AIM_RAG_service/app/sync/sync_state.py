"""Per-tenant sync bookkeeping in Mongo: incremental cursor, run history, and a
short-lived lock so a cron run and a manual run don't overlap.

Collections (in the tenant DB):
    _sync_state   one doc per {corporate_id, domain}
    _sync_lock    one doc per {corporate_id, domain}, TTL-guarded
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from pymongo.errors import DuplicateKeyError

from app.mongo_client import get_mongo_collection

_STATE = "_sync_state"
_LOCK = "_sync_lock"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(corporate_id: str, domain: str) -> str:
    return f"{corporate_id}:{domain}"


def _state_coll(db_name: str):
    return get_mongo_collection(_STATE, db_name)


def _lock_coll(db_name: str):
    return get_mongo_collection(_LOCK, db_name)


def get_state(db_name: str, corporate_id: str, domain: str) -> Dict[str, Any]:
    return _state_coll(db_name).find_one({"_id": _key(corporate_id, domain)}) or {}


def get_cursor(db_name: str, corporate_id: str, domain: str) -> Optional[str]:
    """Max `modifiedon` string seen on the last successful sync (None on first run)."""
    return get_state(db_name, corporate_id, domain).get("last_cursor")


def set_cursor(db_name: str, corporate_id: str, domain: str, cursor: Optional[str]) -> None:
    if not cursor:
        return
    _state_coll(db_name).update_one(
        {"_id": _key(corporate_id, domain)},
        {"$set": {"last_cursor": cursor, "cursor_updated_at": _now()},
         "$setOnInsert": {"corporate_id": corporate_id, "domain": domain}},
        upsert=True,
    )


def record_run(
    db_name: str,
    corporate_id: str,
    domain: str,
    *,
    mode: str,
    status: str,
    counts: Dict[str, Any],
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    _state_coll(db_name).update_one(
        {"_id": _key(corporate_id, domain)},
        {
            "$set": {
                "last_run_at": _now(),
                "last_run_mode": mode,
                "last_run_status": status,
                "last_run_counts": counts,
                "last_run_duration_ms": duration_ms,
                "last_run_error": error,
            },
            "$setOnInsert": {"corporate_id": corporate_id, "domain": domain},
        },
        upsert=True,
    )


def acquire_lock(
    db_name: str, corporate_id: str, domain: str, *, ttl_sec: int = 1800
) -> bool:
    """True if this process now holds the lock; False if another run holds it."""
    now = _now()
    try:
        _lock_coll(db_name).update_one(
            {"_id": _key(corporate_id, domain), "expires_at": {"$lt": now}},
            {"$set": {"expires_at": now + timedelta(seconds=ttl_sec), "acquired_at": now}},
            upsert=True,
        )
        return True
    except DuplicateKeyError:
        return False


def release_lock(db_name: str, corporate_id: str, domain: str) -> None:
    _lock_coll(db_name).delete_one({"_id": _key(corporate_id, domain)})
