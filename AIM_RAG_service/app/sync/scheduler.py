"""Background sync scheduler (stdlib threading — no extra dependency).

Active only when ``AVAAL_SYNC_ENABLED=1``. For every tenant in
``AVAAL_SYNC_TENANTS`` it runs an incremental sync every
``AVAAL_SYNC_INTERVAL_MIN`` minutes and one full + reconcile sync per day at
``AVAAL_SYNC_FULL_HOUR`` (UTC).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger("sync.scheduler")

_thread: threading.Thread | None = None
_stop = threading.Event()

_OFF = {"", "0", "false", "no", "off"}


def _enabled() -> bool:
    return os.environ.get("AVAAL_SYNC_ENABLED", "0").strip().lower() not in _OFF


def _tenants() -> list[str]:
    return [t.strip() for t in os.environ.get("AVAAL_SYNC_TENANTS", "").split(",") if t.strip()]


def _run_all(mode: str) -> None:
    from app.sync.order_sync import sync_orders

    for cid in _tenants():
        if _stop.is_set():
            return
        try:
            sync_orders(cid, mode=mode)
        except Exception:  # noqa: BLE001
            logger.exception("scheduled %s sync failed for %s", mode, cid)


def _loop() -> None:
    interval = max(60, int(os.environ.get("AVAAL_SYNC_INTERVAL_MIN", "15")) * 60)
    full_hour = int(os.environ.get("AVAAL_SYNC_FULL_HOUR", "2"))
    last_incremental = 0.0
    last_full_date = None
    _stop.wait(30)  # let startup settle
    while not _stop.is_set():
        now = datetime.now(timezone.utc)
        try:
            if now.hour == full_hour and last_full_date != now.date():
                logger.info("scheduler: daily full sync")
                _run_all("full")
                last_full_date = now.date()
                last_incremental = time.monotonic()
            elif time.monotonic() - last_incremental >= interval:
                _run_all("incremental")
                last_incremental = time.monotonic()
        except Exception:  # noqa: BLE001
            logger.exception("scheduler tick failed")
        _stop.wait(60)


def start_scheduler() -> None:
    global _thread
    if not _enabled():
        logger.info("Avaal sync scheduler disabled (AVAAL_SYNC_ENABLED)")
        return
    if not _tenants():
        logger.warning("AVAAL_SYNC_ENABLED set but AVAAL_SYNC_TENANTS empty — not starting")
        return
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="avaal-sync", daemon=True)
    _thread.start()
    logger.info("Avaal sync scheduler started: tenants=%s interval=%smin",
                _tenants(), os.environ.get("AVAAL_SYNC_INTERVAL_MIN", "15"))


def stop_scheduler() -> None:
    _stop.set()
