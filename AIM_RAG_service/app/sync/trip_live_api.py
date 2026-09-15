"""
Live trip sync daemon — polls the Avaal Web API every SLEEP_SECONDS and
keeps MongoDB in sync with the latest PAGE_SIZE trip records.

  python -m app.sync.trip_live_api
  (or)  python app/sync/trip_live_api.py

Runs forever until Ctrl+C. Multi-tenant: syncs every corporate_id in
CORPORATE_IDS from ONE process — every cycle, all tenants are hit
CONCURRENTLY (ThreadPoolExecutor, one thread per tenant), so the cycle's
total time is bounded by the SLOWEST tenant, not the sum of all of them.
Same design as app/sync/order_live_api.py — see that file's docstring for
the full rationale (crash-safety, 30s cadence, daily purge, parallel
embedding). DB_NAME_OVERRIDES maps AFMQA -> chatbot_db; every other
corporate_id falls back to using its own id as the Mongo database name
(e.g. AFN01992 -> Mongo db "AFN01992") unless an override is added for it.

API contract (confirmed against a live browser request, NOT guessed):
  POST https://beta.afmsuite.ai/api/Trip/gettriplist
  header: corporateid: <tenant>
  body:   {"filter": {...}}   <- lowercase "filter" here (order's API needed
          capital "Filter" — the two endpoints are NOT consistent, don't
          copy one body shape onto the other).
  response: {"total_count": N, "detailstrips": "[ ...JSON string... ]", ...}
          the trip records are inside "detailstrips" (not "details"), and
          the whole response may itself be wrapped in a 1-element JSON
          array — see fetch_latest_trips() for the unwrap.

USE_API_DATE_FILTER is OFF by default here: the working curl we captured had
isdatefilterapplicable=false and we don't yet know a confirmed datetype code
for trips (order's was "CD"/"OD" — trip's equivalent hasn't been verified).
Leaving it off is safe: PAGE_SIZE is tiny (15) so we always just take the
most-recently-modified trips and let the client-side `is_recent_record`
check filter out anything older than RECENT_MONTHS.

Each cycle, PER TENANT (in parallel):
  1. POST gettriplist (header corporateid: <tenant>, body {"filter": {...}},
     pageno=1, pagesize=PAGE_SIZE) -> the PAGE_SIZE most recent records.
  2. Keep only records whose `createdon` is within the last calendar month
     (RECENT_MONTHS) — re-checked client-side (API date filter is off, see
     above).
  3. For each record, compare by `tripid` against that tenant's collection:
       - tripid already stored -> delete the old doc, insert the fresh one
         (refresh, with a new embedding).
       - tripid not stored yet -> insert it as a new doc.
  4. Embeddings for the whole cycle's records are computed CONCURRENTLY
     (a second, inner ThreadPoolExecutor) since Bedrock's Titan endpoint only
     takes one text per call — see `_embed_texts`. So a full cycle has two
     levels of parallelism: one thread per tenant, and within each tenant's
     thread, one thread per record being embedded.

Safety / operational notes (read before leaving this running unattended):
  - This process must stay running for the sync to keep happening — closing
    the terminal / sleeping the PC / a crash stops it. For a real always-on
    deployment, wrap it as a Windows Service (e.g. via NSSM) or a scheduled
    task that restarts it, rather than relying on a terminal staying open.
  - One tenant's cycle failing (network blip, Mongo hiccup, API error) is
    caught and logged per-tenant; it does NOT kill the daemon or block the
    other tenants — the next cycle just retries that tenant after
    SLEEP_SECONDS.
  - The next cycle starts SLEEP_SECONDS after ALL tenants in the PREVIOUS
    cycle finish, not on a fixed clock tick — so a slow cycle can never
    overlap with the next.
  - Logs go to both the console and ROTATING file `trip_live_api.log` next
    to this script (so a long-running process doesn't grow the log forever).
    Every log line names its corporate_id, since one file now covers every
    tenant.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Set

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.embedding_client import get_embeddings
from app.mongo_client import _to_python_types, get_mongo_collection

logger = logging.getLogger("app.sync.trip_live_api")

# ===================== CONFIG — edit these =====================
API_BASE = "https://beta.afmsuite.ai"
LISTTRIP_PATH = "/api/Trip/gettriplist"
API_VERSION = "1.0"
# Every tenant this ONE process syncs, hit concurrently each cycle.
CORPORATE_IDS = [
    "AFMQA", "AFN01992", "AFN01856", "AFN01813", "AFN01619",
    "AFN01514", "AFN00292", "AFN00861", "AFN01801", "AFN00681",
]
EXTRA_HEADERS: Dict[str, str] = {}

# Body wrapper key is lowercase "filter" for this endpoint (confirmed from a
# working browser request) — every field below must be present since this
# DTO appears to reject missing fields the same way the order one does.
# pageno/pagesize/totalcount are overwritten per page automatically.
BASE_FILTER: Dict[str, Any] = {
    "isdatefilterapplicable": False,
    "datetype": None,
    "fromdate": None,
    "todate": None,
    "isquickfilter": True,
    "bycalendardatetype": None,
    "byaccounting": None,
    "byshipmenttype": None,
    "bycalendar": None,
    "createdby": None,
    "ordernumber": None,
    "tripnumber": None,
    "customerordernumber": None,
    "bolpodnumber": None,
    "containernumber": None,
    "pickuplocationcode": None,
    "pickupcity": None,
    "pickupstate": None,
    "pickupcountry": None,
    "pickupreferencenos": None,
    "deliverylocationcode": None,
    "deliverycity": None,
    "deliverystate": None,
    "deliverycountry": None,
    "deliveryreferencenos": None,
    "tripstatus": None,
    "settlementstatus": None,
    "triptype": None,
    "trucknumber": None,
    "trailernumber": None,
    "firstdriver": None,
    "seconddriver": None,
    "customername": None,
    "carriername": None,
    "companyname": None,
    "tripshipmenttype": None,
    "status": "A",
    "searchvalue": "",
    "sortcolumn": "ModifiedOn",
    "sortorder": "DESC",
    "usercode": "",
    "pickupregionid": None,
    "deliveryregionid": None,
    "vinnumber": None,
    "salesmancode": None,
    "equipmenttype": None,
    "companycode": "",
}
PAGE_SIZE = 15
MAX_PAGES = 1             # only the latest page each cycle (PAGE_SIZE records)
REQUEST_TIMEOUT = 60      # seconds — must stay well under SLEEP_SECONDS

# See module docstring — off until a confirmed trip datetype code is known.
USE_API_DATE_FILTER = False
API_DATE_TYPE = "CD"

DB_NAME_OVERRIDES = {"AFMQA": "chatbot_db"}
COLLECTION_NAME = "Avaal_trip"
NAMESPACE = "avaal_trips"
METADATA_TYPE = "avaal_trip"
DUPLICATE_FIELD = "tripnumber"
ID_FIELD = "tripid"
# Keys in the source payload that may hold the list of trip records.
DETAIL_KEYS = ("detailstrips", "details", "Details")

WITH_EMBEDDINGS = True

# Only keep records whose `createdon` is within the last calendar month.
FILTER_RECENT_ONLY = True
DATE_FIELD = "createdon"
RECENT_MONTHS = 1

INSERT_BATCH_SIZE = 200
EMBED_BATCH_SIZE = 25

# Titan embedding guard rails.
MAX_EMBED_TOKENS = 8192
EST_CHARS_PER_TOKEN = 1.4
MAX_EMBED_CHARS = int(MAX_EMBED_TOKENS * EST_CHARS_PER_TOKEN * 0.85)
# Bedrock's Titan endpoint takes ONE text per call — no batch API. We fan
# calls out across threads (I/O-bound) instead of looping sequentially.
EMBED_MAX_WORKERS = 10

# Gap AFTER each cycle finishes, before the next one starts.
SLEEP_SECONDS = 30

# --- Daily purge: delete anything older than RECENT_MONTHS -----------------
# Runs independently of the 30-second ingest cycle. "Independent" here means:
# a separate function, checked once per cycle but only ACTUALLY executing
# when >= CLEANUP_INTERVAL_HOURS have passed since it last ran. The cutoff is
# computed fresh from datetime.now() every time it runs, so it rolls forward
# on its own each day — no hardcoded date anywhere.
CLEANUP_INTERVAL_HOURS = 24
# Last-run timestamp is persisted in Mongo (not just kept in memory) so a
# restart of this process doesn't forget it and purge more than once a day.
SYNC_STATE_COLLECTION = "sync_state"
SYNC_STATE_DOC_ID = "trip_live_api_cleanup"

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trip_live_api.log")
# ===============================================================


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _db_name(corporate_id: str) -> str:
    return DB_NAME_OVERRIDES.get(corporate_id.upper(), corporate_id)


def _api_url() -> str:
    return f"{API_BASE.rstrip('/')}{LISTTRIP_PATH}"


def _parse_details_payload(payload: Any) -> List[Dict[str, Any]]:
    """Pull the list of trip dicts out of one gettriplist response."""
    details = None
    if isinstance(payload, dict):
        for key in DETAIL_KEYS:
            if key in payload:
                details = payload[key]
                break
    else:
        details = payload

    if details is None:
        return []
    if isinstance(details, str):
        details = details.strip()
        if not details:
            return []
        details = json.loads(details)
    if isinstance(details, dict):
        details = [details]
    if not isinstance(details, list):
        return []
    return [r for r in details if isinstance(r, dict)]


def _extract_total_count(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in ("total_count", "totalcount", "TotalCount", "Total_Count"):
        if key in payload:
            return payload[key]
    return None


def _build_filter(pageno: int, from_date: str | None, to_date: str | None) -> Dict[str, Any]:
    flt = {
        **BASE_FILTER,
        "pageno": pageno,
        "pagesize": PAGE_SIZE,
        "totalcount": PAGE_SIZE,
    }
    if USE_API_DATE_FILTER and from_date and to_date:
        flt["isdatefilterapplicable"] = True
        flt["datetype"] = API_DATE_TYPE
        flt["fromdate"] = from_date
        flt["todate"] = to_date
    return flt


def fetch_latest_trips(
    corporate_id: str, from_date: str | None = None, to_date: str | None = None
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch up to MAX_PAGES pages (PAGE_SIZE records each) from gettriplist
    for one tenant. Takes corporate_id as a parameter (not a module global)
    so this is safe to call concurrently from multiple tenant threads at
    once."""
    session = requests.Session()
    headers = {
        "Content-Type": "application/json; ver=1.0",
        "Accept": "application/json",
        "corporateid": corporate_id,
        **EXTRA_HEADERS,
    }
    params = {"api-version": API_VERSION}
    url = _api_url()

    all_records: List[Dict[str, Any]] = []
    reported_total = None
    pages = 0

    for pageno in range(1, MAX_PAGES + 1):
        body = {"filter": _build_filter(pageno, from_date, to_date)}
        resp = session.post(
            url, params=params, headers=headers, json=body, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        # Unlike order's listorder (bare object), gettriplist wraps its
        # single result object in a 1-element JSON array: [{...}]. Unwrap it
        # so the rest of the pipeline can treat `data` as a plain dict.
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]

        if reported_total is None:
            reported_total = _extract_total_count(data)

        page_records = _parse_details_payload(data)
        pages = pageno
        if not page_records:
            break
        all_records.extend(page_records)

        if reported_total and len(all_records) >= int(reported_total):
            break
        if len(page_records) < PAGE_SIZE:
            break

    stats = {
        "pages_fetched": pages,
        "reported_total_count": reported_total,
        "records_received": len(all_records),
    }
    return all_records, stats


def _subtract_months(moment: datetime.datetime, months: int) -> datetime.datetime:
    month_index = (moment.year * 12 + (moment.month - 1)) - months
    year, month = divmod(month_index, 12)
    month += 1
    if month == 12:
        next_month_first = datetime.datetime(year + 1, 1, 1)
    else:
        next_month_first = datetime.datetime(year, month + 1, 1)
    last_day = (next_month_first - datetime.timedelta(days=1)).day
    day = min(moment.day, last_day)
    return moment.replace(year=year, month=month, day=day)


def _parse_datetime(value: Any) -> datetime.datetime | None:
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(raw)
        except ValueError:
            try:
                dt = datetime.datetime.fromisoformat(raw[:19])
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def is_recent_record(trip: Dict[str, Any], cutoff: datetime.datetime) -> bool:
    created = _parse_datetime(trip.get(DATE_FIELD))
    if created is None:
        return False
    return created >= cutoff


def build_page_content(trip: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(trip.keys()):
        value = trip[key]
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _truncate_for_embedding(text: str) -> str:
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding text is %s chars; truncating to %s", len(text), MAX_EMBED_CHARS
    )
    return text[:MAX_EMBED_CHARS]


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "input token" in msg or "too many input tokens" in msg


def _embed_one_with_retry(embeddings, text: str) -> List[float]:
    attempt = text
    for _ in range(8):
        try:
            return embeddings.embed_query(attempt)
        except Exception as exc:  # noqa: BLE001
            if not _is_token_limit_error(exc):
                raise
            new_len = max(400, int(len(attempt) * 0.6))
            if new_len >= len(attempt):
                raise
            attempt = attempt[:new_len]
    raise RuntimeError("Could not shrink text below the Titan token limit")


def _embed_texts(embeddings, texts: List[str]) -> List[List[float]]:
    """Embed every text concurrently — see order_live_api.py's _embed_texts
    for the full rationale (Bedrock's Titan endpoint is one-text-per-call)."""
    if not texts:
        return []

    results: List[List[float] | None] = [None] * len(texts)
    max_workers = max(1, min(EMBED_MAX_WORKERS, len(texts)))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(_embed_one_with_retry, embeddings, text): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()

    return results  # type: ignore[return-value]


def build_document(
    trip: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
    source_document: str,
    database: str,
    corporate_id: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    for key, value in trip.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = NAMESPACE
    doc["page_content"] = build_page_content(trip)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": METADATA_TYPE,
        "source_document": source_document,
        "collection": COLLECTION_NAME,
        "database": database,
        "corporate_id": corporate_id,
        "tripid": trip.get(ID_FIELD),
        "tripnumber": trip.get(DUPLICATE_FIELD),
        "ingested_at": ingested_at,
        "embedding_dimensions": len(embedding),
        "structured": True,
    }
    return doc


def load_existing_keys(collection, field: str) -> Set[str]:
    existing: Set[str] = set()
    projection = {field: 1, f"metadata.{field}": 1}
    for doc in collection.find({}, projection):
        value = doc.get(field)
        if value in (None, ""):
            value = (doc.get("metadata") or {}).get(field)
        if value not in (None, ""):
            existing.add(str(value).strip())
    return existing


def run_one_cycle(corporate_id: str) -> Dict[str, Any]:
    """Fetch the latest PAGE_SIZE trips for ONE tenant and sync them into
    that tenant's own Mongo database. Raises on unrecoverable errors
    (network, Mongo) — the caller decides how to handle a failed cycle.
    Takes corporate_id as a parameter (not a module global) so this is safe
    to call concurrently from multiple tenant threads at once."""
    database = _db_name(corporate_id)
    source_document = f"api:{_api_url()}?corporateid={corporate_id}"

    now = datetime.datetime.now(datetime.timezone.utc)
    ingested_at = now.isoformat()
    recent_cutoff = (
        _subtract_months(now, RECENT_MONTHS).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if FILTER_RECENT_ONLY
        else None
    )
    from_date = to_date = None
    if recent_cutoff is not None:
        from_date = recent_cutoff.strftime("%Y-%m-%d")
        to_date = (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    t0 = time.perf_counter()
    records, fetch_stats = fetch_latest_trips(corporate_id, from_date, to_date)
    t_fetch = time.perf_counter() - t0

    if not records:
        return {
            "ok": True,
            "corporate_id": corporate_id,
            "database": database,
            "note": "gettriplist returned no records this cycle",
            **fetch_stats,
            "timing_seconds": {"fetch": round(t_fetch, 2), "total": round(t_fetch, 2)},
        }

    # --- transform / filter: decide refresh (tripid match) vs new ---------
    t1 = time.perf_counter()
    collection = get_mongo_collection(COLLECTION_NAME, database, ensure_indexes=True)
    existing_tripids: Set[str] = load_existing_keys(collection, ID_FIELD)

    embeddings = get_embeddings() if WITH_EMBEDDINGS else None

    pending: List[Dict[str, Any]] = []
    ids_to_refresh: List[Any] = []
    skipped_old = 0
    refreshed_count = 0
    new_count = 0
    for trip in records:
        if recent_cutoff is not None and not is_recent_record(trip, recent_cutoff):
            skipped_old += 1
            continue
        key_value = trip.get(ID_FIELD)
        key_str = str(key_value).strip() if key_value not in (None, "") else ""
        if key_str and key_str in existing_tripids:
            ids_to_refresh.append(key_value)
            refreshed_count += 1
        else:
            new_count += 1
        pending.append(trip)
    t_transform = time.perf_counter() - t1

    # --- delete stale matches, then embed (optional) + insert everything --
    t2 = time.perf_counter()
    deleted = 0
    if ids_to_refresh:
        del_result = collection.delete_many(
            {"namespace": NAMESPACE, ID_FIELD: {"$in": ids_to_refresh}}
        )
        deleted = del_result.deleted_count

    inserted = 0
    batch: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal inserted, batch
        if not batch:
            return
        collection.insert_many(batch, ordered=False)
        inserted += len(batch)
        batch = []

    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        chunk = pending[start : start + EMBED_BATCH_SIZE]
        if embeddings is not None:
            texts = [_truncate_for_embedding(build_page_content(o)) for o in chunk]
            vectors = _embed_texts(embeddings, texts)
        else:
            vectors = [[] for _ in chunk]
        for trip, vector in zip(chunk, vectors):
            batch.append(
                build_document(
                    trip, vector, ingested_at, source_document, database, corporate_id
                )
            )
        if len(batch) >= INSERT_BATCH_SIZE:
            flush()
    flush()
    t_insert = time.perf_counter() - t2

    collection.create_index([("namespace", 1), (ID_FIELD, 1)])
    collection.create_index([("namespace", 1), (DUPLICATE_FIELD, 1)])

    stored = collection.count_documents(
        {"namespace": NAMESPACE, "metadata.type": METADATA_TYPE}
    )

    total = time.perf_counter() - t0
    return {
        "ok": True,
        "corporate_id": corporate_id,
        "database": database,
        "collection": COLLECTION_NAME,
        "with_embeddings": WITH_EMBEDDINGS,
        **fetch_stats,
        "documents_skipped_old": skipped_old,
        "documents_new": new_count,
        "documents_matched_refreshed": refreshed_count,
        "documents_deleted_before_refresh": deleted,
        "documents_inserted": inserted,
        "documents_in_namespace": stored,
        "timing_seconds": {
            "fetch": round(t_fetch, 2),
            "transform_filter": round(t_transform, 2),
            "delete_embed_insert": round(t_insert, 2),
            "total": round(total, 2),
        },
    }


def purge_old_records(corporate_id: str) -> Dict[str, Any]:
    """Delete every document older than RECENT_MONTHS for ONE tenant. `now`
    (and therefore the cutoff) is computed fresh at the moment this actually
    runs, so the cutoff rolls forward on its own each day this executes —
    nothing here is a hardcoded date."""
    database = _db_name(corporate_id)
    collection = get_mongo_collection(COLLECTION_NAME, database, ensure_indexes=False)

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = _subtract_months(now, RECENT_MONTHS).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    result = collection.delete_many(
        {
            "namespace": NAMESPACE,
            "metadata.type": METADATA_TYPE,
            DATE_FIELD: {"$lt": cutoff.isoformat()},
        }
    )
    return {
        "ran_at": now.isoformat(),
        "cutoff": cutoff.isoformat(),
        "deleted_count": result.deleted_count,
    }


def _load_last_cleanup_at(state_collection) -> datetime.datetime | None:
    doc = state_collection.find_one({"_id": SYNC_STATE_DOC_ID})
    if not doc:
        return None
    return _parse_datetime(doc.get("last_run_at"))


def _save_last_cleanup_at(state_collection, when: datetime.datetime) -> None:
    state_collection.update_one(
        {"_id": SYNC_STATE_DOC_ID},
        {"$set": {"last_run_at": when.isoformat()}},
        upsert=True,
    )


def maybe_run_daily_cleanup(corporate_id: str) -> Dict[str, Any] | None:
    """Run purge_old_records() for ONE tenant, only if CLEANUP_INTERVAL_HOURS
    have passed since it last ran for that tenant. The "last ran" timestamp
    is persisted in that tenant's own Mongo database (a dedicated sync_state
    doc), not just kept in memory, so restarting this process never causes
    it to purge more than once a day, and one tenant's schedule never
    affects another's."""
    database = _db_name(corporate_id)
    state_collection = get_mongo_collection(
        SYNC_STATE_COLLECTION, database, ensure_indexes=False
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    last_run = _load_last_cleanup_at(state_collection)
    if last_run is not None and (now - last_run) < datetime.timedelta(
        hours=CLEANUP_INTERVAL_HOURS
    ):
        return None

    result = purge_old_records(corporate_id)
    _save_last_cleanup_at(state_collection, now)
    return result


def _process_one_tenant(corporate_id: str, cycle_no: int) -> None:
    """Runs in its own thread — one tenant's sync + daily-cleanup check.
    Every exception is caught and logged here (never re-raised) so one
    tenant's failure can never affect the other tenants running concurrently
    in this same cycle, nor kill the daemon."""
    try:
        status = run_one_cycle(corporate_id)
        logger.info("[%s] cycle %s: %s", corporate_id, cycle_no, status)
    except Exception:  # noqa: BLE001 — one bad tenant must not kill the daemon
        logger.exception(
            "[%s] cycle %s failed; will retry after %ss",
            corporate_id, cycle_no, SLEEP_SECONDS,
        )

    try:
        cleanup_result = maybe_run_daily_cleanup(corporate_id)
        if cleanup_result is not None:
            logger.info("[%s] daily cleanup ran: %s", corporate_id, cleanup_result)
    except Exception:  # noqa: BLE001 — same rule: never kill the daemon
        logger.exception(
            "[%s] daily cleanup check failed on cycle %s (will retry next cycle)",
            corporate_id, cycle_no,
        )


def main() -> int:
    _setup_logging()
    logger.info(
        "Starting live trip sync: corporateids=%s every %ss (%s records/cycle "
        "per tenant, all tenants concurrent), daily purge of records older than "
        "%s month(s) every %sh",
        CORPORATE_IDS,
        SLEEP_SECONDS,
        PAGE_SIZE,
        RECENT_MONTHS,
        CLEANUP_INTERVAL_HOURS,
    )

    cycle_no = 0
    try:
        while True:
            cycle_no += 1
            cycle_start = time.perf_counter()

            # All tenants hit concurrently — cycle time is bounded by the
            # SLOWEST tenant, not the sum of every tenant's own time.
            with ThreadPoolExecutor(max_workers=len(CORPORATE_IDS)) as pool:
                futures = [
                    pool.submit(_process_one_tenant, corp_id, cycle_no)
                    for corp_id in CORPORATE_IDS
                ]
                for future in as_completed(futures):
                    future.result()  # re-raises only truly unexpected bugs in
                    # _process_one_tenant itself (it already catches its own
                    # per-tenant errors), so those still surface below.

            elapsed = time.perf_counter() - cycle_start
            logger.info(
                "cycle %s finished in %.2fs (%d tenants); sleeping %ss before next cycle",
                cycle_no,
                elapsed,
                len(CORPORATE_IDS),
                SLEEP_SECONDS,
            )
            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C) after %s cycles", cycle_no)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
