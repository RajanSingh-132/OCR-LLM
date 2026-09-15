"""
Live invoice sync daemon — polls the Avaal Web API every SLEEP_SECONDS and
keeps MongoDB in sync with the latest PAGE_SIZE invoice records.

  python -m app.sync.invoice_live_api
  (or)  python app/sync/invoice_live_api.py

Runs forever until Ctrl+C. Multi-tenant: syncs every corporate_id in
CORPORATE_IDS from ONE process — every cycle, all tenants are hit
CONCURRENTLY (ThreadPoolExecutor, one thread per tenant), so the cycle's
total time is bounded by the SLOWEST tenant, not the sum of all of them.
Same design as app/sync/order_live_api.py — see that file's docstring for
the full rationale (crash-safety, 30s cadence, daily purge, parallel
embedding). DB_NAME_OVERRIDES maps AFMQA -> chatbot_db; every other
corporate_id falls back to using its own id as the Mongo database name
(e.g. AFN01992 -> Mongo db "AFN01992") unless an override is added for it.

API contract (confirmed against a live browser request AND a raw response
dump, NOT guessed):
  POST https://beta.afmsuite.ai/api/Invoice/getinvoicelistdata
  header: corporateid: <tenant>
  body:   {"filter": {...}}   <- lowercase "filter", same as trip's API.
  response: a BARE JSON ARRAY of invoice records directly — e.g.
          [{...}, {...}, ...] — NOT wrapped in a "details"/"detailstrips"
          key like order/trip. Each record carries its own `totalcount`
          field (a per-row SQL COUNT(*) OVER() value, not very reliable —
          it did not match the actual row count in testing, so it's only
          used for logging/stats, never to decide when to stop paginating).

Field casing is inconsistent in this API's records (e.g. "InvoiceID" /
"InvoiceNumber" are PascalCase, "createdon" / "companycode" are lowercase) —
`_first_present()` tries several casings for the id/duplicate fields, same
approach as scripts/ingest/invoice_ingest.py.

USE_API_DATE_FILTER is OFF by default: the working curl had
IsDateFilterApplied=false and there's no separate "datetype" field on this
endpoint (unlike order/trip) to say which date FromDate/Todate applies to,
so we don't guess. PAGE_SIZE is tiny (15) and the client-side
`is_recent_record` check still filters out anything older than
RECENT_MONTHS, so correctness doesn't depend on the API-side filter.

Each cycle, PER TENANT (in parallel):
  1. POST getinvoicelistdata (header corporateid: <tenant>, body
     {"filter": {...}}, PageNo/pageno=1, PageSize/pagesize=PAGE_SIZE) -> the
     PAGE_SIZE most recent records.
  2. Keep only records whose date field (DATE_FIELDS, first present wins) is
     within the last calendar month (RECENT_MONTHS) — checked client-side.
  3. For each record, compare by invoiceid (ID_FIELDS, first present wins)
     against that tenant's collection:
       - invoiceid already stored -> delete the old doc, insert the fresh
         one (refresh, with a new embedding).
       - invoiceid not stored yet -> insert it as a new doc.
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
  - Logs go to both the console and ROTATING file `invoice_live_api.log`
    next to this script (so a long-running process doesn't grow the log
    forever). Every log line names its corporate_id, since one file now
    covers every tenant.
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

logger = logging.getLogger("app.sync.invoice_live_api")

# ===================== CONFIG — edit these =====================
API_BASE = "https://beta.afmsuite.ai"
LISTINVOICE_PATH = "/api/Invoice/getinvoicelistdata"
API_VERSION = "1.0"
# Every tenant this ONE process syncs, hit concurrently each cycle.
CORPORATE_IDS = [
    "AFMQA", "AFN01992", "AFN01856", "AFN01813", "AFN01619",
    "AFN01514", "AFN00292", "AFN00861", "AFN01801", "AFN00681",
]
EXTRA_HEADERS: Dict[str, str] = {}

# Body wrapper key is lowercase "filter" (confirmed from a working browser
# request). Every field below must be present — the working request sent
# BOTH PascalCase and lowercase duplicates for a few fields
# (SearchValue/searchvalue, PageNo/pageno, PageSize/pagesize,
# TotalCount/totalcount); keep both, we don't know which one the server
# actually reads. pageno/pagesize/PageNo/PageSize/*totalcount are
# overwritten per page automatically.
BASE_FILTER: Dict[str, Any] = {
    "IsFilterApplied": True,
    "ByDate": "",
    "ByInvoiceStatus": "",
    "ByCurrency": "",
    "ByAmount": "",
    "ByCustomer": "",
    "ByInvoiceDate": "",
    "InvoiceId": "",
    "OrderNumber": "",
    "InvoiceNumber": "",
    "CustomerOrderNumber": "",
    "CustomerCode": "",
    "FactoringCompanyCode": "",
    "AmountRange": "",
    "ChkInvoiceStatus": "",
    "IsDateFilterApplied": False,
    "FromDate": "",
    "Todate": "",
    "CompanyCode": "",
    "SearchValue": "",
    "searchvalue": "",
    "SortColumn": "ModifiedOn",
    "SortOrder": "desc",
    "BatchPrint": False,
    "TripNumber": "",
    "VinNumber": "",
    "TruckNumber": "",
    "TrailerNumber": "",
    "DriverName": "",
    "CarrierName": "",
    "IsExcept": 0,
    "EmailSent": "",
    "PickupRefNum": "",
}
PAGE_SIZE = 15
MAX_PAGES = 1             # only the latest page each cycle (PAGE_SIZE records)
REQUEST_TIMEOUT = 60      # seconds — must stay well under SLEEP_SECONDS

# See module docstring — off until a confirmed invoice date-filter target
# field is known.
USE_API_DATE_FILTER = False

DB_NAME_OVERRIDES = {"AFMQA": "chatbot_db"}
COLLECTION_NAME = "Avaal_invoice"
NAMESPACE = "avaal_invoices"
METADATA_TYPE = "avaal_invoice"
# Source field casing is inconsistent — try each in order, first present wins.
DUPLICATE_FIELDS = ("InvoiceNumber", "invoicenumber")
ID_FIELDS = ("InvoiceID", "invoiceid", "id")
# In case the server ever wraps records in a details-style key instead of a
# bare list (order/trip both do this — invoice currently does not).
DETAIL_KEYS = ("details", "Details", "detailsinvoices")

WITH_EMBEDDINGS = True

# Only keep records whose `createdon` is within the last calendar month.
FILTER_RECENT_ONLY = True
DATE_FIELDS = ("createdon", "CreatedOn", "InvoiceDate")
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
SYNC_STATE_DOC_ID = "invoice_live_api_cleanup"

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoice_live_api.log")
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


def _first_present(record: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _db_name(corporate_id: str) -> str:
    return DB_NAME_OVERRIDES.get(corporate_id.upper(), corporate_id)


def _api_url() -> str:
    return f"{API_BASE.rstrip('/')}{LISTINVOICE_PATH}"


def _parse_invoice_response(payload: Any) -> tuple[List[Dict[str, Any]], Any]:
    """Returns (records, reported_total).

    Primary shape (confirmed live): a bare JSON array of invoice dicts, each
    carrying its own `totalcount` field. Falls back to a details-style
    wrapper (dict, or a 1-element list wrapping a dict) in case the server
    ever changes shape, matching how order/trip's endpoints behave.
    """
    data = payload
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        if any(k in data[0] for k in DETAIL_KEYS):
            data = data[0]

    if isinstance(data, dict):
        for key in DETAIL_KEYS:
            if key in data:
                details = data[key]
                if isinstance(details, str):
                    details = json.loads(details) if details.strip() else []
                records = (
                    [r for r in details if isinstance(r, dict)]
                    if isinstance(details, list)
                    else []
                )
                total = data.get("total_count") or data.get("totalcount")
                return records, total
        return [], None

    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]
        total = None
        if records:
            total = (
                records[0].get("totalcount")
                or records[0].get("TotalCount")
                or records[0].get("total_count")
            )
        return records, total

    return [], None


def _build_filter(pageno: int, from_date: str | None, to_date: str | None) -> Dict[str, Any]:
    flt = {
        **BASE_FILTER,
        "PageNo": pageno,
        "pageno": pageno,
        "PageSize": PAGE_SIZE,
        "pagesize": PAGE_SIZE,
        "TotalCount": PAGE_SIZE,
        "totalcount": PAGE_SIZE,
    }
    if USE_API_DATE_FILTER and from_date and to_date:
        flt["IsDateFilterApplied"] = True
        flt["FromDate"] = from_date
        flt["Todate"] = to_date
    return flt


def fetch_latest_invoices(
    corporate_id: str, from_date: str | None = None, to_date: str | None = None
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch up to MAX_PAGES pages (PAGE_SIZE records each) from
    getinvoicelistdata for one tenant. Takes corporate_id as a parameter
    (not a module global) so this is safe to call concurrently from
    multiple tenant threads at once."""
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

        page_records, page_total = _parse_invoice_response(data)
        if reported_total is None:
            reported_total = page_total

        pages = pageno
        if not page_records:
            break
        all_records.extend(page_records)

        # `totalcount` on this endpoint has been unreliable in testing (it
        # did not match the actual row count returned) — only used for
        # logging/stats, never to decide when to stop paginating. The
        # page-size check below is the real stop condition.
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
            dt = None
            for fmt in (
                "%m/%d/%Y %I:%M:%S %p",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(raw[:19])
                except ValueError:
                    return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def is_recent_record(invoice: Dict[str, Any], cutoff: datetime.datetime) -> bool:
    created = _parse_datetime(_first_present(invoice, DATE_FIELDS))
    if created is None:
        return False
    return created >= cutoff


def build_page_content(invoice: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(invoice.keys()):
        value = invoice[key]
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
    invoice: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
    source_document: str,
    database: str,
    corporate_id: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    for key, value in invoice.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = NAMESPACE
    doc["page_content"] = build_page_content(invoice)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": METADATA_TYPE,
        "source_document": source_document,
        "collection": COLLECTION_NAME,
        "database": database,
        "corporate_id": corporate_id,
        "invoiceid": _first_present(invoice, ID_FIELDS),
        "invoicenumber": _first_present(invoice, DUPLICATE_FIELDS),
        "ingested_at": ingested_at,
        "embedding_dimensions": len(embedding),
        "structured": True,
    }
    return doc


def load_existing_keys(collection, fields) -> Set[str]:
    existing: Set[str] = set()
    projection: Dict[str, int] = {"metadata.invoiceid": 1}
    for field in fields:
        projection[field] = 1
    for doc in collection.find({}, projection):
        value = None
        for field in fields:
            value = doc.get(field)
            if value not in (None, ""):
                break
        if value in (None, ""):
            value = (doc.get("metadata") or {}).get("invoiceid")
        if value not in (None, ""):
            existing.add(str(value).strip())
    return existing


def run_one_cycle(corporate_id: str) -> Dict[str, Any]:
    """Fetch the latest PAGE_SIZE invoices for ONE tenant and sync them into
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
    records, fetch_stats = fetch_latest_invoices(corporate_id, from_date, to_date)
    t_fetch = time.perf_counter() - t0

    if not records:
        return {
            "ok": True,
            "corporate_id": corporate_id,
            "database": database,
            "note": "getinvoicelistdata returned no records this cycle",
            **fetch_stats,
            "timing_seconds": {"fetch": round(t_fetch, 2), "total": round(t_fetch, 2)},
        }

    # --- transform / filter: decide refresh (invoiceid match) vs new ------
    t1 = time.perf_counter()
    collection = get_mongo_collection(COLLECTION_NAME, database, ensure_indexes=True)
    existing_invoiceids: Set[str] = load_existing_keys(collection, ID_FIELDS)

    embeddings = get_embeddings() if WITH_EMBEDDINGS else None

    pending: List[Dict[str, Any]] = []
    ids_to_refresh: List[Any] = []
    skipped_old = 0
    refreshed_count = 0
    new_count = 0
    for invoice in records:
        if recent_cutoff is not None and not is_recent_record(invoice, recent_cutoff):
            skipped_old += 1
            continue
        key_value = _first_present(invoice, ID_FIELDS)
        key_str = str(key_value).strip() if key_value not in (None, "") else ""
        if key_str and key_str in existing_invoiceids:
            ids_to_refresh.append(key_value)
            refreshed_count += 1
        else:
            new_count += 1
        pending.append(invoice)
    t_transform = time.perf_counter() - t1

    # --- delete stale matches, then embed (optional) + insert everything --
    t2 = time.perf_counter()
    deleted = 0
    if ids_to_refresh:
        del_result = collection.delete_many(
            {"namespace": NAMESPACE, "metadata.invoiceid": {"$in": ids_to_refresh}}
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
        for invoice, vector in zip(chunk, vectors):
            batch.append(
                build_document(
                    invoice, vector, ingested_at, source_document, database, corporate_id
                )
            )
        if len(batch) >= INSERT_BATCH_SIZE:
            flush()
    flush()
    t_insert = time.perf_counter() - t2

    collection.create_index([("namespace", 1), ("metadata.invoiceid", 1)])
    collection.create_index([("namespace", 1), ("metadata.invoicenumber", 1)])

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
    # createdon here has no timezone suffix in the source data (naive local
    # string, e.g. "2026-09-03T12:43:44.677377") — stored verbatim as-is, so
    # compare against the same naive-looking cutoff string (UTC clock, no
    # offset) to match how it's actually stored.
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%f")
    result = collection.delete_many(
        {
            "namespace": NAMESPACE,
            "metadata.type": METADATA_TYPE,
            "$or": [
                {"createdon": {"$lt": cutoff.isoformat()}},
                {"createdon": {"$lt": cutoff_str}},
            ],
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
        "Starting live invoice sync: corporateids=%s every %ss (%s records/cycle "
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
