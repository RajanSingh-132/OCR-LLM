"""
Live order ingest — pulls orders straight from the Avaal Web API and stores
them in MongoDB (no file needed).

  python -m scripts.ingest.order_api_ingest

Flow:
  1. POST {API_BASE}/api/Order/listorder  (header: corporateid: <CORPORATE_ID>,
     body {"Filter": {...}} with isdate/datetype=CD/fromdate/todate set to the
     last-month window), paging through every result page.
  2. Parse the stringified `details` JSON array from each page.
  3. Client-side safety filter: keep only records whose `createdon` is on/after
     the exact cutoff (RECENT_MONTHS back from now).
  4. For each record, compare by `orderid` against the target collection:
       - orderid already stored  -> delete the old doc, then insert the fresh
         one (refresh — new embedding included).
       - orderid not stored yet  -> insert it as a new doc.
  5. Store the resulting documents in Mongo, each with a Titan embedding.

WITH_EMBEDDINGS controls whether a Titan embedding is computed and stored per
document (set False for a pure fetch+insert timing run). The terminal always
prints a timing breakdown (fetch / transform / delete+embed+insert / total)
at the end regardless of the setting.

The MongoDB database is the corporate id itself (AFMQA -> chatbot_db via the
override), matching app/tenants/mapping.py.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.embedding_client import get_embeddings
from app.mongo_client import _to_python_types, get_mongo_collection

logger = logging.getLogger("scripts.ingest.order_api")

# ===================== CONFIG — edit these =====================
API_BASE = "http://173.209.153.108:5000"
LISTORDER_PATH = "/api/Order/listorder"
API_VERSION = "1.0"
# corporateid header == the tenant's Postgres DB name on the API server, and
# also the MongoDB database we write into (AFMQA -> chatbot_db via override).
# `username` / `usercode` in BASE_FILTER below MUST belong to this same tenant
# or listorder returns 0 rows.
CORPORATE_ID = "AFMQA"
# Extra HTTP headers if the server needs them (auth token etc.). Confirmed by
# a working browser call that only `corporateid` + content-type are required
# — no bearer token, no Origin/Referer.
EXTRA_HEADERS: Dict[str, str] = {}
# `Filter` payload sent to listorder — capital "Filter" is required by the API
# (lowercase "filter" silently no-ops and returns total_count: 0). Every field
# below must be present (the DTO has additionalProperties: false and several
# non-nullable fields like orderoutid), so keep the empty-string / -1 filler
# values even for filters you don't use. Pagination keys (pageno/pagesize) are
# overwritten automatically per page.
BASE_FILTER: Dict[str, Any] = {
    "dataviewtype": "D",
    "companycode": "",
    "ordernumber": "",
    "customercode": "",
    "salesmancode": "",
    "customerordernumber": "",
    "orderstatus": "",
    "shipmenttype": "",
    "pickuplocation": "",
    "pickupcity": "",
    "deliverycity": "",
    "isdate": False,
    "datetype": "ORDERDATE",
    "fromdate": "",
    "todate": "",
    "pickupstatecode": "",
    "deliverystatecode": "",
    "status": "A",
    "currencycode": "",
    "csa": "",
    "hazmat": "",
    "overdimension": "",
    "searchvalue": "",
    "orderformtype": "",
    "equipmenttype": "",
    "usertype": "A",
    "username": "Rahul Agrawal",
    "sortcolumn": "ModifiedOn",
    "sortorder": "DESC",
    "accountingstatus": "",
    "statuscondition": "",
    "pickupcountrycode": "",
    "deliverycountrycode": "",
    "usercode": "USR00001",
    "orderoutid": -1,
    "carriercode": "",
    "tripnumber": "",
    "pickuprefnum": "",
    "pendingaccessorial": False,
    "advancesearchwhere": "",
}
PAGE_SIZE = 15
# Capped at 1 page (= PAGE_SIZE records, 15 total) for this refresh/embedding
# test run. Raise MAX_PAGES back up (e.g. 500) once you want the full
# last-month sync instead of a fixed 15-record sample.
MAX_PAGES = 1
REQUEST_TIMEOUT = 180    # seconds per API call

# Push the "last one month" window into the API request instead of pulling the
# tenant's entire order history (AFN00342 has ~207k). The API only honours two
# date types: "CD" = createdon, "OD" = orderdate. "CREATEDON"/"MODIFIEDON" are
# silently ignored (they return everything), so keep this as "CD".
USE_API_DATE_FILTER = True
API_DATE_TYPE = "CD"

DB_NAME_OVERRIDES = {"AFMQA": "chatbot_db"}
COLLECTION_NAME = "Avaal_order"
NAMESPACE = "avaal_orders"
METADATA_TYPE = "avaal_order"
DUPLICATE_FIELD = "ordernumber"
ID_FIELD = "orderid"

WITH_EMBEDDINGS = True    # <-- on: every fetched record gets a Titan embedding

# Only keep records whose `createdon` is within the last calendar month.
FILTER_RECENT_ONLY = True
DATE_FIELD = "createdon"
RECENT_MONTHS = 1

INSERT_BATCH_SIZE = 200
EMBED_BATCH_SIZE = 25

# Dump the combined raw API records to this file for inspection (None = skip).
# Off by default now — MongoDB itself holds every inserted record (orderid,
# page_content, embedding, metadata), so there's nothing this local file adds
# that Mongo doesn't already have. Set a path here again if you ever want a
# quick local copy without opening Atlas/Compass.
RAW_DUMP_PATH = None

# Titan embedding guard rails (only used when WITH_EMBEDDINGS = True).
MAX_EMBED_TOKENS = 8192
EST_CHARS_PER_TOKEN = 1.4
MAX_EMBED_CHARS = int(MAX_EMBED_TOKENS * EST_CHARS_PER_TOKEN * 0.85)
# Bedrock's Titan embedding endpoint takes ONE text per invoke_model call —
# there is no multi-text batch API. embed_documents() therefore calls it once
# per text, sequentially. We fan those calls out across threads instead (they
# are I/O-bound network calls, so GIL isn't a bottleneck) to run them
# concurrently. Keep this modest — too high risks AWS Bedrock throttling
# (ThrottlingException) on the account's requests-per-second limit.
EMBED_MAX_WORKERS = 10
# ===============================================================


def _db_name(corporate_id: str) -> str:
    return DB_NAME_OVERRIDES.get(corporate_id.upper(), corporate_id)


def _api_url() -> str:
    return f"{API_BASE.rstrip('/')}{LISTORDER_PATH}"


def _parse_details_payload(payload: Any) -> List[Dict[str, Any]]:
    """Pull the list of order dicts out of one listorder response."""
    if isinstance(payload, dict):
        details = payload.get("details", payload.get("Details"))
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


def _build_filter(pageno: int, from_date: str | None, to_date: str | None) -> Dict[str, Any]:
    flt = {**BASE_FILTER, "pageno": pageno, "pagesize": PAGE_SIZE}
    if USE_API_DATE_FILTER and from_date and to_date:
        flt["isdate"] = True
        flt["datetype"] = API_DATE_TYPE
        flt["fromdate"] = from_date
        flt["todate"] = to_date
    return flt


def fetch_all_orders(
    from_date: str | None = None, to_date: str | None = None
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Page through listorder and return (records, stats)."""
    session = requests.Session()
    headers = {
        "Content-Type": "application/json; ver=1.0",
        "Accept": "application/json",
        "corporateid": CORPORATE_ID,
        **EXTRA_HEADERS,
    }
    params = {"api-version": API_VERSION}
    url = _api_url()

    all_records: List[Dict[str, Any]] = []
    reported_total = None
    pages = 0

    for pageno in range(1, MAX_PAGES + 1):
        body = {"Filter": _build_filter(pageno, from_date, to_date)}
        resp = session.post(
            url, params=params, headers=headers, json=body, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        if reported_total is None and isinstance(data, dict):
            reported_total = data.get("total_count")

        page_records = _parse_details_payload(data)
        pages = pageno
        logger.info(
            "page %s: %s records (running total %s / %s)",
            pageno,
            len(page_records),
            len(all_records) + len(page_records),
            reported_total,
        )
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


def is_recent_record(order: Dict[str, Any], cutoff: datetime.datetime) -> bool:
    created = _parse_datetime(order.get(DATE_FIELD))
    if created is None:
        return False
    return created >= cutoff


def build_page_content(order: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(order.keys()):
        value = order[key]
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
    """Embed every text concurrently instead of one-by-one.

    Bedrock's Titan endpoint only accepts one text per call, so
    `embeddings.embed_documents()` loops sequentially inside langchain_aws —
    15 texts there means 15 network round-trips back-to-back. We fan the same
    15 calls out across a thread pool instead: they're independent I/O-bound
    HTTP requests, so running them concurrently collapses the wall-clock time
    from "sum of every call" down to roughly "the slowest single call" (bound
    by EMBED_MAX_WORKERS and whatever AWS Bedrock throttles at).
    """
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
            results[i] = future.result()  # re-raises if that call ultimately failed

    return results  # type: ignore[return-value]


def build_document(
    order: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
    source_document: str,
    database: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    for key, value in order.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = NAMESPACE
    doc["page_content"] = build_page_content(order)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": METADATA_TYPE,
        "source_document": source_document,
        "collection": COLLECTION_NAME,
        "database": database,
        "corporate_id": CORPORATE_ID,
        "orderid": order.get(ID_FIELD),
        "ordernumber": order.get(DUPLICATE_FIELD),
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


def ingest_orders_from_api() -> Dict[str, Any]:
    database = _db_name(CORPORATE_ID)
    source_document = f"api:{_api_url()}?corporateid={CORPORATE_ID}"

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
        logger.info(
            "Requesting %s in [%s .. %s]; client-side cutoff %s >= %s",
            API_DATE_TYPE,
            from_date,
            to_date,
            DATE_FIELD,
            recent_cutoff.isoformat(),
        )

    t0 = time.perf_counter()
    records, fetch_stats = fetch_all_orders(from_date, to_date)
    t_fetch = time.perf_counter() - t0

    if RAW_DUMP_PATH:
        try:
            with open(RAW_DUMP_PATH, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, default=str)
            logger.info("Raw pull written to %s", RAW_DUMP_PATH)
        except OSError as exc:
            logger.warning("Could not write raw dump: %s", exc)

    if not records:
        return {
            "ok": False,
            "error": "listorder returned no records",
            "hint": (
                "listorder returned an empty 'details' list. Check that "
                "CORPORATE_ID, BASE_FILTER['username'] and BASE_FILTER['usercode'] "
                "all belong to the same tenant, and that the body uses capital "
                "'Filter'. A wrong datetype ('CREATEDON' instead of 'CD') is "
                "ignored, not errored."
            ),
            "database": database,
            **fetch_stats,
        }

    # --- transform / filter: decide refresh (orderid match) vs new ---------
    t1 = time.perf_counter()
    collection = get_mongo_collection(COLLECTION_NAME, database, ensure_indexes=True)

    existing_orderids: Set[str] = load_existing_keys(collection, ID_FIELD)
    logger.info("Loaded %s existing %s values", len(existing_orderids), ID_FIELD)

    embeddings = get_embeddings() if WITH_EMBEDDINGS else None

    pending: List[Dict[str, Any]] = []
    ids_to_refresh: List[Any] = []   # raw orderid values whose old doc gets deleted first
    skipped_old = 0
    refreshed_count = 0
    new_count = 0
    for order in records:
        if recent_cutoff is not None and not is_recent_record(order, recent_cutoff):
            skipped_old += 1
            continue
        key_value = order.get(ID_FIELD)
        key_str = str(key_value).strip() if key_value not in (None, "") else ""
        if key_str and key_str in existing_orderids:
            # Same orderid already stored -> drop the old doc; this record
            # replaces it with a fresh embedding below.
            ids_to_refresh.append(key_value)
            refreshed_count += 1
        else:
            new_count += 1
        pending.append(order)
    t_transform = time.perf_counter() - t1

    # --- delete stale matches, then embed (optional) + insert everything ---
    t2 = time.perf_counter()
    deleted = 0
    if ids_to_refresh:
        del_result = collection.delete_many(
            {"namespace": NAMESPACE, ID_FIELD: {"$in": ids_to_refresh}}
        )
        deleted = del_result.deleted_count
        logger.info(
            "Deleted %s existing docs (orderid match) before re-insert", deleted
        )

    inserted = 0
    batch: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal inserted, batch
        if not batch:
            return
        collection.insert_many(batch, ordered=False)
        inserted += len(batch)
        logger.info("Inserted %s / %s", inserted, len(pending))
        batch = []

    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        chunk = pending[start : start + EMBED_BATCH_SIZE]
        if embeddings is not None:
            texts = [_truncate_for_embedding(build_page_content(o)) for o in chunk]
            vectors = _embed_texts(embeddings, texts)
        else:
            vectors = [[] for _ in chunk]
        for order, vector in zip(chunk, vectors):
            batch.append(
                build_document(order, vector, ingested_at, source_document, database)
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
        "corporate_id": CORPORATE_ID,
        "database": database,
        "collection": COLLECTION_NAME,
        "namespace": NAMESPACE,
        "with_embeddings": WITH_EMBEDDINGS,
        **fetch_stats,
        "documents_skipped_old": skipped_old,
        "documents_new": new_count,
        "documents_matched_refreshed": refreshed_count,
        "documents_deleted_before_refresh": deleted,
        "documents_inserted": inserted,
        "documents_in_namespace": stored,
        "recent_cutoff": (
            recent_cutoff.isoformat() if recent_cutoff is not None else None
        ),
        "timing_seconds": {
            "fetch": round(t_fetch, 2),
            "transform_filter": round(t_transform, 2),
            "delete_embed_insert": round(t_insert, 2),
            "total": round(total, 2),
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status = ingest_orders_from_api()

    print("\n=== Order API ingest ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("========================\n")
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
