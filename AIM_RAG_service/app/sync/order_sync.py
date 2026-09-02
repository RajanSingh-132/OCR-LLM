"""Live Avaal API → Mongo sync for the orders domain.

    sync_orders(corporate_id, mode="incremental" | "full", dry_run=False)

- incremental: page ModifiedOn-DESC, stop at the stored cursor, upsert changed rows
- full:        page everything, upsert, then mark rows not seen as `is_stale`

Reuses the tenant mapping, the read-path collection, and the document builder so
the query planner / analytics / RAG keep working unchanged.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from pymongo import ReplaceOne

from app.embedding_client import get_embeddings
from app.mongo_client import get_mongo_collection
from app.sync.avaal_client import AvaalClient
from app.sync.documents import (
    build_order_document,
    build_page_content,
    content_hash,
    truncate_for_embedding,
)
from app.sync.sync_state import (
    acquire_lock,
    get_cursor,
    record_run,
    release_lock,
    set_cursor,
)
from app.tenants.mapping import get_tenant_config

logger = logging.getLogger("sync.order_sync")

DOMAIN = "orders"
_SOURCE = "live_api"


def _ensure_indexes(collection) -> None:
    try:
        collection.create_index([("namespace", 1), ("orderid", 1)])
        collection.create_index([("namespace", 1), ("ordernumber", 1)])
        collection.create_index([("namespace", 1), ("modifiedon", -1)])
    except Exception as exc:  # index creation must never abort a sync
        logger.warning("index ensure failed: %s", exc)


def _embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return get_embeddings().embed_documents([truncate_for_embedding(t) for t in texts])


def _process_batch(
    collection,
    records: List[dict],
    *,
    namespace: str,
    metadata_type: str,
    synced_at: datetime,
    counts: Dict[str, int],
    dry_run: bool,
) -> None:
    """Upsert one page of records; embed only the ones whose text changed."""
    prepared = []
    for rec in records:
        oid = rec.get("orderid")
        if oid is None:
            continue
        text = build_page_content(rec)
        prepared.append((oid, rec, text, content_hash(text)))

    if not prepared:
        return

    ids = [p[0] for p in prepared]
    prev_by_id = {
        d["orderid"]: d
        for d in collection.find(
            {"namespace": namespace, "orderid": {"$in": ids}},
            {"orderid": 1, "content_hash": 1, "embedding": 1},
        )
    }

    to_embed_idx = [
        i for i, (oid, _r, _t, h) in enumerate(prepared)
        if (prev_by_id.get(oid) or {}).get("content_hash") != h
    ]
    embeddings = (
        _embed([prepared[i][2] for i in to_embed_idx]) if (to_embed_idx and not dry_run) else []
    )
    emb_by_idx = dict(zip(to_embed_idx, embeddings))

    ops = []
    for i, (oid, rec, text, h) in enumerate(prepared):
        prev = prev_by_id.get(oid)
        changed = (prev or {}).get("content_hash") != h
        if changed:
            counts["reembedded"] += 1 if not dry_run else 0
            emb = emb_by_idx.get(i) or []
        else:
            emb = (prev or {}).get("embedding") or []
            counts["unchanged"] += 1
        counts["updated" if prev else "inserted"] += 1
        if dry_run:
            continue
        doc = build_order_document(
            rec,
            embedding=emb,
            namespace=namespace,
            metadata_type=metadata_type,
            source_document=_SOURCE,
            synced_at=synced_at,
            page_content=text,
            hash_value=h,
            is_stale=bool(rec.get("isdeleted")) or rec.get("isactive") is False,
        )
        ops.append(
            ReplaceOne({"namespace": namespace, "orderid": oid}, doc, upsert=True)
        )

    if ops:
        collection.bulk_write(ops, ordered=False)


def sync_orders(
    corporate_id: str,
    *,
    mode: str = "incremental",
    dry_run: bool = False,
) -> Dict[str, Any]:
    mode = "full" if str(mode).lower() == "full" else "incremental"
    tenant = get_tenant_config(corporate_id)
    db_name = tenant.database
    namespace = tenant.namespace_for(DOMAIN)
    metadata_type = tenant.metadata_type_for(DOMAIN)
    collection = get_mongo_collection(tenant.collection_for(DOMAIN), db_name)

    if not dry_run and not acquire_lock(db_name, corporate_id, DOMAIN):
        logger.info("sync_orders: %s already running — skipped", corporate_id)
        return {"status": "skipped_locked", "corporate_id": corporate_id, "mode": mode}

    started = time.monotonic()
    synced_at = datetime.now(timezone.utc)
    counts = {
        "pages": 0, "fetched": 0, "inserted": 0, "updated": 0,
        "reembedded": 0, "unchanged": 0, "stale_marked": 0,
    }
    cursor = get_cursor(db_name, corporate_id, DOMAIN) if mode == "incremental" else None
    max_modified = cursor or ""
    seen_ordernumbers: set = set()
    status, error = "ok", None

    try:
        if not dry_run:
            _ensure_indexes(collection)
        client = AvaalClient()
        stop = False
        for batch in client.iter_pages(
            corporate_id, modified_since=(cursor if mode == "incremental" else None)
        ):
            counts["pages"] += 1
            fresh: List[dict] = []
            for rec in batch:
                modon = str(rec.get("modifiedon") or "")
                if mode == "incremental" and cursor and modon and modon <= cursor:
                    stop = True
                    break
                fresh.append(rec)
                counts["fetched"] += 1
                if rec.get("ordernumber") is not None:
                    seen_ordernumbers.add(str(rec.get("ordernumber")))
                if modon > max_modified:
                    max_modified = modon
            _process_batch(
                collection, fresh,
                namespace=namespace, metadata_type=metadata_type,
                synced_at=synced_at, counts=counts, dry_run=dry_run,
            )
            if stop:
                break

        if mode == "full" and not dry_run and seen_ordernumbers:
            res = collection.update_many(
                {
                    "namespace": namespace,
                    "ordernumber": {"$nin": list(seen_ordernumbers)},
                    "is_stale": {"$ne": True},
                },
                {"$set": {
                    "is_stale": True,
                    "stale_reason": "not_in_full_sync",
                    "synced_at": synced_at,
                }},
            )
            counts["stale_marked"] = res.modified_count

        if not dry_run and max_modified:
            set_cursor(db_name, corporate_id, DOMAIN, max_modified)
    except Exception as exc:  # noqa: BLE001 — one tenant must not crash the scheduler
        logger.exception("sync_orders failed for %s", corporate_id)
        status, error = "error", str(exc)
    finally:
        if not dry_run:
            release_lock(db_name, corporate_id, DOMAIN)

    duration_ms = int((time.monotonic() - started) * 1000)
    if not dry_run:
        record_run(
            db_name, corporate_id, DOMAIN,
            mode=mode, status=status, counts=counts,
            duration_ms=duration_ms, error=error,
        )
    result = {
        "status": status, "mode": mode, "corporate_id": corporate_id,
        "dry_run": dry_run, "cursor": max_modified or cursor,
        "duration_ms": duration_ms, "error": error, **counts,
    }
    logger.info("sync_orders done: %s", result)
    return result
