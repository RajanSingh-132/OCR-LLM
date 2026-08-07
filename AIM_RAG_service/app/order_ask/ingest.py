"""
Ingest orderdata.txt into Mongo Avaal_db as structured documents.

Each order = 1 document with:
- all business fields at top level (for Atlas dropdown + $sum calculations)
- namespace / source metadata
- page_content (text for RAG)
- embedding (Bedrock 1024-d) for semantic search
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List

from app.order_ask.config import (
    AVAAL_COLLECTION_NAME,
    AVAAL_NAMESPACE,
    AVAAL_ORDERS_JSON_PATH,
    AVAAL_SOURCE_DOCUMENT,
)
from app.embedding_client import get_models
from app.mongo_client import MONGO_DB_NAME, _to_python_types, get_mongo_collection

logger = logging.getLogger("order_ask.ingest")

EMBED_BATCH_SIZE = int(os.environ.get("AVAAL_EMBED_BATCH_SIZE", "25"))
INSERT_BATCH_SIZE = int(os.environ.get("AVAAL_INSERT_BATCH_SIZE", "100"))


def _load_order_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Orders file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        # Either bare list of orders, or wrapper list like getorderlist.txt
        if payload and isinstance(payload[0], dict) and "details" in payload[0]:
            details = payload[0]["details"]
        elif payload and isinstance(payload[0], dict) and "orderid" in payload[0]:
            return payload
        else:
            details = payload
    elif isinstance(payload, dict):
        details = payload.get("details", payload)
    else:
        raise ValueError("Unsupported orders file format")

    if isinstance(details, str):
        details = json.loads(details)

    if not isinstance(details, list):
        raise ValueError("Orders 'details' must be a list of records")

    records = [r for r in details if isinstance(r, dict)]
    logger.info("Loaded %s order records from %s", len(records), path)
    return records


def _build_page_content(order: Dict[str, Any]) -> str:
    """Compact searchable text; full structured fields live on the document itself."""
    preferred = [
        "orderid",
        "ordernumber",
        "tempordernumber",
        "orderdate",
        "customername",
        "customercode",
        "companycode",
        "salesmanname",
        "salesmancode",
        "currencycode",
        "totalfreight",
        "grosstotalfreight",
        "freightcharges",
        "fuelcharges",
        "othercharges",
        "taxes",
        "totaltaxamount",
        "pretaxamount",
        "offeredamount",
        "orderstatus",
        "statuscode",
        "loadtypelucode",
        "pickuplocationname",
        "pickupfulladdress",
        "pickupdate",
        "deliverylocationname",
        "deliveryfulladdress",
        "deliverydate",
        "commodityname",
        "weight",
        "weightunit",
        "distance",
        "distanceunit",
        "ordernotes",
    ]
    lines = []
    for key in preferred:
        if key in order and order[key] not in (None, "", [], {}):
            lines.append(f"{key}: {order[key]}")
    # Keep a truncated full JSON for completeness in RAG
    full_json = json.dumps(order, ensure_ascii=False, default=str)
    if len(full_json) > 6000:
        full_json = full_json[:6000] + "..."
    lines.append("full_order_json: " + full_json)
    return "\n".join(lines)


def _build_document(
    order: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    # Spread all order fields at top level (structured / calculation-ready)
    for key, value in order.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = AVAAL_NAMESPACE
    doc["page_content"] = _build_page_content(order)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": "avaal_order",
        "source_document": AVAAL_SOURCE_DOCUMENT,
        "collection": AVAAL_COLLECTION_NAME,
        "database": MONGO_DB_NAME,
        "orderid": order.get("orderid"),
        "ordernumber": order.get("ordernumber"),
        "ingested_at": ingested_at,
        "embedding_dimensions": len(embedding),
        "structured": True,
    }
    return doc


def ingest_avaal_orders(
    path: str = None,
    replace_namespace: bool = True,
    with_embeddings: bool = True,
) -> Dict[str, Any]:
    path = path or AVAAL_ORDERS_JSON_PATH
    records = _load_order_records(path)
    if not records:
        return {"ok": False, "error": "No order records found", "path": path}

    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)

    if replace_namespace:
        deleted = collection.delete_many({"namespace": AVAAL_NAMESPACE}).deleted_count
        logger.info("Cleared %s existing docs in namespace=%s", deleted, AVAAL_NAMESPACE)

    embeddings = None
    if with_embeddings:
        embeddings, _ = get_models()

    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    inserted = 0
    batch_docs: List[Dict[str, Any]] = []

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        chunk = records[start : start + EMBED_BATCH_SIZE]
        texts = [_build_page_content(order) for order in chunk]

        if with_embeddings and embeddings is not None:
            vectors = embeddings.embed_documents(texts)
        else:
            vectors = [[] for _ in chunk]

        for order, vector in zip(chunk, vectors):
            batch_docs.append(_build_document(order, vector, ingested_at))

        if len(batch_docs) >= INSERT_BATCH_SIZE:
            collection.insert_many(batch_docs, ordered=False)
            inserted += len(batch_docs)
            logger.info("Inserted %s / %s", inserted, len(records))
            batch_docs = []

        print(
            f"[Avaal ingest] processed {min(start + len(chunk), len(records))} / {len(records)}"
        )

    if batch_docs:
        collection.insert_many(batch_docs, ordered=False)
        inserted += len(batch_docs)

    # Helpful indexes for calculations / lookups
    collection.create_index([("namespace", 1), ("orderid", 1)])
    collection.create_index([("namespace", 1), ("ordernumber", 1)])
    collection.create_index([("namespace", 1), ("customercode", 1)])
    collection.create_index([("namespace", 1), ("taxes", 1)])
    collection.create_index([("namespace", 1), ("totalfreight", 1)])
    collection.create_index([("namespace", 1), ("grosstotalfreight", 1)])

    stored = collection.count_documents({"namespace": AVAAL_NAMESPACE, "metadata.type": "avaal_order"})
    sample = collection.find_one(
        {"namespace": AVAAL_NAMESPACE, "metadata.type": "avaal_order"},
        {"embedding": 0},
    )

    status = {
        "ok": True,
        "database": MONGO_DB_NAME,
        "collection": AVAAL_COLLECTION_NAME,
        "namespace": AVAAL_NAMESPACE,
        "source_path": path,
        "source_document": AVAAL_SOURCE_DOCUMENT,
        "records_loaded": len(records),
        "documents_inserted": inserted,
        "documents_in_namespace": stored,
        "with_embeddings": with_embeddings,
        "sample_orderid": (sample or {}).get("orderid"),
        "sample_top_level_fields": sorted(
            [k for k in (sample or {}).keys() if k not in ("_id", "page_content", "embedding", "metadata", "namespace")]
        )[:25],
    }
    logger.info("Ingest complete: %s", status)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ingest_avaal_orders(with_embeddings=True, replace_namespace=True)
    print("\n=== Avaal ingest result ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("===========================\n")
