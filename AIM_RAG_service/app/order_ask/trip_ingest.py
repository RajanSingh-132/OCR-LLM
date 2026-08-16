"""
Ingest Tripdata.txt into Mongo Avaal_trip_db as structured documents.

Each trip = 1 document with:
- all business fields at top level (for filters / joins on orderid)
- namespace / source metadata
- page_content (text for RAG)
- embedding (Bedrock 1024-d) for semantic search

Tripdata.txt shape:
[
  {
    "total_count": 1000,
    "detailstrips": "<JSON-stringified array of trip objects>",
    ...
  }
]
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.embedding_client import get_models
from app.mongo_client import MONGO_DB_NAME, _to_python_types, get_mongo_collection
from app.order_ask.config import (
    AVAAL_TRIP_COLLECTION_NAME,
    AVAAL_TRIP_NAMESPACE,
    AVAAL_TRIP_SOURCE_DOCUMENT,
    AVAAL_TRIPS_JSON_PATH,
)

logger = logging.getLogger("order_ask.trip_ingest")

EMBED_BATCH_SIZE = int(os.environ.get("AVAAL_TRIP_EMBED_BATCH_SIZE", "25"))
INSERT_BATCH_SIZE = int(os.environ.get("AVAAL_TRIP_INSERT_BATCH_SIZE", "100"))


def _fix_invalid_json_escapes(text: str) -> str:
    """Trip exports sometimes contain \\& and other illegal JSON escapes."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in '"\\/bfnrt':
                out.append(ch)
                out.append(nxt)
                i += 2
            elif (
                nxt == "u"
                and i + 5 < n
                and all(c in "0123456789abcdefABCDEF" for c in text[i + 2 : i + 6])
            ):
                out.append(text[i : i + 6])
                i += 6
            else:
                # drop the backslash, keep the character (e.g. \& -> &)
                out.append(nxt)
                i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _load_json_payload(path: str) -> Any:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_fix_invalid_json_escapes(raw))


def _parse_detailstrips(detailstrips: Any) -> List[Dict[str, Any]]:
    if detailstrips is None:
        return []
    if isinstance(detailstrips, list):
        return [r for r in detailstrips if isinstance(r, dict)]
    if isinstance(detailstrips, str):
        text = detailstrips.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = json.loads(_fix_invalid_json_escapes(text))
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    raise ValueError("Unsupported detailstrips format")


def _load_trip_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trips file not found: {path}")

    payload = _load_json_payload(path)

    if isinstance(payload, list):
        if not payload:
            return []
        first = payload[0]
        if isinstance(first, dict) and (
            "detailstrips" in first or "details" in first or "tripid" in first
        ):
            if "tripid" in first or "tripnumber" in first:
                records = [r for r in payload if isinstance(r, dict)]
            else:
                records = []
                for wrapper in payload:
                    if not isinstance(wrapper, dict):
                        continue
                    records.extend(
                        _parse_detailstrips(
                            wrapper.get("detailstrips")
                            if wrapper.get("detailstrips") is not None
                            else wrapper.get("details")
                        )
                    )
        else:
            raise ValueError("Unsupported trips file list format")
    elif isinstance(payload, dict):
        if "detailstrips" in payload or "details" in payload:
            records = _parse_detailstrips(
                payload.get("detailstrips")
                if payload.get("detailstrips") is not None
                else payload.get("details")
            )
        elif "tripid" in payload or "tripnumber" in payload:
            records = [payload]
        else:
            raise ValueError("Unsupported trips file object format")
    else:
        raise ValueError("Unsupported trips file format")

    logger.info("Loaded %s trip records from %s", len(records), path)
    return records


def _normalize_order_link_fields(trip: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure join-friendly orderid / orderids / ordernumber fields."""
    out = dict(trip)
    orderids_raw = out.get("orderids")
    order_id_list: List[str] = []
    if isinstance(orderids_raw, str) and orderids_raw.strip():
        order_id_list = [p.strip() for p in re.split(r"[,\s]+", orderids_raw) if p.strip()]
    elif isinstance(orderids_raw, list):
        order_id_list = [str(x).strip() for x in orderids_raw if str(x).strip()]

    if order_id_list:
        out["orderids_list"] = order_id_list
        # Primary link used for simple joins / indexes
        if not out.get("orderid"):
            out["orderid"] = order_id_list[0]

    # Keep numeric ids as-is; also store string copies for flexible matching
    if out.get("tripid") is not None and "tripid_str" not in out:
        out["tripid_str"] = str(out["tripid"])
    if out.get("orderid") is not None and "orderid_str" not in out:
        out["orderid_str"] = str(out["orderid"])
    return out


def _build_page_content(trip: Dict[str, Any]) -> str:
    preferred = [
        "tripid",
        "tripnumber",
        "tripstatus",
        "triptype",
        "triptypemain",
        "orderids",
        "orderid",
        "ordernumber",
        "customername",
        "customercodes",
        "companycode",
        "companyname",
        "carriername",
        "carriercode",
        "salesmannames",
        "salesmancodes",
        "pickuplocationname",
        "pickupfulladdress",
        "pickupcity",
        "pickupstate",
        "pickupcountry",
        "firstpickupdate",
        "deliverylocationname",
        "deliveryfulladdress",
        "deliverycity",
        "deliverystate",
        "deliverycountry",
        "lastdeliverydate",
        "commodity",
        "equipmenttype",
        "totaldistance",
        "triptotaldistance",
        "totalloaddistance",
        "totalemptydistance",
        "distanceunit",
        "totalweight",
        "totalquantity",
        "rate",
        "ratetypevalue",
        "offeredamount",
        "totalofferedamount",
        "totaltaxamount",
        "totaladdition",
        "totaldeduction",
        "trucknumber",
        "truckcode",
        "firstdrivername",
        "seconddrivername",
        "settlementstatus",
        "rejectednotes",
        "drivingcarriernotes",
    ]
    lines: List[str] = []
    for key in preferred:
        if key in trip and trip[key] not in (None, "", [], {}):
            lines.append(f"{key}: {trip[key]}")

    # Compact JSON for completeness (RAG)
    full_json = json.dumps(trip, ensure_ascii=False, default=str)
    lines.append("full_trip_json: " + full_json)
    return "\n".join(lines)


def _build_document(
    trip: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
) -> Dict[str, Any]:
    normalized = _normalize_order_link_fields(trip)
    doc: Dict[str, Any] = {}
    for key, value in normalized.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = AVAAL_TRIP_NAMESPACE
    doc["page_content"] = _build_page_content(normalized)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": "avaal_trip",
        "source_document": AVAAL_TRIP_SOURCE_DOCUMENT,
        "collection": AVAAL_TRIP_COLLECTION_NAME,
        "database": MONGO_DB_NAME,
        "tripid": normalized.get("tripid"),
        "tripnumber": normalized.get("tripnumber"),
        "orderid": normalized.get("orderid"),
        "ordernumber": normalized.get("ordernumber"),
        "orderids": normalized.get("orderids"),
        "ingested_at": ingested_at,
        "embedding_dimensions": len(embedding),
        "structured": True,
    }
    return doc


def ingest_avaal_trips(
    path: Optional[str] = None,
    replace_namespace: bool = True,
    with_embeddings: bool = True,
) -> Dict[str, Any]:
    path = path or AVAAL_TRIPS_JSON_PATH
    records = _load_trip_records(path)
    if not records:
        return {"ok": False, "error": "No trip records found", "path": path}

    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)

    if replace_namespace:
        deleted = collection.delete_many({"namespace": AVAAL_TRIP_NAMESPACE}).deleted_count
        logger.info(
            "Cleared %s existing docs in namespace=%s collection=%s",
            deleted,
            AVAAL_TRIP_NAMESPACE,
            AVAAL_TRIP_COLLECTION_NAME,
        )

    embeddings = None
    if with_embeddings:
        embeddings, _ = get_models()

    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    inserted = 0
    batch_docs: List[Dict[str, Any]] = []
    embed_dims = 0

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        chunk = records[start : start + EMBED_BATCH_SIZE]
        texts = [_build_page_content(_normalize_order_link_fields(trip)) for trip in chunk]

        if with_embeddings and embeddings is not None:
            vectors = embeddings.embed_documents(texts)
        else:
            vectors = [[] for _ in chunk]

        for trip, vector in zip(chunk, vectors):
            if vector and not embed_dims:
                embed_dims = len(vector)
            batch_docs.append(_build_document(trip, vector, ingested_at))

        if len(batch_docs) >= INSERT_BATCH_SIZE:
            collection.insert_many(batch_docs, ordered=False)
            inserted += len(batch_docs)
            logger.info("Inserted %s / %s", inserted, len(records))
            batch_docs = []

        print(
            f"[Avaal trip ingest] processed {min(start + len(chunk), len(records))} / {len(records)}"
        )

    if batch_docs:
        collection.insert_many(batch_docs, ordered=False)
        inserted += len(batch_docs)

    collection.create_index([("namespace", 1), ("tripid", 1)])
    collection.create_index([("namespace", 1), ("tripnumber", 1)])
    collection.create_index([("namespace", 1), ("orderid", 1)])
    collection.create_index([("namespace", 1), ("ordernumber", 1)])
    collection.create_index([("namespace", 1), ("orderids", 1)])
    collection.create_index([("namespace", 1), ("tripstatus", 1)])
    collection.create_index([("namespace", 1), ("carriername", 1)])

    stored = collection.count_documents(
        {"namespace": AVAAL_TRIP_NAMESPACE, "metadata.type": "avaal_trip"}
    )
    sample = collection.find_one(
        {"namespace": AVAAL_TRIP_NAMESPACE, "metadata.type": "avaal_trip"},
        {"embedding": 0},
    )

    status = {
        "ok": True,
        "database": MONGO_DB_NAME,
        "collection": AVAAL_TRIP_COLLECTION_NAME,
        "namespace": AVAAL_TRIP_NAMESPACE,
        "source_path": path,
        "source_document": AVAAL_TRIP_SOURCE_DOCUMENT,
        "records_loaded": len(records),
        "documents_inserted": inserted,
        "documents_in_namespace": stored,
        "with_embeddings": with_embeddings,
        "embedding_dimensions": embed_dims or (1024 if with_embeddings else 0),
        "sample_tripid": (sample or {}).get("tripid"),
        "sample_orderid": (sample or {}).get("orderid"),
        "sample_top_level_fields": sorted(
            [
                k
                for k in (sample or {}).keys()
                if k not in ("_id", "page_content", "embedding", "metadata", "namespace")
            ]
        )[:25],
    }
    logger.info("Trip ingest complete: %s", status)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ingest_avaal_trips(with_embeddings=True, replace_namespace=True)
    print("\n=== Avaal trip ingest result ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("================================\n")
