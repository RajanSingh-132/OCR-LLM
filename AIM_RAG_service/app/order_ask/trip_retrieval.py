"""
Avaal trip retrieval helpers for /api/v1/orders/ask.

Uses dedicated Mongo collection Avaal_trip_db + namespace avaal_trips.
Joins to Avaal_db orders via orderid / orderids_list / ordernumber.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.embedding_client import get_models
from app.mongo_client import get_mongo_collection
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import (
    AVAAL_RAG_MIN_SCORE,
    AVAAL_TRIP_COLLECTION_NAME,
    AVAAL_TRIP_NAMESPACE,
)
from app.rag_retrieval import MongoVectorStore

_LIST_PROJECTION = {
    "embedding": 0,
    "page_content": 0,
}

_TRIP_PREFERRED_FIELDS = [
    "tripid",
    "tripnumber",
    "tripstatus",
    "triptype",
    "triptypemain",
    "tripvariant",
    "orderids",
    "orderid",
    "ordernumber",
    "itemscount",
    "tripitemscount",
    "customername",
    "customercodes",
    "customerphone",
    "customerorderrefno",
    "companycode",
    "companyname",
    "carriername",
    "carriercode",
    "carrierphone",
    "pickuplocationname",
    "pickupfulladdress",
    "pickupcity",
    "pickupstate",
    "pickupcountry",
    "pickupcountrycode",
    "pickuppostalcode",
    "firstpickupdate",
    "deliverylocationname",
    "deliveryfulladdress",
    "deliverycity",
    "deliverystate",
    "deliverycountry",
    "deliverycountrycode",
    "deliverypostalcode",
    "lastdeliverydate",
    "commodity",
    "equipmenttype",
    "distanceunit",
    "totaldistance",
    "triptotaldistance",
    "totalloaddistance",
    "totalemptydistance",
    "ebdistance",
    "eedistance",
    "trucknumber",
    "truckcode",
    "customtrucknumber",
    "platenumber",
    "firstdrivername",
    "firstdrivercode",
    "firstdriverphone",
    "firstdrivercell1",
    "seconddrivername",
    "seconddrivercode",
    "seconddriverphone",
    "seconddrivercell1",
    "settlementstatus",
    "offeredamount",
    "totalofferedamount",
    "rate",
    "ratetypevalue",
    "totalweight",
    "totalquantity",
]


def _trip_base_match() -> Dict[str, Any]:
    return {"namespace": AVAAL_TRIP_NAMESPACE, "metadata.type": "avaal_trip"}


def trip_order_count(doc: Dict[str, Any]) -> int:
    """Number of orders linked to a trip (best/worst trip metric)."""
    ids = doc.get("orderids_list")
    if isinstance(ids, list) and ids:
        return len([x for x in ids if str(x).strip()])
    raw = doc.get("orderids") or ""
    if isinstance(raw, str) and raw.strip():
        return len([p for p in re.split(r"[,\s]+", raw) if p.strip()])
    if doc.get("orderid") not in (None, ""):
        return 1
    try:
        return int(doc.get("itemscount") or doc.get("tripitemscount") or 0)
    except (TypeError, ValueError):
        return 0


def trip_distance_value(doc: Dict[str, Any]) -> Optional[float]:
    for key in ("totaldistance", "triptotaldistance", "totalloaddistance"):
        raw = doc.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
    return None


def extract_trip_token(question: str) -> Optional[str]:
    """Extract ETP#### / trip id / trip number from question."""
    q = question or ""
    m = re.search(r"\b(ETP\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"\btrip(?:\s*(?:id|number|no\.?|#))?\s*[:#]?\s*([A-Za-z]*\d+)\b",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        tok = m.group(1).strip()
        if not re.fullmatch(r"20\d{2}", tok):
            return tok.upper() if tok.upper().startswith("ETP") else tok

    # Bare ETP-like or numeric only when question clearly about a trip
    if re.search(r"\btrips?\b", q, re.I):
        m = re.search(r"\b(\d{4,})\b", q)
        if m and not re.fullmatch(r"20\d{2}", m.group(1)):
            # Avoid stealing clear MRP/TORD order tokens
            if not re.search(r"\b(MRP\d+|TORD\d+)\b", q, re.I):
                return m.group(1)
    return None


def find_trip_by_id_or_number(token: str) -> Optional[Dict[str, Any]]:
    token = (token or "").strip()
    if not token:
        return None
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    base = _trip_base_match()
    candidates: List[Dict[str, Any]] = [
        {**base, "tripnumber": {"$regex": f"^{re.escape(token)}$", "$options": "i"}},
        {**base, "tripid_str": token},
    ]
    if token.isdigit():
        try:
            candidates.append({**base, "tripid": int(token)})
        except ValueError:
            pass
    # Also match ordernumber fragments on trip docs (MRP on trip)
    if re.match(r"^(MRP|TORD)\d+$", token, re.I):
        candidates.append(
            {
                **base,
                "ordernumber": {"$regex": re.escape(token), "$options": "i"},
            }
        )
        candidates.append(
            {
                **base,
                "orderids": {"$regex": re.escape(token), "$options": "i"},
            }
        )

    for query in candidates:
        doc = collection.find_one(query, _LIST_PROJECTION)
        if doc:
            checkpoint("TRIP", "exact trip found", token=token, trip=doc.get("tripnumber"))
            return doc
    checkpoint("TRIP", "exact trip not found", token=token)
    return None


def find_trips_for_order(order_token: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Find trips linked to an order id / MRP number."""
    token = (order_token or "").strip()
    if not token:
        return []
    limit = max(1, min(int(limit or 10), 50))
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    base = _trip_base_match()
    or_clauses: List[Dict[str, Any]] = [
        {"ordernumber": {"$regex": re.escape(token), "$options": "i"}},
        {"orderids": {"$regex": re.escape(token), "$options": "i"}},
        {"orderid_str": token},
        {"orderids_list": token},
    ]
    if token.isdigit():
        or_clauses.append({"orderid": token})
        or_clauses.append({"orderid": int(token)})
        or_clauses.append({"orderids_list": str(int(token))})

    cursor = collection.find({**base, "$or": or_clauses}, _LIST_PROJECTION).limit(limit)
    docs = list(cursor)
    checkpoint("TRIP", "trips for order", order=token, found=len(docs))
    return docs


def find_related_orders_for_trip(trip_doc: Dict[str, Any], *, limit: int = 20) -> List[Dict[str, Any]]:
    """Load order docs from Avaal_db linked to this trip."""
    from app.order_ask.rag_retrieval import find_order_by_id_or_number

    tokens: List[str] = []
    ids = trip_doc.get("orderids_list")
    if isinstance(ids, list):
        tokens.extend(str(x).strip() for x in ids if str(x).strip())
    raw = trip_doc.get("orderids")
    if isinstance(raw, str) and raw.strip():
        for p in re.split(r"[,\s]+", raw):
            if p.strip() and p.strip() not in tokens:
                tokens.append(p.strip())
    onums = trip_doc.get("ordernumber")
    if isinstance(onums, str) and onums.strip():
        for p in re.split(r"[,\s]+", onums):
            if p.strip() and p.strip() not in tokens:
                tokens.append(p.strip())
    if trip_doc.get("orderid") not in (None, "") and str(trip_doc["orderid"]) not in tokens:
        tokens.append(str(trip_doc["orderid"]))

    orders: List[Dict[str, Any]] = []
    seen = set()
    for tok in tokens[:limit]:
        doc = find_order_by_id_or_number(tok)
        if not doc:
            continue
        key = str(doc.get("orderid") or doc.get("ordernumber") or tok)
        if key in seen:
            continue
        seen.add(key)
        orders.append(doc)
    checkpoint("TRIP", "related orders", trip=trip_doc.get("tripnumber"), orders=len(orders))
    return orders


def format_trip_doc_for_context(doc: Dict[str, Any], max_fields: int = 80) -> str:
    lines: List[str] = []
    used = set()
    clip_keys = {
        "customername",
        "customercodes",
        "customerphone",
        "customerorderrefno",
        "ordernumber",
        "orderids",
        "salesmannames",
        "salesmancodes",
    }
    for key in _TRIP_PREFERRED_FIELDS:
        if key not in doc or doc[key] in (None, "", [], {}):
            continue
        val = doc[key]
        if key in clip_keys:
            val = _clip_list_field(val, 12 if key in ("ordernumber", "orderids") else 6)
        lines.append(f"{key}: {val}")
        used.add(key)
        if len(lines) >= max_fields:
            break
    # Extra useful scalars not already listed
    skip = used | {
        "_id",
        "embedding",
        "page_content",
        "metadata",
        "namespace",
        "full_trip_json",
        "orderids_list",
    }
    for key, value in doc.items():
        if key in skip or value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)) and key not in ("taxes",):
            continue
        lines.append(f"{key}: {value}")
        if len(lines) >= max_fields:
            break
    lines.append(f"linked_order_count: {trip_order_count(doc)}")
    dist = trip_distance_value(doc)
    if dist is not None:
        lines.append(f"parsed_total_distance: {dist}")
    return "\n".join(lines)


def _clip_list_field(value: Any, max_items: int = 8) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
    else:
        parts = [p.strip() for p in re.split(r"\s*,\s*", str(value)) if p.strip()]
    if len(parts) <= max_items:
        return ", ".join(parts) if parts else str(value)
    return ", ".join(parts[:max_items]) + f" (+{len(parts) - max_items} more)"


def format_trip_list_for_context(payload: Dict[str, Any]) -> str:
    lines = [
        "TRIP LIST RESULT:",
        f"total_matching: {payload.get('total_matching')}",
        f"returned: {payload.get('returned')}",
    ]
    for i, row in enumerate(payload.get("trips") or [], 1):
        lines.append(
            f"{i}. tripnumber={row.get('tripnumber')} tripid={row.get('tripid')} "
            f"status={row.get('tripstatus')} customer={row.get('customername')} "
            f"driver={row.get('firstdrivername')} truck={row.get('trucknumber')} "
            f"distance={row.get('totaldistance') or row.get('triptotaldistance')} "
            f"orders={row.get('ordernumber') or row.get('orderids')} "
            f"order_count={row.get('linked_order_count')} "
            f"pickup_country={row.get('pickupcountry')} delivery_country={row.get('deliverycountry')}"
        )
    return "\n".join(lines)


def search_trips(
    *,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 15,
    sort_by: str = "tripid",
    ascending: bool = False,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 15), 50))
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    query: Dict[str, Any] = {**_trip_base_match()}
    if filters:
        query.update(filters)
    sort_dir = 1 if ascending else -1
    sort_field = sort_by if sort_by else "tripid"
    total = collection.count_documents(query)
    cursor = (
        collection.find(query, _LIST_PROJECTION)
        .sort(sort_field, sort_dir)
        .limit(limit)
    )
    trips = []
    for doc in cursor:
        row = {
            "tripid": doc.get("tripid"),
            "tripnumber": doc.get("tripnumber"),
            "tripstatus": doc.get("tripstatus"),
            "customername": doc.get("customername"),
            "firstdrivername": doc.get("firstdrivername"),
            "firstdriverphone": doc.get("firstdriverphone") or doc.get("firstdrivercell1"),
            "trucknumber": doc.get("trucknumber") or doc.get("customtrucknumber"),
            "totaldistance": doc.get("totaldistance"),
            "triptotaldistance": doc.get("triptotaldistance"),
            "totalloaddistance": doc.get("totalloaddistance"),
            "totalemptydistance": doc.get("totalemptydistance"),
            "ordernumber": doc.get("ordernumber"),
            "orderids": doc.get("orderids"),
            "pickupcountry": doc.get("pickupcountry"),
            "deliverycountry": doc.get("deliverycountry"),
            "linked_order_count": trip_order_count(doc),
        }
        trips.append(row)
    checkpoint("TRIP", "search_trips", total=total, returned=len(trips))
    return {
        "total_matching": total,
        "returned": len(trips),
        "trips": trips,
        "filters": filters or {},
    }


def get_avaal_trip_vectorstore(embeddings=None) -> Optional[MongoVectorStore]:
    if embeddings is None:
        embeddings, _ = get_models()
    collection = get_mongo_collection(AVAAL_TRIP_COLLECTION_NAME)
    exists = collection.count_documents(_trip_base_match(), limit=1) > 0
    if not exists:
        return None
    return MongoVectorStore(
        collection=collection,
        embeddings=embeddings,
        namespace=AVAAL_TRIP_NAMESPACE,
    )


def retrieve_avaal_trips(
    question: str,
    k: int = 8,
    embeddings=None,
    min_score: Optional[float] = None,
) -> List[Document]:
    vectorstore = get_avaal_trip_vectorstore(embeddings=embeddings)
    if vectorstore is None:
        checkpoint("TRIP_RAG", "no trip vectorstore")
        return []
    threshold = AVAAL_RAG_MIN_SCORE if min_score is None else min_score
    docs = vectorstore.similarity_search(
        query=question, k=k, fetch_k=max(k * 4, 40)
    )
    kept: List[Document] = []
    for doc in docs:
        score = (doc.metadata or {}).get("similarity_score")
        if score is None or float(score) >= threshold:
            kept.append(doc)
    checkpoint("TRIP_RAG", "semantic", requested=k, raw=len(docs), kept=len(kept))
    return kept


def build_trip_rag_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        parts.append(
            f"[Trip match {i} score={meta.get('similarity_score')} "
            f"tripnumber={meta.get('tripnumber')} tripid={meta.get('tripid')}]\n"
            f"{(doc.page_content or '')[:2500]}"
        )
    return "\n\n".join(parts)
