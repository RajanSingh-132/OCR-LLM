"""
Avaal orders retrieval helpers for /api/v1/orders/ask.

Uses dedicated Mongo collection Avaal_db + namespace avaal_orders.
Builds on MongoVectorStore from app.rag_retrieval (PDF store) without changing PDF flow.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.embedding_client import get_models
from app.mongo_client import get_mongo_collection
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import (
    AVAAL_COLLECTION_NAME,
    AVAAL_NAMESPACE,
    AVAAL_RAG_MIN_SCORE,
)
from app.rag_retrieval import MongoVectorStore

# Projection used for list/search responses (no embeddings)
_LIST_PROJECTION = {
    "embedding": 0,
    "page_content": 0,
}


def get_avaal_vectorstore(embeddings=None) -> Optional[MongoVectorStore]:
    """Return vectorstore bound to Avaal_db / avaal_orders if data exists."""
    if embeddings is None:
        embeddings, _ = get_models()

    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    exists = collection.count_documents(
        {"namespace": AVAAL_NAMESPACE, "metadata.type": "avaal_order"},
        limit=1,
    ) > 0
    if not exists:
        return None

    return MongoVectorStore(
        collection=collection,
        embeddings=embeddings,
        namespace=AVAAL_NAMESPACE,
    )


def retrieve_avaal_orders(
    question: str,
    k: int = 10,
    embeddings=None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """Semantic retrieve top-k Avaal order chunks (weak matches dropped)."""
    vectorstore = get_avaal_vectorstore(embeddings=embeddings)
    if vectorstore is None:
        checkpoint("RAG", "no vectorstore / empty Avaal_db")
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
    checkpoint(
        "RAG",
        "semantic retrieve",
        requested=k,
        raw=len(docs),
        kept=len(kept),
        min_score=threshold,
    )
    return kept


def _base_order_match(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        "namespace": AVAAL_NAMESPACE,
        "metadata.type": "avaal_order",
    }
    filters = filters or {}
    if filters.get("orderstatus"):
        match["orderstatus"] = filters["orderstatus"]
    if filters.get("currencycode"):
        match["currencycode"] = filters["currencycode"]
    if filters.get("customercode"):
        match["customercode"] = {
            "$regex": f"^{re.escape(str(filters['customercode']))}$",
            "$options": "i",
        }
    if filters.get("companycode"):
        match["companycode"] = {
            "$regex": f"^{re.escape(str(filters['companycode']))}$",
            "$options": "i",
        }
    if filters.get("customername"):
        match["customername"] = {
            "$regex": re.escape(str(filters["customername"])),
            "$options": "i",
        }
    if filters.get("pickup_location"):
        match["pickuplocationname"] = {
            "$regex": re.escape(str(filters["pickup_location"])),
            "$options": "i",
        }
    if filters.get("delivery_location"):
        match["deliverylocationname"] = {
            "$regex": re.escape(str(filters["delivery_location"])),
            "$options": "i",
        }
    # Date prefix match on ISO-like strings (e.g. 2026-08-06)
    if filters.get("orderdate"):
        match["orderdate"] = {
            "$regex": f"^{re.escape(str(filters['orderdate']))}",
            "$options": "i",
        }
    if filters.get("pickupdate"):
        match["pickupdate"] = {
            "$regex": f"^{re.escape(str(filters['pickupdate']))}",
            "$options": "i",
        }
    if filters.get("deliverydate"):
        match["deliverydate"] = {
            "$regex": f"^{re.escape(str(filters['deliverydate']))}",
            "$options": "i",
        }
    return match


def search_orders(
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 15,
    sort_by: str = "orderid",
    ascending: bool = False,
) -> Dict[str, Any]:
    """
    Structured Mongo list/search — accurate filtered order lists.
    Returns count + compact order rows (not embeddings).
    """
    limit = max(1, min(int(limit or 15), 50))
    allowed_sort = {
        "orderid",
        "ordernumber",
        "totalfreight",
        "grosstotalfreight",
        "taxes",
        "totaltaxamount",
        "offeredamount",
        "distance",
        "orderdate",
        "pickupdate",
        "deliverydate",
        "customername",
        "orderstatus",
    }
    if sort_by not in allowed_sort:
        sort_by = "orderid"

    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    match = _base_order_match(filters)
    total = collection.count_documents(match)
    cursor = (
        collection.find(match, _LIST_PROJECTION)
        .sort(sort_by, 1 if ascending else -1)
        .limit(limit)
    )
    rows = []
    for doc in cursor:
        rows.append(
            {
                "orderid": doc.get("orderid"),
                "ordernumber": doc.get("ordernumber"),
                "customername": doc.get("customername"),
                "orderstatus": doc.get("orderstatus"),
                "currencycode": doc.get("currencycode"),
                "totalfreight": doc.get("totalfreight"),
                "grosstotalfreight": doc.get("grosstotalfreight"),
                "taxes": doc.get("taxes"),
                "distance": doc.get("distance"),
                "distanceunit": doc.get("distanceunit"),
                "pickuplocationname": doc.get("pickuplocationname"),
                "deliverylocationname": doc.get("deliverylocationname"),
                "orderdate": doc.get("orderdate"),
                "pickupdate": doc.get("pickupdate"),
                "deliverydate": doc.get("deliverydate"),
                "companycode": doc.get("companycode"),
            }
        )
    checkpoint(
        "LIST",
        "mongo filtered search",
        filters=filters or {},
        total=total,
        returned=len(rows),
        limit=limit,
        sort_by=sort_by,
        ascending=ascending,
    )
    return {
        "filters": filters or {},
        "sort_by": sort_by,
        "ascending": ascending,
        "total_matching": total,
        "returned": len(rows),
        "orders": rows,
    }


def list_recent_orders(limit: int = 10) -> Dict[str, Any]:
    """Most recent orders by orderid desc."""
    return search_orders(filters=None, limit=limit, sort_by="orderid", ascending=False)


def format_order_list_for_context(payload: Dict[str, Any]) -> str:
    lines = [
        "ORDER LIST RESULT (exact Mongo filters — do not invent rows):",
        f"filters={payload.get('filters')}",
        f"total_matching={payload.get('total_matching')}",
        f"returned={payload.get('returned')}",
    ]
    for i, row in enumerate(payload.get("orders") or [], start=1):
        lines.append(
            f"[{i}] orderid={row.get('orderid')} ordernumber={row.get('ordernumber')} "
            f"customer={row.get('customername')} status={row.get('orderstatus')} "
            f"currency={row.get('currencycode')} freight={row.get('totalfreight')} "
            f"taxes={row.get('taxes')}"
        )
    if not payload.get("orders"):
        lines.append("(no orders matched these filters)")
    return "\n".join(lines)


def find_order_by_id_or_number(token: str) -> Optional[Dict[str, Any]]:
    """Exact lookup by orderid or ordernumber when user asks for one order."""
    if not token:
        return None
    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    token = token.strip()
    query_filter: Dict[str, Any] = {
        "namespace": AVAAL_NAMESPACE,
        "metadata.type": "avaal_order",
    }
    if token.isdigit():
        query_filter["$or"] = [
            {"orderid": int(token)},
            {"ordernumber": token},
            {"tempordernumber": token},
        ]
    else:
        query_filter["$or"] = [
            {"ordernumber": {"$regex": f"^{re.escape(token)}$", "$options": "i"}},
            {"tempordernumber": {"$regex": f"^{re.escape(token)}$", "$options": "i"}},
        ]

    doc = collection.find_one(query_filter, {"embedding": 0})
    checkpoint(
        "EXACT",
        "order lookup",
        token=token,
        found=bool(doc),
        orderid=(doc or {}).get("orderid"),
        ordernumber=(doc or {}).get("ordernumber"),
    )
    return doc


def extract_order_token(question: str) -> Optional[str]:
    """Extract order number / id tokens including MRP#### and TORD####."""
    q = question or ""
    # Avaal order numbers like MRP3301 / MRP3298
    m = re.search(r"\b(MRP\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(TORD\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(
        r"\border(?:\s*(?:id|number|no\.?|#))?\s*[:#]?\s*([A-Za-z]*\d+)\b",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        tok = m.group(1)
        # Ignore bare years mistaken from dates
        if re.fullmatch(r"20\d{2}", tok):
            pass
        else:
            return tok.upper() if tok.upper().startswith("MRP") else tok
    # Digits: skip years and date fragments (2026-08-06 / 07/13/2026)
    for m in re.finditer(r"\b(\d{4,})\b", q):
        tok = m.group(1)
        if re.fullmatch(r"20\d{2}", tok):
            continue
        # skip if this number sits inside a date pattern
        start, end = m.span()
        window = q[max(0, start - 3) : min(len(q), end + 3)]
        if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", window) or re.search(
            rf"{re.escape(tok)}[-/]\d", q[start : end + 6]
        ):
            continue
        return tok
    return None


# Preferred field order so full order details are useful in LLM context
_ORDER_DETAIL_PRIORITY = [
    "orderid",
    "ordernumber",
    "tempordernumber",
    "orderstatus",
    "orderdate",
    "customername",
    "customercode",
    "companycode",
    "salesmanname",
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
    "distance",
    "distanceunit",
    "weight",
    "weightunit",
    "pickuplocationname",
    "pickupfulladdress",
    "pickupdate",
    "deliverylocationname",
    "deliveryfulladdress",
    "deliverydate",
    "commodityname",
    "ordernotes",
    "allcarriersname",
]


def format_order_doc_for_context(doc: Dict[str, Any], max_fields: int = 120) -> str:
    skip = {"_id", "embedding", "page_content", "metadata", "namespace"}
    lines = []
    seen = set()

    def _add(key: str, value: Any) -> None:
        nonlocal lines
        if key in skip or key in seen:
            return
        if value in (None, "", [], {}):
            return
        seen.add(key)
        lines.append(f"{key}: {value}")

    for key in _ORDER_DETAIL_PRIORITY:
        if key in doc:
            _add(key, doc.get(key))
            if len(lines) >= max_fields:
                return "\n".join(lines)

    for key, value in doc.items():
        _add(key, value)
        if len(lines) >= max_fields:
            break
    return "\n".join(lines)


def build_rag_context(docs: List[Document], max_chars_per_doc: int = 2500) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        header = (
            f"[Match {i}] orderid={meta.get('orderid')} "
            f"ordernumber={meta.get('ordernumber')} "
            f"score={meta.get('similarity_score')}"
        )
        parts.append(header + "\n" + (doc.page_content or "")[:max_chars_per_doc])
    return "\n\n".join(parts)
