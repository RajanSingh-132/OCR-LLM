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
        # Accurate list: case-insensitive partial match on customer name
        match["customername"] = {
            "$regex": re.escape(str(filters["customername"])),
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
                "taxes": doc.get("taxes"),
                "grosstotalfreight": doc.get("grosstotalfreight"),
                "pickuplocationname": doc.get("pickuplocationname"),
                "deliverylocationname": doc.get("deliverylocationname"),
                "orderdate": doc.get("orderdate"),
            }
        )
    checkpoint(
        "LIST",
        "mongo filtered search",
        filters=filters or {},
        total=total,
        returned=len(rows),
        limit=limit,
    )
    return {
        "filters": filters or {},
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
    q = question or ""
    m = re.search(r"\b(TORD\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(
        r"\border(?:\s*(?:id|number|no\.?|#))?\s*[:#]?\s*(\d+)\b",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4,})\b", q)
    if m:
        return m.group(1)
    return None


def format_order_doc_for_context(doc: Dict[str, Any], max_fields: int = 80) -> str:
    skip = {"_id", "embedding", "page_content", "metadata", "namespace"}
    lines = []
    for key, value in doc.items():
        if key in skip:
            continue
        if value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {value}")
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
