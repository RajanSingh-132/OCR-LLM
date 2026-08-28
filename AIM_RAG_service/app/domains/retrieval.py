"""Domain-aware Mongo retrieval (orders / invoices / trips / future domains)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.domains.registry import get_domain_profile
from app.embedding_client import get_models
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import AVAAL_RAG_MIN_SCORE
from app.rag_retrieval import MongoVectorStore
from app.tenants.context import get_active_domain
from app.tenants.router import (
    get_domain_collection,
    get_domain_metadata_type,
    get_domain_namespace,
)

_LIST_PROJECTION = {"embedding": 0, "page_content": 0}


def _profile():
    return get_domain_profile(get_active_domain())


def _base_match(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    domain = get_active_domain()
    match: Dict[str, Any] = {
        "namespace": get_domain_namespace(domain),
        "metadata.type": get_domain_metadata_type(domain),
    }
    if filters:
        match.update(filters)
    return match


def _pick_sort_field(requested: str) -> str:
    profile = _profile()
    allowed = set(profile.sort_fields)
    if requested in allowed:
        return requested
    if profile.default_sort in allowed:
        return profile.default_sort
    return profile.sort_fields[0] if profile.sort_fields else "_id"


def get_vectorstore(embeddings=None) -> Optional[MongoVectorStore]:
    if embeddings is None:
        embeddings, _ = get_models()

    domain = get_active_domain()
    collection = get_domain_collection(domain)
    namespace = get_domain_namespace(domain)
    metadata_type = get_domain_metadata_type(domain)
    exists = collection.count_documents(
        {"namespace": namespace, "metadata.type": metadata_type},
        limit=1,
    ) > 0
    if not exists:
        return None

    return MongoVectorStore(
        collection=collection,
        embeddings=embeddings,
        namespace=namespace,
    )


def semantic_retrieve(
    question: str,
    k: int = 10,
    embeddings=None,
    min_score: Optional[float] = None,
) -> List[Document]:
    domain = get_active_domain()
    vectorstore = get_vectorstore(embeddings=embeddings)
    if vectorstore is None:
        checkpoint("RAG", f"no vectorstore / empty {domain}")
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
        domain=domain,
        requested=k,
        raw=len(docs),
        kept=len(kept),
    )
    return kept


def search_records(
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 15,
    sort_by: str = "",
    ascending: bool = False,
) -> Dict[str, Any]:
    domain = get_active_domain()
    profile = _profile()
    sort_by = _pick_sort_field(sort_by or profile.default_sort)

    collection = get_domain_collection(domain)
    match = _base_match(filters)
    total = collection.count_documents(match)
    cursor = (
        collection.find(match, _LIST_PROJECTION)
        .sort(sort_by, 1 if ascending else -1)
        .limit(limit)
    )

    rows = []
    for doc in cursor:
        row = {field: doc.get(field) for field in profile.list_fields if field in doc}
        if not row:
            row = {
                k: v
                for k, v in doc.items()
                if k not in ("_id", "embedding", "page_content", "metadata")
            }
        rows.append(row)

    checkpoint(
        "LIST",
        "mongo filtered search",
        domain=domain,
        total=total,
        returned=len(rows),
    )
    return {
        "domain": domain,
        "filters": filters or {},
        "sort_by": sort_by,
        "ascending": ascending,
        "total_matching": total,
        "returned": len(rows),
        "records": rows,
    }


def list_recent_records(limit: int = 10) -> Dict[str, Any]:
    profile = _profile()
    return search_records(
        filters=None,
        limit=limit,
        sort_by=profile.default_sort,
        ascending=False,
    )


def count_records(filters: Optional[Dict[str, Any]] = None) -> int:
    domain = get_active_domain()
    collection = get_domain_collection(domain)
    return collection.count_documents(_base_match(filters))


def sum_numeric_field(
    field: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sum a numeric Mongo field for the active domain (exact field name from registry)."""
    domain = get_active_domain()
    collection = get_domain_collection(domain)
    match = _base_match(filters)
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "total": {
                    "$sum": {
                        "$convert": {
                            "input": f"${field}",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    }
                },
            }
        },
    ]
    rows = list(collection.aggregate(pipeline))
    total = float(rows[0]["total"]) if rows else 0.0
    checkpoint("SUM", "field aggregate", domain=domain, field=field, total=total)
    return {
        "domain": domain,
        "field": field,
        "filters": filters or {},
        "sum_total": total,
    }


def find_record_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    domain = get_active_domain()
    profile = _profile()
    collection = get_domain_collection(domain)
    token = token.strip()

    query_filter: Dict[str, Any] = {
        "namespace": get_domain_namespace(domain),
        "metadata.type": get_domain_metadata_type(domain),
    }

    ors: List[Dict[str, Any]] = []
    if token.isdigit():
        num = int(token)
        for field in profile.id_fields:
            ors.append({field: num})
        for field in profile.number_fields:
            ors.append({field: token})
            ors.append({field: num})
    else:
        for field in profile.number_fields:
            ors.append({field: {"$regex": f"^{re.escape(token)}$", "$options": "i"}})

    if not ors:
        return None

    query_filter["$or"] = ors
    doc = collection.find_one(query_filter, {"embedding": 0})
    checkpoint("LOOKUP", "exact record", domain=domain, token=token, found=bool(doc))
    return doc


def format_record_list_for_context(payload: Dict[str, Any]) -> str:
    profile = _profile()
    label = profile.label.upper()
    lines = [
        f"{label} LIST RESULT (exact Mongo filters — do not invent rows):",
        f"domain={payload.get('domain')}",
        f"filters={payload.get('filters')}",
        f"total_matching={payload.get('total_matching')}",
        f"returned={payload.get('returned')}",
    ]
    for i, row in enumerate(payload.get("records") or [], start=1):
        parts = [f"{k}={row.get(k)}" for k in profile.list_fields if row.get(k) not in (None, "")]
        if not parts:
            parts = [f"{k}={v}" for k, v in list(row.items())[:12]]
        lines.append(f"[{i}] " + " ".join(parts))
    if not payload.get("records"):
        lines.append(f"(no {profile.label.lower()} records matched these filters)")
    return "\n".join(lines)


def format_record_doc_for_context(doc: Dict[str, Any], max_fields: int = 120) -> str:
    profile = _profile()
    lines = [f"{profile.label.upper()} RECORD:"]
    count = 0
    for key, value in doc.items():
        if key in ("_id", "embedding", "page_content"):
            continue
        if value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {value}")
        count += 1
        if count >= max_fields:
            lines.append("...(truncated)")
            break
    return "\n".join(lines)


def build_rag_context(docs: List[Document], max_chars_per_doc: int = 2500) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        text = (doc.page_content or "")[:max_chars_per_doc]
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)
