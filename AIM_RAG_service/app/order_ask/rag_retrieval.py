"""
Avaal orders retrieval helpers for /api/v1/orders/ask.

Uses dedicated Mongo collection Avaal_order + namespace avaal_orders.
Builds on MongoVectorStore from app.rag_retrieval (PDF store) without changing PDF flow.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.embedding_client import get_models
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import AVAAL_RAG_MIN_SCORE
from app.rag_retrieval import MongoVectorStore
from app.tenants.router import (
    get_orders_collection,
    get_orders_metadata_type,
    get_orders_namespace,
)

# Projection used for list/search responses (no embeddings)
_LIST_PROJECTION = {
    "embedding": 0,
    "page_content": 0,
}


def get_avaal_vectorstore(embeddings=None) -> Optional[MongoVectorStore]:
    """Return vectorstore bound to Avaal_order / avaal_orders if data exists."""
    if embeddings is None:
        embeddings, _ = get_models()

    collection = get_orders_collection()
    namespace = get_orders_namespace()
    metadata_type = get_orders_metadata_type()
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


def retrieve_avaal_orders(
    question: str,
    k: int = 10,
    embeddings=None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """Semantic retrieve top-k Avaal order chunks (weak matches dropped)."""
    vectorstore = get_avaal_vectorstore(embeddings=embeddings)
    if vectorstore is None:
        checkpoint("RAG", "no vectorstore / empty Avaal_order")
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


def _side_address_ors(
    *,
    side: str,
    pattern: str,
    include_location_names: bool = False,
) -> List[Dict[str, Any]]:
    """Build $or clauses against pickup/delivery address (+ optional location names)."""
    ors: List[Dict[str, Any]] = []
    rx = {"$regex": pattern, "$options": "i"}
    if side in ("pickup", "both"):
        ors.append({"pickupfulladdress": rx})
        if include_location_names:
            ors.append({"pickuplocationname": rx})
    if side in ("delivery", "both"):
        ors.append({"deliveryfulladdress": rx})
        if include_location_names:
            ors.append({"deliverylocationname": rx})
    return ors


def _state_address_pattern(state: str) -> str:
    """Match state/province inside address: ', CA,' or ', California,'."""
    from app.order_ask.field_catalog import STATE_ALIASES, resolve_state_token

    code = resolve_state_token(state) or (state.strip().upper() if len(state.strip()) <= 3 else None)
    tokens: List[str] = []
    if code:
        tokens.append(code)
        tokens.extend(STATE_ALIASES.get(code, []))
    else:
        tokens.append(state.strip())
    # unique, longest-first so "new york" beats "york" noise
    seen = set()
    ordered = []
    for t in sorted(tokens, key=lambda x: -len(x)):
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            ordered.append(re.escape(t))
    alt = "|".join(ordered)
    return rf"(?:^|,\s*)(?:{alt})(?:\s*,|$)"


def _pin_pattern(pin: str) -> str:
    """US zip or Canadian postal with flexible spacing."""
    pin = (pin or "").strip()
    # Canadian: A1A 1A1 or A1A1A1
    m = re.fullmatch(r"([A-Za-z]\d[A-Za-z])\s?(\d[A-Za-z]\d)", pin, re.I)
    if m:
        return rf"\b{m.group(1)}\s?{m.group(2)}\b"
    return rf"\b{re.escape(pin)}\b"


def _base_order_match(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        "namespace": get_orders_namespace(),
        "metadata.type": get_orders_metadata_type(),
    }
    filters = filters or {}
    if filters.get("orderstatus"):
        match["orderstatus"] = filters["orderstatus"]
    if filters.get("accountingstatus"):
        # Exact case-insensitive match (Atlas-safe)
        st = str(filters["accountingstatus"])
        match["accountingstatus"] = st
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
    if filters.get("salesmanname"):
        match["salesmanname"] = {
            "$regex": re.escape(str(filters["salesmanname"])),
            "$options": "i",
        }
    if filters.get("commodityname"):
        match["commodityname"] = {
            "$regex": re.escape(str(filters["commodityname"])),
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

    # Geo filters live inside address strings — AND each condition via $or sides
    side = str(filters.get("location_side") or "both").lower()
    if side not in ("pickup", "delivery", "both"):
        side = "both"
    and_parts: List[Dict[str, Any]] = []

    if filters.get("outstatus"):
        st = str(filters["outstatus"])
        and_parts.append(
            {
                "$or": [
                    {"outstatus": st},
                    {"outsourcedetails.outStatus": st},
                ]
            }
        )

    if filters.get("pin"):
        ors = _side_address_ors(side=side, pattern=_pin_pattern(str(filters["pin"])))
        if ors:
            and_parts.append({"$or": ors})

    if filters.get("state"):
        ors = _side_address_ors(
            side=side, pattern=_state_address_pattern(str(filters["state"]))
        )
        if ors:
            and_parts.append({"$or": ors})

    if filters.get("country"):
        country = str(filters["country"]).strip()
        country_rx = rf"\b{re.escape(country)}\b"
        if country.lower() in {"us", "usa", "u.s.", "u.s.a."}:
            country_rx = r"\b(?:United\s+States|USA|U\.S\.A\.?|U\.S\.?)\b"
        elif country.lower() == "canada":
            country_rx = r"\bCanada\b"
        ors = _side_address_ors(side=side, pattern=country_rx)
        if ors:
            and_parts.append({"$or": ors})

    if filters.get("city"):
        city_rx = rf"\b{re.escape(str(filters['city']).strip())}\b"
        ors = _side_address_ors(
            side=side, pattern=city_rx, include_location_names=True
        )
        if ors:
            and_parts.append({"$or": ors})

    if filters.get("address"):
        addr_rx = re.escape(str(filters["address"]).strip())
        ors = _side_address_ors(side=side, pattern=addr_rx)
        if ors:
            and_parts.append({"$or": ors})

    if filters.get("location"):
        loc_rx = re.escape(str(filters["location"]).strip())
        ors = _side_address_ors(
            side=side, pattern=loc_rx, include_location_names=True
        )
        if ors:
            and_parts.append({"$or": ors})

    if and_parts:
        match["$and"] = and_parts

    return match


def search_orders(
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 15,
    sort_by: str = "orderid",
    ascending: bool = False,
    *,
    match: Optional[Dict[str, Any]] = None,
    sort_expr: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Structured Mongo list/search — accurate filtered order lists.
    Returns count + compact order rows (not embeddings).

    Pass a prebuilt ``match`` to bypass ``_base_order_match(filters)`` (used by
    the LLM query planner, which builds its own operator-DSL match).

    Pass ``sort_expr`` (a Mongo aggregation expression) to sort on a computed
    value instead of a raw field — e.g. a parsed date for US-format date columns
    whose lexical order is meaningless.
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
    if sort_expr is None and sort_by not in allowed_sort:
        sort_by = "orderid"

    collection = get_orders_collection()
    if match is None:
        match = _base_order_match(filters)
    total = collection.count_documents(match)
    if sort_expr is not None:
        cursor = collection.aggregate(
            [
                {"$match": match},
                {"$addFields": {"__sort": sort_expr}},
                {"$sort": {"__sort": 1 if ascending else -1, "orderid": -1}},
                {"$limit": limit},
                {"$project": _LIST_PROJECTION},
            ],
            maxTimeMS=8000,
        )
    else:
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
                "accountingstatus": doc.get("accountingstatus"),
                "outstatus": doc.get("outstatus")
                or ((doc.get("outsourcedetails") or {}).get("outStatus")),
                "currencycode": doc.get("currencycode"),
                "totalfreight": doc.get("totalfreight"),
                "grosstotalfreight": doc.get("grosstotalfreight"),
                "taxes": doc.get("taxes"),
                "distance": doc.get("distance"),
                "distanceunit": doc.get("distanceunit"),
                "pickuplocationname": doc.get("pickuplocationname"),
                "pickupfulladdress": doc.get("pickupfulladdress"),
                "deliverylocationname": doc.get("deliverylocationname"),
                "deliveryfulladdress": doc.get("deliveryfulladdress"),
                "orderdate": doc.get("orderdate"),
                "pickupdate": doc.get("pickupdate"),
                "deliverydate": doc.get("deliverydate"),
                "companycode": doc.get("companycode"),
                "salesmanname": doc.get("salesmanname"),
                "commodityname": doc.get("commodityname"),
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
    filters = payload.get("filters") or {}
    show_geo = any(
        filters.get(k)
        for k in (
            "pin",
            "state",
            "city",
            "address",
            "location",
            "pickup_location",
            "delivery_location",
        )
    )
    for i, row in enumerate(payload.get("orders") or [], start=1):
        line = (
            f"[{i}] orderid={row.get('orderid')} ordernumber={row.get('ordernumber')} "
            f"customer={row.get('customername')} status={row.get('orderstatus')} "
            f"accounting={row.get('accountingstatus')} outstatus={row.get('outstatus')} "
            f"currency={row.get('currencycode')} freight={row.get('totalfreight')} "
            f"taxes={row.get('taxes')} "
            f"pickup_loc={row.get('pickuplocationname')} "
            f"delivery_loc={row.get('deliverylocationname')}"
        )
        if show_geo:
            line += (
                f" | pickup_address={row.get('pickupfulladdress')} "
                f"| delivery_address={row.get('deliveryfulladdress')}"
            )
        lines.append(line)
    if not payload.get("orders"):
        lines.append("(no orders matched these filters)")
    return "\n".join(lines)


def find_order_by_id_or_number(token: str) -> Optional[Dict[str, Any]]:
    """Exact lookup by orderid or ordernumber when user asks for one order."""
    if not token:
        return None
    collection = get_orders_collection()
    token = token.strip()
    query_filter: Dict[str, Any] = {
        "namespace": get_orders_namespace(),
        "metadata.type": get_orders_metadata_type(),
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
    """Extract order number / id tokens — formats may vary (MRP/TORD/alphanumeric)."""
    q = question or ""
    ql = q.lower()
    # Common Avaal prefixes — require at least one digit so "order" is not a token
    for pat in (
        r"\b(MRP[A-Za-z0-9-]*\d[A-Za-z0-9-]*)\b",
        r"\b(TORD[A-Za-z0-9-]*\d[A-Za-z0-9-]*)\b",
        r"\b(TMP[A-Za-z0-9-]*\d[A-Za-z0-9-]*)\b",
    ):
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    pin_context = bool(
        re.search(r"\b(pin\s*code|pincode|pin|zip\s*code|zip|postal\s*code|postal)\b", ql)
    )
    m = re.search(
        r"\border(?:\s*(?:id|number|no\.?|#))\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,30})\b",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        tok = m.group(1)
        if re.fullmatch(r"20\d{2}", tok):
            pass
        elif pin_context and re.fullmatch(r"\d{5}(?:-\d{4})?", tok):
            pass
        elif tok.lower() in {
            "status", "details", "detail", "list", "recent", "confirmed",
            "quoted", "dispatched", "delivered", "cancelled", "invoiced",
            "confirm", "order", "orders", "some", "any", "wise",
        }:
            pass
        else:
            return tok.upper()

    # Generic alphanumeric order-like token (letters + digits), skip trip prefixes
    m = re.search(r"\b([A-Za-z]{2,6}\d{2,})\b", q)
    if m:
        tok = m.group(1).upper()
        if not re.match(r"^(ETP|TRO|TRIP)", tok) and tok.lower() not in {
            "order", "orders",
        }:
            return tok

    for m in re.finditer(r"\b(\d{4,})\b", q):
        tok = m.group(1)
        if re.fullmatch(r"20\d{2}", tok):
            continue
        start, end = m.span()
        window = q[max(0, start - 3) : min(len(q), end + 3)]
        if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", window) or re.search(
            rf"{re.escape(tok)}[-/]\d", q[start : end + 6]
        ):
            continue
        pre = q[max(0, start - 24) : start].lower()
        if re.fullmatch(r"\d{5}(?:-\d{4})?", tok) and (
            pin_context or re.search(r"\b(pin|zip|postal|pincode|code)\b", pre)
        ):
            continue
        # "give me 20 order" — digit is a limit, not an order id
        if re.search(rf"\b(give|get|show|list|top|last|only|just)\s+{re.escape(tok)}\b", ql):
            continue
        if re.search(rf"\b{re.escape(tok)}\s+(orders?|invoices?|trips?)\b", ql):
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
