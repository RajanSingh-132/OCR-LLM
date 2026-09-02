"""Turn a raw Avaal order record into the Mongo document shape the read path
expects (`namespace`, `page_content`, `embedding`, `metadata.type`, plus the
sync bookkeeping fields).

Shared by the live sync and the manual `scripts/ingest/order_ingest.py`.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from app.mongo_client import _to_python_types

logger = logging.getLogger("sync.documents")

# Titan Text Embeddings V2 caps input at 8192 tokens; order text is ID/number
# heavy so keep the embedded slice well under that. Full record is still stored.
MAX_EMBED_CHARS = 18000


def build_page_content(order: Dict[str, Any]) -> str:
    """Deterministic text blob of every populated top-level key (sorted)."""
    lines = []
    for key in sorted(order.keys()):
        value = order[key]
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def truncate_for_embedding(text: str) -> str:
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding text %s chars > %s — truncating", len(text), MAX_EMBED_CHARS
    )
    return text[:MAX_EMBED_CHARS]


def build_order_document(
    order: Dict[str, Any],
    *,
    embedding: List[float],
    namespace: str,
    metadata_type: str,
    source_document: str,
    synced_at: Any,
    page_content: Optional[str] = None,
    hash_value: Optional[str] = None,
    is_stale: bool = False,
) -> Dict[str, Any]:
    text = page_content if page_content is not None else build_page_content(order)
    doc: Dict[str, Any] = {k: _to_python_types(v) for k, v in order.items()}
    doc["namespace"] = namespace
    doc["page_content"] = text
    doc["embedding"] = [float(x) for x in (embedding or [])]
    doc["content_hash"] = hash_value or content_hash(text)
    doc["synced_at"] = synced_at
    doc["is_stale"] = bool(is_stale)
    doc["metadata"] = {
        "type": metadata_type,
        "source_document": source_document,
        "orderid": order.get("orderid"),
        "ordernumber": order.get("ordernumber"),
        "synced_at": synced_at,
        "embedding_dimensions": len(embedding or []),
        "structured": True,
    }
    return doc
