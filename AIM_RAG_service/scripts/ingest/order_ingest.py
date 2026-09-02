"""
Manual order ingest — edit CONFIG, then run:

  python -m scripts.ingest.order_ingest

Uses Bedrock Titan embeddings only (no LLM).
Skips records when ordernumber already exists in the target collection.
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.embedding_client import get_embeddings
from app.mongo_client import _to_python_types, get_mongo_collection
from app.sync.payload import read_json_payload_text, unwrap_order_payload

logger = logging.getLogger("scripts.ingest.order")

# ===================== CONFIG — edit these =====================
FILE_PATH = r"D:\Desktop\OCR-LLM\AFN01514order.txt"
DB_NAME = "AFN01514"
COLLECTION_NAME = "Avaal_order"
NAMESPACE = "avaal_orders"
DUPLICATE_FIELD = "ordernumber"
WITH_EMBEDDINGS = True
SKIP_DUPLICATES = True
EMBED_BATCH_SIZE = 25
INSERT_BATCH_SIZE = 100
# Titan Text Embeddings V2 accepts at most 8192 input tokens per request.
# Cap the text sent for embedding safely below that ceiling (order text is
# number/ID heavy, so it packs more tokens per character than prose).
# The full record is still stored verbatim in `page_content`; only the
# embedding input is trimmed for unusually large orders.
MAX_EMBED_CHARS = 18000
# ===============================================================


def load_order_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Orders file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = read_json_payload_text(f.read())

    records = unwrap_order_payload(payload)
    if not records:
        raise ValueError(f"No order records found in {path}")
    logger.info("Loaded %s order records from %s", len(records), path)
    return records


def build_page_content(order: Dict[str, Any]) -> str:
    """Build RAG text from all keys present in the source record."""
    lines = []
    for key in sorted(order.keys()):
        value = order[key]
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _truncate_for_embedding(text: str) -> str:
    """Keep embedding input under the Titan 8192-token limit."""
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding text is %s chars; truncating to %s chars to stay under the "
        "Titan token limit",
        len(text),
        MAX_EMBED_CHARS,
    )
    return text[:MAX_EMBED_CHARS]


def build_document(
    order: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
    source_document: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    for key, value in order.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = NAMESPACE
    doc["page_content"] = build_page_content(order)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": "avaal_order",
        "source_document": source_document,
        "collection": COLLECTION_NAME,
        "database": DB_NAME,
        "orderid": order.get("orderid"),
        "ordernumber": order.get("ordernumber"),
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


def ingest_orders() -> Dict[str, Any]:
    records = load_order_records(FILE_PATH)
    if not records:
        return {"ok": False, "error": "No order records found", "path": FILE_PATH}

    collection = get_mongo_collection(COLLECTION_NAME, DB_NAME, ensure_indexes=True)
    source_document = os.path.basename(FILE_PATH)

    existing_keys: Set[str] = set()
    if SKIP_DUPLICATES:
        existing_keys = load_existing_keys(collection, DUPLICATE_FIELD)
        logger.info(
            "Loaded %s existing %s values for duplicate skip",
            len(existing_keys),
            DUPLICATE_FIELD,
        )

    embeddings = get_embeddings() if WITH_EMBEDDINGS else None
    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    inserted = 0
    skipped = 0
    batch_docs: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal inserted, batch_docs
        if not batch_docs:
            return
        collection.insert_many(batch_docs, ordered=False)
        inserted += len(batch_docs)
        logger.info("Inserted %s records so far", inserted)
        batch_docs = []

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        chunk = records[start : start + EMBED_BATCH_SIZE]
        pending: List[Dict[str, Any]] = []

        for order in chunk:
            key_value = order.get(DUPLICATE_FIELD)
            key_str = (
                str(key_value).strip()
                if key_value not in (None, "")
                else ""
            )
            if SKIP_DUPLICATES and key_str:
                if key_str in existing_keys:
                    skipped += 1
                    continue
                existing_keys.add(key_str)
            pending.append(order)

        if not pending:
            print(
                f"[order ingest] processed "
                f"{min(start + len(chunk), len(records))} / {len(records)}"
            )
            continue

        texts = [
            _truncate_for_embedding(build_page_content(order))
            for order in pending
        ]
        if embeddings is not None:
            vectors = embeddings.embed_documents(texts)
        else:
            vectors = [[] for _ in pending]

        for order, vector in zip(pending, vectors):
            batch_docs.append(
                build_document(order, vector, ingested_at, source_document)
            )

        if len(batch_docs) >= INSERT_BATCH_SIZE:
            flush_batch()

        print(
            f"[order ingest] processed "
            f"{min(start + len(chunk), len(records))} / {len(records)}"
        )

    flush_batch()

    collection.create_index([("namespace", 1), ("orderid", 1)])
    collection.create_index([("namespace", 1), ("ordernumber", 1)])

    stored = collection.count_documents(
        {"namespace": NAMESPACE, "metadata.type": "avaal_order"}
    )
    sample = collection.find_one(
        {"namespace": NAMESPACE, "metadata.type": "avaal_order"},
        {"embedding": 0},
    )

    return {
        "ok": True,
        "database": DB_NAME,
        "collection": COLLECTION_NAME,
        "namespace": NAMESPACE,
        "source_path": FILE_PATH,
        "source_document": source_document,
        "records_loaded": len(records),
        "documents_inserted": inserted,
        "documents_skipped_duplicates": skipped,
        "documents_in_namespace": stored,
        "with_embeddings": WITH_EMBEDDINGS,
        "skip_duplicates": SKIP_DUPLICATES,
        "sample_orderid": (sample or {}).get("orderid"),
        "sample_ordernumber": (sample or {}).get("ordernumber"),
        "sample_field_count": len([k for k in (sample or {}) if k != "_id"]),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.isfile(FILE_PATH):
        print(f"ERROR: File not found: {FILE_PATH}")
        return 1

    status = ingest_orders()

    print("\n=== Order ingest ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("====================\n")
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
