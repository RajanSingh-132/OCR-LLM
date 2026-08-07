"""
One-time / safe setup for Avaal_db collection used by /api/v1/orders/ask.

Creates the collection (if missing), indexes, and a setup marker doc.
Does NOT ingest order records — put JSON in avaal_orders/data/ later.
"""
import datetime
import logging

from app.order_ask.config import (
    AVAAL_COLLECTION_NAME,
    AVAAL_NAMESPACE,
    AVAAL_SOURCE_DOCUMENT,
    AVAAL_DATA_DIR,
    AVAAL_ORDERS_JSON_PATH,
)
from app.mongo_client import get_mongo_collection, MONGO_DB_NAME

logger = logging.getLogger("order_ask.setup")


def ensure_avaal_data_dir() -> str:
    os_makedirs = __import__("os").makedirs
    os_makedirs(AVAAL_DATA_DIR, exist_ok=True)
    return AVAAL_DATA_DIR


def setup_avaal_collection() -> dict:
    """
    Ensure Mongo DB + Avaal_db collection exist with indexes.
    Returns a small status dict.
    """
    ensure_avaal_data_dir()

    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)

    # Extra index helpful for order-id style metadata later
    collection.create_index([("namespace", 1), ("metadata.orderid", 1)])
    collection.create_index([("namespace", 1), ("metadata.ordernumber", 1)])

    marker_filter = {
        "namespace": AVAAL_NAMESPACE,
        "metadata.type": "collection_setup_marker",
        "metadata.source_document": AVAAL_SOURCE_DOCUMENT,
    }
    marker_doc = {
        "namespace": AVAAL_NAMESPACE,
        "page_content": (
            f"Avaal orders collection setup marker. "
            f"Place static JSON at: {AVAAL_ORDERS_JSON_PATH}"
        ),
        "embedding": [],
        "metadata": {
            "type": "collection_setup_marker",
            "source_document": AVAAL_SOURCE_DOCUMENT,
            "collection": AVAAL_COLLECTION_NAME,
            "database": MONGO_DB_NAME,
            "setup_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "orders_ingested": False,
        },
    }
    collection.update_one(marker_filter, {"$set": marker_doc}, upsert=True)

    total_docs = collection.count_documents({})
    namespace_docs = collection.count_documents({"namespace": AVAAL_NAMESPACE})

    status = {
        "ok": True,
        "database": MONGO_DB_NAME,
        "collection": AVAAL_COLLECTION_NAME,
        "namespace": AVAAL_NAMESPACE,
        "source_document": AVAAL_SOURCE_DOCUMENT,
        "data_dir": AVAAL_DATA_DIR,
        "expected_json_path": AVAAL_ORDERS_JSON_PATH,
        "total_documents_in_collection": total_docs,
        "documents_in_namespace": namespace_docs,
        "message": (
            f"Collection '{AVAAL_COLLECTION_NAME}' ready on DB '{MONGO_DB_NAME}'. "
            f"Put your static orders JSON at: {AVAAL_ORDERS_JSON_PATH}"
        ),
    }
    logger.info(status["message"])
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = setup_avaal_collection()
    for key, value in result.items():
        print(f"{key}: {value}")
