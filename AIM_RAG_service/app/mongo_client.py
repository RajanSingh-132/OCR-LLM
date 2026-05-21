"""
MongoDB client and database operations
"""
import os
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

MONGO_URI = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME = os.environ.get("DB_NAME", "")
MONGO_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ocr_db")


def _to_python_types(value):
    """Convert numpy types and complex Python objects to JSON-serializable types"""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            k: _to_python_types(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _to_python_types(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _to_python_types(v)
            for v in value
        ]

    if isinstance(value, set):
        return [
            _to_python_types(v)
            for v in value
        ]

    return value


def get_mongo_collection():
    """Get MongoDB collection with proper indexes"""
    if not MONGO_URI or not MONGO_DB_NAME:
        raise ValueError(
            "Mongo config missing. Please set "
            "mongo_db.MONGO_URI and mongo_db.DB_NAME"
        )

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=15000
    )

    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    # Indexes for fast namespace and source-document filtering.
    collection.create_index([("namespace", 1)])
    collection.create_index(
        [("namespace", 1), ("metadata.source_document", 1)]
    )

    return collection
