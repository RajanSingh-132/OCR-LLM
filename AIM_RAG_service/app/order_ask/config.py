"""
Avaal Orders Ask API — Mongo collection config.

DB: same as .env DB_NAME (chatbot_db)
Collection (ask-only): Avaal_db
"""
import os
from dotenv import load_dotenv

# app/order_ask/config.py -> service root is 3 levels up
_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
load_dotenv(os.path.join(_SERVICE_ROOT, ".env"))

AVAAL_COLLECTION_NAME = os.environ.get("AVAAL_COLLECTION_NAME", "Avaal_db")
AVAAL_NAMESPACE = os.environ.get("AVAAL_NAMESPACE", "avaal_orders")
AVAAL_SOURCE_DOCUMENT = os.environ.get(
    "AVAAL_SOURCE_DOCUMENT",
    "orderdata.txt",
)

AVAAL_DATA_DIR = os.path.join(_SERVICE_ROOT, "avaal_orders", "data")

AVAAL_ORDERS_JSON_PATH = os.environ.get(
    "AVAAL_ORDERS_JSON_PATH",
    os.path.join(_SERVICE_ROOT, "orderdata.txt"),
)

# Conversation sessions (same DB, separate collection)
AVAAL_SESSION_COLLECTION = os.environ.get(
    "AVAAL_SESSION_COLLECTION",
    "avaal_chat_sessions",
)
AVAAL_SESSION_MAX_TURNS = int(os.environ.get("AVAAL_SESSION_MAX_TURNS", "20"))

# Drop weak semantic matches below this score (cosine similarity)
AVAAL_RAG_MIN_SCORE = float(os.environ.get("AVAAL_RAG_MIN_SCORE", "0.28"))
