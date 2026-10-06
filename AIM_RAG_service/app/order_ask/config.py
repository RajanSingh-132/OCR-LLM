"""
Avaal Orders Ask API — Mongo collection config + terminal checkpoints.

DB: same as .env DB_NAME (chatbot_db)
Collection (ask-only): Avaal_order

checkpoint() / CheckpointTimer live here because this module imports nothing
else from the app, so every ask module can import them without a circular
import.
"""
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

# app/order_ask/config.py -> service root is 3 levels up
_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
load_dotenv(os.path.join(_SERVICE_ROOT, ".env"))

AVAAL_COLLECTION_NAME = os.environ.get("AVAAL_COLLECTION_NAME", "Avaal_order")
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


# ---------------------------------------------------------------------------
# Terminal checkpoints — printed as [CHECKPOINT] lines so you can watch the
# ask pipeline while the server runs.
# ---------------------------------------------------------------------------
_checkpoint_logger = logging.getLogger("order_ask.checkpoint")


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def checkpoint(step: str, detail: str = "", **extra: Any) -> None:
    """Print + log a visible runtime checkpoint."""
    bits = [f"[CHECKPOINT] {step}"]
    if detail:
        bits.append(f"- {detail}")
    if extra:
        kv = ", ".join(f"{k}={_short(v)}" for k, v in extra.items())
        bits.append(f"| {kv}")
    line = " ".join(bits)
    print(line, flush=True)
    _checkpoint_logger.info(line)


class CheckpointTimer:
    """Simple step timer for one ask request."""

    def __init__(self, label: str = "ask"):
        self.label = label
        self.t0 = time.perf_counter()
        self.last = self.t0

    def mark(self, step: str, detail: str = "", **extra: Any) -> None:
        now = time.perf_counter()
        step_ms = int((now - self.last) * 1000)
        total_ms = int((now - self.t0) * 1000)
        checkpoint(
            step,
            detail,
            step_ms=step_ms,
            total_ms=total_ms,
            **extra,
        )
        self.last = now
