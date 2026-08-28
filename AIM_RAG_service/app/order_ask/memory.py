"""
Conversation memory for Avaal OrderBot (session turns + last entities).

Stored in Mongo collection avaal_chat_sessions (same DB ).
"""
from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from app.mongo_client import get_mongo_collection
from app.order_ask.checkpoint import checkpoint
from app.order_ask.config import AVAAL_SESSION_COLLECTION, AVAAL_SESSION_MAX_TURNS


def new_session_id() -> str:
    return str(uuid.uuid4())


def _sessions():
    return get_mongo_collection(AVAAL_SESSION_COLLECTION)


def load_session(session_id: Optional[str]) -> Dict[str, Any]:
    """Load or create a chat session."""
    if not session_id:
        session_id = new_session_id()
        checkpoint("MEMORY", "new session created", session_id=session_id)
        return {
            "session_id": session_id,
            "corporate_id": None,
            "last_domain": None,
            "turns": [],
            "last_order_token": None,
            "last_entities": {},
            "created": True,
        }

    doc = _sessions().find_one({"session_id": session_id})
    if not doc:
        checkpoint("MEMORY", "session not found — starting fresh", session_id=session_id)
        return {
            "session_id": session_id,
            "corporate_id": None,
            "last_domain": None,
            "turns": [],
            "last_order_token": None,
            "last_entities": {},
            "created": True,
        }

    turns = list(doc.get("turns") or [])[-AVAAL_SESSION_MAX_TURNS:]
    checkpoint(
        "MEMORY",
        "session loaded",
        session_id=session_id,
        turns=len(turns),
        last_order=doc.get("last_order_token"),
    )
    return {
        "session_id": session_id,
        "corporate_id": doc.get("corporate_id"),
        "last_domain": doc.get("last_domain"),
        "turns": turns,
        "last_order_token": doc.get("last_order_token"),
        "last_entities": doc.get("last_entities") or {},
        "created": False,
    }


def format_history_for_prompt(turns: List[Dict[str, Any]], max_turns: int = 6) -> str:
    """Compact chat history for Anthropic prompts."""
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns[-max_turns:]:
        role = (t.get("role") or "user").upper()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def save_turn(
    session_id: str,
    question: str,
    answer: str,
    *,
    corporate_id: Optional[str] = None,
    domain: Optional[str] = None,
    order_token: Optional[str] = None,
    entities: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    intent: Optional[str] = None,
) -> None:
    """Append user+assistant turns and update sticky entities."""
    now = datetime.datetime.utcnow().isoformat() + "Z"
    user_turn = {"role": "user", "content": question, "ts": now, "intent": intent}
    bot_turn = {
        "role": "assistant",
        "content": answer,
        "ts": now,
        "mode": mode,
        "order_token": order_token,
    }

    update: Dict[str, Any] = {
        "$push": {
            "turns": {
                "$each": [user_turn, bot_turn],
                "$slice": -AVAAL_SESSION_MAX_TURNS,
            }
        },
        "$set": {"updated_at": now},
        "$setOnInsert": {
            "session_id": session_id,
            "created_at": now,
        },
    }
    set_fields: Dict[str, Any] = {}
    if corporate_id:
        set_fields["corporate_id"] = corporate_id
    if domain:
        set_fields["last_domain"] = domain
    if order_token:
        set_fields["last_order_token"] = order_token
    if entities:
        # merge sticky entities
        set_fields["last_entities"] = entities
    if set_fields:
        update["$set"].update(set_fields)

    _sessions().update_one({"session_id": session_id}, update, upsert=True)
    checkpoint(
        "MEMORY",
        "turn saved",
        session_id=session_id,
        order_token=order_token,
        mode=mode,
    )
