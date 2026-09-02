"""Unwrap whatever envelope the Avaal order-list API (or a saved dump) returns
into a plain ``list[dict]`` of order records.

Handles, in one place, every shape seen so far:
- a bare JSON array                              ``[ {...}, {...} ]``
- ``{"details": [ {...} ]}``                     (list)
- ``{"details": "[ {...} ]"}``                   (escaped-JSON string)
- ``{"data": [...]}`` / ``{"result": [...]}`` / ``{"orders": [...]}``
- ``{"data": {"details"|"list"|"items": [...]}}``  (one level of nesting)
- a single order object                          ``{ "orderid": ... }``
"""
from __future__ import annotations

import json
from typing import Any, List

# Keys that, at the top level or one level down, hold the list of records.
_LIST_KEYS = ("details", "data", "result", "results", "orders", "list", "items", "rows")
# A record looks like an order if it carries one of these identifying keys.
_RECORD_MARKERS = ("orderid", "ordernumber", "OrderId", "OrderNumber")


def _looks_like_record(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in _RECORD_MARKERS)


def _coerce_list(value: Any) -> Any:
    """A JSON string holding an array/object → the parsed value; else unchanged."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("[", "{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def unwrap_order_payload(payload: Any) -> List[dict]:
    """Return the list of order-record dicts from an already-parsed payload."""
    payload = _coerce_list(payload)

    if isinstance(payload, list):
        if payload and _looks_like_record(payload[0]):
            return [r for r in payload if isinstance(r, dict)]
        # e.g. [{"details": [...]}]
        if payload and isinstance(payload[0], dict):
            inner = unwrap_order_payload(payload[0])
            if inner:
                return inner
        return [r for r in payload if isinstance(r, dict)]

    if isinstance(payload, dict):
        if _looks_like_record(payload):
            # a single order object
            return [payload]
        for key in _LIST_KEYS:
            if key in payload:
                candidate = _coerce_list(payload[key])
                if isinstance(candidate, list):
                    return [r for r in candidate if isinstance(r, dict)]
                if isinstance(candidate, dict):
                    nested = unwrap_order_payload(candidate)
                    if nested:
                        return nested
        # last resort: first list-of-dicts value anywhere in the dict
        for value in payload.values():
            candidate = _coerce_list(value)
            if (
                isinstance(candidate, list)
                and candidate
                and isinstance(candidate[0], dict)
            ):
                return [r for r in candidate if isinstance(r, dict)]

    return []


def read_json_payload_text(raw: str) -> Any:
    """Parse a possibly-truncated JSON dump file (mirrors the old ingest reader)."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty payload")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        wrapped = raw.rstrip(",").strip()
        if not wrapped.startswith("{"):
            wrapped = "{" + wrapped
        if not wrapped.endswith("}"):
            wrapped = wrapped + "}"
        return json.loads(wrapped)
