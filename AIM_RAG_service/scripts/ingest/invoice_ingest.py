"""
Manual invoice ingest — edit CONFIG, then run:

  python -m scripts.ingest.invoice_ingest

Uses Bedrock Titan embeddings only (no LLM).
Skips records when InvoiceNumber already exists in the target collection.
Only ingests records whose `createdon` is within the last calendar month.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.embedding_client import get_embeddings
from app.mongo_client import _to_python_types, get_mongo_collection

logger = logging.getLogger("scripts.ingest.invoice")

# ===================== CONFIG — edit these =====================
FILE_PATH = r"D:\Desktop\OCR-LLM\invoiceafmqa.txt"
DB_NAME = "chatbot_db"
COLLECTION_NAME = "Avaal_invoice"
NAMESPACE = "avaal_invoices"
METADATA_TYPE = "avaal_invoice"
# Source keys can be mixed-case; the first present wins.
DUPLICATE_FIELDS = ("InvoiceNumber", "invoicenumber")
ID_FIELDS = ("InvoiceID", "invoiceid", "id")
WITH_EMBEDDINGS = True
SKIP_DUPLICATES = True
# Only ingest records whose `createdon` falls within the last calendar month
# (relative to "now"). Anything older that is present in the source file is
# skipped — no embedding, no insert. Set to False to ingest every record.
FILTER_RECENT_ONLY = True
DATE_FIELDS = ("createdon", "CreatedOn", "InvoiceDate")
RECENT_MONTHS = 1
EMBED_BATCH_SIZE = 25
INSERT_BATCH_SIZE = 100
# Titan Text Embeddings V2 accepts at most 8192 input tokens per request.
# Invoice text is number/ID/punctuation heavy, so the tokenizer splits it far
# more aggressively than prose — empirically ~1.5 chars per token (vs ~4 for
# English text). A plain character cap therefore cannot guarantee we stay
# under the token ceiling, so we do two things:
#   1. Trim to MAX_EMBED_CHARS up front (cheap, handles the common case).
#   2. If Bedrock still rejects a text for "Too many input tokens", shrink it
#      further and retry (see `_embed_texts` / `_embed_one_with_retry`).
# The full record is always stored verbatim in `page_content`; only the
# embedding input is trimmed for unusually large invoices.
MAX_EMBED_TOKENS = 8192
EST_CHARS_PER_TOKEN = 1.4
MAX_EMBED_CHARS = int(MAX_EMBED_TOKENS * EST_CHARS_PER_TOKEN * 0.85)  # ~9700
# Keys in the source payload that may hold the list of invoice records.
DETAIL_KEYS = ("details", "invoices", "detailsinvoices")
# ===============================================================


def _first_present(record: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _read_json_payload(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Empty file: {path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        wrapped = raw.rstrip(",").strip()
        if not wrapped.startswith(("{", "[")):
            wrapped = "{" + wrapped
        if wrapped.startswith("{") and not wrapped.endswith("}"):
            wrapped = wrapped + "}"
        return json.loads(wrapped)


def _extract_details(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in DETAIL_KEYS:
            if key in payload:
                return payload[key]
        return payload
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            for key in DETAIL_KEYS:
                if key in payload[0]:
                    return payload[0][key]
            return [r for r in payload if isinstance(r, dict)]
        return payload
    raise ValueError("Unsupported file format")


def load_invoice_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invoices file not found: {path}")

    payload = _read_json_payload(path)
    details = _extract_details(payload)

    if isinstance(details, str):
        details = json.loads(details)

    if not isinstance(details, list):
        raise ValueError("Invoice details must resolve to a list of records")

    records = [r for r in details if isinstance(r, dict)]
    logger.info("Loaded %s invoice records from %s", len(records), path)
    return records


def _subtract_months(moment: datetime.datetime, months: int) -> datetime.datetime:
    """Return `moment` shifted back by whole calendar months.

    The day-of-month is clamped when the target month is shorter (e.g. going
    back one month from the 31st lands on the 28th/30th).
    """
    month_index = (moment.year * 12 + (moment.month - 1)) - months
    year, month = divmod(month_index, 12)
    month += 1
    if month == 12:
        next_month_first = datetime.datetime(year + 1, 1, 1)
    else:
        next_month_first = datetime.datetime(year, month + 1, 1)
    last_day = (next_month_first - datetime.timedelta(days=1)).day
    day = min(moment.day, last_day)
    return moment.replace(year=year, month=month, day=day)


def _parse_datetime(value: Any) -> datetime.datetime | None:
    """Best-effort parse of an ISO-ish datetime string into an aware UTC dt."""
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(raw)
        except ValueError:
            dt = None
            for fmt in (
                "%m/%d/%Y %I:%M:%S %p",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(raw[:19])
                except ValueError:
                    return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def is_recent_record(invoice: Dict[str, Any], cutoff: datetime.datetime) -> bool:
    """True when the record's date field is on/after `cutoff`."""
    created = _parse_datetime(_first_present(invoice, DATE_FIELDS))
    if created is None:
        return False
    return created >= cutoff


def build_page_content(invoice: Dict[str, Any]) -> str:
    """Build RAG text from all keys present in the source record."""
    lines = []
    for key in sorted(invoice.keys()):
        value = invoice[key]
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _truncate_for_embedding(text: str) -> str:
    """First-pass trim to keep embedding input near the Titan 8192-token limit."""
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding text is %s chars; truncating to %s chars to stay under the "
        "Titan token limit",
        len(text),
        MAX_EMBED_CHARS,
    )
    return text[:MAX_EMBED_CHARS]


def _is_token_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "input token" in msg or "too many input tokens" in msg


def _embed_one_with_retry(embeddings, text: str) -> List[float]:
    """Embed a single text, shrinking and retrying if Bedrock rejects it."""
    attempt = text
    for _ in range(8):
        try:
            return embeddings.embed_query(attempt)
        except Exception as exc:  # noqa: BLE001
            if not _is_token_limit_error(exc):
                raise
            new_len = max(400, int(len(attempt) * 0.6))
            if new_len >= len(attempt):
                raise
            logger.warning(
                "Embedding rejected for token limit (%s chars); retrying with %s",
                len(attempt),
                new_len,
            )
            attempt = attempt[:new_len]
    raise RuntimeError("Could not shrink text below the Titan token limit")


def _embed_texts(embeddings, texts: List[str]) -> List[List[float]]:
    """Batch-embed; on a token-limit rejection fall back to per-text retry."""
    try:
        return embeddings.embed_documents(texts)
    except Exception as exc:  # noqa: BLE001
        if not _is_token_limit_error(exc):
            raise
        logger.warning(
            "Batch embed hit the token limit; falling back to per-text embedding "
            "with shrink-and-retry"
        )
        return [_embed_one_with_retry(embeddings, t) for t in texts]


def build_document(
    invoice: Dict[str, Any],
    embedding: List[float],
    ingested_at: str,
    source_document: str,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    for key, value in invoice.items():
        doc[key] = _to_python_types(value)

    doc["namespace"] = NAMESPACE
    doc["page_content"] = build_page_content(invoice)
    doc["embedding"] = [float(x) for x in embedding]
    doc["metadata"] = {
        "type": METADATA_TYPE,
        "source_document": source_document,
        "collection": COLLECTION_NAME,
        "database": DB_NAME,
        "invoiceid": _first_present(invoice, ID_FIELDS),
        "invoicenumber": _first_present(invoice, DUPLICATE_FIELDS),
        "ingested_at": ingested_at,
        "embedding_dimensions": len(embedding),
        "structured": True,
    }
    return doc


def load_existing_keys(collection, fields) -> Set[str]:
    existing: Set[str] = set()
    projection: Dict[str, int] = {}
    for field in fields:
        projection[field] = 1
        projection[f"metadata.{field}"] = 1
    projection["metadata.invoicenumber"] = 1
    for doc in collection.find({}, projection):
        value = None
        for field in fields:
            value = doc.get(field)
            if value not in (None, ""):
                break
        if value in (None, ""):
            meta = doc.get("metadata") or {}
            value = meta.get("invoicenumber")
            for field in fields:
                if value not in (None, ""):
                    break
                value = meta.get(field)
        if value not in (None, ""):
            existing.add(str(value).strip())
    return existing


def ingest_invoices() -> Dict[str, Any]:
    records = load_invoice_records(FILE_PATH)
    if not records:
        return {"ok": False, "error": "No invoice records found", "path": FILE_PATH}

    collection = get_mongo_collection(COLLECTION_NAME, DB_NAME, ensure_indexes=True)
    source_document = os.path.basename(FILE_PATH)

    existing_keys: Set[str] = set()
    if SKIP_DUPLICATES:
        existing_keys = load_existing_keys(collection, DUPLICATE_FIELDS)
        logger.info(
            "Loaded %s existing invoice numbers for duplicate skip",
            len(existing_keys),
        )

    embeddings = get_embeddings() if WITH_EMBEDDINGS else None
    now = datetime.datetime.now(datetime.timezone.utc)
    ingested_at = now.isoformat()
    recent_cutoff = (
        _subtract_months(now, RECENT_MONTHS).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if FILTER_RECENT_ONLY
        else None
    )
    if recent_cutoff is not None:
        logger.info(
            "Only ingesting records with createdon >= %s",
            recent_cutoff.isoformat(),
        )
    inserted = 0
    skipped = 0
    skipped_old = 0
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

        for invoice in chunk:
            if recent_cutoff is not None and not is_recent_record(
                invoice, recent_cutoff
            ):
                skipped_old += 1
                continue

            key_value = _first_present(invoice, DUPLICATE_FIELDS)
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
            pending.append(invoice)

        if not pending:
            print(
                f"[invoice ingest] processed "
                f"{min(start + len(chunk), len(records))} / {len(records)}"
            )
            continue

        texts = [
            _truncate_for_embedding(build_page_content(invoice))
            for invoice in pending
        ]
        if embeddings is not None:
            vectors = _embed_texts(embeddings, texts)
        else:
            vectors = [[] for _ in pending]

        for invoice, vector in zip(pending, vectors):
            batch_docs.append(
                build_document(invoice, vector, ingested_at, source_document)
            )

        if len(batch_docs) >= INSERT_BATCH_SIZE:
            flush_batch()

        print(
            f"[invoice ingest] processed "
            f"{min(start + len(chunk), len(records))} / {len(records)}"
        )

    flush_batch()

    collection.create_index([("namespace", 1), ("metadata.invoiceid", 1)])
    collection.create_index([("namespace", 1), ("metadata.invoicenumber", 1)])

    stored = collection.count_documents(
        {"namespace": NAMESPACE, "metadata.type": METADATA_TYPE}
    )
    sample = collection.find_one(
        {"namespace": NAMESPACE, "metadata.type": METADATA_TYPE},
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
        "documents_skipped_old": skipped_old,
        "recent_filter": (
            recent_cutoff.isoformat() if recent_cutoff is not None else None
        ),
        "documents_in_namespace": stored,
        "with_embeddings": WITH_EMBEDDINGS,
        "skip_duplicates": SKIP_DUPLICATES,
        "sample_invoiceid": (sample or {}).get("metadata", {}).get("invoiceid"),
        "sample_invoicenumber": (sample or {})
        .get("metadata", {})
        .get("invoicenumber"),
        "sample_field_count": len([k for k in (sample or {}) if k != "_id"]),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.isfile(FILE_PATH):
        print(f"ERROR: File not found: {FILE_PATH}")
        return 1

    status = ingest_invoices()

    print("\n=== Invoice ingest ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("======================\n")
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
