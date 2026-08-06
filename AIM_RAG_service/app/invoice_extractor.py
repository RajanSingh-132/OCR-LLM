import asyncio
import datetime
import json
import os
from typing import Dict, List, Tuple

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_models
from app.invoice_prompt import INVOICE_EXTRACTION_PROMPT
from app.lan_chain_rag_semantic_parent import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    _load_file_pages,
    ingest_pdf_and_return_json_async,
)
from app.mongo_client import _to_python_types, get_mongo_collection


INVOICE_MONGO_COLLECTION = "invoice_json"
INVOICE_VECTOR_NAMESPACE = "invoice_json"
INVOICE_RECORD_NAMESPACE = "invoice_json_records"
SUPPORTED_INVOICE_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


def clean_json_response(text_response: str) -> str:
    cleaned = (text_response or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def is_empty_value(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        val_strip = val.strip().upper()
        return val_strip == "" or val_strip in ("N/A", "NA", "NOT AVAILABLE", "NONE", "NULL")
    if isinstance(val, (list, dict)):
        return len(val) == 0
    return False


def clean_empty_fields(data):
    if isinstance(data, dict):
        return {
            k: clean_empty_fields(v)
            for k, v in data.items()
            if not is_empty_value(v) and clean_empty_fields(v) is not None
        }
    elif isinstance(data, list):
        return [
            clean_empty_fields(item)
            for item in data
            if not is_empty_value(item) and clean_empty_fields(item) is not None
        ]
    return data

def normalize_invoice_payload(payload) -> Dict[str, object]:
    if isinstance(payload, list):
        invoices = payload
        payload = {
            "document_type": "invoice",
            "invoices": invoices
        }

    if not isinstance(payload, dict):
        return {
            "document_type": "invoice",
            "invoices": []
        }

    invoices = payload.get("invoices")
    if isinstance(invoices, dict):
        invoices = [invoices]
    elif not isinstance(invoices, list):
        invoices = []

    normalized_invoices = []
    for invoice in invoices:
        if not isinstance(invoice, dict):
            continue

        line_items = invoice.get("line_items")
        if isinstance(line_items, dict):
            invoice["line_items"] = [line_items]
        elif not isinstance(line_items, list):
            invoice["line_items"] = []

        normalized_invoices.append(invoice)

    payload["document_type"] = payload.get("document_type") or "invoice"
    payload["invoices"] = normalized_invoices
    return payload


def build_common_invoice_data(
        batch_id: str,
        filename: str,
        record_id: str,
        extracted_json: Dict[str, object]
) -> List[Dict[str, object]]:
    common_data = []
    invoices = extracted_json.get("invoices", [])

    if not isinstance(invoices, list):
        return common_data

    for index, invoice in enumerate(invoices, start=1):
        if not isinstance(invoice, dict):
            continue

        common_data.append({
            "batch_id": batch_id,
            "source_file": filename,
            "record_id": record_id,
            "invoice_index": index,
            "vendor_name": invoice.get("vendor_name"),
            "vendor_address": invoice.get("vendor_address"),
            "vendor_phone": invoice.get("vendor_phone"),
            "vendor_email": invoice.get("vendor_email"),
            "vendor_tax_id": invoice.get("vendor_tax_id"),
            "customer_name": invoice.get("customer_name"),
            "customer_address": invoice.get("customer_address"),
            "customer_phone": invoice.get("customer_phone"),
            "customer_email": invoice.get("customer_email"),
            "invoice_number": invoice.get("invoice_number"),
            "invoice_date": invoice.get("invoice_date"),
            "due_date": invoice.get("due_date"),
            "purchase_order_number": invoice.get("purchase_order_number"),
            "currency": invoice.get("currency"),
            "subtotal": invoice.get("subtotal"),
            "discount_amount": invoice.get("discount_amount"),
            "tax_amount": invoice.get("tax_amount"),
            "shipping_amount": invoice.get("shipping_amount"),
            "total_amount": invoice.get("total_amount"),
            "amount_paid": invoice.get("amount_paid"),
            "balance_due": invoice.get("balance_due"),
            "payment_terms": invoice.get("payment_terms"),
            "payment_method": invoice.get("payment_method"),
            "bank_details": invoice.get("bank_details"),
            "line_items": invoice.get("line_items") or [],
            "notes": invoice.get("notes")
        })

    return common_data


def extract_invoice_json_sync(file_bytes: bytes, filename: str) -> Tuple[Dict[str, object], str]:
    _, llm = get_models()
    pages = _load_file_pages(
        file_bytes=file_bytes,
        filename=filename,
        llm=llm
    )

    full_text = "\n".join(
        page.page_content
        for page in pages
        if page.page_content
    ).strip()

    if not full_text:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")

    prompt = PromptTemplate.from_template(INVOICE_EXTRACTION_PROMPT)
    chain = prompt | llm
    response = chain.invoke({"text": full_text[:16000]})
    text_response = response.content if hasattr(response, "content") else str(response)
    cleaned = clean_json_response(text_response)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {
            "document_type": "invoice",
            "invoices": [],
            "raw_model_output": text_response
        }

    return normalize_invoice_payload(payload), full_text


def store_invoice_record_sync(
        batch_id: str,
        filename: str,
        extracted_json: Dict[str, object],
        raw_text: str
) -> str:
    embeddings, _ = get_models()
    page_content = json.dumps(
        extracted_json,
        ensure_ascii=False,
        indent=2
    )
    embedding = embeddings.embed_query(page_content[:4000])

    now = datetime.datetime.utcnow().isoformat()
    collection = get_mongo_collection(INVOICE_MONGO_COLLECTION)
    result = collection.insert_one(_to_python_types({
        "namespace": INVOICE_RECORD_NAMESPACE,
        "batch_id": batch_id,
        "file_name": filename,
        "document_type": "invoice",
        "page_content": page_content,
        "embedding": [float(x) for x in embedding],
        "extracted_json": extracted_json,
        "raw_text": raw_text,
        "metadata": {
            "source_document": filename,
            "batch_id": batch_id,
            "type": "invoice_extraction",
            "created_at": now,
            "mongo_collection": INVOICE_MONGO_COLLECTION
        },
        "created_at": now
    }))
    return str(result.inserted_id)


async def extract_and_store_invoice_async(
        batch_id: str,
        file_bytes: bytes,
        filename: str
) -> Dict[str, object]:
    extracted_json, raw_text = await asyncio.to_thread(
        extract_invoice_json_sync,
        file_bytes,
        filename
    )
    extracted_json = clean_empty_fields(extracted_json)

    record_id = await asyncio.to_thread(
        store_invoice_record_sync,
        batch_id,
        filename,
        extracted_json,
        raw_text
    )
    common_data = build_common_invoice_data(
        batch_id=batch_id,
        filename=filename,
        record_id=record_id,
        extracted_json=extracted_json
    )
    common_data = [clean_empty_fields(item) for item in common_data]

    return {
        "file_name": filename,
        "record_id": record_id,
        "invoice_count": len(extracted_json.get("invoices", [])),
        "common_data": common_data,
        "extracted_json": extracted_json
    }


async def ingest_invoice_file_async(
        base_dir: str,
        batch_id: str,
        file_bytes: bytes,
        filename: str
):
    return await ingest_pdf_and_return_json_async(
        base_dir=base_dir,
        file_bytes=file_bytes,
        filename=filename,
        collection_name=INVOICE_VECTOR_NAMESPACE,
        mongo_collection_name=INVOICE_MONGO_COLLECTION,
        replace_namespace=False,
        extra_metadata={
            "batch_id": batch_id,
            "type": "invoice_source_chunk"
        }
    )


def is_supported_invoice_file(filename: str) -> bool:
    return os.path.splitext(filename or "")[1].lower() in SUPPORTED_INVOICE_EXTENSIONS


def allowed_invoice_extensions_text() -> str:
    return ", ".join(sorted(SUPPORTED_INVOICE_EXTENSIONS))
