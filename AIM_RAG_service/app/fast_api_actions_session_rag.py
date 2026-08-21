import os
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.lan_chain_rag_semantic_parent import (
    ingest_pdf_and_return_json_async,
    extract_dynamic_kv_from_pdf_async,
)

logger = logging.getLogger("api")

from fastapi.openapi.utils import get_openapi

# Initialize FastAPI application
app = FastAPI(title="ocr")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ocr",
        version="0.1.0",
        routes=app.routes,
    )
    # Fix Swagger UI not showing file upload inputs for OpenAPI 3.1.0+
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = schema.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "array":
                items = prop.get("items", {})
                if items.get("contentMediaType") == "application/octet-stream":
                    items["format"] = "binary"
            elif prop.get("contentMediaType") == "application/octet-stream":
                prop["format"] = "binary"
    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi


@app.post("/api/v1/upload/pdf_dynamic_extract")
async def upload_pdf_dynamic_extract(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
):
    logger.info(f"Received file upload request: {file.filename} (content_type={file.content_type})")

    file_ext = os.path.splitext(file.filename)[1].lower()

    allowed_pdf = {".pdf"}
    allowed_images = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    allowed_word = {".docx", ".doc"}
    allowed_extensions = allowed_pdf | allowed_images | allowed_word

    if file_ext not in allowed_extensions:
        logger.warning(f"Rejected upload: invalid extension '{file_ext}' for file {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF, image, or Word files ({', '.join(sorted(allowed_extensions))}) are allowed."
        )

    content_type = (file.content_type or "").lower()
    allowed_word_content_types = {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    }
    is_pdf = content_type == "application/pdf"
    is_image = content_type.startswith("image/")
    is_word = file_ext in allowed_word and (
        content_type in allowed_word_content_types or content_type == ""
    )
    if content_type and not (is_pdf or is_image or is_word):
        logger.warning(f"Rejected upload: invalid content-type '{file.content_type}' for file {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid content-type. Expected application/pdf, an image type, or a Word document type."
        )

    # pwd = os.path.dirname(os.path.realpath(__file__))  # only needed for Mongo ingest

    try:
        print(f"[pdf_extract] API start — file={file.filename}")
        # Read the file directly into memory (RAM)
        file_bytes = await file.read()
        logger.info(f"Read uploaded file {file.filename} into memory ({len(file_bytes)} bytes)")
        print(f"[pdf_extract] file.read() done — {len(file_bytes)} bytes")

        # 1. Dynamically extract JSON from PDF purely in-memory
        # (Claude Sonnet vision OCR if needed → Claude Sonnet JSON — no MongoDB dependency)
        parsed_json = await extract_dynamic_kv_from_pdf_async(
            file_bytes=file_bytes,
            filename=file.filename
        )
        print(f"[pdf_extract] extract_dynamic_kv_from_pdf_async() done — file={file.filename}")

        print(f"[pdf_extract] background Mongo ingest SKIPPED (commented out)")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during in-memory processing: {e}", exc_info=True)
        print(f"[pdf_extract] API FAILED — {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    finally:
        await file.close()

    print(f"[pdf_extract] API complete — returning extracted_json for {file.filename}")
    return {
        "extracted_json": parsed_json
    }

# ==================== ORDER MANAGEMENT API (Avaal_db + Anthropic) ====================

class OrderQuery(BaseModel):
    question: str
    # Kept for API compatibility; Avaal ask always uses Avaal_db / avaal_orders.
    collection_name: str = "avaal_orders"
    # Pass back session_id from previous response to continue conversation.
    session_id: str | None = None

@app.post("/api/v1/orders/ask")
async def ask_order_question(query: OrderQuery):
    """
    Query order data using RAG with embeddings.
    
    Process:
    1. Auto Ingest getorderlist.json or live GET API with embeddings
    2. Retrieve relevant orders based on the question
    3. Use LLM to generate intelligent response

    Send the same session_id on follow-up turns to keep memory
    (e.g. "uska status?" after looking up an order).

    Watch the server terminal for [CHECKPOINT] logs.
    """
    try:
        logger.info(
            "Received Avaal order query: %s | session_id=%s",
            query.question,
            query.session_id,
        )
        from app.order_ask.rag_engine import answer_order_question

        result = await asyncio.to_thread(
            answer_order_question,
            query.question,
            True,
            10,
            query.session_id,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing order query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process order query: {str(e)}"
        )
