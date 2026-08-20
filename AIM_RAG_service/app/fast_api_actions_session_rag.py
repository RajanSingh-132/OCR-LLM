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
# from app.prompt import ORDER_ANALYSIS_PROMPT  # unused after Avaal ask rewrite
# Invoice extract API temporarily disabled
# from app.invoice_extractor import (
#     INVOICE_MONGO_COLLECTION,
#     allowed_invoice_extensions_text,
#     extract_and_store_invoice_async,
#     ingest_invoice_file_async,
#     is_supported_invoice_file,
#     clean_empty_fields,
# )

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
        # (Groq vision OCR if needed → Claude JSON — no MongoDB dependency)
        parsed_json = await extract_dynamic_kv_from_pdf_async(
            file_bytes=file_bytes,
            filename=file.filename
        )
        print(f"[pdf_extract] extract_dynamic_kv_from_pdf_async() done — file={file.filename}")

        # 2. MongoDB embed/ingest disabled on this branch (does not affect JSON extract)
        # background_tasks.add_task(
        #     ingest_pdf_and_return_json_async,
        #     base_dir=pwd,
        #     file_bytes=file_bytes,
        #     filename=file.filename
        # )
        # print(f"[pdf_extract] background ingest scheduled — file={file.filename}")
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


# Invoice extract API temporarily disabled
# @app.post("/api/v1/invoices/extract")
# async def upload_invoice_extract(
#         request: Request,
#         background_tasks: BackgroundTasks,
#         files: List[UploadFile] = File(...),
# ):
#     """
#     Extract one or more invoice documents into a common invoice JSON schema.
#
#     This endpoint stores invoice extraction records and invoice embeddings in
#     the separate MongoDB collection named `invoice_json`.
#     """
#     _ = request
#
#     if not files:
#         raise HTTPException(
#             status_code=400,
#             detail="At least one invoice file is required."
#         )
#
#     batch_id = str(uuid.uuid4())
#     base_dir = os.path.dirname(os.path.realpath(__file__))
#     results = []
#     failed_files = []
#
#     for file in files:
#         logger.info(
#             f"Received invoice upload: {file.filename} "
#             f"(content_type={file.content_type}, batch_id={batch_id})"
#         )
#
#         try:
#             if not is_supported_invoice_file(file.filename):
#                 raise ValueError(
#                     f"Only invoice PDF or image files "
#                     f"({allowed_invoice_extensions_text()}) are allowed."
#                 )
#
#             content_type = file.content_type or ""
#             if (
#                     content_type
#                     and content_type != "application/octet-stream"
#                     and content_type != "application/pdf"
#                     and not content_type.startswith("image/")
#             ):
#                 raise ValueError(
#                     "Invalid content-type. Expected application/pdf or an image type."
#                 )
#
#             file_bytes = await file.read()
#             if not file_bytes:
#                 raise ValueError(f"Uploaded file '{file.filename}' is empty.")
#
#             result = await extract_and_store_invoice_async(
#                 batch_id=batch_id,
#                 file_bytes=file_bytes,
#                 filename=file.filename
#             )
#             results.append(result)
#
#             background_tasks.add_task(
#                 ingest_invoice_file_async,
#                 base_dir=base_dir,
#                 batch_id=batch_id,
#                 file_bytes=file_bytes,
#                 filename=file.filename
#             )
#
#         except Exception as exc:
#             logger.error(
#                 f"Invoice extraction failed for {file.filename}: {exc}",
#                 exc_info=True
#             )
#             failed_files.append({
#                 "file_name": file.filename,
#                 "error": str(exc)
#             })
#         finally:
#             await file.close()
#
#     if not results:
#         raise HTTPException(
#             status_code=400,
#             detail={
#                 "message": "Invoice extraction failed for all uploaded files.",
#                 "failed_files": failed_files
#             }
#         )
#
#     # Calculate common fields across all extracted invoices
#     all_invoices = []
#     for item in results:
#         all_invoices.extend(item.get("extracted_json", {}).get("invoices", []))
#
#     common_fields = []
#     if all_invoices:
#         invoice_keys = [set(inv.keys()) for inv in all_invoices]
#         common_keys = set.intersection(*invoice_keys)
#         common_fields = [k for k in all_invoices[0].keys() if k in common_keys]
#
#     return {
#         "status": "success" if not failed_files else "partial_success",
#         "total_files": len(files),
#         "processed_files": len(results),
#         "failed_files": failed_files,
#         "invoices": all_invoices,
#         "common_fields": common_fields
#     }


# ==================== ORDER MANAGEMENT API (Avaal_db + Anthropic) ====================

class OrderQuery(BaseModel):
    question: str
    # Kept for API compatibility; Avaal ask always uses Avaal_db / avaal_orders.
    collection_name: str = "avaal_orders"
    # Pass back session_id from previous response to continue conversation.
    session_id: str | None = None


# Legacy live-API helpers kept commented for reference (not used by Avaal ask flow).
# async def ingest_order_file_async(file_path: str, collection_name: str):
#     base_dir = os.path.dirname(os.path.realpath(__file__))
#     pwd = os.path.dirname(base_dir)
#     return await asyncio.to_thread(
#         data_ingestion,
#         base_dir=pwd,
#         file_paths=[file_path],
#         collection_name=collection_name
#     )
#
# import httpx
# async def fetch_orders_from_api() -> list:
#     api_url = "http://192.168.1.22:2090/api/Order/listorder"
#     headers = {"corporateid": "AFMQA", "Content-Type": "application/json"}
#     async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
#         response = await client.get(api_url, headers=headers)
#         response.raise_for_status()
#         return response.json()


# ==============================================================================
# PLACE TO PUT YOUR GET API:
# Update the URL, query parameters, and authentication headers here as needed.
# ==============================================================================
import httpx
async def fetch_orders_from_api() -> list:
    """
    Fetch order data live from an external GET API.
    Configure this function with your actual external API endpoint.
    """
    # 1. PLACE YOUR GET API URL HERE:
    api_url = "http://192.168.1.22:2090/api/Order/listorder" 
    
    # 2. PLACE YOUR HEADERS / AUTHENTICATION TOKEN HERE (If needed):
    # Note: If your local API does not require Authorization, you can comment this out.
    headers = {
        "corporateid":"AFMQA",
        "Content-Type": "application/json"
    }
    
    # 3. HTTP GET request to pull the orders JSON payload:
    try:
        # We set verify=False because local IP HTTPS endpoints (e.g. 192.168.x.x)
        # usually use self-signed SSL certificates.
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as ce:
        logger.error(f"Failed to connect to API at {api_url}: {ce}")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not connect to your GET API link ({api_url}). "
                "This usually means your local server is offline, the port is wrong, "
                "or there is a local firewall block. "
                "If running on this same computer, please try changing '192.168.1.22' to 'localhost' or '127.0.0.1', or check if it uses HTTP instead of HTTPS."
            )
        )
    except httpx.ConnectTimeout as ct:
        logger.error(f"Timeout connecting to API at {api_url}: {ct}")
        raise HTTPException(
            status_code=504,
            detail=f"Connection timed out while trying to reach your GET API link ({api_url}). Check if the server is running."
        )
    except httpx.HTTPStatusError as hse:
        logger.error(f"API returned HTTP error: {hse.response.status_code} - {hse.response.text}")
        raise HTTPException(
            status_code=hse.response.status_code,
            detail=f"Your GET API link returned an HTTP error: {hse.response.status_code}. Response: {hse.response.text[:200]}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching orders from API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while fetching from your GET API link: {str(e)}"
        )


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
