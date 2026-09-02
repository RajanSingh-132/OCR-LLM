import os
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.lan_chain_rag_semantic_parent import (
    ingest_pdf_and_return_json_async,
    extract_dynamic_kv_from_pdf_async,
    pdf_extract_ckpt,
)
from app import postgres_client

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

# ==================== ORDER MANAGEMENT API (Avaal_order + Anthropic) ====================

class OrderQuery(BaseModel):
    question: str
    # Identifies which company/tenant DB to use (routing wired in a later step).
    corporate_id: str
    # Kept for API compatibility; ask uses tenant orders collection (default Avaal_order).
    # collection_name: str = "avaal_orders"
    # Pass back session_id from previous response to continue conversation.
    session_id: str | None = None

@app.post("/api/v1/orders/ask")
async def ask_order_question(query: OrderQuery):
    """
    Query order data using RAG with embeddings.

    Process:
    0. Bucket 1 fast path: date-range / country / last-N-days COUNT
       questions are answered directly from live Postgres
       (fn_getorders_regular), no LLM call, no Mongo. See
       app/order_ask/bucket1_live.py — BUCKET1_PLAN.md.
    0b. Dynamic query builder (env AVAAL_DYNAMIC_QUERY_BUILDER=1 only):
       LLM -> validated JSON payload -> constrained Postgres query.
       When enabled it OWNS every non-Bucket-1 order question — an
       unsafe/unsupported question gets an honest message, never the
       Mongo pipeline. Off by default → step 1 behaviour is unchanged.
    1. Otherwise falls through to the existing Mongo + LLM pipeline.

    Send the same session_id and corporate_id on follow-up turns to keep memory
    (e.g. "uska status?" after looking up an order).

    Watch the server terminal for [CHECKPOINT] logs.
    """
    try:
        logger.info(
            "Received Avaal order query: %s | corporate_id=%s | session_id=%s",
            query.question,
            query.corporate_id,
            query.session_id,
        )

        # --- Bucket 1 fast path: live Postgres, no LLM ---
        try:
            from app.order_ask.bucket1_live import try_answer as bucket1_try_answer

            bucket1_result = await bucket1_try_answer(query.corporate_id, query.question)
            if bucket1_result is not None:
                logger.info("Bucket1 fast path answered: %s", bucket1_result.get("source"))
                return {
                    "answer": bucket1_result["answer"],
                    "session_id": query.session_id,
                    "source": bucket1_result["source"],
                }
        except postgres_client.TenantNotProvisionedError:
            # Not one of our live-Postgres test tenants yet — fall through
            # to the existing Mongo pipeline rather than failing the request.
            pass
        except Exception as bucket1_err:
            # Bucket 1 is a best-effort fast path — any failure here should
            # fall through to the normal pipeline, not break the request.
            logger.warning("Bucket1 fast path failed, falling back: %s", bucket1_err)

        # --- Dynamic query builder (env-flagged, off by default) ---
        if os.environ.get("AVAAL_DYNAMIC_QUERY_BUILDER", "0").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            try:
                from app.order_ask.dynamic_query_flow import run_dynamic_query_flow

                dq_result = await run_dynamic_query_flow(
                    query.corporate_id, query.question, query.session_id
                )
                logger.info("Dynamic query builder answered: %s", dq_result.get("source"))
                return {
                    "answer": dq_result["answer"],
                    "session_id": query.session_id,
                    "source": dq_result["source"],
                }
            except postgres_client.TenantNotProvisionedError:
                # Not a live-Postgres tenant — fall through to Mongo pipeline.
                pass

        # --- Existing pipeline (Mongo-backed today) ---
        from app.order_ask.rag_engine import answer_order_question

        result = await asyncio.to_thread(
            answer_order_question,
            query.question,
            True,
            10,
            query.session_id,
            query.corporate_id,
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


# ==================== LIVE AVAAL API → MONGO SYNC ====================

class OrderSyncRequest(BaseModel):
    corporate_id: str
    # "incremental" (default) = only rows changed since last cursor
    # "full" = pull everything and mark rows no longer present as stale
    mode: str = "incremental"


@app.post("/api/v1/orders/sync")
async def sync_orders_endpoint(body: OrderSyncRequest, request: Request):
    """Pull fresh orders from the live Avaal API into this tenant's Mongo store.

    Protected by the `X-Sync-Token` header (must equal env `AVAAL_SYNC_TOKEN`).
    """
    expected = os.environ.get("AVAAL_SYNC_TOKEN", "")
    if not expected or request.headers.get("X-Sync-Token") != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-Sync-Token")
    try:
        from app.sync.order_sync import sync_orders

        return await asyncio.to_thread(
            sync_orders, body.corporate_id.strip(), mode=body.mode
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Order sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="order sync failed")


@app.on_event("startup")
def _start_avaal_sync_scheduler():
    try:
        from app.sync.scheduler import start_scheduler

        start_scheduler()
    except Exception as e:  # never block API startup on the scheduler
        logger.warning("Avaal sync scheduler did not start: %s", e)
