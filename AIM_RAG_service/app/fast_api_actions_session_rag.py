import os
import json
import logging
import asyncio
import threading
import queue
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.lan_chain_rag_semantic_parent import (
    ingest_pdf_and_return_json_async,
    extract_dynamic_kv_from_pdf_async,
    pdf_extract_ckpt,
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
    1. Auto Ingest getorderlist.json or live GET API with embeddings
    2. Retrieve relevant orders based on the question
    3. Use LLM to generate intelligent response

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


_STREAM_SENTINEL = object()


async def _sse_from_sync_generator(sync_gen_factory):
    """
    Bridges a blocking sync generator (Mongo + Bedrock + Claude .stream() calls)
    into an async generator the event loop can await on, without blocking it.

    A worker thread drives the sync generator and pushes each event onto a
    plain queue.Queue; this coroutine awaits queue.get() via the default
    executor so other requests keep being served while we wait for the next
    chunk.
    """
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            for event in sync_gen_factory():
                q.put(event)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            q.put({"type": "error", "error": str(exc)})
        finally:
            q.put(_STREAM_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    loop = asyncio.get_event_loop()
    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is _STREAM_SENTINEL:
            break
        yield event


@app.post("/api/v1/orders/ask/stream")
async def ask_order_question_stream(query: OrderQuery):
    """
    Same Q&A pipeline as POST /api/v1/orders/ask, but streamed over
    Server-Sent Events (SSE) so the answer appears token-by-token instead of
    only after the full reply is ready.

    Each SSE frame is `data: <json>\\n\\n` with one of:
      {"type": "chunk", "text": "..."}     — a piece of the answer text; append
                                              these in order to build the reply
                                              live as it streams in
      {"type": "final", "data": {...}}     — the complete response object,
                                              identical in shape to what
                                              POST /api/v1/orders/ask returns
                                              (session_id, matches, calculation,
                                              analytics, etc.) — sent once, last
      {"type": "error", "error": "..."}    — sent instead of "final" on failure

    Everything before the last Claude call (tenant/session resolve, query
    planner, Mongo retrieval/tools) still runs to completion first — only the
    final answer-generation call actually streams token-by-token.
    """
    logger.info(
        "Received Avaal order query (stream): %s | corporate_id=%s | session_id=%s",
        query.question,
        query.corporate_id,
        query.session_id,
    )
    from app.order_ask.rag_engine import stream_order_question

    async def event_source():
        try:
            async for event in _sse_from_sync_generator(
                lambda: stream_order_question(
                    query.question, True, 10, query.session_id, query.corporate_id
                )
            ):
                etype = event.get("type")
                if etype == "chunk":
                    payload = {"type": "chunk", "text": event.get("text", "")}
                elif etype == "final":
                    payload = {"type": "final", "data": event.get("response")}
                else:
                    payload = {"type": "error", "error": event.get("error") or "unknown error"}
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error(f"Error streaming order query: {str(exc)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx: don't buffer the whole response before forwarding it
            "X-Accel-Buffering": "no",
        },
    )
