import os
import logging
import json
import asyncio
import uuid
from typing import List
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.lan_chain_rag_semantic_parent import (
    ingest_pdf_and_return_json_async,
    extract_dynamic_kv_from_pdf_async,
    data_ingestion,
    get_vectorstore,
    get_models,
    get_mongo_collection
)
from app.prompt import ORDER_ANALYSIS_PROMPT
from app.semanticstore import GETORDERLIST_PATH
from app.invoice_extractor import (
    INVOICE_MONGO_COLLECTION,
    allowed_invoice_extensions_text,
    extract_and_store_invoice_async,
    ingest_invoice_file_async,
    is_supported_invoice_file,
    clean_empty_fields,
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
    allowed_extensions = allowed_pdf | allowed_images

    if file_ext not in allowed_extensions:
        logger.warning(f"Rejected upload: invalid extension '{file_ext}' for file {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF or image files ({', '.join(allowed_extensions)}) are allowed."
        )

    if file.content_type != "application/pdf" and not file.content_type.startswith("image/"):
        logger.warning(f"Rejected upload: invalid content-type '{file.content_type}' for file {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid content-type. Expected application/pdf or an image type."
        )

    pwd = os.path.dirname(os.path.realpath(__file__))

    try:
        # Read the file directly into memory (RAM)
        file_bytes = await file.read()
        logger.info(f"Read uploaded file {file.filename} into memory ({len(file_bytes)} bytes)")
            
        # 1. Dynamically extract JSON from PDF purely in-memory
        parsed_json = await extract_dynamic_kv_from_pdf_async(
            file_bytes=file_bytes,
            filename=file.filename
        )

        # 2. Embed into MongoDB asynchronously in the background purely in-memory
        background_tasks.add_task(
            ingest_pdf_and_return_json_async,
            base_dir=pwd,
            file_bytes=file_bytes,
            filename=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during in-memory processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    finally:
        await file.close()

    return {
        "extracted_json": parsed_json
    }


@app.post("/api/v1/invoices/extract")
async def upload_invoice_extract(
        request: Request,
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
):
    """
    Extract one or more invoice documents into a common invoice JSON schema.

    This endpoint stores invoice extraction records and invoice embeddings in
    the separate MongoDB collection named `invoice_json`.
    """
    _ = request

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one invoice file is required."
        )

    batch_id = str(uuid.uuid4())
    base_dir = os.path.dirname(os.path.realpath(__file__))
    results = []
    failed_files = []

    for file in files:
        logger.info(
            f"Received invoice upload: {file.filename} "
            f"(content_type={file.content_type}, batch_id={batch_id})"
        )

        try:
            if not is_supported_invoice_file(file.filename):
                raise ValueError(
                    f"Only invoice PDF or image files "
                    f"({allowed_invoice_extensions_text()}) are allowed."
                )

            content_type = file.content_type or ""
            if (
                    content_type
                    and content_type != "application/octet-stream"
                    and content_type != "application/pdf"
                    and not content_type.startswith("image/")
            ):
                raise ValueError(
                    "Invalid content-type. Expected application/pdf or an image type."
                )

            file_bytes = await file.read()
            if not file_bytes:
                raise ValueError(f"Uploaded file '{file.filename}' is empty.")

            result = await extract_and_store_invoice_async(
                batch_id=batch_id,
                file_bytes=file_bytes,
                filename=file.filename
            )
            results.append(result)

            background_tasks.add_task(
                ingest_invoice_file_async,
                base_dir=base_dir,
                batch_id=batch_id,
                file_bytes=file_bytes,
                filename=file.filename
            )

        except Exception as exc:
            logger.error(
                f"Invoice extraction failed for {file.filename}: {exc}",
                exc_info=True
            )
            failed_files.append({
                "file_name": file.filename,
                "error": str(exc)
            })
        finally:
            await file.close()

    if not results:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invoice extraction failed for all uploaded files.",
                "failed_files": failed_files
            }
        )

    # Calculate common fields across all extracted invoices
    all_invoices = []
    for item in results:
        all_invoices.extend(item.get("extracted_json", {}).get("invoices", []))

    common_fields = []
    if all_invoices:
        invoice_keys = [set(inv.keys()) for inv in all_invoices]
        common_keys = set.intersection(*invoice_keys)
        common_fields = [k for k in all_invoices[0].keys() if k in common_keys]

    return {
        "status": "success" if not failed_files else "partial_success",
        "total_files": len(files),
        "processed_files": len(results),
        "failed_files": failed_files,
        "invoices": all_invoices,
        "common_fields": common_fields
    }


# ==================== ORDER MANAGEMENT API ====================

class OrderQuery(BaseModel):
    question: str
    collection_name: str = "orders"


async def ingest_order_file_async(file_path: str, collection_name: str):
    """Ingest order data from file to MongoDB with embeddings"""
    base_dir = os.path.dirname(os.path.realpath(__file__))
    pwd = os.path.dirname(base_dir)
    
    return await asyncio.to_thread(
        data_ingestion,
        base_dir=pwd,
        file_paths=[file_path],
        collection_name=collection_name
    )


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

    """
    try:
        logger.info(f"Received order query: {query.question}")
        
        # Get models (embeddings + LLM)
        embeddings, llm = get_models()
        
        # ----------------------------------------------------------------------
        # STRICT LIVE GET API PIPELINE
        # Local JSON file path checking and Option A (local ingestion) are fully commented out.
        # This pipeline strictly and exclusively calls the live GET API.
        # ----------------------------------------------------------------------
        # order_file_path = GETORDERLIST_PATH
        # local_file_exists = os.path.exists(order_file_path)
        
        # 1. Force the GET API source name and print a loud console log for Vercel
        source_name = "live_api_orders.json"
        print(">>> [AIM RAG SYSTEM] STRICT LIVE GET API PIPELINE ACTIVE. Local files bypassed.")
        logger.info(f"Selected Strictly Live GET API Pipeline (Source: {source_name})")
        
        # 2. Check if the active source data is already ingested in MongoDB
        collection = get_mongo_collection()
        existing_count = collection.count_documents({
            "namespace": query.collection_name,
            "metadata.source_document": source_name
        }, limit=1)
        
        if existing_count > 0:
            print(f">>> [AIM RAG SYSTEM] Data already indexed in MongoDB (Source: {source_name}). Skipping ingestion.")
            logger.info(f"Source {source_name} already indexed. Proceeding to search.")
        else:
            print(f">>> [AIM RAG SYSTEM] DB is empty for source: '{source_name}'. Starting live API ingestion...")
            logger.info(f"Starting ingestion for {source_name}...")
            success = False

            # ==================================================================
            # OPTION A: LOCAL JSON FILE INGESTION (NOW FULLY COMMENTED OUT)
            # ==================================================================
            # if local_file_exists:
            #     print(f">>> [AIM RAG SYSTEM] Reading and ingesting from local file path: {order_file_path}")
            #     logger.info(f"Ingesting from local file: {order_file_path}")
            #     success = await ingest_order_file_async(order_file_path, query.collection_name)
            
            # ==================================================================
            # OPTION B: STRICT LIVE GET API INGESTION (ACTIVE)
            # ==================================================================
            print(">>> [AIM RAG SYSTEM] Calling external Live GET API...")
            logger.info("Calling external Live GET API...")
            
            # Fetch data from GET API live
            live_orders_json = await fetch_orders_from_api()
            json_bytes = json.dumps(live_orders_json).encode("utf-8")
            
            print(">>> [AIM RAG SYSTEM] Received live JSON data from GET API. Ingesting to MongoDB...")
            logger.info("Received live API data. Starting MongoDB in-memory ingestion...")
            
            success = await asyncio.to_thread(
                data_ingestion,
                base_dir=None,
                file_paths=None,
                collection_name=query.collection_name,
                file_bytes=json_bytes,
                filename=source_name  # Triggers JSON list parsing in-memory
            )
            
            if success:
                print(f">>> [AIM RAG SYSTEM] Live API Ingestion COMPLETED SUCCESSFULLY for source: '{source_name}'.")
                logger.info(f"Ingestion successful for {source_name}")
            else:
                print(f">>> [AIM RAG SYSTEM] ERROR: Live API Ingestion FAILED for source: '{source_name}'.")
                logger.error(f"Ingestion failed for {source_name}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to ingest order data from Live GET API. Please check logs or API endpoint."
                )
        
        # Get vectorstore for the order collection
        vectorstore = get_vectorstore(
            embeddings,
            None,
            query.collection_name,
            _docs=None  # Load existing documents
        )
        
        if not vectorstore:
            raise HTTPException(
                status_code=500,
                detail="Failed to access order database"
            )
        
        # Retrieve relevant orders using similarity search
        retrieved_docs = vectorstore.similarity_search(
            query=query.question,
            k=10
        )
        
        if not retrieved_docs:
            return {
                "question": query.question,
                "answer": "No relevant order data found matching your query.",
                "matches": [],
                "collection": query.collection_name
            }
        
        # Combine retrieved context
        context = "\n\n".join([doc.page_content[:1000] for doc in retrieved_docs])
        
        # Use LLM to generate answer with prompt template
        from langchain_core.prompts import PromptTemplate
        from app.prompt import ORDERBOT_CONVERSATION_PROMPT
        
        prompt_template = PromptTemplate.from_template(ORDERBOT_CONVERSATION_PROMPT)
        
        chain = prompt_template | llm
        
        response = chain.invoke({
            "context": context,
            "question": query.question
        })
        
        # Extract text content from response
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        return {
            "answer": answer_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing order query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process order query: {str(e)}"
        )
