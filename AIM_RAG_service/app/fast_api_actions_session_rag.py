import os
import logging
import json
import asyncio
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

logger = logging.getLogger("api")

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


@app.post("/api/v1/orders/ask")
async def ask_order_question(query: OrderQuery):
    """
    Query order data using RAG with embeddings.
    
    Process:
    1. Auto Ingest getorderlist.json with embeddings
    2. Retrieve relevant orders based on the question
    3. Use LLM to generate intelligent response

    """
    try:
        logger.info(f"Received order query: {query.question}")
        
        # Get models (embeddings + LLM)
        embeddings, llm = get_models()
        
        # Check if getorderlist.txt exists and needs ingestion
        order_file_path = GETORDERLIST_PATH
        
        # Ingest order data if not already in DB
        collection = get_mongo_collection()
        existing_count = collection.count_documents({"namespace": query.collection_name}, limit=1)
        
        if existing_count == 0:
            if not os.path.exists(order_file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Order file not found at {order_file_path}"
                )
            
            logger.info(f"Ingesting order data from {order_file_path}")
            success = await ingest_order_file_async(order_file_path, query.collection_name)
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to ingest order data"
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


