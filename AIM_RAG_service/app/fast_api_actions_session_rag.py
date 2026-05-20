import os
import logging
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks

from app.lan_chain_rag_semantic_parent import (
    ingest_pdf_and_return_json_async,
    extract_dynamic_kv_from_pdf_async
)

logger = logging.getLogger("api")

# Initialize FastAPI application
app = FastAPI(title="ocr")


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


