
import asyncio
import os
import re
import base64
import mimetypes
import io
import pypdf
import pypdfium2
from typing import List
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_aws import BedrockEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_core.prompts import PromptTemplate

import json

# ---------------- CONFIG (.env) ----------------
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Commented out: replaced by Anthropic

# AWS Bedrock Configuration
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")

SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}
SUPPORTED_DATA_EXTENSIONS = {".json", ".txt"}





# ---------------- MODELS ----------------

# def get_models():
#     embeddings = NomicEmbeddings(
#         model=NOMIC_EMBED_MODEL,
#         nomic_api_key=NOMIC_EMBED_API_KEY,
#         dimensionality=NOMIC_EMBED_DIMENSION
#     )

#     # llm = AzureChatOpenAI(
#     #     azure_endpoint=data["AZURE_OPENAI_ENDPOINT"],
#     #     api_key=data["AZURE_OPENAI_KEY"],
#     #     api_version=data["AZURE_OPENAI_API_VERSION"],
#     #     deployment_name=data['AZURE_OPENAI_DEPLOYMENT_NAME'],
#     #     temperature=0.2
#     # )

#     # return embeddings, llm

from app.embedding_client import get_models, get_vision_llm
from app.mongo_client import get_mongo_collection, _to_python_types
from app.rag_retrieval import get_vectorstore
from app.prompt import DYNAMIC_EXTRACTION_PROMPT


# ---------------- METADATA ----------------

def extract_metadata(
        text: str,
        file_path: str = "",
        page: int = None,
        is_amendment: bool = False
):
    metadata = {
        "source_document": os.path.basename(file_path),
    }
    return metadata


def _extract_text_from_image(image_path: str = None, image_bytes: bytes = None, filename: str = None, llm = None) -> str:
    try:
        if image_bytes is not None:
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            source_name = filename or "image.jpg"
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()
                ).decode("utf-8")
            source_name = os.path.basename(image_path)

        mime_type, _ = mimetypes.guess_type(image_path or source_name)
        mime_type = mime_type or "image/jpeg"

        prompt = (
            "Extract readable text from this image exactly as present. "
            "If the image has no readable text, reply with: NO_TEXT_FOUND"
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        )

        # --- Groq vision LLM (commented out, replaced by Anthropic) ---
        # vision_llm = None
        # if GROQ_API_KEY:
        #     try:
        #         from langchain_groq import ChatGroq
        #         vision_llm = ChatGroq(
        #             model="meta-llama/llama-4-scout-17b-16e-instruct",
        #             groq_api_key=GROQ_API_KEY,
        #             temperature=0.0
        #         )
        #     except Exception as e:
        #         print(f"Error initializing vision ChatGroq: {e}")
        # -------------------------------------------------------------------

        # Always use the dedicated vision LLM (llama-4-scout) for multimodal image input.
        # The generic `llm` parameter is text-only (llama-3.3-70b) and does not accept
        # list-style content, which would raise: "messages[0].content must be a string".
        vision_llm = get_vision_llm()

        response = vision_llm.invoke([message])
        extracted = response.content if hasattr(response, "content") else str(response)
        extracted = (extracted or "").strip()

        if not extracted or extracted.upper() == "NO_TEXT_FOUND":
            return (
                f"No readable text found in image {source_name}"
            )

        return extracted

    except Exception as e:
        print(f"Image text extraction error: {str(e)}")
        return f"Image file {source_name}"


def _load_file_pages(file_path: str = None, file_bytes: bytes = None, filename: str = None, llm = None):
    if file_path:
        file_ext = os.path.splitext(file_path)[1].lower()
        source_name = os.path.basename(file_path)
    elif filename:
        file_ext = os.path.splitext(filename)[1].lower()
        source_name = filename
    else:
        raise ValueError("Either file_path or filename must be provided.")

    if file_ext in SUPPORTED_PDF_EXTENSIONS:
        # Resolve bytes whether the file came in-memory or from disk
        if file_bytes is None:
            with open(file_path, "rb") as f:
                file_bytes_local = f.read()
        else:
            file_bytes_local = file_bytes

        # --- TIER 1: Text layer extraction (pypdf, pure Python) ---
        reader = pypdf.PdfReader(io.BytesIO(file_bytes_local))
        page_texts = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")

        # --- TIER 2 & 3: Dynamic OCR — no fixed threshold ---
        # Strategy: for EVERY page, check if embedded images exist.
        # If images exist → OCR them and compare with text layer → keep the richer result.
        # This is fully dynamic: we don't guess based on character count.
        # A page with 200 chars of text but a full-page embedded photo still gets OCR'd.

        for i, page in enumerate(reader.pages):
            text_layer = page_texts[i].strip()

            # --- TIER 2: pypdf embedded image extraction (pure Python, Vercel-safe) ---
            # Photo-based PDFs (camera shot → PDF) store the image directly inside
            # the PDF as an embedded XObject. pypdf can extract those bytes without
            # any native binary, making this path fully compatible with Vercel.
            tier2_ocr_text = ""
            try:
                embedded_images = page.images  # list of ImageFile objects (pypdf 3+)
                if embedded_images:
                    page_ocr_parts = []
                    for idx, img_obj in enumerate(embedded_images):
                        try:
                            img_data = img_obj.data           # raw image bytes
                            img_name = getattr(img_obj, "name", None) or \
                                       f"{source_name}_p{i}_img{idx}.png"
                            ocr_text = _extract_text_from_image(
                                image_bytes=img_data,
                                filename=img_name,
                                llm=llm
                            )
                            if ocr_text and "NO_TEXT_FOUND" not in ocr_text.upper():
                                page_ocr_parts.append(ocr_text)
                        except Exception as img_err:
                            print(f"Embedded image OCR error page {i} img {idx}: {img_err}")
                    if page_ocr_parts:
                        tier2_ocr_text = "\n".join(page_ocr_parts)
            except Exception as emb_err:
                print(f"pypdf embedded image extraction failed page {i}: {emb_err}")

            if tier2_ocr_text:
                # Dynamically pick the richer result: OCR text vs text layer
                # More content = more extracted information from the invoice
                if len(tier2_ocr_text.strip()) >= len(text_layer):
                    page_texts[i] = tier2_ocr_text
                # else: text layer was already richer, keep it
                continue  # page resolved, skip Tier 3

            # --- TIER 3: pypdfium2 page rendering (local fallback, may fail on Vercel) ---
            # Only reached when page has NO embedded images (not a photo PDF).
            # Useful for PDFs where content is drawn as vector/path graphics.
            # Trigger only if text layer is also sparse (dynamic: fewer than 30 real words).
            real_words = [w for w in text_layer.split() if len(w) > 1]
            if len(real_words) < 30:
                try:
                    pdf_doc = pypdfium2.PdfDocument(io.BytesIO(file_bytes_local))
                    pdf_page = pdf_doc[i]
                    bitmap = pdf_page.render(scale=2.08, rotation=0)  # ~150 DPI
                    pil_image = bitmap.to_pil()
                    img_bytes_io = io.BytesIO()
                    pil_image.save(img_bytes_io, format="PNG")
                    img_bytes = img_bytes_io.getvalue()
                    ocr_text = _extract_text_from_image(
                        image_bytes=img_bytes,
                        filename=f"{source_name}_page{i}.png",
                        llm=llm
                    )
                    if ocr_text and len(ocr_text.strip()) > len(text_layer):
                        page_texts[i] = ocr_text
                    pdf_doc.close()
                except Exception as render_err:
                    print(f"pypdfium2 render fallback failed page {i} of {source_name}: {render_err}")

        docs = []
        for i, text in enumerate(page_texts):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_document": source_name,
                        "page": i
                    }
                )
            )
        return docs


    if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
        if file_bytes is not None:
            image_text = _extract_text_from_image(
                image_bytes=file_bytes,
                filename=source_name,
                llm=llm
            )
        else:
            image_text = _extract_text_from_image(
                image_path=file_path,
                llm=llm
            )
        return [
            Document(
                page_content=image_text,
                metadata={
                    "source_document": source_name,
                    "page": 0
                }
            )
        ]

    if file_ext in SUPPORTED_DATA_EXTENSIONS:
        try:
            if file_bytes is not None:
                content = file_bytes.decode("utf-8")
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            if file_ext == ".json":
                try:
                    data = json.loads(content)
                    formatted_text = json.dumps(data, indent=2)
                except json.JSONDecodeError:
                    formatted_text = content
            else:
                formatted_text = content
            
            return [
                Document(
                    page_content=formatted_text,
                    metadata={
                        "source_document": source_name,
                        "page": 0,
                        "file_type": file_ext
                    }
                )
            ]
        except Exception as e:
            print(f"Error loading {file_ext} file: {str(e)}")
            raise ValueError(f"Failed to load {file_ext} file: {str(e)}")

    raise ValueError(
        f"Unsupported file type for ingestion: {file_ext}"
    )


# ---------------- INGESTION ----------------

def data_ingestion(
        base_dir: str,
        file_paths: list = None,
        collection_name: str = 'legal_documents',
        file_bytes: bytes = None,
        filename: str = None
):
    embeddings, llm = get_models()

    all_chunks = []

    # --------------------------------
    # HYBRID CHUNKING
    # --------------------------------

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    # --------------------------------
    # GLOBAL STRUCTURES
    # --------------------------------

    reference_map = {}

    chunk_registry = {}

    chunk_id_counter = 0



    loop_paths = [filename] if file_bytes is not None else file_paths
    if not loop_paths:
        return False

    for file_path in loop_paths:

        if file_bytes is None:
            if not os.path.exists(file_path):
                continue

        try:
            pages = _load_file_pages(
                file_path=file_path if file_bytes is None else None,
                file_bytes=file_bytes,
                filename=file_path if file_bytes is not None else None,
                llm=llm
            )
        except Exception as e:
            print(
                f"Skipping unsupported file {file_path}: {str(e)}"
            )
            continue

        for page_num, p in enumerate(pages):

            # --------------------------------
            # STEP 1: RECURSIVE SPLITTING
            # --------------------------------

            initial_chunks = recursive_splitter.split_text(
                p.page_content
            )

            # --------------------------------
            # STEP 2: SEMANTIC SPLITTING
            # --------------------------------

            semantic_docs = semantic_splitter.create_documents(
                initial_chunks
            )

            chunks = [
                d.page_content
                for d in semantic_docs
            ]

            prev_chunk_id = None

            for c in chunks:

                try:

                    chunk_id = f"chunk_{chunk_id_counter}"

                    chunk_id_counter += 1

                    # --------------------------------
                    # BASE METADATA
                    # --------------------------------

                    meta = extract_metadata(
                        c,
                        file_path
                    )

                    # --------------------------------
                    # BUILD METADATA
                    # --------------------------------

                    meta.update({
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "prev_chunk_id": prev_chunk_id,
                        "next_chunk_id": None
                    })

                    # --------------------------------
                    # LINK CHUNKS
                    # --------------------------------

                    if (
                            prev_chunk_id
                            and prev_chunk_id in chunk_registry
                    ):
                        chunk_registry[
                            prev_chunk_id
                        ].metadata[
                            "next_chunk_id"
                        ] = chunk_id

                    doc = Document(
                        page_content=c,
                        metadata=meta
                    )

                    all_chunks.append(doc)

                    chunk_registry[chunk_id] = doc

                    prev_chunk_id = chunk_id

                except Exception as e:

                    print(
                        f"Error processing chunk "
                        f"in {file_path}: {str(e)}"
                    )

    # --------------------------------
    # STORE VECTOR DB
    # --------------------------------

    if not all_chunks:
        return False

    vectorstore = get_vectorstore(
        embeddings,
        None,
        collection_name,
        _docs=all_chunks
    )

    return vectorstore is not None




def ingest_pdf_and_return_json_sync(
        base_dir: str,
        file_path: str = None,
        collection_name: str = "legal_documents",
        preview_limit: int = 5,
        file_bytes: bytes = None,
        filename: str = None
):
    if file_bytes is not None and filename is not None:
        file_ext = os.path.splitext(filename)[1].lower()
        source_name = filename
    else:
        file_ext = os.path.splitext(file_path)[1].lower()
        source_name = os.path.basename(file_path)

    allowed_extensions = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_DATA_EXTENSIONS
    if file_ext not in allowed_extensions:
        return {
            "ingestion_success": False,
            "error": f"Only PDF, image, or data files ({', '.join(allowed_extensions)}) are supported in this endpoint."
        }

    success = data_ingestion(
        base_dir=base_dir,
        file_paths=[file_path] if file_path else None,
        collection_name=collection_name,
        file_bytes=file_bytes,
        filename=filename
    )

    if not success:
        return {
            "ingestion_success": False,
            "error": "Ingestion failed for the PDF file."
        }

    collection = get_mongo_collection()

    query_filter = {
        "namespace": collection_name,
        "metadata.source_document": source_name
    }

    total_chunks = collection.count_documents(query_filter)

    preview_docs = list(
        collection.find(
            query_filter,
            {
                "_id": 0,
                "page_content": 1,
                "metadata": 1,
                "embedding": 1
            }
        ).limit(max(1, preview_limit))
    )

    chunk_preview = []

    for item in preview_docs:
        metadata = item.get("metadata", {})
        page_text = item.get("page_content", "")
        emb = item.get("embedding", [])

        chunk_preview.append({
            "chunk_id": metadata.get("chunk_id"),
            "source_document": metadata.get("source_document"),
            "page": metadata.get("page"),
            "parent_article": metadata.get("parent_article"),
            "text_preview": page_text[:350],
            "embedding_dimensions": len(emb),
            "metadata": metadata
        })

    return {
        "ingestion_success": True,
        "collection_name": collection_name,
        "source_document": source_name,
        "total_chunks_stored": total_chunks,
        "preview_chunks": chunk_preview
    }


async def ingest_pdf_and_return_json_async(
        base_dir: str,
        file_path: str = None,
        collection_name: str = "legal_documents",
        preview_limit: int = 5,
        file_bytes: bytes = None,
        filename: str = None
):
    return await asyncio.to_thread(
        ingest_pdf_and_return_json_sync,
        base_dir,
        file_path,
        collection_name,
        preview_limit,
        file_bytes,
        filename
    )


# ---------------- DYNAMIC PDF & IMAGE JSON EXTRACTION ----------------

def extract_dynamic_kv_from_pdf_sync(file_path: str = None, file_bytes: bytes = None, filename: str = None):
    import json
    try:
        if file_bytes is not None and filename is not None:
            source_name = filename
            pwd = ""
        else:
            source_name = os.path.basename(file_path)
            pwd = os.path.dirname(os.path.realpath(file_path))

        collection = get_mongo_collection()
        
        # 1. Check if the document already exists in MongoDB
        query_filter = {
            "namespace": "legal_documents",
            "metadata.source_document": source_name
        }
        exists = collection.find_one(query_filter)
        
        if not exists:
            # Dynamically ingest the document (supports both PDFs and Images) to populate the database
            data_ingestion(
                base_dir=pwd,
                file_paths=[file_path] if file_path else None,
                collection_name="legal_documents",
                file_bytes=file_bytes,
                filename=filename
            )
            
        # 2. Retrieve all stored chunk texts dynamically from MongoDB
        docs = list(collection.find(query_filter).sort("metadata.chunk_id", 1))
        full_text = "\n".join([doc.get("page_content", "") for doc in docs]).strip()

        # If DB is empty OR stale (all-blank chunks from before OCR fix), fall back to direct load
        if not full_text:
            if exists:
                collection.delete_many(query_filter)
            _, llm = get_models()
            pages = _load_file_pages(
                file_path=file_path,
                file_bytes=file_bytes,
                filename=filename,
                llm=llm
            )
            full_text = "\n".join([p.page_content for p in pages if p.page_content]).strip()

        if not full_text:
            # Nothing could be extracted — return an informative error instead of all-null JSON
            return {"error": f"No text could be extracted from '{source_name}'. The file may be corrupted or empty."}
            
        # 3. Use Groq Cloud Llama 3.3 to dynamically extract all key-value pairs as a flat JSON object
        _, llm = get_models()
        
        prompt = PromptTemplate.from_template(DYNAMIC_EXTRACTION_PROMPT)
        chain = prompt | llm
        
        # Use full text to perform complete dynamic extraction
        response = chain.invoke({"text": full_text[:12000]})
        
        # Extract string content from response
        text_response = response.content if hasattr(response, "content") else str(response)
        
        cleaned = text_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            try:
                # Try parsing as JSON lines (multiple JSON objects separated by newlines)
                result = []
                for line in cleaned.strip().split('\n'):
                    line = line.strip()
                    if line:
                        result.append(json.loads(line))
                if result:
                    return result
            except json.JSONDecodeError:
                pass
            return {"raw_extracted_text": text_response}
    except Exception as e:
        print("Failed to dynamically extract JSON:", str(e))
        return {"error": str(e)}


async def extract_dynamic_kv_from_pdf_async(file_path: str = None, file_bytes: bytes = None, filename: str = None):
    return await asyncio.to_thread(
        extract_dynamic_kv_from_pdf_sync,
        file_path,
        file_bytes,
        filename
    )

