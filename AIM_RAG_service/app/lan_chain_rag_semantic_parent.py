
import asyncio
import os
import re
import base64
import mimetypes
import io
import pypdf
from typing import List
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient

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

MONGO_URI = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME = os.environ.get("DB_NAME", "")
MONGO_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ocr_db")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

_embeddings_cache = None
_llm_cache = None

def get_models():
    global _embeddings_cache, _llm_cache
    if _embeddings_cache is None or _llm_cache is None:
        # Set AWS credentials as environment variables for Bedrock
        os.environ["AWS_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_KEY
        
        # Use AWS Bedrock embeddings
        embeddings = BedrockEmbeddings(
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"dimensions": 1024}
        )
            
        # Optimizing: Use high-speed cloud Groq LLM if API key is provided.
        # This shifts inference to Groq's hardware, reducing generation time to 200-500ms!
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
            except Exception as e:
                print(f"Error initializing ChatGroq, falling back to OllamaLLM: {e}")
                llm = OllamaLLM(model="phi3:mini", format="json")
        else:
            llm = OllamaLLM(model="phi3:mini", format="json")
        
        _embeddings_cache = embeddings
        _llm_cache = llm
        
    return _embeddings_cache, _llm_cache


# ---------------- MONGO VECTOR STORE ----------------

def _to_python_types(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            k: _to_python_types(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _to_python_types(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _to_python_types(v)
            for v in value
        ]

    if isinstance(value, set):
        return [
            _to_python_types(v)
            for v in value
        ]

    return value


def get_mongo_collection():
    if not MONGO_URI or not MONGO_DB_NAME:
        raise ValueError(
            "Mongo config missing. Please set "
            "mongo_db.MONGO_URI and mongo_db.DB_NAME"
        )

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=15000
    )

    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    # Indexes for fast namespace and source-document filtering.
    collection.create_index([("namespace", 1)])
    collection.create_index(
        [("namespace", 1), ("metadata.source_document", 1)]
    )

    return collection


class MongoRetriever:

    def __init__(
            self,
            vectorstore,
            search_kwargs=None
    ):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {}

    def invoke(self, query: str):
        k = self.search_kwargs.get("k", 15)
        fetch_k = self.search_kwargs.get("fetch_k", max(k, 15))

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )


class MongoVectorStore:

    def __init__(
            self,
            collection,
            embeddings,
            namespace
    ):
        self.collection = collection
        self.embeddings = embeddings
        self.namespace = namespace

    def as_retriever(
            self,
            search_type="mmr",
            search_kwargs=None
    ):
        # Current implementation uses cosine similarity + reranking.
        # The search_type argument is accepted for API compatibility.
        _ = search_type

        return MongoRetriever(
            vectorstore=self,
            search_kwargs=search_kwargs
        )

    def add_documents(self, docs):
        payload = []

        for doc in docs:
            metadata = _to_python_types(
                dict(doc.metadata)
            )

            chunk_embedding = metadata.pop(
                "chunk_embedding",
                None
            )

            if chunk_embedding is None:
                chunk_embedding = self.embeddings.embed_query(
                    doc.page_content[:4000]
                )

            chunk_embedding = [
                float(x)
                for x in chunk_embedding
            ]

            payload.append({
                "namespace": self.namespace,
                "page_content": doc.page_content,
                "embedding": chunk_embedding,
                "metadata": metadata
            })

        if payload:
            self.collection.insert_many(payload)

    def similarity_search(
            self,
            query,
            k=15,
            fetch_k=60
    ):
        query_embedding = np.array(
            self.embeddings.embed_query(query)
        )

        if np.linalg.norm(query_embedding) == 0:
            return []

        docs = list(
            self.collection.find(
                {"namespace": self.namespace},
                {
                    "_id": 0,
                    "page_content": 1,
                    "metadata": 1,
                    "embedding": 1
                }
            )
        )

        if not docs:
            return []

        # Vectorized similarity search optimization:
        # Instead of iterating sequentially in a slow Python loop, we construct a 2D matrix
        # and compute all cosine similarities in a single optimized NumPy linear algebra call.
        # This speeds up retrieval matching tremendously as documents increase.
        valid_docs = [doc for doc in docs if doc.get("embedding") is not None]
        if not valid_docs:
            return []

        embeddings_matrix = np.array([doc["embedding"] for doc in valid_docs])

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        doc_norms = np.linalg.norm(embeddings_matrix, axis=1)

        dot_products = np.dot(embeddings_matrix, query_embedding)

        denoms = query_norm * doc_norms
        scores = np.zeros_like(dot_products)
        valid_denoms = denoms > 0
        scores[valid_denoms] = dot_products[valid_denoms] / denoms[valid_denoms]

        scored = list(zip(scores, valid_docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        candidate_limit = max(k, fetch_k)
        top = scored[:candidate_limit]

        return [
            Document(
                page_content=item["page_content"],
                metadata={
                    **item.get("metadata", {}),
                    "embedding": item.get("embedding"),
                    "similarity_score": float(score)
                }
            )
            for score, item in top[:k]
        ]


def get_vectorstore(
        embeddings,
        _unused_url,
        collection_name,
        _docs=None
):
    try:
        collection = get_mongo_collection()

        vectorstore = MongoVectorStore(
            collection=collection,
            embeddings=embeddings,
            namespace=collection_name
        )

        # --------------------------------
        # LOAD EXISTING NAMESPACE
        # --------------------------------

        if _docs is None:
            exists = collection.count_documents(
                {"namespace": collection_name},
                limit=1
            ) > 0

            return vectorstore if exists else None

        # --------------------------------
        # RECREATE NAMESPACE DATA
        # --------------------------------

        collection.delete_many({"namespace": collection_name})

        vectorstore.add_documents(_docs)

        return vectorstore

    except Exception as e:

        print(f"Vector store error: {str(e)}")

        return None


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

        # Initialize a vision-capable LLM for image OCR on Groq
        vision_llm = None
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                vision_llm = ChatGroq(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.0
                )
            except Exception as e:
                print(f"Error initializing vision ChatGroq: {e}")
        
        if vision_llm is None:
            try:
                from langchain_ollama import OllamaLLM
                vision_llm = OllamaLLM(model="llama3.2-vision")
            except Exception:
                vision_llm = llm

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
        if file_bytes is not None:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
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
        else:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                if "source_document" not in doc.metadata:
                    doc.metadata["source_document"] = os.path.basename(doc.metadata.get("source", file_path))
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

    # --------------------------------
    # REFERENCE EXTRACTION
    # --------------------------------

    def extract_legal_references(text: str):

        refs = []

        eu_patterns = [
            r'Regulation\s*\(EU\)\s*\d+/\d+',
            r'Directive\s*\(EU\)\s*\d+/\d+',
            r'Article\s+\d+\s+of\s+Regulation\s*\(EU\)\s*\d+/\d+'
        ]

        us_patterns = [
            r'\b\d+\s+U\.S\.C\.?\s*§+\s*\d+[a-zA-Z\-]*',
            r'Section\s+\d+\s+of\s+the\s+[A-Za-z\s]+Act',
            r'\b\d+\s+CFR\s+\d+'
        ]

        for pattern in eu_patterns + us_patterns:
            refs.extend(
                re.findall(
                    pattern,
                    text,
                    re.IGNORECASE
                )
            )

        return list(set(refs))

    # --------------------------------
    # NORMALIZATION
    # --------------------------------

    def normalize_reference(ref: str):

        ref_lower = ref.lower()

        if "2016/679" in ref_lower or "gdpr" in ref_lower:
            return "EU_GDPR_2016_679"

        if "2022/1925" in ref_lower:
            return "EU_DMA_2022_1925"

        if "15 u.s.c" in ref_lower and "45" in ref_lower:
            return "US_15_USC_45"

        return re.sub(
            r'\W+',
            '_',
            ref.strip().upper()
        )

    # --------------------------------
    # REGISTER REFERENCES
    # --------------------------------

    def register_reference(ref, chunk_id):

        norm = normalize_reference(ref)

        if norm not in reference_map:
            reference_map[norm] = {
                "aliases": set(),
                "jurisdiction":
                    "EU" if "EU" in ref
                    else "US" if "U.S." in ref
                    else "Unknown",
                "type":
                    "Regulation"
                    if "Regulation" in ref
                    else "Statute",
                "chunks": set()
            }

        reference_map[norm]["aliases"].add(ref)

        reference_map[norm]["chunks"].add(chunk_id)

        return norm

    # --------------------------------
    # INGEST FILES
    # --------------------------------

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

    allowed_extensions = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
    if file_ext not in allowed_extensions:
        return {
            "ingestion_success": False,
            "error": f"Only PDF or image files ({', '.join(allowed_extensions)}) are supported in this endpoint."
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
        
        if not docs:
            # Fallback to direct file loading using our unified load pages method if DB is empty
            _, llm = get_models()
            pages = _load_file_pages(
                file_path=file_path,
                file_bytes=file_bytes,
                filename=filename,
                llm=llm
            )
            full_text = "\n".join([p.page_content for p in pages if p.page_content])
        else:
            full_text = "\n".join([doc.get("page_content", "") for doc in docs])
            
        # 3. Use Groq Cloud Llama 3.3 to dynamically extract all key-value pairs as a flat JSON object
        _, llm = get_models()
        
        prompt = PromptTemplate.from_template("""
        You are an expert dynamic data extractor. Analyze the document text and extract all meaningful key-value pairs as a flat JSON object.
        Look for key document metadata such as:
        - "carrier_name" or "company_name"
        - "total_amount" or "settled_amount"
        - "invoice_or_load_number"
        - "date"
        - "phone_or_email"
        - "stops" (if any stop/pickup/delivery list exists)
        - "tax_details"

        CRITICAL RULES:-
        - If any field is not found, return null for that field.
        - Dynamically create key-value pairs for any other relevant information found in the text.
        - DO NOT extract, include, or generate any "additional_notes" key in the JSON object under any circumstances.
        - Do Not Hallucinate.
        Return ONLY a valid JSON object. Do not include markdown formatting like ```json or any conversational text.
        TEXT:
        {text}
        """)
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

