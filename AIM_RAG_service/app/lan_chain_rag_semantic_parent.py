
import asyncio
import os
import re
import base64
import mimetypes
import io
import logging
import pypdf
# Fillable/DynamicPDF AcroForms often emit noisy "Multiple definitions... /Q" warnings.
logging.getLogger("pypdf").setLevel(logging.ERROR)
try:
    import pypdfium2          # native binary — works locally, may fail on Vercel
    _PYPDFIUM2_AVAILABLE = True
except Exception:
    pypdfium2 = None          # Tier 3 rendering disabled; Tiers 1/2/2.5 still work
    _PYPDFIUM2_AVAILABLE = False
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
SUPPORTED_WORD_EXTENSIONS = {".docx", ".doc"}
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

from app.embedding_client import (
    get_models,
    get_anthropic_llm,
    get_vision_llm,
    get_vision_model_names,
    ANTHROPIC_LLM_MODEL,
)
from app.mongo_client import get_mongo_collection, _to_python_types
from app.rag_retrieval import get_vectorstore
from app.prompt import DYNAMIC_EXTRACTION_PROMPT
from app.word_extractor import extract_text_from_word_bytes, load_word_text_and_images


# ---------------- Colored terminal checkpoints (pdf_dynamic_extract) ----------------
_CKPT_RESET = "\033[0m"
_CKPT_BOLD = "\033[1m"
_CKPT_COLORS = {
    "start": "\033[96m",   # cyan
    "ok": "\033[92m",      # green
    "warn": "\033[93m",    # yellow
    "llm": "\033[95m",     # magenta
    "vision": "\033[94m",  # blue
    "fail": "\033[91m",    # red
    "info": "\033[96m",
}
_CKPT_ANSI_READY = False


def _enable_ckpt_ansi():
    """Enable VT100 colors on Windows consoles (no-op elsewhere)."""
    global _CKPT_ANSI_READY
    if _CKPT_ANSI_READY:
        return
    _CKPT_ANSI_READY = True
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def pdf_extract_ckpt(fn: str, msg: str, level: str = "info"):
    """Print a colored checkpoint line for pdf_dynamic_extract flow."""
    _enable_ckpt_ansi()
    color = _CKPT_COLORS.get(level, _CKPT_COLORS["info"])
    print(f"{_CKPT_BOLD}{color}[pdf_extract] {fn} — {msg}{_CKPT_RESET}", flush=True)


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


def _normalize_image_bytes_for_vision(image_bytes: bytes, filename: str = None):
    """
    Convert raw/embedded PDF image bytes into a vision-safe PNG.
    pypdf often returns raw streams that are not valid JPEG/PNG files.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")

    # Already a normal JPEG/PNG — keep as-is when Pillow can open it.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "pillow is required to do image extraction. Install with: pip install pillow"
        ) from exc

    bio = io.BytesIO(image_bytes)
    try:
        with Image.open(bio) as img:
            img.load()
            # Claude vision accepts common RGB/RGBA PNG reliably.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue(), "image/png", (filename or "image.png")
    except Exception:
        # Last chance: if bytes already look like JPEG/PNG, pass through.
        if image_bytes.startswith(b"\xff\xd8\xff"):
            name = filename or "image.jpg"
            if not name.lower().endswith((".jpg", ".jpeg")):
                name = f"{os.path.splitext(name)[0]}.jpg"
            return image_bytes, "image/jpeg", name
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            name = filename or "image.png"
            if not name.lower().endswith(".png"):
                name = f"{os.path.splitext(name)[0]}.png"
            return image_bytes, "image/png", name
        raise ValueError("Could not decode image bytes into a valid image")


def _extract_text_from_image(image_path: str = None, image_bytes: bytes = None, filename: str = None, llm = None) -> str:
    source_name = filename or (os.path.basename(image_path) if image_path else "image.jpg")
    pdf_extract_ckpt("_extract_text_from_image()", f"start — {source_name}", "vision")
    try:
        if image_bytes is not None:
            raw_bytes = image_bytes
        else:
            with open(image_path, "rb") as image_file:
                raw_bytes = image_file.read()

        try:
            normalized_bytes, mime_type, source_name = _normalize_image_bytes_for_vision(
                raw_bytes,
                filename=source_name,
            )
        except Exception as norm_err:
            print(f"Image normalize skipped/failed for {source_name}: {norm_err}")
            # Avoid sending clearly invalid bytes to Claude vision.
            raise ValueError(f"invalid image data: {norm_err}") from norm_err

        image_data = base64.b64encode(normalized_bytes).decode("utf-8")

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

        # Vision OCR: Anthropic Claude Sonnet only (Groq removed).
        last_error = None
        response = None
        used_provider = None
        for model_name in get_vision_model_names():
            try:
                vision_llm = get_vision_llm(model_name)
                response = vision_llm.invoke([message])
                used_provider = f"Claude:{model_name}"
                break
            except Exception as model_error:
                last_error = model_error
                print(
                    f"Image OCR failed with Claude vision model '{model_name}': "
                    f"{model_error}"
                )

        if response is None:
            raise last_error or RuntimeError(
                "No Claude vision model could extract image text."
            )

        extracted = response.content if hasattr(response, "content") else str(response)
        extracted = (extracted or "").strip()

        if not extracted or extracted.upper() == "NO_TEXT_FOUND":
            print(
                f"[pdf_extract] _extract_text_from_image() done — no text in "
                f"{source_name} (provider={used_provider})"
            )
            return (
                f"No readable text found in image {source_name}"
            )

        print(
            f"[pdf_extract] _extract_text_from_image() done — "
            f"{source_name}, chars={len(extracted)}, provider={used_provider}"
        )
        return extracted

    except Exception as e:
        print(f"[pdf_extract] _extract_text_from_image() FAILED — {source_name}: {e}")
        return f"Image file {source_name}"

def _carve_jpeg_from_pdf_bytes(pdf_bytes: bytes) -> list:
    """Pure Python JPEG byte carving from raw PDF bytes.
    Phone-photo PDFs embed the camera JPEG directly in the file content stream.
    pypdf's page.images only finds XObject images; this finds ALL JPEGs including
    inline image streams, which is the common format for mobile-app PDFs.
    No native binary needed — works on Vercel, Railway, everywhere.
    """
    images = []
    JPEG_SOI = b'\xff\xd8\xff'   # JPEG Start Of Image marker
    JPEG_EOI = b'\xff\xd9'       # JPEG End Of Image marker
    MIN_JPEG_SIZE = 10_000       # real photos are > 10 KB

    pos = 0
    while pos < len(pdf_bytes) - 3:
        start = pdf_bytes.find(JPEG_SOI, pos)
        if start == -1:
            break
        end = pdf_bytes.find(JPEG_EOI, start + 3)
        if end == -1:
            break
        jpeg_data = pdf_bytes[start: end + 2]  # include EOI marker
        if len(jpeg_data) >= MIN_JPEG_SIZE:
            images.append(jpeg_data)
        pos = end + 2

    return images


def _extract_acroform_fields_text(reader) -> str:
    """Read fillable AcroForm field values from a PDF.

    BOLs generated by tools like DynamicPDF/Nitro often store shipper, consignee,
    BOL #, weights, etc. only in form field /V values — not in the page text layer.
    Returns a labeled text block for the LLM, or "" when no filled fields exist.
    """
    try:
        fields = reader.get_fields()
    except Exception as form_err:
        print(f"AcroForm get_fields failed: {form_err}")
        return ""

    if not fields:
        return ""

    lines = []
    for name, info in fields.items():
        if not name:
            continue
        value = None
        if isinstance(info, dict):
            value = info.get("/V")
        else:
            value = getattr(info, "value", None)
            if value is None and isinstance(info, str):
                value = info

        if value is None:
            continue

        # Checkbox / radio style values
        if hasattr(value, "get_object"):
            try:
                value = value.get_object()
            except Exception:
                pass

        text_val = str(value).strip()
        if not text_val:
            continue

        # Skip empty placeholder markers
        if text_val in {"/", "Off", "Yes", "No"} and str(name).lower().endswith(
            ("check", "chk", "box", "flag")
        ):
            continue

        field_name = str(name).strip()
        # Skip pure UI label fields that only echo another label (e.g. "SCAC :")
        if field_name.lower().endswith("label") and text_val.endswith(":"):
            continue

        lines.append(f"{field_name}: {text_val}")

    if not lines:
        return ""

    return "=== PDF FORM FIELD VALUES ===\n" + "\n".join(lines)


def _load_file_pages(file_path: str = None, file_bytes: bytes = None, filename: str = None, llm = None):
    if file_path:
        file_ext = os.path.splitext(file_path)[1].lower()
        source_name = os.path.basename(file_path)
    elif filename:
        file_ext = os.path.splitext(filename)[1].lower()
        source_name = filename
    else:
        raise ValueError("Either file_path or filename must be provided.")

    print(f"[pdf_extract] _load_file_pages() start — {source_name} ({file_ext})")

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
        print(
            f"[pdf_extract] Tier1 pypdf text done — pages={len(page_texts)}, "
            f"chars={sum(len(t) for t in page_texts)}"
        )

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
                        print(
                            f"[pdf_extract] Tier2 embedded-image OCR done — "
                            f"page={i}, chars={len(tier2_ocr_text)}"
                        )
            except Exception as emb_err:
                print(f"[pdf_extract] Tier2 embedded image FAILED page {i}: {emb_err}")

            # --- TIER 2.5: JPEG byte carving (pure Python, Vercel-safe) ---
            # Handles phone-photo PDFs that use inline image streams instead of
            # XObject images. Scans raw PDF bytes for JPEG SOI/EOI markers.
            if not tier2_ocr_text:
                try:
                    carved_jpegs = _carve_jpeg_from_pdf_bytes(file_bytes_local)
                    if carved_jpegs:
                        carve_ocr_parts = []
                        for idx, jpeg_data in enumerate(carved_jpegs):
                            ocr_text = _extract_text_from_image(
                                image_bytes=jpeg_data,
                                filename=f"{source_name}_carved{idx}.jpg",
                                llm=llm
                            )
                            if ocr_text and "NO_TEXT_FOUND" not in ocr_text.upper():
                                carve_ocr_parts.append(ocr_text)
                        if carve_ocr_parts:
                            tier2_ocr_text = "\n".join(carve_ocr_parts)
                            print(
                                f"[pdf_extract] Tier2.5 JPEG carve OCR done — "
                                f"page={i}, images={len(carve_ocr_parts)}"
                            )
                except Exception as carve_err:
                    print(f"[pdf_extract] Tier2.5 JPEG carve FAILED page {i}: {carve_err}")

            if tier2_ocr_text:
                # Do NOT replace a rich text layer with OCR-only (OCR can be longer but
                # messier and causes Claude to null most JSON fields).
                # Rich text → keep text layer + append OCR (logo/brand from images).
                # Sparse text (scanned) → OCR primary.
                real_words = [w for w in text_layer.split() if len(w) > 1]
                if len(real_words) >= 30 and text_layer:
                    page_texts[i] = (
                        text_layer
                        + "\n\n=== EMBEDDED IMAGE OCR ===\n"
                        + tier2_ocr_text.strip()
                    )
                    print(
                        f"[pdf_extract] Tier2 merge text+OCR — page={i}, "
                        f"text_chars={len(text_layer)}, ocr_chars={len(tier2_ocr_text)}"
                    )
                elif len(tier2_ocr_text.strip()) >= len(text_layer):
                    page_texts[i] = tier2_ocr_text
                    print(
                        f"[pdf_extract] Tier2 OCR primary (sparse text layer) — page={i}"
                    )
                # else: text layer already richer and sparse OCR — keep text_layer
                continue  # page resolved, skip Tier 3

            # --- TIER 3: pypdfium2 page rendering (local fallback, may fail on Vercel) ---
            # Only reached when page has NO embedded images (not a photo PDF).
            # Useful for PDFs where content is drawn as vector/path graphics.
            # Trigger only if pypdfium2 loaded AND text layer is sparse (< 30 real words).
            real_words = [w for w in text_layer.split() if len(w) > 1]
            if _PYPDFIUM2_AVAILABLE and len(real_words) < 30:
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
                    if ocr_text and "NO_TEXT_FOUND" not in (ocr_text or "").upper():
                        if text_layer and len(real_words) >= 10:
                            page_texts[i] = (
                                text_layer
                                + "\n\n=== PAGE RENDER OCR ===\n"
                                + ocr_text.strip()
                            )
                        elif len(ocr_text.strip()) > len(text_layer):
                            page_texts[i] = ocr_text
                        print(f"[pdf_extract] Tier3 pypdfium2 OCR done — page={i}")
                    pdf_doc.close()
                except Exception as render_err:
                    print(f"[pdf_extract] Tier3 pypdfium2 FAILED page {i}: {render_err}")

        # --- AcroForm fields (fillable PDF values) ---
        # Some BOLs keep shipper/consignee/BOL/weight only in form /V values.
        # Merge those into page 0 so the LLM sees them; no-op for normal text PDFs.
        try:
            form_text = _extract_acroform_fields_text(reader)
            if form_text:
                if page_texts:
                    base = (page_texts[0] or "").strip()
                    page_texts[0] = (base + "\n\n" + form_text).strip() if base else form_text
                else:
                    page_texts = [form_text]
                print(f"[pdf_extract] AcroForm fields merged — {source_name}")
        except Exception as form_merge_err:
            print(f"[pdf_extract] AcroForm merge FAILED — {source_name}: {form_merge_err}")

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
        print(
            f"[pdf_extract] _load_file_pages() done — PDF {source_name}, "
            f"pages={len(docs)}, chars={sum(len(d.page_content or '') for d in docs)}"
        )
        return docs


    if file_ext in SUPPORTED_WORD_EXTENSIONS:
        if file_bytes is None:
            with open(file_path, "rb") as f:
                file_bytes_local = f.read()
        else:
            file_bytes_local = file_bytes

        word_text, word_images = load_word_text_and_images(
            file_bytes=file_bytes_local,
            filename=source_name,
        )
        text_layer = (word_text or "").strip()
        real_words = [w for w in text_layer.split() if len(w) > 1]

        # Pasted order screenshots live in word/media — OCR when body text is empty/sparse.
        ocr_text = ""
        if word_images and (len(real_words) < 30 or not text_layer):
            page_ocr_parts = []
            # Cap to avoid extreme multi-icon docs; largest images first.
            for idx, (img_name, img_data) in enumerate(word_images[:5]):
                try:
                    part = _extract_text_from_image(
                        image_bytes=img_data,
                        filename=f"{source_name}_{img_name or f'img{idx}'}",
                        llm=llm,
                    )
                    if part and "NO_TEXT_FOUND" not in part.upper():
                        page_ocr_parts.append(part)
                except Exception as img_err:
                    print(f"Word embedded image OCR error {source_name} img {idx}: {img_err}")
            if page_ocr_parts:
                ocr_text = "\n".join(page_ocr_parts).strip()

        if ocr_text and len(ocr_text) >= len(text_layer):
            final_text = ocr_text
        else:
            final_text = text_layer

        if not final_text:
            raise ValueError(
                f"No readable text found in '{source_name}'. "
                "If the Word file has a pasted order image, ensure the image is embedded "
                "and vision OCR is configured; or export to PDF/image and re-upload."
            )

        print(
            f"[pdf_extract] _load_file_pages() done — Word {source_name}, "
            f"chars={len(final_text)}"
        )
        return [
            Document(
                page_content=final_text,
                metadata={
                    "source_document": source_name,
                    "page": 0,
                    "file_type": file_ext,
                },
            )
        ]

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
        print(
            f"[pdf_extract] _load_file_pages() done — Image {source_name}, "
            f"chars={len(image_text or '')}"
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
            
            # Check if text content can be parsed as JSON
            is_json = False
            try:
                data = json.loads(content.strip())
                is_json = True
            except json.JSONDecodeError:
                pass
                
            if is_json:
                records = []
                # Extract records from data (handle lists, details arrays, or serialized nested strings)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "details" in item:
                            details_val = item["details"]
                            if isinstance(details_val, str):
                                try:
                                    details_val = json.loads(details_val)
                                except Exception:
                                    pass
                            if isinstance(details_val, list):
                                records.extend(details_val)
                            else:
                                records.append(item)
                        else:
                            records.append(item)
                elif isinstance(data, dict):
                    if "details" in data:
                        details_val = data["details"]
                        if isinstance(details_val, str):
                            try:
                                details_val = json.loads(details_val)
                            except Exception:
                                pass
                        if isinstance(details_val, list):
                            records.extend(details_val)
                        else:
                            records.append(data)
                    else:
                        records.append(data)
                else:
                    records.append(data)
                
                # Create a separate Document for each record to prevent fragmenting fields during chunking
                docs = []
                for idx, record in enumerate(records):
                    record_text = json.dumps(record, indent=2)
                    metadata = {
                        "source_document": source_name,
                        "page": idx,
                        "file_type": file_ext,
                        "no_split": True
                    }
                    if isinstance(record, dict):
                        if "ordernumber" in record:
                            metadata["ordernumber"] = record["ordernumber"]
                        if "orderid" in record:
                            metadata["orderid"] = record["orderid"]
                    docs.append(
                        Document(
                            page_content=record_text,
                            metadata=metadata
                        )
                    )
                return docs
            else:
                # Fallback to loading the file as a single plain text Document
                return [
                    Document(
                        page_content=content,
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
        filename: str = None,
        mongo_collection_name: str = None,
        replace_namespace: bool = True,
        extra_metadata: dict = None
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

            # Skip splitting if the Document is flagged as no_split (e.g. structured JSON records)
            if p.metadata.get("no_split"):
                chunks = [p.page_content]
            else:
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

                    if extra_metadata:
                        meta.update(extra_metadata)

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
        _docs=all_chunks,
        mongo_collection_name=mongo_collection_name,
        replace_namespace=replace_namespace
    )

    return vectorstore is not None




def ingest_pdf_and_return_json_sync(
        base_dir: str,
        file_path: str = None,
        collection_name: str = "legal_documents",
        preview_limit: int = 5,
        file_bytes: bytes = None,
        filename: str = None,
        mongo_collection_name: str = None,
        replace_namespace: bool = True,
        extra_metadata: dict = None
):
    if file_bytes is not None and filename is not None:
        file_ext = os.path.splitext(filename)[1].lower()
        source_name = filename
    else:
        file_ext = os.path.splitext(file_path)[1].lower()
        source_name = os.path.basename(file_path)

    print(f"[pdf_extract] ingest_pdf_and_return_json_sync() start — {source_name}")

    allowed_extensions = (
        SUPPORTED_PDF_EXTENSIONS
        | SUPPORTED_IMAGE_EXTENSIONS
        | SUPPORTED_WORD_EXTENSIONS
        | SUPPORTED_DATA_EXTENSIONS
    )
    if file_ext not in allowed_extensions:
        return {
            "ingestion_success": False,
            "error": f"Only PDF, image, Word, or data files ({', '.join(sorted(allowed_extensions))}) are supported in this endpoint."
        }

    success = data_ingestion(
        base_dir=base_dir,
        file_paths=[file_path] if file_path else None,
        collection_name=collection_name,
        file_bytes=file_bytes,
        filename=filename,
        mongo_collection_name=mongo_collection_name,
        replace_namespace=replace_namespace,
        extra_metadata=extra_metadata
    )

    if not success:
        print(f"[pdf_extract] ingest FAILED — {source_name}")
        return {
            "ingestion_success": False,
            "error": "Ingestion failed for the PDF file."
        }

    collection = get_mongo_collection(mongo_collection_name)

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

    print(
        f"[pdf_extract] ingest_pdf_and_return_json_sync() done — "
        f"{source_name}, chunks={total_chunks}"
    )
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
        filename: str = None,
        mongo_collection_name: str = None,
        replace_namespace: bool = True,
        extra_metadata: dict = None
):
    return await asyncio.to_thread(
        ingest_pdf_and_return_json_sync,
        base_dir,
        file_path,
        collection_name,
        preview_limit,
        file_bytes,
        filename,
        mongo_collection_name,
        replace_namespace,
        extra_metadata
    )


# ---------------- DYNAMIC PDF & IMAGE JSON EXTRACTION ----------------

def _is_invalid_commodity_value(value) -> bool:
    """True when commodity is clearly package/weight text, not a product description."""
    if value is None:
        return False
    if isinstance(value, list):
        return any(_is_invalid_commodity_value(v) for v in value)
    text = str(value).strip().lower()
    if not text:
        return False
    # pcs / packages / pallets with optional weight — classic false commodity
    if re.search(
        r"\b\d+(\.\d+)?\s*(pcs?|pieces?|pkgs?|packages?|pallets?|skids?|qty)\b",
        text,
    ):
        return True
    if re.search(r"\b\d+([.,]\d+)?\s*(lbs?|lb|kg|kgs|pounds?)\b", text) and not re.search(
        r"[a-zA-Z]{3,}",
        re.sub(r"\b(lbs?|lb|kg|kgs|pounds?|pcs?|pieces?|and|,)\b", " ", text),
    ):
        # mostly numbers + mass units, no real product words
        return True
    if re.search(r"\bpcs?\b", text) and re.search(r"\b(lbs?|lb|kg)\b", text):
        return True
    return False


_SHIPMENT_FIELD_KEYS = (
    "commodity",
    "pickup_location",
    "pickup_date",
    "pickup_time",
    "pickup_refrence_no",
    "distance",
    "delivery_location",
    "delivery_date",
    "delivery_time",
    "delivery_refrence_no",
    "ValueOfgoods",
    "Equipment",
    "No.OfPackage",
    "weight",
    "temperature",
    "dimention",
    "pickupNote",
    "DeliveryNotes",
    "Copmliancehandling",
)

_SHIPMENT_LINE_KEYS = ("commodity", "No.OfPackage", "weight", "dimention")


def _fill_shipment_keys(row: dict) -> dict:
    """Keep every shipment field present; do not drop keys."""
    out = {}
    for key in _SHIPMENT_FIELD_KEYS:
        out[key] = row.get(key, None) if isinstance(row, dict) else None
    return out


def _sanitize_one_shipment(row: dict) -> dict:
    filled = _fill_shipment_keys(row if isinstance(row, dict) else {})
    commodity = filled.get("commodity")
    if isinstance(commodity, list):
        if len(commodity) == 1:
            filled["commodity"] = commodity[0]
        elif len(commodity) == 0:
            filled["commodity"] = None
    if _is_invalid_commodity_value(filled.get("commodity")):
        filled["commodity"] = None
    return filled


def _zip_line_value(value, index: int, *, copy_scalar: bool):
    """Map a field onto shipment row i. Lists zip by index; missing → None."""
    if isinstance(value, list):
        if index < len(value):
            item = value[index]
            return None if item in ("", []) else item
        return None
    if value in (None, "", []):
        return None
    return value if copy_scalar else (value if index == 0 else None)


def _split_shipment_by_commodity_list(shipment: dict) -> list:
    commodities = [c for c in shipment.get("commodity") or [] if c not in (None, "")]
    rows = []
    for i, commodity in enumerate(commodities):
        row = {}
        for key in _SHIPMENT_FIELD_KEYS:
            if key == "commodity":
                row[key] = commodity
            elif key in _SHIPMENT_LINE_KEYS:
                # Per-line lists zip. A single document total (string) is copied to every row.
                row[key] = _zip_line_value(
                    shipment.get(key), i, copy_scalar=True
                )
            else:
                row[key] = shipment.get(key)
        rows.append(_sanitize_one_shipment(row))
    return rows


def _normalize_extracted_shipment(parsed: dict) -> dict:
    """
    1 commodity → shipment object.
    2+ commodities → shipment array of full objects.
    If the model still returns one object with commodity:[...], split it here.
    """
    if not isinstance(parsed, dict):
        return parsed
    shipment = parsed.get("shipment")

    if isinstance(shipment, list):
        if (
            len(shipment) == 1
            and isinstance(shipment[0], dict)
            and isinstance(shipment[0].get("commodity"), list)
            and len([c for c in shipment[0].get("commodity") or [] if c not in (None, "")]) > 1
        ):
            parsed["shipment"] = _split_shipment_by_commodity_list(shipment[0])
            return parsed
        parsed["shipment"] = [
            _sanitize_one_shipment(item)
            for item in shipment
            if isinstance(item, dict)
        ]
        if len(parsed["shipment"]) == 1:
            parsed["shipment"] = parsed["shipment"][0]
        elif not parsed["shipment"]:
            parsed["shipment"] = _fill_shipment_keys({})
        return parsed

    if not isinstance(shipment, dict):
        return parsed

    commodity = shipment.get("commodity")
    if isinstance(commodity, list) and len([c for c in commodity if c not in (None, "")]) > 1:
        parsed["shipment"] = _split_shipment_by_commodity_list(shipment)
        return parsed

    parsed["shipment"] = _sanitize_one_shipment(shipment)
    return parsed


def _sanitize_extracted_commodity(parsed):
    """Null invalid commodity; split multi-commodity shipment arrays."""
    if not isinstance(parsed, dict):
        return parsed
    parsed = _normalize_extracted_shipment(parsed)
    if "commodity" in parsed and _is_invalid_commodity_value(parsed.get("commodity")):
        parsed["commodity"] = None
    return parsed


def _parse_llm_json_response(text_response: str):
    """
    Parse LLM output into a real JSON object (not a string with \\n escapes).
    Prefers the full extraction dict; never returns a nested scalar array
    like ["0.000","0.000"] from shipment.weight when the outer object truncates.
    """
    cleaned = (text_response or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Strip a leading/trailing quoted JSON blob
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        try:
            maybe = json.loads(cleaned)
            if isinstance(maybe, str):
                cleaned = maybe.strip()
        except json.JSONDecodeError:
            pass

    def _looks_like_extract(obj) -> bool:
        if isinstance(obj, dict):
            return any(
                k in obj
                for k in ("customerinfo", "shipment", "Revenue", "raw_extracted_text", "error")
            )
        return False

    def _try_load(candidate: str):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed.strip())
            except json.JSONDecodeError:
                return None
        return parsed

    # 1) Prefer full text / outermost object {...}
    dict_candidates = [cleaned]
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        snippet = cleaned[obj_start : obj_end + 1].strip()
        if snippet not in dict_candidates:
            dict_candidates.append(snippet)

    for candidate in dict_candidates:
        parsed = _try_load(candidate)
        if isinstance(parsed, dict):
            return _sanitize_extracted_commodity(parsed)

    # 2) Only accept top-level arrays that look like a list of extract objects
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        parsed = _try_load(cleaned[arr_start : arr_end + 1].strip())
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            if any(_looks_like_extract(x) for x in parsed):
                return [_sanitize_extracted_commodity(x) for x in parsed]
            # list of dicts but not our schema — still return first dict if keys look useful
            return parsed

    # 3) NDJSON lines (dict lines only)
    try:
        result = []
        for line in cleaned.split("\n"):
            line = line.strip().rstrip(",")
            if not line or not line.startswith("{"):
                continue
            parsed = _try_load(line)
            if isinstance(parsed, dict):
                result.append(_sanitize_extracted_commodity(parsed))
        if result:
            return result if len(result) > 1 else result[0]
    except Exception:
        pass

    return {"raw_extracted_text": text_response}


def extract_dynamic_kv_from_pdf_sync(file_path: str = None, file_bytes: bytes = None, filename: str = None):
    """Extract key-value pairs from a PDF or image as JSON.
    Reads text DIRECTLY from the file — does NOT trigger embedding or MongoDB.
    Embedding/storage is handled separately by the background task.
    This makes the function fast and Vercel-compatible.
    """
    try:
        if file_bytes is not None and filename is not None:
            source_name = filename
        else:
            source_name = os.path.basename(file_path)

        print(f"[pdf_extract] extract_dynamic_kv_from_pdf_sync() start — {source_name}")

        _, llm = get_models()
        pdf_extract_ckpt("get_models()", "ready", "ok")
        pages = _load_file_pages(
            file_path=file_path,
            file_bytes=file_bytes,
            filename=filename,
            llm=llm
        )
        full_text = "\n".join([p.page_content for p in pages if p.page_content]).strip()
        print(
            f"[pdf_extract] text load done — pages={len(pages)}, chars={len(full_text)}"
        )

        if not full_text:
            print(f"[pdf_extract] extract STOPPED — no text in {source_name}")
            return {"error": f"No text could be extracted from '{source_name}'. The file may be corrupted or empty."}

        # JSON extract: Claude via LLM_MODEL (Groq text path not used). Flow unchanged.
        anthropic_llm = get_anthropic_llm()
        prompt = PromptTemplate.from_template(DYNAMIC_EXTRACTION_PROMPT)
        extract_vars = {"text": full_text[:12000]}


        try:
            print(
                f"[pdf_extract] Claude JSON extract start — "
                f"model={ANTHROPIC_LLM_MODEL}, doc_chars={len(extract_vars['text'])}"
            )
            try:
                llm_bound = anthropic_llm.bind(max_tokens=4096)
            except Exception:
                llm_bound = anthropic_llm
            response = (prompt | llm_bound).invoke(extract_vars)
            print(
                f"[pdf_extract] Claude JSON extract done — "
                f"model={ANTHROPIC_LLM_MODEL}"
            )
        except Exception as claude_exc:
            print(f"[pdf_extract] Claude JSON extract FAILED — {claude_exc}")
            raise

        text_response = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_llm_json_response(text_response)
        print(f"[pdf_extract] _parse_llm_json_response() done — {source_name}")
        print(f"[pdf_extract] extract_dynamic_kv_from_pdf_sync() complete — {source_name}")
        return parsed
    except Exception as e:
        print(f"[pdf_extract] extract_dynamic_kv_from_pdf_sync() FAILED — {e}")
        return {"error": str(e)}


async def extract_dynamic_kv_from_pdf_async(file_path: str = None, file_bytes: bytes = None, filename: str = None):
    print("[pdf_extract] extract_dynamic_kv_from_pdf_async() → thread")
    return await asyncio.to_thread(
        extract_dynamic_kv_from_pdf_sync,
        file_path,
        file_bytes,
        filename
    )
    pdf_extract_ckpt("extract_dynamic_kv_from_pdf_async()", "thread done", "ok")
    return result

