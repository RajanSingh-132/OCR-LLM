import os
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

_embeddings_cache = None
_llm_cache = None
_vision_llm_cache = {}
_anthropic_llm_cache = None

# Read env vars
# Text extract LLM is Claude; Groq is used only for vision OCR.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")  # text LLM disabled
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GROQ_VISION_FALLBACK_MODELS = os.environ.get("GROQ_VISION_FALLBACK_MODELS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Used by /api/v1/orders/ask and /api/v1/upload/pdf_dynamic_extract (text LLM)
ANTHROPIC_LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")


def get_models():
    """Return (embeddings, llm). Embeddings=Bedrock; LLM=Anthropic Claude (text only)."""
    global _embeddings_cache, _llm_cache
    if _embeddings_cache is None or _llm_cache is None:
        if BEDROCK_ACCESS_KEY:
            os.environ["AWS_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY
        if BEDROCK_SECRET_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_KEY

        embeddings = BedrockEmbeddings(
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"dimensions": 1024},
        )

        # --- Groq text LLM (disabled: Claude used for extract/ask) ---
        # if not GROQ_API_KEY:
        #     raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
        # groq_kwargs = {
        #     "model": GROQ_LLM_MODEL,
        #     "groq_api_key": GROQ_API_KEY,
        #     "temperature": 0.0,
        #     "max_tokens": int(os.environ.get("GROQ_LLM_MAX_TOKENS", "2500")),
        # }
        # if "gpt-oss" in (GROQ_LLM_MODEL or "").lower():
        #     groq_kwargs["reasoning_effort"] = os.environ.get(
        #         "GROQ_REASONING_EFFORT", "low"
        #     )
        # llm = ChatGroq(**groq_kwargs)

        llm = get_anthropic_llm()

        _embeddings_cache = embeddings
        _llm_cache = llm
        print("[pdf_extract] get_models() ready — Bedrock embeddings + Claude text LLM")

    return _embeddings_cache, _llm_cache


def get_anthropic_llm():
    """Anthropic Claude — PDF text extract + /orders/ask."""
    global _anthropic_llm_cache
    if _anthropic_llm_cache is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it in .env for Claude extract/ask."
            )
        _anthropic_llm_cache = ChatAnthropic(
            model=ANTHROPIC_LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        print(f"[pdf_extract] get_anthropic_llm() ready — model={ANTHROPIC_LLM_MODEL}")
    return _anthropic_llm_cache


def get_vision_model_names():
    """Vision model candidates — Groq only (image OCR)."""
    model_names = [GROQ_VISION_MODEL]
    model_names.extend(
        model.strip()
        for model in GROQ_VISION_FALLBACK_MODELS.split(",")
        if model.strip()
    )
    return list(dict.fromkeys(model_names))


def get_vision_llm(model_name: str = None):
    """Vision-capable LLM for image OCR — Groq (not Claude)."""
    global _vision_llm_cache
    model_name = model_name or GROQ_VISION_MODEL

    if model_name not in _vision_llm_cache:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env for Groq vision OCR."
            )
        _vision_llm_cache[model_name] = ChatGroq(
            model=model_name,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
        )
        print(f"[pdf_extract] get_vision_llm() ready — Groq vision model={model_name}")
    return _vision_llm_cache[model_name]
