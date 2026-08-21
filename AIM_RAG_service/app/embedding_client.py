import os
# from langchain_groq import ChatGroq  # disabled: vision + JSON use Anthropic Sonnet
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

_embeddings_cache = None
_llm_cache = None
_vision_llm_cache = {}
_anthropic_llm_cache = None

# Vision OCR + JSON extract + /orders/ask: Anthropic Claude (LLM_MODEL).
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
# GROQ_VISION_FALLBACK_MODELS = os.environ.get("GROQ_VISION_FALLBACK_MODELS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")


def get_models():
    """Return (embeddings, llm). Embeddings=Bedrock; LLM=Anthropic Sonnet."""
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

        llm = get_anthropic_llm()

        _embeddings_cache = embeddings
        _llm_cache = llm
        print(
            "[pdf_extract] get_models() ready — Bedrock embeddings + "
            f"Claude LLM ({ANTHROPIC_LLM_MODEL})"
        )

    return _embeddings_cache, _llm_cache


def get_anthropic_llm():
    """Anthropic Claude Sonnet — JSON extract + /orders/ask + vision OCR."""
    global _anthropic_llm_cache
    if _anthropic_llm_cache is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it in .env for Claude extract/ask/vision."
            )
        _anthropic_llm_cache = ChatAnthropic(
            model=ANTHROPIC_LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        print(f"[pdf_extract] get_anthropic_llm() ready — model={ANTHROPIC_LLM_MODEL}")
    return _anthropic_llm_cache


def get_vision_model_names():
    """Vision model — Anthropic Sonnet only (Groq removed)."""
    return [ANTHROPIC_LLM_MODEL]


def get_vision_llm(model_name: str = None):
    """Vision OCR — Anthropic Claude Sonnet (same client as JSON extract)."""
    global _vision_llm_cache
    model_name = model_name or ANTHROPIC_LLM_MODEL

    if model_name not in _vision_llm_cache:
        # Same ChatAnthropic client accepts multimodal image messages
        _vision_llm_cache[model_name] = get_anthropic_llm()
        print(
            f"[pdf_extract] get_vision_llm() ready — "
            f"Claude vision model={model_name}"
        )
    return _vision_llm_cache[model_name]
