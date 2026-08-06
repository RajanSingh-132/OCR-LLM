import os
from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic  # Anthropic (commented out, replaced by Groq)
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

_embeddings_cache = None
_llm_cache = None
_vision_llm_cache = {}

# Read env vars
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GROQ_VISION_FALLBACK_MODELS = os.environ.get(
    "GROQ_VISION_FALLBACK_MODELS",
    ""
)
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # Commented out
# LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")  # Commented out
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")


def get_models():
    """Return (embeddings, llm) pair with simple caching. Uses Groq LLM."""
    global _embeddings_cache, _llm_cache
    if _embeddings_cache is None or _llm_cache is None:
        # Set AWS credentials as environment variables for Bedrock client libs
        if BEDROCK_ACCESS_KEY:
            os.environ["AWS_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY
        if BEDROCK_SECRET_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_KEY

        # Initialize Bedrock embeddings
        embeddings = BedrockEmbeddings(
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"dimensions": 1024}
        )

        # Initialize Groq LLM
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
        )

        # --- Anthropic LLM (commented out) ---
        # if not ANTHROPIC_API_KEY:
        #     raise ValueError("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")
        # llm = ChatAnthropic(
        #     model=LLM_MODEL,
        #     anthropic_api_key=ANTHROPIC_API_KEY,
        #     temperature=0.0,
        # )
        # --------------------------------------

        _embeddings_cache = embeddings
        _llm_cache = llm

    return _embeddings_cache, _llm_cache


def get_vision_model_names():
    """Return Groq vision model candidates for image OCR."""
    model_names = [GROQ_VISION_MODEL]
    model_names.extend(
        model.strip()
        for model in GROQ_VISION_FALLBACK_MODELS.split(",")
        if model.strip()
    )
    return list(dict.fromkeys(model_names))


def get_vision_llm(model_name: str = None):
    """Return a vision-capable Groq LLM for image OCR.
    Uses a separate cache from the main text LLM."""
    global _vision_llm_cache
    model_name = model_name or GROQ_VISION_MODEL
    if model_name not in _vision_llm_cache:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
        _vision_llm_cache[model_name] = ChatGroq(
            model=model_name,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
        )
    return _vision_llm_cache[model_name]
