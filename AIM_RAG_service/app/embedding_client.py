import os
import boto3
from botocore.config import Config as BotoConfig
# from langchain_groq import ChatGroq  # disabled: vision + JSON use Anthropic Sonnet
from langchain_xai import ChatXAI  # /orders/ask final-answer + fallback LLM
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"))

_embeddings_cache = None
_llm_cache = None
_vision_llm_cache = {}
_anthropic_llm_cache = None
_planner_llm_cache = None
_xai_llm_cache = None

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
PLANNER_LLM_MODEL = os.environ.get("PLANNER_LLM_MODEL", "claude-haiku-4-5-20251001")
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")
BEDROCK_MAX_POOL_CONNECTIONS = int(os.environ.get("BEDROCK_MAX_POOL_CONNECTIONS", "130"))


def get_embeddings():
    """Return Bedrock embeddings only (no LLM loaded)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        if BEDROCK_ACCESS_KEY:
            os.environ["AWS_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY
        if BEDROCK_SECRET_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_KEY

        boto_client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=BotoConfig(max_pool_connections=BEDROCK_MAX_POOL_CONNECTIONS),
        )
        _embeddings_cache = BedrockEmbeddings(
            client=boto_client,
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"dimensions": 1024},
        )
        print(
            f"[embeddings] Bedrock ready — model={BEDROCK_MODEL}, "
            f"max_pool_connections={BEDROCK_MAX_POOL_CONNECTIONS}"
        )

    return _embeddings_cache


def get_models():
    """Return (embeddings, llm). Embeddings=Bedrock; LLM=Anthropic Sonnet."""
    global _llm_cache
    embeddings = get_embeddings()
    if _llm_cache is None:
        _llm_cache = get_anthropic_llm()
        print(
            "[pdf_extract] get_models() ready — Bedrock embeddings + "
            f"Claude LLM ({ANTHROPIC_LLM_MODEL})"
        )

    return embeddings, _llm_cache


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


def get_planner_llm():
    """Faster/cheaper Claude model for LLM query planners (JSON-plan
    generation only — order/trip/invoice query_planner.py). Falls back to
    the main Sonnet client if this model can't be reached (e.g. not enabled
    on the account yet), so the planner keeps working either way — just
    without the speed-up."""
    global _planner_llm_cache
    if _planner_llm_cache is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it in .env for Claude extract/ask/vision."
            )
        try:
            candidate = ChatAnthropic(
                model=PLANNER_LLM_MODEL,
                anthropic_api_key=ANTHROPIC_API_KEY,
                temperature=0.0,
            )
            candidate.invoke("ping")  # fail fast here, not on the first real question
            _planner_llm_cache = candidate
            print(f"[planner] fast planner LLM ready — model={PLANNER_LLM_MODEL}")
        except Exception as exc:
            print(
                f"[planner] {PLANNER_LLM_MODEL} unavailable ({exc}); "
                f"falling back to {ANTHROPIC_LLM_MODEL} for the planner"
            )
            _planner_llm_cache = get_anthropic_llm()
    return _planner_llm_cache


def get_xai_llm():
    """
    xAI Grok — the LLM for /orders/ask(/stream): final-answer generation
    (rag_engine.py), intent-classify fallback (intent.py), domain-detect
    fallback (domains/detect.py), and the dynamic-analytics fallback
    (dynamic_analytics.py) all use this client. Query planners
    (order/trip/invoice) stay on get_planner_llm() (Claude Haiku) —
    unchanged.
    """
    global _xai_llm_cache
    if _xai_llm_cache is None:
        if not XAI_API_KEY:
            raise ValueError(
                "XAI_API_KEY is not set. Add it in .env for the /orders/ask LLM."
            )
        _xai_llm_cache = ChatXAI(
            model_name=XAI_MODEL,
            xai_api_key=XAI_API_KEY,
            temperature=0.0,
            max_retries=1,
        )
        print(f"[xai] get_xai_llm() ready — model={XAI_MODEL}")
    return _xai_llm_cache


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
