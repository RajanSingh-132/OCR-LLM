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
_planner_llm_cache = None

# Vision OCR + JSON extract + /orders/ask: Anthropic Claude (LLM_MODEL).
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
# GROQ_VISION_FALLBACK_MODELS = os.environ.get("GROQ_VISION_FALLBACK_MODELS", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
# Query planner (question -> JSON plan) is a structured-output task, not a
# reasoning-heavy one — a faster/cheaper model answers it well and cuts the
# planner call's latency, which was ~5s of a ~7.5s total request in testing.
# The final user-facing answer still always uses ANTHROPIC_LLM_MODEL (Sonnet)
# via get_anthropic_llm() — only the JSON-plan step uses this one.
PLANNER_LLM_MODEL = os.environ.get("PLANNER_LLM_MODEL", "claude-haiku-4-5-20251001")
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")


def get_embeddings():
    """Return Bedrock embeddings only (no LLM loaded)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        if BEDROCK_ACCESS_KEY:
            os.environ["AWS_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY
        if BEDROCK_SECRET_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_KEY

        _embeddings_cache = BedrockEmbeddings(
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"dimensions": 1024},
        )
        print(f"[embeddings] Bedrock ready — model={BEDROCK_MODEL}")

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
