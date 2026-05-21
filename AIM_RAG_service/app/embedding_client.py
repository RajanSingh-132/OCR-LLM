import os
# from langchain_groq import ChatGroq  # Groq LLM (commented out, replaced by Anthropic)
from langchain_anthropic import ChatAnthropic
from langchain_aws import BedrockEmbeddings

_embeddings_cache = None
_llm_cache = None

# Read env vars
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Groq API key (commented out)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
BEDROCK_MODEL = os.environ.get("bedrockmodel", "amazon.titan-embed-text-v2:0")
BEDROCK_ACCESS_KEY = os.environ.get("accesskey", "")
BEDROCK_SECRET_KEY = os.environ.get("secretaccesskey", "")
BEDROCK_REGION = os.environ.get("awsregion", "us-east-1")


def get_models():
    """Return (embeddings, llm) pair with simple caching. Uses Anthropic LLM."""
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

        # Initialize Anthropic LLM
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")

        llm = ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )

        # --- Groq LLM (commented out) ---
        # if not GROQ_API_KEY:
        #     raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
        # llm = ChatGroq(
        #     model="llama-3.3-70b-versatile",
        #     groq_api_key=GROQ_API_KEY,
        #     temperature=0.0,
        # )
        # ---------------------------------

        _embeddings_cache = embeddings
        _llm_cache = llm

    return _embeddings_cache, _llm_cache
