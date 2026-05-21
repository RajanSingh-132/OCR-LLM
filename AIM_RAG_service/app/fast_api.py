"""
Small connector exposing the FastAPI `app` from existing module.
"""
from app.fast_api_actions_session_rag import app

__all__ = ["app"]
