"""
Order Ask package for /api/v1/orders/ask.

Import the orchestrator directly when needed:
  from app.order_ask.rag_engine import answer_order_question
"""

__all__ = ["answer_order_question"]


def __getattr__(name: str):
    if name == "answer_order_question":
        from app.order_ask.rag_engine import answer_order_question

        return answer_order_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
