"""
Advanced Avaal OrderBot orchestrator for /api/v1/orders/ask.

Flow:
1) Load session memory (conversation)
2) Understand intent
3) Extract entities (accurate lists + follow-ups)
4) Plan + run tools (power-user Q&A)
5) Anthropic answer with length matched to question
6) Save turn + print terminal checkpoints
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_anthropic_llm
from app.order_ask.calculation_engine import (
    format_calculation_result_for_context,
    list_formula_catalog_for_prompt,
)
from app.order_ask.checkpoint import CheckpointTimer, checkpoint
from app.order_ask.config import AVAAL_COLLECTION_NAME, AVAAL_NAMESPACE
from app.order_ask.entities import extract_entities
from app.order_ask.intent import understand_question
from app.order_ask.memory import (
    format_history_for_prompt,
    load_session,
    new_session_id,
    save_turn,
)
from app.order_ask.prompts import (
    ORDER_ASK_PROMPT,
    ORDER_FORMULA_PROMPT,
    ORDER_GREETING_PROMPT,
    ORDERBOT_CONVERSATION_PROMPT,
    ORDERBOT_STRUCTURED_PROMPT,
)
from app.order_ask.tools import execute_tools, plan_tools

logger = logging.getLogger("order_ask.rag_engine")

_STRUCTURED_MODES = frozenset(
    {
        "analytics",
        "trip_analytics",
        "exact_trip",
        "calculation",
        "trip_rag",
    }
)

def _invoke_anthropic(
    prompt_text: str,
    variables: Dict[str, Any],
    max_tokens: int = 300,
) -> str:
    llm = get_anthropic_llm()
    try:
        llm_bound = llm.bind(max_tokens=max(32, int(max_tokens)))
    except Exception:
        llm_bound = llm
    chain = PromptTemplate.from_template(prompt_text) | llm_bound
    response = chain.invoke(variables)
    return response.content if hasattr(response, "content") else str(response)


def answer_order_question(
    question: str,
    conversational: bool = True,
    k: int = 10,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Q&A entry for Avaal orders (conversation + accurate lists + tools).
    """
    timer = CheckpointTimer("ask")
    question = (question or "").strip()
    if not session_id:
        session_id = new_session_id()

    checkpoint("=" * 48, "")
    checkpoint("ASK_START", "new request", question=question[:120], session_id=session_id)

    if not question:
        return {
            "answer": "Please ask a question about Avaal orders.",
            "mode": "empty",
            "intent": "empty",
            "matches": [],
            "calculation": None,
            "list_result": None,
            "tools_used": [],
            "session_id": session_id,
            "collection": AVAAL_COLLECTION_NAME,
            "namespace": AVAAL_NAMESPACE,
            "question": question,
        }

    # 1) Conversation memory
    session = load_session(session_id)
    session_id = session["session_id"]
    history = format_history_for_prompt(session.get("turns") or [])
    timer.mark("MEMORY_READY", "history loaded", turns=len(session.get("turns") or []))

    # 2) Understand intent
    intent_info = understand_question(question, history=history)
    intent = intent_info.get("intent") or "open_qa"
    style = intent_info.get("response_style") or "medium"
    max_tokens = int(intent_info.get("max_tokens_hint") or 280)
    retrieve_k = int(intent_info.get("retrieve_k") or 0)
    if retrieve_k < 0:
        retrieve_k = 0
    if retrieve_k > k:
        retrieve_k = k
    timer.mark("INTENT_DONE", intent=intent, style=style)

    # 3) Entities (sticky session for follow-ups)
    entities = extract_entities(
        question,
        session_order_token=session.get("last_order_token"),
        session_entities=session.get("last_entities") or {},
    )
    if intent_info.get("order_token") and not entities.get("order_token"):
        entities["order_token"] = intent_info["order_token"]
    if intent_info.get("trip_token") and not entities.get("trip_token"):
        entities["trip_token"] = intent_info["trip_token"]
    timer.mark("ENTITIES_DONE", entities=entities)

    # Greeting / thanks: quick reply, no tools
    if intent in ("greeting", "thanks", "chitchat"):
        checkpoint("ROUTE", "greeting — skip tools/RAG")
        answer = _invoke_anthropic(
            ORDER_GREETING_PROMPT,
            {"question": question, "history": history},
            max_tokens=max_tokens,
        )
        timer.mark("LLM_DONE", mode="greeting")
        save_turn(
            session_id,
            question,
            answer,
            entities=entities,
            mode="greeting",
            intent=intent,
        )
        timer.mark("ASK_END", "complete")
        return {
            "answer": answer,
            "mode": "greeting",
            "intent": intent,
            "response_style": style,
            "matches": [],
            "calculation": None,
            "list_result": None,
            "analytics": None,
            "tools_used": [],
            "session_id": session_id,
            "entities": entities,
            "collection": AVAAL_COLLECTION_NAME,
            "namespace": AVAAL_NAMESPACE,
            "question": question,
        }

    # 4) Power-user tools
    tool_names = plan_tools(intent, entities, intent_info, question=question)
    tool_result = execute_tools(
        tool_names,
        question=question,
        entities=entities,
        retrieve_k=retrieve_k or 5,
    )
    timer.mark(
        "TOOLS_DONE",
        tools=tool_result.get("tools_run"),
        matches=len(tool_result.get("matches") or []),
    )

    context_blocks = tool_result.get("context_blocks") or []
    matches = tool_result.get("matches") or []
    calc_payload = tool_result.get("calculation")
    list_payload = tool_result.get("list_result")
    analytics_payload = tool_result.get("analytics")
    tools_used = tool_result.get("tools_run") or []
    active_order = tool_result.get("active_order_token") or entities.get("order_token")
    active_trip = tool_result.get("active_trip_token") or entities.get("trip_token")

    mode = intent
    if "run_trip_analytics" in tools_used:
        mode = "trip_analytics"
    elif "get_trip" in tools_used or "trips_for_order" in tools_used:
        mode = "exact_trip"
    elif "semantic_trip_rag" in tools_used:
        mode = "trip_rag"
    elif "run_analytics" in tools_used:
        mode = "analytics"
    elif "run_calculation" in tools_used and not matches:
        mode = "calculation"
    elif "search_orders" in tools_used or "list_recent" in tools_used:
        mode = "list"
    elif "get_order" in tools_used:
        mode = "exact_order"
    elif "compare_orders" in tools_used:
        mode = "compare"
    elif "semantic_rag" in tools_used:
        mode = "rag"

    if not context_blocks:
        checkpoint("ROUTE", "no context — clarify")
        answer = _invoke_anthropic(
            ORDER_GREETING_PROMPT,
            {"question": question, "history": history},
            max_tokens=min(max_tokens, 120),
        )
        timer.mark("LLM_DONE", mode="clarify")
        save_turn(
            session_id,
            question,
            answer,
            order_token=active_order,
            entities=entities,
            mode="clarify",
            intent=intent,
        )
        timer.mark("ASK_END", "complete")
        return {
            "answer": answer,
            "mode": "clarify",
            "intent": intent,
            "response_style": style,
            "matches": [],
            "calculation": None,
            "list_result": None,
            "analytics": None,
            "tools_used": tools_used,
            "session_id": session_id,
            "entities": entities,
            "collection": AVAAL_COLLECTION_NAME,
            "namespace": AVAAL_NAMESPACE,
            "question": question,
        }

    context = "\n\n".join(context_blocks)
    tools_label = ", ".join(tools_used) if tools_used else "none"

    try:
        checkpoint("LLM", "Anthropic answer", mode=mode, max_tokens=max_tokens)
        if calc_payload and mode == "calculation" and not matches:
            answer = _invoke_anthropic(
                ORDER_FORMULA_PROMPT,
                {
                    "formula_catalog": list_formula_catalog_for_prompt(),
                    "calculation_result": format_calculation_result_for_context(
                        calc_payload
                    ),
                    "question": question,
                    "response_style": style,
                    "history": history,
                },
                max_tokens=max_tokens,
            )
        else:
            if conversational:
                if mode in _STRUCTURED_MODES:
                    prompt = ORDERBOT_STRUCTURED_PROMPT
                else:
                    prompt = ORDERBOT_CONVERSATION_PROMPT
            else:
                prompt = ORDER_ASK_PROMPT
            answer = _invoke_anthropic(
                prompt,
                {
                    "context": context,
                    "question": question,
                    "intent": intent,
                    "response_style": style,
                    "history": history,
                    "tools_used": tools_label,
                },
                max_tokens=max_tokens,
            )
    except Exception as exc:
        logger.error("Anthropic answer failed: %s", exc, exc_info=True)
        checkpoint("LLM", "FAILED", error=str(exc))
        raise

    timer.mark("LLM_DONE", mode=mode, answer_chars=len(answer or ""))

    # Persist sticky entities for next turn
    sticky = dict(session.get("last_entities") or {})
    sticky.update({k: v for k, v in entities.items() if k != "from_session"})
    if active_trip:
        sticky["trip_token"] = active_trip
    save_turn(
        session_id,
        question,
        answer,
        order_token=active_order,
        entities=sticky,
        mode=mode,
        intent=intent,
    )
    timer.mark("ASK_END", "complete", session_id=session_id)
    checkpoint("=" * 48, "")

    return {
        "answer": answer,
        "mode": mode,
        "intent": intent,
        "response_style": style,
        "matches": matches[:20],
        "calculation": calc_payload,
        "analytics": analytics_payload,
        "list_result": list_payload,
        "tools_used": tools_used,
        "session_id": session_id,
        "entities": entities,
        "collection": AVAAL_COLLECTION_NAME,
        "namespace": AVAAL_NAMESPACE,
        "question": question,
    }
