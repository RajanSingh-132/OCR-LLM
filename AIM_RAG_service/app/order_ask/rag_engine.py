"""
Advanced Avaal OrderBot orchestrator for /api/v1/orders/ask.

Flow:
1) Resolve tenant (corporate_id -> DB + collections)
2) Load session memory (conversation)
3) Understand intent
4) Extract entities (accurate lists + follow-ups)
5) Plan + run tools (power-user Q&A)
6) Anthropic answer with length matched to question
7) Save turn + print terminal checkpoints
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
)
from app.domains.detect import detect_domain
from app.domains.lookup import get_domain_prompts, get_lookup_module
from app.domains.retrieval import format_list_answer_for_user
from app.order_ask.tools import execute_tools, plan_tools
from app.tenants.context import AskContext
from app.tenants.mapping import InvalidCorporateIdError, get_tenant_config
from app.tenants.models import TenantConfig

logger = logging.getLogger("order_ask.rag_engine")


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


def _tenant_fields(
    tenant: Optional[TenantConfig],
    corporate_id: str = "",
    domain: str = "orders",
) -> Dict[str, Any]:
    if tenant is None:
        return {
            "corporate_id": corporate_id or None,
            "company_name": None,
            "database": None,
            "domain": domain,
            "collection": None,
            "namespace": None,
        }
    meta = tenant.to_response_meta(domain)
    return {
        "corporate_id": meta["corporate_id"],
        "company_name": meta["company_name"],
        "database": meta["database"],
        "domain": domain,
        "collection": meta["collection"],
        "namespace": meta["namespace"],
    }


def _response(
    *,
    tenant: Optional[TenantConfig],
    corporate_id: str,
    session_id: str,
    question: str,
    domain: str = "orders",
    **fields: Any,
) -> Dict[str, Any]:
    return {
        **_tenant_fields(tenant, corporate_id, domain),
        "session_id": session_id,
        "question": question,
        **fields,
    }


def answer_order_question(
    question: str,
    conversational: bool = True,
    k: int = 10,
    session_id: Optional[str] = None,
    corporate_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Q&A entry for Avaal orders (conversation + accurate lists + tools).
    """
    timer = CheckpointTimer("ask")
    question = (question or "").strip()
    corporate_id = (corporate_id or "").strip()
    if not session_id:
        session_id = new_session_id()

    checkpoint("=" * 48, "")
    checkpoint(
        "ASK_START",
        "new request",
        question=question[:120],
        session_id=session_id,
        corporate_id=corporate_id or None,
    )

    if not corporate_id:
        return _response(
            tenant=None,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            answer="corporate_id is required.",
            mode="error",
            intent="validation",
            matches=[],
            calculation=None,
            list_result=None,
            tools_used=[],
        )

    try:
        tenant = get_tenant_config(corporate_id)
    except InvalidCorporateIdError as exc:
        return _response(
            tenant=None,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            answer=str(exc),
            mode="error",
            intent="validation",
            matches=[],
            calculation=None,
            list_result=None,
            tools_used=[],
        )

    checkpoint(
        "TENANT",
        "matched existing DB",
        corporate_id=tenant.corporate_id,
        database=tenant.database,
        orders_collection=tenant.collection_for("orders"),
    )

    if not question:
        return _response(
            tenant=tenant,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            answer="Please ask a question about Avaal orders.",
            mode="empty",
            intent="empty",
            matches=[],
            calculation=None,
            list_result=None,
            tools_used=[],
        )

    # 1) Conversation memory
    session = load_session(session_id)
    session_id = session["session_id"]
    stored_corporate_id = (session.get("corporate_id") or "").strip()
    if stored_corporate_id and stored_corporate_id != corporate_id:
        return _response(
            tenant=tenant,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            answer=(
                "This session belongs to a different company. "
                "Start a new session with the correct corporate_id."
            ),
            mode="error",
            intent="validation",
            matches=[],
            calculation=None,
            list_result=None,
            tools_used=[],
        )

    history = format_history_for_prompt(session.get("turns") or [])
    domain = detect_domain(
        question,
        history_hint=str(session.get("last_domain") or ""),
        chat_history=history,
    )

    with AskContext(tenant, domain):
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
        timer.mark("ENTITIES_DONE", entities=entities)

        # Greeting / thanks / ask-for-id: quick reply, no tools
        domain_prompts = get_domain_prompts(domain)
        if intent in ("greeting", "thanks", "chitchat", "ask_for_record_id"):
            checkpoint("ROUTE", f"{intent} — skip tools/RAG")
            if intent == "ask_for_record_id":
                # Fixed sweet ask — never leak MRP/TORD/ETP examples via LLM.
                answer = {
                    "orders": (
                        "Please provide the order number or order id and "
                        "I’ll look it up for you."
                    ),
                    "invoices": (
                        "Please provide the invoice number or invoice id and "
                        "I’ll look it up for you."
                    ),
                    "trips": (
                        "Please provide the trip number or trip id and "
                        "I’ll look it up for you."
                    ),
                }.get(
                    domain,
                    "Please provide the number or id and I’ll look it up for you.",
                )
            else:
                answer = _invoke_anthropic(
                    domain_prompts.greeting,
                    {"question": question, "history": history},
                    max_tokens=max_tokens,
                )
            timer.mark("LLM_DONE", mode=intent)
            save_turn(
                session_id,
                question,
                answer,
                corporate_id=corporate_id,
                domain=domain,
                entities=entities,
                mode=intent,
                intent=intent,
            )
            timer.mark("ASK_END", "complete")
            return _response(
                tenant=tenant,
                corporate_id=corporate_id,
                session_id=session_id,
                question=question,
                domain=domain,
                answer=answer,
                mode=intent,
                intent=intent,
                response_style=style,
                matches=[],
                calculation=None,
                list_result=None,
                analytics=None,
                tools_used=[],
                entities=entities,
            )

        # 4) Power-user tools
        tool_names = plan_tools(intent, entities, intent_info)
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

        mode = intent
        if "run_analytics" in tools_used:
            mode = "analytics"
        elif "run_calculation" in tools_used or "count_records" in tools_used or "sum_field" in tools_used:
            if not matches:
                mode = "calculation"
        elif "search_records" in tools_used or "list_recent" in tools_used:
            mode = "list"
        elif "get_record" in tools_used:
            mode = "exact_record"
        elif "compare_records" in tools_used:
            mode = "compare"
        elif "semantic_rag" in tools_used:
            mode = "rag"

        if not context_blocks:
            checkpoint("ROUTE", "no context — clarify")
            answer = _invoke_anthropic(
                domain_prompts.greeting,
                {"question": question, "history": history},
                max_tokens=min(max_tokens, 120),
            )
            timer.mark("LLM_DONE", mode="clarify")
            save_turn(
                session_id,
                question,
                answer,
                corporate_id=corporate_id,
                domain=domain,
                order_token=active_order,
                entities=entities,
                mode="clarify",
                intent=intent,
            )
            timer.mark("ASK_END", "complete")
            return _response(
                tenant=tenant,
                corporate_id=corporate_id,
                session_id=session_id,
                question=question,
                domain=domain,
                answer=answer,
                mode="clarify",
                intent=intent,
                response_style=style,
                matches=[],
                calculation=None,
                list_result=None,
                analytics=None,
                tools_used=tools_used,
                entities=entities,
            )

        context = "\n\n".join(context_blocks)
        tools_label = ", ".join(tools_used) if tools_used else "none"

        lookup_mod = get_lookup_module(domain)
        lookup_intents = {lookup_mod.intent_name, "order_lookup", "invoice_lookup", "trip_lookup"}

        # List replies need more room so Claude can write all returned rows.
        if mode == "list" and list_payload:
            returned = int(list_payload.get("returned") or 0)
            max_tokens = max(max_tokens, min(2500, 200 + returned * 90))

        try:
            # List answers: format from Mongo rows in code (full N rows, $0 LLM).
            # Avoids mid-sentence cutoff when max_tokens is too small for 15+ items.
            if mode == "list" and list_payload:
                checkpoint(
                    "LLM",
                    "skip — deterministic list answer",
                    mode=mode,
                    domain=domain,
                    returned=list_payload.get("returned"),
                )
                answer = format_list_answer_for_user(domain, list_payload)
            else:
                checkpoint("LLM", "Anthropic answer", mode=mode, domain=domain, max_tokens=max_tokens)
                if calc_payload and mode == "calculation" and not matches:
                    formula_prompt = domain_prompts.formula or ORDER_FORMULA_PROMPT
                    answer = _invoke_anthropic(
                        formula_prompt,
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
                elif mode == "exact_record" and intent in lookup_intents:
                    from app.domains.lookup.invoices.prompts import INVOICE_LOOKUP_PROMPT
                    from app.domains.lookup.orders.prompts import ORDER_LOOKUP_PROMPT
                    from app.domains.lookup.trips.prompts import TRIP_LOOKUP_PROMPT

                    lookup_prompts = {
                        "orders": ORDER_LOOKUP_PROMPT,
                        "invoices": INVOICE_LOOKUP_PROMPT,
                        "trips": TRIP_LOOKUP_PROMPT,
                    }
                    answer = _invoke_anthropic(
                        lookup_prompts.get(domain, ORDER_LOOKUP_PROMPT),
                        {
                            "context": context,
                            "question": question,
                            "history": history,
                        },
                        max_tokens=max_tokens,
                    )
                else:
                    prompt = (
                        domain_prompts.conversation if conversational else domain_prompts.ask
                    )
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

        sticky = dict(session.get("last_entities") or {})
        sticky.update({k: v for k, v in entities.items() if k != "from_session"})
        save_turn(
            session_id,
            question,
            answer,
            corporate_id=corporate_id,
            domain=domain,
            order_token=active_order,
            entities=sticky,
            mode=mode,
            intent=intent,
        )
        timer.mark("ASK_END", "complete", session_id=session_id)
        checkpoint("=" * 48, "")

        return _response(
            tenant=tenant,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            domain=domain,
            answer=answer,
            mode=mode,
            intent=intent,
            response_style=style,
            matches=matches[:20],
            calculation=calc_payload,
            analytics=analytics_payload,
            list_result=list_payload,
            tools_used=tools_used,
            entities=entities,
        )
