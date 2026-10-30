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
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate

from app.embedding_client import get_anthropic_llm, get_xai_llm
from app.Ai_Assistant.calculation_engine import (
    format_calculation_result_for_context,
    list_formula_catalog_for_prompt,
)
from app.Ai_Assistant.config import CheckpointTimer, checkpoint
from app.Ai_Assistant.entities import extract_entities
from app.Ai_Assistant.intent import classify_intent_common, understand_question
from app.Ai_Assistant.memory import (
    format_history_for_prompt,
    load_session,
    new_session_id,
    save_turn,
)
from app.Ai_Assistant.prompts import ORDER_FORMULA_PROMPT
from app.System_prompt.common import NOT_IN_DATA_REPLY, OUT_OF_SCOPE_REPLY
from app.domains.detect import (
    _STRONG_SCORE as STRONG_DOMAIN_SCORE,
    _score_domains,
    detect_domain_local,
)
from app.domains.registry import (
    ROUTING_HINTS,
    domain_to_collection,
    resolve_collection,
)
from app.domains.lookup import get_domain_prompts, get_lookup_module
from app.Ai_Assistant.tools import execute_tools, plan_tools
from app.tenants.context import AskContext
from app.tenants.mapping import InvalidCorporateIdError, get_tenant_config
from app.tenants.models import TenantConfig

logger = logging.getLogger("order_ask.rag_engine")

# A bare record token ("MRP12345", "TORD9981", "123456") — cheap deterministic
# fast path, never worth a planner LLM call.
_BARE_TOKEN_RE = re.compile(
    r"^\s*(MRP[A-Za-z0-9-]*\d[A-Za-z0-9-]*|TORD[A-Za-z0-9-]*\d[A-Za-z0-9-]*|"
    r"TMP[A-Za-z0-9-]*\d[A-Za-z0-9-]*|[A-Za-z]{2,6}\d{2,}|\d{4,})\s*$",
    re.IGNORECASE,
)

_MORE_FOLLOWUP_RE = re.compile(
    r"^\s*(more|more\s+details?|tell\s+me\s+more|continue|elaborate|"
    r"aur\s+(batao|do|details?)|details?\s+aur|full\s+details?)\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Short "yes / go ahead / do it" replies — meaningless on their own; when the bot
# just offered to do something they mean "execute what you offered".
_AFFIRM_RE = re.compile(
    r"^\s*(yes|yes\s*please|yes\s*go|yep|yeah|ya|yup|sure|ok|okay|k|alright|"
    r"go\s*ahead|go\s*for\s*it|please(\s*do)?|do\s*it|proceed|continue|"
    r"sounds?\s*good|absolutely|definitely|correct|right|"
    r"haan|haan\s*ji|haanji|ha|kar\s*do|dikha\s*do|dikhao|batao|"
    r"show\s*me|fetch\s*it|get\s*it)\b[\s!.]*$",
    re.IGNORECASE,
)

_OFFER_RE = re.compile(
    r"would you like|shall i\b|should i\b|do you want me|want me to|"
    r"let me know if you|i can (fetch|pull|show|get|provide|run|give|help|do)|"
    r"i'?ll (fetch|pull|show|get|run|give)|would you like me to",
    re.IGNORECASE,
)


def _last_user_and_bot(turns: list) -> tuple[str, str]:
    """Most recent prior user question + assistant answer from stored turns."""
    prev_user = ""
    prev_bot = ""
    for turn in reversed(turns or []):
        role = turn.get("role")
        if role == "assistant" and not prev_bot:
            prev_bot = (turn.get("content") or "").strip()
        elif role == "user" and not prev_user:
            prev_user = (turn.get("content") or "").strip()
        if prev_user and prev_bot:
            break
    return prev_user, prev_bot


def _is_affirmation_followup(question: str, turns: list) -> str:
    """
    If `question` is a bare 'yes / continue / go ahead' AND the bot's last answer
    offered to do something, return the prior user question to replay. Else "".
    """
    if not _AFFIRM_RE.match((question or "").strip()):
        return ""
    prev_user, prev_bot = _last_user_and_bot(turns)
    if not prev_user or _AFFIRM_RE.match(prev_user):
        return ""
    if _OFFER_RE.search(prev_bot):
        return prev_user
    return ""

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


_WORD_RE = re.compile(r"\S+\s+")


def _invoke_anthropic_stream(
    prompt_text: str,
    variables: Dict[str, Any],
    max_tokens: int = 300,
):
    """
    Same call as _invoke_anthropic but yields the answer word-by-word (each
    piece = one word + its trailing whitespace) instead of the LLM's raw,
    multi-word network chunks — gives a smooth one-word-at-a-time typing
    effect on the client.

    The LLM's stream() yields arbitrarily-sized text pieces (several words at
    once, or a partial word split across pieces) — buffer them and only emit
    a word once we've seen the whitespace after it, so a word is never split
    across two SSE events.

    Final-answer model: xAI Grok (see get_xai_llm()) — swapped in from Claude
    Sonnet. The Claude implementation is commented out below for a quick
    rollback (just swap which line runs).
    """
    # llm = get_anthropic_llm()  # Claude Sonnet — disabled, kept for rollback
    llm = get_xai_llm()
    try:
        llm_bound = llm.bind(max_tokens=max(32, int(max_tokens)))
    except Exception:
        llm_bound = llm
    chain = PromptTemplate.from_template(prompt_text) | llm_bound

    buffer = ""
    for chunk in chain.stream(variables):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if not text:
            continue
        buffer += text
        last_end = 0
        for m in _WORD_RE.finditer(buffer):
            yield m.group(0)
            last_end = m.end()
        buffer = buffer[last_end:]
    if buffer:
        # Last word of the answer has no trailing whitespace to wait for.
        yield buffer


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
        # Which collection the answer actually came from (order / trip /
        # invoice) — may differ from the payload `collection`, see
        # _pick_domain(). None on validation errors (nothing was queried).
        "related": None if fields.get("mode") == "error" else domain_to_collection(domain),
        "session_id": session_id,
        "question": question,
        **fields,
    }


def _pick_domain(question: str, payload_domain: str) -> str:
    """
    Domain to answer from. The payload `collection` is the default, but a
    question that clearly names another record type ("how many trips…" with
    collection=order, or a trip/invoice number like ETP123 / INO4) is answered
    from that type's collection instead. Keyword-only — no LLM call.

    The payload domain is kept when the question also strongly names the
    payload type ("orders on trip ETP123"). A question that names no record
    type at all falls back to ROUTING_HINTS ("driver count per company" ->
    trips, "which currency does X use" -> orders), else the payload domain.
    """
    question_domain, _reason, scores = detect_domain_local(question)
    from app.Ai_Assistant.Orders.profitability import is_profitability_or_lane_question

    if is_profitability_or_lane_question(question) and max(
        scores.get("trips", 0.0), scores.get("invoices", 0.0)
    ) < STRONG_DOMAIN_SCORE:
        # Profit / margin / lane questions are computed from order revenue and
        # carrier cost — only the orders domain holds both.
        return "orders"
    if max(scores.values(), default=0.0) < STRONG_DOMAIN_SCORE:
        for hint_domains, pattern in ROUTING_HINTS:
            if re.search(pattern, question or "", re.I):
                return payload_domain if payload_domain in hint_domains else hint_domains[0]
        return payload_domain
    if question_domain == payload_domain:
        return payload_domain
    if scores.get(question_domain, 0.0) < STRONG_DOMAIN_SCORE:
        return payload_domain
    if _score_domains(question.lower()).get(payload_domain, 0.0) >= STRONG_DOMAIN_SCORE:
        return payload_domain
    return question_domain


def _answer_order_question_gen(
    question: str,
    conversational: bool = True,
    k: int = 10,
    session_id: Optional[str] = None,
    corporate_id: Optional[str] = None,
    collection: Optional[str] = None,
):
    """
    Generator core for /api/v1/orders/ask (conversation + accurate lists + tools).

    `collection` (order / trip / invoice) picks the domain — and so the Mongo
    collection — directly; see app/domains/registry.py COLLECTION_TO_DOMAIN.

    Yields events while it runs:
      {"type": "chunk", "text": "..."}   — a piece of the answer as Claude streams it
      {"type": "final", "response": {...}} — the full response dict (same shape the
                                              non-streaming API has always returned),
                                              yielded exactly once, last.

    `answer_order_question()` below drains this and returns only the final dict
    (existing non-streaming behavior, unchanged). `stream_order_question()` exposes
    the raw events for the SSE endpoint.
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
        yield {"type": "final", "response": _response(
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
        )}
        return

    try:
        domain = resolve_collection(collection or "")
    except ValueError as exc:
        yield {"type": "final", "response": _response(
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
        )}
        return

    try:
        tenant = get_tenant_config(corporate_id)
    except InvalidCorporateIdError as exc:
        yield {"type": "final", "response": _response(
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
        )}
        return

    checkpoint(
        "TENANT",
        "matched existing DB",
        corporate_id=tenant.corporate_id,
        database=tenant.database,
        orders_collection=tenant.collection_for("orders"),
    )

    if not question:
        yield {"type": "final", "response": _response(
            tenant=tenant,
            corporate_id=corporate_id,
            session_id=session_id,
            question=question,
            domain=domain,
            answer="Please ask a question about Avaal orders.",
            mode="empty",
            intent="empty",
            matches=[],
            calculation=None,
            list_result=None,
            tools_used=[],
        )}
        return

    # 1) Conversation memory
    session = load_session(session_id, database=tenant.database)
    session_id = session["session_id"]
    stored_corporate_id = (session.get("corporate_id") or "").strip()
    if stored_corporate_id and stored_corporate_id != corporate_id:
        yield {"type": "final", "response": _response(
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
        )}
        return

    history = format_history_for_prompt(session.get("turns") or [])
    payload_domain = domain
    domain = _pick_domain(question, payload_domain)
    checkpoint(
        "DOMAIN",
        "from payload collection" if domain == payload_domain
        else "question names another collection — switched",
        collection=collection,
        domain=domain,
    )

    with AskContext(tenant, domain):
        timer.mark("MEMORY_READY", "history loaded", turns=len(session.get("turns") or []))

        # "yes" / "go ahead" / "continue" right after the bot offered to do
        # something → replay the prior question with the full conversation context.
        effective_question = question
        replay_of = _is_affirmation_followup(question, session.get("turns") or [])
        if replay_of:
            effective_question = replay_of
            checkpoint(
                "ROUTE",
                "affirmation follow-up → replay prior ask",
                prior=replay_of[:80],
            )

        # 2) Route — a deterministic fast-path handles plain "how many
        #    <records> were created <period>" questions first (single Mongo
        #    count_documents(), zero LLM calls); the LLM query planner is
        #    primary for everything else; the regex engine
        #    (understand_question + extract_entities + plan_tools) stays
        #    wired as the automatic fallback.
        planner_plan = None
        precomputed_tool_result = None
        execute_query_plan = None
        fast_path_hit = False
        if domain in ("orders", "invoices", "trips") and not replay_of:
            try:
                from app.Ai_Assistant.fast_path import try_count_fast_path

                precomputed_tool_result = try_count_fast_path(
                    effective_question, domain
                )
            except Exception as exc:
                logger.warning("Fast-path count check failed: %s", exc, exc_info=True)
                precomputed_tool_result = None
            if precomputed_tool_result is None and domain == "orders":
                # Profit / margin / lane questions — deterministic engine,
                # no planner LLM call.
                from app.Ai_Assistant.Orders.profitability import (
                    format_profitability_for_context,
                    is_profitability_or_lane_question,
                    profitability,
                )

                if is_profitability_or_lane_question(effective_question):
                    try:
                        payload = profitability(effective_question)
                        precomputed_tool_result = {
                            "context_blocks": [format_profitability_for_context(payload)],
                            "matches": [],
                            "calculation": None,
                            "analytics": payload,
                            "list_result": None,
                            "tools_run": ["run_analytics"],
                            "active_order_token": "",
                            "domain": "orders",
                        }
                    except Exception as exc:
                        logger.warning("Profitability engine failed: %s", exc, exc_info=True)
                        precomputed_tool_result = None
            if precomputed_tool_result is not None:
                fast_path_hit = True
                intent = "analytics"
                style = "short"
                max_tokens = 150
                if (precomputed_tool_result.get("analytics") or {}).get(
                    "analytics_type"
                ) == "profitability":
                    style = "medium"
                    max_tokens = 900
                retrieve_k = 0
                entities = {"analytics": "dynamic"}
                # Defensive: nothing downstream should read this for the
                # fast-path (the "more follow-up" branch can't co-match a
                # count question), but keep it defined to avoid a NameError
                # if that ever changes.
                intent_info = {
                    "intent": intent,
                    "response_style": style,
                    "max_tokens_hint": max_tokens,
                    "retrieve_k": retrieve_k,
                }
                checkpoint(
                    "ROUTE",
                    "deterministic fast-path — LLM planner skipped",
                    domain=domain,
                )

        if not fast_path_hit and domain in ("orders", "invoices", "trips") and (
            replay_of
            or (
                classify_intent_common(question) is None
                and not _BARE_TOKEN_RE.match(question or "")
                and not _MORE_FOLLOWUP_RE.match(question or "")
            )
        ):
            try:
                if domain == "orders":
                    from app.Ai_Assistant.Orders.order_query_planner import (
                        PLANNER_ENABLED,
                        execute_query_plan,
                        run_query_planner,
                    )
                elif domain == "trips":
                    from app.Ai_Assistant.Trips.trip_query_planner import (
                        TRIP_PLANNER_ENABLED as PLANNER_ENABLED,
                        execute_trip_query_plan as execute_query_plan,
                        run_trip_query_planner as run_query_planner,
                    )
                else:
                    from app.Ai_Assistant.Invoices.invoice_query_planner import (
                        INVOICE_PLANNER_ENABLED as PLANNER_ENABLED,
                        execute_invoice_query_plan as execute_query_plan,
                        run_invoice_query_planner as run_query_planner,
                    )

                if PLANNER_ENABLED:
                    planner_plan = run_query_planner(
                        effective_question, history=history
                    )
            except Exception as exc:
                logger.warning("Query planner failed: %s", exc, exc_info=True)
                checkpoint("PLANNER", "planning error — regex fallback", error=str(exc))
                planner_plan = None

        if planner_plan is not None:
            intent_info = planner_plan.to_intent_info()
            intent = intent_info["intent"]
            style = intent_info["response_style"]
            max_tokens = int(intent_info["max_tokens_hint"])
            retrieve_k = 0
            entities = planner_plan.to_entities()
            if planner_plan.task == "not_in_data":
                # In-scope question whose answer isn't stored (payroll, company
                # status, drivers with no trips) — fixed reply below; never
                # hand it to the regex fallback, which runs unrelated tools.
                checkpoint("ROUTE", "planner: not in data", reason=planner_plan.reason)
            else:
                try:
                    if planner_plan.task == "ops_summary":
                        # Orders + trips + invoices together — no single planner can.
                        from app.Ai_Assistant.ops_summary import build_ops_summary

                        precomputed_tool_result = build_ops_summary()
                    else:
                        precomputed_tool_result = execute_query_plan(
                            planner_plan, question=effective_question
                        )
                    checkpoint(
                        "ROUTE",
                        "LLM query planner",
                        task=planner_plan.task,
                        intent=intent,
                        reason=planner_plan.reason,
                    )
                except Exception as exc:
                    logger.warning("Query plan execution failed: %s", exc, exc_info=True)
                    checkpoint("PLANNER", "execute error — regex fallback", error=str(exc))
                    planner_plan = None
                    precomputed_tool_result = None

        if planner_plan is None and not fast_path_hit:
            # 2b) Understand intent (regex + Claude fallback)
            intent_info = understand_question(effective_question, history=history)
            intent = intent_info.get("intent") or "open_qa"
            style = intent_info.get("response_style") or "medium"
            max_tokens = int(intent_info.get("max_tokens_hint") or 280)
            retrieve_k = int(intent_info.get("retrieve_k") or 0)
            if retrieve_k < 0:
                retrieve_k = 0
            if retrieve_k > k:
                retrieve_k = k

            # 3) Entities (sticky session for follow-ups)
            entities = extract_entities(
                effective_question,
                session_order_token=session.get("last_order_token"),
                session_entities=session.get("last_entities") or {},
            )
            if intent_info.get("order_token") and not entities.get("order_token"):
                entities["order_token"] = intent_info["order_token"]
            # Intent-LLM filters fill gaps local entity extraction missed
            intent_filters = intent_info.get("filters") or {}
            if isinstance(intent_filters, dict):
                for key, value in intent_filters.items():
                    if value in (None, "", [], {}):
                        continue
                    if key == "limit":
                        if not entities.get("limit"):
                            try:
                                entities["limit"] = int(value)
                            except (TypeError, ValueError):
                                pass
                        continue
                    if not entities.get(key):
                        entities[key] = value

            # Follow-up "more" / "more details" → continue same record (not greeting)
            sticky_token = (
                entities.get("order_token")
                or session.get("last_order_token")
                or ""
            )
            if _MORE_FOLLOWUP_RE.match(effective_question) and sticky_token:
                lookup_mod = get_lookup_module(domain)
                intent = lookup_mod.intent_name
                entities["order_token"] = sticky_token
                entities["record_token"] = sticky_token
                intent_info = {
                    **intent_info,
                    "intent": intent,
                    "needs_exact_order": True,
                    "needs_rag": False,
                    "response_style": "detailed",
                    "max_tokens_hint": max(max_tokens, 900),
                    "retrieve_k": 0,
                    "reason": "more_followup_sticky_token",
                }
                max_tokens = int(intent_info["max_tokens_hint"])
                style = "detailed"
                checkpoint(
                    "ROUTE",
                    "more follow-up → lookup sticky token",
                    domain=domain,
                    token=sticky_token,
                )

        timer.mark("INTENT_DONE", intent=intent, style=style)
        timer.mark("ENTITIES_DONE", entities=entities)

        # Follow-up "more" / "more details" → continue same record (not greeting)
        sticky_token = (
            entities.get("order_token")
            or session.get("last_order_token")
            or ""
        )
        if _MORE_FOLLOWUP_RE.match(question) and sticky_token:
            lookup_mod = get_lookup_module(domain)
            intent = lookup_mod.intent_name
            entities["order_token"] = sticky_token
            entities["record_token"] = sticky_token
            intent_info = {
                **intent_info,
                "intent": intent,
                "needs_exact_order": True,
                "needs_rag": False,
                "response_style": "detailed",
                "max_tokens_hint": max(max_tokens, 900),
                "retrieve_k": 0,
                "reason": "more_followup_sticky_token",
            }
            max_tokens = int(intent_info["max_tokens_hint"])
            style = "detailed"
            checkpoint(
                "ROUTE",
                "more follow-up → lookup sticky token",
                domain=domain,
                token=sticky_token,
            )

        # Greeting / thanks / ask-for-id / out-of-scope: quick reply, no tools
        domain_prompts = get_domain_prompts(domain)
        if precomputed_tool_result is None and intent in (
            "greeting",
            "thanks",
            "chitchat",
            "ask_for_record_id",
            "out_of_scope",
            "not_in_data",
        ):
            checkpoint("ROUTE", f"{intent} — skip tools/RAG")
            if intent == "out_of_scope":
                # Fixed reply — never let the LLM answer general knowledge
                # (see OUT_OF_SCOPE_POLICY in app/System_prompt/common.py).
                answer = OUT_OF_SCOPE_REPLY
                yield {"type": "chunk", "text": answer}
            elif intent == "not_in_data":
                answer = NOT_IN_DATA_REPLY
                yield {"type": "chunk", "text": answer}
            elif intent == "ask_for_record_id":
                # Fixed sweet ask — never leak MRP/TORD/ETP examples via LLM,
                # and never ask for the internal id (business number only —
                # see ID PROTECTION policy in app/System_prompt/common.py).
                answer = {
                    "orders": (
                        "Please provide the order number and "
                        "I’ll look it up for you."
                    ),
                    "invoices": (
                        "Please provide the invoice number and "
                        "I’ll look it up for you."
                    ),
                    "trips": (
                        "Please provide the trip number and "
                        "I’ll look it up for you."
                    ),
                }.get(
                    domain,
                    "Please provide the number and I’ll look it up for you.",
                )
                yield {"type": "chunk", "text": answer}
            else:
                # Fast path: short greeting prompt only — no DB, no heavy CORE policy.
                parts = []
                for piece in _invoke_anthropic_stream(
                    domain_prompts.greeting,
                    {"question": question},
                    max_tokens=min(max_tokens, 120),
                ):
                    parts.append(piece)
                    yield {"type": "chunk", "text": piece}
                answer = "".join(parts)
            timer.mark("LLM_DONE", mode=intent)
            save_turn(
                session_id,
                question,
                answer,
                database=tenant.database,
                corporate_id=corporate_id,
                domain=domain,
                entities=entities,
                mode=intent,
                intent=intent,
            )
            timer.mark("ASK_END", "complete")
            yield {"type": "final", "response": _response(
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
            )}
            return

        # 4) Power-user tools (skipped when the query planner already ran them)
        if precomputed_tool_result is not None:
            tool_result = precomputed_tool_result
        else:
            tool_names = plan_tools(intent, entities, intent_info)
            tool_result = execute_tools(
                tool_names,
                question=effective_question,
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
            parts = []
            for piece in _invoke_anthropic_stream(
                domain_prompts.greeting,
                {"question": effective_question, "history": history},
                max_tokens=min(max_tokens, 120),
            ):
                parts.append(piece)
                yield {"type": "chunk", "text": piece}
            answer = "".join(parts)
            timer.mark("LLM_DONE", mode="clarify")
            save_turn(
                session_id,
                question,
                answer,
                database=tenant.database,
                corporate_id=corporate_id,
                domain=domain,
                order_token=active_order,
                entities=entities,
                mode="clarify",
                intent=intent,
            )
            timer.mark("ASK_END", "complete")
            yield {"type": "final", "response": _response(
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
            )}
            return

        context = "\n\n".join(context_blocks)
        tools_label = ", ".join(tools_used) if tools_used else "none"

        lookup_mod = get_lookup_module(domain)
        lookup_intents = {lookup_mod.intent_name, "order_lookup", "invoice_lookup", "trip_lookup"}

        # List replies need more room so Claude can write all returned rows.
        if mode == "list" and list_payload:
            returned = int(list_payload.get("returned") or 0)
            max_tokens = max(max_tokens, min(2500, 200 + returned * 90))

        try:
            # Always LLM: intent already understood; context from DB tools;
            # Claude formats the user-facing answer (no direct DB dump).
            checkpoint("LLM", "Anthropic answer", mode=mode, domain=domain, max_tokens=max_tokens)
            if calc_payload and mode == "calculation" and not matches:
                formula_prompt = domain_prompts.formula or ORDER_FORMULA_PROMPT
                stream_prompt = formula_prompt
                stream_vars = {
                    "formula_catalog": list_formula_catalog_for_prompt(),
                    "calculation_result": format_calculation_result_for_context(
                        calc_payload
                    ),
                    "question": effective_question,
                    "response_style": style,
                    "history": history,
                }
            elif mode == "exact_record" and intent in lookup_intents:
                stream_prompt = (
                    domain_prompts.lookup
                    or domain_prompts.conversation
                    or domain_prompts.ask
                )
                stream_vars = {
                    "context": context,
                    "question": effective_question,
                    "history": history,
                }
            else:
                stream_prompt = (
                    domain_prompts.conversation if conversational else domain_prompts.ask
                )
                stream_vars = {
                    "context": context,
                    "question": effective_question,
                    "intent": intent,
                    "response_style": style,
                    "history": history,
                    "tools_used": tools_label,
                }

            parts = []
            for piece in _invoke_anthropic_stream(
                stream_prompt, stream_vars, max_tokens=max_tokens
            ):
                parts.append(piece)
                yield {"type": "chunk", "text": piece}
            answer = "".join(parts)
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
            database=tenant.database,
            corporate_id=corporate_id,
            domain=domain,
            order_token=active_order,
            entities=sticky,
            mode=mode,
            intent=intent,
        )
        timer.mark("ASK_END", "complete", session_id=session_id)
        checkpoint("=" * 48, "")

        yield {"type": "final", "response": _response(
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
        )}


def _keep_in_asked_chat(response: Dict[str, Any], collection: Optional[str]) -> Dict[str, Any]:
    """The answer belongs to the chat the question was typed in.

    `related` is set to the payload collection (the client shows the answer in
    that chat); `answered_from` says which collection the data actually came
    from (e.g. a trucks question asked in the Orders chat is answered from
    trips but stays in the Orders chat).
    """
    try:
        asked_in = domain_to_collection(resolve_collection(collection or ""))
    except ValueError:
        return response
    if response.get("related") is not None:
        response["answered_from"] = response["related"]
        response["related"] = asked_in
    return response


def answer_order_question(
    question: str,
    conversational: bool = True,
    k: int = 10,
    session_id: Optional[str] = None,
    corporate_id: Optional[str] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Q&A entry for Avaal orders (conversation + accurate lists + tools).

    Non-streaming — drains _answer_order_question_gen() and returns only its
    final response dict.
    """
    final_response: Dict[str, Any] = {}
    for event in _answer_order_question_gen(
        question, conversational, k, session_id, corporate_id, collection
    ):
        if event.get("type") == "final":
            final_response = _keep_in_asked_chat(event["response"], collection)
    return final_response


def stream_order_question(
    question: str,
    conversational: bool = True,
    k: int = 10,
    session_id: Optional[str] = None,
    corporate_id: Optional[str] = None,
    collection: Optional[str] = None,
):
    """
    Streaming Q&A entry — same pipeline as answer_order_question(), but yields
    events as they happen instead of blocking for the full answer:

      {"type": "chunk", "text": "..."}     — one piece of the answer text
      {"type": "final", "response": {...}} — the complete response dict
                                              (same shape answer_order_question()
                                              returns), yielded once, last

    Used by the SSE endpoint (POST /api/v1/orders/ask/stream).
    """
    for event in _answer_order_question_gen(
        question, conversational, k, session_id, corporate_id, collection
    ):
        if event.get("type") == "final":
            event = {
                **event,
                "response": _keep_in_asked_chat(event["response"], collection),
            }
        yield event
