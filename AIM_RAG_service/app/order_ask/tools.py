"""
Power-user tool layer for Avaal OrderBot.

Planner picks tools from intent + entities; tools hit Mongo / calculation engine.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.embedding_client import get_models
from app.order_ask.analytics import (
    format_analytics_for_context,
    run_analytics,
)
from app.order_ask.calculation_engine import (
    execute_formulas,
    format_calculation_result_for_context,
    is_calculation_question,
    match_formulas,
    run_calculation_engine,
)
from app.order_ask.checkpoint import checkpoint
from app.order_ask.entities import entities_to_mongo_filters, has_geo_or_list_filters
from app.order_ask.rag_retrieval import (
    build_rag_context,
    find_order_by_id_or_number,
    format_order_doc_for_context,
    format_order_list_for_context,
    list_recent_orders,
    retrieve_avaal_orders,
    search_orders,
)

TOOL_GET_ORDER = "get_order"
TOOL_SEARCH_ORDERS = "search_orders"
TOOL_LIST_RECENT = "list_recent"
TOOL_RUN_CALCULATION = "run_calculation"
TOOL_RUN_ANALYTICS = "run_analytics"
TOOL_COMPARE_ORDERS = "compare_orders"
TOOL_SEMANTIC_RAG = "semantic_rag"


def plan_tools(intent: str, entities: Dict[str, Any], intent_info: Dict[str, Any]) -> List[str]:
    """Decide which tools to run (ordered)."""
    tools: List[str] = []
    intent = (intent or "").lower()

    if intent in ("greeting", "thanks", "chitchat", "empty"):
        return []

    if (
        intent == "analytics"
        or intent_info.get("needs_analytics")
        or entities.get("analytics")
        or entities.get("country")
    ):
        tools.append(TOOL_RUN_ANALYTICS)

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if TOOL_RUN_ANALYTICS not in tools:
            tools.append(TOOL_RUN_CALCULATION)

    if intent == "order_lookup" or intent_info.get("needs_exact_order") or entities.get("order_token"):
        token = entities.get("order_token") or intent_info.get("order_token")
        # Skip fake tokens: years, or pin/zip digits mistaken as order ids
        pin = str(entities.get("pin") or "")
        fake_pin_token = bool(
            token
            and pin
            and re.sub(r"\s+", "", str(token)).upper()
            == re.sub(r"\s+", "", pin).upper()
        )
        if token and not fake_pin_token and not (
            TOOL_RUN_ANALYTICS in tools and re.fullmatch(r"20\d{2}", str(token))
        ):
            if not entities.get("order_token") and intent_info.get("order_token"):
                entities["order_token"] = intent_info["order_token"]
            if entities.get("order_token") or (
                intent_info.get("order_token")
                and not re.fullmatch(r"20\d{2}", str(intent_info.get("order_token")))
            ):
                tools.append(TOOL_GET_ORDER)

    if intent in ("list_filter", "list_orders", "filter") or (
        entities.get("sort_by") and intent != "analytics"
    ):
        tools.append(TOOL_SEARCH_ORDERS)

    # Pin / state / city / address / customer / location etc. → always filter DB
    if (
        has_geo_or_list_filters(entities)
        and TOOL_RUN_ANALYTICS not in tools
        and TOOL_SEARCH_ORDERS not in tools
        and TOOL_GET_ORDER not in tools
    ):
        tools.append(TOOL_SEARCH_ORDERS)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent == "compare":
        tools.append(TOOL_COMPARE_ORDERS)

    if intent_info.get("needs_rag") or intent == "open_qa":
        filters = entities_to_mongo_filters(entities)
        if filters and TOOL_SEARCH_ORDERS not in tools:
            tools.append(TOOL_SEARCH_ORDERS)
        elif (
            TOOL_GET_ORDER not in tools
            and TOOL_SEARCH_ORDERS not in tools
            and TOOL_RUN_ANALYTICS not in tools
        ):
            tools.append(TOOL_SEMANTIC_RAG)

    seen = set()
    ordered = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    checkpoint("TOOLS", "plan", tools=ordered, intent=intent)
    return ordered


def _compare_tokens(entities: Dict[str, Any], question: str) -> List[str]:
    tokens = []
    if entities.get("order_token"):
        tokens.append(str(entities["order_token"]))
    for m in re.finditer(r"\b(MRP\d+|TORD\d+|\d{4,})\b", question or "", re.I):
        tok = m.group(1)
        if re.fullmatch(r"20\d{2}", tok):
            continue
        if tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= 2:
            break
    return tokens[:2]


def execute_tools(
    tool_names: List[str],
    *,
    question: str,
    entities: Dict[str, Any],
    retrieve_k: int = 5,
) -> Dict[str, Any]:
    """
    Run planned tools and return context blocks + structured payloads.
    """
    context_blocks: List[str] = []
    matches: List[Dict[str, Any]] = []
    calc_payload = None
    list_payload = None
    analytics_payload = None
    tools_run: List[str] = []
    active_order_token = entities.get("order_token")

    for name in tool_names:
        checkpoint("TOOL_RUN", f"start {name}")
        if name == TOOL_GET_ORDER:
            token = entities.get("order_token")
            doc = find_order_by_id_or_number(token) if token else None
            if doc:
                context_blocks.append(
                    "EXACT ORDER RECORD:\n" + format_order_doc_for_context(doc)
                )
                matches.append(
                    {
                        "orderid": doc.get("orderid"),
                        "ordernumber": doc.get("ordernumber"),
                        "match_type": "exact",
                    }
                )
                active_order_token = str(doc.get("orderid") or doc.get("ordernumber") or token)
                tools_run.append(name)
            else:
                context_blocks.append(f"EXACT ORDER RECORD: not found for token={token}")
                tools_run.append(name)

        elif name == TOOL_SEARCH_ORDERS:
            filters = entities_to_mongo_filters(entities)
            limit = int(entities.get("limit") or 15)
            sort_by = entities.get("sort_by") or "orderid"
            ascending = bool(entities.get("ascending", False))
            list_payload = search_orders(
                filters=filters,
                limit=limit,
                sort_by=sort_by,
                ascending=ascending,
            )
            context_blocks.append(format_order_list_for_context(list_payload))
            for row in list_payload.get("orders") or []:
                matches.append(
                    {
                        "orderid": row.get("orderid"),
                        "ordernumber": row.get("ordernumber"),
                        "match_type": "filter",
                    }
                )
            tools_run.append(name)

        elif name == TOOL_LIST_RECENT:
            limit = int(entities.get("limit") or 10)
            list_payload = list_recent_orders(limit=limit)
            context_blocks.append(format_order_list_for_context(list_payload))
            for row in list_payload.get("orders") or []:
                matches.append(
                    {
                        "orderid": row.get("orderid"),
                        "ordernumber": row.get("ordernumber"),
                        "match_type": "recent",
                    }
                )
            tools_run.append(name)

        elif name == TOOL_RUN_CALCULATION:
            filters = entities_to_mongo_filters(entities)
            formula_ids = match_formulas(question)
            if not formula_ids and is_calculation_question(question):
                formula_ids = [
                    "order_count",
                    "total_revenue",
                    "total_freight",
                    "total_taxes",
                ]
            if filters:
                calc_payload = execute_formulas(formula_ids, filters=filters)
                calc_payload["matched_formula_ids"] = formula_ids
                calc_payload["question"] = question
            else:
                calc_payload = run_calculation_engine(question)
            context_blocks.append(format_calculation_result_for_context(calc_payload))
            tools_run.append(name)

        elif name == TOOL_RUN_ANALYTICS:
            analytics_payload = run_analytics(question, entities=entities)
            context_blocks.append(format_analytics_for_context(analytics_payload))
            tools_run.append(name)

        elif name == TOOL_COMPARE_ORDERS:
            tokens = _compare_tokens(entities, question)
            parts = []
            for tok in tokens:
                doc = find_order_by_id_or_number(tok)
                if doc:
                    parts.append(
                        f"ORDER {tok}:\n" + format_order_doc_for_context(doc, max_fields=40)
                    )
                    matches.append(
                        {
                            "orderid": doc.get("orderid"),
                            "ordernumber": doc.get("ordernumber"),
                            "match_type": "compare",
                        }
                    )
                else:
                    parts.append(f"ORDER {tok}: not found")
            context_blocks.append("COMPARE ORDERS:\n" + "\n\n".join(parts))
            tools_run.append(name)

        elif name == TOOL_SEMANTIC_RAG:
            embeddings, _ = get_models()
            docs = retrieve_avaal_orders(question, k=max(1, retrieve_k), embeddings=embeddings)
            if docs:
                context_blocks.append("RETRIEVED ORDER CONTEXT:\n" + build_rag_context(docs))
                for doc in docs:
                    meta = doc.metadata or {}
                    matches.append(
                        {
                            "orderid": meta.get("orderid"),
                            "ordernumber": meta.get("ordernumber"),
                            "similarity_score": meta.get("similarity_score"),
                            "match_type": "semantic",
                        }
                    )
            else:
                context_blocks.append("RETRIEVED ORDER CONTEXT: no strong semantic matches")
            tools_run.append(name)

        checkpoint("TOOL_RUN", f"done {name}", blocks=len(context_blocks), matches=len(matches))

    return {
        "context_blocks": context_blocks,
        "matches": matches,
        "calculation": calc_payload,
        "analytics": analytics_payload,
        "list_result": list_payload,
        "tools_run": tools_run,
        "active_order_token": active_order_token,
    }
