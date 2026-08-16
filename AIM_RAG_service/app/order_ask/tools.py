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
from app.order_ask.trip_analytics import (
    format_trip_analytics_for_context,
    is_trip_analytics_question,
    run_trip_analytics,
)
from app.order_ask.trip_retrieval import (
    build_trip_rag_context,
    find_related_orders_for_trip,
    find_trip_by_id_or_number,
    find_trips_for_order,
    format_trip_doc_for_context,
    format_trip_list_for_context,
    retrieve_avaal_trips,
    search_trips,
)

TOOL_GET_ORDER = "get_order"
TOOL_SEARCH_ORDERS = "search_orders"
TOOL_LIST_RECENT = "list_recent"
TOOL_RUN_CALCULATION = "run_calculation"
TOOL_RUN_ANALYTICS = "run_analytics"
TOOL_COMPARE_ORDERS = "compare_orders"
TOOL_SEMANTIC_RAG = "semantic_rag"
TOOL_GET_TRIP = "get_trip"
TOOL_RUN_TRIP_ANALYTICS = "run_trip_analytics"
TOOL_SEARCH_TRIPS = "search_trips"
TOOL_SEMANTIC_TRIP_RAG = "semantic_trip_rag"
TOOL_TRIPS_FOR_ORDER = "trips_for_order"


def plan_tools(
    intent: str,
    entities: Dict[str, Any],
    intent_info: Dict[str, Any],
    question: str = "",
) -> List[str]:
    """Decide which tools to run (ordered)."""
    tools: List[str] = []
    intent = (intent or "").lower()
    q = question or ""

    if intent in ("greeting", "thanks", "chitchat", "empty"):
        return []

    # Trip analytics (best/worst/longest/status) before order analytics
    if (
        intent == "trip_analytics"
        or intent_info.get("needs_trip_analytics")
        or entities.get("analytics")
        in (
            "best_trip",
            "worst_trip",
            "best_worst_trips",
            "longest_trip",
            "shortest_trip",
            "trip_status_summary",
        )
        or is_trip_analytics_question(q)
    ):
        tools.append(TOOL_RUN_TRIP_ANALYTICS)

    if intent == "trip_lookup" or intent_info.get("needs_exact_trip") or entities.get("trip_token"):
        tools.append(TOOL_GET_TRIP)

    if intent == "trips_for_order" or intent_info.get("needs_trips_for_order") or entities.get(
        "want_trip_for_order"
    ):
        tools.append(TOOL_TRIPS_FOR_ORDER)
        if entities.get("order_token") and TOOL_GET_ORDER not in tools:
            tools.append(TOOL_GET_ORDER)

    if (
        intent == "analytics"
        or intent_info.get("needs_analytics")
        or (
            entities.get("analytics")
            and entities.get("analytics")
            not in (
                "best_trip",
                "worst_trip",
                "best_worst_trips",
                "longest_trip",
                "shortest_trip",
                "trip_status_summary",
            )
        )
        or entities.get("country")
    ) and TOOL_RUN_TRIP_ANALYTICS not in tools:
        tools.append(TOOL_RUN_ANALYTICS)

    if intent == "calculation" or intent_info.get("needs_calculation"):
        if TOOL_RUN_ANALYTICS not in tools and TOOL_RUN_TRIP_ANALYTICS not in tools:
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
        # Don't treat ETP as order
        if token and re.match(r"^ETP\d+$", str(token), re.I):
            fake_pin_token = True
            if not entities.get("trip_token"):
                entities["trip_token"] = str(token).upper()
                if TOOL_GET_TRIP not in tools:
                    tools.append(TOOL_GET_TRIP)
        if token and not fake_pin_token and not (
            TOOL_RUN_ANALYTICS in tools and re.fullmatch(r"20\d{2}", str(token))
        ):
            if not entities.get("order_token") and intent_info.get("order_token"):
                entities["order_token"] = intent_info["order_token"]
            if entities.get("order_token") or (
                intent_info.get("order_token")
                and not re.fullmatch(r"20\d{2}", str(intent_info.get("order_token")))
            ):
                if TOOL_GET_ORDER not in tools:
                    tools.append(TOOL_GET_ORDER)
                # Cross-link: order ↔ trip
                if entities.get("want_trip_for_order") or (
                    entities.get("order_token")
                    and re.search(r"\btrips?\b", q, re.I)
                    and TOOL_GET_TRIP not in tools
                ):
                    if TOOL_TRIPS_FOR_ORDER not in tools:
                        tools.append(TOOL_TRIPS_FOR_ORDER)

    if intent in ("list_filter", "list_orders", "filter") or (
        entities.get("sort_by") and intent not in ("analytics", "trip_analytics")
    ):
        tools.append(TOOL_SEARCH_ORDERS)

    if intent in ("trip_list", "list_trips"):
        tools.append(TOOL_SEARCH_TRIPS)

    # Pin / state / city / address / customer / location etc. → always filter DB
    if (
        has_geo_or_list_filters(entities)
        and TOOL_RUN_ANALYTICS not in tools
        and TOOL_RUN_TRIP_ANALYTICS not in tools
        and TOOL_SEARCH_ORDERS not in tools
        and TOOL_GET_ORDER not in tools
        and TOOL_GET_TRIP not in tools
    ):
        tools.append(TOOL_SEARCH_ORDERS)

    if intent == "list_recent":
        tools.append(TOOL_LIST_RECENT)

    if intent == "compare":
        tools.append(TOOL_COMPARE_ORDERS)

    if intent_info.get("needs_trip_rag") or intent == "trip_open_qa":
        if TOOL_GET_TRIP not in tools and TOOL_RUN_TRIP_ANALYTICS not in tools:
            tools.append(TOOL_SEMANTIC_TRIP_RAG)

    if intent_info.get("needs_rag") or intent == "open_qa":
        filters = entities_to_mongo_filters(entities)
        if filters and TOOL_SEARCH_ORDERS not in tools:
            tools.append(TOOL_SEARCH_ORDERS)
        elif (
            TOOL_GET_ORDER not in tools
            and TOOL_SEARCH_ORDERS not in tools
            and TOOL_RUN_ANALYTICS not in tools
            and TOOL_GET_TRIP not in tools
            and TOOL_RUN_TRIP_ANALYTICS not in tools
            and TOOL_SEMANTIC_TRIP_RAG not in tools
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
    trip_analytics_payload = None
    trip_list_payload = None
    tools_run: List[str] = []
    active_order_token = entities.get("order_token")
    active_trip_token = entities.get("trip_token")

    for name in tool_names:
        checkpoint("TOOL_RUN", f"start {name}")
        if name == TOOL_GET_TRIP:
            token = entities.get("trip_token") or active_trip_token
            doc = find_trip_by_id_or_number(token) if token else None
            if doc:
                context_blocks.append(
                    "EXACT TRIP RECORD:\n" + format_trip_doc_for_context(doc)
                )
                matches.append(
                    {
                        "tripid": doc.get("tripid"),
                        "tripnumber": doc.get("tripnumber"),
                        "match_type": "exact_trip",
                    }
                )
                active_trip_token = str(
                    doc.get("tripnumber") or doc.get("tripid") or token
                )
                # Join related orders when question mentions order / which order
                if re.search(
                    r"\b(order|orders|ordernumber|orderid|kis order|which order)\b",
                    question or "",
                    re.I,
                ) or entities.get("want_orders_for_trip"):
                    related = find_related_orders_for_trip(doc)
                    if related:
                        parts = [
                            format_order_doc_for_context(o, max_fields=35) for o in related[:8]
                        ]
                        context_blocks.append(
                            "RELATED ORDERS FOR TRIP:\n" + "\n\n".join(parts)
                        )
                        for o in related:
                            matches.append(
                                {
                                    "orderid": o.get("orderid"),
                                    "ordernumber": o.get("ordernumber"),
                                    "match_type": "trip_join_order",
                                }
                            )
                tools_run.append(name)
            else:
                context_blocks.append(f"EXACT TRIP RECORD: not found for token={token}")
                tools_run.append(name)

        elif name == TOOL_TRIPS_FOR_ORDER:
            token = entities.get("order_token") or active_order_token
            trips = find_trips_for_order(token) if token else []
            if trips:
                parts = [format_trip_doc_for_context(t, max_fields=50) for t in trips]
                context_blocks.append(
                    f"TRIPS LINKED TO ORDER {token}:\n" + "\n\n".join(parts)
                )
                for t in trips:
                    matches.append(
                        {
                            "tripid": t.get("tripid"),
                            "tripnumber": t.get("tripnumber"),
                            "match_type": "order_join_trip",
                        }
                    )
                active_trip_token = str(
                    trips[0].get("tripnumber") or trips[0].get("tripid") or ""
                ) or active_trip_token
            else:
                context_blocks.append(f"TRIPS LINKED TO ORDER {token}: none found")
            tools_run.append(name)

        elif name == TOOL_RUN_TRIP_ANALYTICS:
            trip_analytics_payload = run_trip_analytics(question, entities=entities)
            context_blocks.append(
                format_trip_analytics_for_context(trip_analytics_payload)
            )
            tools_run.append(name)

        elif name == TOOL_SEARCH_TRIPS:
            limit = int(entities.get("limit") or 15)
            trip_list_payload = search_trips(limit=limit)
            context_blocks.append(format_trip_list_for_context(trip_list_payload))
            for row in trip_list_payload.get("trips") or []:
                matches.append(
                    {
                        "tripid": row.get("tripid"),
                        "tripnumber": row.get("tripnumber"),
                        "match_type": "trip_list",
                    }
                )
            tools_run.append(name)

        elif name == TOOL_SEMANTIC_TRIP_RAG:
            embeddings, _ = get_models()
            docs = retrieve_avaal_trips(
                question, k=max(1, retrieve_k), embeddings=embeddings
            )
            if docs:
                context_blocks.append(
                    "RETRIEVED TRIP CONTEXT:\n" + build_trip_rag_context(docs)
                )
                for doc in docs:
                    meta = doc.metadata or {}
                    matches.append(
                        {
                            "tripid": meta.get("tripid"),
                            "tripnumber": meta.get("tripnumber"),
                            "similarity_score": meta.get("similarity_score"),
                            "match_type": "semantic_trip",
                        }
                    )
            else:
                context_blocks.append(
                    "RETRIEVED TRIP CONTEXT: no strong semantic matches"
                )
            tools_run.append(name)

        elif name == TOOL_GET_ORDER:
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
        "analytics": analytics_payload or trip_analytics_payload,
        "list_result": list_payload or trip_list_payload,
        "tools_run": tools_run,
        "active_order_token": active_order_token,
        "active_trip_token": active_trip_token,
    }
