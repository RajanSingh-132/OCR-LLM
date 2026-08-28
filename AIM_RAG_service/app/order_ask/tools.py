"""
Power-user tool layer — domain-aware via app/domains/rules/.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.domains.registry import get_domain_profile
from app.domains.retrieval import (
    build_rag_context,
    count_records,
    find_record_by_token,
    format_record_doc_for_context,
    format_record_list_for_context,
    list_recent_records,
    search_records,
    semantic_retrieve,
    sum_numeric_field,
)
from app.domains.rules import get_domain_rules
from app.domains.rules.base import build_calc_result, format_domain_calc_context
from app.domains.lookup import get_lookup_module
from app.domains.rules.invoices import needs_sum as invoice_needs_sum
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
from app.order_ask.entities import entities_to_mongo_filters
from app.tenants.context import get_active_domain

TOOL_GET_RECORD = "get_record"
TOOL_SEARCH = "search_records"
TOOL_LIST_RECENT = "list_recent"
TOOL_RUN_CALCULATION = "run_calculation"
TOOL_RUN_ANALYTICS = "run_analytics"
TOOL_COMPARE = "compare_records"
TOOL_SEMANTIC_RAG = "semantic_rag"
TOOL_COUNT = "count_records"
TOOL_SUM_FIELD = "sum_field"


def plan_tools(
    intent: str,
    entities: Dict[str, Any],
    intent_info: Dict[str, Any],
) -> List[str]:
    """Decide which tools to run using domain-specific rules."""
    domain = get_active_domain()
    rules = get_domain_rules(domain)
    return rules.plan_tools(intent, entities, intent_info)


def _record_token(entities: Dict[str, Any], intent_info: Dict[str, Any]) -> str:
    return (
        entities.get("record_token")
        or entities.get("order_token")
        or intent_info.get("record_token")
        or intent_info.get("order_token")
        or ""
    )


def _compare_tokens(entities: Dict[str, Any], question: str, domain: str) -> List[str]:
    lookup = get_lookup_module(domain)
    tokens: List[str] = []
    tok = _record_token(entities, {})
    if tok:
        tokens.append(str(tok))
    pattern = lookup.compare_token_pattern
    if pattern is not None:
        for m in pattern.finditer(question or ""):
            t = m.group(1)
            if re.fullmatch(r"20\d{2}", t):
                continue
            if t not in tokens:
                tokens.append(t)
            if len(tokens) >= 2:
                break
    return tokens[:2]


def _match_summary(rec: Dict[str, Any], match_type: str) -> Dict[str, Any]:
    domain = get_active_domain()
    profile = get_domain_profile(domain)
    out: Dict[str, Any] = {"domain": domain, "match_type": match_type}
    for field in (*profile.id_fields, *profile.number_fields):
        if rec.get(field) not in (None, ""):
            out[field] = rec.get(field)
    return out


def execute_tools(
    tool_names: List[str],
    *,
    question: str,
    entities: Dict[str, Any],
    retrieve_k: int = 5,
) -> Dict[str, Any]:
    """Run planned tools and return context blocks + structured payloads."""
    domain = get_active_domain()
    profile = get_domain_profile(domain)
    label = profile.label.upper()
    rules = get_domain_rules(domain)

    context_blocks: List[str] = []
    matches: List[Dict[str, Any]] = []
    calc_payload = None
    list_payload = None
    analytics_payload = None
    tools_run: List[str] = []
    active_order_token = _record_token(entities, {})

    for name in tool_names:
        checkpoint("TOOL_RUN", f"start {name}", domain=domain)
        if name == TOOL_GET_RECORD:
            token = _record_token(entities, {})
            doc = find_record_by_token(token) if token else None
            if doc:
                context_blocks.append(
                    f"EXACT {label} RECORD:\n" + format_record_doc_for_context(doc)
                )
                matches.append(_match_summary(doc, "exact"))
                active_order_token = str(token)
                tools_run.append(name)
            else:
                context_blocks.append(f"EXACT {label} RECORD: not found for token={token}")
                tools_run.append(name)

        elif name == TOOL_SEARCH:
            filters = entities_to_mongo_filters(entities, domain=domain)
            limit = int(entities.get("limit") or 15)
            sort_by = entities.get("sort_by") or profile.default_sort
            ascending = bool(entities.get("ascending", False))
            list_payload = search_records(
                filters=filters,
                limit=limit,
                sort_by=sort_by,
                ascending=ascending,
            )
            context_blocks.append(format_record_list_for_context(list_payload))
            for row in list_payload.get("records") or []:
                matches.append(_match_summary(row, "filter"))
            tools_run.append(name)

        elif name == TOOL_LIST_RECENT:
            limit = int(entities.get("limit") or 10)
            list_payload = list_recent_records(limit=limit)
            context_blocks.append(format_record_list_for_context(list_payload))
            for row in list_payload.get("records") or []:
                matches.append(_match_summary(row, "recent"))
            tools_run.append(name)

        elif name == TOOL_COUNT:
            filters = entities_to_mongo_filters(entities, domain=domain)
            total = count_records(filters if filters else None)
            calc_payload = build_calc_result(
                domain=domain,
                question=question,
                total_count=total,
                filters=filters,
            )
            context_blocks.append(format_domain_calc_context(calc_payload))
            tools_run.append(name)

        elif name == TOOL_SUM_FIELD:
            filters = entities_to_mongo_filters(entities, domain=domain)
            field = invoice_needs_sum(entities, question) or "TotalAmount"
            agg = sum_numeric_field(field, filters if filters else None)
            calc_payload = build_calc_result(
                domain=domain,
                question=question,
                sum_field=field,
                sum_total=agg.get("sum_total"),
                filters=filters,
            )
            context_blocks.append(format_domain_calc_context(calc_payload))
            tools_run.append(name)

        elif name == TOOL_RUN_CALCULATION and domain == "orders":
            filters = entities_to_mongo_filters(entities, domain=domain)
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

        elif name == TOOL_RUN_ANALYTICS and domain == "orders":
            analytics_payload = run_analytics(question, entities=entities)
            context_blocks.append(format_analytics_for_context(analytics_payload))
            tools_run.append(name)

        elif name == TOOL_COMPARE:
            tokens = _compare_tokens(entities, question, domain)
            parts = []
            for tok in tokens:
                doc = find_record_by_token(tok)
                if doc:
                    parts.append(
                        f"{label} {tok}:\n" + format_record_doc_for_context(doc, max_fields=40)
                    )
                    matches.append(_match_summary(doc, "compare"))
                else:
                    parts.append(f"{label} {tok}: not found")
            context_blocks.append(f"COMPARE {label}S:\n" + "\n\n".join(parts))
            tools_run.append(name)

        elif name == TOOL_SEMANTIC_RAG:
            embeddings, _ = get_models()
            docs = semantic_retrieve(question, k=max(1, retrieve_k), embeddings=embeddings)
            if docs:
                context_blocks.append(
                    f"RETRIEVED {label} CONTEXT:\n" + build_rag_context(docs)
                )
                for doc in docs:
                    meta = doc.metadata or {}
                    matches.append({**meta, "domain": domain, "match_type": "semantic"})
            else:
                context_blocks.append(f"RETRIEVED {label} CONTEXT: no strong semantic matches")
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
        "domain": domain,
    }
