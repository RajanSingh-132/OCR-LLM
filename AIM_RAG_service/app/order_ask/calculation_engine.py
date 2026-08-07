"""
Formula catalog + calculation engine for Avaal order Q&A.

AI-facing formulas map natural language → exact Mongo aggregations / expressions.
LLM must not invent totals; it explains CALCULATION RESULT blocks from this engine.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.mongo_client import get_mongo_collection
from app.order_ask.config import AVAAL_COLLECTION_NAME, AVAAL_NAMESPACE


# ---------------------------------------------------------------------------
# Formula registry (what the AI is allowed to understand / request)
# ---------------------------------------------------------------------------

FORMULA_CATALOG: Dict[str, Dict[str, Any]] = {
    "order_count": {
        "id": "order_count",
        "label": "Order Count",
        "description": "Total number of orders in Avaal_db.",
        "op": "count",
        "field": None,
        "aliases": [
            "how many orders",
            "order count",
            "number of orders",
            "total orders",
            "count of orders",
        ],
    },
    "total_taxes": {
        "id": "total_taxes",
        "label": "Total Taxes",
        "description": "Sum of taxes across all orders.",
        "op": "sum",
        "field": "taxes",
        "aliases": [
            "total tax",
            "total taxes",
            "sum of tax",
            "sum of taxes",
            "tax total",
            "all taxes",
        ],
    },
    "total_tax_amount": {
        "id": "total_tax_amount",
        "label": "Total Tax Amount",
        "description": "Sum of totaltaxamount across all orders.",
        "op": "sum",
        "field": "totaltaxamount",
        "aliases": ["total tax amount", "sum totaltaxamount"],
    },
    "total_freight": {
        "id": "total_freight",
        "label": "Total Freight",
        "description": "Sum of totalfreight across all orders.",
        "op": "sum",
        "field": "totalfreight",
        "aliases": [
            "total freight",
            "sum freight",
            "freight total",
            "all freight",
        ],
    },
    "total_revenue": {
        "id": "total_revenue",
        "label": "Total Revenue (Gross Freight)",
        "description": "Sum of grosstotalfreight treated as revenue.",
        "op": "sum",
        "field": "grosstotalfreight",
        "aliases": [
            "total revenue",
            "revenue",
            "gross revenue",
            "total gross freight",
            "grosstotalfreight",
            "gross total freight",
        ],
    },
    "total_fuel_charges": {
        "id": "total_fuel_charges",
        "label": "Total Fuel Charges",
        "description": "Sum of fuelcharges.",
        "op": "sum",
        "field": "fuelcharges",
        "aliases": ["total fuel", "fuel charges", "sum fuel", "total fuel charges"],
    },
    "total_other_charges": {
        "id": "total_other_charges",
        "label": "Total Other Charges",
        "description": "Sum of othercharges.",
        "op": "sum",
        "field": "othercharges",
        "aliases": ["total other charges", "other charges", "sum other charges"],
    },
    "total_offered_amount": {
        "id": "total_offered_amount",
        "label": "Total Offered Amount",
        "description": "Sum of offeredamount.",
        "op": "sum",
        "field": "offeredamount",
        "aliases": ["total offered", "offered amount", "sum offered"],
    },
    "total_pretax_amount": {
        "id": "total_pretax_amount",
        "label": "Total Pre-Tax Amount",
        "description": "Sum of pretaxamount.",
        "op": "sum",
        "field": "pretaxamount",
        "aliases": ["total pretax", "pre tax total", "pretax amount"],
    },
    "total_distance": {
        "id": "total_distance",
        "label": "Total Distance",
        "description": "Sum of distance.",
        "op": "sum",
        "field": "distance",
        "aliases": ["total distance", "sum distance", "all miles", "total miles"],
    },
    "avg_taxes": {
        "id": "avg_taxes",
        "label": "Average Taxes",
        "description": "Average taxes per order.",
        "op": "avg",
        "field": "taxes",
        "aliases": ["average tax", "avg tax", "average taxes", "mean tax"],
    },
    "avg_freight": {
        "id": "avg_freight",
        "label": "Average Freight",
        "description": "Average totalfreight per order.",
        "op": "avg",
        "field": "totalfreight",
        "aliases": ["average freight", "avg freight", "mean freight"],
    },
    "avg_revenue": {
        "id": "avg_revenue",
        "label": "Average Revenue",
        "description": "Average grosstotalfreight per order.",
        "op": "avg",
        "field": "grosstotalfreight",
        "aliases": ["average revenue", "avg revenue", "mean revenue"],
    },
    "net_after_tax_estimate": {
        "id": "net_after_tax_estimate",
        "label": "Net After Tax (Estimate)",
        "description": "grosstotalfreight - taxes (sum-level estimate).",
        "op": "expression",
        "expression": "sum_grosstotalfreight - sum_taxes",
        "depends_on": ["grosstotalfreight", "taxes"],
        "aliases": [
            "net after tax",
            "revenue after tax",
            "freight minus tax",
            "net revenue",
        ],
    },
    "freight_plus_fuel": {
        "id": "freight_plus_fuel",
        "label": "Freight + Fuel",
        "description": "totalfreight + fuelcharges (sum-level).",
        "op": "expression",
        "expression": "sum_totalfreight + sum_fuelcharges",
        "depends_on": ["totalfreight", "fuelcharges"],
        "aliases": ["freight plus fuel", "freight and fuel", "freight + fuel"],
    },
}


def list_formula_catalog_for_prompt() -> str:
    """Human-readable formula list for LLM prompts."""
    lines = []
    for fid, meta in FORMULA_CATALOG.items():
        aliases = ", ".join(meta.get("aliases", [])[:6])
        lines.append(
            f"- {fid}: {meta.get('label')} | {meta.get('description')} | "
            f"op={meta.get('op')} field={meta.get('field')} | aliases: {aliases}"
        )
    return "\n".join(lines)


def _base_match(extra_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        "namespace": AVAAL_NAMESPACE,
        "metadata.type": "avaal_order",
    }
    if extra_filters:
        match.update(extra_filters)
    return match


def _detect_filters(question: str) -> Dict[str, Any]:
    """Lightweight filters the calculation engine understands."""
    q = question or ""
    filters: Dict[str, Any] = {}

    m = re.search(r"\bcustomer(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)", q, re.I)
    if m:
        filters["customercode"] = m.group(1).upper()

    m = re.search(r"\bcurrency\s*[:=]?\s*([A-Za-z]{3})\b", q, re.I)
    if m:
        filters["currencycode"] = m.group(1).upper()

    m = re.search(r"\bcompany(?:\s*code)?\s*[:=]?\s*([A-Za-z0-9_-]+)", q, re.I)
    if m:
        filters["companycode"] = m.group(1).upper()

    return filters


def match_formulas(question: str) -> List[str]:
    """Return formula ids matched from natural language."""
    q = (question or "").lower().strip()
    matched: List[str] = []

    # Longer aliases first
    scored: List[Tuple[int, str]] = []
    for fid, meta in FORMULA_CATALOG.items():
        for alias in meta.get("aliases", []):
            alias_l = alias.lower()
            if alias_l in q:
                scored.append((len(alias_l), fid))
                break
        # also allow formula id mention
        if fid.replace("_", " ") in q or fid in q:
            scored.append((len(fid), fid))

    scored.sort(key=lambda x: -x[0])
    for _, fid in scored:
        if fid not in matched:
            matched.append(fid)

    # Generic fallbacks
    if not matched:
        if re.search(r"\b(how many|count)\b.*\border", q) or re.search(
            r"\border.*\b(count|how many)\b", q
        ):
            matched.append("order_count")
        elif "tax" in q and re.search(r"\b(total|sum|all)\b", q):
            matched.append("total_taxes")
        elif "revenue" in q:
            matched.append("total_revenue")
        elif "freight" in q and re.search(r"\b(total|sum|all)\b", q):
            matched.append("total_freight")
        elif re.search(r"\b(total|sum|aggregate|overall)\b", q):
            matched.extend(["order_count", "total_revenue", "total_freight", "total_taxes"])

    return matched


def is_calculation_question(question: str) -> bool:
    q = (question or "").lower()
    if match_formulas(q):
        return True
    if re.search(r"\b(total|sum|average|avg|count|how many|aggregate|formula|calculate|calculation)\b", q):
        return True
    return any(
        w in q
        for w in (
            "revenue",
            "tax",
            "taxes",
            "freight",
            "fuel",
            "offered",
            "pretax",
            "distance",
            "profit",
            "net after",
        )
    )


def _run_field_aggs(
    fields: List[str],
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    collection = get_mongo_collection(AVAAL_COLLECTION_NAME)
    match = _base_match(filters)
    group: Dict[str, Any] = {"_id": None, "order_count": {"$sum": 1}}
    for field in fields:
        group[f"sum_{field}"] = {"$sum": f"${field}"}
        group[f"avg_{field}"] = {"$avg": f"${field}"}
        group[f"min_{field}"] = {"$min": f"${field}"}
        group[f"max_{field}"] = {"$max": f"${field}"}

    rows = list(collection.aggregate([{"$match": match}, {"$group": group}]))
    if not rows:
        return {"order_count": 0}

    row = rows[0]
    row.pop("_id", None)
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def _eval_expression(expression: str, numbers: Dict[str, float]) -> Optional[float]:
    """
    Safe eval for simple + - * / expressions over known sum_/avg_ keys only.
    """
    expr = expression
    # Replace tokens longest-first
    for key in sorted(numbers.keys(), key=len, reverse=True):
        expr = expr.replace(key, str(numbers[key]))

    if not re.fullmatch(r"[0-9eE\.\+\-\*/\(\) \t]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - guarded charset
    except Exception:
        return None


def execute_formulas(
    formula_ids: List[str],
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one or more catalog formulas and return structured results."""
    if not formula_ids:
        return {}

    fields_needed: List[str] = []
    for fid in formula_ids:
        meta = FORMULA_CATALOG.get(fid) or {}
        if meta.get("op") in ("sum", "avg") and meta.get("field"):
            fields_needed.append(meta["field"])
        for dep in meta.get("depends_on") or []:
            fields_needed.append(dep)

    fields_needed = list(dict.fromkeys(fields_needed))
    base = _run_field_aggs(fields_needed or ["taxes"], filters=filters)

    results = {
        "order_count": base.get("order_count", 0),
        "filters_applied": filters or {},
        "formulas": [],
    }

    number_map = {
        k: float(v)
        for k, v in base.items()
        if isinstance(v, (int, float)) and k != "order_count"
    }
    number_map["order_count"] = float(base.get("order_count") or 0)

    for fid in formula_ids:
        meta = FORMULA_CATALOG.get(fid)
        if not meta:
            continue

        item: Dict[str, Any] = {
            "formula_id": fid,
            "label": meta.get("label"),
            "description": meta.get("description"),
            "op": meta.get("op"),
            "field": meta.get("field"),
        }

        op = meta.get("op")
        field = meta.get("field")
        if op == "count":
            item["value"] = base.get("order_count", 0)
            item["formula"] = "COUNT(orders)"
        elif op == "sum" and field:
            item["value"] = base.get(f"sum_{field}", 0)
            item["formula"] = f"SUM({field})"
            item["avg"] = base.get(f"avg_{field}")
            item["min"] = base.get(f"min_{field}")
            item["max"] = base.get(f"max_{field}")
        elif op == "avg" and field:
            item["value"] = base.get(f"avg_{field}", 0)
            item["formula"] = f"AVG({field})"
        elif op == "expression":
            expr = meta.get("expression") or ""
            value = _eval_expression(expr, number_map)
            item["value"] = round(value, 4) if value is not None else None
            item["formula"] = expr
            item["components"] = {
                k: number_map.get(k)
                for k in re.findall(r"sum_[a-z0-9_]+|avg_[a-z0-9_]+|order_count", expr)
            }
        else:
            item["value"] = None

        results["formulas"].append(item)

    return results


def run_calculation_engine(question: str) -> Dict[str, Any]:
    """
    End-to-end: question → matched formulas → Mongo execution → AI-ready payload.
    """
    filters = _detect_filters(question)
    formula_ids = match_formulas(question)
    if not formula_ids and is_calculation_question(question):
        formula_ids = ["order_count", "total_revenue", "total_freight", "total_taxes"]

    payload = execute_formulas(formula_ids, filters=filters)
    payload["matched_formula_ids"] = formula_ids
    payload["question"] = question
    return payload


def format_calculation_result_for_context(payload: Dict[str, Any]) -> str:
    """Plain-text block injected into LLM context."""
    lines = ["CALCULATION RESULT (exact engine — do not recalculate):"]
    filters = payload.get("filters_applied") or {}
    if filters:
        lines.append("Filters: " + ", ".join(f"{k}={v}" for k, v in filters.items()))
    lines.append(f"order_count: {payload.get('order_count')}")

    for item in payload.get("formulas") or []:
        lines.append(
            f"- {item.get('label')} [{item.get('formula_id')}]: "
            f"value={item.get('value')} | formula={item.get('formula')}"
        )
        if item.get("avg") is not None:
            lines.append(
                f"  avg={item.get('avg')} min={item.get('min')} max={item.get('max')}"
            )
        if item.get("components"):
            lines.append(f"  components={item.get('components')}")

    lines.append(
        "Use these values as ground truth. Explain them clearly to the user."
    )
    return "\n".join(lines)
