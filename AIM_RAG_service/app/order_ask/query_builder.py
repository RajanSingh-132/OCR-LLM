"""
Query builder — turns a ValidatedQuery into a real database call.

Pure execution: this module TRUSTS its input already passed
`payload_validator.validate_payload()` and does NOT re-validate. It is the
only place order questions on the dynamic path become SQL.

Two paths:

  procedure  -> app/postgres_client.py wrappers for the three `_ai`
               functions. NOT DEPLOYED YET — gated on
               AVAAL_AI_PROCEDURES_LIVE (default off). While inactive,
               `execute_validated_query` raises ProceduresNotLiveError so
               the caller can return an honest "not available yet" reply.

  dynamic    -> a parameterized SELECT against trnorder. Identifiers are
               only ever the allowlisted names the validator already
               approved; every value goes through a psycopg placeholder.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Tuple

from psycopg import sql

from app import postgres_client
from app.order_ask import query_allowlist as qa
from app.order_ask.checkpoint import checkpoint
from app.order_ask.payload_validator import ValidatedQuery

logger = logging.getLogger("order_ask.query_builder")

AI_PROCEDURES_LIVE = os.environ.get(
    "AVAAL_AI_PROCEDURES_LIVE", "0"
).strip().lower() in ("1", "true", "yes", "on")

_OP_SQL = {
    "=": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<=",
}


class ProceduresNotLiveError(RuntimeError):
    """Raised when a procedure-path query is requested but the `_ai`
    functions haven't been deployed yet."""

    def __init__(self, procedure_kind: str | None = None):
        self.procedure_kind = procedure_kind
        super().__init__(
            f"AI aggregate/procedure functions not deployed yet "
            f"(procedure_kind={procedure_kind!r})"
        )


# ---------------------------------------------------------------------------
# response shaping
# ---------------------------------------------------------------------------
def shape_rows(rows: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    """Strip internal/join-key columns before anything reaches the LLM or
    the user. Second line of defence on top of the allowlist."""
    cleaned: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        cleaned.append(
            {k: v for k, v in row.items() if k not in qa.ALLOWED_JOIN_KEYS}
        )
    return cleaned


# ---------------------------------------------------------------------------
# procedure path
# ---------------------------------------------------------------------------
async def _run_procedure(corporate_id: str, vq: ValidatedQuery) -> Tuple[int, List[Dict[str, Any]]]:
    if not AI_PROCEDURES_LIVE:
        raise ProceduresNotLiveError(vq.procedure_kind)

    params = vq.procedure_params or {}

    if vq.procedure_kind == "regular":
        total, details = await postgres_client.fetch_orders(
            corporate_id, params, fn_name="fn_getorders_regular_ai"
        )
        checkpoint("QUERYBUILD", "procedure regular", total=total, returned=len(details))
        return total, details

    if vq.procedure_kind == "summary":
        rows = await postgres_client.fetch_orders_summary(corporate_id, params)
        checkpoint("QUERYBUILD", "procedure summary", returned=len(rows))
        return len(rows), rows

    # count | freight
    rows = await postgres_client.fetch_orders_aggregate(
        corporate_id, vq.procedure_kind, params
    )
    if vq.top_n:
        rows = rows[: vq.top_n]
    checkpoint(
        "QUERYBUILD",
        f"procedure {vq.procedure_kind}",
        groups=len(rows),
        group_by=params.get("p_groupby"),
        top_n=vq.top_n,
    )
    return len(rows), rows


# ---------------------------------------------------------------------------
# dynamic path
# ---------------------------------------------------------------------------
def _filter_clause(f: Dict[str, Any]) -> Tuple[sql.Composable, List[Any]]:
    op = f["op"]
    value = f["value"]
    pg_type = f.get("type") or ""

    # Compare timestamp columns as calendar dates so an inclusive range end
    # like "2026-08-31" actually covers that whole day (mirrors the ::date
    # cast every existing order procedure uses).
    if ("timestamp" in pg_type or pg_type == "date") and op in (
        "between", "=", "!=", ">", ">=", "<", "<=", "in",
    ):
        field = sql.SQL("{}::date").format(sql.Identifier(f["field"]))
    else:
        field = sql.Identifier(f["field"])

    if op == "between":
        return (
            sql.SQL("{} BETWEEN {} AND {}").format(field, sql.Placeholder(), sql.Placeholder()),
            [value[0], value[1]],
        )
    if op == "in":
        return sql.SQL("{} = ANY({})").format(field, sql.Placeholder()), [list(value)]
    if op == "ilike":
        return sql.SQL("{} ILIKE {}").format(sql.Identifier(f["field"]), sql.Placeholder()), [value]
    return sql.SQL("{} " + _OP_SQL[op] + " {}").format(field, sql.Placeholder()), [value]


def build_dynamic_sql(vq: ValidatedQuery) -> Tuple[sql.Composed, List[Any]]:
    table = sql.Identifier("public", vq.table)
    params: List[Any] = []

    where_parts = [sql.SQL("isactive = TRUE AND isdeleted = FALSE")]
    for f in vq.filters or []:
        clause, p = _filter_clause(f)
        where_parts.append(clause)
        params.extend(p)
    where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)

    if vq.aggregate:
        agg_fn = sql.SQL(vq.aggregate.upper())
        agg_arg = (
            sql.SQL("*") if vq.aggregate == "count"
            else sql.Identifier(vq.aggregate_field)
        )
        value_expr = sql.SQL("{}({}) AS value").format(agg_fn, agg_arg)

        if vq.group_by:
            gb = sql.Identifier(vq.group_by)
            query = (
                sql.SQL("SELECT {} AS group_key, {} FROM {}").format(gb, value_expr, table)
                + where_sql
                + sql.SQL(" GROUP BY {} ORDER BY value DESC NULLS LAST LIMIT {}").format(
                    gb, sql.Literal(vq.limit)
                )
            )
        else:
            query = sql.SQL("SELECT {} FROM {}").format(value_expr, table) + where_sql
        return query, params

    cols = sql.SQL(", ").join(sql.Identifier(c) for c in vq.select_fields)
    query = (
        sql.SQL("SELECT {} FROM {}").format(cols, table)
        + where_sql
        + sql.SQL(" LIMIT {}").format(sql.Literal(vq.limit))
    )
    return query, params


_SELECT_ONLY_RE = re.compile(r"^\s*SELECT\s", re.IGNORECASE)


def _assert_select_only(rendered: str) -> None:
    """Last-ditch guard: the builder only ever composes SELECTs, but confirm
    it before execution. Identifiers are allowlisted and quoted, values are
    placeheld, so neither can smuggle in a second statement — this catches a
    future bug in build_dynamic_sql(), not an injection."""
    if not _SELECT_ONLY_RE.match(rendered):
        raise ValueError(f"query_builder produced a non-SELECT statement: {rendered[:80]!r}")
    # psycopg forbids multiple statements in one execute() anyway; belt & braces.
    if ";" in rendered.rstrip().rstrip(";"):
        raise ValueError("query_builder produced multiple statements")


async def _run_dynamic(corporate_id: str, vq: ValidatedQuery) -> Tuple[int, List[Dict[str, Any]]]:
    query, params = build_dynamic_sql(vq)

    from psycopg.rows import dict_row

    from app.postgres_client import readonly_connection, resolve_tenant_db_name

    db_name = resolve_tenant_db_name(corporate_id)
    # readonly_connection pins the session READ ONLY + a statement timeout, so
    # a stray write (bug or crafted payload) is refused by Postgres itself.
    async with readonly_connection(db_name) as conn:
        rendered = query.as_string(conn)
        _assert_select_only(rendered)
        checkpoint("QUERYBUILD", "dynamic SQL", sql=rendered, params=params)

        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()

    rows = [dict(r) for r in (rows or [])]
    return len(rows), rows


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
async def execute_validated_query(
    corporate_id: str, vq: ValidatedQuery
) -> Tuple[int, List[Dict[str, Any]]]:
    """Run a validated query. Returns (row_count, rows) — same shape as
    postgres_client.fetch_orders() for pipeline consistency."""
    if vq.kind == "procedure":
        return await _run_procedure(corporate_id, vq)
    return await _run_dynamic(corporate_id, vq)
