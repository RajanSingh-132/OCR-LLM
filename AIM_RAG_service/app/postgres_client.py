"""
Postgres client — live order data source.

Uses psycopg3 (async) instead of asyncpg: psycopg ships prebuilt binary
wheels for Windows/newer Python versions, avoiding the need for
Microsoft C++ Build Tools that asyncpg's source build requires.

Replaces the Mongo "Avaal_order" collection as the source of truth for
order data. Session memory (app/order_ask/memory.py) is UNCHANGED and
stays on Mongo — this module only handles the order-data path.

Deployment shape assumed (per current testing setup):
    - ONE Postgres server
    - MULTIPLE databases on it (one per tenant/corporate_id)
    - Same credentials/host/port work for every database, only the
      database name changes per tenant.

Env vars expected (add to .env, do NOT hardcode):
    PG_HOST=...
    PG_PORT=5432
    PG_USER=...
    PG_PASSWORD=...
    PG_SSLMODE=require        # or "disable" for local testing

Per-tenant DB name resolution reuses the same pattern as
app/tenants/mapping.py (corporate_id -> db name), see
resolve_tenant_db_name() below — wire this to the real mapping once
available.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

logger = logging.getLogger("postgres_client")

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
load_dotenv(os.path.join(_SERVICE_ROOT, ".env"))

PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_SSLMODE = os.environ.get("PG_SSLMODE", "require")

# Pool sizing — tune once you have real concurrency numbers.
PG_POOL_MIN = int(os.environ.get("PG_POOL_MIN", "1"))
PG_POOL_MAX = int(os.environ.get("PG_POOL_MAX", "5"))
PG_COMMAND_TIMEOUT = float(os.environ.get("PG_COMMAND_TIMEOUT", "10"))  # seconds

# One pool PER DATABASE NAME (not per corporate_id directly — several
# corporate_ids could theoretically share a db name, mirrors the Mongo
# override pattern in tenants/mapping.py).
_pools: Dict[str, AsyncConnectionPool] = {}


class PostgresConfigError(RuntimeError):
    pass


class TenantNotProvisionedError(RuntimeError):
    """Raised when a corporate_id has no known Postgres database yet."""

    def __init__(self, corporate_id: str):
        self.corporate_id = corporate_id
        super().__init__(f"No Postgres database mapped for corporate_id={corporate_id!r}")


def _require_config() -> None:
    missing = [
        name
        for name, val in (
            ("PG_HOST", PG_HOST),
            ("PG_USER", PG_USER),
            ("PG_PASSWORD", PG_PASSWORD),
        )
        if not val
    ]
    if missing:
        raise PostgresConfigError(
            f"Postgres config missing: {', '.join(missing)}. Set these in .env."
        )


def _conninfo(db_name: str) -> str:
    sslmode = PG_SSLMODE if PG_SSLMODE else "require"
    return (
        f"host={PG_HOST} port={PG_PORT} dbname={db_name} "
        f"user={PG_USER} password={PG_PASSWORD} sslmode={sslmode}"
    )


# ---------------------------------------------------------------------------
# Tenant -> database name resolution
#
# TODO: replace this stub with the real 600-tenant mapping once available.
# For now, testing set of 3-4 known db names, same pattern as
# tenants/mapping.py's _CORPORATE_DB_OVERRIDES.
# ---------------------------------------------------------------------------
_TEST_TENANT_DB_MAP: Dict[str, str] = {
    "AFN01801": "AFN01801",
    "AFN01992": "AFN01992",
    "AFN01514": "AFN01514",
    "AFMQA"   : "AFMQA"
}


def resolve_tenant_db_name(corporate_id: str) -> str:
    """Map a corporate_id to its Postgres database name.

    Replace this with the real lookup (config table / service) once
    available. Raises TenantNotProvisionedError if unknown, so callers
    can surface the same friendly "still provisioning" message used for
    Mongo (see tenants.mapping.pending_tenant_user_message).
    """
    cid = (corporate_id or "").strip().upper()
    db_name = _TEST_TENANT_DB_MAP.get(cid)
    if not db_name:
        raise TenantNotProvisionedError(corporate_id)
    return db_name


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------
async def get_pool(db_name: str) -> AsyncConnectionPool:
    """Get (or lazily create) a connection pool for one database on the
    shared server. Pools are cached per db_name for the life of the process.
    """
    _require_config()
    pool = _pools.get(db_name)
    if pool is not None:
        return pool

    logger.info("postgres: creating pool for db=%s host=%s", db_name, PG_HOST)
    pool = AsyncConnectionPool(
        conninfo=_conninfo(db_name),
        min_size=PG_POOL_MIN,
        max_size=PG_POOL_MAX,
        timeout=PG_COMMAND_TIMEOUT,
        open=False,
    )
    await pool.open()
    _pools[db_name] = pool
    return pool


async def close_all_pools() -> None:
    """Call on app shutdown."""
    for db_name, pool in list(_pools.items()):
        await pool.close()
        logger.info("postgres: closed pool for db=%s", db_name)
    _pools.clear()


# Statement timeout (ms) for every order-data query — a slow aggregate must
# never hang a request. Tune via env if needed.
PG_STATEMENT_TIMEOUT_MS = int(os.environ.get("PG_STATEMENT_TIMEOUT_MS", "45000"))


@asynccontextmanager
async def readonly_connection(db_name: str):
    """Yield a connection pinned READ ONLY with a statement timeout.

    Used for the dynamic query builder (LLM-shaped SQL structure) and for
    the pure-read functions (aggregate / summary). A stray
    INSERT/UPDATE/DELETE/DDL — bug or crafted question — is refused by
    Postgres itself ("cannot execute ... in a read-only transaction").
    Fails closed: if the session can't be set read-only, nothing runs.
    """
    pool = await get_pool(db_name)
    async with pool.connection() as conn:
        await conn.set_read_only(True)          # raises -> caller aborts
        await conn.set_autocommit(False)
        # SET does not take bind params; the value is an int we cast ourselves.
        _timeout_ms = int(PG_STATEMENT_TIMEOUT_MS)
        async with conn.cursor() as _cur:
            await _cur.execute(f"SET statement_timeout = {_timeout_ms}")
        if not conn.read_only:                  # paranoia: confirm it took
            raise RuntimeError("could not pin connection to READ ONLY")
        try:
            yield conn
        finally:
            try:
                await conn.rollback()            # never commit anything
                await conn.set_read_only(False)  # clean state for next borrower
            except Exception:
                logger.debug("readonly_connection: cleanup on return failed")


@asynccontextmanager
async def guarded_connection(db_name: str):
    """Statement-timeout + never-commit connection for the two vetted
    ``fn_getorders_regular[_ai]`` functions ONLY.

    Those functions build a session-local pagination scratch table
    (``CREATE TEMP TABLE page ON COMMIT DROP`` + ``DROP TABLE IF EXISTS
    page``), so Postgres refuses to run them inside a READ ONLY
    transaction. No LLM-shaped SQL ever reaches here — only these two
    functions, called with parameter-bound args. The transaction is always
    rolled back, so nothing (not even the temp table) persists, and the
    functions themselves never touch a business table.
    """
    pool = await get_pool(db_name)
    async with pool.connection() as conn:
        await conn.set_autocommit(False)
        _timeout_ms = int(PG_STATEMENT_TIMEOUT_MS)
        async with conn.cursor() as _cur:
            await _cur.execute(f"SET statement_timeout = {_timeout_ms}")
        try:
            yield conn
        finally:
            try:
                await conn.rollback()            # never commit anything
            except Exception:
                logger.debug("guarded_connection: rollback on return failed")


# ---------------------------------------------------------------------------
# fn_getorders_regular caller
# ---------------------------------------------------------------------------

# Full param order per the function signature you provided. Every param
# has a SQL-side default, so callers only need to pass what the question
# actually needs — everything else stays None/default.
_ORDER_FN_PARAM_ORDER = [
    "p_dataviewtype", "p_companycode", "p_ordernumber", "p_customercode",
    "p_salesmancode", "p_customerordernumber", "p_orderstatus", "p_vinnumber",
    "p_shipmenttype", "p_pickuplocation", "p_deliverylocation", "p_pickupcity",
    "p_deliverycity", "p_isdate", "p_datetype", "p_fromdate", "p_todate",
    "p_pickupstatecode", "p_deliverystatecode", "p_status", "p_currencycode",
    "p_csa", "p_hazmat", "p_overdimension", "p_searchvalue", "p_orderformtype",
    "p_equipmenttype", "p_usertype", "p_username", "p_pageno", "p_pagesize",
    "p_sortcolumn", "p_sortorder", "p_accountingstatus", "p_statuscondition",
    "p_pickupcountrycode", "p_deliverycountrycode", "p_usercode",
    "p_orderoutid", "p_carriercode", "p_tripnumber", "p_pickuprefnum",
    "p_pendingaccessorial", "p_orderid",
]

_ORDER_FN_DEFAULTS: Dict[str, Any] = {
    "p_dataviewtype": "D",
    "p_companycode": "A",
    "p_isdate": False,
    "p_status": "A",
    "p_currencycode": "A",
    "p_searchvalue": "",
    "p_pageno": 1,
    "p_pagesize": 10,
    "p_sortcolumn": "ModifiedOn",
    "p_sortorder": "DESC",
    "p_orderoutid": -1,
    "p_pendingaccessorial": False,
    "p_orderid": None,
}


async def fetch_orders(
    corporate_id: str,
    params: Optional[Dict[str, Any]] = None,
    fn_name: str = "fn_getorders_regular",
) -> Tuple[int, List[Dict[str, Any]]]:
    """Call fn_getorders_regular (or a signature-compatible variant) for
    one tenant.

    `params` should only contain the keys the caller actually wants to
    override (produced by the entities->params mediator, see Phase 2).
    Everything else falls back to the SQL function's own defaults.

    `fn_name` lets the dynamic query builder target `fn_getorders_regular_ai`
    (same 45-param signature, per the contract) without a second copy of
    this plumbing.

    Returns (total_count, details) — details is already a list of dicts
    (jsonb decoded), matching the shape the rest of the pipeline expects
    from the old Mongo path.
    """
    if fn_name not in ("fn_getorders_regular", "fn_getorders_regular_ai"):
        raise ValueError(f"fetch_orders: unexpected fn_name {fn_name!r}")

    db_name = resolve_tenant_db_name(corporate_id)

    merged: Dict[str, Any] = dict(_ORDER_FN_DEFAULTS)
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})

    # Direct-PK fast path: if p_orderid is set, force page 1/size 1 —
    # mirrors the function's own internal short-circuit, kept explicit
    # here too so callers get the cheap path even before hitting SQL.
    if merged.get("p_orderid"):
        merged["p_pageno"] = 1
        merged["p_pagesize"] = 1

    args = [merged.get(name) for name in _ORDER_FN_PARAM_ORDER]

    placeholders = ", ".join(["%s"] * len(args))
    query = f"""
        SELECT total_count, details
        FROM public.{fn_name}({placeholders})
    """

    # fn_getorders_regular[_ai] needs a session temp table -> guarded_connection
    # (timeout + always-rollback), not readonly_connection.
    async with guarded_connection(db_name) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, args)
            row = await cur.fetchone()

    if row is None:
        return 0, []

    total_count = row["total_count"] or 0
    details_raw = row["details"]
    # psycopg decodes jsonb to Python objects automatically in most
    # setups, but handle the raw-string case defensively too.
    if isinstance(details_raw, str):
        details = json.loads(details_raw) if details_raw else []
    else:
        details = details_raw or []

    return total_count, details


# ---------------------------------------------------------------------------
# Aggregate _ai functions — NOT DEPLOYED YET (see BUCKET1_PLAN / the _ai
# contract). These wrappers are written against the agreed signature so the
# dynamic query builder needs no rework once the CREATE FUNCTION statements
# are run in pgAdmin. Callers must gate on AVAAL_AI_PROCEDURES_LIVE.
# ---------------------------------------------------------------------------
_AGG_FN_PARAM_ORDER = [
    "p_companycode", "p_isdate", "p_datetype", "p_fromdate", "p_todate",
    "p_orderstatus", "p_pickupcountrycode", "p_deliverycountrycode",
    "p_status", "p_groupby",
]
_AGG_FN_DEFAULTS: Dict[str, Any] = {
    "p_companycode": "A",
    "p_isdate": False,
    "p_status": "A",
}
_AGG_FN_NAMES = {
    "count": "fn_getorders_aggregate_count_ai",
    "freight": "fn_getorders_aggregate_freight_ai",
}
_AGG_FN_VALUE_COL = {
    "count": "order_count",
    "freight": "total_freight",
}


async def fetch_orders_aggregate(
    corporate_id: str,
    kind: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Call fn_getorders_aggregate_count_ai / _freight_ai for one tenant.

    `kind` is "count" or "freight". Returns a list of
    {"group_key": <str|None>, "value": <number>} rows, already sorted
    descending by value (the function does that server-side).
    """
    if kind not in _AGG_FN_NAMES:
        raise ValueError(f"fetch_orders_aggregate: bad kind {kind!r}")

    db_name = resolve_tenant_db_name(corporate_id)

    merged: Dict[str, Any] = dict(_AGG_FN_DEFAULTS)
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})
    args = [merged.get(name) for name in _AGG_FN_PARAM_ORDER]

    fn_name = _AGG_FN_NAMES[kind]
    value_col = _AGG_FN_VALUE_COL[kind]
    placeholders = ", ".join(["%s"] * len(args))
    query = f"SELECT group_key, {value_col} AS value FROM public.{fn_name}({placeholders})"

    async with readonly_connection(db_name) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, args)
            rows = await cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        out.append({"group_key": r.get("group_key"), "value": r.get("value")})
    return out


_SUMMARY_FN_PARAM_ORDER = [
    "p_companycode", "p_isdate", "p_datetype", "p_fromdate", "p_todate",
    "p_orderstatus", "p_pickupcountrycode", "p_deliverycountrycode",
    "p_status", "p_limit",
]
_SUMMARY_FN_DEFAULTS: Dict[str, Any] = {
    "p_companycode": "A",
    "p_isdate": False,
    "p_status": "A",
    "p_limit": 25,
}
_SUMMARY_FN_COLUMNS = [
    "ordernumber", "orderstatus", "customername", "pickuplocation",
    "deliverylocation", "orderdate", "totalfreight", "ordernotes",
]


async def fetch_orders_summary(
    corporate_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Call fn_getorders_summary_ai for one tenant — a filtered list of
    orders with a fixed, user-friendly column set (no detail blob).

    Returns a list of row dicts (see _SUMMARY_FN_COLUMNS).
    """
    db_name = resolve_tenant_db_name(corporate_id)

    merged: Dict[str, Any] = dict(_SUMMARY_FN_DEFAULTS)
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})
    args = [merged.get(name) for name in _SUMMARY_FN_PARAM_ORDER]

    placeholders = ", ".join(["%s"] * len(args))
    cols = ", ".join(_SUMMARY_FN_COLUMNS)
    query = f"SELECT {cols} FROM public.fn_getorders_summary_ai({placeholders})"

    async with readonly_connection(db_name) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, args)
            rows = await cur.fetchall()

    return [dict(r) for r in (rows or [])]
