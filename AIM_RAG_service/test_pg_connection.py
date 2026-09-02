"""
Manual connection test — run this yourself to confirm the Postgres
connection, tenant DB mapping, and fn_getorders_regular call all work
BEFORE handing the rest of the migration to the agent.

Usage:
    python test_pg_connection.py <corporate_id> [order_id]

Examples:
    python test_pg_connection.py AFN01514
    python test_pg_connection.py AFN01514 57060
"""
import asyncio
import sys

# Windows defaults to ProactorEventLoop, which psycopg's async mode does
# not support. Force SelectorEventLoop on Windows before anything else runs.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.postgres_client import (
    fetch_orders,
    resolve_tenant_db_name,
    get_pool,
    close_all_pools,
)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pg_connection.py <corporate_id> [order_id]")
        sys.exit(1)

    corporate_id = sys.argv[1]
    order_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"--- Testing corporate_id={corporate_id!r} ---")

    # Step 1: tenant -> db name resolution (pure Python, no DB involved)
    try:
        db_name = resolve_tenant_db_name(corporate_id)
        print(f"[OK] Step 1/3 — resolved db_name = {db_name!r}")
    except Exception as e:
        print(f"[FAIL] Step 1/3 — resolve_tenant_db_name: {e}")
        return

    # Step 2: RAW CONNECTION ONLY — no procedure call yet. This isolates
    # whether the failure is connection-level (host/port/auth/event loop)
    # before we ever touch fn_getorders_regular.
    try:
        pool = await get_pool(db_name)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
        print(f"[OK] Step 2/3 — raw connection + SELECT 1 succeeded: {result}")
    except Exception as e:
        print(f"[FAIL] Step 2/3 — connection failed (host/port/auth/event loop issue): "
              f"{type(e).__name__}: {e}")
        await close_all_pools()
        return

    # Step 3: the actual fn_getorders_regular call — only reached if the
    # raw connection above already succeeded, so any failure here is
    # specifically about the procedure call itself, not connectivity.
    params = {}
    if order_id:
        params["p_orderid"] = order_id
        print(f"[..] Step 3/3 — calling fn_getorders_regular with p_orderid={order_id}")
    else:
        params["p_pagesize"] = 3
        print("[..] Step 3/3 — calling fn_getorders_regular with default filters, pagesize=3")

    try:
        total_count, details = await fetch_orders(corporate_id, params)
        print(f"[OK] Step 3/3 — total_count = {total_count}")
        print(f"[OK] Step 3/3 — details returned = {len(details)} row(s)")
        if details:
            first = details[0]
            print("--- First row sample fields ---")
            for key in ("ordernumber", "orderstatus", "customername", "orderid"):
                if key in first:
                    print(f"  {key}: {first[key]}")
        else:
            print("[WARN] No rows returned — check filters or that this db has data")
    except Exception as e:
        print(f"[FAIL] Step 3/3 — fn_getorders_regular call failed "
              f"(connection was fine, this is a procedure/SQL-level error): "
              f"{type(e).__name__}: {e}")
    finally:
        await close_all_pools()


if __name__ == "__main__":
    asyncio.run(main())










