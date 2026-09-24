"""Test the dynamic tenant flow for orders — ONE pass, then exit.

  1. Run TENANT_QUERY (app/tenants/tenant_provisioning.py, LIMIT 10) on
     Postgres AFM_Manager -> company codes.
  2. For each code, create the MongoDB database of the same name with
     Avaal_order / Avaal_trip / Avaal_invoice (skipped if it already exists).
  3. For each ready tenant, one by one: call /api/Order/listorder with that
     corporateid, embed, and store into <db>.Avaal_order.

This WRITES to MongoDB (creates databases/collections, inserts orders) and
calls Bedrock for embeddings — same as one cycle of order_live_api.py.

Run from the AIM_RAG_service folder:
    Windows:  .venv\\Scripts\\python test_order_dynamic_tenants.py
    Linux:    .venv/bin/python test_order_dynamic_tenants.py
"""

import logging
import os
import sys

SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from app.sync.order_live_api import _db_name, prepare_tenants, run_one_cycle  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    print("\n=== Step 1 + 2: Postgres tenant list -> MongoDB databases ===")
    try:
        tenants = prepare_tenants()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to load tenants from Postgres: {type(exc).__name__}: {exc}")
        return 1
    print(f"{len(tenants)} tenants ready: {tenants}")

    print("\n=== Step 3: listorder API -> MongoDB, one tenant at a time ===")
    rows = []
    for code in tenants:
        try:
            status = run_one_cycle(code)
            rows.append((
                code,
                _db_name(code),
                "OK",
                status.get("records_received", "-"),
                status.get("documents_inserted", 0),
                status.get("note", ""),
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append((code, _db_name(code), "FAILED", "-", "-", f"{type(exc).__name__}: {exc}"[:80]))

    print(f"\n{'Tenant':<10} {'Mongo DB':<12} {'Result':<7} {'Fetched':>7} {'Inserted':>8}  Note")
    print("-" * 70)
    for code, db, result, fetched, inserted, note in rows:
        print(f"{code:<10} {db:<12} {result:<7} {fetched!s:>7} {inserted!s:>8}  {note}")

    failed = sum(1 for r in rows if r[2] == "FAILED")
    print(f"\nDone: {len(rows) - failed} OK, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
