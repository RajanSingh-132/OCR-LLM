"""Test: which MongoDB database names would the tenant query produce?

Runs the tenant query on AFM_Manager (same PG_* settings from .env) and
prints, for every company code returned:
  - the MongoDB database name it maps to (same name as the company code)
  - the 3 collections that would live inside it
  - whether that database / collections already exist in MongoDB

Read-only: nothing is created in Postgres or MongoDB.

Run from the AIM_RAG_service folder:
    Windows:  .venv\\Scripts\\python test_tenant_db_names.py
    Linux:    .venv/bin/python test_tenant_db_names.py
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SERVICE_ROOT, ".env"))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

# Database that holds mstcompany.
MANAGER_DB = "AFM_Manager"

TENANT_QUERY = """
SELECT companycode
FROM mstcompany
WHERE activeyn = 'Y'
  AND companycode NOT IN (
      'AFMDATH',
      'AFMDemo ',
      'AFMDEMOA',
      'AFMQAA',
      'AFMQAM',
      'AFMQAM1'
  )
ORDER BY companycode
LIMIT 50;
"""

COLLECTIONS = [
    os.environ.get("AVAAL_COLLECTION_NAME", "Avaal_order"),
    os.environ.get("AVAAL_TRIPS_COLLECTION_NAME", "Avaal_trip"),
    os.environ.get("AVAAL_INVOICE_COLLECTION_NAME", "Avaal_invoice"),
]


def load_company_codes() -> list[str]:
    with psycopg.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", "5432")),
        user=os.environ["PG_USER"],
        password=os.environ.get("PG_PASSWORD", ""),
        sslmode=os.environ.get("PG_SSLMODE", "prefer"),
        dbname=MANAGER_DB,
        connect_timeout=15,
    ) as conn, conn.cursor() as cur:
        cur.execute(TENANT_QUERY)
        # strip(): some codes are stored with trailing spaces (e.g. 'AFMDemo ').
        return [row[0].strip() for row in cur.fetchall() if row[0] and row[0].strip()]


def load_mongo_state() -> dict[str, set[str]] | None:
    """{db_name: {collection names}} for every Mongo DB, or None if Mongo is unreachable."""
    try:
        from app.mongo_client import get_mongo_client

        client = get_mongo_client()
        return {
            name: set(client[name].list_collection_names())
            for name in client.list_database_names()
        }
    except Exception as exc:  # noqa: BLE001
        print(f"(MongoDB check skipped: {type(exc).__name__}: {exc})\n")
        return None


def main() -> int:
    try:
        codes = load_company_codes()
    except Exception as exc:  # noqa: BLE001
        print("FAILED: could not run the tenant query on PostgreSQL")
        print(f"  {type(exc).__name__}: {exc}")
        return 1

    print(f"Query returned {len(codes)} company codes from {MANAGER_DB}.mstcompany\n")
    if not codes:
        return 0

    mongo = load_mongo_state()

    header = f"{'#':>3}  {'Mongo DB name':<15} {'DB exists':<10} collections (Y = already exists)"
    print(header)
    print("-" * len(header))
    for i, code in enumerate(codes, 1):
        if mongo is None:
            db_state, coll_state = "?", ", ".join(COLLECTIONS)
        else:
            existing = mongo.get(code)
            db_state = "yes" if existing is not None else "NO"
            coll_state = ", ".join(
                f"{c}[{'Y' if existing and c in existing else 'N'}]" for c in COLLECTIONS
            )
        print(f"{i:>3}  {code:<15} {db_state:<10} {coll_state}")

    if mongo is not None:
        missing = [c for c in codes if c not in mongo]
        print(f"\nSummary: {len(codes) - len(missing)} DBs already in MongoDB, "
              f"{len(missing)} would be newly created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
