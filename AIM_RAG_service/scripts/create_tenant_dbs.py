"""
Create one Mongo database per tenant, each with the 3 Avaal collections
(Avaal_order, Avaal_trip, Avaal_invoice) and the same indexes the live sync
daemons (app/sync/*_live_api.py) create.

  python scripts/create_tenant_dbs.py                  # uses DB_NAMES below
  python scripts/create_tenant_dbs.py AFN01900 ABC01   # only these names

Safe to run again: an existing database / collection / index is left as it
is (nothing is dropped or emptied). MONGO_URI is read from .env.

Note: the database is created with EXACTLY the name given. The sync daemons
and /ask map AFMQA to "AFMQA" (DB_NAME_OVERRIDES / _CORPORATE_DB_OVERRIDES),
so a database named "AFMQA" is not used by them unless that override is removed.
"""
import os
import re
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.mongo_client import get_mongo_client

# ===================== CONFIG — edit these =====================
# Add a new tenant here, then run this script.
DB_NAMES: List[str] = ["AFN01800", "AFN01807", "AFN01808", "AFN01809", "AFN01812", "AFN01813", "AFN01817", "AFN01818", "AFN01823"]

# collection -> (namespace, id field, duplicate field), matching the sync daemons.
COLLECTIONS: Dict[str, Tuple[str, str, str]] = {
    "Avaal_order": ("avaal_orders", "orderid", "ordernumber"),
    "Avaal_trip": ("avaal_trips", "tripid", "tripnumber"),
    "Avaal_invoice": ("avaal_invoices", "metadata.invoiceid", "metadata.invoicenumber"),
}
# ===============================================================

# Same rule as app/tenants/mapping.py: letters, digits, underscore, hyphen; max 63.
_DB_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,63}$")


def _index_specs(id_field: str, duplicate_field: str) -> List[List[Tuple[str, int]]]:
    return [
        [("namespace", 1)],
        [("namespace", 1), ("metadata.source_document", 1)],
        [("namespace", 1), (id_field, 1)],
        [("namespace", 1), (duplicate_field, 1)],
    ]


def create_tenant_db(client, db_name: str) -> Dict[str, str]:
    db = client[db_name]
    existing = set(db.list_collection_names())
    status: Dict[str, str] = {}

    for name, (_namespace, id_field, duplicate_field) in COLLECTIONS.items():
        if name in existing:
            status[name] = "already exists"
        else:
            db.create_collection(name)
            status[name] = "created"

        collection = db[name]
        for keys in _index_specs(id_field, duplicate_field):
            collection.create_index(keys)

    return status


def main(argv: List[str]) -> int:
    names = [n.strip() for n in (argv or DB_NAMES) if n.strip()]
    if not names:
        print("No database names given.")
        return 1

    invalid = [n for n in names if not _DB_NAME_RE.fullmatch(n)]
    if invalid:
        print(f"Invalid database name(s): {invalid} "
              "(use letters, numbers, underscore, hyphen only, max 63 chars)")
        return 1

    client = get_mongo_client()
    client.admin.command("ping")
    before = set(client.list_database_names())

    print("\n=== Tenant DB setup ===")
    for db_name in names:
        db_state = "already exists" if db_name in before else "created"
        print(f"\n[{db_name}] database {db_state}")
        for collection, state in create_tenant_db(client, db_name).items():
            print(f"  {collection}: {state} (indexes ok)")
    print("\n=======================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
