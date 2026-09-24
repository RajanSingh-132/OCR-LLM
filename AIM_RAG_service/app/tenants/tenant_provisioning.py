"""
Dynamic tenant list: Postgres (AFM_Manager.mstcompany) -> MongoDB.

1. `load_company_codes()` runs TENANT_QUERY once on AFM_Manager using the
   same PG_* settings from .env and returns the company codes.
2. `ensure_mongo_database()` makes sure a MongoDB database with that name
   exists with the Avaal_order / Avaal_trip / Avaal_invoice collections.
   A database or collection that already exists is left untouched — only
   missing ones get created.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List

import psycopg
from dotenv import load_dotenv
from pymongo.errors import CollectionInvalid

from app.mongo_client import get_mongo_client

logger = logging.getLogger("tenants.provisioning")

_SERVICE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
load_dotenv(os.path.join(_SERVICE_ROOT, ".env"))

# Postgres database that holds mstcompany.
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

TENANT_COLLECTIONS = [
    os.environ.get("AVAAL_COLLECTION_NAME", "Avaal_order"),
    os.environ.get("AVAAL_TRIPS_COLLECTION_NAME", "Avaal_trip"),
    os.environ.get("AVAAL_INVOICE_COLLECTION_NAME", "Avaal_invoice"),
]


def load_company_codes() -> List[str]:
    """Run TENANT_QUERY on AFM_Manager and return the company codes."""
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
        rows = cur.fetchall()
    # strip(): codes are stored padded with spaces (e.g. 'AFMQA   ').
    codes = [str(row[0]).strip() for row in rows if row[0] is not None]
    return [c for c in codes if c]


def ensure_mongo_database(database: str) -> Dict[str, object]:
    """Create `database` and any of TENANT_COLLECTIONS missing from it."""
    client = get_mongo_client()
    db_existed = database in client.list_database_names()
    db = client[database]
    existing = set(db.list_collection_names())

    created: List[str] = []
    for name in TENANT_COLLECTIONS:
        if name in existing:
            continue
        try:
            db.create_collection(name)
            created.append(name)
        except CollectionInvalid:
            pass  # created meanwhile by another process
    return {
        "database": database,
        "db_existed": db_existed,
        "collections_created": created,
    }
