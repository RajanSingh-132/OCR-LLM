"""Quick check that the PG_* settings in .env can reach PostgreSQL.

Run from the AIM_RAG_service folder:
    Windows:  .venv\\Scripts\\python test_pg_connection.py
    Linux:    .venv/bin/python test_pg_connection.py

Read-only: it only runs SELECT queries, nothing is created or changed.
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

host = os.environ.get("PG_HOST", "")
port = int(os.environ.get("PG_PORT", "5432"))
user = os.environ.get("PG_USER", "")
password = os.environ.get("PG_PASSWORD", "")
sslmode = os.environ.get("PG_SSLMODE", "prefer")
# .env has no PG_DATABASE yet; "postgres" exists on every server.
dbname = os.environ.get("PG_DATABASE", "postgres")
timeout = int(float(os.environ.get("PG_COMMAND_TIMEOUT", "10")))

print("Settings read from .env:")
print(f"  PG_HOST     = {host or '(missing)'}")
print(f"  PG_PORT     = {port}")
print(f"  PG_USER     = {user or '(missing)'}")
print(f"  PG_PASSWORD = {'(set)' if password else '(missing)'}")
print(f"  PG_SSLMODE  = {sslmode}")
print(f"  PG_DATABASE = {dbname}")
print()

if not host or not user:
    print("FAILED: PG_HOST or PG_USER is missing in .env")
    sys.exit(1)

try:
    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        sslmode=sslmode,
        connect_timeout=timeout,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_database(), current_user")
            version, current_db, current_user = cur.fetchone()
            print("SUCCESS: connected to PostgreSQL")
            print(f"  Server   : {version.split(',')[0]}")
            print(f"  Database : {current_db}")
            print(f"  User     : {current_user}")

            cur.execute(
                "SELECT datname FROM pg_database "
                "WHERE NOT datistemplate ORDER BY datname"
            )
            names = [name for (name,) in cur.fetchall()]
            print(f"\nDatabases this user can see: {len(names)}")
            print("  First 10: " + ", ".join(names[:10]))
except Exception as exc:  # noqa: BLE001
    print("FAILED: could not connect to PostgreSQL")
    print(f"  {type(exc).__name__}: {exc}")
    sys.exit(1)
