"""
Ingest all records from orderdata.txt into Mongo Avaal_db.

  python -m scripts.ingest_avaal_orders
"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.order_ask.ingest import ingest_avaal_orders


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status = ingest_avaal_orders(with_embeddings=True, replace_namespace=True)
    print("\n=== Avaal ingest ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("====================\n")
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
