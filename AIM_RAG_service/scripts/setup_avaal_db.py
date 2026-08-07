"""
Run once to create/verify Avaal_db collection + indexes.

  python -m scripts.setup_avaal_db
"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.order_ask.setup import setup_avaal_collection


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    status = setup_avaal_collection()
    print("\n=== Avaal DB setup ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("======================\n")
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
