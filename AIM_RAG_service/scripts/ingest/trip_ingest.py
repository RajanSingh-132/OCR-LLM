"""
Manual trip ingest — not yet standalone.

Copy the pattern from scripts/ingest/order_ingest.py and adjust:
  DUPLICATE_FIELD = "tripnumber"
  metadata type = "avaal_trip"
"""
import sys

print(
    "trip_ingest.py is not configured yet.\n"
    "Use scripts/ingest/order_ingest.py as the template.",
    file=sys.stderr,
)
raise SystemExit(1)
