"""
Manual invoice ingest — not yet standalone.

Copy the pattern from scripts/ingest/order_ingest.py and adjust:
  DUPLICATE_FIELD = "InvoiceNumber" or "invoicenumber"
  metadata type = "avaal_invoice"
"""
import sys

print(
    "invoice_ingest.py is not configured yet.\n"
    "Use scripts/ingest/order_ingest.py as the template.",
    file=sys.stderr,
)
raise SystemExit(1)
