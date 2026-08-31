"""Domain-specific intent classification prompts (used when local rules miss)."""

INTENT_ORDERS_SUFFIX = """
Active domain: orders (Avaal_order).

Always fill filters when the user mentions location or status:
- city / state / country / pin → filters.city|state|country + location_side
- orderstatus / accountingstatus / outstatus → corresponding filter
- "summary" / "how many" / "city wise" / "state wise" → intent=analytics, needs_analytics=true
- specific order number/id → order_lookup, needs_exact_order=true, full details path
- filtered list without count → list_filter
- recent/latest/only N without geo → list_recent

ORDER STATUS (orderstatus): Quoted, Confirmed, Dispatched, Started, In-Transit,
  Partially Delivered, Delivered, Cancelled, Rejected.
OUTSOURCE STATUS (outstatus): Open, Planned, Assigned, Quoted, Delivered.
ACCOUNTING STATUS (accountingstatus): Invoiced, PartiallyPaid, Paid, Restricted.
Do NOT map Invoiced/Paid to orderstatus — those are accountingstatus.
Typos: ontorio→ON/Ontario, confrm→Confirmed.
"""

INTENT_INVOICES_SUFFIX = """
Active domain: invoices (Avaal_invoice).
Statuses: Paid, Open, PartiallyPaid, BadDebt, OverDue.

Fill filters for status / customer / company / location when present.
invoice_lookup for a specific invoice number/id (full details).
list_filter for filtered lists; list_recent for recent/some.
analytics for status counts, country-wise, best/worst, due next week, period.
calculation for total amount sum.
Do NOT use order-only analytics.
"""

INTENT_TRIPS_SUFFIX = """
Active domain: trips (Avaal_trip).
TRIP STATUS: Planned, Dispatched, Started, In-Transit, Delivered, Rejected.

Fill filters for tripstatus / driver / customer / pickupcity / deliverycity / country.
trip_lookup for a specific trip number/id (full details).
list_filter for filtered lists; list_recent for recent.
analytics for status counts, best/worst distance, country-wise trips.
Do NOT use order-only analytics (best customer, city-wise orders).
"""

DOMAIN_INTENT_SUFFIX = {
    "orders": INTENT_ORDERS_SUFFIX,
    "invoices": INTENT_INVOICES_SUFFIX,
    "trips": INTENT_TRIPS_SUFFIX,
}
