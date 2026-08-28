"""Domain-specific intent classification prompts (used when local rules miss)."""

INTENT_ORDERS_SUFFIX = """
Active domain: orders (Avaal_order).
Focus on freight orders, MRP/TORD numbers, order status, customer/geo filters, analytics.
Use intent order_lookup for specific order numbers (MRP####, TORD####).
"""

INTENT_INVOICES_SUFFIX = """
Active domain: invoices (Avaal_invoice).
Fields: InvoiceID, InvoiceNumber, InvoiceStatus, CustomerName, TotalAmount, InvoiceDate, DueDate.
Use intent invoice_lookup for a specific invoice number/id.
Use intent list_filter for invoice status questions (e.g. "give me some invoice status") — NOT lookup.
Use calculation for how many invoices or total invoice amount (sum).
Use list_filter for paid/open/overdue invoices or customer-filtered lists.
Use list_recent for recent/latest invoices.
Do NOT use order analytics intents (best customer, state-wise orders, etc.).
"""

INTENT_TRIPS_SUFFIX = """
Active domain: trips (Avaal_trip).
Fields: TripID, TripNumber, DriverName, TruckNumber, TrailerNumber, TripStatus.
Use intent trip_lookup for a specific trip number/id.
Use intent list_filter for trip status / driver / list questions — NOT lookup.
Use calculation for how many trips (count only).
Use list_filter for driver/truck/trailer/status filtered lists.
Use list_recent for recent/latest trips.
Do NOT use order-only analytics.
"""

DOMAIN_INTENT_SUFFIX = {
    "orders": INTENT_ORDERS_SUFFIX,
    "invoices": INTENT_INVOICES_SUFFIX,
    "trips": INTENT_TRIPS_SUFFIX,
}
