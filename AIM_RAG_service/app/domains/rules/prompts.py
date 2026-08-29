"""Domain-specific intent classification prompts (used when local rules miss)."""

INTENT_ORDERS_SUFFIX = """
Active domain: orders (Avaal_order).
Focus on freight orders, MRP/TORD numbers, status filters, customer/geo filters, analytics.

ORDER STATUS (orderstatus): Quoted, Confirmed, Dispatched, Started, In-Transit,
  Partially Delivered, Delivered, Cancelled, Rejected.
OUTSOURCE STATUS (outstatus): Open, Planned, Assigned, Quoted, Delivered.
ACCOUNTING STATUS (accountingstatus): Invoiced, PartiallyPaid, Paid, Restricted.

Use intent order_lookup for specific order numbers (MRP####, TORD####).
Use intent list_filter for status-filtered lists (confirmed / invoiced / outsource assigned / …).
Use intent analytics for status counts (how many Confirmed / Invoiced / Assigned / …).
Do NOT map Invoiced/Paid to orderstatus — those are accountingstatus.
"""

INTENT_INVOICES_SUFFIX = """
Active domain: invoices (Avaal_invoice).
Statuses: Paid, Open, PartiallyPaid, BadDebt, OverDue.
Fields: InvoiceID, InvoiceNumber, InvoiceStatus, CustomerName, CompanyName,
TotalAmount, PreTaxAmount, freightcharges, othercharges, outstandinamount,
ExchangeRate, InvoiceDate, DueDate, pickuplocation, deliverylocation,
InvoiceOrderNumbers, commodityname.

Use intent invoice_lookup for a specific invoice number/id (MR####, INO#, AIN####, …).
Use intent list_filter for status / customer / company filtered lists.
Use intent list_recent for recent/latest/some invoices (some => max 10).
Use intent analytics for:
  - status counts (how many Paid / BadDebt / …)
  - best/worst invoice by amount (Paid preferred for best)
  - country-wise invoice counts
  - customer with most/least invoices
  - last week/month invoice counts (optional status)
  - due next week invoices
Use calculation for total invoice amount sum when asked.
Do NOT use order-only analytics.
"""

INTENT_TRIPS_SUFFIX = """
Active domain: trips (Avaal_trip).
Real fields include: tripid, tripnumber, tripstatus, triptype,
firstdrivername/phone, seconddrivername/phone, trucknumber,
customername, commodity, salesmannames,
pickuplocationname, pickupfulladdress, pickupcity, pickupcountry, firstpickupdate,
deliverylocationname, deliveryfulladdress, deliverycity, deliverycountry, lastdeliverydate,
totalloaddistance, triptotaldistance, distanceunit.

TRIP STATUS (tripstatus): Planned, Dispatched, Started, In-Transit, Delivered, Rejected.

Use intent trip_lookup for a specific trip number/id (ETP####, TRO####, tripid).
Use intent list_filter for status / driver / customer / type / country filtered lists
  (e.g. list Planned trips, Dispatched trips, In-Transit trips).
Use intent list_recent for recent/latest trips.
Use intent analytics for:
  - status counts (how many Planned / Dispatched / Started / In-Transit / Delivered / Rejected)
  - best trip (highest distance) / worst trip (lowest distance)
  - country-wise total trips
Use calculation for how many trips (count only).
Do NOT use order-only analytics (best customer, city-wise orders).
"""

DOMAIN_INTENT_SUFFIX = {
    "orders": INTENT_ORDERS_SUFFIX,
    "invoices": INTENT_INVOICES_SUFFIX,
    "trips": INTENT_TRIPS_SUFFIX,
}
