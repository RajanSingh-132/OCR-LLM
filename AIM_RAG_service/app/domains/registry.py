from __future__ import annotations

from typing import Dict

from app.domains.models import DomainProfile

DOMAINS: Dict[str, DomainProfile] = {
    "orders": DomainProfile(
        name="orders",
        label="Order",
        keywords=(
            r"\borders?\b",
            r"\bordernumber\b",
            r"\bfreight\b",
            r"\bdispatch",
            r"\bconfirmed\b",
            r"\bquoted\b",
            r"\bmrp\d+",
            r"\btord\d+",
            r"\bprofit(?:able|ability)?\b",
            r"\bmargin\b",
            r"\blos(?:e|es|ing|s)\b[^.?!]{0,20}\bmoney\b",
        ),
        strong_keywords=(
            r"\borders?\b",
            r"\bordernumber\b",
            r"\bmrp\d+",
            r"\btord\d+",
            # Profitability is an orders-only analysis in this system (revenue
            # = grosstotalfreight, freight = totalfreight — both order fields)
            # — force it to always route to the orders domain deterministically
            # instead of falling through to the LLM domain-classifier, which
            # would otherwise decide this on a question with no order/trip/
            # invoice keyword at all (e.g. "most profitable customer").
            r"\bprofit(?:able|ability)?\b",
            r"\bmargin\b",
            r"\blos(?:e|es|ing|s)\b[^.?!]{0,20}\bmoney\b",
        ),
        id_fields=("orderid",),
        number_fields=("ordernumber", "tempordernumber"),
        sort_fields=("orderid", "orderdate", "pickupdate", "totalfreight", "grosstotalfreight"),
        default_sort="orderid",
        list_fields=(
            "orderid",
            "ordernumber",
            "customername",
            "orderstatus",
            "accountingstatus",
            "outstatus",
            "statuscode",
            "currencycode",
            "totalfreight",
            "taxes",
            "orderdate",
            "pickuplocationname",
            "pickupfulladdress",
            "deliverylocationname",
            "deliveryfulladdress",
        ),
    ),
    "invoices": DomainProfile(
        name="invoices",
        label="Invoice",
        keywords=(
            r"\binvoices?\b",
            r"\binvoicenumber\b",
            r"\binvoiceid\b",
            r"\bpaid\b",
            r"\bopen\b",
            r"\bdue\s*date\b",
            r"\bbilling\b",
            r"\btotalamount\b",
        ),
        strong_keywords=(
            r"\binvoices?\b",
            r"\binvoicenumber\b",
            r"\b(?:MR|INO|AIN|UGY)\d+",
        ),
        id_fields=("InvoiceID", "invoiceid"),
        number_fields=("InvoiceNumber", "invoicenumber"),
        sort_fields=("InvoiceID", "InvoiceDate", "DueDate", "TotalAmount"),
        default_sort="InvoiceID",
        list_fields=(
            "InvoiceID",
            "InvoiceNumber",
            "CustomerName",
            "InvoiceStatus",
            "TotalAmount",
            "PreTaxAmount",
            "freightcharges",
            "othercharges",
            "outstandinamount",
            "CurrencyCode",
            "ExchangeRate",
            "InvoiceDate",
            "DueDate",
            "CompanyName",
            "InvoiceOrderNumbers",
            "commodityname",
            "pickuplocation",
            "deliverylocation",
        ),
    ),
    "trips": DomainProfile(
        name="trips",
        label="Trip",
        keywords=(
            r"\btrips?\b",
            r"\btripno\b",
            r"\btrip\s*no\b",
            r"\btripnumber\b",
            r"\btripid\b",
            r"\bdriver\b",
            r"\btruck\b",
            r"\btrailer\b",
            r"\btotalloaddistance\b",
            r"\b(?:ETP|TRO|TRP|TRIP)[A-Za-z0-9-]{2,}\b",
        ),
        strong_keywords=(
            r"\btrips?\b",
            r"\btripnumber\b",
            r"\btripno\b",
            r"\b(?:ETP|TRO|TRP|TRIP)[A-Za-z0-9-]{2,}\b",
        ),
        id_fields=("tripid", "TripID", "id"),
        number_fields=("tripnumber", "TripNumber", "tripno", "TripNo"),
        sort_fields=(
            "tripid",
            "TripID",
            "totalloaddistance",
            "triptotaldistance",
            "firstpickupdate",
            "lastdeliverydate",
            "_id",
        ),
        default_sort="tripid",
        list_fields=(
            "tripid",
            "tripnumber",
            "tripstatus",
            "triptype",
            "firstdrivername",
            "firstdriverphone",
            "seconddrivername",
            "seconddriverphone",
            "trucknumber",
            "customername",
            "companyname",
            "commodity",
            "salesmannames",
            "pickuplocationname",
            "pickupcity",
            "pickupcountry",
            "firstpickupdate",
            "deliverylocationname",
            "deliverycity",
            "deliverycountry",
            "lastdeliverydate",
            "totalloaddistance",
            "triptotaldistance",
            "distanceunit",
        ),
    ),
}

DEFAULT_DOMAIN = "orders"

# Words that don't name a record type but point at the domains that hold the
# data: (domains that can answer, pattern). rag_engine._pick_domain uses these
# ONLY when the question names no record type at all ("driver count per
# company", "which currency does X use") — an explicit "order" / "trip" /
# "invoice" always decides first. On a match the payload domain is kept when
# it is listed, else the first listed domain is used. First match wins:
# currency outranks drivers because trips have no currency field.
ROUTING_HINTS = (
    (("orders", "invoices"), r"\b(?:usd|cad|currency|currencies)\b"),
    (("trips",), r"\b(?:drivers?|trucks?|trailers?|miles?|mileage|distance)\b"),
)

# Request payload `collection` value -> domain (which picks the Mongo
# collection via app/tenants/mapping.py DEFAULT_COLLECTIONS):
#   order -> Avaal_order, trip -> Avaal_trip, invoice -> Avaal_invoice.
# Add payment / receipt here once their domains exist.
COLLECTION_TO_DOMAIN: Dict[str, str] = {
    "order": "orders",
    "trip": "trips",
    "invoice": "invoices",
}


def resolve_collection(collection: str) -> str:
    """Map a payload `collection` value ("order", "Orders", ...) to its domain.
    Raises ValueError for anything not in COLLECTION_TO_DOMAIN."""
    key = (collection or "").strip().lower()
    if key in COLLECTION_TO_DOMAIN:
        return COLLECTION_TO_DOMAIN[key]
    if key.endswith("s") and key[:-1] in COLLECTION_TO_DOMAIN:
        return COLLECTION_TO_DOMAIN[key[:-1]]
    allowed = ", ".join(COLLECTION_TO_DOMAIN)
    raise ValueError(f"Invalid collection {collection!r}. Use one of: {allowed}.")


def domain_to_collection(domain: str) -> str | None:
    """Inverse of resolve_collection: "trips" -> "trip" (the response `related`)."""
    for key, value in COLLECTION_TO_DOMAIN.items():
        if value == domain:
            return key
    return None


def get_domain_profile(domain: str) -> DomainProfile:
    key = (domain or DEFAULT_DOMAIN).lower()
    if key not in DOMAINS:
        raise KeyError(f"Unknown domain: {domain!r}")
    return DOMAINS[key]
