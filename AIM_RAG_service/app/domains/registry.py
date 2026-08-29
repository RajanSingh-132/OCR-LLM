from __future__ import annotations

from typing import Dict, List

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
        ),
        strong_keywords=(r"\borders?\b", r"\bordernumber\b", r"\bmrp\d+", r"\btord\d+"),
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
            r"\b(?:ETP|TRO|TRIP)[A-Za-z0-9-]{2,}\b",
        ),
        strong_keywords=(
            r"\btrips?\b",
            r"\btripnumber\b",
            r"\btripno\b",
            r"\b(?:ETP|TRO|TRIP)[A-Za-z0-9-]{2,}\b",
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


def list_domains() -> List[str]:
    return list(DOMAINS.keys())


def get_domain_profile(domain: str) -> DomainProfile:
    key = (domain or DEFAULT_DOMAIN).lower()
    if key not in DOMAINS:
        raise KeyError(f"Unknown domain: {domain!r}")
    return DOMAINS[key]
