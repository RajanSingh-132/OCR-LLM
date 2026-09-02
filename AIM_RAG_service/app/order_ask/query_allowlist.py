"""
Query builder allowlist — order data.

This is the single source of truth for what the constrained dynamic
query builder is permitted to touch. Nothing outside this file's
allowlists can ever be referenced in an assembled query, regardless of
what the LLM outputs.

Table list is drawn directly from what fn_getorders_regular already
joins — these are the tables already proven relevant to order
questions, not a guess. Internal/audit columns (createdby, modifiedby,
raw orderid as an output field) are deliberately excluded from
ALLOWED_COLUMNS even though orderid remains allowed as a JOIN KEY
internally (see ALLOWED_JOIN_KEYS) — the response shaper (Step 1)
strips it from anything returned to the user regardless, but keeping
it out of ALLOWED_COLUMNS means the query builder can't even select it
as an output column in the first place.
"""
from __future__ import annotations

from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Approved procedures — the fast, pre-optimized path. Prefer these when a
# question fits one of them; the dynamic query builder is for what these
# don't cleanly cover.
# ---------------------------------------------------------------------------
ALLOWED_PROCEDURES: Set[str] = {
    "fn_getorders_regular_ai",
    "fn_getorders_aggregate_count_ai",
    "fn_getorders_aggregate_freight_ai",
    "fn_getorders_summary_ai",
}

# ---------------------------------------------------------------------------
# Tables the dynamic query builder may reference — the full set already
# joined in fn_getorders_regular. Nothing outside this set is queryable,
# no matter what the LLM's payload says.
# ---------------------------------------------------------------------------
ALLOWED_TABLES: Set[str] = {
    "trnorder",
    "trnorderitem",
    "trnorderstatus",
    "trnordersoutdetails",
    "trnorderappointment",
    "trnorderitemrevenue",
    "trnextracharge",
    "trnordertaxes",
    "trnorderpickupdeliver",
    "trntriporderitemdetail",
    "trntripassets",
    "trninvoicedetail",
    "trninvoice",
    "trnsettlement",
    "trnreceiptdetails",
    "trnbid",
    "trnuploadedfiles",
    "mstprimarycclinfo",   # customers / carriers
    "mstemployee",         # salesmen
    "mstdriver",
    "msttruck",
    "msttrailer",
    "mstlookup",
    "mstdocumenttype",
    "vw_locationdetails",  # pickup/delivery location details
}

# ---------------------------------------------------------------------------
# Per-table column allowlist — SELECT/filter/groupby may only reference
# these. Deliberately excludes: orderid (as an output column — still
# usable as a join key, see ALLOWED_JOIN_KEYS below), createdby,
# modifiedby, and any other raw audit/internal column.
# ---------------------------------------------------------------------------
ALLOWED_COLUMNS: Dict[str, List[str]] = {
    "trnorder": [
        "ordernumber", "tempordernumber", "companycode", "customercode",
        "salesmancode", "customerorderno", "currencycode", "enquirydate",
        "quotationdate", "orderdate", "offeredamount", "totalamount",
        "ordernotes", "pono", "referno", "pendingaccessorial",
        "scaleticketno", "isactive", "isdeleted", "createdon", "modifiedon",
    ],
    "trnorderitem": [
        "orderid", "commodityname", "valueofgoods", "quantity", "quantityunit",
        "equipmenttype", "weight", "weightunit", "hazmat", "overdimension",
        "csa", "ctpat", "fast", "pip", "pilotcar", "pickupstopcode",
        "deliverystopcode", "pickupdate", "deliverydate", "pickupnumber",
        "deliverynumber",
    ],
    "trnorderstatus": [
        "orderid", "status", "statuscode", "modulecode",
    ],
    "trnordersoutdetails": [
        "orderid", "outtype", "outcarriercode", "outstatus", "outrate",
        "outvalue", "outofferedamount", "outcurrency",
    ],
    "trnorderappointment": [
        "orderid", "appointmentno", "starttime", "appointmenttype", "locationcode",
    ],
    "trnorderitemrevenue": [
        "orderid", "revenueattribute", "amount", "rate", "ratemethodtypelucode",
    ],
    "trnextracharge": [
        "parentrefcode", "convertedamount",
    ],
    "trnordertaxes": [
        "orderid", "taxname", "taxpercentage", "taxamount",
    ],
    "trnorderpickupdeliver": [
        "orderid", "actiontype", "stopcode", "chain",
    ],
    "trntriporderitemdetail": [
        "orderid", "tripid",
    ],
    "trntripassets": [
        "tripid", "reftype", "refcode",
    ],
    "trninvoicedetail": [
        "orderid", "invoiceid", "customerordernumber", "pickupdate", "deliverydate",
    ],
    "trninvoice": [
        "invoiceid", "invoicenumber", "invoicestatus",
    ],
    "trnsettlement": [
        "orderoutid", "settlementtype", "settledstatus", "paidstatus",
    ],
    "trnreceiptdetails": [
        "invoiceid", "receiptid",
    ],
    "trnbid": [
        "orderid", "numberofcarriers",
    ],
    "trnuploadedfiles": [
        "parentmodulerefcode", "parentmoduletypecode", "documenttypecode",
    ],
    "mstprimarycclinfo": [
        "primaryinfocode", "name", "phone",
    ],
    "mstemployee": [
        "employeecode", "firstname", "lastname",
    ],
    "mstdriver": [
        "drivercode", "firstname", "lastname",
    ],
    "msttruck": [
        "truckcode", "trucknumber",
    ],
    "msttrailer": [
        "trailercode", "trailernumber",
    ],
    "mstlookup": [
        "lucode", "lutype", "displayname",
    ],
    "mstdocumenttype": [
        "documenttypecode", "documenttype",
    ],
    "vw_locationdetails": [
        "primaryinfocode", "name", "fulladdress", "countrycode", "statecode", "city",
    ],
}

# orderid is allowed as a JOIN KEY across tables (required to connect them)
# but is NOT in ALLOWED_COLUMNS, so it can never be selected as an output
# field. The response shaper (Step 1) is the second line of defense on top
# of this — even if it somehow ended up in a result set, it gets stripped
# before reaching the LLM/user.
ALLOWED_JOIN_KEYS: Set[str] = {
    "orderid", "ordernumber", "tripid", "orderoutid", "invoiceid",
    "customercode", "salesmancode", "drivercode", "truckcode",
    "trailercode", "outcarriercode", "primaryinfocode", "pickupstopcode",
    "deliverystopcode", "employeecode", "lucode", "documenttypecode",
    "parentrefcode", "parentmodulerefcode",
}

# ---------------------------------------------------------------------------
# Allowed filter operators — nothing outside this set can appear in an
# assembled WHERE clause, regardless of LLM output.
# ---------------------------------------------------------------------------
ALLOWED_OPERATORS: Set[str] = {
    "=", "!=", ">", ">=", "<", "<=", "between", "in", "ilike",
}

# ---------------------------------------------------------------------------
# Allowed aggregate functions for groupby/aggregate queries.
# ---------------------------------------------------------------------------
ALLOWED_AGGREGATES: Set[str] = {"count", "sum", "avg", "min", "max"}

# Hard cap — any assembled query must include this LIMIT if none is
# specified, and any requested limit above this is clamped down to it.
MAX_ROW_LIMIT = 500


def is_table_allowed(table: str) -> bool:
    return table in ALLOWED_TABLES


def is_column_allowed(table: str, column: str) -> bool:
    if column in ALLOWED_JOIN_KEYS:
        return True
    return column in ALLOWED_COLUMNS.get(table, [])


def is_operator_allowed(op: str) -> bool:
    return op in ALLOWED_OPERATORS


def is_aggregate_allowed(agg: str) -> bool:
    return agg in ALLOWED_AGGREGATES


# ---------------------------------------------------------------------------
# CONFIRMED against live schema (information_schema.columns dump,
# AFN01801) — not guessed. One nuance worth flagging: "orderid" is
# bigint on trnorder (the PK) but integer everywhere else it appears as
# a foreign key (trnorderitem, trnorderstatus, trnordersoutdetails,
# trnorderappointment, trnorderitemrevenue, trnordertaxes,
# trnorderpickupdeliver, trntriporderitemdetail, trninvoicedetail,
# trnbid). Both are safely comparable/joinable in Postgres without an
# explicit cast, but the query builder should not assume a single
# uniform type for "orderid" across tables if it ever needs to
# generate an explicit CAST.
# ---------------------------------------------------------------------------
ALLOWED_COLUMN_TYPES: Dict[str, Dict[str, str]] = {
    "trnorder": {
        "ordernumber": "character varying", "tempordernumber": "character varying",
        "companycode": "character varying", "customercode": "character varying",
        "salesmancode": "character varying", "customerorderno": "character varying",
        "currencycode": "character varying", "enquirydate": "timestamp with time zone",
        "quotationdate": "timestamp with time zone", "orderdate": "timestamp with time zone",
        "offeredamount": "numeric", "totalamount": "numeric",
        "ordernotes": "character varying", "pono": "character varying",
        "referno": "character varying", "pendingaccessorial": "boolean",
        "scaleticketno": "character varying", "isactive": "boolean",
        "isdeleted": "boolean", "createdon": "timestamp with time zone",
        "modifiedon": "timestamp with time zone",
    },
    "trnorderitem": {
        "orderid": "integer", "commodityname": "character varying",
        "valueofgoods": "numeric", "quantity": "numeric",
        "quantityunit": "character varying", "equipmenttype": "character varying",
        "weight": "numeric", "weightunit": "character varying",
        "hazmat": "boolean", "overdimension": "boolean", "csa": "boolean",
        "ctpat": "boolean", "fast": "boolean", "pip": "boolean",
        "pilotcar": "boolean", "pickupstopcode": "character varying",
        "deliverystopcode": "character varying",
        "pickupdate": "timestamp with time zone", "deliverydate": "timestamp with time zone",
        "pickupnumber": "character varying", "deliverynumber": "character varying",
    },
    "trnorderstatus": {
        "orderid": "integer", "status": "character varying",
        "statuscode": "character varying", "modulecode": "character varying",
    },
    "trnordersoutdetails": {
        "orderid": "integer", "outtype": "character varying",
        "outcarriercode": "character varying", "outstatus": "character varying",
        "outrate": "numeric", "outvalue": "numeric",
        "outofferedamount": "numeric", "outcurrency": "character varying",
    },
    "trnorderappointment": {
        "orderid": "integer", "appointmentno": "character varying",
        "starttime": "timestamp with time zone", "appointmenttype": "character",
        "locationcode": "character varying",
    },
    "trnorderitemrevenue": {
        "orderid": "integer", "revenueattribute": "character varying",
        "amount": "numeric", "rate": "numeric",
        "ratemethodtypelucode": "character varying",
    },
    "trnextracharge": {
        "parentrefcode": "character varying", "convertedamount": "numeric",
    },
    "trnordertaxes": {
        "orderid": "integer", "taxname": "character varying",
        "taxpercentage": "numeric", "taxamount": "numeric",
    },
    "trnorderpickupdeliver": {
        "orderid": "integer", "actiontype": "character",
        "stopcode": "character varying", "chain": "integer",
    },
    "trntriporderitemdetail": {
        "orderid": "integer", "tripid": "integer",
    },
    "trntripassets": {
        "tripid": "integer", "reftype": "character varying",
        "refcode": "character varying",
    },
    "trninvoicedetail": {
        "orderid": "integer", "invoiceid": "integer",
        "customerordernumber": "character varying",
        "pickupdate": "timestamp with time zone", "deliverydate": "timestamp with time zone",
    },
    "trninvoice": {
        "invoiceid": "bigint", "invoicenumber": "character varying",
        "invoicestatus": "character varying",
    },
    "trnsettlement": {
        "orderoutid": "integer", "settlementtype": "character varying",
        "settledstatus": "character varying", "paidstatus": "character varying",
    },
    "trnreceiptdetails": {
        "invoiceid": "integer", "receiptid": "integer",
    },
    "trnbid": {
        "orderid": "integer", "numberofcarriers": "integer",
    },
    "trnuploadedfiles": {
        "parentmodulerefcode": "character varying",
        "parentmoduletypecode": "character varying",
        "documenttypecode": "character varying",
    },
    "mstprimarycclinfo": {
        "primaryinfocode": "character varying", "name": "character varying",
        "phone": "character varying",
    },
    "mstemployee": {
        "employeecode": "character varying", "firstname": "character varying",
        "lastname": "character varying",
    },
    "mstdriver": {
        "drivercode": "character varying", "firstname": "character varying",
        "lastname": "character varying",
    },
    "msttruck": {
        "truckcode": "character varying", "trucknumber": "character varying",
    },
    "msttrailer": {
        "trailercode": "character varying", "trailernumber": "character varying",
    },
    "mstlookup": {
        "lucode": "character varying", "lutype": "character varying",
        "displayname": "character varying",
    },
    "mstdocumenttype": {
        "documenttypecode": "character varying", "documenttype": "character varying",
    },
    "vw_locationdetails": {
        "primaryinfocode": "character varying", "name": "character varying",
        "fulladdress": "text", "countrycode": "character varying",
        "statecode": "character varying", "city": "character varying",
    },
}


def get_column_type(table: str, column: str) -> str | None:
    """Returns the confirmed Postgres data type for an allowed column,
    or None if the table/column isn't in the allowlist at all."""
    return ALLOWED_COLUMN_TYPES.get(table, {}).get(column)
