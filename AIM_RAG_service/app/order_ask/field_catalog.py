"""
Filterable field catalog for Avaal order Q&A.

Teaches the AI which dimensions users can ask about, and maps natural
language → Mongo filter keys. Pin/state/city live inside address strings:
  pickupfulladdress / deliveryfulladdress
  e.g. "1241 OLD TEMESCAL ROAD #103, CORONA, CA, 92881, United States, ..."
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# JSON schema the LLM sees in prompts — keep compact and accurate.
FILTERABLE_FIELDS_JSON: Dict[str, Any] = {
    "assistant": "Avaal AI assistant",
    "dataset": "Avaal transport orders",
    "rule": (
        "When the user asks by any of these fields, filter the FULL order set "
        "and answer only from matching rows. If no rows match, give a sweet short apology."
    ),
    "address_shape": (
        "pickupfulladdress / deliveryfulladdress look like: "
        "STREET, CITY, STATE_OR_PROVINCE, PIN_OR_ZIP, Country, email, phone"
    ),
    "fields": [
        {
            "key": "customername",
            "aliases": ["customer", "customer name", "client", "shipper"],
            "db_fields": ["customername"],
            "example_questions": [
                "orders for customer 1-BRIDGE LOGISTICS",
                "show orders of customer Avaal Group",
            ],
        },
        {
            "key": "customercode",
            "aliases": ["customer code", "account code"],
            "db_fields": ["customercode"],
            "example_questions": ["orders with customer code PRI02810"],
        },
        {
            "key": "companycode",
            "aliases": ["company", "company code"],
            "db_fields": ["companycode"],
            "example_questions": ["orders for company AVAAL"],
        },
        {
            "key": "orderstatus",
            "aliases": ["status", "order status"],
            "db_fields": ["orderstatus"],
            "values": [
                "Quoted",
                "Cancelled",
                "Confirmed",
                "Dispatched",
                "Delivered",
                "Invoiced",
            ],
            "example_questions": ["list confirmed orders", "show delivered orders"],
        },
        {
            "key": "pin",
            "aliases": ["pin", "pin code", "pincode", "zip", "zip code", "postal", "postal code"],
            "db_fields": ["pickupfulladdress", "deliveryfulladdress"],
            "match": "regex inside address text (US 5-digit zip or Canadian postal)",
            "side": "pickup | delivery | both",
            "example_questions": [
                "orders with pin 92881",
                "delivery zip 79927",
                "pickup postal code M5V 2T6",
            ],
        },
        {
            "key": "state",
            "aliases": ["state", "province", "region"],
            "db_fields": ["pickupfulladdress", "deliveryfulladdress"],
            "match": "state/province code or name inside address (CA, TX, Ontario, …)",
            "side": "pickup | delivery | both",
            "example_questions": [
                "orders in California",
                "delivery to TX",
                "pickup from Ontario",
            ],
        },
        {
            "key": "city",
            "aliases": ["city", "town"],
            "db_fields": ["pickupfulladdress", "deliveryfulladdress", "pickuplocationname", "deliverylocationname"],
            "side": "pickup | delivery | both",
            "example_questions": [
                "orders in Corona",
                "delivery city Socorro",
                "pickup from Toronto",
            ],
        },
        {
            "key": "address",
            "aliases": ["address", "full address", "street"],
            "db_fields": ["pickupfulladdress", "deliveryfulladdress"],
            "side": "pickup | delivery | both",
            "example_questions": [
                "orders with address ALAMEDA",
                "pickup address TEMESCAL",
            ],
        },
        {
            "key": "location",
            "aliases": ["location", "place", "warehouse", "facility"],
            "db_fields": [
                "pickuplocationname",
                "deliverylocationname",
                "pickupfulladdress",
                "deliveryfulladdress",
            ],
            "side": "pickup | delivery | both",
            "example_questions": [
                "orders pickup location WELLINGTON",
                "delivery location WERNER",
                "orders at location FOODS",
            ],
        },
        {
            "key": "country",
            "aliases": ["country", "Canada", "USA", "United States"],
            "db_fields": ["pickupfulladdress", "deliveryfulladdress"],
            "side": "pickup | delivery | both",
            "example_questions": [
                "how many customers in Canada",
                "orders delivering to USA",
            ],
        },
        {
            "key": "orderdate",
            "aliases": ["order date", "dated", "on date"],
            "db_fields": ["orderdate"],
            "example_questions": ["orders on 2026-08-06"],
        },
        {
            "key": "pickupdate",
            "aliases": ["pickup date"],
            "db_fields": ["pickupdate"],
            "example_questions": ["pickup on 07/13/2026"],
        },
        {
            "key": "deliverydate",
            "aliases": ["delivery date", "drop date"],
            "db_fields": ["deliverydate"],
            "example_questions": ["delivery date 2026-08-06"],
        },
        {
            "key": "currencycode",
            "aliases": ["currency", "CAD", "USD"],
            "db_fields": ["currencycode"],
            "example_questions": ["CAD orders", "orders in USD"],
        },
        {
            "key": "salesmanname",
            "aliases": ["salesman", "sales person"],
            "db_fields": ["salesmanname"],
            "example_questions": ["orders by salesman Avaal QA"],
        },
        {
            "key": "commodityname",
            "aliases": ["commodity", "product"],
            "db_fields": ["commodityname"],
            "example_questions": ["orders with commodity FOOD"],
        },
    ],
}

# US states + Canadian provinces (code → display / aliases)
STATE_ALIASES: Dict[str, List[str]] = {
    "AL": ["alabama"],
    "AK": ["alaska"],
    "AZ": ["arizona"],
    "AR": ["arkansas"],
    "CA": ["california"],
    "CO": ["colorado"],
    "CT": ["connecticut"],
    "DE": ["delaware"],
    "FL": ["florida"],
    "GA": ["georgia"],
    "HI": ["hawaii"],
    "ID": ["idaho"],
    "IL": ["illinois"],
    "IN": ["indiana"],
    "IA": ["iowa"],
    "KS": ["kansas"],
    "KY": ["kentucky"],
    "LA": ["louisiana"],
    "ME": ["maine"],
    "MD": ["maryland"],
    "MA": ["massachusetts"],
    "MI": ["michigan"],
    "MN": ["minnesota"],
    "MS": ["mississippi"],
    "MO": ["missouri"],
    "MT": ["montana"],
    "NE": ["nebraska"],
    "NV": ["nevada"],
    "NH": ["new hampshire"],
    "NJ": ["new jersey"],
    "NM": ["new mexico"],
    "NY": ["new york"],
    "NC": ["north carolina"],
    "ND": ["north dakota"],
    "OH": ["ohio"],
    "OK": ["oklahoma"],
    "OR": ["oregon"],
    "PA": ["pennsylvania"],
    "RI": ["rhode island"],
    "SC": ["south carolina"],
    "SD": ["south dakota"],
    "TN": ["tennessee"],
    "TX": ["texas"],
    "UT": ["utah"],
    "VT": ["vermont"],
    "VA": ["virginia"],
    "WA": ["washington"],
    "WV": ["west virginia"],
    "WI": ["wisconsin"],
    "WY": ["wyoming"],
    "DC": ["district of columbia", "washington dc"],
    "ON": ["ontario"],
    "QC": ["quebec", "québec"],
    "BC": ["british columbia"],
    "AB": ["alberta"],
    "MB": ["manitoba"],
    "SK": ["saskatchewan"],
    "NS": ["nova scotia"],
    "NB": ["new brunswick"],
    "NL": ["newfoundland", "newfoundland and labrador"],
    "PE": ["prince edward island", "pei"],
    "NT": ["northwest territories"],
    "YT": ["yukon"],
    "NU": ["nunavut"],
}

# Reverse lookup: alias/name → canonical code
_NAME_TO_STATE: Dict[str, str] = {}
for _code, _names in STATE_ALIASES.items():
    _NAME_TO_STATE[_code.lower()] = _code
    for _n in _names:
        _NAME_TO_STATE[_n.lower()] = _code


def resolve_state_token(raw: str) -> str | None:
    """Return canonical state/province code if recognized."""
    if not raw:
        return None
    key = raw.strip().lower()
    return _NAME_TO_STATE.get(key)


def format_field_catalog_for_prompt() -> str:
    """Compact JSON block injected into answer / intent prompts."""
    return json.dumps(FILTERABLE_FIELDS_JSON, ensure_ascii=False, indent=2)


def catalog_filter_keys() -> List[str]:
    return [f["key"] for f in FILTERABLE_FIELDS_JSON["fields"]]
