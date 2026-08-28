from __future__ import annotations

import logging
import os
import re
from typing import Dict

from dotenv import load_dotenv

from app.tenants.models import TenantConfig

logger = logging.getLogger("tenants.mapping")

_SERVICE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
load_dotenv(os.path.join(_SERVICE_ROOT, ".env"))

DEFAULT_NAMESPACES = {
    "orders": "avaal_orders",
    "invoices": "avaal_invoices",
    "trips": "avaal_trips",
}
DEFAULT_METADATA_TYPES = {
    "orders": "avaal_order",
    "invoices": "avaal_invoice",
    "trips": "avaal_trip",
}
DEFAULT_COLLECTIONS = {
    "orders": os.environ.get("AVAAL_COLLECTION_NAME", "Avaal_order"),
    "invoices": os.environ.get("AVAAL_INVOICE_COLLECTION_NAME", "Avaal_invoice"),
    "trips": os.environ.get("AVAAL_TRIPS_COLLECTION_NAME", "Avaal_trip"),
}

# Optional prefix if all tenant DBs share one, e.g. "corp_" -> corp_ABC123
TENANT_DB_PREFIX = os.environ.get("TENANT_DB_PREFIX", "")

# MongoDB DB names: letters, digits, underscore, hyphen; max 63 bytes
_CORPORATE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,63}$")


class InvalidCorporateIdError(ValueError):
    def __init__(self, corporate_id: str, reason: str = ""):
        self.corporate_id = corporate_id
        self.reason = reason
        message = f"Invalid corporate_id: {corporate_id!r}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


# Backward-compatible alias for callers that still catch UnknownTenantError
class UnknownTenantError(InvalidCorporateIdError):
    pass


def validate_corporate_id(raw: str) -> str:
    corporate_id = (raw or "").strip()
    if not corporate_id:
        raise InvalidCorporateIdError(raw, "corporate_id is required")
    if not _CORPORATE_ID_RE.fullmatch(corporate_id):
        raise InvalidCorporateIdError(
            corporate_id,
            "use letters, numbers, underscore, hyphen only (max 63 chars)",
        )
    return corporate_id


# Hardcoded override: this corporate_id uses shared chatbot_db (not AFMQA as DB name)
_CORPORATE_DB_OVERRIDES = {
    "AFMQA": "chatbot_db",
}


def _database_name(corporate_id: str) -> str:
    override = _CORPORATE_DB_OVERRIDES.get(corporate_id.upper())
    if override:
        return override
    return f"{TENANT_DB_PREFIX}{corporate_id}"


def verify_tenant_database_exists(corporate_id: str, database: str) -> None:
    """
    Ensure the mapped Mongo database already exists.
    Does NOT create anything — corporate_id must match a pre-provisioned DB.
    """
    from app.mongo_client import get_mongo_client, list_mongo_database_names

    get_mongo_client().admin.command("ping")
    existing = set(list_mongo_database_names())
    if database not in existing:
        raise InvalidCorporateIdError(
            corporate_id,
            (
                f"Mongo database {database!r} not found. "
                "corporate_id must match an existing company database name."
            ),
        )


def get_tenant_config(corporate_id: str, *, verify_exists: bool = True) -> TenantConfig:
    """
    Map corporate_id -> existing Mongo database (same name, optional prefix/override).

    Does NOT create databases or collections. If verify_exists=True (default),
    raises when the mapped database is missing on the server.
    """
    corporate_id = validate_corporate_id(corporate_id)
    database = _database_name(corporate_id)

    if verify_exists:
        verify_tenant_database_exists(corporate_id, database)

    logger.info(
        "Tenant matched: corporate_id=%s database=%s collections=%s",
        corporate_id,
        database,
        DEFAULT_COLLECTIONS,
    )

    return TenantConfig(
        corporate_id=corporate_id,
        company_name=corporate_id,
        database=database,
        collections=dict(DEFAULT_COLLECTIONS),
        namespaces=dict(DEFAULT_NAMESPACES),
        metadata_types=dict(DEFAULT_METADATA_TYPES),
    )


def list_tenant_ids() -> list[str]:
    """Dynamic mode — tenants are not pre-registered."""
    return []
