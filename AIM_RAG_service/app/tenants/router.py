from __future__ import annotations

from pymongo.collection import Collection

from app.mongo_client import get_mongo_collection
from app.tenants.context import get_active_domain, require_tenant


def get_domain_collection(domain: str = None) -> Collection:
    domain = domain or get_active_domain()
    tenant = require_tenant()
    return get_mongo_collection(
        tenant.collection_for(domain),
        db_name=tenant.database,
    )


def get_domain_namespace(domain: str = None) -> str:
    domain = domain or get_active_domain()
    return require_tenant().namespace_for(domain)


def get_domain_metadata_type(domain: str = None) -> str:
    domain = domain or get_active_domain()
    return require_tenant().metadata_type_for(domain)


def get_orders_collection() -> Collection:
    return get_domain_collection("orders")


def get_orders_namespace() -> str:
    return get_domain_namespace("orders")


def get_orders_metadata_type() -> str:
    return get_domain_metadata_type("orders")
