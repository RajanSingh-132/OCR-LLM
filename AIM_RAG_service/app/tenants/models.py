from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TenantConfig:
    corporate_id: str
    company_name: str
    database: str
    collections: Dict[str, str]
    namespaces: Dict[str, str]
    metadata_types: Dict[str, str]

    def collection_for(self, domain: str) -> str:
        try:
            return self.collections[domain]
        except KeyError as exc:
            raise KeyError(
                f"Tenant {self.corporate_id!r} has no collection for domain {domain!r}"
            ) from exc

    def namespace_for(self, domain: str) -> str:
        return self.namespaces[domain]

    def metadata_type_for(self, domain: str) -> str:
        return self.metadata_types[domain]

    def to_response_meta(self, domain: str = "orders") -> Dict[str, str]:
        return {
            "corporate_id": self.corporate_id,
            "company_name": self.company_name,
            "database": self.database,
            "collection": self.collection_for(domain),
            "namespace": self.namespace_for(domain),
        }
