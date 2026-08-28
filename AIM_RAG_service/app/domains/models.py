from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class DomainProfile:
    """Config for one data domain (orders, invoices, trips, ...)."""

    name: str
    label: str
    keywords: Tuple[str, ...] = ()
    strong_keywords: Tuple[str, ...] = ()
    id_fields: Tuple[str, ...] = ()
    number_fields: Tuple[str, ...] = ()
    sort_fields: Tuple[str, ...] = ()
    default_sort: str = "_id"
    list_fields: Tuple[str, ...] = ()

    def all_lookup_fields(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for key in (*self.id_fields, *self.number_fields):
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out
