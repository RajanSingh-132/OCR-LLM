from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

from app.tenants.models import TenantConfig

_tenant_ctx: ContextVar[Optional[TenantConfig]] = ContextVar("tenant_ctx", default=None)
_domain_ctx: ContextVar[str] = ContextVar("domain_ctx", default="orders")


class AskContext:
    """Bind tenant (DB) + active domain (collection) for one ask request."""

    def __init__(self, tenant: TenantConfig, domain: str):
        self.tenant = tenant
        self.domain = (domain or "orders").lower()
        self._tenant_token: Optional[Token] = None
        self._domain_token: Optional[Token] = None

    def __enter__(self) -> TenantConfig:
        self._tenant_token = _tenant_ctx.set(self.tenant)
        self._domain_token = _domain_ctx.set(self.domain)
        return self.tenant

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._domain_token is not None:
            _domain_ctx.reset(self._domain_token)
        if self._tenant_token is not None:
            _tenant_ctx.reset(self._tenant_token)


# Backward-compatible alias
TenantContext = AskContext


def require_tenant() -> TenantConfig:
    tenant = _tenant_ctx.get()
    if tenant is None:
        raise RuntimeError(
            "Tenant context is not set. Resolve corporate_id before running tools."
        )
    return tenant


def get_current_tenant() -> Optional[TenantConfig]:
    return _tenant_ctx.get()


def get_active_domain() -> str:
    return _domain_ctx.get() or "orders"
