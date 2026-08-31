"""
Backward-compatible re-exports — canonical prompts live in app.System_prompt.order_prompt.
"""

from app.System_prompt.order_prompt import (  # noqa: F401
    ORDER_ASK_PROMPT,
    ORDER_CONVERSATION_PROMPT,
    ORDER_FORMULA_PROMPT,
    ORDER_GREETING_PROMPT,
    ORDER_LOOKUP_PROMPT,
    ORDER_SYSTEM_PROMPT,
)

# Alias used by older code
ORDER_CORE_POLICY = ORDER_SYSTEM_PROMPT

try:
    from app.order_ask.field_catalog import format_field_catalog_for_prompt

    FILTERABLE_FIELDS_JSON = (
        format_field_catalog_for_prompt().replace("{", "{{").replace("}", "}}")
    )
except Exception:  # pragma: no cover
    FILTERABLE_FIELDS_JSON = "{}"
