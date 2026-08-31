"""
Backward-compatible re-exports — canonical prompts live in app.System_prompt.invoice_prompt.
"""

from app.System_prompt.invoice_prompt import (  # noqa: F401
    INVOICE_ASK_PROMPT,
    INVOICE_CONVERSATION_PROMPT,
    INVOICE_GREETING_PROMPT,
    INVOICE_LOOKUP_PROMPT,
    INVOICE_SYSTEM_PROMPT,
)

INVOICE_CORE_POLICY = INVOICE_SYSTEM_PROMPT
