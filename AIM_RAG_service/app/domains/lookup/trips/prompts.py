"""
Backward-compatible re-exports — canonical prompts live in app.System_prompt.trip_prompt.
"""

from app.System_prompt.trip_prompt import (  # noqa: F401
    TRIP_ASK_PROMPT,
    TRIP_CONVERSATION_PROMPT,
    TRIP_GREETING_PROMPT,
    TRIP_LOOKUP_PROMPT,
    TRIP_SYSTEM_PROMPT,
)

TRIP_CORE_POLICY = TRIP_SYSTEM_PROMPT
