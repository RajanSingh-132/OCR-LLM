"""
Terminal checkpoints for Avaal order ask flow.

Prints clear [CHECKPOINT] banners to stdout so you can watch the pipeline
while the server runs.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("order_ask.checkpoint")


def checkpoint(step: str, detail: str = "", **extra: Any) -> None:
    """Print + log a visible runtime checkpoint."""
    bits = [f"[CHECKPOINT] {step}"]
    if detail:
        bits.append(f"- {detail}")
    if extra:
        kv = ", ".join(f"{k}={_short(v)}" for k, v in extra.items())
        bits.append(f"| {kv}")
    line = " ".join(bits)
    print(line, flush=True)
    logger.info(line)


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


class CheckpointTimer:
    """Simple step timer for one ask request."""

    def __init__(self, label: str = "ask"):
        self.label = label
        self.t0 = time.perf_counter()
        self.last = self.t0

    def mark(self, step: str, detail: str = "", **extra: Any) -> None:
        now = time.perf_counter()
        step_ms = int((now - self.last) * 1000)
        total_ms = int((now - self.t0) * 1000)
        checkpoint(
            step,
            detail,
            step_ms=step_ms,
            total_ms=total_ms,
            **extra,
        )
        self.last = now
