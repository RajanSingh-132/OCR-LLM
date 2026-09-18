"""
Typo-tolerant trigger-word matching for the fast/deterministic regex paths
(fast_path.py, domains/registry.py, analytics.py, domains/rules/orders.py).

Flow (exactly as designed): tokenize -> try exact match first (cheap, and
what almost every question hits — no typo, no extra work) -> only for
questions where NO exact match was found, fuzzy-match each token against
the trigger words within a small bounded edit distance -> threshold check
(max_distance) decides accept/reject -> caller treats a fuzzy hit the same
as an exact one (the "corrected" intent), then proceeds into the SAME
Mongo/LLM pipeline as before — nothing downstream changes.

No LLM call, no external dependency (pure Python) — this stays cheap enough
to run on every question without hurting the fast-path's whole reason to
exist (skip the LLM call for simple questions).

Distance used is Damerau-Levenshtein (adjacent-transposition counts as ONE
edit, not two) because the typos actually seen in this project so far
("mnay" for "many", "cosutomer" for "customer") are both adjacent-letter
swaps — the single most common real-world typing mistake — which plain
Levenshtein would score as distance 2 and miss at max_distance=1.
"""
from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _damerau_levenshtein_le(a: str, b: str, max_dist: int) -> bool:
    """True if edit distance (insert/delete/substitute/adjacent-transpose)
    between a and b is <= max_dist. Cheap length-difference short-circuit
    before doing the actual DP table."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False

    # Standard bounded Damerau-Levenshtein DP (small strings — trigger words
    # and question tokens are a handful of characters, this is microseconds).
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition

    return d[la][lb] <= max_dist


def _allowed_distance(word_len: int) -> int:
    """How many edits a word is allowed to be off by, scaled to its length —
    a fixed distance (e.g. always 2) makes short words dangerously
    ambiguous: "total" vs "trial" is distance 2, and those are two
    completely different, both-common English words. Longer words have
    more "room" before a typo could land on a different real word."""
    if word_len <= 6:
        return 1
    return 2


def fuzzy_contains_any(
    question: str,
    trigger_words: Iterable[str],
    *,
    exact_re: "re.Pattern | None" = None,
    min_word_len: int = 4,
) -> bool:
    """
    1) If exact_re is given and matches -> True immediately (no fuzzy work at
       all — this is the zero-typo common case, unchanged speed).
    2) Otherwise tokenize the question and fuzzy-match each token against
       trigger_words, with the allowed edit distance scaled to word length
       (see _allowed_distance). Words shorter than min_word_len are skipped
       entirely on the fuzzy pass — too easy to false-match something
       unrelated (e.g. "an" vs "any").
    """
    q = question or ""
    if exact_re is not None and exact_re.search(q):
        return True

    tokens = [t for t in _tokenize(q) if len(t) >= min_word_len]
    if not tokens:
        return False

    triggers = [t.lower() for t in trigger_words if len(t) >= min_word_len]
    for token in tokens:
        for trigger in triggers:
            allowed = min(_allowed_distance(len(token)), _allowed_distance(len(trigger)))
            if _damerau_levenshtein_le(token, trigger, allowed):
                return True
    return False
