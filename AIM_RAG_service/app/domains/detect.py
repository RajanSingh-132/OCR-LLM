"""Detect which domain (orders / invoices / trips) a question belongs to."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from app.domains.prompts import DOMAIN_CLASSIFY_PROMPT
from app.domains.registry import DEFAULT_DOMAIN, DOMAINS, get_domain_profile
from app.embedding_client import get_anthropic_llm
from app.order_ask.checkpoint import checkpoint

logger = logging.getLogger("domains.detect")

_STRONG_SCORE = 3.0


def _score_domains(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, profile in DOMAINS.items():
        score = 0.0
        for pattern in profile.strong_keywords:
            if re.search(pattern, text, re.I):
                score += _STRONG_SCORE
        for pattern in profile.keywords:
            if re.search(pattern, text, re.I):
                score += 1.0
        if score > 0:
            scores[name] = score
    return scores


def detect_domain_local(question: str, *, history_hint: str = "") -> Tuple[str, str, dict[str, float]]:
    """
    Keyword-only domain pick. Returns (domain, reason, scores).
    Strong keyword match (score >= 3) is treated as confident.
    """
    text = f"{history_hint} {question or ''}".lower().strip()
    if not text:
        return DEFAULT_DOMAIN, "empty_question_default_orders", {}

    scores = _score_domains(text)
    if not scores:
        return DEFAULT_DOMAIN, "no_keyword_match", scores

    domain = max(scores, key=scores.get)
    top = scores[domain]
    tied = [name for name, score in scores.items() if score == top]
    if len(tied) > 1:
        return domain, f"keyword_tie_scores={scores}", scores
    if top >= _STRONG_SCORE:
        return domain, f"strong_keyword_scores={scores}", scores
    return domain, f"weak_keyword_scores={scores}", scores


def classify_domain_with_anthropic(
    question: str,
    *,
    history: str = "(no prior turns)",
    last_domain: str = "",
) -> Dict[str, Any]:
    """Ask Claude which collection domain the question belongs to."""
    checkpoint("DOMAIN", "Anthropic classify", question=(question or "")[:80])
    llm = get_anthropic_llm()
    try:
        llm = llm.bind(max_tokens=120)
    except Exception:
        pass

    chain = PromptTemplate.from_template(DOMAIN_CLASSIFY_PROMPT) | llm
    raw = chain.invoke(
        {
            "question": question or "",
            "history": history or "(no prior turns)",
            "last_domain": last_domain or "none",
        }
    )
    text = raw.content if hasattr(raw, "content") else str(raw)
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Domain JSON parse failed; defaulting. raw=%s", text[:300])
        data = {}

    domain = str(data.get("domain") or DEFAULT_DOMAIN).lower().strip()
    if domain not in DOMAINS:
        logger.warning("Unknown domain from LLM %r; defaulting to %s", domain, DEFAULT_DOMAIN)
        domain = DEFAULT_DOMAIN

    return {
        "domain": domain,
        "reason": data.get("reason") or "anthropic_classify",
        "normalized_terms": data.get("normalized_terms") or "",
        "method": "llm",
    }


def detect_domain_detailed(
    question: str,
    *,
    history_hint: str = "",
    chat_history: str = "",
) -> Dict[str, Any]:
    """
    Hybrid domain detection:
    1) Fast keyword path when a strong match exists (exact invoice/order/trip words).
    2) Claude when keywords miss, tie, or only weak signals (typos: invoi, ord, trp).
    """
    local_domain, local_reason, scores = detect_domain_local(
        question, history_hint=history_hint
    )

    use_llm = False
    if not scores:
        use_llm = True
    else:
        top_score = max(scores.values())
        tied = sum(1 for score in scores.values() if score == top_score) > 1
        if top_score < _STRONG_SCORE or tied:
            use_llm = True

    if not use_llm:
        result = {
            "domain": local_domain,
            "method": "local",
            "reason": local_reason,
            "scores": scores,
        }
        checkpoint(
            "DOMAIN",
            "detected",
            domain=local_domain,
            method="local",
            reason=local_reason,
        )
        return result

    try:
        llm_result = classify_domain_with_anthropic(
            question,
            history=chat_history or "(no prior turns)",
            last_domain=history_hint or "",
        )
        result = {
            "domain": llm_result["domain"],
            "method": "llm",
            "reason": llm_result.get("reason") or "anthropic_classify",
            "normalized_terms": llm_result.get("normalized_terms") or "",
            "local_fallback": local_domain,
            "local_reason": local_reason,
            "scores": scores,
        }
        checkpoint(
            "DOMAIN",
            "detected",
            domain=result["domain"],
            method="llm",
            reason=result["reason"],
            normalized=result.get("normalized_terms"),
        )
        return result
    except Exception as exc:
        logger.error("Anthropic domain classify failed: %s", exc, exc_info=True)
        fallback = local_domain if scores else DEFAULT_DOMAIN
        result = {
            "domain": fallback,
            "method": "fallback",
            "reason": f"llm_error:{exc}",
            "scores": scores,
        }
        checkpoint(
            "DOMAIN",
            "detected",
            domain=fallback,
            method="fallback",
            reason=result["reason"],
        )
        return result


def detect_domain(
    question: str,
    *,
    history_hint: str = "",
    chat_history: str = "",
) -> str:
    """Return domain name (orders / invoices / trips)."""
    return detect_domain_detailed(
        question,
        history_hint=history_hint,
        chat_history=chat_history,
    )["domain"]


def domain_label(domain: str) -> str:
    return get_domain_profile(domain).label
