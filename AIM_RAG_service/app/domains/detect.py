"""Detect which domain (orders / invoices / trips) a question belongs to."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from app.domains.prompts import DOMAIN_CLASSIFY_PROMPT
from app.domains.registry import DEFAULT_DOMAIN, DOMAINS
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

    # Avaal trip number prefixes are definitive (ETP4455, TRO4725, …).
    if re.search(r"\b(?:etp|tro|trip)[a-z0-9-]{2,}\b", text, re.I):
        return "trips", "trip_number_prefix", {"trips": _STRONG_SCORE + 1.0}

    # Invoice numbers like MR3932 / INO4 / AIN12225 (not MRP/TORD/"invoice").
    if re.search(
        r"\b(?:mr(?!p)|ino|ain|ugy|inv)[a-z0-9-]*\d[a-z0-9-]*\b",
        text,
        re.I,
    ):
        return "invoices", "invoice_number_prefix", {"invoices": _STRONG_SCORE + 1.0}

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


def _is_bare_record_token(question: str) -> bool:
    """True when the message is mostly just an id/number (follow-up after asking for it)."""
    q = (question or "").strip()
    if not q:
        return False
    # Allow short wrappers: "this", "check", "lookup", etc. + token
    q2 = re.sub(
        r"^(?:please\s+)?(?:check|lookup|look\s*up|get|show|find|this|that|here)\s+",
        "",
        q,
        flags=re.I,
    ).strip()
    if len(q2.split()) > 2:
        return False
    tok = q2.split()[0] if q2 else ""
    tok = tok.strip(".,:;\"'")
    if not tok or len(tok) < 2:
        return False
    if tok.lower() in {
        "hi", "hello", "hey", "thanks", "thank", "ok", "okay", "yes", "no",
        "order", "orders", "invoice", "invoices", "trip", "trips",
    }:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{1,40}", tok))


def detect_domain_detailed(
    question: str,
    *,
    history_hint: str = "",
    chat_history: str = "",
) -> Dict[str, Any]:
    """
    Hybrid domain detection:
    1) Fast keyword path when a strong match exists (exact invoice/order/trip words).
    2) Sticky last_domain when user sends only a bare number/id after we asked for one.
    3) Claude when keywords miss, tie, or only weak signals (typos: invoi, ord, trp).
    """
    local_domain, local_reason, scores = detect_domain_local(
        question, history_hint=history_hint
    )

    # After "please provide order/invoice/trip number", user often replies with only the id.
    last = (history_hint or "").strip().lower()
    if last in DOMAINS and _is_bare_record_token(question):
        # Prefix rules from local detection still win when definitive
        if "prefix" not in local_reason:
            result = {
                "domain": last,
                "method": "local",
                "reason": f"bare_token_sticky_last_domain={last}",
                "scores": scores,
            }
            checkpoint(
                "DOMAIN",
                "detected",
                domain=last,
                method="local",
                reason=result["reason"],
            )
            return result

    use_llm = False
    if not scores:
        use_llm = True
    else:
        top_score = max(scores.values())
        tied = sum(1 for score in scores.values() if score == top_score) > 1
        if top_score < _STRONG_SCORE or tied:
            use_llm = True

    # Pure greeting/thanks — never burn an LLM call on domain classify.
    # Default to last sticky domain or orders; greeting answer does not need DB.
    from app.order_ask.intent import GREETING_RE, THANKS_RE

    q_stripped = (question or "").strip()
    if GREETING_RE.match(q_stripped) or THANKS_RE.match(q_stripped):
        sticky = (history_hint or "").strip().lower()
        domain = sticky if sticky in DOMAINS else DEFAULT_DOMAIN
        result = {
            "domain": domain,
            "method": "local",
            "reason": "greeting_or_thanks_skip_domain_llm",
            "scores": scores,
        }
        checkpoint(
            "DOMAIN",
            "detected",
            domain=domain,
            method="local",
            reason=result["reason"],
        )
        return result

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
