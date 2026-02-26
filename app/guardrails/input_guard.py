"""
Input guardrails.

1. Length check — reject inputs over max_input_chars
2. PII scrubbing — strip common PII patterns before text reaches the LLM
   (email, UK/US phone, NHS number patterns)
   Production upgrade: Microsoft Presidio for comprehensive entity recognition.
3. Injection heuristics — flag prompts that try to override system instructions
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── PII patterns (regex fallback) ─────────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
    (r"(\+44\s?|0)7\d{3}\s?\d{6}", "[UK_PHONE]"),
    (r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[US_PHONE]"),
    (r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b", "[PHONE]"),
]

# ── Prompt injection heuristics ───────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore (all |previous |above |all previous |all above )?instructions",
    r"disregard (your |the )?(system |above )?prompt",
    r"forget (all |your |previous |prior )?instructions",
    r"you are now",
    r"act as (a |an )?(?!assistant)",
    r"jailbreak",
    r"DAN mode",
    r"override (your |all |previous )?instructions",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@lru_cache
def _get_presidio_analyzer():
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


@lru_cache
def _get_presidio_anonymizer():
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def _regex_scrub_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _presidio_scrub_pii(text: str) -> str:
    from presidio_anonymizer.entities import OperatorConfig

    analyzer = _get_presidio_analyzer()
    anonymizer = _get_presidio_anonymizer()

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
    )
    if not results:
        return text

    operators = {
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
    }
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text


def scrub_pii(text: str) -> str:
    settings = get_settings()
    if not settings.enable_pii_scrubbing:
        return text

    if settings.pii_guard_provider.lower() == "presidio":
        try:
            return _presidio_scrub_pii(text)
        except Exception as exc:
            logger.warning("Presidio PII scrub failed, using regex", error=str(exc))

    return _regex_scrub_pii(text)


def _rebuff_injection_detect(text: str) -> bool:
    settings = get_settings()
    try:
        from rebuff import Rebuff

        rebuff = Rebuff(api_key=settings.rebuff_api_key or None)
        result = rebuff.detect_injection(text)
        if isinstance(result, dict):
            flagged = result.get("injection") or result.get("is_injection") or result.get("detected")
            score = result.get("score") or result.get("confidence") or 0.0
        else:
            flagged = getattr(result, "injection", None) or getattr(result, "is_injection", None)
            score = getattr(result, "score", None) or getattr(result, "confidence", None) or 0.0
        if flagged is None:
            return score >= settings.rebuff_block_threshold
        return bool(flagged) or score >= settings.rebuff_block_threshold
    except Exception as exc:
        logger.warning("Rebuff injection detection failed", error=str(exc))
        return False


@lru_cache
def _get_local_injection_pipeline():
    from transformers import pipeline

    settings = get_settings()
    return pipeline(
        "text-classification",
        model=settings.local_injection_model,
        truncation=True,
    )


def _local_injection_detect(text: str) -> bool:
    settings = get_settings()
    try:
        classifier = _get_local_injection_pipeline()
        results = classifier(text)
        if isinstance(results, list) and results:
            result = results[0]
        else:
            result = results

        label = str(result.get("label", "")).upper()
        score = float(result.get("score", 0.0))

        injection_labels = {"INJECTION", "PROMPT_INJECTION", "JAILBREAK", "UNSAFE", "MALICIOUS"}
        if any(tag in label for tag in injection_labels):
            return score >= settings.local_injection_block_threshold

        if any(tag in label for tag in {"SAFE", "BENIGN", "ALLOW"}):
            return False

        return score >= settings.local_injection_block_threshold
    except Exception as exc:
        logger.warning("Local injection detection failed", error=str(exc))
        return False


def validate_input(question: str) -> str:
    """
    Validates and sanitises user input.
    Returns the cleaned question or raises GuardrailViolationError.
    """
    settings = get_settings()

    if len(question) > settings.max_input_chars:
        raise GuardrailViolationError(
            f"Input exceeds maximum length of {settings.max_input_chars} characters."
        )

    if settings.enable_prompt_injection_guard:
        if settings.injection_guard_provider.lower() == "rebuff":
            if _rebuff_injection_detect(question):
                logger.warning("Prompt injection detected (Rebuff)", snippet=question[:100])
                raise GuardrailViolationError("Input contains disallowed content.")

        if settings.injection_guard_provider.lower() == "local":
            if _local_injection_detect(question):
                logger.warning("Prompt injection detected (local)", snippet=question[:100])
                raise GuardrailViolationError("Input contains disallowed content.")

        if _INJECTION_RE.search(question):
            logger.warning("Prompt injection attempt detected", snippet=question[:100])
            raise GuardrailViolationError("Input contains disallowed content.")

    cleaned = scrub_pii(question)

    if cleaned != question:
        logger.info("PII scrubbed from input")

    return cleaned
