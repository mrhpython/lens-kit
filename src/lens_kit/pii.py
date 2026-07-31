"""Deterministic PII detection — regex-first, no LLM, stdlib-only.

Runs BEFORE any LLM call in the gate (Lens 0). Catches the structured
patterns an LLM misses unreliably: API keys, credit cards, NI numbers,
SSNs, credentials in code, connection strings.

Why this layer exists: in the author's internal eval (2026-03), the
LLM-driven Rights lens caught only ~50% of planted structured-PII
patterns; these regexes catch 100% of them. The LLM Rights lens remains
the second pass for context-dependent judgment (dark patterns,
manipulation, ethical concerns) — and for emails/phone numbers, which are
often legitimate in deliverables (contact lines) and therefore stay
warning-severity here and NEVER fail the gate deterministically.

Halt semantics: the five types listed in ``_HALT_TYPES`` set ``halt=True``
— the gate stops before the text is sent to any provider. Everything else
is reported via ``scan()``/``scrub()`` and the ``lens-kit scrub`` CLI.
"""

import re
from dataclasses import dataclass, field


@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int
    severity: str = "critical"


@dataclass
class PIIScanResult:
    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    scrubbed_text: str = ""
    halt: bool = False
    halt_reason: str = ""


# ── Patterns ─────────────────────────────────────────────────────
# Order matters: more specific patterns first to avoid partial matches.

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # UK National Insurance number: 2 letters, 6 digits, 1 letter (A-D suffix)
    # Broad match — for PII purposes, catch ANY 2-letter + 6-digit + letter
    # pattern rather than validating HMRC prefix rules. Over-match is safer.
    ("ni_number", re.compile(
        r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b',
        re.IGNORECASE
    ), "critical"),

    # Credit card numbers (Visa, MC, Amex, Discover) — with optional spaces/dashes
    ("credit_card", re.compile(
        r'\b(?:'
        r'4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'  # Visa
        r'|5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'  # MasterCard
        r'|3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}'  # Amex
        r'|6(?:011|5\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'  # Discover
        r')\b'
    ), "critical"),

    # US SSN: 3-2-4 digit pattern
    ("ssn", re.compile(
        r'\b\d{3}[\s-]\d{2}[\s-]\d{4}\b'
    ), "critical"),

    # API keys — common formats (sk-, sk_live_, xai-, ghp_, AKIA, etc.)
    ("api_key", re.compile(
        r'\b(?:'
        r'sk-[a-zA-Z0-9_-]{20,}'  # Anthropic, OpenAI
        r'|sk_(?:live|test)_[a-zA-Z0-9]{16,}'  # Stripe-style
        r'|xai-[a-zA-Z0-9_-]{20,}'  # xAI
        r'|ghp_[a-zA-Z0-9]{36,}'  # GitHub PAT
        r'|AKIA[A-Z0-9]{16}'  # AWS access key
        r'|glpat-[a-zA-Z0-9_-]{20,}'  # GitLab PAT
        r')\b'
    ), "critical"),

    # Generic secret/password in assignment (key=value or key: value)
    # Matches standalone words AND env var suffixes (_PASS, _SECRET, etc.)
    ("credential", re.compile(
        r'(?:\w*(?:password|passwd|pass|secret|token|api_key|apikey|auth_token|access_token|private_key)\w*)'
        r'\s*[=:]\s*["\']?[^\s"\']{8,}["\']?',
        re.IGNORECASE
    ), "critical"),

    # Database connection strings with credentials
    ("connection_string", re.compile(
        r'(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s]+:[^@\s]+@[^\s]+',
        re.IGNORECASE
    ), "critical"),

    # Email addresses
    ("email", re.compile(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    ), "warning"),

    # UK phone numbers: +44, 07, 01, 02
    # Note: \b doesn't work before + (non-word char), use (?<!\w) instead
    ("phone_uk", re.compile(
        r'(?<!\w)(?:\+44\s?|0)(?:7\d{3}|\d{3,4})\s?\d{3}\s?\d{3,4}\b'
    ), "warning"),

    # US phone numbers (same +\b pitfall as phone_uk: (?<!\w), not \b,
    # so the "+1 (" prefix lands inside the match and gets redacted too)
    ("phone_us", re.compile(
        r'(?<!\w)(?:\+1[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b'
    ), "warning"),

    # UK UTR (Unique Taxpayer Reference): 10 digits
    ("utr", re.compile(
        r'\bUTR[\s:]*\d{10}\b',
        re.IGNORECASE
    ), "critical"),

    # Destructive commands
    ("destructive_cmd", re.compile(
        r'\brm\s+-rf\s+/',
        re.IGNORECASE
    ), "critical"),
]

# Critical types that stop the gate before any provider call. Deliberately
# NOT everything critical: destructive_cmd is a content hazard, not a
# secret — it fails review, it doesn't need to block the LLM from seeing
# the text.
_HALT_TYPES = ("ni_number", "ssn", "credit_card", "credential", "api_key")

# Context patterns that indicate already-scrubbed text (skip these)
_REDACTED_PATTERN = re.compile(r'\[REDACTED\]', re.IGNORECASE)


def scan(text: str) -> PIIScanResult:
    """Scan text for PII using regex patterns. No LLM calls."""
    matches: list[PIIMatch] = []
    halt = False
    halt_reason = ""

    for pii_type, pattern, severity in _PATTERNS:
        for m in pattern.finditer(text):
            matched_value = m.group()

            # Skip if this region is already [REDACTED]
            # Check surrounding context for redaction markers
            context_start = max(0, m.start() - 20)
            context_end = min(len(text), m.end() + 20)
            context = text[context_start:context_end]
            if _REDACTED_PATTERN.search(context):
                continue

            # Skip email-like patterns in generic descriptions
            if pii_type == "email" and matched_value.startswith("example@"):
                continue

            matches.append(PIIMatch(
                pii_type=pii_type,
                value=matched_value,
                start=m.start(),
                end=m.end(),
                severity=severity,
            ))

            if severity == "critical" and pii_type in _HALT_TYPES:
                halt = True
                if not halt_reason:
                    halt_reason = f"Critical PII detected: {pii_type}"

    return PIIScanResult(
        has_pii=len(matches) > 0,
        matches=matches,
        scrubbed_text=scrub(text) if matches else text,
        halt=halt,
        halt_reason=halt_reason,
    )


def scrub(text: str, replacement: str = "[REDACTED]") -> str:
    """Replace all PII patterns with [REDACTED]. Deterministic, no LLM."""
    result = text
    all_matches: list[tuple[int, int, str]] = []

    for pii_type, pattern, severity in _PATTERNS:
        for m in pattern.finditer(result):
            context_start = max(0, m.start() - 20)
            context_end = min(len(result), m.end() + 20)
            context = result[context_start:context_end]
            if _REDACTED_PATTERN.search(context):
                continue
            all_matches.append((m.start(), m.end(), pii_type))

    # Sort by start position descending to replace from end first
    all_matches.sort(key=lambda x: x[0], reverse=True)

    for start, end, pii_type in all_matches:
        result = result[:start] + replacement + result[end:]

    return result
