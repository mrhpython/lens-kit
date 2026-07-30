"""Deterministic filter layer — detect freely, filter deterministically.

The LLM lenses detect generally (signatures.py). This layer suppresses
known-false-positive classes AFTER detection, using vocabulary supplied by
the customer profile (config.py). No LLM calls happen here; every function
is pure and unit-testable offline.

Empty-vocabulary semantics (the empty-profile default):
- Every suppression list defaults to empty -> NOTHING is suppressed.
  An empty profile means pure LLM detection (more false positives, never
  fewer true positives).
- One deliberate exception: ``text_has_recommendations`` with all three
  trigger lists empty returns True (Structure lens runs on everything).
  Empty vocabulary must never silently disable a lens.
"""
from dataclasses import dataclass, field
import re


def _lower_all(items: list) -> list:
    return [s.lower() for s in items]


@dataclass
class LensVocabulary:
    """All deterministic-filter vocabulary, externalized from the gate.

    Ship-with-agency values live in profiles/agency-example.yaml. Every
    field defaults to empty (= no suppression). Fields map 1:1 to the
    ``vocabulary:`` block of a profile YAML.
    """
    # Truth lens
    truth_whitelist: list = field(default_factory=list)            # rhetoric that never carries numbers
    internal_source_phrases: list = field(default_factory=list)    # internal tool outputs trusted as self-sourced
    first_party_signals: list = field(default_factory=list)        # full-text gate for first-party filtering
    first_party_context: list = field(default_factory=list)        # narrow-window operational-reporting phrases
    external_claim_signals: list = field(default_factory=list)     # never-suppress markers
    future_signals: list = field(default_factory=list)             # forward-looking -> Extrapolation territory
    present_tense_override: list = field(default_factory=list)     # present-tense fabrication -> stays Truth

    # Contradiction lens
    scenario_labels: list = field(default_factory=list)            # bull case / bear case / scenario markers

    # Extrapolation lens
    past_tense_signals: list = field(default_factory=list)         # past-tense -> Truth territory
    future_counter_signals: list = field(default_factory=list)     # future markers that out-prioritize past-tense
    confidence_keywords: list = field(default_factory=list)        # already-qualified projections

    # Structure lens
    recommendation_signals: list = field(default_factory=list)     # imperative/plan language triggers
    financial_plan_signals: list = field(default_factory=list)     # financial projections-with-targets triggers
    marketing_plan_signals: list = field(default_factory=list)     # structured marketing-plan triggers
    precondition_keywords: list = field(default_factory=list)      # preconditions-present detection
    rollback_keywords: list = field(default_factory=list)          # rollback-present detection
    citation_overlap_signals: list = field(default_factory=list)   # citation complaints belong to Truth/Extrapolation
    stated_basis_phrases: list = field(default_factory=list)       # financial models with stated basis

    @classmethod
    def from_dict(cls, data: dict) -> "LensVocabulary":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown vocabulary keys: {sorted(unknown)}")
        return cls(**{k: list(v) for k, v in data.items()})

    def to_dict(self) -> dict:
        return {k: list(getattr(self, k)) for k in self.__dataclass_fields__}


class DeterministicFilters:
    """The gate's post-detection suppression layer, parameterized by vocabulary.

    Takes a DEFENSIVE COPY of the vocabulary at construction — mutating the
    caller's LensVocabulary (or Profile) afterwards does not change filter
    behavior. Calibration loops that vary vocabulary must build a new
    DeterministicFilters (or LensGate) per variant.
    """

    # Methodology markers — part of the lens method itself, not domain vocab.
    QUALIFICATION_MARKERS = ["[PROJECTION]", "[ESTIMATE]"]

    def __init__(self, vocab: LensVocabulary | None = None):
        source = vocab or LensVocabulary()
        self.vocab = LensVocabulary.from_dict(source.to_dict())  # deep copy via round-trip

    # ── shared helpers ───────────────────────────────────────────

    @staticmethod
    def get_violation_context(text: str, location: str, window: int = 100) -> str:
        """Get text context around a violation location for filter matching."""
        if not location:
            return text[:window]
        idx = text.lower().find(location.lower()[:30])
        if idx == -1:
            return text[:window]
        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(location) + window // 2)
        return text[start:end]

    @staticmethod
    def _violation_in_lines(violation: dict, line_set: set) -> bool:
        loc = violation.get("location", "")
        match = re.search(r'line\s*(\d+)', loc, re.IGNORECASE)
        if match:
            return int(match.group(1)) in line_set
        return False

    # ── Truth filters ────────────────────────────────────────────

    def filter_truth_whitelist(self, violations: list) -> list:
        """Remove truth violations matching whitelisted business rhetoric."""
        whitelist = _lower_all(self.vocab.truth_whitelist)
        if not whitelist:
            return violations
        filtered = []
        for v in violations:
            issue_lower = v.get("issue", str(v)).lower()
            if any(phrase in issue_lower for phrase in whitelist):
                continue
            filtered.append(v)
        return filtered

    def filter_truth_tabular(self, text: str, violations: list) -> list:
        """Suppress truth violations for tabular/structured data sections.

        Vocabulary-free: table detection is structural, not domain-specific.
        """
        tabular_lines = set()
        for i, line in enumerate(text.split('\n')):
            if line.count('|') >= 2 or re.match(r'^\s*\S+\s{2,}\S+\s{2,}\S+', line):
                tabular_lines.add(i)
        if not tabular_lines:
            return violations
        return [v for v in violations if not self._violation_in_lines(v, tabular_lines)]

    def filter_truth_named_source(self, violations: list, text: str) -> list:
        """Suppress truth violations for claims from INTERNAL/known data sources.

        Only internal tool output is trusted as self-sourced. External
        attributions are NOT suppressed — the attribution may be fabricated.
        """
        phrases = _lower_all(self.vocab.internal_source_phrases)
        if not phrases:
            return violations
        filtered = []
        for v in violations:
            location = v.get("location", "")
            context = self.get_violation_context(text, location, window=100)
            if any(p in context.lower() for p in phrases):
                continue
            filtered.append(v)
        return filtered

    def _filter_first_party(self, violations: list, text: str) -> list:
        """Shared first-party suppression used by Truth and Extrapolation."""
        signals = _lower_all(self.vocab.first_party_signals)
        text_lower = text.lower()
        if not signals or not any(sig in text_lower for sig in signals):
            return violations  # Not first-party text — no filtering

        fp_context = _lower_all(self.vocab.first_party_context)
        external = _lower_all(self.vocab.external_claim_signals)
        kept = []
        for v in violations:
            location = v.get("location", v.get("issue", ""))
            context = self.get_violation_context(text, location, window=80).lower()
            issue_lower = v.get("issue", str(v)).lower()
            # External-facing claims are never suppressed
            if any(sig in issue_lower for sig in external):
                kept.append(v)
                continue
            if any(fp in context for fp in fp_context):
                continue  # First-party operational data — entity is its own source
            kept.append(v)
        return kept

    def filter_truth_first_party(self, violations: list, text: str) -> list:
        return self._filter_first_party(violations, text)

    def filter_truth_forward_looking(self, violations: list, text: str) -> list:
        """Route forward-looking claims to Extrapolation; keep present-tense in Truth."""
        future = _lower_all(self.vocab.future_signals)
        present = _lower_all(self.vocab.present_tense_override)
        if not future and not present:
            return violations
        filtered = []
        for v in violations:
            issue_lower = v.get("issue", "").lower()
            location = v.get("location", v.get("issue", ""))
            # Step 1: check ISSUE TEXT for temporal signals (most reliable)
            if any(pt in issue_lower for pt in present):
                filtered.append(v)  # present-tense claim -> Truth
                continue
            if any(sig in issue_lower for sig in future):
                continue  # forward-looking claim -> Extrapolation
            # Step 2: no temporal markers in issue — narrow context as tiebreaker
            narrow = self.get_violation_context(text, location, window=40).lower()
            if any(pt in narrow for pt in present):
                filtered.append(v)
                continue
            if any(sig in narrow for sig in future):
                continue
            # Step 3: no temporal signals at all -> keep (Truth checks facts by default)
            filtered.append(v)
        return filtered

    # ── Contradiction filters ────────────────────────────────────

    def is_scenario_text(self, text: str) -> bool:
        """True if the input text uses explicit scenario labels.

        Differing numbers across labeled scenarios are analysis, not
        contradictions. Check is purely on input text.
        """
        labels = _lower_all(self.vocab.scenario_labels)
        if not labels:
            return False
        text_lower = text.lower()
        return any(label in text_lower for label in labels)

    # ── Extrapolation filters ────────────────────────────────────

    def filter_extrapolation_first_party(self, violations: list, text: str) -> list:
        return self._filter_first_party(violations, text)

    def filter_extrapolation_past_tense(self, violations: list, text: str) -> list:
        """Past-tense fabrications are Truth territory; future signals win ties."""
        past = _lower_all(self.vocab.past_tense_signals)
        future = _lower_all(self.vocab.future_counter_signals)
        if not past:
            return violations
        kept = []
        for v in violations:
            location = v.get("location", v.get("issue", ""))
            context = self.get_violation_context(text, location, window=60).lower()
            if any(fc in context for fc in future):
                kept.append(v)  # future takes priority — keep as extrapolation
                continue
            if any(sig in context for sig in past):
                continue
            kept.append(v)
        return kept

    def filter_extrapolation_qualified(self, violations: list, text: str) -> list:
        """Suppress extrapolation violations for already-qualified projections."""
        keywords = _lower_all(self.vocab.confidence_keywords)
        if not keywords:
            return violations
        filtered = []
        for v in violations:
            location = v.get("location", v.get("issue", ""))
            context = self.get_violation_context(text, location, window=150)
            has_marker = any(m in context for m in self.QUALIFICATION_MARKERS)
            has_confidence = any(kw in context.lower() for kw in keywords)
            if has_marker and has_confidence:
                continue
            filtered.append(v)
        return filtered

    # ── Structure filters ────────────────────────────────────────

    def text_has_recommendations(self, text: str) -> bool:
        """True if text contains plan/recommendation language needing
        precondition/rollback checks.

        With ALL trigger lists empty (empty profile): returns True —
        Structure runs on everything rather than being silently disabled.
        """
        rec = _lower_all(self.vocab.recommendation_signals)
        fin = _lower_all(self.vocab.financial_plan_signals)
        mkt = _lower_all(self.vocab.marketing_plan_signals)
        if not rec and not fin and not mkt:
            return True
        text_lower = text.lower()
        return (
            any(sig in text_lower for sig in rec)
            or any(sig in text_lower for sig in fin)
            or any(sig in text_lower for sig in mkt)
        )

    def filter_structure_complete(self, violations: list, text: str) -> list:
        """Override Structure to PASS if both preconditions and rollback present."""
        pre = _lower_all(self.vocab.precondition_keywords)
        rb = _lower_all(self.vocab.rollback_keywords)
        if not pre or not rb:
            return violations
        text_lower = text.lower()
        if any(kw in text_lower for kw in pre) and any(kw in text_lower for kw in rb):
            return []
        return violations

    def filter_structure_stated_basis(self, violations: list, text: str) -> list:
        """Suppress structure violations for financial models with stated basis."""
        phrases = _lower_all(self.vocab.stated_basis_phrases)
        if not phrases:
            return violations
        text_lower = text.lower()
        if any(p in text_lower for p in phrases):
            return [v for v in violations if "source" not in v.get("issue", "").lower()]
        return violations

    def filter_structure_citation_overlap(self, violations: list) -> list:
        """Remove Structure violations about missing citations/markers —
        Truth and Extrapolation own those."""
        signals = _lower_all(self.vocab.citation_overlap_signals)
        if not signals:
            return violations
        filtered = []
        for v in violations:
            issue_lower = v.get("issue", str(v)).lower()
            if any(sig in issue_lower for sig in signals):
                continue
            filtered.append(v)
        return filtered
