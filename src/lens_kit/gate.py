"""LensGate — the composed 10-lens validation pipeline as one dspy.Module.

Extracted from the author's internal validation gate (not published).
Behavior-preserving for the sequential path; all
provider/domain specifics are externalized to a Profile (config.py).

Lens order and semantics:
  Rights runs FIRST and can HALT the pipeline (critical PII/harm).
  Truth, Causality, Contradiction, Extrapolation, Structure are core gates.
  ConsciousScan (JudgmentDetection) is non-blocking — flags, never fails.
  Consistency is a core gate (cross-section agreement).
  Relevance is warning-only and requires caller-supplied context.
  ViolationCrossCheck is a post-hoc safety net (warning-level injections).

Detect-freely / filter-deterministically split: LLM signatures stay general;
all deterministic suppression lives in filters.DeterministicFilters and is
driven by profile vocabulary.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import dspy

from .claims import ClaimExtractor
from .config import Profile
from .filters import DeterministicFilters
from .pii import scan as pii_scan
from . import signatures as S

CANONICAL_LENSES = (
    "rights", "truth", "causality", "definitionalIntegrity", "contradiction",
    "extrapolation", "structure", "consistency", "relevance", "consciousScan",
)


def validate_lens_labels(per_lens: dict, *, where: str = "") -> None:
    """Hard error on lens-label typos (fail closed — never a silent 0).

    A per_lens key outside CANONICAL_LENSES can never match gate output, so
    the example's labels never intersect the verdict and it silently scores
    0 — poisoning training metrics and eval summaries without a trace.
    Design choice: load-time ValueError, not a warning. A warning scrolls
    past in a 4,000-rollout compile; a typo'd gold label is a data bug the
    run must not spend money on.
    """
    unknown = [k for k in per_lens if k not in CANONICAL_LENSES]
    if unknown:
        loc = f" in {where}" if where else ""
        raise ValueError(
            f"Unknown lens label(s) {unknown}{loc} — canonical lenses: "
            f"{', '.join(CANONICAL_LENSES)}. A non-canonical label can never "
            f"match gate output and would silently score 0; fix the label "
            f"(label edits force a new run designation — see the RUNS.md rule)."
        )


# ── Result Types ─────────────────────────────────────────────────

@dataclass
class LensViolation:
    lens: str
    severity: str  # "critical" | "warning"
    issue: str
    auto_fixable: bool = False


@dataclass
class LensResult:
    """Structured gate verdict.

    Pass/fail has two INDEPENDENT fail paths:
    1. Severity escalation — any violation with severity "critical" or
       "high" fails the gate.
    2. per_lens override — if ANY lens other than relevance reports
       violations (``per_lens[name] is False``), the gate fails, even when
       every violation is warning-severity. Relevance is the only
       warning-only lens; consciousScan emits flags, never violations.

    On a Rights HALT the pipeline stops immediately: ``per_lens`` contains
    ONLY the "rights" entry. Downstream lenses never ran — absence from
    ``per_lens`` means "not evaluated", not "passed".
    """
    passed: bool
    halted: bool = False
    halt_reason: str = ""
    violations: list = field(default_factory=list)
    fixed_text: str = ""
    consciousness_flags: list = field(default_factory=list)
    per_lens: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # per-lens timing when include_timings=True

    def to_dict(self) -> dict:
        out = {
            "passed": self.passed,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "violations": [
                {"lens": v.lens, "severity": v.severity, "issue": v.issue}
                for v in self.violations
            ],
            "consciousness_flags": self.consciousness_flags,
            "per_lens": self.per_lens,
        }
        if self.timings:
            out["timings"] = self.timings
        return out


# ── Domain Auto-Detection ────────────────────────────────────────

def _format_conflict(v: dict) -> str:
    """Render a contradiction violation honestly.

    Intra-passage conflicts come back with claim_b empty or identical to
    claim_a, which the raw f-string rendered as "'X' vs 'X'" / "'X' vs ''".
    """
    a, b = v.get("claim_a", ""), v.get("claim_b", "")
    if not b or a == b:
        return f"CONFLICT (both conflicting clauses are inside this single passage): '{a}'"
    return f"CONFLICT: '{a}' vs '{b}'"


def detect_domain(text: str, domain_keywords: dict) -> str:
    """Auto-detect validation domain via profile keyword lists.

    ``domain_keywords`` maps domain name -> keyword list; insertion order is
    priority order. Empty mapping -> always 'general'.
    """
    text_lower = text.lower()
    for domain, keywords in domain_keywords.items():
        if any(str(w).lower() in text_lower for w in keywords):
            return domain
    return "general"


# ── Lens Gate Pipeline ───────────────────────────────────────────

class LensGate(dspy.Module):
    """Full 10-lens validation pipeline as a single DSPy Module.

    Args:
        profile:   Profile carrying vocabulary, domain rules, and domain
                   detection keywords. None -> empty profile (pure LLM
                   detection, no deterministic suppression).
        auto_fix:  Run the AutoFix error-correction pass on critical
                   auto-fixable violations.
        fast_mode: Use plain Predict (N=1) for Extrapolation instead of
                   BestOfN(N=3). Faster, less strict. (Truth is always N=1
                   since 2026-08-10 — see the constructor note.)
        parallel:  Run independent lenses concurrently per dependency stage
                   (default True). Proven verdict-identical to the sequential
                   reference (mocked byte-identical test + live holdout
                   non-inferiority, 2026-06-17). parallel=False selects the
                   unchanged sequential path (the equivalence oracle).
    """

    def __init__(self, profile: Profile | None = None, auto_fix: bool = True,
                 fast_mode: bool = False, parallel: bool = True):
        super().__init__()
        self.lens_profile = profile or Profile.empty()
        self.filters = DeterministicFilters(self.lens_profile.vocabulary)
        self.auto_fix_enabled = auto_fix
        self.fast_mode = fast_mode
        self.parallel = parallel

        # Core lens gates.
        # Truth: plain Predict (N=1) since 2026-08-10. It previously ran
        # BestOfN(N=3) with a strictness reward (more violations = better) —
        # designed when missing citations were the standard. Under the
        # adjudicated standard (label epoch v4: "uncited market figure =
        # clean") that reward is a false-positive amplifier, and every banked
        # eval number measures the N=1 path — the gate now runs what the
        # harness measures. _truth_strictness_reward is retained for
        # checkpoint-era compatibility. Extrapolation keeps BestOfN(N=3)
        # pending its own review; fast_mode = N=1 there.
        self.truth = dspy.Predict(S.TruthValidation)
        self.causality = dspy.Predict(S.CausalityValidation)
        # Definitional Integrity: independent blocking lens (plain Predict, like
        # causality — not BestOfN). Owns undefined-term + equivocation only;
        # derivation defects stay with causality. v1 has no precondition
        # short-circuit (see DefinitionalIntegrityCheck v2 note).
        self.definitional_integrity = dspy.Predict(S.DefinitionalIntegrityCheck)
        self.contradiction = dspy.Predict(S.ContradictionCheck)
        extrap_pred = dspy.Predict(S.ExtrapolationCheck)
        if fast_mode:
            self.extrapolation = extrap_pred
        else:
            self.extrapolation = dspy.BestOfN(
                module=extrap_pred,
                N=3,
                reward_fn=self._extrap_strictness_reward,
                threshold=1.0,
            )
        self.rights = dspy.Predict(S.RightsCheck)
        self.structure = dspy.Predict(S.StructureCheck)

        # ConsciousScan (non-blocking)
        self.judgment = dspy.Predict(S.JudgmentDetection)

        # Consistency (cross-section agreement)
        self.consistency = dspy.Predict(S.ConsistencyCheck)

        # Relevance (audience fit — requires context)
        self.relevance = dspy.Predict(S.RelevanceCheck)

        # Cross-check safety net (post-processing meta-lens)
        self.cross_check = dspy.Predict(S.ViolationCrossCheck)

        # Claim extraction pre-layer
        self.claim_extractor = ClaimExtractor()

        # Auto-fix (error correction). Always instantiated so the module's
        # parameter tree is identical regardless of the auto_fix flag —
        # keeps dspy checkpoint save/load symmetric. USE is gated on
        # self.auto_fix_enabled in forward().
        self.fixer = dspy.ChainOfThought(S.AutoFix)

    # ── parsing helpers ──────────────────────────────────────────

    @staticmethod
    def _parse_violations(raw) -> list:
        """Parse violation JSON from DSPy output. Always returns list[dict]."""
        try:
            parsed = []
            if isinstance(raw, str):
                raw = raw.strip()
                if raw.startswith("["):
                    parsed = json.loads(raw)
                else:
                    match = re.search(r'\[.*\]', raw, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group())
            if isinstance(raw, list):
                parsed = raw
            return [
                v if isinstance(v, dict) else {"issue": str(v), "severity": "warning"}
                for v in parsed
            ]
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _is_true(val) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "yes", "1")

    @staticmethod
    def _parse_cross_check(raw) -> list:
        """Parse cross-check results into violation list."""
        if not raw:
            return []
        cleaned = str(raw).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, dict) and "issue" in v]
            return []
        except (json.JSONDecodeError, ValueError):
            return []

    # ── BestOfN reward functions ─────────────────────────────────

    @staticmethod
    def _truth_strictness_reward(args: dict, pred) -> float:
        """Prefer the result that catches MORE violations (cap at 5 = 1.0)."""
        violations_str = getattr(pred, "violations", "[]")
        try:
            violations = json.loads(violations_str) if isinstance(violations_str, str) else violations_str
            count = len(violations) if isinstance(violations, list) else 0
        except (json.JSONDecodeError, TypeError):
            count = 0
        return min(count / 5.0, 1.0)

    @staticmethod
    def _extrap_strictness_reward(args: dict, pred) -> float:
        """Strictness by violation count plus bonus for projection markers."""
        violations_str = getattr(pred, "violations", "[]")
        try:
            violations = json.loads(violations_str) if isinstance(violations_str, str) else violations_str
            count = len(violations) if isinstance(violations, list) else 0
        except (json.JSONDecodeError, TypeError):
            count = 0

        score = min(count / 5.0, 1.0)
        result_lower = violations_str.lower() if isinstance(violations_str, str) else ""
        if "[projection]" in result_lower or "[estimate]" in result_lower:
            score = min(score + 0.1, 1.0)
        if "confidence" in result_lower:
            score = min(score + 0.05, 1.0)
        return score

    # ── pipeline ─────────────────────────────────────────────────

    def forward(self, text: str, domain: str = "general", context: str = "",
                include_timings: bool = False) -> LensResult:
        """Sequential 10-lens validation.

        include_timings=True records per-lens wall time in result.timings.

        Fail semantics (see LensResult): critical/high severity fails the
        gate, AND any non-relevance lens reporting violations fails the
        gate even at warning severity.
        """
        # ── Lens 0: deterministic PII pre-pass (regex, no LLM call) ──
        # Critical structured PII (cards, API keys, NI/SSN, credentials)
        # halts the gate BEFORE the text is sent to any provider — the
        # secret never leaves the process. Emails/phone numbers are context
        # judgment calls (contact lines are often legitimate) and stay with
        # the LLM Rights lens; use ``lens_kit.pii.scan`` or the
        # ``lens-kit scrub`` CLI for the full deterministic findings.
        pii = pii_scan(text)
        if pii.halt:
            result = LensResult(passed=False, halted=True,
                                halt_reason=pii.halt_reason,
                                fixed_text=pii.scrubbed_text)
            result.per_lens["rights"] = False
            result.violations = [
                LensViolation("rights", m.severity,
                              f"Deterministic PII ({m.pii_type}): '{m.value}'")
                for m in pii.matches if m.severity == "critical"]
            return result

        if self.parallel:
            return self._forward_parallel(text, domain, context, include_timings)
        return self._forward_sequential(text, domain, context, include_timings)

    def _forward_sequential(self, text: str, domain: str = "general", context: str = "",
                            include_timings: bool = False) -> LensResult:
        """Sequential reference path (unchanged semantics; split out of
        forward() so the PII pre-pass wraps both execution paths once)."""
        t_total_start = time.monotonic()
        timings = {}

        result = LensResult(passed=True, fixed_text=text)
        current_text = text
        all_violations = []
        flt = self.filters

        # Auto-detect domain when caller didn't specify one
        if domain == "general":
            detected = detect_domain(text, self.lens_profile.domain_detection)
            if detected != "general":
                domain = detected

        # Enrich domain context with profile strictness rules
        domain_key = domain.lower().strip() if domain else "general"
        domain_rules = self.lens_profile.domain_rules.get(domain_key, "")
        enriched_domain = f"{domain}\n\n{domain_rules}" if domain_rules else domain

        # ── Claim Extraction Pre-Layer ──
        # Claim context is ADDITIVE — appended to enriched_domain for
        # Truth, Extrapolation, and Consistency lenses only.
        t0 = time.monotonic()
        claims = self.claim_extractor.extract(text)
        routed = self.claim_extractor.route_claims(claims) if claims else {
            "truth": [], "extrapolation": [], "consistency": []}
        truth_claim_context = self.claim_extractor.format_for_lens(routed["truth"], "truth")
        extrap_claim_context = self.claim_extractor.format_for_lens(routed["extrapolation"], "extrapolation")
        timings["claim_extraction"] = round(time.monotonic() - t0, 3)

        truth_domain = enriched_domain + truth_claim_context
        extrap_domain = enriched_domain + extrap_claim_context

        # ── Lens 1: Rights (HALT gate — runs first) ──
        t0 = time.monotonic()
        rights = self.rights(text=current_text)
        timings["rights"] = round(time.monotonic() - t0, 3)
        rights_violations = self._parse_violations(rights.violations)
        result.per_lens["rights"] = not self._is_true(rights.has_violations)
        if self._is_true(rights.halt):
            result.passed = False
            result.halted = True
            result.halt_reason = (rights_violations[0].get("issue", "Rights violation")
                                  if rights_violations else "Critical rights violation")
            all_violations.extend([
                LensViolation("rights", "critical", v.get("issue", str(v)))
                for v in rights_violations
            ])
            result.violations = all_violations
            return result  # HALT — don't continue
        if self._is_true(rights.has_violations):
            if getattr(rights, "scrubbed_text", "") and rights.scrubbed_text != current_text:
                current_text = rights.scrubbed_text
            all_violations.extend([
                LensViolation("rights", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in rights_violations
            ])

        # ── Lens 2: Truth ──
        t0 = time.monotonic()
        truth = self.truth(text=current_text, domain=truth_domain)
        truth_violations = self._parse_violations(truth.violations)
        truth_violations = flt.filter_truth_whitelist(truth_violations)
        truth_violations = flt.filter_truth_tabular(current_text, truth_violations)
        truth_violations = flt.filter_truth_named_source(truth_violations, current_text)
        truth_violations = flt.filter_truth_first_party(truth_violations, current_text)
        truth_violations = flt.filter_truth_forward_looking(truth_violations, current_text)
        has_real_violations = bool(truth_violations) and self._is_true(truth.has_violations)
        result.per_lens["truth"] = not has_real_violations
        if has_real_violations:
            all_violations.extend([
                LensViolation("truth", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in truth_violations
            ])
            if getattr(truth, "fixed_text", "") and truth.fixed_text != current_text:
                current_text = truth.fixed_text
        timings["truth"] = round(time.monotonic() - t0, 3)

        # ── Lens 3: Causality ──
        t0 = time.monotonic()
        causality = self.causality(text=current_text, domain=domain_key)
        causality_violations = self._parse_violations(causality.violations)
        result.per_lens["causality"] = not self._is_true(causality.has_violations)
        if self._is_true(causality.has_violations):
            all_violations.extend([
                LensViolation("causality", v.get("severity", "warning"), v.get("issue", str(v)))
                for v in causality_violations
            ])
        timings["causality"] = round(time.monotonic() - t0, 3)

        # ── Lens 3b: Definitional Integrity ──
        # Independent blocking lens (mirrors causality). Owns undefined
        # load-bearing term + equivocation / meaning-shift; derivation defects
        # (missing mechanism, circular causation) stay with causality. Placed
        # AFTER causality and BEFORE contradiction so reporting order reflects
        # "definitions precede propositions" — v1 does NOT short-circuit
        # downstream lenses (that is the v2 precondition refinement).
        t0 = time.monotonic()
        definitional = self.definitional_integrity(text=current_text, domain=domain_key)
        definitional_violations = self._parse_violations(definitional.violations)
        result.per_lens["definitionalIntegrity"] = not self._is_true(definitional.has_violations)
        if self._is_true(definitional.has_violations):
            all_violations.extend([
                LensViolation("definitionalIntegrity", v.get("severity", "warning"), v.get("issue", str(v)))
                for v in definitional_violations
            ])
        timings["definitional_integrity"] = round(time.monotonic() - t0, 3)

        # ── Lens 4: Contradiction ──
        # Use ORIGINAL text, not current_text — Truth auto-fix inserts
        # [UNKNOWN] markers that leak into contradiction quotes.
        t0 = time.monotonic()
        contradiction = self.contradiction(text=text)
        contradiction_violations = self._parse_violations(contradiction.violations)
        # Scenario check on ORIGINAL text too — a Rights scrub of current_text
        # must not be able to strip scenario labels and re-activate
        # contradiction findings. (Divergence from the source gate, which
        # checked current_text against its own stated intent.)
        if flt.is_scenario_text(text):
            contradiction_violations = []
            has_contradiction = False
        else:
            has_contradiction = self._is_true(contradiction.has_violations)
        result.per_lens["contradiction"] = not has_contradiction
        if has_contradiction:
            result.passed = False  # Contradictions are always critical
            all_violations.extend([
                LensViolation("contradiction", "critical", _format_conflict(v))
                for v in contradiction_violations
            ])
        timings["contradiction"] = round(time.monotonic() - t0, 3)

        # ── Lens 5: Extrapolation ──
        # Use ORIGINAL text — Truth auto-fix markers cause cascading FPs.
        t0 = time.monotonic()
        extrapolation = self.extrapolation(text=text, domain=extrap_domain)
        extrap_violations = self._parse_violations(extrapolation.violations)
        extrap_violations = flt.filter_extrapolation_qualified(extrap_violations, text)
        extrap_violations = flt.filter_extrapolation_first_party(extrap_violations, text)
        extrap_violations = flt.filter_extrapolation_past_tense(extrap_violations, text)
        has_extrap = self._is_true(extrapolation.has_violations) and bool(extrap_violations)
        result.per_lens["extrapolation"] = not has_extrap
        if has_extrap:
            all_violations.extend([
                LensViolation("extrapolation", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in extrap_violations
            ])
            if getattr(extrapolation, "fixed_text", "") and extrapolation.fixed_text != current_text:
                current_text = extrapolation.fixed_text
        timings["extrapolation"] = round(time.monotonic() - t0, 3)

        # ── Lens 6: Structure ──
        # Only run on text containing plans/recommendations (with empty
        # profile vocabulary the trigger returns True — always run).
        t0 = time.monotonic()
        if flt.text_has_recommendations(text):
            structure = self.structure(text=text, domain=enriched_domain)  # ORIGINAL text
            structure_violations = self._parse_violations(structure.violations)
            structure_violations = flt.filter_structure_complete(structure_violations, text)
            structure_violations = flt.filter_structure_stated_basis(structure_violations, text)
            structure_violations = flt.filter_structure_citation_overlap(structure_violations)
            has_structure = self._is_true(structure.has_violations) and bool(structure_violations)
            result.per_lens["structure"] = not has_structure
            if has_structure:
                all_violations.extend([
                    LensViolation("structure", v.get("severity", "warning"), v.get("issue", str(v)))
                    for v in structure_violations
                ])
        else:
            result.per_lens["structure"] = True  # Auto-pass: no recommendations to check
        timings["structure"] = round(time.monotonic() - t0, 3)

        # ── Auto-Fix (Error Correction) ──
        t0 = time.monotonic()
        auto_fix_ran = False
        critical_violations = [v for v in all_violations if v.severity == "critical" and v.auto_fixable]
        if self.auto_fix_enabled and critical_violations:
            auto_fix_ran = True
            try:
                fix_input = json.dumps([{"lens": v.lens, "issue": v.issue} for v in critical_violations])
                fixed = self.fixer(text=current_text, violations=fix_input)
                if getattr(fixed, "fixed_text", ""):
                    current_text = fixed.fixed_text
            except Exception:
                pass  # Auto-fix failure is non-fatal
        timings["auto_fix"] = round(time.monotonic() - t0, 3)

        # ── Lens 8: Consistency (cross-section agreement) ──
        # Reads ORIGINAL text — auto-fix [UNKNOWN] markers would mask inconsistencies.
        t0 = time.monotonic()
        consistency_violations = []
        try:
            consistency = self.consistency(text=text)
            consistency_violations = self._parse_violations(consistency.violations)
            result.per_lens["consistency"] = not self._is_true(consistency.has_violations)
            if self._is_true(consistency.has_violations):
                all_violations.extend([
                    LensViolation("consistency",
                        v.get("severity", "warning"),
                        f"INCONSISTENT: {v.get('entity', '?')}: '{v.get('value_a', '')}' vs '{v.get('value_b', '')}'")
                    for v in consistency_violations
                ])
        except Exception:
            result.per_lens["consistency"] = True  # Skip on error, don't block
        timings["consistency"] = round(time.monotonic() - t0, 3)

        # ── Lens 9: Relevance (audience fit — warning only) ──
        t0 = time.monotonic()
        if context:
            try:
                relevance = self.relevance(text=current_text, context=context)
                relevance_violations = self._parse_violations(relevance.violations)
                result.per_lens["relevance"] = not self._is_true(relevance.has_violations)
                if self._is_true(relevance.has_violations):
                    all_violations.extend([
                        LensViolation("relevance", "warning", v.get("issue", str(v)))
                        for v in relevance_violations
                    ])
            except Exception:
                result.per_lens["relevance"] = True
        # No context = skip relevance lens silently
        timings["relevance"] = round(time.monotonic() - t0, 3)

        # ── Lens 7: ConsciousScan (non-blocking) ──
        t0 = time.monotonic()
        try:
            judgment = self.judgment(text=current_text)
            result.consciousness_flags = self._parse_violations(judgment.flags)
        except Exception:
            result.consciousness_flags = []
        timings["conscious_scan"] = round(time.monotonic() - t0, 3)

        # ── Cross-check safety net — catch what individual lenses missed ──
        t0 = time.monotonic()
        lens_summary = json.dumps({
            "truth": len(truth_violations) if truth_violations else 0,
            "extrapolation": len(extrap_violations) if extrap_violations else 0,
            "rights": "HALTED" if result.halted else (len(rights_violations) if rights_violations else 0),
            "contradiction": len(contradiction_violations) if contradiction_violations else 0,
            "consistency": len(consistency_violations) if consistency_violations else 0,
        }, default=str)
        try:
            cross_result = self.cross_check(text=text, lens_results=lens_summary)
            missed = self._parse_cross_check(cross_result.missed_violations)
            for v in missed:
                target = v.get("lens", "truth")
                issue = v.get("issue", "")
                location = v.get("location", "cross-check")
                all_violations.append(
                    LensViolation(target, "warning", f"{issue} (cross-check: {location})", True)
                )
        except Exception:
            pass  # Safety net should never crash the pipeline
        timings["cross_check"] = round(time.monotonic() - t0, 3)

        # ── Final result ──
        result.violations = all_violations
        result.fixed_text = current_text
        if any(v.severity in ("critical", "high") for v in all_violations):
            result.passed = False
        if any(not passed for lens, passed in result.per_lens.items() if lens != "relevance"):
            result.passed = False

        timings["total"] = round(time.monotonic() - t_total_start, 3)
        timings["auto_fix_ran"] = auto_fix_ran
        if include_timings:
            result.timings = timings

        return result

    # ── parallel path (proven verdict-identical to forward(); see plan) ──

    @staticmethod
    def _call_with_lm(predictor, kwargs, lm):
        """Run a predictor, re-entering the captured dspy LM context inside the
        worker thread (dspy LM is thread-local; workers don't inherit it).
        lm is None in no-LM test runs -> call directly."""
        if lm is None:
            return predictor(**kwargs)
        with dspy.context(lm=lm):
            return predictor(**kwargs)

    def _dispatch_stage(self, jobs, lm):
        if not jobs:
            return {}
        if len(jobs) == 1:
            name, predictor, kwargs = jobs[0]
            try:
                return {name: self._call_with_lm(predictor, kwargs, lm)}
            except Exception as e:
                return {name: e}
        results = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {ex.submit(self._call_with_lm, predictor, kwargs, lm): name
                    for name, predictor, kwargs in jobs}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    results[name] = e
        return results

    def _forward_parallel(self, text: str, domain: str = "general", context: str = "",
                          include_timings: bool = False) -> LensResult:
        """Stage-scheduled equivalent of forward(): independent lenses dispatch
        concurrently per stage, but raw predictions are folded into the result
        in the SAME canonical lens order as sequential. Verdict-identical."""
        t_total_start = time.monotonic()
        timings = {}
        result = LensResult(passed=True, fixed_text=text)
        current_text = text
        all_violations = []
        flt = self.filters
        captured_lm = getattr(dspy.settings, "lm", None)

        # Domain detection + enrichment (identical to sequential)
        if domain == "general":
            detected = detect_domain(text, self.lens_profile.domain_detection)
            if detected != "general":
                domain = detected
        domain_key = domain.lower().strip() if domain else "general"
        domain_rules = self.lens_profile.domain_rules.get(domain_key, "")
        enriched_domain = f"{domain}\n\n{domain_rules}" if domain_rules else domain

        # ── Stage A: claim_extraction ∥ Rights ──
        t0 = time.monotonic()
        stageA = self._dispatch_stage([
            ("rights", self.rights, dict(text=current_text)),
            ("claims", self.claim_extractor.extract, dict(text=text)),
        ], captured_lm)
        # Sequential runs claim_extraction (un-wrapped) BEFORE rights, so a
        # claim-extractor failure must PROPAGATE out of the gate, not be
        # swallowed to "no claims". Unwrap claims first to match that order.
        claims = self._unwrap(stageA["claims"])
        rights = self._unwrap(stageA["rights"])  # no try/except in sequential -> re-raise
        routed = self.claim_extractor.route_claims(claims) if claims else {
            "truth": [], "extrapolation": [], "consistency": []}
        truth_claim_context = self.claim_extractor.format_for_lens(routed["truth"], "truth")
        extrap_claim_context = self.claim_extractor.format_for_lens(routed["extrapolation"], "extrapolation")
        timings["claim_extraction"] = round(time.monotonic() - t0, 3)
        truth_domain = enriched_domain + truth_claim_context
        extrap_domain = enriched_domain + extrap_claim_context

        # Rights post-process (verbatim semantics from forward())
        t0 = time.monotonic()
        rights_violations = self._parse_violations(rights.violations)
        result.per_lens["rights"] = not self._is_true(rights.has_violations)
        if self._is_true(rights.halt):
            result.passed = False
            result.halted = True
            result.halt_reason = (rights_violations[0].get("issue", "Rights violation")
                                  if rights_violations else "Critical rights violation")
            all_violations.extend([
                LensViolation("rights", "critical", v.get("issue", str(v)))
                for v in rights_violations])
            result.violations = all_violations
            timings["rights"] = round(time.monotonic() - t0, 3)
            return result  # HALT — stages B–E never dispatch
        if self._is_true(rights.has_violations):
            if getattr(rights, "scrubbed_text", "") and rights.scrubbed_text != current_text:
                current_text = rights.scrubbed_text
            all_violations.extend([
                LensViolation("rights", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in rights_violations])
        timings["rights"] = round(time.monotonic() - t0, 3)

        # ── Stage B: Truth(ct₁) ∥ Contradiction(text) ∥ Extrapolation(text) ∥ Structure(text) ∥ Consistency(text) ──
        run_structure = flt.text_has_recommendations(text)
        jobsB = [
            ("truth", self.truth, dict(text=current_text, domain=truth_domain)),
            ("contradiction", self.contradiction, dict(text=text)),
            ("extrapolation", self.extrapolation, dict(text=text, domain=extrap_domain)),
            ("consistency", self.consistency, dict(text=text)),
        ]
        if run_structure:
            jobsB.append(("structure", self.structure, dict(text=text, domain=enriched_domain)))
        stageB = self._dispatch_stage(jobsB, captured_lm)

        # Truth post-process FIRST (its fixed_text feeds Stage C). Verbatim semantics.
        t0 = time.monotonic()
        truth = self._unwrap(stageB["truth"])
        truth_violations = self._parse_violations(truth.violations)
        truth_violations = flt.filter_truth_whitelist(truth_violations)
        truth_violations = flt.filter_truth_tabular(current_text, truth_violations)
        truth_violations = flt.filter_truth_named_source(truth_violations, current_text)
        truth_violations = flt.filter_truth_first_party(truth_violations, current_text)
        truth_violations = flt.filter_truth_forward_looking(truth_violations, current_text)
        has_real_violations = bool(truth_violations) and self._is_true(truth.has_violations)
        result.per_lens["truth"] = not has_real_violations
        if has_real_violations:
            all_violations.extend([
                LensViolation("truth", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in truth_violations])
            if getattr(truth, "fixed_text", "") and truth.fixed_text != current_text:
                current_text = truth.fixed_text
        timings["truth"] = round(time.monotonic() - t0, 3)

        # ── Stage C: Causality(ct₂) ∥ DefinitionalIntegrity(ct₂) ──
        stageC = self._dispatch_stage([
            ("causality", self.causality, dict(text=current_text, domain=domain_key)),
            ("definitional", self.definitional_integrity, dict(text=current_text, domain=domain_key)),
        ], captured_lm)

        # Canonical post-process order: causality, DI, contradiction, extrapolation, structure
        t0 = time.monotonic()
        causality = self._unwrap(stageC["causality"])
        causality_violations = self._parse_violations(causality.violations)
        result.per_lens["causality"] = not self._is_true(causality.has_violations)
        if self._is_true(causality.has_violations):
            all_violations.extend([
                LensViolation("causality", v.get("severity", "warning"), v.get("issue", str(v)))
                for v in causality_violations])
        timings["causality"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        definitional = self._unwrap(stageC["definitional"])
        definitional_violations = self._parse_violations(definitional.violations)
        result.per_lens["definitionalIntegrity"] = not self._is_true(definitional.has_violations)
        if self._is_true(definitional.has_violations):
            all_violations.extend([
                LensViolation("definitionalIntegrity", v.get("severity", "warning"), v.get("issue", str(v)))
                for v in definitional_violations])
        timings["definitional_integrity"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        contradiction = self._unwrap(stageB["contradiction"])
        contradiction_violations = self._parse_violations(contradiction.violations)
        if flt.is_scenario_text(text):
            contradiction_violations = []
            has_contradiction = False
        else:
            has_contradiction = self._is_true(contradiction.has_violations)
        result.per_lens["contradiction"] = not has_contradiction
        if has_contradiction:
            result.passed = False
            all_violations.extend([
                LensViolation("contradiction", "critical", _format_conflict(v))
                for v in contradiction_violations])
        timings["contradiction"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        extrapolation = self._unwrap(stageB["extrapolation"])
        extrap_violations = self._parse_violations(extrapolation.violations)
        extrap_violations = flt.filter_extrapolation_qualified(extrap_violations, text)
        extrap_violations = flt.filter_extrapolation_first_party(extrap_violations, text)
        extrap_violations = flt.filter_extrapolation_past_tense(extrap_violations, text)
        has_extrap = self._is_true(extrapolation.has_violations) and bool(extrap_violations)
        result.per_lens["extrapolation"] = not has_extrap
        if has_extrap:
            all_violations.extend([
                LensViolation("extrapolation", v.get("severity", "warning"), v.get("issue", str(v)), True)
                for v in extrap_violations])
            if getattr(extrapolation, "fixed_text", "") and extrapolation.fixed_text != current_text:
                current_text = extrapolation.fixed_text
        timings["extrapolation"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        if run_structure:
            structure = self._unwrap(stageB["structure"])
            structure_violations = self._parse_violations(structure.violations)
            structure_violations = flt.filter_structure_complete(structure_violations, text)
            structure_violations = flt.filter_structure_stated_basis(structure_violations, text)
            structure_violations = flt.filter_structure_citation_overlap(structure_violations)
            has_structure = self._is_true(structure.has_violations) and bool(structure_violations)
            result.per_lens["structure"] = not has_structure
            if has_structure:
                all_violations.extend([
                    LensViolation("structure", v.get("severity", "warning"), v.get("issue", str(v)))
                    for v in structure_violations])
        else:
            result.per_lens["structure"] = True
        timings["structure"] = round(time.monotonic() - t0, 3)

        # ── Stage D: Auto-fix (conditional on critical auto-fixable so far) ──
        t0 = time.monotonic()
        auto_fix_ran = False
        critical_violations = [v for v in all_violations if v.severity == "critical" and v.auto_fixable]
        if self.auto_fix_enabled and critical_violations:
            auto_fix_ran = True
            try:
                fix_input = json.dumps([{"lens": v.lens, "issue": v.issue} for v in critical_violations])
                fixed = self._call_with_lm(self.fixer, dict(text=current_text, violations=fix_input), captured_lm)
                if getattr(fixed, "fixed_text", ""):
                    current_text = fixed.fixed_text
            except Exception:
                pass
        timings["auto_fix"] = round(time.monotonic() - t0, 3)

        # Consistency post-process (call was in Stage B; try/except in sequential).
        t0 = time.monotonic()
        consistency_violations = []
        consistency = stageB["consistency"]
        if isinstance(consistency, Exception):
            result.per_lens["consistency"] = True
        else:
            try:
                consistency_violations = self._parse_violations(consistency.violations)
                result.per_lens["consistency"] = not self._is_true(consistency.has_violations)
                if self._is_true(consistency.has_violations):
                    all_violations.extend([
                        LensViolation("consistency", v.get("severity", "warning"),
                            f"INCONSISTENT: {v.get('entity', '?')}: '{v.get('value_a', '')}' vs '{v.get('value_b', '')}'")
                        for v in consistency_violations])
            except Exception:
                result.per_lens["consistency"] = True
        timings["consistency"] = round(time.monotonic() - t0, 3)

        # lens_summary uses counts available now (verbatim from sequential)
        lens_summary = json.dumps({
            "truth": len(truth_violations) if truth_violations else 0,
            "extrapolation": len(extrap_violations) if extrap_violations else 0,
            "rights": "HALTED" if result.halted else (len(rights_violations) if rights_violations else 0),
            "contradiction": len(contradiction_violations) if contradiction_violations else 0,
            "consistency": len(consistency_violations) if consistency_violations else 0,
        }, default=str)

        # ── Stage E: Relevance(ct₄) ∥ ConsciousScan(ct₄) ∥ Cross-check(text, summary) ──
        jobsE = [
            ("judgment", self.judgment, dict(text=current_text)),
            ("cross_check", self.cross_check, dict(text=text, lens_results=lens_summary)),
        ]
        if context:
            jobsE.append(("relevance", self.relevance, dict(text=current_text, context=context)))
        stageE = self._dispatch_stage(jobsE, captured_lm)

        # Canonical order: relevance, consciousScan, cross-check
        t0 = time.monotonic()
        if context:
            relevance = stageE["relevance"]
            if isinstance(relevance, Exception):
                result.per_lens["relevance"] = True
            else:
                try:
                    relevance_violations = self._parse_violations(relevance.violations)
                    result.per_lens["relevance"] = not self._is_true(relevance.has_violations)
                    if self._is_true(relevance.has_violations):
                        all_violations.extend([
                            LensViolation("relevance", "warning", v.get("issue", str(v)))
                            for v in relevance_violations])
                except Exception:
                    result.per_lens["relevance"] = True
        timings["relevance"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        judgment = stageE["judgment"]
        if isinstance(judgment, Exception):
            result.consciousness_flags = []
        else:
            try:
                result.consciousness_flags = self._parse_violations(judgment.flags)
            except Exception:
                result.consciousness_flags = []
        timings["conscious_scan"] = round(time.monotonic() - t0, 3)

        t0 = time.monotonic()
        cross_result = stageE["cross_check"]
        if not isinstance(cross_result, Exception):
            try:
                missed = self._parse_cross_check(cross_result.missed_violations)
                for v in missed:
                    target = v.get("lens", "truth")
                    issue = v.get("issue", "")
                    location = v.get("location", "cross-check")
                    all_violations.append(
                        LensViolation(target, "warning", f"{issue} (cross-check: {location})", True))
            except Exception:
                pass
        timings["cross_check"] = round(time.monotonic() - t0, 3)

        # ── Final result (verbatim from sequential) ──
        result.violations = all_violations
        result.fixed_text = current_text
        if any(v.severity in ("critical", "high") for v in all_violations):
            result.passed = False
        if any(not passed for lens, passed in result.per_lens.items() if lens != "relevance"):
            result.passed = False

        timings["total"] = round(time.monotonic() - t_total_start, 3)
        timings["auto_fix_ran"] = auto_fix_ran
        if include_timings:
            result.timings = timings
        return result

    @staticmethod
    def _unwrap(pred_or_exc):
        """Re-raise for lenses that sequential did NOT wrap in try/except."""
        if isinstance(pred_or_exc, Exception):
            raise pred_or_exc
        return pred_or_exc
