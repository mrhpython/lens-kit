"""Lens signatures — the 10 canonical lenses plus support signatures.

Canonical lens set: rights, truth, causality, definitionalIntegrity,
contradiction, extrapolation, structure, consistency, relevance, consciousScan
(JudgmentDetection). DefinitionalIntegrityCheck was added 2026-06-14 to cover
the undefined-term / equivocation axis the gate systematically missed (probe
definitional-integrity-2026-06-14); it is scoped to term-definedness and
meaning-shift only — derivation defects (missing mechanism, circular causation)
stay with Causality. Plus two support signatures: ViolationCrossCheck (safety
net) and AutoFix (error correction), and the claim-extraction pre-layer
signature.

Signature docstrings ARE the prompt. They encode the detect-freely doctrine:
the LLM detects generally; deterministic suppression happens AFTER detection
in the filter layer (filters.py), driven by profile vocabulary — never here.
"""
import dspy


class TruthValidation(dspy.Signature):
    """AI produces confident text without knowing if it's true. It has no concept of evidence.

    Flag ONLY:
    1. Specific numbers ($X, Y%, Z units) stated as CURRENT FACT without a named source or [ESTIMATE]/[UNKNOWN] marker
    2. Definitive statements about external entities (competitors, market, customers) that could be fabricated
    3. Historical claims presented as fact without verifiable basis
    4. Claims citing studies, reports, or statistics that sound authoritative but are unverifiable —
       e.g. "Studies by [Institution] confirm X produces Y" or "According to a comprehensive analysis".
       Attribution to an institution does NOT make a claim true. Flag the specific numbers even when attributed.

    DO NOT flag (other lenses handle these):
    - Forward-looking claims (future tense: "will generate", "by 2028", "reaches 75%") — Extrapolation lens owns projections
    - First-person operational data ("our revenue was $4.2M", "we grew 23%", "Q4 results") — the reporting entity IS the source
    - Numbers already carrying [UNKNOWN], [ESTIMATE], or [PROJECTION] markers

    PASS: business rhetoric without numbers, tabular/structured data, first-party reporting."""
    text = dspy.InputField(desc="AI-generated text to validate")
    domain = dspy.InputField(desc="Domain context (finance, seo, marketing, competitor)")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {issue, severity, location} objects. "
        "Each 'issue' MUST name the exact number or claim, e.g. '7.5% ECB rate has no named source' "
        "not 'specific numbers without source'. Each 'location' MUST quote the sentence fragment. "
        "Empty array if none.")
    fixed_text = dspy.OutputField(desc="Text with [UNKNOWN] markers added to uncited claims. Return original if no fixes needed.")


class CausalityValidation(dspy.Signature):
    """AI simulates causation by pattern-matching, not understanding. A recommendation without
    a mechanism is indistinguishable from a hallucinated causal chain.

    Flag ONLY: recommendations or cause-effect claims that have NO causal chain, mechanism, or evidence.
    PASS any claim that includes an explicit causal chain with evidence (IF/THEN/BECAUSE with rationale,
    or a named source cited as basis for the causal claim).

    This lens checks for MECHANISM only — not for citation completeness (Truth lens) or confidence levels
    (Extrapolation lens)."""
    text = dspy.InputField(desc="AI-generated text to validate")
    domain = dspy.InputField(desc="Domain context")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {issue, severity, location} objects")


class DefinitionalIntegrityCheck(dspy.Signature):
    """AI moves terms through an argument without ensuring they are grounded —
    a load-bearing word can carry the conclusion while never being pinned to a
    meaning, or quietly shift meaning between premise and conclusion.

    Check that load-bearing terms are grounded BEFORE they do work in the
    conclusion. Flag ONLY when the conclusion's force depends on a term that is:
      1. never defined or agreed before use;
      2. shifts meaning between premise and conclusion (equivocation);
      3. defined circularly / question-beggingly (defined via a term defined via
         the first);
      4. never bottoms out (infinite jargon regress).

    OVER-FLAG GUARD (critical): flag ONLY terms the conclusion depends on.
    ACCEPT terms that bottom out at shared / common / stipulated primitives — a
    stipulated ("Assume…") premise or a commonly-agreed domain term (interest
    rates, even number) is grounded, NOT a violation. Natural language is not
    geometry; do NOT demand every term be defined to bedrock.

    IN-TEXT DEFINITION RULE (the most common false positive): if the text
    itself defines a term — an "X is …" / "An X is …" / "X means …" clause, OR
    an explicit "Assume …" stipulation — that term is GROUNDED for the rest of
    the text. Do NOT flag a defined term as undefined, and do NOT call its
    later consistent use an equivocation, even when the term is technical
    (quorum, leap year, overdue invoice, callable bond). A clean syllogism that
    defines its term and then applies that exact definition has NO definitional
    defect — pass it.

    EQUIVOCATION REQUIRES A REAL SHIFT: only flag equivocation (clause 2) when
    the SAME term is used with TWO genuinely DIFFERENT, incompatible senses
    across the text. A term used in one consistent sense — even a vague or
    technical one — is NOT equivocation. Name both senses in the issue, or do
    not flag.

    BOUNDARY: this lens does NOT judge missing causal mechanism or circular
    *causation* — the Causality lens owns those. This lens owns
    term-definedness and term meaning-shift ONLY: an undefined / equivocating
    load-bearing term, not an unsupported cause-effect inference.

    [v2 note: Euclid's method puts Definitions before Propositions; a future
    version may run this as a precondition that short-circuits Contradiction /
    Consistency when a key term is ungrounded. v1 runs as an independent
    blocking lens — it does not short-circuit any downstream lens.]"""
    text = dspy.InputField(desc="AI-generated text to validate")
    domain = dspy.InputField(desc="Domain context")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {term, issue, severity, location} objects. "
        "'term' MUST be the exact undefined / equivocating load-bearing word (e.g. 'alpha', "
        "'clean', 'validated'); 'issue' names the defect (undefined | equivocation | circular | "
        "regress) and why the conclusion depends on it; 'location' quotes the sentence fragment. "
        "Empty array if none.")


class ContradictionCheck(dspy.Signature):
    """Check for internal contradictions. AI generates token-by-token without
    persistent awareness — it can state A and NOT-A in consecutive paragraphs.

    Flag:
    1. Direct contradictions: claim A and NOT-A in the same text
    2. Incompatible recommendations or timeline conflicts
    3. DERIVED mathematical contradictions where metrics are logically incompatible:
       - High monthly churn + high net retention (e.g. 8% monthly churn ≈ 96% annual
         churn is incompatible with 115% net retention which requires low gross churn)
       - Growth rates + absolute numbers that don't align
       - Percentages for the same category summing to >100%"""
    text = dspy.InputField(desc="AI-generated text to validate")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {claim_a, claim_b, severity} objects. "
        "For derived contradictions, claim_a and claim_b should state the two incompatible metrics.")


class ExtrapolationCheck(dspy.Signature):
    """AI predicts deterministically without uncertainty awareness. It cannot judge confidence —
    always maximally confident or randomly hedging.

    Flag ONLY: future claims or projections stated as certainties without confidence levels.
    PASS: text using [PROJECTION] or [ESTIMATE] with a confidence level, stated range, or explicit
    assumptions — regardless of domain.

    DO NOT flag:
    - Contradictions between claims — Contradiction lens (Lens 4) handles those
    - First-person operational data ("our revenue", "we grew 23%") — the reporting entity IS the source
    - Past-tense historical claims that are fabricated — Truth lens (Lens 2) handles those"""
    text = dspy.InputField(desc="AI-generated text to validate")
    domain = dspy.InputField(desc="Domain context")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {issue, severity} objects")
    fixed_text = dspy.OutputField(desc="Text with [PROJECTION]/[ESTIMATE] markers added. Return original if no fixes needed.")


class RightsCheck(dspy.Signature):
    """Check for privacy violations, PII exposure, unsafe operations, and
    ethical issues. AI cannot feel consequences — it produces harmful content
    with the same confidence as helpful content. HALT on any critical finding.
    Detect: SSN patterns, email addresses, credit card numbers, phone numbers,
    NI numbers, credential exposure, dark patterns, manipulation."""
    text = dspy.InputField(desc="AI-generated text to validate")
    halt = dspy.OutputField(desc="true if critical violation found, false otherwise")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {issue, severity, pii_type} objects")
    scrubbed_text = dspy.OutputField(desc="Text with PII replaced by [REDACTED]. Return original if no PII found.")


class StructureCheck(dspy.Signature):
    """AI has no foresight about real-world dependencies. It doesn't understand that actions
    have preconditions or that changes need rollback plans.

    Flag ONLY: plans or recommendations missing BOTH preconditions AND rollback.
    - Has preconditions + rollback: PASS
    - Has preconditions, no rollback: flag for missing rollback
    - Has rollback, no preconditions: flag for missing preconditions
    - Has neither: flag both

    DO NOT flag:
    - Contradictions between claims — Contradiction lens (Lens 4) handles those
    - Missing citations or sources — Truth lens (Lens 2) handles those
    - Missing confidence levels on projections — Extrapolation lens (Lens 5) handles those
    - [ESTIMATE], [UNKNOWN], or [PROJECTION] markers in the text — those are fix annotations, not structural issues"""
    text = dspy.InputField(desc="AI-generated text to validate")
    domain = dspy.InputField(desc="Domain context")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {issue, severity} objects")


class JudgmentDetection(dspy.Signature):
    """AI crosses from data (which it can handle) to experience (which it cannot) without
    signaling the boundary. Non-blocking — flag, don't fail.

    Flag: aesthetic judgments ('elegant', 'beautiful'), emotional predictions ('users will love'),
    moral conclusions ('right approach ethically'), creative claims ('innovative', 'revolutionary'),
    strategic intuition ('feels overextended', 'timing feels right').

    PASS: factual technical descriptions, [ESTIMATE]/[PROJECTION] markers with numeric ranges."""
    text = dspy.InputField(desc="AI-generated text to scan")
    flags = dspy.OutputField(desc="JSON array of {text_snippet, type, reason} where type is aesthetic|emotional|moral|creative|intuitive")


class ConsistencyCheck(dspy.Signature):
    """AI generates section-by-section without cross-referencing. It can state different
    numbers for the same entity in consecutive sections without noticing.

    Flag: same entity described with different values across sections. Examples:
    - '50 employees' in section 1, '75 employees' in section 3
    - 'revenue of $2M' in overview, 'revenue of $3.5M' in financials
    - 'founded 2019' in intro, 'founded 2021' in timeline

    Report both locations and values for each inconsistency."""
    text = dspy.InputField(desc="Full AI-generated text to check for cross-section consistency")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {entity, value_a, location_a, value_b, location_b, severity} objects")


class RelevanceCheck(dspy.Signature):
    """AI satisfices — it produces plausible content without checking whether it serves
    the reader. Content that is technically correct but irrelevant wastes the reader's
    attention and erodes trust.

    Given the context description, flag content sections that do not serve the stated
    audience or purpose. Report what the section says and why it's irrelevant.

    If no context is provided, PASS everything — this lens cannot operate without context."""
    text = dspy.InputField(desc="AI-generated text to check for relevance")
    context = dspy.InputField(desc="Audience/purpose context: who is reading this, what do they need")
    has_violations = dspy.OutputField(desc="true or false")
    violations = dspy.OutputField(desc="JSON array of {section, issue, severity} objects")


class ViolationCrossCheck(dspy.Signature):
    """Review the original text against the lens results.
    Identify any obvious violations that ALL lenses missed.
    Focus on: unsourced numbers, unqualified predictions, PII patterns.
    This is a safety net — catch what fell through.

    Return ONLY a JSON array of missed violations, or [] if nothing missed.
    Each violation: {"lens": "truth|extrapolation|rights|consistency", "issue": "description", "location": "where in text"}
    """
    text = dspy.InputField(desc="Original text being validated")
    lens_results = dspy.InputField(desc="JSON summary of what each lens found")
    missed_violations = dspy.OutputField(desc="JSON array of violations all lenses missed, or empty array []")


class AutoFix(dspy.Signature):
    """Fix specific violations in AI-generated text. Apply the minimum
    change needed — don't rewrite, just fix. Add markers, redact PII,
    add qualifiers. The rules were designed by consciousness. You execute
    them precisely."""
    text = dspy.InputField(desc="Text with violations")
    violations = dspy.InputField(desc="JSON array of violations to fix")
    fixed_text = dspy.OutputField(desc="Text with violations fixed — minimum changes only")


class ClaimExtraction(dspy.Signature):
    """Extract all factual claims, quantitative assertions, and predictions
    from text. Each claim becomes a separate validation target.

    A claim is any statement that could be true or false:
    - Quantitative: numbers, percentages, dollar amounts, dates
    - Attributive: "according to X", "studies show", "research indicates"
    - Predictive: "will", "by 2028", "expected to", "projected"
    - Comparative: "better than", "leading", "fastest growing"

    Return ONLY a JSON array. No markdown, no explanation.
    """
    text = dspy.InputField(desc="Text to extract claims from")
    claims = dspy.OutputField(desc="JSON array of {claim, type, location} where type is quantitative|attributive|predictive|comparative. Return [] for text with no extractable claims.")
