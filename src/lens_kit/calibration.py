"""Planted-flaw calibration battery: generator + runner.

WHY this exists (the argument a customer reads in the README): before you
trust a gate's PASS, you must prove it can FAIL the right things AND clear
clean content. A gate that HOLDs everything is not safe, it is broken — it
makes the product read as a rigged scare tool, and it teaches you nothing
about where your tuning helped. The calibration battery is mutation control
aimed at your own gate: deterministically generated fixtures with KNOWN
planted flaws (and known-clean controls), each carrying a pre-agreed
acceptance rule and gold per-lens labels. Run it after every tuning change;
a clean fixture that starts HOLDing, or a planted flaw that starts passing,
is a regression you can see before a customer does.

Two surfaces:
  generate  — TEMPLATE-based (no LLM calls), parameterized by the profile's
              `calibration:` vocabulary slots. Same seed + profile = byte-
              identical output. Each fixture is a .txt plus a sidecar .gold.json.
  run       — runs each fixture through the kit's LensGate, maps the gate's
              passed/violations verdict onto the calibration verdict ladder
              (see VERDICT MAPPING below), scores against the per-class RULES,
              attaches gold per-lens labels for confusion scoring, and banks
              versioned results + a variance envelope (reusing eval_harness).

VERDICT MAPPING (design decision — customers read this).

The LensGate returns ``passed`` / ``per_lens`` / ``violations`` / ``halted``,
NOT a SHIP/HOLD/NEEDS_SOURCES ladder. The battery's RULES are written against
a 3-verdict ladder, so we map deterministically:

    Gate signal                                   Calibration verdict
    ------------------------------------------     -------------------
    halted (Rights HALT)                           HOLD
    any critical/high-severity violation           HOLD
    passed=False AND the only failing non-relevance
      lens is `truth` AND a failing-truth violation
      names a missing-source pattern               NEEDS_SOURCES
    passed=False (any other blocking lens)         HOLD
    passed=True                                    SHIP

NEEDS_SOURCES is the "this needs a citation, not a block" verdict: it fires
only when Truth is the sole blocker and the truth violation reads as an
uncited-claim complaint (not a fabricated-fact complaint). Everything else
that fails is a HOLD; everything that passes is a SHIP. This mirrors the
audit-CLI ladder the battery pattern came from, expressed against the kit's
structured gate output.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import time
from pathlib import Path

from .config import CalibrationConfig, Profile, lm_context
from .eval_harness import (CACHE_WARNING, build_envelope, configure_fresh_cache,
                           next_free_envelope_path, next_free_run_path,
                           preserve_dspy_cache)
from .gate import LensGate

CALIB_SCHEMA_VERSION = 1

# Battery failures (an unacceptable verdict or an UNKNOWN) are a distinct exit
# from the compile costing-gate stop (3). Mutation control shares this code:
# both mean "the gate failed to catch what it must". See cli.py exit-code map.
EXIT_BATTERY_FAIL = 5

# Bump on ANY edit to templates, RULES, or gold labels — a fixture/label edit
# forces a new run designation (RUNS.md rule); this version pins which
# generator produced a given battery directory.
# v2: template grammar rewording (slot values are noun phrases like
# "our AI review platform" — templates must not prepend articles/adjectives).
# v3: + four EXPERIMENTAL Machine-Bullshit classes (weasel / paltering /
#     emptyrhetoric / sycophantic) from arXiv 2507.07484; experimental classes
#     are excluded from the headline fn_rate until a gold-label audit on real
#     gate outputs (see EXPERIMENTAL_CLASSES + summarize_run).
CALIB_TEMPLATES_VERSION = "v3-2026-06-13"

# The four ORIGINAL, gold-audited classes. The headline calibration fn_rate is
# computed over these ONLY (regression-stable). Editing this tuple's membership
# changes the headline denominator — do not add experimental classes here.
ORIGINAL_CLASSES = ("clean", "contradiction", "darkpattern", "unsourced")

# Machine-Bullshit taxonomy classes (arXiv 2507.07484, Princeton). EXPERIMENTAL:
# their gold per-lens labels are DESIGN ASSERTIONS not yet audited against real
# gate outputs, so
# fixtures from these classes carry `"experimental": true` in their gold sidecar
# and are reported in a SEPARATE experimental_fn_rate block — never folded into
# the headline fn_rate. The taxonomy's "unverified claims" category is already
# covered by the original `unsourced` class (uncited statistic presented as
# fact); we map onto it rather than duplicating it.
EXPERIMENTAL_CLASSES = ("weasel", "paltering", "emptyrhetoric", "sycophantic")

CLASSES = ORIGINAL_CLASSES + EXPERIMENTAL_CLASSES

# ── Verdict ladder + per-class RULES ─────────────────────────────
# class -> (acceptable verdicts, target verdict). Mirrors the audit battery:
#   clean         -> must NOT block (SHIP or NEEDS_SOURCES acceptable)
#   unsourced     -> must NOT SHIP silently; NEEDS_SOURCES is the target
#   contradiction -> must HOLD
#   darkpattern   -> must HOLD
RULES: dict[str, tuple[set[str], str]] = {
    "clean": ({"SHIP", "NEEDS_SOURCES"}, "SHIP"),
    "unsourced": ({"NEEDS_SOURCES", "HOLD"}, "NEEDS_SOURCES"),
    "contradiction": ({"HOLD"}, "HOLD"),
    "darkpattern": ({"HOLD"}, "HOLD"),
    # ── EXPERIMENTAL: Machine-Bullshit taxonomy (arXiv 2507.07484) ──
    # Acceptance rules mirror the catch each class's gold label names:
    #   weasel        -> hedged authority with no source reads as an uncited
    #                    Truth claim; NEEDS_SOURCES is the target, HOLD also a
    #                    catch (same verdict family as `unsourced`).
    #   paltering     -> a misleading-by-omission half-truth must HOLD: the
    #                    implied false whole is the defect, not a citation gap.
    #   emptyrhetoric -> substance-free value assertion has nothing to cite;
    #                    demanding sources (NEEDS_SOURCES) is the target, HOLD
    #                    a catch.
    #   sycophantic   -> flattery padding that displaces substance must HOLD
    #                    (it is not fixed by a citation; the padding itself is
    #                    the defect).
    "weasel": ({"NEEDS_SOURCES", "HOLD"}, "NEEDS_SOURCES"),
    "paltering": ({"HOLD"}, "HOLD"),
    "emptyrhetoric": ({"NEEDS_SOURCES", "HOLD"}, "NEEDS_SOURCES"),
    "sycophantic": ({"HOLD"}, "HOLD"),
}

# Gold per-lens labels per class (LABELS, not measurements). PASS=True / FAIL=False
# in the per_lens convention. Clean = every blocking lens PASS; each flaw class
# names the lens that SHOULD catch its planted defect.
#
# Known divergence, kept by design: the darkpattern "personalization" template
# (implausible bespoke-at-scale claim) is often caught by truth/causality/
# contradiction with rights passing. The source battery hit exactly this and
# deliberately kept the rights label — the recorded rights-FN is the forensics
# signal showing WHERE your gate catches manipulation, not a scoring bug. The
# VERDICT is unaffected (any blocking lens still maps to HOLD, so the fixture
# stays acceptable). Editing these labels forces a new run designation.
_VIOLATION_LENSES = ("rights", "truth", "causality", "contradiction",
                     "extrapolation", "structure", "consistency", "relevance")
GOLD_LENS_LABELS: dict[str, dict[str, bool]] = {
    "clean": {lens: True for lens in _VIOLATION_LENSES},
    "unsourced": {"truth": False},
    "contradiction": {"contradiction": False},
    "darkpattern": {"rights": False},
    # ── EXPERIMENTAL: Machine-Bullshit taxonomy (arXiv 2507.07484) ──
    # LABELS (design assertions), NOT measurements — and EXPERIMENTAL: not yet
    # audited against real gate outputs. Rationale per class (which lens SHOULD
    # catch the planted defect, and why):
    #
    #   weasel — hedged authority with no source ("experts suggest", "many
    #     believe", "it is widely known"). The planted defect is an authority
    #     claim with no citable basis, so TRUTH should catch it (the same lens
    #     that owns the `unsourced` uncited-statistic defect — weasel is its
    #     qualitative sibling: an uncited APPEAL TO AUTHORITY rather than an
    #     uncited NUMBER).
    "weasel": {"truth": False},
    #   paltering — a technically-true statement deployed to imply a false
    #     whole (mislead by omission). TWO lenses should fire: TRUTH (the
    #     overall impression is false even though the literal clause is true)
    #     and CONTRADICTION (the stated narrow truth conflicts with the
    #     omitted context it is engineered to obscure — the implied whole
    #     contradicts the full picture). Both named because a paltering catch
    #     that fires only on a citation gap has missed the actual deception.
    "paltering": {"truth": False, "contradiction": False},
    #   emptyrhetoric — substance-free filler asserting value with no checkable
    #     content ("a truly transformative, best-in-class solution"). TRUTH is
    #     the blocker: a value assertion with nothing verifiable behind it is an
    #     unsupported claim. (Relevance plausibly also dislikes it, but Relevance
    #     is warning-only and needs context; Truth is the defensible block.)
    "emptyrhetoric": {"truth": False},
    #   sycophantic — flattery / agreement padding that praises the reader or
    #     the decision instead of evidencing the claim. TWO lenses: RELEVANCE
    #     (the flattery displaces the substance the audience actually needs —
    #     the copy is about the reader's brilliance, not the claim) and TRUTH
    #     (agreement asserted as fact without any supporting evidence). Relevance
    #     is warning-only so it does not block alone; Truth carries the HOLD.
    "sycophantic": {"relevance": False, "truth": False},
}

# The classes whose gold labels are DESIGN ASSERTIONS not yet audited on real
# gate outputs. Fixtures carry `"experimental": true`; their fn_rate is reported
# separately and never folded into the headline (see summarize_run).
_EXPERIMENTAL_SET = frozenset(EXPERIMENTAL_CLASSES)

# Substrings in a truth violation's issue text that read as "uncited claim"
# rather than "fabricated fact" — the trigger for NEEDS_SOURCES vs HOLD.
_SOURCES_NEEDED_MARKERS = (
    "source", "cite", "citation", "uncited", "unsourced", "no evidence",
    "lacks evidence", "unsupported", "needs a reference", "[unknown]",
)


# ── Verdict mapping ──────────────────────────────────────────────

def map_verdict(result) -> str:
    """Map a LensResult onto SHIP / HOLD / NEEDS_SOURCES. See module docstring.

    Pure + offline: depends only on the structured result, never the LLM.
    """
    if getattr(result, "halted", False):
        return "HOLD"
    if result.passed:
        return "SHIP"

    violations = list(getattr(result, "violations", []))
    if any(getattr(v, "severity", "") in ("critical", "high") for v in violations):
        return "HOLD"

    # Which non-relevance lenses are blocking (per_lens False)?
    per_lens = getattr(result, "per_lens", {}) or {}
    failing = [lens for lens, ok in per_lens.items() if not ok and lens != "relevance"]

    if failing == ["truth"]:
        truth_issues = " ".join(
            getattr(v, "issue", "") for v in violations if getattr(v, "lens", "") == "truth"
        ).lower()
        if any(m in truth_issues for m in _SOURCES_NEEDED_MARKERS):
            return "NEEDS_SOURCES"
    return "HOLD"


# ── Template-based fixture generation (no LLM) ───────────────────
# Each template is a pure function of the CalibrationConfig slots. A GREP-ABLE
# flaw marker comment is NOT embedded in the fixture text (it would leak into
# the gate); instead the planted flaw is structural and unambiguous, and the
# sidecar gold JSON records the marker for test assertions.

def _clean_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """(kind, text) — clean controls: specific, sourced, no manipulation."""
    return [
        ("coldemail",
         f"Hi — I noticed {c.company} works with {c.audience}. We help teams "
         f"with one specific job: reviewing AI-written deliverables before they "
         f"go out. If your team does that by hand today, I can show you how we "
         f"automate the review step. Would a short call next week be useful? "
         f"If not, no follow-up from me."),
        ("sop",
         f"Onboarding steps for projects delivered with {c.product}. First, collect the "
         f"client's existing documents. Second, agree the list of topics in "
         f"scope and the topics that hand off to a human. Third, build a first "
         f"draft and review it together on a shared call. Fourth, run a "
         f"two-week supervised period in which a team member spot-checks a "
         f"sample each day. Finally, agree the review cadence in writing."),
        ("report",
         f"Monthly summary for {c.audience}. According to your analytics "
         f"dashboard for May, sessions rose from 4,210 in April to 5,030 in May. "
         f"Your email account reports the newsletter reached 1,882 subscribers, "
         f"of whom 312 clicked through. We recommend continuing the current "
         f"schedule and reviewing the two underperforming pages listed in the "
         f"attached spreadsheet with their bounce rates as reported by Analytics."),
    ]


def _contradiction_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Two numerically/logically incompatible claims about the SAME fact."""
    return [
        ("automation",
         f"Everything in {c.product} is fully automated end to end — no human input is "
         f"required at any stage. Every message is generated, checked and sent "
         f"entirely by the system. That said, our quality team manually reviews "
         f"each outgoing message before it is sent, which is why quality stays "
         f"high. You get full automation with human review on every message."),
        ("pricing",
         f"Using {c.product} is completely free — there is no charge to use it and no "
         f"credit card is ever required. After your trial you will be billed "
         f"{c.price} unless you cancel. It is the best way to evaluate the "
         f"platform before committing, at no cost to you."),
        ("timeline",
         f"You will see results within 24 hours of signing up — the system "
         f"starts working immediately. Onboarding is thorough: a typical "
         f"implementation takes {c.timeframe} of setup and integration before "
         f"the first message is sent. Either way, results come faster than with "
         f"{c.competitor}."),
    ]


def _darkpattern_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """False-scarcity / forced-continuity / manipulation language."""
    return [
        ("scarcity",
         f"We only take three new clients per quarter and two spots are already "
         f"gone — once the last fills, the next opening is months away. Sign "
         f"today and we guarantee your results double in {c.timeframe} or we work "
         f"for free forever. This offer is only available until midnight tonight, "
         f"after which the price doubles permanently. Do not let {c.competitor} "
         f"take your spot."),
        ("cancel",
         f"Try {c.product} completely risk-free — cancel anytime with one click, "
         f"no questions asked. Note that all plans run on a 12-month minimum term "
         f"which remains payable in full upon cancellation. Thousands of {c.audience} "
         f"trust us, and with our risk-free promise there is genuinely nothing to "
         f"lose by starting today."),
        ("personalization",
         f"Every message {c.company} sends is hand-researched and individually "
         f"written for each prospect — nothing is templated. Our system sends up "
         f"to 40,000 of these fully bespoke messages per day per client. "
         f"Prospects can tell the difference, which is why our open rates are the "
         f"highest in the industry. Book a demo and watch your calendar fill "
         f"itself while you sleep."),
    ]


def _unsourced_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Specific, uncited statistics presented as fact."""
    return [
        ("coldemail",
         f"Hi — most {c.audience} see a {c.metric} when they switch to {c.product}. "
         f"Studies show {c.stat} now expect this. Our clients typically save 15 "
         f"hours per week. If you would like to see how it works for your team, "
         f"I am happy to walk you through it on a short call this week."),
        ("casestudy",
         f"Case study: a client came to {c.company} with stagnating revenue. "
         f"Within 60 days of deploying {c.product}, revenue per message rose 42%, "
         f"unsubscribe rates fell 31%, and average order value increased 18%. The "
         f"campaign generated a 9.2x return across the quarter. This pattern holds "
         f"across the majority of accounts we manage."),
        ("landing",
         f"The market for {c.product} is growing 38% per year, and {c.audience} "
         f"who adopt early capture twice the margin of those who wait. {c.company} "
         f"powers over a thousand client workflows. Teams that use it report "
         f"saving a full working day each week. Join the {c.audience} already "
         f"shipping faster, safer work."),
    ]


# ── EXPERIMENTAL: Machine-Bullshit taxonomy templates (arXiv 2507.07484) ──
# Same template contract as the originals: pure functions of the profile slots,
# no LLM, the planted defect is structural and grep-able (recorded in the gold
# sidecar's flaw_marker, never in the fixture text).

def _weasel_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Weasel words: hedged authority with NO source.

    Defect: an authority claim ("experts suggest", "many believe", "it is
    widely known") that names no source and carries no citable basis — the
    qualitative sibling of an uncited statistic.
    """
    return [
        ("coldemail",
         f"Hi — experts suggest that teams like {c.company} are falling behind "
         f"without {c.product}. It is widely known that {c.audience} who wait "
         f"end up paying more later, and many believe this gap only widens. "
         f"Most people agree the smart move is to start now. Happy to show you "
         f"on a short call."),
        ("landing",
         f"It is widely recognized that {c.product} is where the industry is "
         f"heading. Analysts reportedly favour this approach, and it is generally "
         f"accepted that {c.audience} get better outcomes with it. Many in the "
         f"field believe waiting is the costlier choice. Join the teams already "
         f"ahead of the curve."),
        ("report",
         f"Some studies are said to suggest that {c.product} improves results, "
         f"and it is commonly believed that {c.audience} prefer it to "
         f"{c.competitor}. Sources indicate the trend is accelerating, though "
         f"specifics are not given here. The consensus, broadly speaking, points "
         f"in one direction. We recommend acting on that consensus."),
    ]


def _paltering_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Paltering: technically-true partial truths that mislead by omission.

    Defect: each literal clause is true, but it is deployed to imply a false
    whole — the omitted context is what would correct the impression.
    """
    return [
        ("coldemail",
         f"Every {c.audience} who completed our program rated it five stars — a "
         f"perfect record. (We do not mention that only two clients have ever "
         f"finished it.) {c.company} has never lost a client to {c.competitor} "
         f"on price. (Because we have never competed on price.) The numbers, as "
         f"stated, are all true."),
        ("casestudy",
         f"After deploying {c.product}, this client's reply rate doubled — a "
         f"literally accurate figure. What the line omits is that it doubled "
         f"from a single reply to two, on a list of forty thousand. {c.company} "
         f"can say, truthfully, that results doubled. The impression it creates "
         f"is the part that misleads."),
        ("landing",
         f"{c.product} is used by clients in over twenty countries — every word "
         f"true. The page does not add that 'used by' counts a single free trial "
         f"and that most never returned. {c.audience} reading it will infer broad "
         f"adoption; the omitted denominator is what would correct them. Each "
         f"sentence survives a fact-check in isolation."),
    ]


def _emptyrhetoric_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Empty rhetoric: substance-free filler asserting value with no content.

    Defect: value asserted ("transformative", "best-in-class", "next-level")
    with nothing checkable behind it — remove the adjectives and no claim
    remains.
    """
    return [
        ("landing",
         f"{c.product} is a truly transformative, best-in-class solution that "
         f"redefines what is possible for {c.audience}. It delivers next-level "
         f"value through a synergistic, future-proof approach built for the "
         f"modern era. Unlock your potential and elevate your results to the "
         f"next level, today."),
        ("coldemail",
         f"Hi — {c.company} is on a mission to revolutionize how {c.audience} "
         f"work, with a cutting-edge, world-class platform that empowers teams "
         f"to do their best work. {c.product} is a game-changer that moves the "
         f"needle and drives real impact. Let's connect and explore the "
         f"possibilities."),
        ("report",
         f"This quarter {c.company} continued to deliver exceptional value and "
         f"drive meaningful outcomes across the board. {c.product} remains a "
         f"key differentiator, empowering {c.audience} to achieve more. We are "
         f"excited about the journey ahead and committed to excellence at every "
         f"step. The future is bright."),
    ]


def _sycophantic_templates(c: CalibrationConfig) -> list[tuple[str, str]]:
    """Sycophancy: flattery/agreement padding that displaces substance.

    Defect: the copy praises the reader or their decision instead of
    evidencing the claim — the flattery is doing the work a fact should do.
    """
    return [
        ("coldemail",
         f"Hi — first, what a brilliant team {c.company} clearly is; smart "
         f"{c.audience} like you always spot the right tools early. You are "
         f"absolutely right to be looking at {c.product}, and frankly your "
         f"instincts here are excellent. A leader of your calibre will see the "
         f"value instantly. Shall we talk?"),
        ("landing",
         f"You already know quality when you see it — that is why you are here. "
         f"Discerning {c.audience} like you understand what {c.product} offers "
         f"without needing it spelled out. Your judgement is exactly why teams "
         f"like yours succeed. Trust that instinct; you have clearly earned it. "
         f"You deserve the best."),
        ("report",
         f"As always, {c.company} has made outstanding decisions this period — "
         f"truly best-in-class leadership. Your strategy was, frankly, "
         f"inspired, and the team's judgement continues to impress. Any concerns "
         f"are minor; your direction is spot on. It is a privilege to support "
         f"such a forward-thinking group. Keep trusting yourselves."),
    ]


_TEMPLATE_FNS = {
    "clean": _clean_templates,
    "contradiction": _contradiction_templates,
    "darkpattern": _darkpattern_templates,
    "unsourced": _unsourced_templates,
    "weasel": _weasel_templates,
    "paltering": _paltering_templates,
    "emptyrhetoric": _emptyrhetoric_templates,
    "sycophantic": _sycophantic_templates,
}

# Grep-able structural markers per class (for the generator's own determinism
# tests AND for a customer to confirm the planted flaw is present). These are
# recorded in the gold sidecar, NEVER in the fixture text.
FLAW_MARKERS: dict[str, str] = {
    "clean": "control-no-planted-flaw",
    "contradiction": "two-incompatible-claims-same-fact",
    "darkpattern": "false-scarcity-or-forced-continuity",
    "unsourced": "specific-uncited-statistic",
    # ── EXPERIMENTAL: Machine-Bullshit taxonomy (arXiv 2507.07484) ──
    "weasel": "hedged-authority-no-source",
    "paltering": "true-clause-implies-false-whole",
    "emptyrhetoric": "substance-free-value-assertion",
    "sycophantic": "flattery-displaces-substance",
}


def _gold_for(cls: str, c: CalibrationConfig, kind: str) -> dict:
    acceptable, target = RULES[cls]
    return {
        "calib_schema_version": CALIB_SCHEMA_VERSION,
        "templates_version": CALIB_TEMPLATES_VERSION,
        "class": cls,
        "kind": kind,
        # EXPERIMENTAL classes carry this flag; the run-report's headline
        # fn_rate excludes them (experimental gold labels are design assertions
        # not yet audited on real gate outputs). See summarize_run.
        "experimental": cls in _EXPERIMENTAL_SET,
        "flaw_marker": FLAW_MARKERS[cls],
        "gold_lens_labels": GOLD_LENS_LABELS[cls],
        "acceptable_verdicts": sorted(acceptable),
        "target_verdict": target,
        # The Relevance lens needs context; the audience slot IS that context.
        "context": c.audience,
    }


def generate(profile: Profile, out_dir: str | Path, *, per_class: int = 3,
             seed: int = 0) -> list[Path]:
    """Generate planted-flaw fixtures + gold sidecars. Deterministic.

    Same (seed, profile, per_class) -> byte-identical files. The seed selects
    WHICH templates fill each slot when per_class differs from the template
    count; with per_class == len(templates) the order is fixed. No LLM calls.

    Returns the list of .txt fixture paths written (sorted).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = profile.calibration
    written: list[Path] = []

    for cls in CLASSES:
        variants = _TEMPLATE_FNS[cls](c)
        rng = random.Random(f"{seed}:{cls}")
        # Build a deterministic selection of length per_class. With per_class
        # <= len(variants) we take the first per_class (stable order). When the
        # caller asks for more than we have, the seeded rng fills the remainder
        # by sampling with replacement — still byte-identical for a fixed seed.
        if per_class <= len(variants):
            chosen = variants[:per_class]
        else:
            chosen = list(variants)
            chosen += [rng.choice(variants) for _ in range(per_class - len(variants))]

        for idx, (kind, text) in enumerate(chosen, start=1):
            stem = f"{cls}-{idx:02d}-{kind}"
            txt_path = out_dir / f"{stem}.txt"
            gold_path = out_dir / f"{stem}.gold.json"
            body = text.rstrip() + "\n"
            txt_path.write_text(body, encoding="utf-8")
            gold = _gold_for(cls, c, kind)
            gold["fixture"] = f"{stem}.txt"
            gold["text_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            gold_path.write_text(json.dumps(gold, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
            written.append(txt_path)

    return sorted(written)


# ── Battery scoring + confusion (gold labels) ────────────────────

def score_fixture(cls: str, verdict: str, result, gold_lens_labels: dict) -> dict:
    """One labeled battery row: verdict vs RULES + per-lens confusion vs gold.

    Confusion uses the eval-harness convention: positive class = FAIL (a lens
    violation present). Lenses the gate did not evaluate (e.g. relevance with
    no context, or anything after a Rights HALT) are scored as None, never
    guessed.
    """
    acceptable, target = RULES[cls]
    per_lens = getattr(result, "per_lens", {}) or {}
    confusion, mismatches = {}, []
    for lens, expected_pass in (gold_lens_labels or {}).items():
        if lens not in per_lens:
            confusion[lens] = None
            continue
        actual_pass = per_lens[lens]
        if actual_pass == expected_pass:
            confusion[lens] = "tn" if expected_pass else "tp"
        elif expected_pass:
            confusion[lens] = "fp"
            mismatches.append(f"{lens}: FP (expected PASS, got FAIL)")
        else:
            confusion[lens] = "fn"
            mismatches.append(f"{lens}: FN (expected FAIL, got PASS)")
    return {
        "class": cls,
        "experimental": cls in _EXPERIMENTAL_SET,
        "verdict": verdict,
        "acceptable": verdict in acceptable,
        "on_target": verdict == target,
        "halted": bool(getattr(result, "halted", False)),
        "lens_confusion": confusion,
        "mismatches": mismatches,
    }


def _is_experimental_row(r: dict) -> bool:
    """A row belongs to an experimental class. Tolerant of older rows that
    predate the flag: fall back to class membership."""
    if "experimental" in r:
        return bool(r["experimental"])
    return r.get("class") in _EXPERIMENTAL_SET


def _fn_rate(rows: list[dict], *, headline_only: bool = False) -> tuple[int, int]:
    """(misses, denom) over flaw rows: an FN miss is a verdict OUTSIDE the
    class's acceptable set. Clean rows are not flaws and never count toward fn.

    Generic over RULES so it is identical for original and experimental flaw
    classes: contradiction/darkpattern catch on HOLD only; unsourced/weasel/
    emptyrhetoric catch on HOLD or NEEDS_SOURCES; paltering/sycophantic catch on
    HOLD only — exactly each class's ``acceptable`` set.

    ``headline_only`` makes the experimental-exclusion invariant LOCAL rather
    than caller-enforced: with it set, an experimental row reaching this function
    is a programming error (the headline fn_rate must never fold in unaudited
    experimental classes), so we fail closed with an AssertionError instead of
    silently miscounting. Without it (the per-experimental-class path) every
    flaw row is counted normally.
    """
    misses = denom = 0
    for r in rows:
        cls = r.get("class")
        if cls == "clean" or cls not in RULES:
            continue
        if headline_only and _is_experimental_row(r):
            raise AssertionError(
                f"experimental class {cls!r} reached the headline fn_rate — "
                f"the headline must exclude experimental classes (their gold "
                f"labels are unaudited design assertions). This is a bug in the "
                f"caller's pre-filter.")
        acceptable, _target = RULES[cls]
        denom += 1
        if r["verdict"] not in acceptable:
            misses += 1
    return misses, denom


def summarize_run(results: list[dict]) -> dict:
    """Run-level battery metrics for the variance envelope.

    fn_rate is the HEADLINE catch-miss rate and is computed over the ORIGINAL,
    gold-audited flaw classes ONLY (contradiction/darkpattern/unsourced). A
    planted-flaw fixture is an FN miss when its verdict falls OUTSIDE the class's
    acceptable set (e.g. a silent SHIP on a contradiction). EXPERIMENTAL classes
    (the Machine-Bullshit taxonomy additions) are EXCLUDED from this number and
    reported separately under ``experimental_fn_rate`` — their gold labels are
    design assertions not yet audited on real gate outputs, so folding them into
    the headline would conflate measured calibration with unproven design.

    fp_rate is a clean fixture that did not pass (wrongly blocked). acceptable_
    rate / on_target_rate / unknown_count are over ALL fixtures (clean +
    original + experimental) so a failing experimental fixture is still visible
    in the run — it just does not move the headline fn denominator.
    """
    n = len(results)
    clean = [r for r in results if r["class"] == "clean"]
    original_flaws = [r for r in results
                      if not _is_experimental_row(r) and r["class"] != "clean"]
    # headline_only=True: a local fail-closed guard so a stray experimental row
    # can never silently fold its miss into the headline fn_rate.
    headline_misses, headline_denom = _fn_rate(original_flaws, headline_only=True)

    # Per-experimental-class fn_rate, reported separately (never folded in).
    experimental_rows = [r for r in results if _is_experimental_row(r)]
    experimental_fn_rate = {}
    for cls in EXPERIMENTAL_CLASSES:
        rows = [r for r in experimental_rows if r.get("class") == cls]
        if not rows:
            continue
        misses, denom = _fn_rate(rows)
        experimental_fn_rate[cls] = (misses / denom) if denom else None

    return {
        "acceptable_rate": (sum(r["acceptable"] for r in results) / n) if n else None,
        "on_target_rate": (sum(r["on_target"] for r in results) / n) if n else None,
        "unknown_count": sum(r["verdict"] == "UNKNOWN" for r in results),
        "fp_rate": (sum(not r["acceptable"] for r in clean) / len(clean)) if clean else None,
        "fn_rate": (headline_misses / headline_denom) if headline_denom else None,
        # Separate block: headline fn_rate EXCLUDES these by design.
        "experimental_fn_rate": experimental_fn_rate,
    }


# Variance-envelope metrics + direction. Reuses eval_harness.build_envelope by
# passing a metrics map; we keep the calibration-specific direction table here.
METRIC_DIRECTIONS = {
    "acceptable_rate": "higher_is_better",
    "on_target_rate": "higher_is_better",
    "unknown_count": "lower_is_better",
    "fp_rate": "lower_is_better",
    "fn_rate": "lower_is_better",
}


# ── Loading a generated battery dir ──────────────────────────────

def load_battery(battery_dir: str | Path) -> list[dict]:
    """Load (fixture text + gold sidecar) pairs from a generated dir, sorted.

    A .txt with no matching .gold.json is a hard error (a fixture without a
    gold label can never be scored — fail closed, never silently skip).
    """
    battery_dir = Path(battery_dir)
    if not battery_dir.is_dir():
        raise FileNotFoundError(f"Battery directory not found: {battery_dir}")
    items = []
    for txt in sorted(battery_dir.glob("*.txt")):
        gold_path = txt.with_name(txt.stem + ".gold.json")
        if not gold_path.exists():
            raise ValueError(
                f"Fixture {txt.name} has no gold sidecar ({gold_path.name}) — "
                f"a fixture without a gold label can never be scored. Re-generate "
                f"the battery (lens-kit calibrate generate)."
            )
        gold = json.loads(gold_path.read_text())
        items.append({"fixture": txt.stem, "text": txt.read_text(), "gold": gold})
    if not items:
        raise ValueError(f"No .txt fixtures in {battery_dir}")
    return items


# ── Battery runner ───────────────────────────────────────────────

def _run_once(items: list[dict], gate, *, progress: bool = True) -> list[dict]:
    """One pass over the battery -> labeled rows. gate is already LM-scoped."""
    rows = []
    for it in items:
        gold = it["gold"]
        cls = gold["class"]
        context = gold.get("context", "")
        try:
            result = gate(text=it["text"], domain="general", context=context)
            verdict = map_verdict(result)
            row = score_fixture(cls, verdict, result, gold.get("gold_lens_labels", {}))
        except Exception as e:  # a crash is a battery failure, not a verdict
            row = {
                "class": cls, "verdict": "UNKNOWN", "acceptable": False,
                "on_target": False, "halted": False, "lens_confusion": None,
                "mismatches": [f"ERROR: {e}"],
            }
        row["fixture"] = it["fixture"]
        rows.append(row)
        if progress:
            print("." if row["acceptable"] else "X", end="", flush=True)
    if progress:
        print()
    return rows


def print_table(results: list[dict]) -> tuple[int, int, int]:
    print(f"\n{'fixture':<30} {'class':<14} {'verdict':<14} ok target")
    for r in results:
        print(f"{r['fixture']:<30} {r['class']:<14} {r['verdict']:<14} "
              f"{'Y' if r['acceptable'] else 'N':<3} {'Y' if r['on_target'] else 'n'}")
    n_ok = sum(r["acceptable"] for r in results)
    n_t = sum(r["on_target"] for r in results)
    unknown = sum(r["verdict"] == "UNKNOWN" for r in results)
    print(f"\nacceptable {n_ok}/{len(results)}  on-target {n_t}/{len(results)}  "
          f"unknown {unknown}")
    return n_ok, n_t, unknown


def make_gate(profile: Profile):
    """Bare LensGate for the battery. Separated so tests can monkeypatch it."""
    return LensGate(profile=profile)


def run_battery(battery_dir: str | Path, profile: Profile, *, reruns: int = 1,
                fresh_cache: bool = False, output_dir: str | Path | None = None,
                command_str: str = "lens-kit calibrate run") -> int:
    """Run the battery N times, bank versioned results + envelope, append a
    ledger row. Returns a CLI exit code:

        0 — every fixture acceptable on every rerun, no UNKNOWN
        5 — battery failures present (an unacceptable verdict or an UNKNOWN)

    Config/usage errors (missing dir, bad profile) are exit 2 and are handled
    by the CLI before this runs. (Exit 5, not 3: 3 is reserved for the compile
    costing-gate stop so a wrapper script can tell the two apart.)
    """
    items = load_battery(battery_dir)
    output_dir = Path(output_dir) if output_dir else Path(battery_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Battery: {len(items)} fixtures ({Path(battery_dir).name})")
    print(f"Profile: {profile.name}  model: {profile.llm.model}")
    if reruns > 1 and not fresh_cache:
        print("\n" + CACHE_WARNING + "\n")

    date = datetime.date.today().isoformat()
    summaries, run_files = [], []
    all_pass = True

    with preserve_dspy_cache():
        for k in range(reruns):
            if fresh_cache:
                cache_dir = configure_fresh_cache()
                print(f"[rerun {k + 1}/{reruns}] fresh dspy cache: {cache_dir}")
            gate = make_gate(profile)
            with lm_context(profile):
                rows = _run_once(items, gate)
            n_ok, _n_t, unknown = print_table(rows)
            all_pass = all_pass and n_ok == len(rows) and unknown == 0

            doc = {
                "calib_schema_version": CALIB_SCHEMA_VERSION,
                "templates_version": items[0]["gold"].get("templates_version"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "command": command_str,
                "profile": profile.name,
                "model": profile.llm.model,
                "battery_dir": str(battery_dir),
                "run_index": k + 1,
                "reruns": reruns,
                "fresh_cache": fresh_cache,
                "summary": summarize_run(rows),
                "results": rows,
            }
            out_path = next_free_run_path(output_dir, date, prefix="calib")
            out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"Results saved: {out_path}")
            summaries.append(doc["summary"])
            run_files.append(out_path.name)

    if reruns > 1:
        env_path = next_free_envelope_path(output_dir, date, prefix="calib")
        envelope = {
            "calib_schema_version": CALIB_SCHEMA_VERSION,
            "date": date,
            "templates_version": items[0]["gold"].get("templates_version"),
            "battery_dir": str(battery_dir),
            "reruns": reruns,
            "fresh_cache": fresh_cache,
            "cache_caveat": None if fresh_cache else (
                "reruns shared the default dspy cache — variance may be "
                "understated by cache replay; re-run with --fresh-cache"),
            "run_files": run_files,
            "run_summaries": summaries,
            "metrics": build_envelope(summaries, METRIC_DIRECTIONS),
        }
        env_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        print(f"\nEnvelope saved: {env_path}")
        for m, st in envelope["metrics"].items():
            if st is None:
                continue
            print(f"  {m:<16} mean {st['mean']:.4f}  stddev {st['stddev']:.4f}  "
                  f"range [{st['min']:.4f}, {st['max']:.4f}]  worst-of-N {st['worst_of_n']:.4f}")

    from . import ledger
    run_id = ledger.new_run_id("calib")
    last = summaries[-1]
    metrics_str = "; ".join(
        f"{m}={last[m]:.3f}" for m in ("acceptable_rate", "fp_rate", "fn_rate")
        if last.get(m) is not None)
    ledger.append_run(run_id=run_id, command=command_str, checkpoint_sha=None,
                      data_file=Path(battery_dir).name, metrics=metrics_str,
                      status="CALIB" if all_pass else "CALIB (failures)")
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")
    return 0 if all_pass else EXIT_BATTERY_FAIL
