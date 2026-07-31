"""lens-kit CLI.

C2 surface:
    lens-kit validate <file> --profile <yaml> [--domain D] [--context C] [--json]
    lens-kit compile <training.json> --profile <yaml> --output <ckpt.json>
        [--max-evals N | --auto light|medium|heavy] [--threads N]
        [--val-split F] [--patience N] [--approve-cost]
        [--fresh-cache] [--warm-probe-cache]
    lens-kit eval <holdout.json> --profile <yaml>
        [--checkpoint ckpt.json | --baseline] [--reruns N] [--fresh-cache]
        [--output-dir DIR]
    lens-kit calibrate generate --profile <yaml> --out <dir> [--per-class N] [--seed N]
    lens-kit calibrate run <dir> --profile <yaml> [--reruns N] [--fresh-cache]
        [--output-dir DIR]
    lens-kit mutate <holdout.json> --profile <yaml> [--seed N] [--per-type N]
        [--output-dir DIR]
    lens-kit label-audit <dataset.json> <eval-results.json>
        [--backup] [--output PATH]
    lens-kit sidecar <ckpt.json> --profile <yaml>
        [--dataset DATA.json] [--eval-results EVAL.json]
        [--train-size N] [--val-size N] [--not-a-claim-of TEXT ...]
    lens-kit consistency markers <source> <rendered...> [--markers ...] [--profile P]
    lens-kit consistency leaks <files...> [--profile P] [--deny ...]
    lens-kit consistency numbers <summary> <body>
    lens-kit consistency all --config <yaml> [--profile P]
    lens-kit catches add [--catch C --pattern P --rule R --artifact-type T
        --domain D --self-catch | --from-json FILE | --seed]
    lens-kit catches relevant <artifact_type | --all> [--domain D]
        [--threshold N] [--format block|json]
    lens-kit catches stats [--threshold N] [--domain D] [--artifact-type T]
    lens-kit catches export --as-labels OUT [--catches PATH] [--domain D]
    lens-kit integrity [--runs-md RUNS.md] [--catches catches.jsonl]
        [--last N] [--strict] [--profile P]
    lens-kit improve --trainset T --holdout H --baseline-checkpoint C
        --profile P [--catches catches.jsonl] [--output DIR] [--domain D]
        [--reruns N] [--allow-single-measurement] [--approve-cost]
        [--max-evals N | --auto light|medium|heavy]
        [--threads N] [--val-split F] [--patience N]
    lens-kit improve --status [--runs-md RUNS.md]
    lens-kit review <results-or-dir> [--port N] [--static PATH]
        [--previous-feedback feedback.json] [--open]

Global: --no-echo-args (record only the subcommand in RUNS.md, not arguments;
    sensitive flag values like --deny are redacted regardless).

Exit codes: 0 = success/validation passed, 1 = validation failed/halted,
2 = usage error or config error,
3 = costing-gate hard stop (compile ONLY — uniquely "stop, this run costs too much"),
4 = pace kill (live pace blew the approved projection; kill record written),
5 = gate failed to catch planted flaws: calibration battery failures
    (calibrate run) OR a missed/rubber-stamped mutant (mutate / eval
    --mutation-control),
6 = consistency violation (dropped marker / forbidden-string leak / orphan
    summary number) — ALSO the integrity --strict family: a breached
    gate-integrity threshold (rubber-stamp shape / stale memory / stale
    mutation control), AND an improve run whose promotion mutation control was
    INCONCLUSIVE (a non-{0,5} result we can't read as clean-or-rubber-stamp; the
    challenger is kept, not promoted). Deterministic tripwire, no score overrides it.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import ConfigError, Profile, configure_from_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lens-kit",
        description="Trainable 10-lens validation gate for AI-generated text.",
    )
    parser.add_argument(
        "--no-echo-args", action="store_true",
        help="Do not echo command arguments into the RUNS.md ledger — record "
             "only the subcommand. (Sensitive flag values like --deny are "
             "redacted regardless; this opts out of echoing ALL args.)")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Run the 10-lens gate on a text file")
    v.add_argument("file", help="Path to the text file to validate")
    v.add_argument("--profile", required=True, help="Path to a profile YAML")
    v.add_argument("--domain", default="general", help="Domain context (default: auto-detect via profile)")
    v.add_argument("--context", default="", help="Audience/purpose context for the Relevance lens")
    v.add_argument("--context-file", help="Read relevance context from file")
    v.add_argument("--json", action="store_true", help="Output verdict as JSON")
    v.add_argument("--no-fix", action="store_true", help="Disable auto-fix")
    v.add_argument("--fast-mode", action="store_true",
                   help="N=1 for Truth/Extrapolation instead of BestOfN N=3 (faster, less strict)")
    v.add_argument("--timing", action="store_true", help="Record per-lens timing in the verdict")

    c = sub.add_parser("compile", help="Compile the gate with GEPA (costing gate + pace monitor)")
    c.add_argument("training", help="Training data JSON (list of {text, per_lens, ...})")
    c.add_argument("--profile", required=True, help="Path to a profile YAML")
    c.add_argument("--output", required=True, help="Output checkpoint path (.json)")
    budget = c.add_mutually_exclusive_group()
    budget.add_argument("--max-evals", type=int, default=None,
                        help="Max full evaluations (rollouts = N x (train+val))")
    budget.add_argument("--auto", choices=["light", "medium", "heavy"], default=None,
                        help="GEPA auto budget (default when neither flag given: light)")
    c.add_argument("--threads", type=int, default=1, help="Parallel metric threads (default: 1)")
    c.add_argument("--val-split", type=float, default=0.2, help="Validation split (default: 0.2)")
    c.add_argument("--patience", type=int, default=10,
                   help="Early stop after N iterations without improvement (default: 10)")
    c.add_argument("--approve-cost", action="store_true",
                   help="Accept a projection above the approval threshold "
                        "(default threshold: 2h; see profile compile.approval_threshold_hours)")
    c.add_argument("--fresh-cache", action="store_true",
                   help="Run the GEPA compile against a fresh dspy cache dir "
                        "(no rollout served from a stale cache; mirrors eval --fresh-cache)")
    c.add_argument("--warm-probe-cache", action="store_true",
                   help="Measure the pace probe against the LIVE dspy cache instead "
                        "of an isolated fresh one. Default is fresh: a warm cache "
                        "replays sub-second pace and arms a false pace kill.")

    e = sub.add_parser("eval", help="Run the holdout eval harness")
    e.add_argument("holdout", help="Holdout JSON (list or {'examples': [...]})")
    e.add_argument("--profile", required=True, help="Path to a profile YAML")
    mode = e.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", default=None, help="Checkpoint to load (remaps legacy keys)")
    mode.add_argument("--baseline", action="store_true",
                      help="Uncompiled baseline gate (also the default when no --checkpoint)")
    e.add_argument("--reruns", type=int, default=1,
                   help="Run N times and bank a variance envelope (default: 1)")
    e.add_argument("--fresh-cache", action="store_true",
                   help="Fresh dspy cache dir per rerun — required for REAL variance "
                        "(without it, cache replay can make reruns identical)")
    e.add_argument("--output-dir", default=".", help="Where versioned results land (default: cwd)")
    e.add_argument("--mutation-control", action="store_true",
                   help="After the eval, seed deterministic mutants from this holdout "
                        "and confirm the gate flags every one (rubber-stamp check). "
                        "A missed mutant makes the whole command exit 5.")
    e.add_argument("--mutation-seed", type=int, default=0,
                   help="Seed for --mutation-control mutant generation (default: 0)")

    cal = sub.add_parser("calibrate", help="Planted-flaw calibration battery (generate / run)")
    cal_sub = cal.add_subparsers(dest="calibrate_command", required=True)

    cg = cal_sub.add_parser("generate", help="Generate planted-flaw fixtures (no LLM, deterministic)")
    cg.add_argument("--profile", required=True, help="Path to a profile YAML (calibration: vocabulary slots)")
    cg.add_argument("--out", required=True, help="Output directory for fixtures + gold sidecars")
    cg.add_argument("--per-class", type=int, default=3,
                    help="Fixtures per class (8 classes: 4 original + 4 "
                         "experimental Machine-Bullshit; default: 3)")
    cg.add_argument("--seed", type=int, default=0,
                    help="Deterministic seed (same seed+profile = identical bytes; default: 0)")

    cr = cal_sub.add_parser("run", help="Run a generated battery through the gate")
    cr.add_argument("battery_dir", help="Directory of generated fixtures + gold sidecars")
    cr.add_argument("--profile", required=True, help="Path to a profile YAML")
    cr.add_argument("--reruns", type=int, default=1,
                    help="Run N times and bank a variance envelope (default: 1)")
    cr.add_argument("--fresh-cache", action="store_true",
                    help="Fresh dspy cache dir per rerun — required for REAL variance")
    cr.add_argument("--output-dir", default=None,
                    help="Where versioned results land (default: the battery dir)")

    m = sub.add_parser("mutate",
                       help="Mutation control: seed mutants from a holdout, the gate MUST flag each")
    m.add_argument("holdout", help="Holdout JSON to seed mutants from (real examples)")
    m.add_argument("--profile", required=True, help="Path to a profile YAML")
    m.add_argument("--seed", type=int, default=0,
                   help="Deterministic seed (same holdout+seed = identical mutants; default: 0)")
    m.add_argument("--per-type", type=int, default=1,
                   help="Mutants per mutation type (3 types; default: 1)")
    m.add_argument("--output-dir", default=None,
                   help="Where the mutation results JSON lands (default: none written)")

    la = sub.add_parser("label-audit",
                        help="Audit gold labels the gate disagrees with (before blaming the model)")
    la.add_argument("dataset", help="Dataset JSON with gold per_lens labels")
    la.add_argument("eval_results", help="A single-run eval results JSON (has per_example)")
    la.add_argument("--backup", action="store_true",
                    help="Create a dated dataset backup (the only sanctioned way to "
                         "make labels editable; relabeling without it is refused)")
    la.add_argument("--output", default=None,
                    help="Audit report path (default: label-audit-<dataset>-<date>.md)")

    sc = sub.add_parser("sidecar", help="Generate a full provenance sidecar next to a checkpoint")
    sc.add_argument("checkpoint", help="Checkpoint JSON to describe")
    sc.add_argument("--profile", required=True, help="Path to a profile YAML")
    sc.add_argument("--dataset", default=None, help="Dataset the checkpoint was tuned/scored on")
    sc.add_argument("--eval-results", default=None,
                    help="Eval results or envelope JSON (records metrics + caveats)")
    sc.add_argument("--train-size", type=int, default=None, help="Training example count")
    sc.add_argument("--val-size", type=int, default=None, help="Validation example count")
    sc.add_argument("--not-a-claim-of", action="append", default=None, metavar="TEXT",
                    help="Extra honest-boundary entry (repeatable; defaults always included)")

    cons = sub.add_parser(
        "consistency",
        help="Deterministic cross-artifact tripwires (no LLM): marker parity, "
             "forbidden-string leaks, summary/body number parity")
    cons_sub = cons.add_subparsers(dest="consistency_command", required=True)

    cm = cons_sub.add_parser(
        "markers",
        help="Marker parity: evidence markers counted in a source must survive "
             "into every rendered output (TRIPWIRE — a subset render under-counts "
             "legitimately; review, don't trust blindly)")
    cm.add_argument("source", help="The source artifact (authoritative marker counts)")
    cm.add_argument("rendered", nargs="+", help="Rendered output file(s) to check")
    cm.add_argument("--markers", nargs="+", default=None,
                    help="Override the marker set (default: profile consistency.markers "
                         "or [UNVERIFIED] [ESTIMATE] [PROJECTION])")
    cm.add_argument("--profile", default=None,
                    help="Profile YAML (for consistency.markers; optional)")

    cl = cons_sub.add_parser(
        "leaks",
        help="Forbidden-string leak scan: any deny-list hit is a violation no "
             "score overrides (case-insensitive literal match)")
    cl.add_argument("files", nargs="+", help="Customer-facing file(s) to scan")
    cl.add_argument("--profile", default=None,
                    help="Profile YAML providing consistency.deny")
    cl.add_argument("--deny", nargs="+", default=None,
                    help="Extra deny terms (added to the profile's consistency.deny)")

    cn = cons_sub.add_parser(
        "numbers",
        help="Number parity: numeric literals in a summary must appear (normalized) "
             "in the body (TRIPWIRE — literal match only, no semantic/derived math)")
    cn.add_argument("summary", help="The summary file (its numbers must be in the body)")
    cn.add_argument("body", help="The body file the summary summarizes")

    ca = cons_sub.add_parser(
        "all",
        help="Run markers + leaks + numbers from one config YAML")
    ca.add_argument("--config", required=True,
                    help="YAML: {markers: {source, rendered}, leaks: {files}, "
                         "numbers: {summary, body}} — only the listed checks run")
    ca.add_argument("--profile", default=None,
                    help="Profile YAML for consistency.markers + consistency.deny")

    cat = sub.add_parser(
        "catches",
        help="Institutional-memory loop: record validator catches, surface "
             "prior patterns before a run, promote recurring patterns to "
             "deterministic checks (file-based, no LLM, no network)")
    cat_sub = cat.add_subparsers(dest="catches_command", required=True)

    cadd = cat_sub.add_parser(
        "add",
        help="Append a catch (a NAMED DEFECT) to catches.jsonl. Routine passes "
             "are rejected by doctrine; a catch without a pattern/rule warns.")
    cadd.add_argument("--catch", default=None,
                      help="What was WRONG (required unless --from-json/--seed)")
    cadd.add_argument("--pattern", default="",
                      help="The general trap this is an instance of (recommended)")
    cadd.add_argument("--rule", default="",
                      help="The forward rule that prevents this next time (recommended)")
    cadd.add_argument("--domain", default="general", help="Domain tag (default: general)")
    cadd.add_argument("--artifact-type", default="",
                      help="Artifact kind, e.g. landing-copy / research-brief")
    cadd.add_argument("--date", default="", help="ISO date (default: today)")
    cadd.add_argument("--self-catch", action="store_true",
                      help="Mark this a validator-discipline failure (a self-catch)")
    cadd.add_argument("--id", default="", help="Explicit record id (default: derived)")
    cadd.add_argument("--snippet-file", default=None,
                      help="Read the offending text (verbatim, exact bytes/newlines) "
                           "from a file — the snippet the validator caught. A file is "
                           "used (not an inline flag) to preserve exact bytes without "
                           "shell-mangling. Required to make this catch exportable as "
                           "a training label (with --lenses-failed).")
    cadd.add_argument("--lenses-failed", default=None,
                      help="Comma-separated canonical lens keys this snippet failed "
                           "(e.g. truth,contradiction). Validated against the 10-lens "
                           "canon; an unknown key is a usage error.")
    cadd.add_argument("--from-json", default=None,
                      help="Read the catch record(s) from a JSON file (object or "
                           "list of objects) — for agent use")
    cadd.add_argument("--seed", action="store_true",
                      help="Copy the shipped EXAMPLE catches into catches.jsonl as "
                           "starter content (documented examples, fully genericized)")

    crel = cat_sub.add_parser(
        "relevant",
        help="Print prior catches for an artifact type, most-recurrent patterns "
             "first — designed to paste into a validator agent's context")
    crel.add_argument("artifact_type", nargs="?", default=None,
                      help="Artifact type to filter by (omit and pass --all for everything)")
    crel.add_argument("--all", action="store_true",
                      help="Surface every catch regardless of artifact type")
    crel.add_argument("--domain", default=None, help="Also filter by domain")
    crel.add_argument("--threshold", type=int, default=None,
                      help="Promote threshold for the [PROMOTE] lines (default: 3)")
    crel.add_argument("--format", choices=["block", "json"], default="block",
                      help="block = the paste surface (default); json = the raw records")

    cstats = cat_sub.add_parser(
        "stats",
        help="Per-pattern recurrence counts; any pattern at threshold gets a "
             "PROMOTE-to-deterministic-check suggestion")
    cstats.add_argument("--threshold", type=int, default=None,
                        help="Recurrence threshold for a PROMOTE line (default: 3)")
    cstats.add_argument("--domain", default=None, help="Filter by domain")
    cstats.add_argument("--artifact-type", default=None, help="Filter by artifact type")

    cexp = cat_sub.add_parser(
        "export",
        help="Export catches (those with snippet + lenses_failed) as GEPA "
             "training labels — the hindsight loop: caught defects become "
             "train-side examples (named lenses FAIL, the rest PASS). Catches "
             "lacking snippet/lenses_failed are skipped WITH a printed count.")
    cexp.add_argument("--as-labels", required=True, metavar="OUT",
                      help="Output JSON path for the training-label list")
    cexp.add_argument("--catches", default=None,
                      help="Path to catches.jsonl (default: ./catches.jsonl)")
    cexp.add_argument("--domain", default=None, help="Filter catches by domain")

    scr = sub.add_parser(
        "scrub",
        help="Deterministic PII scan/scrub (regex, no LLM): scrubbed text to "
             "stdout, findings (type + offsets, never the value) to stderr; "
             "exit 6 on any finding (tripwire)")
    scr.add_argument("file", help="Text file to scan")
    scr.add_argument("--json", action="store_true",
                     help="Emit {findings, scrubbed_text} JSON to stdout instead "
                          "(findings carry type/severity/offsets only — raw "
                          "values are never echoed)")

    intg = sub.add_parser(
        "integrity",
        help="Gate-integrity trend monitor: pass-rate trend, runs/days since the "
             "last catch, mutation-control recency. Report-only by default; "
             "--strict exits 6 on a breached threshold (tripwire, not oracle — "
             "a flag means 'look here', absence of a flag is NOT health)")
    intg.add_argument("--runs-md", default="RUNS.md",
                      help="Path to the RUNS.md ledger (default: ./RUNS.md)")
    intg.add_argument("--catches", default="catches.jsonl",
                      help="Path to catches.jsonl (default: ./catches.jsonl)")
    intg.add_argument("--last", type=int, default=10,
                      help="Window: consider the last N eval runs (default: 10)")
    intg.add_argument("--strict", action="store_true",
                      help="Exit 6 when a threshold is breached (default: "
                           "report-only, exit 0)")
    intg.add_argument("--profile", default=None,
                      help="Profile YAML providing integrity thresholds "
                           "(optional; conservative defaults otherwise)")

    imp = sub.add_parser(
        "improve",
        help="Autoresearch challenger/promote loop: build a challenger trainset "
             "(base + catches export labels), recompile THROUGH the costing gate, "
             "eval baseline + challenger on the frozen holdout (fresh cache on "
             "both), and promote the challenger ONLY if it beats baseline beyond "
             "the variance envelope AND passes mutation control. Both outcomes "
             "(PROMOTED / KEPT-BASELINE) get a ledger row + provenance sidecar.")
    imp.add_argument("--trainset", help="Base training data JSON (the incumbent's)")
    imp.add_argument("--holdout", help="Frozen holdout JSON (both gates eval'd on it)")
    imp.add_argument("--baseline-checkpoint",
                     help="The incumbent checkpoint the challenger must beat")
    imp.add_argument("--catches", default=None,
                     help="catches.jsonl to export labels from (default: ./catches.jsonl)")
    imp.add_argument("--output", default=".",
                     help="Output dir for the challenger/promoted checkpoint + "
                          "sidecars (default: cwd)")
    imp.add_argument("--profile", help="Path to a profile YAML")
    imp.add_argument("--domain", default=None,
                     help="Only export catches in this domain into the challenger")
    imp.add_argument("--reruns", type=int, default=1,
                     help="Eval reruns for the variance envelope (default: 1 = a "
                          "single measurement). With reruns==1 a nominal win is "
                          "NOT promoted unless --allow-single-measurement is set "
                          "(a single draw is not a bound). Use >1 to bound it.")
    imp.add_argument("--allow-single-measurement", action="store_true",
                     help="Permit promotion on a single eval measurement (no "
                          "variance envelope). Without it, reruns==1 keeps the "
                          "baseline and asks for --reruns N. The single-measurement "
                          "caveat is recorded in the ledger + sidecar either way.")
    imp.add_argument("--approve-cost", action="store_true",
                     help="Pass through to the challenger compile's costing gate "
                          "(accept a projection above the approval threshold)")
    imp.add_argument("--threads", type=int, default=1,
                     help="Parallel metric threads for the challenger compile")
    budget_i = imp.add_mutually_exclusive_group()
    budget_i.add_argument("--max-evals", type=int, default=None,
                          help="Challenger compile budget: max full evaluations")
    budget_i.add_argument("--auto", choices=["light", "medium", "heavy"],
                          default=None, help="Challenger compile GEPA auto budget")
    imp.add_argument("--val-split", type=float, default=0.2,
                     help="Challenger compile validation split (default: 0.2)")
    imp.add_argument("--patience", type=int, default=10,
                     help="Challenger compile early-stop patience (default: 10)")
    imp.add_argument("--status", action="store_true",
                     help="Report the last improve outcome(s) from the ledger and "
                          "exit (incumbent, last challenger result, promoted-or-not)")
    imp.add_argument("--runs-md", default="RUNS.md",
                     help="Ledger path for --status (default: ./RUNS.md)")

    rv = sub.add_parser("review",
                        help="Human review surface: render eval / calibration / "
                             "mutation artifacts as self-contained HTML + feedback.json")
    rv.add_argument("results", help="A results JSON file or a directory of them")
    rv.add_argument("--port", type=int, default=0,
                    help="Server port (default: 0 = OS-chosen ephemeral)")
    rv.add_argument("--previous-feedback", default=None,
                    help="A prior feedback.json shown as diff context")
    rv.add_argument("--static", default=None,
                    help="Write standalone HTML to this path instead of serving")
    rv.add_argument("--open", action="store_true",
                    help="Open the page in a browser (server mode only)")

    return parser


def _print_human(result) -> None:
    print("LENS GATE REPORT")
    print("-" * 40)
    for lens, passed in result.per_lens.items():
        print(f"  {lens:16s} {'PASS' if passed else 'FAIL'}")
    if result.consciousness_flags:
        print(f"\n  consciousness flags: {len(result.consciousness_flags)}")
        for flag in result.consciousness_flags:
            ftype = flag.get("type", "unknown") if isinstance(flag, dict) else str(flag)
            snippet = flag.get("text_snippet", "")[:60] if isinstance(flag, dict) else ""
            print(f"    [{ftype}] {snippet}")
    print("-" * 40)
    print(f"Overall: {'PASS' if result.passed else 'FAIL'}")
    if result.halted:
        print(f"HALTED: {result.halt_reason}")
    if result.violations:
        print(f"\nViolations ({len(result.violations)}):")
        for v in result.violations:
            print(f"  [{v.lens}/{v.severity}] {v.issue[:100]}")


def _cmd_validate(args) -> int:
    text_path = Path(args.file)
    if not text_path.exists():
        print(f"error: file not found: {text_path}", file=sys.stderr)
        return 2
    try:
        profile = Profile.load(args.profile)
        configure_from_profile(profile)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Import here so `lens-kit compile` etc. never need a working LM setup.
    from .gate import LensGate

    context = args.context
    if args.context_file:
        ctx_path = Path(args.context_file)
        if not ctx_path.exists():
            print(f"error: context file not found: {ctx_path}", file=sys.stderr)
            return 2
        context = ctx_path.read_text().strip()

    gate = LensGate(profile=profile, auto_fix=not args.no_fix, fast_mode=args.fast_mode)
    result = gate(text=text_path.read_text(), domain=args.domain,
                  context=context, include_timings=args.timing)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)
    return 0 if result.passed else 1


def _load_profile(path: str):
    try:
        return Profile.load(path), None
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return None, 2


def _command_str(argv, *, no_echo_args=False, redact_sensitive=True) -> str:
    """The command string recorded in the ledger.

    no_echo_args: collapse to just the subcommand (``lens-kit consistency``) —
        the global ``--no-echo-args`` opt-out for anyone who wants NO arguments
        echoed into RUNS.md at all.
    redact_sensitive (default): mask sensitive flag VALUES (e.g. ``--deny``)
        with ``<redacted>`` so deny-list terms never land in the ledger in the
        clear (S1 review WARN), while keeping the command legible.
    """
    parts = [str(p) for p in (argv if argv is not None else sys.argv[1:])]
    # The global --no-echo-args flag is itself part of raw argv; never record it.
    parts = [p for p in parts if p != "--no-echo-args"]
    if no_echo_args:
        # Keep the leading subcommand token(s) only (e.g. "consistency leaks"),
        # drop everything that looks like a flag or an argument value.
        kept = []
        for p in parts:
            if p.startswith("-"):
                break
            kept.append(p)
        return "lens-kit " + " ".join(kept) + (" [args redacted]" if len(parts) > len(kept) else "")
    if redact_sensitive:
        from .ledger import redact_command
        return "lens-kit " + redact_command(parts)
    return "lens-kit " + shlex.join(str(p) for p in parts)


def _cmd_compile(args, argv) -> int:
    training = Path(args.training)
    if not training.exists():
        print(f"error: training data not found: {training}", file=sys.stderr)
        return 2
    profile, err = _load_profile(args.profile)
    if err:
        return err
    # Import here: validate/eval must never need the optimizer stack.
    from .compile_harness import run_compile

    try:
        return run_compile(
            training, profile, args.output,
            auto=args.auto, max_evals=args.max_evals, threads=args.threads,
            val_split=args.val_split, patience=args.patience,
            approve_cost=args.approve_cost, fresh_cache=args.fresh_cache,
            warm_probe_cache=args.warm_probe_cache, command_str=_command_str(argv),
        )
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_eval(args, argv) -> int:
    holdout = Path(args.holdout)
    if not holdout.exists():
        print(f"error: holdout file not found: {holdout}", file=sys.stderr)
        return 2
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if args.reruns < 1:
        print("error: --reruns must be >= 1", file=sys.stderr)
        return 2
    profile, err = _load_profile(args.profile)
    if err:
        return err
    from .eval_harness import run_eval_command

    try:
        rc = run_eval_command(
            holdout, profile, checkpoint=args.checkpoint,
            reruns=args.reruns, fresh_cache=args.fresh_cache,
            output_dir=args.output_dir, command_str=_command_str(argv),
        )
        if rc == 0 and args.mutation_control:
            from .mutation import run_mutation_control
            print("\n" + "=" * 40 + "\nMutation control (post-eval)\n" + "=" * 40)
            # A missed mutant overrides a green eval: a gate that can't fail a
            # planted flaw shouldn't be trusted just because the holdout passed.
            return run_mutation_control(
                holdout, profile, seed=args.mutation_seed,
                output_dir=args.output_dir, command_str=_command_str(argv))
        return rc
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_calibrate(args, argv) -> int:
    profile, err = _load_profile(args.profile)
    if err:
        return err

    if args.calibrate_command == "generate":
        if args.per_class < 1:
            print("error: --per-class must be >= 1", file=sys.stderr)
            return 2
        from .calibration import generate

        written = generate(profile, args.out, per_class=args.per_class, seed=args.seed)
        print(f"Generated {len(written)} fixtures (+ gold sidecars) in {args.out}")
        for p in written:
            print(f"  {p.name}")
        return 0

    if args.calibrate_command == "run":
        battery_dir = Path(args.battery_dir)
        if not battery_dir.is_dir():
            print(f"error: battery directory not found: {battery_dir}", file=sys.stderr)
            return 2
        if args.reruns < 1:
            print("error: --reruns must be >= 1", file=sys.stderr)
            return 2
        from .calibration import run_battery

        try:
            return run_battery(
                battery_dir, profile, reruns=args.reruns,
                fresh_cache=args.fresh_cache, output_dir=args.output_dir,
                command_str=_command_str(argv),
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    print(f"error: unknown calibrate command: {args.calibrate_command}", file=sys.stderr)
    return 2


def _cmd_mutate(args, argv) -> int:
    holdout = Path(args.holdout)
    if not holdout.exists():
        print(f"error: holdout file not found: {holdout}", file=sys.stderr)
        return 2
    if args.per_type < 1:
        print("error: --per-type must be >= 1", file=sys.stderr)
        return 2
    profile, err = _load_profile(args.profile)
    if err:
        return err
    from .mutation import run_mutation_control

    try:
        return run_mutation_control(
            holdout, profile, seed=args.seed, per_type=args.per_type,
            output_dir=args.output_dir, command_str=_command_str(argv))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_label_audit(args, argv) -> int:
    dataset = Path(args.dataset)
    eval_results = Path(args.eval_results)
    if not dataset.exists():
        print(f"error: dataset not found: {dataset}", file=sys.stderr)
        return 2
    if not eval_results.exists():
        print(f"error: eval-results not found: {eval_results}", file=sys.stderr)
        return 2
    from .label_audit import run_label_audit

    try:
        return run_label_audit(
            dataset, eval_results, backup=args.backup,
            output_path=args.output, command_str=_command_str(argv))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_sidecar(args, argv) -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    if args.dataset and not Path(args.dataset).exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    if args.eval_results and not Path(args.eval_results).exists():
        print(f"error: eval-results not found: {args.eval_results}", file=sys.stderr)
        return 2
    profile, err = _load_profile(args.profile)
    if err:
        return err
    from .sidecar import generate_sidecar

    try:
        sidecar, path = generate_sidecar(
            checkpoint, profile=profile,
            train_size=args.train_size, val_size=args.val_size,
            dataset_path=args.dataset, eval_results_path=args.eval_results,
            extra_not_a_claim_of=args.not_a_claim_of)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"Provenance sidecar written: {path}")
    print(f"  sha256: {sidecar['sha256']}")
    print(f"  not_a_claim_of: {len(sidecar['not_a_claim_of'])} entries")
    return 0


def _missing_files(paths) -> str | None:
    for p in paths:
        if not Path(p).exists():
            return str(p)
    return None


def _consistency_profile(path):
    """Load consistency config from a profile, or None. Returns (cfg, err_rc)."""
    if not path:
        return None, None
    profile, err = _load_profile(path)
    if err:
        return None, err
    return profile.consistency, None


def _cmd_consistency(args, argv) -> int:
    from . import consistency as cons

    cmd = args.consistency_command
    # --deny values are redacted in the ledger by default; --no-echo-args drops
    # all args (S1 review WARN: deny terms must not land in RUNS.md in the clear).
    cstr = _command_str(argv, no_echo_args=args.no_echo_args)

    if cmd == "markers":
        missing = _missing_files([args.source, *args.rendered])
        if missing:
            print(f"error: file not found: {missing}", file=sys.stderr)
            return 2
        markers = args.markers
        if markers is None and args.profile:
            cfg, err = _consistency_profile(args.profile)
            if err:
                return err
            markers = cfg.markers
        if markers is None:
            markers = list(cons.DEFAULT_MARKERS)
        result = cons.check_markers(args.source, args.rendered, markers=markers)
        return cons.run_consistency([result], command_str=cstr)

    if cmd == "leaks":
        missing = _missing_files(args.files)
        if missing:
            print(f"error: file not found: {missing}", file=sys.stderr)
            return 2
        deny = []
        if args.profile:
            cfg, err = _consistency_profile(args.profile)
            if err:
                return err
            deny.extend(cfg.deny)
        if args.deny:
            deny.extend(args.deny)
        result = cons.check_leaks(args.files, deny)
        return cons.run_consistency([result], command_str=cstr)

    if cmd == "numbers":
        missing = _missing_files([args.summary, args.body])
        if missing:
            print(f"error: file not found: {missing}", file=sys.stderr)
            return 2
        result = cons.check_numbers(args.summary, args.body)
        return cons.run_consistency([result], command_str=cstr)

    if cmd == "all":
        if not Path(args.config).exists():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            return 2
        import yaml
        config = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(config, dict):
            print("error: --config must be a YAML mapping", file=sys.stderr)
            return 2

        cfg = None
        if args.profile:
            cfg, err = _consistency_profile(args.profile)
            if err:
                return err

        results = []
        if "markers" in config:
            mc = config["markers"]
            source, rendered = mc.get("source"), mc.get("rendered", [])
            if not source or not rendered:
                print("error: consistency.all markers needs {source, rendered}",
                      file=sys.stderr)
                return 2
            missing = _missing_files([source, *rendered])
            if missing:
                print(f"error: file not found: {missing}", file=sys.stderr)
                return 2
            markers = mc.get("markers") or (cfg.markers if cfg
                                            else list(cons.DEFAULT_MARKERS))
            results.append(cons.check_markers(source, rendered, markers=markers))
        if "leaks" in config:
            lc = config["leaks"]
            files = lc.get("files", [])
            if not files:
                print("error: consistency.all leaks needs {files}", file=sys.stderr)
                return 2
            missing = _missing_files(files)
            if missing:
                print(f"error: file not found: {missing}", file=sys.stderr)
                return 2
            deny = list(cfg.deny) if cfg else []
            deny.extend(lc.get("deny", []))
            results.append(cons.check_leaks(files, deny))
        if "numbers" in config:
            nc = config["numbers"]
            summary, body = nc.get("summary"), nc.get("body")
            if not summary or not body:
                print("error: consistency.all numbers needs {summary, body}",
                      file=sys.stderr)
                return 2
            missing = _missing_files([summary, body])
            if missing:
                print(f"error: file not found: {missing}", file=sys.stderr)
                return 2
            results.append(cons.check_numbers(summary, body))

        if not results:
            print("error: --config selected no checks (use keys markers/leaks/numbers)",
                  file=sys.stderr)
            return 2
        return cons.run_consistency(results, command_str=cstr)

    print(f"error: unknown consistency command: {cmd}", file=sys.stderr)
    return 2


def _cmd_catches(args, argv) -> int:
    from . import catches as cat

    cmd = args.catches_command
    path = cat.catches_path()

    if cmd == "add":
        return _catches_add(args, cat, path)

    if cmd == "export":
        return _catches_export(args, cat, argv)

    if cmd == "relevant":
        if not args.all and not args.artifact_type:
            print("error: pass an <artifact_type> or --all", file=sys.stderr)
            return 2
        try:
            records = cat.load_catches(path)
        except cat.CatchError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        artifact_type = None if args.all else args.artifact_type
        filtered = cat.filter_records(records, artifact_type=artifact_type,
                                      domain=args.domain)
        threshold = (args.threshold if args.threshold is not None
                     else cat.DEFAULT_PROMOTE_THRESHOLD)
        if args.format == "json":
            print(json.dumps(filtered, indent=2, ensure_ascii=False))
        else:
            print(cat.render_relevant_block(
                filtered, artifact_type=artifact_type, domain=args.domain,
                threshold=threshold), end="")
        return 0

    if cmd == "stats":
        try:
            records = cat.load_catches(path)
        except cat.CatchError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        records = cat.filter_records(
            records,
            artifact_type=args.artifact_type if args.artifact_type else None,
            domain=args.domain)
        threshold = (args.threshold if args.threshold is not None
                     else cat.DEFAULT_PROMOTE_THRESHOLD)
        print(cat.render_stats(records, threshold=threshold), end="")
        return 0

    print(f"error: unknown catches command: {cmd}", file=sys.stderr)
    return 2


def _catches_add(args, cat, path) -> int:
    """add: --seed copy | --from-json record(s) | flag-driven single record."""
    if args.seed:
        seed = cat.load_seed()
        for rec in seed:
            built, _ = cat.build_catch(
                catch=rec.get("catch", ""), pattern=rec.get("pattern", ""),
                rule=rec.get("rule", ""), domain=rec.get("domain", "general"),
                artifact_type=rec.get("artifact_type", ""),
                date=rec.get("date", ""), self_catch=rec.get("self_catch", False),
                catch_id=rec.get("id", ""))
            cat.append_catch(built, path)
        print(f"Seeded {len(seed)} example catch(es) into {path.name} "
              f"(documented examples — replace with your own as you record real catches)")
        return 0

    if args.from_json is not None:
        jp = Path(args.from_json)
        if not jp.exists():
            print(f"error: --from-json file not found: {jp}", file=sys.stderr)
            return 2
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"error: --from-json is not valid JSON: {e}", file=sys.stderr)
            return 2
        records = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(r, dict) for r in records):
            print("error: --from-json must be an object or a list of objects",
                  file=sys.stderr)
            return 2
        # ATOMIC batch: build + validate EVERY record before appending ANY.
        # A mid-batch doctrine rejection must not leave a partial write — an
        # agent that fixes the bad record and retries the whole batch would
        # otherwise duplicate the records that had already landed.
        built_all = []
        for idx, rec in enumerate(records, 1):
            try:
                built, warnings = cat.build_catch(
                    catch=rec.get("catch", ""), pattern=rec.get("pattern", ""),
                    rule=rec.get("rule", ""), domain=rec.get("domain", "general"),
                    artifact_type=rec.get("artifact_type", ""),
                    date=rec.get("date", ""),
                    self_catch=rec.get("self_catch", False),
                    catch_id=rec.get("id", ""),
                    snippet=rec.get("snippet", ""),
                    lenses_failed=rec.get("lenses_failed"))
            except cat.CatchError as e:
                print(f"error: record {idx} of {len(records)}: {e} "
                      f"(nothing appended — the batch is all-or-nothing)",
                      file=sys.stderr)
                return 2
            built_all.append((built, warnings))
        for built, warnings in built_all:
            cat.append_catch(built, path)
            for w in warnings:
                print(f"warning: {w}", file=sys.stderr)
        print(f"Appended {len(built_all)} catch(es) to {path.name}")
        return 0

    if args.catch is None:
        print("error: catches add needs --catch (or --from-json / --seed)",
              file=sys.stderr)
        return 2
    snippet = ""
    if args.snippet_file is not None:
        sp = Path(args.snippet_file)
        if not sp.exists():
            print(f"error: --snippet-file not found: {sp}", file=sys.stderr)
            return 2
        snippet = sp.read_text(encoding="utf-8")  # verbatim, no strip
    try:
        built, warnings = cat.build_catch(
            catch=args.catch, pattern=args.pattern, rule=args.rule,
            domain=args.domain, artifact_type=args.artifact_type,
            date=args.date, self_catch=args.self_catch, catch_id=args.id,
            snippet=snippet, lenses_failed=args.lenses_failed)
    except cat.CatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cat.append_catch(built, path)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"Appended catch {built.id} to {path.name}")
    return 0


def _catches_export(args, cat, argv) -> int:
    """export --as-labels: caught defects -> GEPA training labels (item 6).

    Appends a ledger row like the other catches commands. Exits 0 even when
    nothing is eligible (an empty memory is a normal early state, not a usage
    error) — the printed count is the honest receipt.
    """
    catches_file = Path(args.catches) if args.catches else cat.catches_path()
    try:
        records = cat.load_catches(catches_file)
    except cat.CatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path = Path(args.as_labels)
    # Holdout-contamination guard: exported labels are TRAIN-side material;
    # landing them beside a frozen holdout is the classic train/test-leak
    # footgun. WARN (stderr, non-fatal) — a tripwire, not a guarantee.
    if cat.holdout_contamination_dir(out_path):
        print(f"warning: {out_path.parent}/ also contains a *holdout* file — "
              f"exported labels are TRAIN-side material; do not let them leak "
              f"into a frozen holdout (heuristic check, not a guarantee).",
              file=sys.stderr)

    try:
        labels, skipped = cat.build_export_labels(records, domain_filter=args.domain)
    except cat.CatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"{len(labels)} catches exported, {skipped} skipped "
          f"(missing snippet/lenses_failed) -> {out_path}")

    from . import ledger
    run_id = ledger.new_run_id("catches-export")
    ledger.append_run(
        run_id=run_id, command=_command_str(argv), checkpoint_sha=None,
        data_file=catches_file.name,
        metrics=f"{len(labels)} labels, {skipped} skipped",
        status="EXPORT")
    print(f"Ledger row appended: {ledger.LEDGER_FILENAME} ({run_id})")
    return 0


def _cmd_scrub(args) -> int:
    """Deterministic PII tripwire. Exit 0 = clean, 6 = findings, 2 = bad input.

    Raw matched values are NEVER printed — findings identify type, severity,
    and character offsets so the caller can locate the hit without the secret
    propagating into logs, CI output, or shell history.
    """
    from .pii import scan

    missing = _missing_files([args.file])
    if missing:
        print(f"error: file not found: {missing}", file=sys.stderr)
        return 2
    text = open(args.file, encoding="utf-8").read()
    result = scan(text)

    findings = [
        {"type": m.pii_type, "severity": m.severity,
         "start": m.start, "end": m.end}
        for m in result.matches
    ]
    if args.json:
        print(json.dumps({
            "clean": not result.has_pii,
            "halt": result.halt,
            "halt_reason": result.halt_reason,
            "findings": findings,
            "scrubbed_text": result.scrubbed_text,
        }, indent=2))
    else:
        print(result.scrubbed_text, end="")
        for f in findings:
            print(f"PII {f['severity'].upper()}: {f['type']} "
                  f"at {f['start']}-{f['end']}", file=sys.stderr)
        if result.halt:
            print(f"HALT: {result.halt_reason}", file=sys.stderr)

    return 6 if result.has_pii else 0


def _cmd_integrity(args, argv) -> int:
    from .config import IntegrityConfig
    from .integrity import run_integrity

    if args.last < 1:
        print("error: --last must be >= 1", file=sys.stderr)
        return 2
    config = IntegrityConfig()
    if args.profile:
        profile, err = _load_profile(args.profile)
        if err:
            return err
        config = profile.integrity
    # The ledger row this command appends lands in the SAME RUNS.md it monitors,
    # so the monitor is itself logged. run_integrity maps IntegrityError -> 2.
    return run_integrity(
        args.runs_md, args.catches, last=args.last, strict=args.strict,
        config=config, command_str=_command_str(argv))


def _cmd_improve(args, argv) -> int:
    from .improve import run_improve, run_improve_status

    # --status is a pure ledger read — no inputs, no profile, no spend.
    if args.status:
        return run_improve_status(ledger_path=args.runs_md)

    # A real improve run requires the four inputs + a profile.
    missing_required = [name for name, val in (
        ("--trainset", args.trainset), ("--holdout", args.holdout),
        ("--baseline-checkpoint", args.baseline_checkpoint),
        ("--profile", args.profile)) if not val]
    if missing_required:
        print(f"error: improve needs {', '.join(missing_required)} "
              f"(or pass --status to read the ledger)", file=sys.stderr)
        return 2

    catches = args.catches if args.catches else "catches.jsonl"
    for label, p in (("trainset", args.trainset), ("holdout", args.holdout),
                     ("baseline-checkpoint", args.baseline_checkpoint),
                     ("catches", catches)):
        if not Path(p).exists():
            print(f"error: {label} not found: {p}", file=sys.stderr)
            return 2

    if args.reruns < 1:
        print("error: --reruns must be >= 1", file=sys.stderr)
        return 2

    profile, err = _load_profile(args.profile)
    if err:
        return err

    try:
        return run_improve(
            trainset=args.trainset, holdout=args.holdout,
            baseline_checkpoint=args.baseline_checkpoint, catches=catches,
            output_dir=args.output, profile=profile, domain_filter=args.domain,
            approve_cost=args.approve_cost, threads=args.threads, auto=args.auto,
            max_evals=args.max_evals, val_split=args.val_split,
            patience=args.patience, reruns=args.reruns,
            allow_single_measurement=args.allow_single_measurement,
            command_str=_command_str(argv))
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _cmd_review(args) -> int:
    if not Path(args.results).exists():
        print(f"error: results path not found: {args.results}", file=sys.stderr)
        return 2
    # Import here so the rest of the CLI never pulls in the http server.
    from .review import run_review

    return run_review(
        args.results, port=args.port,
        previous_feedback=args.previous_feedback,
        static=args.static, open_browser=args.open)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "compile":
        return _cmd_compile(args, argv)
    if args.command == "eval":
        return _cmd_eval(args, argv)
    if args.command == "calibrate":
        return _cmd_calibrate(args, argv)
    if args.command == "mutate":
        return _cmd_mutate(args, argv)
    if args.command == "label-audit":
        return _cmd_label_audit(args, argv)
    if args.command == "sidecar":
        return _cmd_sidecar(args, argv)
    if args.command == "consistency":
        return _cmd_consistency(args, argv)
    if args.command == "catches":
        return _cmd_catches(args, argv)
    if args.command == "scrub":
        return _cmd_scrub(args)
    if args.command == "integrity":
        return _cmd_integrity(args, argv)
    if args.command == "improve":
        return _cmd_improve(args, argv)
    if args.command == "review":
        return _cmd_review(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def entry():  # console_scripts entry point
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
