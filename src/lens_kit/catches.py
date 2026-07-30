"""catches.jsonl — the institutional-memory loop (no LLM, no network).

A validator agent's value compounds only if its catches are remembered. This
module is the file-based memory: every defect a validator finds is appended to
a ``catches.jsonl`` in the working directory as one record, prior catches are
surfaced before a run so the agent walks in already knowing the recurring
traps, and a pattern that recurs enough times is flagged for promotion to a
deterministic check (the ``consistency`` command is the worked example — its
checks were built from exactly this kind of history).

DOCTRINE: routine passes are NOT recorded. The memory stays high-signal only
if it holds defects, not "looked fine" notes — a memory full of passes is a
memory no one reads. ``add`` therefore rejects an empty or pass-like ``catch``
(a catch must name a defect) and warns when a catch lands without a
generalizing ``pattern``/``rule`` (a catch without a generalization is half a
catch — you'll re-learn it the next time it appears in a new shape).

THE FLYWHEEL: agent catches a defect -> the same pattern recurs -> at a
threshold ``stats`` says "promote this to a deterministic check" -> the check
runs in pure Python forever after, cheaper and earlier than any LLM pass. The
shipped ``consistency`` checks (marker parity, leak scan, number parity) are
the first three patterns to make that journey.

No graph dependency, no service: one JSONL file you can read, diff, and commit.
"""
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2

CATCHES_FILENAME = "catches.jsonl"
SCHEMA_VERSION = 1
DEFAULT_PROMOTE_THRESHOLD = 3

# Fields stored per record, in stable write order.
#
# ``snippet`` and ``lenses_failed`` are OPTIONAL, backward-compatible additions
# (item 5): they serialize ONLY when present, so a row written before they
# existed stays byte-identical on round-trip and the stable paste-block / JSONL
# format contracts are unaffected. They appear LAST in write order — appended
# after the original fields — so older readers ignore them and the original
# field order is preserved verbatim.
FIELDS = ("id", "date", "domain", "artifact_type", "catch", "pattern", "rule",
          "self_catch", "schema_version", "snippet", "lenses_failed")

# Optional fields that are omitted from the record entirely when empty (the
# backward-compat contract: an old row gains no empty keys, a new row carries
# only the fields it actually has).
_OPTIONAL_FIELDS = ("snippet", "lenses_failed")

# A ``catch`` that is empty or pass-like is rejected: the memory is for
# defects, not routine passes (doctrine). Two mechanical rules — they stop the
# obvious routine-pass entries, they are not a semantic judge of catch quality:
#   1. exact match against the known one-liner pass phrases below;
#   2. closed-vocabulary: a SHORT catch (<= _PASS_MAX_WORDS words) composed
#      ENTIRELY of routine-pass vocabulary ("all lenses passed", "no issues
#      found", "everything checks out") names no defect. A real catch always
#      carries at least one content word outside this vocabulary (the defect
#      itself: "marker", "fabricated", "stale", ...), so a mention of passing
#      INSIDE a defect description ("the gate passed a fabricated stat")
#      survives.
_PASS_LIKE = (
    "", "pass", "passed", "passes", "pass.", "clean", "clean.", "ok", "okay",
    "looks good", "lgtm", "no issues", "no issue", "no violations",
    "no violation", "no problems", "no problem", "nothing wrong",
    "all good", "n/a", "na", "none", "-", "fine", "looks fine",
)

# Closed vocabulary for rule 2. Keep CONTENT words (e.g. "wrong", "stale",
# "marker") OUT of this set: a short vague defect like "the flags are wrong"
# must not be swallowed as a routine pass.
_PASS_VOCAB = frozenset((
    # quantifiers / negations
    "all", "every", "each", "everything", "nothing", "no", "none", "zero",
    # what gets checked
    "issue", "issues", "problem", "problems", "violation", "violations",
    "defect", "defects", "concern", "concerns", "flag", "flags",
    "finding", "findings", "error", "errors",
    "lens", "lenses", "check", "checks", "test", "tests", "gate", "gates",
    "validation", "validations", "scan", "scans", "review", "reviews",
    # pass verdicts
    "pass", "passed", "passes", "passing", "clean", "cleanly", "clear",
    "good", "great", "fine", "ok", "okay", "lgtm", "verified",
    "found", "detected", "noted", "observed", "reported", "present",
    # connective filler
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "out",
    "looks", "look", "looking", "and", "with", "here", "there", "it",
))
_PASS_MAX_WORDS = 6


class CatchError(Exception):
    """Raised on a malformed/doctrine-violating add. Maps to exit 2."""


# The canonical lens keys a ``lenses_failed`` entry is validated against. Sourced
# from the gate so the catch memory and the gate can never drift to different
# lens names. Imported lazily inside the validator to keep this module free of
# the dspy import chain that gate.py pulls in (load_catches / stats must run with
# no LLM stack installed).
def _canonical_lenses() -> tuple:
    from .gate import CANONICAL_LENSES
    return CANONICAL_LENSES


def validate_lenses_failed(lenses) -> list:
    """Normalize + validate a ``lenses_failed`` value -> a clean ``list[str]``.

    Accepts a list of lens keys or a comma-separated string (the CLI shape).
    Each key is validated against the canonical 10-lens set; an unknown key is a
    CatchError (-> exit 2) — fail closed, same doctrine as the gate's label
    validation, because a typo'd lens key would silently never invert in
    ``catches export --as-labels`` and the label would score wrong forever.
    Order is preserved; duplicates are collapsed (first occurrence wins).
    Empty / None -> [].
    """
    if lenses is None:
        return []
    if isinstance(lenses, str):
        items = [p.strip() for p in lenses.split(",")]
    else:
        items = [str(p).strip() for p in lenses]
    items = [p for p in items if p]
    canon = _canonical_lenses()
    unknown = [p for p in items if p not in canon]
    if unknown:
        raise CatchError(
            f"unknown lens key(s) {unknown} in lenses_failed — canonical "
            f"lenses: {', '.join(canon)}. A non-canonical key can never invert "
            f"correctly in `catches export`; fix the key.")
    seen = set()
    out = []
    for p in items:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ── Record shape ─────────────────────────────────────────────────

@dataclass
class Catch:
    """One institutional-memory record. ``catch`` is the only required field.

    ``snippet`` (the offending text, verbatim — the thing the validator caught)
    and ``lenses_failed`` (the lens keys it failed) are OPTIONAL. They are
    serialized only when present (see ``to_record``), so a record without them
    round-trips to the exact original FIELDS bytes — old rows stay valid and the
    stable paste-block / format tests are untouched.
    """
    catch: str
    pattern: str = ""
    rule: str = ""
    domain: str = "general"
    artifact_type: str = ""
    date: str = ""
    self_catch: bool = False
    id: str = ""
    schema_version: int = SCHEMA_VERSION
    snippet: str = ""
    lenses_failed: list = field(default_factory=list)

    def to_record(self) -> "OrderedDict":
        """Ordered dict in the stable FIELDS order (stable JSONL bytes).

        The optional ``snippet`` / ``lenses_failed`` are emitted ONLY when
        non-empty (appended last, after ``schema_version``). An empty optional
        field is omitted entirely so a record without it is byte-identical to a
        pre-extension row.
        """
        rec = OrderedDict()
        rec["id"] = self.id
        rec["date"] = self.date
        rec["domain"] = self.domain
        rec["artifact_type"] = self.artifact_type
        rec["catch"] = self.catch
        rec["pattern"] = self.pattern
        rec["rule"] = self.rule
        rec["self_catch"] = bool(self.self_catch)
        rec["schema_version"] = int(self.schema_version)
        if self.snippet:
            rec["snippet"] = self.snippet
        if self.lenses_failed:
            rec["lenses_failed"] = list(self.lenses_failed)
        return rec


def is_pass_like(catch: str) -> bool:
    """True if ``catch`` names no defect (empty or a routine-pass phrase).

    Doctrine heuristic: the memory excludes routine passes BY DESIGN to stay
    high-signal. Pass-like means (1) one of the known "nothing wrong" phrases
    (case-insensitive, trailing punctuation ignored), or (2) a short text
    (<= 6 words) built entirely from routine-pass vocabulary — "all lenses
    passed", "no issues found", "everything checks out". A real catch always
    contains a content word outside that vocabulary, so defect descriptions
    that MENTION passing are not rejected.
    """
    norm = " ".join((catch or "").split()).strip().lower().rstrip(".!?").strip()
    if norm in _PASS_LIKE:
        return True
    words = [w.strip(",;:'\"()") for w in norm.split()]
    words = [w for w in words if w]
    if not words:
        return True
    if len(words) <= _PASS_MAX_WORDS and all(
            w in _PASS_VOCAB or w.isdigit() for w in words):
        return True
    return False


def normalize_pattern(pattern: str) -> str:
    """Canonical key for recurrence counting: lowercased, whitespace-collapsed.

    Two catches that phrase the same pattern with different spacing/case count
    as the same recurring pattern. Punctuation is kept (it can be meaningful),
    only case and runs of whitespace are normalized.
    """
    return " ".join((pattern or "").split()).lower()


def build_catch(*, catch: str, pattern: str = "", rule: str = "",
                domain: str = "general", artifact_type: str = "",
                date: str = "", self_catch: bool = False,
                catch_id: str = "", snippet: str = "",
                lenses_failed=None) -> tuple:
    """Validate + construct a Catch. Returns ``(Catch, warnings)``.

    Doctrine enforcement (raises CatchError -> exit 2):
      - ``catch`` must name a defect: empty or pass-like text is rejected.
      - ``lenses_failed`` keys must be canonical (validate_lenses_failed).
    Warnings (returned, not fatal — a half-catch is still recorded):
      - missing ``pattern`` AND/OR ``rule``: a catch without a generalization
        is half a catch (you'll re-learn it in its next shape).
    Optional (item 5): ``snippet`` (the offending text, verbatim) and
    ``lenses_failed`` (canonical lens keys; list or comma-string). Both are
    stored only when present — see Catch.to_record. ``snippet`` is NOT stripped
    (verbatim bytes are the point — it becomes the training-label ``text``).
    Defaults filled here: ``date`` -> today (ISO), ``id`` -> derived from
    date + artifact_type + a short content hash for a stable, readable id.
    """
    if is_pass_like(catch):
        raise CatchError(
            "catch must NAME A DEFECT — empty or pass-like entries are rejected "
            "by doctrine (routine passes are excluded to keep the memory "
            "high-signal). Record what was WRONG, or do not record.")

    lenses = validate_lenses_failed(lenses_failed)

    warnings = []
    if not (pattern or "").strip():
        warnings.append("no pattern: a catch without a generalization is half a "
                        "catch — add what GENERAL trap this is an instance of.")
    if not (rule or "").strip():
        warnings.append("no rule: add the forward rule that would PREVENT this "
                        "next time (the half a catch you keep is the rule).")

    date = (date or "").strip() or _date.today().isoformat()
    artifact_type = (artifact_type or "").strip()
    domain = (domain or "general").strip() or "general"
    cid = (catch_id or "").strip() or _derive_id(date, artifact_type, catch)

    rec = Catch(
        catch=catch.strip(), pattern=(pattern or "").strip(),
        rule=(rule or "").strip(), domain=domain, artifact_type=artifact_type,
        date=date, self_catch=bool(self_catch), id=cid,
        snippet=(snippet or ""), lenses_failed=lenses)
    return rec, warnings


def _derive_id(date: str, artifact_type: str, catch: str) -> str:
    """Stable, readable id: ``<date>-<artifact-slug>-<6-hex>`` of the catch text."""
    import hashlib
    slug = "".join(c if c.isalnum() else "-" for c in artifact_type.lower()).strip("-")
    slug = slug or "catch"
    digest = hashlib.sha256(catch.strip().encode("utf-8")).hexdigest()[:6]
    return f"{date}-{slug}-{digest}"


# ── Read / write the JSONL store ─────────────────────────────────

def catches_path(working_dir=None) -> Path:
    return (Path(working_dir) if working_dir else Path.cwd()) / CATCHES_FILENAME


def load_catches(path) -> list:
    """Read every record from a catches.jsonl. Missing file -> []. Skips blanks.

    Malformed JSON lines raise CatchError (with the 1-based line number) rather
    than being silently dropped — a memory file you can't fully read is a
    memory you can't trust.
    """
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise CatchError(f"{p.name}:{lineno}: malformed JSON ({e})") from e
        if not isinstance(rec, dict):
            raise CatchError(f"{p.name}:{lineno}: record is not a JSON object")
        out.append(rec)
    return out


def append_catch(catch: Catch, path) -> Path:
    """Append one record as a single JSON line (creates the file if absent)."""
    p = Path(path)
    line = json.dumps(catch.to_record(), ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return p


def seed_path() -> Path:
    """Path to the shipped example catches (package data)."""
    return Path(__file__).parent / "catches_seed.jsonl"


def load_seed() -> list:
    """The shipped, fully-genericized example catches (6 records)."""
    return load_catches(seed_path())


# ── Recurrence + promotion ───────────────────────────────────────

def pattern_counts(records: list) -> tuple:
    """Returns ``(counts, display)``.

    ``counts``: OrderedDict of normalized-pattern -> count, most-recurrent
    first. ``display``: normalized-pattern -> the first-seen original phrasing
    (for human-readable output). Records with an empty pattern are excluded
    from recurrence (they have no generalization to recur on). Insertion order
    within an equal count is the order of first appearance — stable across
    runs on the same file.
    """
    counter = Counter()
    display = {}
    for rec in records:
        norm = normalize_pattern(rec.get("pattern", ""))
        if not norm:
            continue
        counter[norm] += 1
        display.setdefault(norm, (rec.get("pattern") or "").strip())
    ordered = OrderedDict()
    for norm, count in counter.most_common():
        ordered[norm] = count
    return ordered, display


def promotable(records: list, threshold: int = DEFAULT_PROMOTE_THRESHOLD) -> list:
    """[(display_pattern, count)] for patterns at/over ``threshold``, busiest first."""
    counts, display = pattern_counts(records)
    return [(display[norm], count) for norm, count in counts.items()
            if count >= threshold]


_PROMOTE_LINE = (
    "PROMOTE: this pattern has recurred {count}x — consider encoding it as a "
    "deterministic check (see lens-kit consistency; extension point: "
    "consistency.py).")


def promote_line(count: int) -> str:
    return _PROMOTE_LINE.format(count=count)


# ── relevant: the paste-into-a-validator-agent surface ───────────

def filter_records(records: list, artifact_type=None, domain=None) -> list:
    """Filter by artifact_type and/or domain (case-insensitive exact match)."""
    out = records
    if artifact_type is not None:
        at = artifact_type.strip().lower()
        out = [r for r in out if (r.get("artifact_type") or "").lower() == at]
    if domain is not None:
        dm = domain.strip().lower()
        out = [r for r in out if (r.get("domain") or "").lower() == dm]
    return out


def order_for_relevant(records: list,
                       threshold: int = DEFAULT_PROMOTE_THRESHOLD) -> list:
    """Order records most-recurrent-pattern first, then first-seen order.

    Stable, documented ordering for the paste block: records whose pattern
    recurs most often come first (a recurring trap is the one most worth
    re-reading before a run); within an equal recurrence count, original file
    order is preserved. Records with no pattern sort last (count treated as 0).
    """
    counts, _ = pattern_counts(records)
    indexed = list(enumerate(records))

    def key(item):
        idx, rec = item
        norm = normalize_pattern(rec.get("pattern", ""))
        count = counts.get(norm, 0)
        return (-count, idx)  # busiest pattern first, then first-seen order

    return [rec for _, rec in sorted(indexed, key=key)]


def render_relevant_block(records: list, *, artifact_type=None, domain=None,
                          threshold: int = DEFAULT_PROMOTE_THRESHOLD) -> str:
    """The STABLE, documented paste-block of prior catches for an agent context.

    Format contract (do not reorder fields — agents and tests depend on it):

        ## PRIOR CATCHES (institutional memory)
        scope: artifact_type=<at|any>, domain=<d|any>; <N> catch(es)
        Read these before validating. Most-recurrent patterns first. A pattern
        marked PROMOTE has recurred enough to deserve a deterministic check.

        [PROMOTE] <pattern>  (<count>x)            # zero or more, busiest first
        ...

        - CATCH: <catch>
          PATTERN: <pattern or "(none)">
          RULE: <rule or "(none)">
          (self-catch)                              # only when self_catch is true
        ... (one block per record, most-recurrent pattern first)

    The PROMOTE lines at the TOP surface promote-ready patterns up front; the
    per-record blocks follow in ``order_for_relevant`` order.
    """
    scope_at = artifact_type if artifact_type else "any"
    scope_dm = domain if domain else "any"
    n = len(records)

    lines = ["## PRIOR CATCHES (institutional memory)",
             f"scope: artifact_type={scope_at}, domain={scope_dm}; {n} catch(es)",
             "Read these before validating. Most-recurrent patterns first. A "
             "pattern marked PROMOTE has recurred enough to deserve a "
             "deterministic check.",
             ""]

    for pattern, count in promotable(records, threshold):
        lines.append(f"[PROMOTE] {pattern}  ({count}x)")
    if promotable(records, threshold):
        lines.append("")

    if not records:
        lines.append("(no prior catches recorded for this scope)")
        return "\n".join(lines) + "\n"

    for rec in order_for_relevant(records, threshold):
        catch = (rec.get("catch") or "").strip()
        pattern = (rec.get("pattern") or "").strip() or "(none)"
        rule = (rec.get("rule") or "").strip() or "(none)"
        lines.append(f"- CATCH: {catch}")
        lines.append(f"  PATTERN: {pattern}")
        lines.append(f"  RULE: {rule}")
        if rec.get("self_catch"):
            lines.append("  (self-catch)")
    return "\n".join(lines) + "\n"


# ── stats rendering ──────────────────────────────────────────────

def render_stats(records: list,
                 threshold: int = DEFAULT_PROMOTE_THRESHOLD) -> str:
    """Per-pattern recurrence counts + a PROMOTE line for any at/over threshold."""
    counts, display = pattern_counts(records)
    total = len(records)
    distinct = len(counts)
    lines = [f"[catches:stats] {total} catch(es), {distinct} distinct pattern(s), "
             f"promote threshold {threshold}"]
    if not counts:
        lines.append("  (no patterns recorded — catches added without a pattern "
                     "are not counted for recurrence)")
        return "\n".join(lines) + "\n"
    for norm, count in counts.items():
        lines.append(f"  {count}x  {display[norm]}")
        if count >= threshold:
            lines.append(f"        {promote_line(count)}")
    return "\n".join(lines) + "\n"


# ── export: catches -> GEPA training labels (the RLHS hindsight loop) ──

def build_export_labels(records: list, *, domain_filter=None) -> tuple:
    """Build GEPA training examples from catches with snippet + lenses_failed.

    Returns ``(labels, skipped)``:
      labels  -> list of {text, domain, per_lens, source, source_id, text_length}
                 in the PROVEN schema (matches the production training shape).
                 ``per_lens`` inverts the catch: every lens in ``lenses_failed``
                 is False (a real planted defect), every OTHER canonical lens is
                 True. ``text`` = the verbatim snippet, ``source`` = "catches",
                 ``source_id`` = the catch id.
      skipped -> count of catches lacking a snippet OR lenses_failed (NOT a
                 silent cap — the caller prints this count).

    Dedupe is by ``source_id`` (the catch id): the first eligible catch per id
    wins, later duplicates are dropped (not counted as skipped — they are
    eligible, just already represented).
    ``domain_filter`` (case-insensitive exact match) narrows the input first.
    A catch's ``lenses_failed`` keys are re-validated here (fail closed); a row
    that somehow carries a non-canonical key raises CatchError rather than
    emitting a label that can never score.
    """
    canon = _canonical_lenses()
    if domain_filter is not None:
        records = filter_records(records, domain=domain_filter)

    labels = []
    skipped = 0
    seen_ids = set()
    for rec in records:
        snippet = rec.get("snippet")
        lenses = rec.get("lenses_failed")
        if not snippet or not lenses:
            skipped += 1
            continue
        sid = (rec.get("id") or "").strip()
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        failed = validate_lenses_failed(lenses)
        per_lens = {lens: (lens not in failed) for lens in canon}
        labels.append({
            "text": snippet,
            "domain": (rec.get("domain") or "general"),
            "per_lens": per_lens,
            "source": "catches",
            "source_id": sid,
            "text_length": len(snippet),
        })
    return labels, skipped


def holdout_contamination_dir(out_path) -> bool:
    """True if ``out_path``'s directory co-locates a likely frozen holdout file.

    Heuristic (documented footgun-reducer, NOT a guarantee): the kit has no
    enforced holdout-filename convention — a holdout is just a user-named JSON
    passed to ``eval`` — so this warns when the OUTPUT directory contains any
    OTHER file whose name matches ``*holdout*`` (case-insensitive). Exported
    labels are TRAIN-side material; landing them beside a frozen holdout is the
    classic train/test-leak footgun. Limits, stated honestly:
      - misses holdouts named without "holdout" in the filename;
      - misses holdouts in a sibling/parent dir;
      - false-positives on an unrelated file that happens to match.
    It is a tripwire that says "look here", never a proof of non-contamination.
    """
    out_path = Path(out_path)
    out_dir = out_path.parent if out_path.parent != Path("") else Path(".")
    if not out_dir.is_dir():
        return False
    out_name = out_path.name.lower()
    for sibling in out_dir.iterdir():
        if not sibling.is_file():
            continue
        name = sibling.name.lower()
        if name == out_name:
            continue  # the export target itself is not a holdout
        if "holdout" in name:
            return True
    return False
