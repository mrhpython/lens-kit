"""Profile configuration — provider/model + domain vocabulary, externalized.

Provider-agnostic by design: the customer brings ANY litellm-compatible
endpoint. The ``llm.model`` field is a litellm model string passed straight
to ``dspy.LM``:

    xai/grok-4-1-fast-reasoning          (xAI)
    openai/gpt-4o                        (OpenAI)
    anthropic/claude-sonnet-4-5          (Anthropic)
    gemini/gemini-2.5-pro                (Google)
    ollama_chat/llama3                   (local Ollama, with api_base)
    openai/meta-llama/Llama-3.3-70B-Instruct  (any OpenAI-compatible host, with api_base)

API keys are NEVER stored in profiles — only the NAME of the environment
variable that holds the key (``api_key_env``). Missing key = hard error
(fail closed), never a silent fallback to another provider.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .filters import LensVocabulary


class ConfigError(Exception):
    """Raised on invalid profile or missing credentials. Fail closed."""


@dataclass
class LLMConfig:
    """Maps 1:1 onto dspy.LM(model=..., api_base=..., api_key=..., ...)."""
    model: str = ""
    api_base: str | None = None
    api_key_env: str | None = None   # NAME of env var holding the key — never a key value
    temperature: float = 0.1
    max_tokens: int = 8000
    extra_body: dict | None = None   # server-side params forwarded verbatim to an
                                     # OpenAI-compatible endpoint. Use to suppress reasoning
                                     # on thinking-capable models (Qwen3.x, etc.):
                                     # {"reasoning_effort": "none",
                                     #  "chat_template_kwargs": {"enable_thinking": false}}

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown llm keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
        }


@dataclass
class CompileConfig:
    """Costing-gate + pace-monitor knobs for `lens-kit compile`.

    cost_per_rollout_usd defaults to None = UNKNOWN: without it the costing
    gate prints projected rollouts + wall-clock hours only, never an invented
    dollar figure. Set it from a measured probe on YOUR endpoint.

    approval_threshold_hours: projections above this require an explicit
    --approve-cost flag (hard stop otherwise). pace_kill_factor: abort the
    run when measured pace exceeds the approved projection by this factor.
    reflection: optional separate LLM block for GEPA's reflection model;
    default = the profile's task LM at temperature 1.0.
    """
    cost_per_rollout_usd: float | None = None
    approval_threshold_hours: float = 2.0
    pace_kill_factor: float = 1.5
    pace_min_calls: int = 5
    probe_examples: int = 3
    reflection: "LLMConfig | None" = None
    # Optional per-lens severity weight map for the compile/eval metric. None
    # (the default, and the shape of every older profile) -> the unweighted F2
    # behavior, byte-identical. When set, names lens->weight to up/down-weight a
    # lens's contribution (e.g. {truth: 2.0, structure: 0.5}); a lens absent from
    # the map weighs 1.0. Validated lazily by the metric (canonical keys, >= 0).
    metric_weights: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "CompileConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown compile keys: {sorted(unknown)}")
        kwargs = dict(data)
        if kwargs.get("reflection") is not None:
            kwargs["reflection"] = LLMConfig.from_dict(kwargs["reflection"])
        if kwargs.get("metric_weights") is not None:
            mw = kwargs["metric_weights"]
            if not isinstance(mw, dict):
                raise ConfigError("compile.metric_weights must be a mapping of "
                                  "lens -> weight")
            kwargs["metric_weights"] = {str(k): float(v) for k, v in mw.items()}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "cost_per_rollout_usd": self.cost_per_rollout_usd,
            "approval_threshold_hours": self.approval_threshold_hours,
            "pace_kill_factor": self.pace_kill_factor,
            "pace_min_calls": self.pace_min_calls,
            "probe_examples": self.probe_examples,
            "reflection": self.reflection.to_dict() if self.reflection else None,
            "metric_weights": (dict(self.metric_weights)
                               if self.metric_weights is not None else None),
        }


@dataclass
class CalibrationConfig:
    """Customer-vocabulary slots for the planted-flaw calibration battery.

    The calibration generator (calibration.py) is TEMPLATE-based and makes no
    LLM calls; these slots are the only customer-specific text it interpolates
    into the planted-flaw fixtures. Defaults are deliberately generic so an
    empty profile still generates a runnable, on-topic battery — but a battery
    that reads like a real customer's content is more representative, so set
    these to YOUR company/product/audience vocabulary.

    Grammar contract: every slot value must read naturally MID-SENTENCE
    (templates never prepend articles/adjectives) — e.g. product
    "our AI review platform" works; "The Platform(TM)." with trailing
    punctuation will read broken.

    Slots:
      company        -> the sender/brand name used in fixtures
      product        -> the thing being sold/described (mid-sentence noun phrase)
      audience       -> who the copy addresses (also the Relevance context)
      competitor     -> a named rival, used in contradiction/darkpattern copy
      output_type    -> what kind of artifact the copy is (e.g. "cold email")
      price          -> a concrete price string used in pricing/contradiction
      stat           -> an uncited statistic (the planted Truth flaw anchor)
      metric         -> a second uncited number for unsourced fixtures
      timeframe      -> a period used in contradiction-timeline fixtures
    """
    company: str = "Acme Agency"
    product: str = "the platform"
    audience: str = "prospective customers"
    competitor: str = "other vendors"
    output_type: str = "marketing copy"
    price: str = "$99/month"
    stat: str = "87% of buyers"
    metric: str = "3x improvement"
    timeframe: str = "90 days"

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown calibration keys: {sorted(unknown)}")
        return cls(**{k: str(v) for k, v in data.items()})

    def to_dict(self) -> dict:
        return {k: str(getattr(self, k)) for k in self.__dataclass_fields__}


@dataclass
class ConsistencyConfig:
    """Deny-list + marker set for `lens-kit consistency` (deterministic, no LLM).

    These checks are TRIPWIRES, not oracles — see consistency.py docstrings
    and the README. The defaults are deliberately generic; ship YOUR internal
    vocabulary in the deny-list so customer-facing copy never leaks it.

    Slots:
      markers   -> evidence markers whose count must survive source -> rendered
                   (marker-parity check). Default: the three honesty markers.
      deny      -> case-insensitive literal strings that must NEVER appear in
                   customer-facing copy (internal lens names/counts, cost data,
                   internal vocabulary). One hit = a consistency violation that
                   no score overrides (recovery 1.10).
    """
    markers: list = field(default_factory=lambda: ["[UNVERIFIED]", "[ESTIMATE]",
                                                   "[PROJECTION]"])
    deny: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ConsistencyConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown consistency keys: {sorted(unknown)}")
        kwargs = {}
        if "markers" in data:
            if not isinstance(data["markers"], list):
                raise ConfigError("consistency.markers must be a list of strings")
            kwargs["markers"] = [str(m) for m in data["markers"]]
        if "deny" in data:
            if not isinstance(data["deny"], list):
                raise ConfigError("consistency.deny must be a list of strings")
            kwargs["deny"] = [str(d) for d in data["deny"]]
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {"markers": list(self.markers), "deny": list(self.deny)}


@dataclass
class IntegrityConfig:
    """Thresholds for `lens-kit integrity` — the gate-integrity trend monitor.

    The monitor is a TRIPWIRE, not an oracle (see integrity.py). These knobs set
    when it raises a flag; a flag means "look here", never "the gate is broken".
    Defaults are conservative: they fire only on the unambiguous rubber-stamp
    shape (a high AND rising pass-rate over enough runs with ZERO catches in the
    window). All keys are optional — a profile with no `integrity:` block uses
    these defaults, so older profiles stay valid.

    Slots:
      pass_rate_warn     -> pass-rate at/above this (over >= min_runs eval runs)
                            is "high"; combined with zero catches + a non-falling
                            trend it trips the rubber-stamp flag.
      min_runs           -> minimum eval runs in the window before the pass-rate
                            tripwire is allowed to fire (too few runs = no trend).
      stale_catch_runs   -> runs since the last named catch at/above this is
                            flagged (the memory has gone quiet).
      stale_catch_days   -> days since the last named catch at/above this is
                            flagged.
      stale_mutation_runs-> eval/validate runs since the last mutation-control
                            run at/above this is flagged (rubber-stamp detector
                            itself has gone stale). Surfaced as "unknown" — never
                            guessed — when the ledger records no mutation run.
    """
    pass_rate_warn: float = 0.95
    min_runs: int = 5
    stale_catch_runs: int = 10
    stale_catch_days: int = 30
    stale_mutation_runs: int = 20

    @classmethod
    def from_dict(cls, data: dict) -> "IntegrityConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown integrity keys: {sorted(unknown)}")
        kwargs = {}
        for k in ("pass_rate_warn",):
            if k in data:
                kwargs[k] = float(data[k])
        for k in ("min_runs", "stale_catch_runs", "stale_catch_days",
                  "stale_mutation_runs"):
            if k in data:
                kwargs[k] = int(data[k])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "pass_rate_warn": self.pass_rate_warn,
            "min_runs": self.min_runs,
            "stale_catch_runs": self.stale_catch_runs,
            "stale_catch_days": self.stale_catch_days,
            "stale_mutation_runs": self.stale_mutation_runs,
        }


@dataclass
class ImproveConfig:
    """Knobs for `lens-kit improve` — the challenger/promote loop.

    fp_regression_tolerance: how much the challenger's false-positive rate may
    rise vs the baseline and still be promotable. Default 0.0 — NO FP increase is
    allowed, because the product claim is "we don't over-flag": a challenger that
    catches more only by flagging more is over-flagging, not improving. A profile
    may set a small documented epsilon when a tiny FP rise buys a large catch
    gain; it must be >= 0 (a negative tolerance would be incoherent). All keys
    optional — a profile with no `improve:` block uses these defaults, so older
    profiles stay valid.
    """
    fp_regression_tolerance: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "ImproveConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown improve keys: {sorted(unknown)}")
        kwargs = {}
        if "fp_regression_tolerance" in data:
            tol = float(data["fp_regression_tolerance"])
            if tol < 0:
                raise ConfigError(
                    f"improve.fp_regression_tolerance must be >= 0 (got {tol}) — "
                    f"a negative FP tolerance is incoherent.")
            kwargs["fp_regression_tolerance"] = tol
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {"fp_regression_tolerance": self.fp_regression_tolerance}


@dataclass
class Profile:
    """A complete customer profile: LLM endpoint + domain vocabulary.

    Empty-profile defaults:
    - llm: unset (make_lm raises until configured)
    - vocabulary: all lists empty -> no deterministic suppression
    - domain_rules: empty -> no domain strictness preambles
    - domain_detection: empty -> domain auto-detection always returns 'general'
    - compile: costing-gate defaults (2h approval threshold, 1.5x pace kill)
    - calibration: generic customer-vocabulary slots (template fixtures)
    """
    name: str = "empty"
    llm: LLMConfig = field(default_factory=LLMConfig)
    vocabulary: LensVocabulary = field(default_factory=LensVocabulary)
    domain_rules: dict = field(default_factory=dict)       # domain -> strictness preamble text
    domain_detection: dict = field(default_factory=dict)   # domain -> keyword list (order = priority)
    compile: CompileConfig = field(default_factory=CompileConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    improve: ImproveConfig = field(default_factory=ImproveConfig)

    @classmethod
    def empty(cls) -> "Profile":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        known = {"name", "llm", "vocabulary", "domain_rules", "domain_detection",
                 "compile", "calibration", "consistency", "integrity", "improve"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown profile keys: {sorted(unknown)}")
        try:
            vocab = LensVocabulary.from_dict(data.get("vocabulary") or {})
        except ValueError as e:
            raise ConfigError(str(e)) from e
        return cls(
            name=data.get("name", "unnamed"),
            llm=LLMConfig.from_dict(data.get("llm") or {}),
            vocabulary=vocab,
            domain_rules=dict(data.get("domain_rules") or {}),
            domain_detection={k: list(v) for k, v in (data.get("domain_detection") or {}).items()},
            compile=CompileConfig.from_dict(data.get("compile") or {}),
            calibration=CalibrationConfig.from_dict(data.get("calibration") or {}),
            consistency=ConsistencyConfig.from_dict(data.get("consistency") or {}),
            integrity=IntegrityConfig.from_dict(data.get("integrity") or {}),
            improve=ImproveConfig.from_dict(data.get("improve") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Profile not found: {path}")
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ConfigError(f"Profile must be a YAML mapping: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "llm": self.llm.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "domain_rules": dict(self.domain_rules),
            "domain_detection": {k: list(v) for k, v in self.domain_detection.items()},
            "compile": self.compile.to_dict(),
            "calibration": self.calibration.to_dict(),
            "consistency": self.consistency.to_dict(),
            "integrity": self.integrity.to_dict(),
            "improve": self.improve.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))


def builtin_profile_path(name: str = "agency-example") -> Path:
    """Path to a profile shipped inside the package."""
    return Path(__file__).parent / "profiles" / f"{name}.yaml"


def make_lm(cfg: LLMConfig):
    """Build a dspy.LM from an LLMConfig. No fallback chain — fail closed.

    Mapping:
      cfg.model       -> dspy.LM(model)        litellm model string, any provider
      cfg.api_base    -> dspy.LM(api_base=...) only when set (local/self-hosted endpoints)
      cfg.api_key_env -> dspy.LM(api_key=os.environ[api_key_env]); missing env var raises
      cfg.temperature, cfg.max_tokens -> passed through
    """
    import dspy  # deferred: config loading must not require a configured LM

    if not cfg.model:
        raise ConfigError("Profile has no llm.model — set a litellm model string "
                          "(e.g. 'openai/gpt-4o', 'ollama_chat/llama3').")
    kwargs = {
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    if cfg.api_key_env:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise ConfigError(
                f"Environment variable '{cfg.api_key_env}' is not set. "
                f"Refusing to run without credentials (fail closed)."
            )
        kwargs["api_key"] = key
    if cfg.extra_body:
        # Server-side params forwarded verbatim to an OpenAI-compatible endpoint
        # (e.g. suppress reasoning on a thinking-capable model). drop_params stops
        # litellm hard-failing if the provider does not recognise a key; the
        # defensive downstream parsers remain the real guard.
        kwargs["extra_body"] = cfg.extra_body
        import litellm
        litellm.drop_params = True
    return dspy.LM(cfg.model, **kwargs)


def configure_from_profile(profile: Profile):
    """Build the LM and set it as the global dspy default.

    Returns ``(lm, previous_lm)`` so the caller can restore the prior
    global state (``dspy.configure(lm=previous_lm)``). ``previous_lm`` is
    None when no LM was configured. Prefer ``lm_context`` for scoped use —
    teacher/student swaps in a compile harness should never leak a global.
    """
    import dspy

    previous_lm = getattr(dspy.settings, "lm", None)
    lm = make_lm(profile.llm)
    dspy.configure(lm=lm)
    return lm, previous_lm


@contextmanager
def lm_context(profile: Profile):
    """Scoped LM override: ``with lm_context(profile) as lm: ...``.

    Builds the LM from the profile (fail-closed) and installs it via
    dspy's thread-local context override; the previous LM is restored on
    exit, including on exception. This is the intended surface for the C2
    compile harness (teacher/student swaps, checkpoint eval runs).
    """
    import dspy

    lm = make_lm(profile.llm)
    with dspy.context(lm=lm):
        yield lm
