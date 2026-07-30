"""Costing gate + pace monitor for the compile harness.

These exist because of two documented real incidents:
1. A per-lens GEPA run projected ~$360 of API spend; it was killed at $14.53
   only because a human happened to be watching. The projection now runs as
   CODE, before the optimizer, and a projection over the approval threshold
   is a HARD STOP without an explicit --approve-cost flag.
2. Long runs whose live pace drifts above projection burn the budget the
   gate approved. The pace monitor aborts with a structured kill record the
   moment measured pace exceeds the approved projection by a configured
   factor.

Projection model:
    projected_rollouts = GEPA's own metric-call budget accounting
        --max-evals N  -> N x (train + val)            (dspy GEPA semantics)
        --auto X       -> GEPA.auto_budget(num_preds, AUTO_RUN_SETTINGS[X],
                                           valset_size)
      plus the probe rollouts and the one post-compile val eval the harness
      itself performs.
    seconds/rollout    = measured on a small live probe chunk (serial).
    projected hours    = rollouts x sec_per_rollout / threads (assumes
                         linear thread scaling — optimistic; treat as floor).
    projected cost     = rollouts x cost_per_rollout_usd, ONLY when the
                         profile sets compile.cost_per_rollout_usd. Default
                         is UNKNOWN -> rollouts + hours are printed, never
                         an invented dollar figure.

Reflection-LM calls are NOT counted (GEPA budgets metric calls only), so
the projection is a floor, not a ceiling.
"""
import threading
import time
from dataclasses import dataclass, field


def measure_pace_fresh_cache(program, examples, probe_n: int = 3,
                             clock=time.monotonic, *, _measure=None) -> float:
    """measure_pace against an ISOLATED fresh dspy cache (restored after).

    The compile pace probe exists to estimate real seconds-per-rollout so the
    pace monitor's kill threshold is set to a true figure. If the probe runs
    against a WARM dspy disk cache (left over from a prior run, an eval, or an
    earlier compile of the same data), every probe call is a cache replay and
    measures sub-second pace — then the FIRST real (uncached) GEPA rollout
    blows that bogus threshold and trips a FALSE pace kill (the warm-cache
    incident: probe 0.11s/rollout, real call 37s/rollout, killed at exit 4).

    A probe that's accidentally cached lies about money, so the probe is
    measured against a brand-new cache directory with the in-process memory
    cache disabled. The process-global cache config is restored on exit
    (including on exception) so the subsequent GEPA compile uses whatever
    cache the caller intended.

    ``_measure`` injects the timing function (defaults to this module's
    measure_pace); the compile harness passes its own module-level reference
    so a single monkeypatch point covers both the warm and fresh probe paths.
    Imports live inside the function so config-only code never pulls in dspy.
    """
    from .eval_harness import configure_fresh_cache, preserve_dspy_cache

    measure = _measure if _measure is not None else measure_pace
    with preserve_dspy_cache():
        configure_fresh_cache()
        return measure(program, examples, probe_n=probe_n, clock=clock)


def gepa_projected_metric_calls(*, num_preds: int, train_size: int, val_size: int,
                                auto: str | None = None,
                                max_evals: int | None = None) -> int:
    """GEPA's own budget accounting for projected metric calls (rollouts).

    Mirrors dspy.teleprompt.GEPA.compile():
      max_full_evals -> max_full_evals * (len(train) + len(val))
      auto           -> GEPA.auto_budget(num_preds, AUTO_RUN_SETTINGS[auto]['n'],
                        valset_size=len(val) or len(train))
    Exactly one of auto / max_evals must be set.
    """
    if (auto is None) == (max_evals is None):
        raise ValueError("Exactly one of auto / max_evals must be set.")
    if max_evals is not None:
        return int(max_evals) * (train_size + val_size)

    from dspy.teleprompt import GEPA
    from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS

    if auto not in AUTO_RUN_SETTINGS:
        raise ValueError(f"Unknown auto setting: {auto!r} "
                         f"(expected one of {sorted(AUTO_RUN_SETTINGS)})")
    # auto_budget does not use self — call it unbound, exactly as GEPA would.
    return int(GEPA.auto_budget(
        None,
        num_preds=num_preds,
        num_candidates=AUTO_RUN_SETTINGS[auto]["n"],
        valset_size=val_size if val_size else train_size,
    ))


def measure_pace(program, examples, probe_n: int = 3,
                 clock=time.monotonic) -> float:
    """Seconds per rollout, measured live on the first probe_n examples.

    These are real (billed) calls — the price of an honest projection.
    """
    sample = list(examples)[:max(1, probe_n)]
    if not sample:
        raise ValueError("Cannot measure pace on an empty example set.")
    t0 = clock()
    for ex in sample:
        program(**ex.inputs())
    elapsed = clock() - t0
    return elapsed / len(sample)


@dataclass
class CostProjection:
    """The pre-run costing receipt. Built BEFORE the optimizer is invoked."""
    projected_rollouts: int
    budget: str                      # e.g. "auto=light" / "max_evals=8"
    train_size: int
    val_size: int
    seconds_per_rollout: float       # measured, serial
    threads: int
    cost_per_rollout_usd: float | None
    threshold_hours: float
    probe_rollouts: int = 0

    @property
    def wall_seconds_per_rollout(self) -> float:
        return self.seconds_per_rollout / max(self.threads, 1)

    @property
    def projected_wall_seconds(self) -> float:
        return self.projected_rollouts * self.wall_seconds_per_rollout

    @property
    def projected_hours(self) -> float:
        return self.projected_wall_seconds / 3600.0

    @property
    def projected_cost_usd(self) -> float | None:
        if self.cost_per_rollout_usd is None:
            return None
        return self.projected_rollouts * self.cost_per_rollout_usd

    @property
    def requires_approval(self) -> bool:
        return self.projected_hours > self.threshold_hours

    def to_dict(self) -> dict:
        return {
            "projected_rollouts": self.projected_rollouts,
            "budget": self.budget,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "probe_rollouts": self.probe_rollouts,
            "seconds_per_rollout": round(self.seconds_per_rollout, 4),
            "threads": self.threads,
            "wall_seconds_per_rollout": round(self.wall_seconds_per_rollout, 4),
            "projected_wall_seconds": round(self.projected_wall_seconds, 1),
            "projected_hours": round(self.projected_hours, 3),
            "cost_per_rollout_usd": self.cost_per_rollout_usd,
            "projected_cost_usd": (round(self.projected_cost_usd, 2)
                                   if self.projected_cost_usd is not None else None),
            "threshold_hours": self.threshold_hours,
            "requires_approval": self.requires_approval,
        }

    def format(self) -> str:
        cost_line = (f"  projected cost:      ${self.projected_cost_usd:.2f} "
                     f"(@ ${self.cost_per_rollout_usd}/rollout from profile)"
                     if self.projected_cost_usd is not None else
                     "  projected cost:      UNKNOWN — set compile.cost_per_rollout_usd "
                     "in the profile to price this; rollouts + hours above are the receipt")
        lines = [
            "COSTING GATE — projection before any compile call",
            "-" * 56,
            f"  budget:              {self.budget}",
            f"  train/val:           {self.train_size}/{self.val_size}",
            f"  projected rollouts:  {self.projected_rollouts} "
            f"(GEPA metric-call budget + probe {self.probe_rollouts} + final val eval)",
            f"  measured pace:       {self.seconds_per_rollout:.2f}s/rollout "
            f"(live probe, serial)",
            f"  threads:             {self.threads} "
            f"(assumes linear scaling — projection is a floor)",
            f"  projected wall time: {self.projected_hours:.2f}h",
            cost_line,
            f"  approval threshold:  {self.threshold_hours:.2f}h"
            + ("  -> EXCEEDED: --approve-cost required" if self.requires_approval
               else "  -> under threshold"),
            "-" * 56,
        ]
        return "\n".join(lines)


class PaceExceeded(RuntimeError):
    """Raised by PaceMonitor when live pace blows the approved projection."""

    def __init__(self, record: dict):
        super().__init__(record.get("reason", "pace exceeded"))
        self.record = record


@dataclass
class PaceMonitor:
    """Aborts a run whose measured pace exceeds the approved projection.

    Thread-safe (GEPA calls the metric from worker threads). Pace is
    wall-clock elapsed / completed rollouts, compared against the approved
    wall pace (probe sec/rollout divided by threads). The first
    `min_calls` rollouts are exempt (startup noise).
    """
    projected_rollouts: int
    approved_wall_seconds_per_rollout: float
    kill_factor: float = 1.5
    min_calls: int = 5
    clock: object = time.monotonic
    _start: float | None = None
    _calls: int = 0
    killed: bool = False
    kill_record: dict | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        self._start = self.clock()

    @property
    def calls(self) -> int:
        return self._calls

    def record_call(self) -> None:
        """Count one completed rollout; raise PaceExceeded on pace blowout."""
        with self._lock:
            if self._start is None:
                self._start = self.clock()
            self._calls += 1
            if self.killed:
                raise PaceExceeded(self.kill_record)
            elapsed = self.clock() - self._start
            pace = elapsed / self._calls
            limit = self.approved_wall_seconds_per_rollout * self.kill_factor
            if self._calls >= self.min_calls and pace > limit:
                self.killed = True
                self.kill_record = self._build_kill_record(elapsed, pace)
                raise PaceExceeded(self.kill_record)

    def _build_kill_record(self, elapsed: float, pace: float) -> dict:
        remaining = max(self.projected_rollouts - self._calls, 0)
        return {
            "kill": True,
            "reason": (f"measured pace {pace:.2f}s/rollout exceeds approved "
                       f"{self.approved_wall_seconds_per_rollout:.2f}s/rollout "
                       f"x kill factor {self.kill_factor}"),
            "percent_done": round(100.0 * self._calls / max(self.projected_rollouts, 1), 1),
            "rollouts_done": self._calls,
            "projected_rollouts": self.projected_rollouts,
            "elapsed_seconds": round(elapsed, 1),
            "measured_seconds_per_rollout": round(pace, 3),
            "approved_wall_seconds_per_rollout": round(
                self.approved_wall_seconds_per_rollout, 3),
            "kill_factor": self.kill_factor,
            "projected_remaining_seconds": round(remaining * pace, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


def monitored_metric(metric, monitor: PaceMonitor):
    """Wrap a GEPA/Evaluate metric so every call ticks the pace monitor."""
    def wrapped(example, pred, trace=None, pred_name=None, pred_trace=None):
        monitor.record_call()
        return metric(example, pred, trace, pred_name, pred_trace)
    wrapped.__name__ = getattr(metric, "__name__", "metric") + "_paced"
    return wrapped


class PaceStopper:
    """gepa stop-callback: stops the optimizer loop once the monitor killed.

    Belt-and-braces for executors that swallow the PaceExceeded raised from
    inside the metric: the optimizer's own stop-condition hook sees
    monitor.killed and halts at the next iteration boundary.
    """

    def __init__(self, monitor: PaceMonitor):
        self.monitor = monitor

    def __call__(self, gepa_state=None) -> bool:
        return self.monitor.killed
