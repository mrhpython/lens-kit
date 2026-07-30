"""lens_kit — trainable 10-lens validation gate for AI-generated text.

Provider-agnostic: bring any litellm-compatible endpoint via a profile YAML.

Quick start:
    from lens_kit import LensGate, Profile, lm_context

    profile = Profile.load("my-profile.yaml")
    gate = LensGate(profile=profile)
    with lm_context(profile):                  # scoped LM, fail-closed, auto-restores
        result = gate(text="AI output to validate", context="who reads this")
    result.passed, result.violations, result.per_lens

    # Or process-global (returns (lm, previous_lm) for manual restore):
    from lens_kit import configure_from_profile
    lm, previous_lm = configure_from_profile(profile)
"""
from .checkpoint import load_checkpoint, save_checkpoint
from .catches import (
    Catch, CatchError, append_catch, build_catch, filter_records, is_pass_like,
    load_catches, load_seed, pattern_counts, promotable, promote_line,
    render_relevant_block, render_stats,
)
from .config import (
    CalibrationConfig, CompileConfig, ConfigError, ConsistencyConfig,
    LLMConfig, Profile, builtin_profile_path, configure_from_profile,
    lm_context, make_lm,
)
from .consistency import (
    DEFAULT_MARKERS, CheckResult, check_leaks, check_markers, check_numbers,
    extract_numbers, normalize_number, run_consistency, scan_leaks,
)
from .filters import DeterministicFilters, LensVocabulary
from .gate import (
    CANONICAL_LENSES, LensGate, LensResult, LensViolation, detect_domain,
    validate_lens_labels,
)
from .label_audit import backup_dataset, find_suspicious, impact_math, run_label_audit
from .mutation import (
    CAUGHT_VERDICTS, MUTATION_TYPES, generate_mutants, mutate_text,
    run_mutation_control,
)
from .sidecar import (
    DEFAULT_NOT_A_CLAIM_OF, build_sidecar, generate_sidecar, write_sidecar,
)

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_LENSES",
    "CAUGHT_VERDICTS",
    "CalibrationConfig",
    "Catch",
    "CatchError",
    "CheckResult",
    "CompileConfig",
    "ConfigError",
    "ConsistencyConfig",
    "DEFAULT_MARKERS",
    "DEFAULT_NOT_A_CLAIM_OF",
    "DeterministicFilters",
    "append_catch",
    "build_catch",
    "filter_records",
    "is_pass_like",
    "load_catches",
    "load_seed",
    "pattern_counts",
    "promotable",
    "promote_line",
    "render_relevant_block",
    "render_stats",
    "LLMConfig",
    "LensGate",
    "LensResult",
    "LensViolation",
    "LensVocabulary",
    "MUTATION_TYPES",
    "Profile",
    "backup_dataset",
    "build_sidecar",
    "builtin_profile_path",
    "check_leaks",
    "check_markers",
    "check_numbers",
    "configure_from_profile",
    "detect_domain",
    "extract_numbers",
    "find_suspicious",
    "generate_mutants",
    "generate_sidecar",
    "impact_math",
    "lm_context",
    "load_checkpoint",
    "make_lm",
    "mutate_text",
    "normalize_number",
    "run_consistency",
    "run_label_audit",
    "run_mutation_control",
    "save_checkpoint",
    "scan_leaks",
    "validate_lens_labels",
    "write_sidecar",
    "__version__",
]
