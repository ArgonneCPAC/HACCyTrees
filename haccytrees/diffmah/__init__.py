"""Batched diffmah fits of core mass histories (see fitter.py)."""

from .fitter import (
    DIFFMAH_FIELDS,
    EXTRA_FIELDS,
    FLAG_AT_BOUND,
    FLAG_HIGH_LOSS,
    FLAG_HIT_CAP,
    FLAG_SKIPPED,
    NOFIT,
    FitConfig,
    cosmic_time,
    cpu_environment,
    default_cpu_config,
    empty_results,
    fit_mahs,
    mah_from_mass,
    result_dtypes,
    set_env,
)
from .parallel import fit_mahs_parallel

__all__ = (
    "DIFFMAH_FIELDS",
    "EXTRA_FIELDS",
    "FLAG_AT_BOUND",
    "FLAG_HIGH_LOSS",
    "FLAG_HIT_CAP",
    "FLAG_SKIPPED",
    "NOFIT",
    "FitConfig",
    "cosmic_time",
    "cpu_environment",
    "default_cpu_config",
    "empty_results",
    "fit_mahs",
    "fit_mahs_parallel",
    "mah_from_mass",
    "result_dtypes",
    "set_env",
)
