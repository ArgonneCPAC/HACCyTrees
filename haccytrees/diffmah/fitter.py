"""Batched diffmah fits of core mass-assembly histories.

Reproduces ``diffmah.fitting_helpers.diffmah_fitter`` (target selection, loss, bounded
parameterization, scipy L-BFGS-B termination rule) for many cores at once: the fits run
inside one jitted, vmapped L-BFGS (optax) so that a GPU, or a single CPU core, handles
thousands of cores per call. Results are statistically equivalent to the per-core scipy
fitter (identical mass-history curves on the fitted points; parameters match where the
loss valley is not degenerate), see docs/diffmah.md in the lc-coreforest project.

JAX, optax and diffmah are imported lazily; this module can be imported without them.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

# columns of diffmah's own fitter output (subvol_N_diffmah_fits.hdf5), in that order
DIFFMAH_FIELDS = (
    "logm0",
    "logtc",
    "early_index",
    "late_index",
    "t_peak",
    "loss",
    "n_points_per_fit",
    "fit_algo",
)
# additional columns written by this fitter
EXTRA_FIELDS = ("fit_flag", "n_iterations")
NOFIT = -99.0

# fit_flag bits
FLAG_SKIPPED = 1  # too few usable points (fit_algo = -1, parameters = NOFIT)
FLAG_HIT_CAP = 2  # optimizer stopped by the iteration cap, not by the tolerance
FLAG_AT_BOUND = (
    4  # a shape parameter within bound_tol of its bound (or early pinned to late)
)
FLAG_HIGH_LOSS = 8  # loss above high_loss

MAH_BOUNDS = {  # diffmah 0.7.3 MAH_PBDICT
    "logm0": (0.0, 17.0),
    "logtc": (-1.0, 1.0),
    "early_index": (0.1, 10.0),
    "late_index": (0.1, 5.0),
    "t_peak": (0.05, 20.0),
}


@dataclasses.dataclass(frozen=True)
class FitConfig:
    """Settings of the batched fitter.

    dtype: "float64" (recommended: makes the loss identical across CPU/CUDA/ROCm/SYCL
        backends) or "float32" (about 2x faster on GPUs whose FP64 is slow).
    maxiter: iteration cap per batch. 300 reproduces scipy's own behaviour on the ~2% of
        cores that creep toward a parameter bound; 100 costs half and only changes those.
    ftol, pgtol: scipy L-BFGS-B defaults (relative loss decrease, max |gradient|).
    batch: cores per optimizer call. GPUs: 65536 or more; a CPU core: 256 (cache).
    lgm_min, dlogm_cut, t_fit_min, npts_min: target selection, diffmah defaults.
    high_loss, bound_tol: thresholds for the FLAG_HIGH_LOSS / FLAG_AT_BOUND bits.
    """

    dtype: str = "float64"
    maxiter: int = 300
    ftol: float = 2.220446049250313e-09
    pgtol: float = 1e-5
    batch: int = 65536
    lgm_min: float = -float("inf")
    dlogm_cut: float = 2.5
    t_fit_min: float = 1.0
    npts_min: int = 3
    high_loss: float = 3e-3
    bound_tol: float = 1e-3


def cosmic_time(simulation) -> np.ndarray:
    """Age of the universe (Gyr) at the simulation's cosmotools steps.

    Uses dsps (what diffmah's own HACC loader uses, so results match the existing fits
    numerically) when it is installed, otherwise haccytrees' Cosmology.
    """
    zarr = simulation.step2z(np.array(simulation.cosmotools_steps))
    c = simulation.cosmo
    try:
        from dsps.cosmology import flat_wcdm

        cosmo = flat_wcdm.CosmoParams(c.Omega_m, c.w0, c.wa, c.h)
        return np.asarray(flat_wcdm.age_at_z(zarr, *cosmo), dtype=np.float64)
    except ImportError:
        a = 1.0 / (1.0 + zarr)
        t0 = c.lookback_time(1e-6)
        return np.asarray(t0 - c.lookback_time(a), dtype=np.float64)


def mah_from_mass(mass: np.ndarray, h: float) -> np.ndarray:
    """Mass history in Msun from a (ncores, nsteps) core mass matrix in Msun/h: cumulative
    peak along the time axis, then divided by h (as diffmah's load_hacc_mahs)."""
    return np.maximum.accumulate(mass, axis=1) / h


@dataclasses.dataclass
class Targets:
    """Vectorized equivalent of diffmah's get_loss_data for many cores."""

    log_mah: np.ndarray  # (n, nt) log10 M, clipped
    weight: np.ndarray  # (n, nt) 1 where the point enters the fit
    npts: np.ndarray  # (n,) non-trivial points (log_mah < logm0) in the fit
    skip: np.ndarray  # (n,) bool
    t_peak: np.ndarray  # (n,)
    u_init: np.ndarray  # (n, 4) unbounded start of (logm0, logtc, early, late)
    u_t_peak: np.ndarray  # (n,) unbounded t_peak (fixed during the fit)


def prepare_targets(tarr: np.ndarray, mahs: np.ndarray, cfg: FitConfig) -> Targets:
    from diffmah import diffmah_kernels as dk
    from diffmah.fitting_helpers.diffmah_fitter_helpers import EPSILON
    import jax.numpy as jnp
    from jax import vmap

    n, nt = mahs.shape
    assert tarr.shape == (nt,)
    clip = EPSILON + 10.0**cfg.lgm_min
    log_mah = np.log10(np.where(mahs <= 10.0**cfg.lgm_min, clip, mahs)).astype(
        np.float64
    )
    logm0 = log_mah[:, -1]
    msk = (log_mah > (logm0[:, None] - cfg.dlogm_cut)) & (log_mah > cfg.lgm_min)
    msk &= tarr[None, :] >= cfg.t_fit_min
    npts = np.sum(msk & (log_mah < logm0[:, None]), axis=1)
    skip = npts < cfg.npts_min
    indx_t_peak = np.argmax(log_mah == logm0[:, None], axis=1)
    t_peak = tarr[indx_t_peak]
    p_init = np.tile(np.array(dk.DEFAULT_MAH_PARAMS, dtype="f4"), (n, 1)).astype(
        np.float64
    )
    p_init[:, 0] = logm0
    p_init[:, 4] = t_peak
    to_u = vmap(lambda p: jnp.array(dk.get_unbounded_mah_params(dk.DiffmahParams(*p))))
    u = np.asarray(to_u(jnp.array(p_init)), dtype=np.float64)
    return Targets(
        log_mah, msk.astype(np.float64), npts, skip, t_peak, u[:, :4], u[:, 4]
    )


_KERNELS: dict[tuple, Any] = {}


def _kernel(cfg: FitConfig):
    """Build (once per config) the jitted, vmapped L-BFGS over a batch of cfg.batch cores."""
    key = (cfg.dtype, cfg.maxiter, cfg.ftol, cfg.pgtol, cfg.batch)
    if key in _KERNELS:
        return _KERNELS[key]
    import jax

    if cfg.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from diffmah import diffmah_kernels as dk
    from jax import jit, lax, vmap

    def loss_fn(u_varied, t, target, w, u_tp, lgt0):
        u_params = dk.DiffmahUParams(*u_varied, u_tp)
        pred = dk._log_mah_kern_u_params(u_params, t, lgt0)
        d = pred - target
        return jnp.sum(w * d * d) / jnp.sum(w)

    opt = optax.lbfgs()
    vg = optax.value_and_grad_from_state(loss_fn)
    ftol, pgtol, maxiter = cfg.ftol, cfg.pgtol, cfg.maxiter

    def fit_one(u0, t, target, w, u_tp, lgt0):
        def cond(c):
            return (~c[5]) & (c[6] < maxiter)

        def body(c):
            params, state, best_p, best_v, prev_v, done, it = c
            value, grad = vg(params, t, target, w, u_tp, lgt0, state=state)
            updates, state = opt.update(
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=loss_fn,
                t=t,
                target=target,
                w=w,
                u_tp=u_tp,
                lgt0=lgt0,
            )
            new_params = optax.apply_updates(params, updates)
            better = value < best_v
            best_p = jnp.where(better, params, best_p)
            best_v = jnp.where(better, value, best_v)
            rel = (prev_v - value) / jnp.maximum(
                jnp.maximum(jnp.abs(prev_v), jnp.abs(value)), 1.0
            )
            done = (rel <= ftol) | (jnp.max(jnp.abs(grad)) <= pgtol)
            return (new_params, state, best_p, best_v, value, done, it + 1)

        state = opt.init(u0)
        c0 = (u0, state, u0, jnp.inf, jnp.inf, jnp.bool_(False), jnp.int32(0))
        params, state, best_p, best_v, prev_v, done, it = lax.while_loop(cond, body, c0)
        v_last = loss_fn(params, t, target, w, u_tp, lgt0)
        better = v_last < best_v
        best_p = jnp.where(better, params, best_p)
        best_v = jnp.where(better, v_last, best_v)
        return best_p, best_v, it

    fit_batch = jit(vmap(fit_one, in_axes=(0, None, 0, 0, 0, None)))
    to_bounded = jit(
        vmap(
            lambda u, utp: jnp.array(
                dk.get_bounded_mah_params(dk.DiffmahUParams(*u, utp))
            )
        )
    )
    _KERNELS[key] = (fit_batch, to_bounded)
    return _KERNELS[key]


def fit_targets(tarr: np.ndarray, tg: Targets, cfg: FitConfig) -> dict[str, np.ndarray]:
    """Run the batched fitter; returns the DIFFMAH_FIELDS + EXTRA_FIELDS arrays (n,)."""
    import jax
    import jax.numpy as jnp

    fit_batch, to_bounded = _kernel(cfg)
    dtype = np.float64 if cfg.dtype == "float64" else np.float32
    n = len(tg.skip)
    B = cfg.batch
    t_j = jnp.array(tarr, dtype=dtype)
    lgt0 = dtype(np.log10(tarr[-1]))
    u_best = np.empty((n, 4))
    loss = np.empty(n)
    niter = np.empty(n, dtype=np.int32)
    for s in range(0, n, B):
        e = min(n, s + B)
        sl = slice(s, e)

        # pad the last batch to B so the kernel is compiled for one shape only
        def pad(a):
            if e - s == B:
                return a
            reps = [(0, B - (e - s))] + [(0, 0)] * (a.ndim - 1)
            return np.pad(a, reps, mode="edge")

        bp, bv, it = fit_batch(
            jnp.array(pad(tg.u_init[sl]), dtype=dtype),
            t_j,
            jnp.array(pad(tg.log_mah[sl] * tg.weight[sl]), dtype=dtype),
            jnp.array(pad(tg.weight[sl]), dtype=dtype),
            jnp.array(pad(tg.u_t_peak[sl]), dtype=dtype),
            lgt0,
        )
        jax.block_until_ready(bv)
        m = e - s
        u_best[sl] = np.asarray(bp)[:m]
        loss[sl] = np.asarray(bv)[:m]
        niter[sl] = np.asarray(it)[:m]
    p = np.asarray(
        to_bounded(jnp.array(u_best, dtype=dtype), jnp.array(tg.u_t_peak, dtype=dtype)),
        dtype=np.float64,
    )

    skip = tg.skip
    out: dict[str, np.ndarray] = {}
    for i, k in enumerate(("logm0", "logtc", "early_index", "late_index", "t_peak")):
        out[k] = np.where(skip, NOFIT, p[:, i])
    out["loss"] = np.where(skip, NOFIT, loss)
    # diffmah records the point count for skipped cores too (it is why they were skipped)
    out["n_points_per_fit"] = tg.npts.astype(np.int64)
    out["fit_algo"] = np.where(skip, -1, 0).astype(np.int64)
    tol = cfg.bound_tol
    at_bound = (
        (p[:, 2] >= MAH_BOUNDS["early_index"][1] - tol)
        | (p[:, 2] <= p[:, 3] + tol)
        | (np.abs(p[:, 1]) >= MAH_BOUNDS["logtc"][1] - tol)
        | (p[:, 3] >= MAH_BOUNDS["late_index"][1] - tol)
    )
    flag = np.zeros(n, dtype=np.int8)
    flag[skip] |= FLAG_SKIPPED
    flag[~skip & (niter >= cfg.maxiter)] |= FLAG_HIT_CAP
    flag[~skip & at_bound] |= FLAG_AT_BOUND
    flag[~skip & (loss > cfg.high_loss)] |= FLAG_HIGH_LOSS
    out["fit_flag"] = flag
    out["n_iterations"] = np.where(skip, 0, niter).astype(np.int16)
    return out


def fit_mahs(
    tarr: np.ndarray, mahs: np.ndarray, cfg: FitConfig = FitConfig()
) -> dict[str, np.ndarray]:
    """Fit mass histories (ncores, nsteps) in Msun on the current JAX device."""
    if len(mahs) == 0:
        return empty_results()
    return fit_targets(tarr, prepare_targets(tarr, mahs, cfg), cfg)


def result_dtypes() -> dict[str, np.dtype]:
    d = {
        k: np.dtype(np.float64)
        for k in ("logm0", "logtc", "early_index", "late_index", "t_peak", "loss")
    }
    d.update(
        n_points_per_fit=np.dtype(np.int64),
        fit_algo=np.dtype(np.int64),
        fit_flag=np.dtype(np.int8),
        n_iterations=np.dtype(np.int16),
    )
    return d


def empty_results() -> dict[str, np.ndarray]:
    return {k: np.empty(0, dtype=dt) for k, dt in result_dtypes().items()}


def cpu_environment(threads: int = 1) -> dict[str, str]:
    """Environment for CPU workers: one XLA thread (the batched fitter does not
    parallelize across threads) and the CPU backend."""
    return {
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": str(threads),
        "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}",
    }


def default_cpu_config(**overrides) -> FitConfig:
    """CPU defaults: small batches (cache), float64, cap 100 (see FitConfig.maxiter)."""
    kw: dict[str, Any] = dict(batch=256, maxiter=100)
    kw.update(overrides)
    return FitConfig(**kw)


def set_env(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ.setdefault(k, v)
