"""Run the batched diffmah fitter on many CPU cores: one single-threaded worker process
per core, each fitting its slice of the cores. XLA's own thread pool does not speed up
the batched fit, but independent processes scale linearly (measured on Crux: 128 ranks x
4.5-7 ms/core with batch 256, i.e. ~20-30k cores/s per node).

Workers are spawned (not forked) so that the fit only ever runs JAX inside them; the
calling process (e.g. an MPI rank of the lightcone script) does not import JAX for the fit
(cosmic_time imports it in the caller when dsps is installed), and the workers do not
re-import the caller's __main__ module (see _no_main_reimport).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .fitter import FitConfig, cpu_environment, empty_results, fit_mahs


def _init_worker(env: dict[str, str], counter, nworkers: int) -> None:
    """runs in every spawned worker before JAX is imported: single-threaded CPU backend and,
    when the caller's affinity mask has at least one CPU per worker, pin this worker to its
    own CPU (which also keeps XLA's thread pool at one thread). With fewer CPUs than workers
    the OS schedules the workers within the mask. The pinning assumes that the caller owns
    its mask: callers that share one mask (unbound MPI ranks, several processes started by
    hand on one node) pin their workers to the same CPUs and should be bound to disjoint
    CPU sets by the launcher."""
    os.environ.update(env)
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= nworkers:
            os.sched_setaffinity(0, {cpus[idx % len(cpus)]})
    except (AttributeError, OSError):
        pass


@contextlib.contextmanager
def _no_main_reimport():
    """Spawned children normally re-import the parent's __main__ module (a console script
    that may import mpi4py, initialize MPI, ...). The workers only need this module, so hide
    __main__'s file/spec while the pool starts its processes (multiprocessing.spawn checks
    __main__.__spec__.name and __main__.__file__ to decide what to re-import)."""
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    had_file = hasattr(main, "__file__")
    saved_file = getattr(main, "__file__", None)
    saved_spec = getattr(main, "__spec__", None)
    try:
        if had_file:
            del main.__file__
        main.__spec__ = None
        yield
    finally:
        if had_file:
            main.__file__ = saved_file
        main.__spec__ = saved_spec


def _worker(args):
    tarr, mahs, cfg = args
    return fit_mahs(tarr, mahs, cfg)


def fit_mahs_parallel(
    tarr: np.ndarray,
    mahs: np.ndarray,
    cfg: FitConfig,
    workers: int,
    *,
    chunk: int = 4096,
) -> dict[str, np.ndarray]:
    """Fit (ncores, nsteps) histories with `workers` CPU processes; returns the same dict
    as fit_mahs, in input order. At most `chunk` cores per task keeps the pickled payload
    small; tasks are made smaller when needed so that every worker gets at least one."""
    n = len(mahs)
    if n == 0:
        return empty_results()
    # always in spawned workers (also for workers == 1), so the CPU backend and the
    # single-threaded XLA settings hold regardless of the caller's JAX state
    workers = max(1, min(workers, n))
    # the pool starts one process per pending task, so there have to be at least `workers`
    # tasks for all of them to run (with the default chunk, n < 4096 * workers would
    # otherwise leave workers unused)
    chunk = max(1, min(chunk, -(-n // workers)))
    # the workers inherit the caller's CPU affinity mask (e.g. the MPI rank's 16 cores) and
    # are pinned to one CPU each within it when it has enough CPUs (_init_worker); each
    # worker runs single-threaded XLA (cpu_environment)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    tasks = [(tarr, mahs[s : s + chunk], cfg) for s in range(0, n, chunk)]
    results = []
    with (
        _no_main_reimport(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(cpu_environment(), counter, workers),
        ) as ex,
    ):
        for r in ex.map(_worker, tasks, chunksize=1):
            results.append(r)
    return {k: np.concatenate([r[k] for r in results]) for k in results[0]}
