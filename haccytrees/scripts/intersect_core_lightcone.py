"""Attach coreforest quantities and diffmah fits to an existing core lightcone.

The simulation wrote a core lightcone in situ: one GenericIO file set per PM step with the
interpolated position, velocity and scale factor of every core that crossed the lightcone in
that step, its in-situ lightcone halo (``fof_halo_tag``, ``fof_halo_replication``) and its
``core_tag``. This script joins every lightcone core by ``core_tag`` to its coreforest row at
the snapshot nearest to the step, adds the forest quantities of the Last Journey lightcone
catalog (host cores, infall properties, peak mass, diffmah fits of the core and its hosts) and
writes the Last Journey layout: ``<dir>/step_<s>/<name>-<s>.<writer>.hdf5`` with ``data/`` and
an ``index/`` group per lightcone halo.

How the work is distributed:

1. The coreforest is read once. Every forest file is split into root-aligned chunks that are
   dealt round-robin to the ranks; each rank keeps dense (cores x snapshots) matrices of the
   fields it needs, and fits diffmah per chunk (on the rank's GPU tile, or CPU workers).
2. A directory maps ``core_tag`` to (owner rank, matrix row): each rank sends the tags of its
   cores to the directory rank ``hash(core_tag) % nranks``.
3. Per lightcone step: the lightcone cores are read (GenericIO, distributed over the ranks),
   sent to the directory rank of their ``core_tag``, looked up, and sent to the rank that owns
   their forest row. That rank adds the forest quantities at the snapshot nearest to the step
   (no forest data moves).
4. The cores are sent to the rank ``hash(group) % nranks`` of their lightcone halo
   ``(octant, fof_halo_tag)``, where the group's central (the most massive central core, or the
   most massive core if the group has no central) gives ``lc_halo_*``. Then the cores are
   distributed on the sphere by the group's angular position (S2 partition over all ranks), and
   collected by ``nwriters`` writer ranks, each writing one file per step.

Column conventions follow the Last Journey catalog (core-lc-9); differences: positions,
velocities, ``theta phi scale_factor`` are the core's own lightcone values; ``lc_halo_*`` is
the position of the group's central core; ``lc_fof_halo_tag`` is the in-situ lightcone halo
tag; ``lc_fof_halo_flag`` is true when the group has no central core; new columns
``radius vel_disp overlap replication`` come from the core lightcone. ``index/unique_id`` packs
``octant << 44 | fof_halo_tag`` (the replication code is a bit pattern per axis, mapped to an
octant index 0..7).
"""

from __future__ import annotations

import dataclasses
import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path

import click
import numpy as np
from mpi4py import MPI

from haccytrees.coretrees import corematrix_reader
from haccytrees.coretrees.assemble import CoretreesAssemblyConfig
from haccytrees.mergertrees.fragments import split_fragment_tag
from haccytrees.utils.hdf5_utils import COMPRESSION_CHOICES, create_dataset
from haccytrees.utils.mpi_error_handler import init_mpi_error_handler

from .intersect_cores_with_lightcone import (
    DEFAULT_DIFFMAH_FIELDS,
    DEFAULT_DIFFMAH_HOST_FIELDS,
    DIFFMAH_MISSING,
    PEAK_MASS_SOURCE,
    cast_diffmah_results,
    diffmah_storage_dtype,
    get_infall_time_indices,
    step_output_base,
    top_host_core_fields,
)

try:  # compressed forests (Blosc2) need the filter plugin to be registered
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

init_mpi_error_handler()

LC_FIELDS = [
    "x", "y", "z", "vx", "vy", "vz", "a", "radius", "vel_disp",
    "fof_halo_tag", "fof_halo_replication", "core_tag", "overlap",
]
# replication code of the core lightcone: one bit per axis (seen: 0, 1 << 10, 1 << 20)
REPLICATION_BITS = (1 << 20, 1 << 10, 1 << 0)
REPLICATION_MASK = sum(REPLICATION_BITS)
TAG_BITS = 44

# forest fields read into the matrices (corematrix_reader adds core_tag host_core snapnum
# central merged)
FOREST_FIELDS = [
    "core_tag",
    "fof_halo_tag",
    "central",
    "infall_fof_halo_center_x",
    "infall_fof_halo_center_y",
    "infall_fof_halo_center_z",
]
# matrices needed for all snapshots (host rows and infall indices look at the history)
FULL_HISTORY = ["core_tag", "central", "host_row", "top_host_row", "secondary_top_host_row"]

OUTPUT_FIELDS = [
    # Last Journey core_fields
    "x", "y", "z", "vx", "vy", "vz", "fof_halo_tag", "central", "core_tag",
    "infall_fof_halo_center_x", "infall_fof_halo_center_y", "infall_fof_halo_center_z",
    "coreforest_file_idx", "coreforest_row_idx",
    "theta", "phi", "scale_factor",
    "lc_halo_x", "lc_halo_y", "lc_halo_z", "lc_halo_theta", "lc_halo_phi",
    "lc_fof_halo_tag", "lc_fof_halo_flag",
    "top_host_tag", "secondary_top_host_tag", "top_host_idx", "secondary_top_host_idx",
    "host_core", "merged", "snapnum",
]
OUTPUT_TAIL = ["indx_t_ult_inf", "indx_t_pen_inf", "mp_obs", "mp0",
               "radius", "vel_disp", "overlap", "replication"]


def log(comm, msg):
    comm.Barrier()
    if comm.rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """deterministic 64-bit hash (used to spread keys evenly over ranks)"""
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hash_rank(keys: np.ndarray, nranks: int) -> np.ndarray:
    return (splitmix64(keys) % np.uint64(nranks)).astype(np.int64)


# exchange_by_dest settings (set by the command line): rounds in flight, and the seconds after
# which a rank that still waits prints the partners it waits for
EXCHANGE = {"window": 8, "report_s": 300.0}
_exchange_state = {"seq": 0, "hosts": None}


def gather_rank_hosts(comm) -> list:
    """node name of every rank, for the exchange_by_dest report; call it once on all ranks"""
    if _exchange_state["hosts"] is None:
        _exchange_state["hosts"] = comm.allgather(MPI.Get_processor_name().split(".")[0])
    return _exchange_state["hosts"]


# GPU diffmah fits with a deadline. In job 9087866 one of 6144 ranks never returned from a JAX
# call on its tile (prepare_targets, waiting for the device), and all other ranks waited for it
# in the collectives at the end of the forest read until the read watchdog ended the run. A hung
# Level Zero call cannot be cancelled, so the fit runs in a daemon thread; when it has not
# returned after DIFFMAH_GPU["timeout_s"], this rank fits that chunk, and all later ones, in CPU
# worker processes (fit_mahs_parallel, same FitConfig with batch 256). The fits of a core depend
# somewhat on the cores that share its batch, so CPU and GPU fits are equivalent, not equal: in
# job 9088287 logm0 agreed within 0.01 dex for 96% of the CPU-fitted cores, as in the earlier
# GPU against CPU comparison.
DIFFMAH_GPU = {"timeout_s": 600.0, "hung": False, "cpu_chunks": 0}


def fit_with_deadline(comm, fit_gpu, fit_cpu, label: str):
    """fit_gpu() in a thread, at most DIFFMAH_GPU["timeout_s"] seconds; fit_cpu() otherwise"""
    if not DIFFMAH_GPU["hung"]:
        box = {}
        test_hang = os.environ.get("INTERSECT_TEST_GPU_HANG") == str(comm.rank)

        def run():
            try:
                if test_hang:  # test of the fallback: this rank's GPU fit never returns
                    time.sleep(1e9)
                box["res"] = fit_gpu()
            except BaseException as e:  # noqa: BLE001 (re-raised in the calling thread)
                box["err"] = e

        th = threading.Thread(target=run, name="diffmah-gpu", daemon=True)
        th.start()
        timeout = DIFFMAH_GPU["timeout_s"]
        th.join(timeout if timeout > 0 else None)
        if not th.is_alive():
            if "err" in box:
                raise box["err"]
            return box["res"]
        DIFFMAH_GPU["hung"] = True
        hosts = _exchange_state["hosts"]
        host = hosts[comm.rank] if hosts else "?"
        print(f"rank {comm.rank} ({host}): the GPU diffmah fit of {label} has not returned after "
              f"{timeout:.0f} s; this rank fits on the CPU from now on", file=sys.stderr, flush=True)
    DIFFMAH_GPU["cpu_chunks"] += 1
    return fit_cpu()


def exchange_by_dest(comm, data: dict, dest: np.ndarray) -> dict:
    """send row i of every array in data to rank dest[i]

    The fields of a row are packed into one record, and the records travel in a shifted
    pairwise schedule of point-to-point messages: in round i a rank receives from rank - i and
    sends to rank + i, with EXCHANGE["window"] rounds in flight, and only pairs with rows
    exchange a message. This replaced one MPI_Alltoallv per field, which hung in exchange 2 of
    the first lightcone step at 6144 ranks (job 9086927). A rank that still waits after
    EXCHANGE["report_s"] seconds prints the partner ranks and nodes it waits for.
    Received rows are ordered by source rank, each source's rows in their original order.
    """
    n = len(dest)
    keys = sorted(data)
    for k in keys:
        assert len(data[k]) == n, (k, len(data[k]), n)
    size, rank = comm.size, comm.rank
    dest = np.asarray(dest, dtype=np.int64)
    order = np.argsort(dest, kind="stable")
    scount = np.bincount(dest, minlength=size).astype(np.int64)
    assert len(scount) == size, "destination rank out of range"
    rcount = np.empty_like(scount)
    comm.Alltoall(scount, rcount)
    rec = np.dtype([(k, data[k].dtype, data[k].shape[1:]) for k in keys])
    isz = rec.itemsize
    assert max(scount.max(initial=0), rcount.max(initial=0)) * isz < 2**31, (
        "message too large; use more ranks"
    )
    send = np.empty(n, dtype=rec)
    for k in keys:
        send[k] = data[k][order]
    recv = np.empty(int(rcount.sum()), dtype=rec)
    sdispl = np.concatenate([[0], np.cumsum(scount)[:-1]]).astype(np.int64)
    rdispl = np.concatenate([[0], np.cumsum(rcount)[:-1]]).astype(np.int64)
    recv[rdispl[rank]:rdispl[rank] + rcount[rank]] = send[sdispl[rank]:sdispl[rank] + scount[rank]]
    sb, rb = send.view(np.uint8), recv.view(np.uint8)
    seq = _exchange_state["seq"]
    _exchange_state["seq"] += 1
    tag = 1000 + seq % 30000
    window = max(1, int(EXCHANGE["window"]))
    t0 = time.time()
    reported = False
    for start in range(1, size, window):
        reqs, what = [], []
        rounds = range(start, min(start + window, size))
        for i in rounds:
            src = (rank - i) % size
            if rcount[src]:
                a, b = rdispl[src] * isz, (rdispl[src] + rcount[src]) * isz
                reqs.append(comm.Irecv([rb[a:b], MPI.BYTE], source=src, tag=tag))
                what.append(("receive from", src, int(rcount[src])))
        for i in rounds:
            dst = (rank + i) % size
            if scount[dst]:
                a, b = sdispl[dst] * isz, (sdispl[dst] + scount[dst]) * isz
                reqs.append(comm.Isend([sb[a:b], MPI.BYTE], dest=dst, tag=tag))
                what.append(("send to", dst, int(scount[dst])))
        while not MPI.Request.Testall(reqs):
            if not reported and time.time() - t0 > EXCHANGE["report_s"]:
                reported = True
                hosts = _exchange_state["hosts"]  # gathered at the start (collective)
                waiting = [w for r, w in zip(reqs, what) if not r.Test()]
                desc = "; ".join(f"{d} rank {p} ({hosts[p] if hosts else '?'}) {c} rows"
                                 for d, p, c in waiting[:12])
                more = f" and {len(waiting) - 12} more" if len(waiting) > 12 else ""
                me = hosts[rank] if hosts else MPI.Get_processor_name()
                print(f"exchange {seq}: rank {rank} ({me}) still waits after {time.time() - t0:.0f} s "
                      f"in rounds {start}..{rounds[-1]} of {size - 1}: {desc}{more}", flush=True)
    if reported:
        print(f"exchange {seq}: rank {rank} finished after {time.time() - t0:.0f} s", flush=True)
    return {k: np.ascontiguousarray(recv[k]) for k in keys}


def s2_home_rank(partition, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """rank of the S2 partition cell of each (theta, phi): the rule of mpipartition 1.5.0's
    s2_distribute, whose per-field Alltoallv exchange_by_dest replaces"""
    assert np.all(theta >= 0) and np.all(theta <= np.pi)
    assert np.all(phi >= 0) and np.all(phi < 2 * np.pi)
    if partition.nranks == 1:
        return np.zeros(len(theta), dtype=np.int64)
    if partition.equal_area:
        ring_idx = np.digitize(theta, partition.ring_thetas)
    elif partition.nranks == 2:
        ring_idx = (theta > partition.theta_cap).astype(np.int32)
    else:
        ring_idx = ((theta - partition.theta_cap) // partition.ring_dtheta).astype(np.int32) + 1
        ring_idx = np.clip(ring_idx, 0, len(partition.ring_segments) + 1)
        ring_idx[theta == np.pi] -= 1
    phi_idx = np.zeros_like(ring_idx, dtype=np.int32)
    on_ring = (ring_idx > 0) & (ring_idx <= len(partition.ring_segments))
    phi_idx[on_ring] = (
        phi[on_ring] / (2 * np.pi) * partition.ring_segments[ring_idx[on_ring] - 1]
    ).astype(np.int32)
    ring_start = np.zeros(len(partition.ring_segments) + 2, dtype=np.int32)
    ring_start[1] = 1
    ring_start[2:] = np.cumsum(partition.ring_segments) + 1
    home = (ring_start[ring_idx] + phi_idx).astype(np.int64)
    assert np.all(home >= 0) and np.all(home < partition.nranks)
    return home


def replication_to_octant(code: np.ndarray) -> np.ndarray:
    code = code.astype(np.int64)
    bad = (code & ~np.int64(REPLICATION_MASK)) != 0
    if np.any(bad):
        raise ValueError(f"unexpected replication codes {np.unique(code[bad])[:10]}")
    octant = np.zeros(len(code), dtype=np.int64)
    for i, b in enumerate(REPLICATION_BITS):
        octant |= ((code & b) != 0).astype(np.int64) << (2 - i)
    return octant


def pack_unique_id(octant: np.ndarray, tag: np.ndarray) -> np.ndarray:
    assert np.all(tag >= 0) and np.all(tag < (1 << TAG_BITS))
    assert np.all(octant >= 0) and np.all(octant < 8)
    return (octant.astype(np.uint64) << np.uint64(TAG_BITS)) + tag.astype(np.uint64)


def nearest_snapshots(lc_steps, cosmotools_steps) -> list[int]:
    """forest column of the snapshot nearest in step to each lightcone step (ties: earlier)"""
    st = np.array(cosmotools_steps)
    return [int(np.argmin(np.abs(st - s))) for s in lc_steps]


# --------------------------------------------------------------------------------------
# forest


def read_forest(comm, config, forest_base: Path, *, col0: int, ncorechunks: int,
                diffmah_fields, diffmah_dtype: str, diffmah_device: str,
                diffmah_workers: int, diffmah_maxiter: int, diffmah_batch: int,
                files: list[int]):
    """Read the forest chunks of this rank. Matrices of the fields in FULL_HISTORY keep all
    snapshots; the others only the columns from col0 on (the lightcone's snapshots)."""
    sim = config.simulation
    if diffmah_fields:
        from haccytrees.diffmah import (
            FitConfig, cosmic_time, default_cpu_config, fit_mahs, fit_mahs_parallel,
            mah_from_mass,
        )
        if diffmah_device == "cpu":
            # the fits run in CPU worker processes; cosmic_time initializes JAX in this
            # process when dsps is installed: keep the MPI ranks off the GPUs for that
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        tarr = cosmic_time(sim)
        if diffmah_device == "gpu":
            cfg = FitConfig(dtype="float64", maxiter=diffmah_maxiter, batch=diffmah_batch)
            # the CPU fallback after a hung GPU fit: the same fit, in the batch size that is
            # fastest on CPUs
            cfg_cpu = dataclasses.replace(cfg, batch=256)
        else:
            cfg = default_cpu_config(dtype="float64", maxiter=diffmah_maxiter, batch=diffmah_batch)
        if comm.rank == 0:
            print(f"diffmah fits on the fly ({diffmah_device}): {cfg}", flush=True)
    import h5py

    if comm.rank == 0:
        with h5py.File(f"{forest_base}.{files[0]}.hdf5") as f:
            available = set(f["data"].keys())
    else:
        available = None
    available = comm.bcast(available, root=0)
    eig = [k for k in top_host_core_fields if k in available]
    include = FOREST_FIELDS + eig + [PEAK_MASS_SOURCE]

    parts = []
    tasks = [(f, c) for f in files for c in range(ncorechunks)]
    t0 = time.time()
    tfit = 0.0
    for j in range(comm.rank, len(tasks), comm.size):
        fidx, chunk = tasks[j]
        m = corematrix_reader(
            f"{forest_base}.{fidx}.hdf5", sim, include_fields=list(include),
            calculate_host_rows=True, calculate_secondary_host_row=True,
            nchunks=ncorechunks, chunknum=chunk,
        )
        m.pop("core_state", None)
        n = m["core_tag"].shape[0]
        part = {"coreforest_file_idx": np.full(n, fidx, dtype=np.uint16),
                "coreforest_row_idx": m.pop("absolute_row_idx").astype(np.int64)}
        mass = m.pop(PEAK_MASS_SOURCE)
        if diffmah_fields:
            t1 = time.time()
            mahs = mah_from_mass(mass, sim.cosmo.h)
            if n == 0:
                from haccytrees.diffmah import empty_results
                res = empty_results()
            elif diffmah_device == "gpu":
                res = fit_with_deadline(
                    comm, lambda: fit_mahs(tarr, mahs, cfg),
                    lambda: fit_mahs_parallel(tarr, mahs, cfg_cpu, diffmah_workers),
                    f"forest file {fidx} chunk {chunk} ({n} cores)")
            else:
                res = fit_mahs_parallel(tarr, mahs, cfg, diffmah_workers)
            part.update(cast_diffmah_results(res, diffmah_fields, diffmah_dtype))
            del mahs
            tfit += time.time() - t1
        mpeak = np.maximum.accumulate(mass, axis=1, out=mass)
        part["mp0"] = mpeak[:, -1].copy()
        part["mp_obs"] = np.ascontiguousarray(mpeak[:, col0:])
        del mass, mpeak
        for k in list(m):
            if k in FULL_HISTORY:
                part[k] = m.pop(k)
            else:
                part[k] = np.ascontiguousarray(m.pop(k)[:, col0:])
        parts.append(part)

    # concatenate; host rows are chunk-local and get shifted
    if parts:
        offset = 0
        for p in parts:
            for k in ("host_row", "top_host_row", "secondary_top_host_row"):
                p[k][p[k] >= 0] += offset
            offset += p["core_tag"].shape[0]
        forest = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    else:
        forest = None
    # ranks without chunks need empty arrays of the right shapes and dtypes
    proto = comm.allgather(
        None if forest is None else {k: (v.dtype.str, v.shape[1:]) for k, v in forest.items()}
    )
    proto = next(p for p in proto if p is not None)
    if forest is None:
        forest = {k: np.empty((0,) + tuple(s), dtype=np.dtype(d)) for k, (d, s) in proto.items()}
    del parts
    n_local = forest["core_tag"].shape[0]
    n_tot = comm.allreduce(n_local)
    tfit_max = comm.allreduce(tfit, op=MPI.MAX)
    n_hung = comm.allreduce(int(DIFFMAH_GPU["hung"]))
    n_cpu = comm.allreduce(DIFFMAH_GPU["cpu_chunks"])
    log(comm, f"forest read: {n_tot} cores in {len(files)} files, {time.time() - t0:.0f} s "
              f"(diffmah {tfit_max:.0f} s max per rank; {n_hung} ranks with a hung GPU fit, "
              f"{n_cpu} chunks fitted on the CPU instead)")
    return forest


def build_directory(comm, forest):
    """core_tag -> (owner rank, row), stored on rank hash(core_tag) % nranks, sorted by tag"""
    tag = forest["core_tag"][:, -1].astype(np.int64)  # every forest core exists at the last snapshot
    assert np.all(tag > 0), "forest core without core_tag at the last snapshot"
    d = {"tag": tag,
         "owner": np.full(len(tag), comm.rank, dtype=np.int32),
         "row": np.arange(len(tag), dtype=np.int64)}
    d = exchange_by_dest(comm, d, hash_rank(tag, comm.size))
    s = np.argsort(d["tag"])
    d = {k: v[s] for k, v in d.items()}
    ndup = int(np.sum(np.diff(d["tag"]) == 0))
    ndup = comm.allreduce(ndup)
    if ndup:
        raise RuntimeError(f"{ndup} duplicate core_tag in the forest")
    return d


def forest_at(forest, rows: np.ndarray, col: int, col0: int, diffmah_fields, host_fields):
    """per-core forest quantities of the given rows at snapshot column col"""
    c = col - col0
    out = {}
    out["fof_halo_tag"] = forest["fof_halo_tag"][rows, c]
    out["central"] = forest["central"][rows, col]
    out["core_tag_forest"] = forest["core_tag"][rows, col]
    for k in ("infall_fof_halo_center_x", "infall_fof_halo_center_y", "infall_fof_halo_center_z"):
        out[k] = forest[k][rows, c]
    out["host_core"] = forest["host_core"][rows, c]
    out["merged"] = forest["merged"][rows, c]
    out["snapnum"] = forest["snapnum"][rows, c]
    out["coreforest_file_idx"] = forest["coreforest_file_idx"][rows]
    out["coreforest_row_idx"] = forest["coreforest_row_idx"][rows]
    out["mp_obs"] = forest["mp_obs"][rows, c]
    out["mp0"] = forest["mp0"][rows]
    for k in diffmah_fields:
        out[k] = forest[k][rows]

    top_row = forest["top_host_row"][rows, col]
    sec_row = forest["secondary_top_host_row"][rows, col]
    top_ok, sec_ok = top_row >= 0, sec_row >= 0
    top_safe, sec_safe = np.where(top_ok, top_row, 0), np.where(sec_ok, sec_row, 0)
    ct = forest["core_tag"][:, col]
    out["top_host_tag"] = np.where(top_ok, ct[top_safe], -1).astype(np.int64)
    out["secondary_top_host_tag"] = np.where(sec_ok, ct[sec_safe], -1).astype(np.int64)
    for k in host_fields:
        v = forest[k]
        out[f"top_host_{k}"] = np.where(top_ok, v[top_safe], DIFFMAH_MISSING).astype(v.dtype)
        out[f"sec_host_{k}"] = np.where(sec_ok, v[sec_safe], DIFFMAH_MISSING).astype(v.dtype)
    for k in top_host_core_fields:
        if k in forest:
            v = forest[k][:, c]
            out[f"top_host_{k}"] = np.where(top_ok, v[top_safe], np.nan).astype(v.dtype)
    ult, pen = get_infall_time_indices(
        forest["host_row"][rows], forest["central"][rows], forest["top_host_row"][rows],
        forest["secondary_top_host_row"][rows], col,
    )
    out["indx_t_ult_inf"] = ult.astype(np.int32)
    out["indx_t_pen_inf"] = pen.astype(np.int32)
    return out


# --------------------------------------------------------------------------------------
# lightcone


def read_lightcone_step(path: str):
    import pygio

    lc = pygio.read_genericio(path, LC_FIELDS, print_stats=False)
    return {k: np.asarray(v) for k, v in lc.items()}


def angles(x, y, z):
    r = np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2 + z.astype(np.float64) ** 2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.fmod(np.mod(np.arctan2(y, x), 2 * np.pi), 2 * np.pi)
    return theta, phi


def group_centrals(cores):
    """sort by (unique_id, central first, most massive first, smallest core_tag); per group the
    first core is the reference (lc_halo_*), and lc_fof_halo_flag marks groups without a central.
    core_tag breaks ties of mass (small halos have equal particle counts), so the reference does
    not depend on the order in which the rows arrived"""
    s = np.lexsort((cores["core_tag"], -cores["mp_obs"].astype(np.float64), 1 - cores["central"],
                    cores["unique_id"]))
    cores = {k: v[s] for k, v in cores.items()}
    uid, first, count = np.unique(cores["unique_id"], return_index=True, return_counts=True)
    ref = np.repeat(first, count)
    for c in "xyz":
        cores[f"lc_halo_{c}"] = cores[c][ref]
    th, ph = angles(cores["lc_halo_x"], cores["lc_halo_y"], cores["lc_halo_z"])
    cores["lc_halo_theta"], cores["lc_halo_phi"] = th, ph
    cores["lc_fof_halo_flag"] = cores["central"][ref] != 1
    return cores


def host_indices(cores):
    """top_host_idx / secondary_top_host_idx: row of the host core (same replication) in this
    file, -1 if it is not in the file; returns the number of hosts not found"""
    dt = np.dtype([("replication", np.int64), ("tag", np.int64)])
    if len(cores["core_tag"]) == 0:
        for name in ("top_host_idx", "secondary_top_host_idx"):
            cores[name] = np.empty(0, dtype=np.int64)
        return {"top_host_tag": 0, "secondary_top_host_tag": 0}
    key = np.empty(len(cores["core_tag"]), dtype=dt)
    key["replication"] = cores["replication"]
    key["tag"] = cores["core_tag"]
    s = np.argsort(key)
    lost = {}
    for name in ("top_host_tag", "secondary_top_host_tag"):
        want = np.empty(len(key), dtype=dt)
        want["replication"] = cores["replication"]
        want["tag"] = cores[name]
        has = cores[name] > 0
        idx = np.minimum(np.searchsorted(key[s], want), len(key) - 1)
        found = has & (key[s][idx] == want)
        out = np.full(len(key), -1, dtype=np.int64)
        out[found] = s[idx[found]]
        cores[name.replace("tag", "idx")] = out
        lost[name] = int(np.sum(has & ~found))
    return lost


def write_step_file(path: str, cores: dict, fields: list[str], attrs: dict,
                    compression: str, level: int):
    import h5py

    uid, offset, count = np.unique(cores["unique_id"], return_index=True, return_counts=True)
    tmp = path + ".partial"
    with h5py.File(tmp, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        g = f.create_group("data")
        for k in fields:
            create_dataset(g, k, np.ascontiguousarray(cores[k]), compression, level)
        g = f.create_group("index")
        create_dataset(g, "unique_id", uid, compression, level)
        create_dataset(g, "offset", offset.astype(np.int64), compression, level)
        create_dataset(g, "count", count.astype(np.int64), compression, level)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--core-lightcone-pattern", required=True,
              help="core lightcone GenericIO files, '#' for the step, e.g. "
              ".../step_#/m000p-#.lc_coreproperties")
@click.option("--timestep-file", required=True, type=click.Path(exists=True, dir_okay=False),
              help="lightcone steps to process (whitespace separated)")
@click.option("--output-base", required=True, help="<dir>/<name>; files go to <dir>/step_<s>/<name>-<s>.<writer>.hdf5")
@click.option("--ncorechunks", type=click.IntRange(1), default=8, show_default=True)
@click.option("--forest-files", default="all", show_default=True,
              help="forest files to read, e.g. 0-1 (debug) or all")
@click.option("--nwriters", type=click.IntRange(1), default=256, show_default=True,
              help="writer ranks (files) per step")
@click.option("--diffmah-fit/--no-diffmah-fit", default=True, show_default=True)
@click.option("--diffmah-device", type=click.Choice(["gpu", "cpu"]), default="gpu", show_default=True)
@click.option("--diffmah-workers", type=int, default=0, help="CPU worker processes per rank [OMP_NUM_THREADS]")
@click.option("--diffmah-maxiter", type=int, default=None, help="iteration cap [gpu 300, cpu 100]")
@click.option("--diffmah-batch", type=int, default=None, help="cores per optimizer call [gpu 65536, cpu 256]")
@click.option("--diffmah-dtype", type=click.Choice(["float32", "float64"]), default="float32", show_default=True)
@click.option("--compression", type=click.Choice(COMPRESSION_CHOICES), default="blosc2-zstd", show_default=True)
@click.option("--compression-level", type=int, default=1, show_default=True)
@click.option("--step-watchdog", type=float, default=1200, show_default=True,
              help="seconds; a rank whose lightcone step takes longer prints the stacks of its "
                   "threads and exits, which ends the run (0: off)")
@click.option("--read-watchdog", type=float, default=3000, show_default=True,
              help="seconds; the same for the forest read, the diffmah fits and the directory "
                   "(0: off); leaves time for a GPU timeout and CPU fits of the rank's chunks")
@click.option("--diffmah-gpu-timeout", type=float, default=600, show_default=True,
              help="seconds; a GPU fit of one forest chunk that has not returned by then is done "
                   "on the CPU, and so are the rank's later chunks (0: no limit)")
@click.option("--exchange-window", type=click.IntRange(1), default=8, show_default=True,
              help="rounds of the pairwise exchange in flight")
@click.option("--exchange-report", type=float, default=300, show_default=True,
              help="seconds; a rank still waiting in an exchange prints the partners it waits for")
def cli(config_file, core_lightcone_pattern, timestep_file, output_base, ncorechunks,
        forest_files, nwriters, diffmah_fit, diffmah_device, diffmah_workers, diffmah_maxiter,
        diffmah_batch, diffmah_dtype, compression, compression_level, step_watchdog,
        read_watchdog, diffmah_gpu_timeout, exchange_window, exchange_report):
    from mpipartition import S2Partition

    comm = MPI.COMM_WORLD
    t_start = time.time()
    EXCHANGE.update(window=exchange_window, report_s=exchange_report)
    DIFFMAH_GPU["timeout_s"] = diffmah_gpu_timeout
    gather_rank_hosts(comm)
    config = CoretreesAssemblyConfig.parse_config(str(config_file))
    sim = config.simulation
    ob = Path(config.output_base)
    forest_base = ob if ob.is_absolute() else config_file.parent / ob
    nfiles = 0
    if comm.rank == 0:
        while Path(f"{forest_base}.{nfiles}.hdf5").exists():
            nfiles += 1
    nfiles = comm.bcast(nfiles, root=0)
    assert nfiles > 0, f"no forest files {forest_base}.N.hdf5"
    if forest_files == "all":
        files = list(range(nfiles))
    else:
        a, b = map(int, forest_files.split("-"))
        files = list(range(a, b + 1))

    lc_steps = [int(s) for s in open(timestep_file).read().split()]
    cols = nearest_snapshots(lc_steps, sim.cosmotools_steps)
    col0 = max(min(cols) - 1, 0)  # one more column for the fallback to the other neighbour
    diffmah_fields = list(DEFAULT_DIFFMAH_FIELDS) + ["fit_flag"] if diffmah_fit else []
    host_fields = list(DEFAULT_DIFFMAH_HOST_FIELDS) if diffmah_fit else []
    if diffmah_workers <= 0:
        diffmah_workers = int(os.environ.get("OMP_NUM_THREADS", "1"))
    maxiter = diffmah_maxiter or (300 if diffmah_device == "gpu" else 100)
    batch = diffmah_batch or (65536 if diffmah_device == "gpu" else 256)
    nwriters = min(nwriters, comm.size)
    if comm.rank == 0:
        print(f"{comm.size} ranks, {len(lc_steps)} lightcone steps {lc_steps[0]}..{lc_steps[-1]}, "
              f"forest columns {col0}..{max(cols)} of {len(sim.cosmotools_steps)}, "
              f"{len(files)} of {nfiles} forest files, {nwriters} writers", flush=True)

    if read_watchdog > 0:
        # attempt 3 of job 9086927 stopped in the collectives at the end of the forest read
        faulthandler.dump_traceback_later(read_watchdog, exit=True)
    forest = read_forest(comm, config, forest_base, col0=col0, ncorechunks=ncorechunks,
                         diffmah_fields=diffmah_fields, diffmah_dtype=diffmah_dtype,
                         diffmah_device=diffmah_device, diffmah_workers=diffmah_workers,
                         diffmah_maxiter=maxiter, diffmah_batch=batch, files=files)
    directory = build_directory(comm, forest)
    log(comm, "directory built")
    faulthandler.cancel_dump_traceback_later()

    s2 = S2Partition()
    writer_of = (np.arange(comm.size) * nwriters) // comm.size  # writer index of each rank
    writer_root = np.searchsorted(writer_of, np.arange(nwriters))  # first rank of each writer
    is_writer = comm.rank in set(writer_root.tolist())
    my_writer = int(writer_of[comm.rank])
    members = np.nonzero(writer_of == my_writer)[0]
    th_ext = s2.all_theta_extents[members]
    ph_ext = s2.all_phi_extents[members]
    if comm.rank == 0:
        Path(output_base).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{output_base}-decomposition.txt", "w") as f:
            f.write("# writer  s2_ranks  theta_extents  phi_extents\n")
            for w in range(nwriters):
                mem = np.nonzero(writer_of == w)[0]
                f.write(f"{w} {mem.tolist()} {s2.all_theta_extents[mem].tolist()} "
                        f"{s2.all_phi_extents[mem].tolist()}\n")

    output_fields = (OUTPUT_FIELDS + diffmah_fields
                     + [f"{h}_{k}" for h in ("top_host", "sec_host") for k in host_fields]
                     + [f"top_host_{k}" for k in top_host_core_fields if k in forest]
                     + OUTPUT_TAIL)
    stats = []
    for step, col in zip(lc_steps, cols):
        t0 = time.time()
        if step_watchdog > 0:
            # a hung collective (the 6144-rank run 9080278 stopped in its first step without an
            # error) shows where every rank waits, and ends the run instead of the walltime
            faulthandler.dump_traceback_later(step_watchdog, exit=True)
        path = core_lightcone_pattern.replace("#", str(step))
        lc = read_lightcone_step(path)
        n_lc = comm.allreduce(len(lc["core_tag"]))
        lc["core_tag"] = lc["core_tag"].astype(np.int64)
        lc["octant"] = replication_to_octant(lc["fof_halo_replication"])
        # 1. to the directory rank of the core_tag, look up owner and row
        lc = exchange_by_dest(comm, lc, hash_rank(lc["core_tag"], comm.size))
        dtag = directory["tag"]
        if len(dtag):
            i = np.minimum(np.searchsorted(dtag, lc["core_tag"]), len(dtag) - 1)
            found = dtag[i] == lc["core_tag"]
        else:
            i = np.zeros(len(lc["core_tag"]), dtype=np.int64)
            found = np.zeros(len(lc["core_tag"]), dtype=bool)
        n_missing = comm.allreduce(int(np.sum(~found)))
        lc = {k: v[found] for k, v in lc.items()}
        lc["row"] = directory["row"][i[found]]
        owner = directory["owner"][i[found]].astype(np.int64)
        # 2. to the owner of the forest row, add the forest quantities at the snapshot
        lc = exchange_by_dest(comm, lc, owner)
        # the nearest snapshot, or the other neighbour when the core is not alive there
        rows = lc.pop("row")
        alt = col + 1 if sim.cosmotools_steps[col] <= step else col - 1
        alt = min(max(alt, col0), len(sim.cosmotools_steps) - 1)
        alive = forest["core_tag"][rows, col] == lc["core_tag"]
        alive_alt = forest["core_tag"][rows, alt] == lc["core_tag"]
        use_alt = ~alive & alive_alt
        keep = alive | use_alt
        n_fallback = comm.allreduce(int(np.sum(use_alt)))
        n_dead = comm.allreduce(int(np.sum(~keep)))
        parts = []
        for m_, c_ in ((alive, col), (use_alt, alt)):
            q = forest_at(forest, rows[m_], c_, col0, diffmah_fields, host_fields)
            assert np.all(q.pop("core_tag_forest") == lc["core_tag"][m_]), "forest row mismatch"
            parts.append((m_, q))
        lc = {k: np.concatenate([v[m_] for m_, _ in parts]) for k, v in lc.items()}
        fq = {k: np.concatenate([q[k] for _, q in parts]) for k in parts[0][1]}
        cores = {
            "x": lc["x"], "y": lc["y"], "z": lc["z"], "vx": lc["vx"], "vy": lc["vy"], "vz": lc["vz"],
            "core_tag": lc["core_tag"], "scale_factor": lc["a"], "radius": lc["radius"],
            "vel_disp": lc["vel_disp"], "overlap": lc["overlap"],
            "replication": lc["fof_halo_replication"].astype(np.int32),
            "lc_fof_halo_tag": lc["fof_halo_tag"].astype(np.int64),
        }
        cores.update(fq)
        th, ph = angles(cores["x"], cores["y"], cores["z"])
        cores["theta"], cores["phi"] = th, ph
        tag = cores["lc_fof_halo_tag"].copy()
        frag = tag < 0
        if np.any(frag):
            tag[frag], _ = split_fragment_tag(tag[frag])
        cores["unique_id"] = pack_unique_id(lc["octant"], tag)
        # 3. to the group's rank: central and lc_halo_*
        cores = exchange_by_dest(comm, cores, hash_rank(cores["unique_id"], comm.size))
        cores = group_centrals(cores)
        # 4. on the sphere by the group's position, then to the writer
        cores = exchange_by_dest(comm, cores, s2_home_rank(s2, cores["lc_halo_theta"], cores["lc_halo_phi"]))
        cores = exchange_by_dest(comm, cores, np.full(len(cores["x"]), writer_root[my_writer], dtype=np.int64))
        lost = {"top_host_tag": 0, "secondary_top_host_tag": 0}
        n_out = n_groups = n_nocentral = 0
        if is_writer:
            cores = group_centrals(cores)  # sorted by group, central first
            lost = host_indices(cores)
            base = step_output_base(output_base, step, True)
            base.parent.mkdir(parents=True, exist_ok=True)
            attrs = {"step": step, "snapnum": col,
                     "theta_extent": [float(th_ext[:, 0].min()), float(th_ext[:, 1].max())],
                     "phi_extent": [float(ph_ext[:, 0].min()), float(ph_ext[:, 1].max())],
                     "theta_extents": th_ext, "phi_extents": ph_ext, "s2_ranks": members}
            write_step_file(f"{base}-{step}.{my_writer}.hdf5", cores, output_fields, attrs,
                            compression, compression_level)
            n_out = len(cores["x"])
            uid, first = np.unique(cores["unique_id"], return_index=True)
            n_groups, n_nocentral = len(uid), int(np.sum(cores["lc_fof_halo_flag"][first]))
        n_out, n_groups, n_nocentral = (comm.allreduce(v) for v in (n_out, n_groups, n_nocentral))
        lost_t = {k: comm.allreduce(v) for k, v in lost.items()}
        dt = time.time() - t0
        rec = {"step": step, "snapshot": sim.cosmotools_steps[col], "lc_cores": n_lc,
               "not_in_forest": n_missing, "other_snapshot": n_fallback,
               "not_alive_at_either": n_dead, "written": n_out, "groups": n_groups,
               "groups_without_central": n_nocentral,
               "lost_top_host_idx": lost_t["top_host_tag"],
               "lost_secondary_top_host_idx": lost_t["secondary_top_host_tag"], "seconds": round(dt, 1)}
        stats.append(rec)
        if comm.rank == 0:
            print(json.dumps(rec), flush=True)
    faulthandler.cancel_dump_traceback_later()
    if comm.rank == 0:
        with open(f"{output_base}-stats.json", "w") as f:
            json.dump(stats, f, indent=1)
    log(comm, f"done in {time.time() - t_start:.0f} s")
    if DIFFMAH_GPU["hung"]:
        # the hung GPU thread could block the interpreter's shutdown (JAX's exit handlers)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.Finalize()
        os._exit(0)


if __name__ == "__main__":
    cli()
