import click
import os
from mpi4py import MPI

from mpipartition import Partition, S2Partition, distribute, s2_distribute
from haccytrees.coretrees.assemble import (
    CoretreesAssemblyConfig,
)
from haccytrees.mergertrees.fragments import split_fragment_tag
from haccytrees.coretrees import corematrix_reader
from pathlib import Path
import pygio
import numpy as np
import h5py

import numba

from haccytrees.utils.mpi_error_handler import init_mpi_error_handler

init_mpi_error_handler()

core_fields = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "fof_halo_tag",
    "central",
    "core_tag",
    "infall_fof_halo_center_x",
    "infall_fof_halo_center_y",
    "infall_fof_halo_center_z",
]

# Fields of the top host core that are copied to each core (at the lightcone step),
# if they exist in the coreforest files.
top_host_core_fields = [
    f"infall_fof_halo_eigS{i}{x}" for i in (1, 2, 3) for x in ("X", "Y", "Z")
]

# diffmah fits (one row per core = per coreforest matrix row, files named by the
# coreforest file index), copied to each core and, for a subset, from the top and
# secondary ("penultimate") host core. Missing fits are marked with DIFFMAH_MISSING
# in the diffmah files, and hosts that do not exist get the same marker.
DEFAULT_DIFFMAH_FIELDS = [
    "early_index",
    "late_index",
    "logm0",
    "logtc",
    "loss",
    "n_points_per_fit",
    "t_peak",
]
DEFAULT_DIFFMAH_HOST_FIELDS = ["early_index", "late_index", "logm0", "logtc", "t_peak"]
DIFFMAH_MISSING = -99

# matrix-row references into the coreforest matrix (chunk-local when read, made
# rank-local when chunks are concatenated in read_corematrix)
host_row_fields = ["host_row", "top_host_row", "secondary_top_host_row"]


def compose_unique_id(rep, tag):
    assert np.all(rep < (1 << 23))
    assert np.all(tag < (1 << 41))
    assert np.all(rep >= 0)
    assert np.all(tag >= 0)
    return (rep.astype(np.uint64) << 41) + tag.astype(np.uint64)


@numba.njit(parallel=True)
def update_host_tag_matrix(host_row, core_tag, host_tag):
    rows, cols = host_row.shape
    # Loop over all entries and update in-place:
    for i in numba.prange(rows):
        for j in range(cols):
            # If host_row is non-negative, assign the corresponding core_tag,
            # otherwise leave it as -1.
            if host_row[i, j] >= 0:
                host_tag[i, j] = core_tag[host_row[i, j], j]
            else:
                host_tag[i, j] = -1


def get_infall_time_indices(
    host_row, is_central, top_host_row, secondary_top_host_row, iz
):
    """Timestep of first infall into penultimate and ultimate hosts, -1 for centrals

    Copied verbatim from diffsky (diffsky/data_loaders/hacc_utils/load_hacc_cores.py,
    commit 3996ab0b) so that the indices stored on the lightcone are the ones diffsky
    computes for a core observed at snapshot column `iz`. The returned values are
    column indices into the coreforest snapshot axis (the simulation's cosmotools
    steps). Note that the penultimate-host search runs over all columns, as in
    diffsky, and argmax returns 0 when no column matches.
    """
    _X = top_host_row
    M_ult_host = host_row == _X[:, iz].reshape((-1, 1))
    indx_t_ult_inf_case2 = np.argmax(M_ult_host[:, : iz + 1], axis=1)

    # Was the core identified prior to iz?
    core_only_minus1 = np.all(host_row[:, : iz + 1] == -1, axis=1)

    # Was the core always a central prior to iz?
    is_central_whole_life = ~np.any(
        (host_row[:, : iz + 1] > -1) & (is_central[:, : iz + 1] < 1), axis=1
    )

    # msk_case1: non-satellites
    msk_case1 = core_only_minus1 | is_central_whole_life

    # indx_t_ult_inf = indx_t_ult_inf_case2 for satellites, -1 otherwise
    indx_t_ult_inf = np.where(msk_case1, -1, indx_t_ult_inf_case2)

    _Y = secondary_top_host_row
    M_pen_host = _X == _Y[:, iz].reshape((-1, 1))
    indx_t_pen_inf_case3 = np.argmax(M_pen_host, axis=1)

    # msk_case3: satellites with an existing secondary host at iz
    msk_case3 = ~msk_case1 & (secondary_top_host_row[:, iz] != -1)

    # indx_t_ult_inf = indx_t_ult_inf_case3 for sats-of-sats, -1 otherwise
    indx_t_pen_inf = np.where(msk_case3, indx_t_pen_inf_case3, -1)

    return indx_t_ult_inf, indx_t_pen_inf


def create_dataset(group, name: str, data: np.ndarray, compression_level: int):
    """gzip-compressed (with byte shuffle) chunked dataset; level 0 writes it plain.
    Empty datasets cannot be chunked and are always written plain."""
    if compression_level > 0 and len(data) > 0:
        group.create_dataset(
            name,
            data=data,
            chunks=True,
            shuffle=True,
            compression="gzip",
            compression_opts=compression_level,
        )
    else:
        group.create_dataset(name, data=data)


def read_diffmah_rows(
    diffmah_pattern: str,
    file_idx: int,
    row_start: int,
    row_end: int,
    fields: list[str],
    float_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """read the diffmah fits of coreforest matrix rows [row_start, row_end) of file file_idx

    Floating point fields are cast to float_dtype (the fit parameters span
    logm0 11.5..17, logtc -1..1, indices 0.1..10, t_peak 1..14 Gyr, loss 1e-9..0.1,
    none of which needs double precision); integer fields are stored as int16.
    """
    filename = diffmah_pattern.replace("#", str(file_idx))
    with h5py.File(filename) as f:
        data = {k: f[k][row_start:row_end] for k in fields}
    for k, v in data.items():
        assert len(v) == row_end - row_start, (
            f"{filename}: {k} has {len(v)} rows in [{row_start}, {row_end}), "
            f"does the diffmah file match the coreforest file?"
        )
        dtype = diffmah_storage_dtype(k, float_dtype)
        if np.issubdtype(dtype, np.integer):
            assert np.all(np.abs(v) < 2 ** (8 * dtype.itemsize - 1))
        data[k] = v.astype(dtype)
    return data


def diffmah_storage_dtype(name: str, float_dtype: str) -> np.dtype:
    """storage type of a diffmah column in the lightcone files (see read_diffmah_rows)"""
    if name == "fit_flag":
        return np.dtype(np.int8)
    if name in ("n_points_per_fit", "fit_algo", "n_iterations"):
        return np.dtype(np.int16)
    return np.dtype(float_dtype)


def cast_diffmah_results(
    results: dict[str, np.ndarray], fields: list[str], float_dtype: str
) -> dict[str, np.ndarray]:
    """select and cast the arrays returned by haccytrees.diffmah.fit_mahs"""
    missing = [k for k in fields if k not in results]
    assert not missing, f"diffmah fitter does not provide {missing}"
    return {k: results[k].astype(diffmah_storage_dtype(k, float_dtype)) for k in fields}


def host_quantities_at_step(
    corematrix: dict[str, np.ndarray],
    snap_num: int,
    diffmah_host_fields: list[str],
) -> dict[str, np.ndarray]:
    """per-core (1d) quantities at snapshot column snap_num that need the full matrix:
    diffmah fits and infall properties of the top and secondary host cores, and the
    diffsky infall time indices. Must be called before the cores are redistributed,
    since the host rows refer to this rank's corematrix."""
    out: dict[str, np.ndarray] = {}
    top_row = corematrix["top_host_row"][:, snap_num]
    sec_row = corematrix["secondary_top_host_row"][:, snap_num]
    top_ok = top_row >= 0
    sec_ok = sec_row >= 0
    top_safe = np.where(top_ok, top_row, 0)
    sec_safe = np.where(sec_ok, sec_row, 0)

    for k in diffmah_host_fields:
        if k in corematrix:
            v = corematrix[k]
            out[f"top_host_{k}"] = np.where(
                top_ok, v[top_safe], DIFFMAH_MISSING
            ).astype(v.dtype)
            out[f"sec_host_{k}"] = np.where(
                sec_ok, v[sec_safe], DIFFMAH_MISSING
            ).astype(v.dtype)

    for k in top_host_core_fields:
        if k in corematrix:
            v = corematrix[k][:, snap_num]
            out[f"top_host_{k}"] = np.where(top_ok, v[top_safe], np.nan).astype(v.dtype)

    indx_t_ult_inf, indx_t_pen_inf = get_infall_time_indices(
        corematrix["host_row"],
        corematrix["central"],
        corematrix["top_host_row"],
        corematrix["secondary_top_host_row"],
        snap_num,
    )
    out["indx_t_ult_inf"] = indx_t_ult_inf.astype(np.int32)
    out["indx_t_pen_inf"] = indx_t_pen_inf.astype(np.int32)
    return out


def read_corematrix(
    partition_cube: Partition,
    coreforest_base: Path,
    config: CoretreesAssemblyConfig,
    *,
    diffmah_pattern: str | None = None,
    diffmah_fields: list[str] = DEFAULT_DIFFMAH_FIELDS,
    diffmah_dtype: str = "float32",
    diffmah_fit: bool = False,
    diffmah_workers: int = 1,
    diffmah_fit_maxiter: int = 100,
    diffmah_fit_batch: int = 256,
    nchunks: int = 8,
    dbg_nfiles: int | None = None,
) -> dict[str, np.ndarray]:
    if diffmah_fit:
        # on-the-fly diffmah fits per chunk (haccytrees.diffmah, CPU worker processes)
        from haccytrees.diffmah import (
            cosmic_time,
            default_cpu_config,
            fit_mahs_parallel,
            mah_from_mass,
        )

        _tarr = cosmic_time(config.simulation)
        _cfg = default_cpu_config(
            dtype="float64", maxiter=diffmah_fit_maxiter, batch=diffmah_fit_batch
        )
        if partition_cube.rank == 0:
            print(
                f"diffmah fits on the fly: {_cfg}, {diffmah_workers} workers/rank",
                flush=True,
            )
    if partition_cube.rank == 0:
        number_of_core_files = 0
        while Path(f"{coreforest_base}.{number_of_core_files}.hdf5").exists():
            number_of_core_files += 1
        # optional fields: only those present in the coreforest files
        with h5py.File(f"{coreforest_base}.0.hdf5") as f:
            available = set(f["data"].keys())
        include_fields = core_fields + [
            k for k in top_host_core_fields if k in available
        ]
        if diffmah_fit:
            include_fields.append("infall_tree_node_mass")
        missing = [k for k in top_host_core_fields if k not in available]
        if missing:
            print(f"Coreforest has no {missing}, skipping top_host_ copies", flush=True)
    else:
        number_of_core_files = None
        include_fields = None
    number_of_core_files = partition_cube.comm.bcast(number_of_core_files, root=0)
    include_fields = partition_cube.comm.bcast(include_fields, root=0)

    if dbg_nfiles is not None:
        number_of_core_files = dbg_nfiles

    assert number_of_core_files > 0

    corematrix: dict[str, np.ndarray] | None = None

    # Just read a tiny bit of data so that we have the right fields we can send to everyone
    if number_of_core_files < partition_cube.nranks:
        if partition_cube.rank == 0:
            corematrix = corematrix_reader(
                f"{coreforest_base}.{0}.hdf5",
                config.simulation,
                include_fields=list(include_fields),
                calculate_host_rows=True,
                calculate_secondary_host_row=True,
                nchunks=10000,
                chunknum=0,
            )
            for k in corematrix.keys():
                corematrix[k] = np.empty(
                    (0,) + corematrix[k].shape[1:], dtype=corematrix[k].dtype
                )
            corematrix["coreforest_file_idx"] = np.empty(
                (0, corematrix["x"].shape[1]), dtype=np.uint16
            )
            corematrix["coreforest_row_idx"] = corematrix.pop("absolute_row_idx")
            corematrix["coreforest_row_idx"] = np.tile(
                corematrix["coreforest_row_idx"].reshape(-1, 1),
                (1, corematrix["x"].shape[1]),
            )
            corematrix["top_host_tag"] = np.empty(
                (0, corematrix["x"].shape[1]), dtype=np.int64
            )
            corematrix["secondary_top_host_tag"] = np.empty(
                (0, corematrix["x"].shape[1]), dtype=np.int64
            )
            if diffmah_pattern is not None:
                for k, v in read_diffmah_rows(
                    diffmah_pattern, 0, 0, 1, diffmah_fields, diffmah_dtype
                ).items():
                    corematrix[k] = np.empty((0,), dtype=v.dtype)
            elif diffmah_fit:
                corematrix.pop("infall_tree_node_mass")
                for k in diffmah_fields:
                    corematrix[k] = np.empty(
                        (0,), dtype=diffmah_storage_dtype(k, diffmah_dtype)
                    )
        corematrix = partition_cube.comm.bcast(corematrix, root=0)

    # Actually reading data
    if partition_cube.rank == 0:
        print(f"Reading {number_of_core_files} core files", flush=True)
    partition_cube.comm.Barrier()

    for j in range(
        partition_cube.rank, number_of_core_files * nchunks, partition_cube.nranks
    ):
        i = j // nchunks
        chunknum = j % nchunks

        _corematrix = corematrix_reader(
            f"{coreforest_base}.{i}.hdf5",
            config.simulation,
            include_fields=list(include_fields),
            calculate_host_rows=True,
            calculate_secondary_host_row=True,
            nchunks=nchunks,
            chunknum=chunknum,
        )
        _corematrix["coreforest_file_idx"] = np.full(
            _corematrix["x"].shape, i, dtype=np.uint16
        )
        _corematrix["coreforest_row_idx"] = _corematrix.pop("absolute_row_idx")
        if diffmah_pattern is not None:
            # diffmah row j of file i is coreforest matrix row j of file i; a chunk
            # covers a contiguous range of matrix rows
            _rows = _corematrix["coreforest_row_idx"]
            if len(_rows) > 0:  # empty when the file has fewer roots than chunks
                assert _rows[-1] - _rows[0] + 1 == len(_rows)
                _corematrix.update(
                    read_diffmah_rows(
                        diffmah_pattern,
                        i,
                        int(_rows[0]),
                        int(_rows[-1]) + 1,
                        diffmah_fields,
                        diffmah_dtype,
                    )
                )
            else:
                _corematrix.update(
                    {
                        k: np.empty(0, dtype=diffmah_storage_dtype(k, diffmah_dtype))
                        for k in diffmah_fields
                    }
                )
        elif diffmah_fit:
            _mahs = mah_from_mass(
                _corematrix.pop("infall_tree_node_mass"), config.simulation.cosmo.h
            )
            _res = fit_mahs_parallel(_tarr, _mahs, _cfg, diffmah_workers)
            del _mahs
            _corematrix.update(
                cast_diffmah_results(_res, diffmah_fields, diffmah_dtype)
            )
        _corematrix["coreforest_row_idx"] = np.tile(
            _corematrix["coreforest_row_idx"].reshape(-1, 1),
            (1, _corematrix["x"].shape[1]),
        )

        _corematrix["top_host_tag"] = np.empty(
            _corematrix["top_host_row"].shape, dtype=np.int64
        )
        update_host_tag_matrix(
            _corematrix["top_host_row"],
            _corematrix["core_tag"],
            _corematrix["top_host_tag"],
        )

        _corematrix["secondary_top_host_tag"] = np.empty(
            _corematrix["secondary_top_host_row"].shape, dtype=np.int64
        )
        update_host_tag_matrix(
            _corematrix["secondary_top_host_row"],
            _corematrix["core_tag"],
            _corematrix["secondary_top_host_tag"],
        )

        if corematrix is None:
            corematrix = _corematrix
        else:
            # host rows are chunk-local: shift them to rows of the concatenated matrix
            offset = corematrix["core_tag"].shape[0]
            for k in host_row_fields:
                if k in _corematrix:
                    _corematrix[k][_corematrix[k] >= 0] += offset
            for k in corematrix.keys():
                corematrix[k] = np.concatenate([corematrix[k], _corematrix[k]], axis=0)

    partition_cube.comm.Barrier()
    if partition_cube.rank == 0:
        print("Reading core files done", flush=True)
    partition_cube.comm.Barrier()
    assert corematrix is not None
    return corematrix


def read_lightcone(partition_cube: Partition, lightcone_path: Path, simulation_np: int):
    """read halo lightcone and distribute halos according to their Lagrangian position

    Parameters
    ----------
    partition_cube : Partition
        partition object
    lightcone_path : Path
        path to lightcone file (genericio)
    simulation_np : int
        number of particles per side of the simulation box

    Returns
    -------
    halo_lc : dict
        dictionary containing the lightcone halos. The halos are sorted by their (non-fragmented) fof_halo_tag
    """
    lc_fields = ["x", "y", "z", "id", "a", "replication"]
    halo_lc = pygio.read_genericio(str(lightcone_path), lc_fields)

    # distribute LC halos according to their Lagrangian position
    # note:: halo LC only contains one fragment per fof halo, which does not need
    # to be fragment 0
    mask_fragment = halo_lc["id"] < 0
    halo_lc["fragment_idx"] = np.zeros_like(halo_lc["id"])
    halo_lc["fof_halo_tag_clean"] = np.copy(halo_lc["id"])
    (
        halo_lc["fof_halo_tag_clean"][mask_fragment],
        halo_lc["fragment_idx"][mask_fragment],
    ) = split_fragment_tag(halo_lc["id"][mask_fragment])
    assert np.all(halo_lc["fof_halo_tag_clean"] >= 0)

    halo_lc["qz"] = (halo_lc["fof_halo_tag_clean"] % simulation_np) / simulation_np
    halo_lc["qy"] = (
        (halo_lc["fof_halo_tag_clean"] // simulation_np) % simulation_np
    ) / simulation_np
    halo_lc["qx"] = (halo_lc["fof_halo_tag_clean"] // simulation_np**2) / simulation_np

    assert np.all(halo_lc["qx"] >= 0)
    assert np.all(halo_lc["qy"] >= 0)
    assert np.all(halo_lc["qz"] >= 0)
    assert np.all(halo_lc["qx"] < 1)
    assert np.all(halo_lc["qy"] < 1)
    assert np.all(halo_lc["qz"] < 1)
    halo_lc = distribute(partition_cube, 1.0, halo_lc, ["qx", "qy", "qz"])

    # order by fof_halo_tag (without fragment index)
    s = np.argsort(halo_lc["fof_halo_tag_clean"])
    halo_lc = {k: v[s] for k, v in halo_lc.items()}

    assert np.all(halo_lc["fof_halo_tag_clean"] >= 0)
    assert np.all(halo_lc["fof_halo_tag_clean"] < (1 << 41))
    assert np.all(halo_lc["replication"] >= 0)
    assert np.all(halo_lc["replication"] < (1 << 22))
    unique_id = compose_unique_id(halo_lc["replication"], halo_lc["fof_halo_tag_clean"])
    # unique_id = (halo_lc["replication"].astype(np.int64) << 41) + halo_lc["fof_halo_tag_clean"]
    num_duplicates = len(unique_id) - len(np.unique(unique_id))
    max_duplicates = 0
    num_duplicated_halos = 0
    if num_duplicates > 0:
        _uq, _idx, _cnt = np.unique(unique_id, return_index=True, return_counts=True)
        _mask = _cnt > 1
        max_duplicates = np.max(_cnt)
        num_duplicated_halos = np.sum(_mask)
        # _prtidx = np.argmax(_cnt)
        # _prtindices = np.nonzero(unique_id == _uq[_prtidx])[0]
        # print(
        #     f"DEBUG:: rank {partition_cube.rank} found {np.sum(_mask)} halos with the same fof_halo_tag/replication",
        #     f"max duplicate count {_cnt[_prtidx]}",
        #     f"fof_halo_tag_clean={halo_lc['fof_halo_tag_clean'][_prtindices]}",
        #     f"fof_halo_tag={halo_lc['id'][_prtindices]}",
        #     f"replication={halo_lc['replication'][_prtindices]}",
        #     f"pos=[{halo_lc['x'][_prtindices]}, {halo_lc['y'][_prtindices]}, {halo_lc['z'][_prtindices]}]",
        #     flush=True
        #     )
        halo_lc = {k: v[_idx] for k, v in halo_lc.items()}
        unique_id = unique_id[_idx]

        # make sure it's sorted by fof_halo_tag_clean
        s = np.argsort(halo_lc["fof_halo_tag_clean"])
        halo_lc = {k: v[s] for k, v in halo_lc.items()}
        unique_id = unique_id[s]

    max_duplicates_global = partition_cube.comm.reduce(
        max_duplicates, op=MPI.MAX, root=0
    )
    num_duplicated_halos_global = partition_cube.comm.reduce(
        num_duplicated_halos, op=MPI.SUM, root=0
    )
    if partition_cube.rank == 0:
        print(
            f"DEBUG:: found {num_duplicated_halos_global} duplicated halos, max duplicates {max_duplicates_global}",
            flush=True,
        )

    assert len(np.unique(unique_id)) == len(halo_lc["id"])
    halo_lc["unique_id"] = unique_id

    unique_tags, unique_idx, unique_reverse, unique_counts = np.unique(
        halo_lc["fof_halo_tag_clean"],
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    # make sure each non-unique fof_halo_tag has the same fragment index
    mask = (
        halo_lc["fragment_idx"] - halo_lc["fragment_idx"][unique_idx][unique_reverse]
        == 0
    )
    if not np.all(mask):
        idx_comp = (unique_idx[unique_reverse])[~mask]
        print_mask = mask.copy()
        print_mask[idx_comp] = False
        print("WARNING: multiple fragment_index found for same fof_halo_tag:")
        print(f"   clean_tag: {halo_lc['fof_halo_tag_clean'][~print_mask]}")
        print(f"   frag_idx : {halo_lc['fragment_idx'][~print_mask]}")
        print(f"   replicat : {halo_lc['replication'][~print_mask]}")
        print(f"   orig_tag : {halo_lc['id'][~print_mask]}", flush=True)

        halo_lc = {k: d[mask] for k, d in halo_lc.items()}
        unique_tags, unique_idx, unique_reverse, unique_counts = np.unique(
            halo_lc["fof_halo_tag_clean"],
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
    partition_cube.comm.Barrier()
    assert np.all(
        halo_lc["fragment_idx"] - halo_lc["fragment_idx"][unique_idx][unique_reverse]
        == 0
    )

    # print("DEBUG max replications", np.max(unique_counts))
    halo_lc["replications_count"] = unique_counts[unique_reverse]

    return halo_lc


def distribute_cores_at_step(
    partition_cube: Partition,
    corematrix: dict[str, np.ndarray],
    snap_num: int,
    simulation_np: int,
    diffmah_host_fields: list[str] = DEFAULT_DIFFMAH_HOST_FIELDS,
):
    # Get all cores at that step (2d fields are per snapshot, 1d fields are per core)
    cores_step = {
        k: (v[:, snap_num] if v.ndim == 2 else v) for k, v in corematrix.items()
    }
    cores_step.update(
        host_quantities_at_step(corematrix, snap_num, diffmah_host_fields)
    )
    mask = cores_step["core_tag"] > 0

    # Make sure the host has the same fof_halo_tag as the core
    # assert np.all(
    #     cores_step["fof_halo_tag"][mask]
    #     == cores_step["fof_halo_tag"][cores_step["top_host_row"][mask]]
    # )

    cores_step = {k: v[mask] for k, v in cores_step.items()}

    # distribute cores by their Lagrangian position
    mask_fragment = cores_step["fof_halo_tag"] < 0
    cores_step["fof_halo_tag_clean"] = np.copy(cores_step["fof_halo_tag"])
    cores_step["fragment_idx"] = np.zeros_like(cores_step["fof_halo_tag"])
    (
        cores_step["fof_halo_tag_clean"][mask_fragment],
        cores_step["fragment_idx"][mask_fragment],
    ) = split_fragment_tag(cores_step["fof_halo_tag"][mask_fragment])
    assert np.all(cores_step["fof_halo_tag_clean"] >= 0)
    cores_step["qz"] = (
        cores_step["fof_halo_tag_clean"] % simulation_np
    ) / simulation_np
    cores_step["qy"] = (
        (cores_step["fof_halo_tag_clean"] // simulation_np) % simulation_np
    ) / simulation_np
    cores_step["qx"] = (
        cores_step["fof_halo_tag_clean"] // simulation_np**2
    ) / simulation_np
    assert np.all(cores_step["qx"] >= 0)
    assert np.all(cores_step["qy"] >= 0)
    assert np.all(cores_step["qz"] >= 0)
    assert np.all(cores_step["qx"] < 1)
    assert np.all(cores_step["qy"] < 1)
    assert np.all(cores_step["qz"] < 1)
    cores_step = distribute(partition_cube, 1.0, cores_step, ["qx", "qy", "qz"])

    return cores_step


@click.command()
@click.argument(
    "config_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--lightcone-pattern",
    required=True,
    type=str,
)
@click.option(
    "--timestep-file",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--output-base",
    required=True,
    type=str,
)
@click.option(
    "--diffmah-pattern",
    type=str,
    default=None,
    help="diffmah fit files, one per coreforest file, with # replaced by the "
    "coreforest file index, e.g. /path/subvol_#_diffmah_fits.hdf5. Row j of the "
    "diffmah file has to be matrix row j (coreforest_row_idx) of the coreforest file.",
)
@click.option(
    "--diffmah-fields",
    type=str,
    default=",".join(DEFAULT_DIFFMAH_FIELDS),
    show_default=True,
    help="comma-separated datasets to copy from the diffmah files to each core",
)
@click.option(
    "--diffmah-host-fields",
    type=str,
    default=",".join(DEFAULT_DIFFMAH_HOST_FIELDS),
    show_default=True,
    help="comma-separated subset of --diffmah-fields also copied from the top host "
    "(top_host_*) and the secondary host (sec_host_*) core at the lightcone step; "
    "empty to disable",
)
@click.option(
    "--diffmah-fit",
    is_flag=True,
    help="compute the diffmah fits on the fly from the coreforest mass histories "
    "(haccytrees.diffmah, CPU worker processes per rank) instead of reading them "
    "with --diffmah-pattern; adds the fit_flag column",
)
@click.option(
    "--diffmah-workers",
    type=int,
    default=0,
    help="worker processes per rank for --diffmah-fit [default: OMP_NUM_THREADS or 1]",
)
@click.option(
    "--diffmah-fit-maxiter",
    type=int,
    default=100,
    show_default=True,
    help="iteration cap of the on-the-fly fitter (scipy-like stopping rule, float64)",
)
@click.option(
    "--diffmah-fit-batch",
    type=int,
    default=256,
    show_default=True,
    help="cores per optimizer call of the on-the-fly fitter (CPU: keep small)",
)
@click.option(
    "--diffmah-dtype",
    type=click.Choice(["float32", "float64"]),
    default="float32",
    show_default=True,
    help="storage type of the diffmah floating point fields (core and host copies)",
)
@click.option(
    "--compression-level",
    type=click.IntRange(0, 9),
    default=4,
    show_default=True,
    help="gzip level for the HDF5 datasets (with byte shuffle); 0 disables compression",
)
@click.option(
    "--ncorechunks",
    type=click.IntRange(1),
    default=8,
    show_default=True,
    help="split every coreforest file into this many root-aligned chunks; the "
    "nfiles*ncorechunks chunks are distributed round-robin over the ranks, so "
    "the per-rank memory is set by ceil(nfiles*ncorechunks/nranks)/ncorechunks files",
)
@click.option(
    "--dbg-ncorefiles",
    type=int,
)
def cli(
    config_file: Path,
    lightcone_pattern: str,
    timestep_file: Path,
    output_base: Path,
    diffmah_pattern: str | None,
    diffmah_fields: str,
    diffmah_host_fields: str,
    diffmah_fit: bool,
    diffmah_workers: int,
    diffmah_fit_maxiter: int,
    diffmah_fit_batch: int,
    diffmah_dtype: str,
    compression_level: int,
    ncorechunks: int,
    dbg_ncorefiles: int | None,
):
    diffmah_fields = [k for k in diffmah_fields.split(",") if k]
    diffmah_host_fields = [k for k in diffmah_host_fields.split(",") if k]
    if diffmah_pattern is not None and diffmah_fit:
        raise click.BadParameter("--diffmah-pattern and --diffmah-fit are exclusive")
    if diffmah_pattern is None and not diffmah_fit:
        diffmah_fields = []
        diffmah_host_fields = []
    if diffmah_fit and "fit_flag" not in diffmah_fields:
        diffmah_fields.append("fit_flag")
    if diffmah_workers <= 0:
        diffmah_workers = int(os.environ.get("OMP_NUM_THREADS", "1"))
    if diffmah_pattern is not None and "#" not in diffmah_pattern:
        raise click.BadParameter(
            "--diffmah-pattern needs a '#' placeholder for the coreforest file index"
        )
    unknown = [k for k in diffmah_host_fields if k not in diffmah_fields]
    if unknown:
        raise click.BadParameter(
            f"--diffmah-host-fields {unknown} not in --diffmah-fields {diffmah_fields}"
        )

    partition_cube = Partition(3)
    partition_s2 = S2Partition()

    if partition_s2.rank == 0:
        with open(str(output_base) + "-decomposition.txt", "w") as f:
            f.write(f"{'index':<10} {'theta':<20} {'phi':<20}\n\n")
            for i in range(partition_s2.nranks):
                _theta = partition_s2.all_theta_extents[i]
                _theta = f"[{_theta[0]:8.6f}, {_theta[1]:8.6f}]"
                _phi = partition_s2.all_phi_extents[i]
                _phi = f"[{_phi[0]:8.6f}, {_phi[1]:8.6f}]"
                f.write(f"{i:<10} {_theta:<20} {_phi:<20}\n")

    config = CoretreesAssemblyConfig.parse_config(str(config_file))
    sim_np = config.simulation.np

    with open(timestep_file) as f:
        timesteps = [int(t) for t in f.read().split()]
    # +1 because of how the lightcone is constructed
    snapnums = [
        config.simulation.cosmotools_steps.index(step) + 1 for step in timesteps
    ]

    # Read all coretrees
    coreforest_base = config_file.parent / config.output_base
    corematrix = read_corematrix(
        partition_cube,
        coreforest_base,
        config,
        diffmah_pattern=diffmah_pattern,
        diffmah_fields=diffmah_fields,
        diffmah_dtype=diffmah_dtype,
        diffmah_fit=diffmah_fit,
        diffmah_workers=diffmah_workers,
        diffmah_fit_maxiter=diffmah_fit_maxiter,
        diffmah_fit_batch=diffmah_fit_batch,
        nchunks=ncorechunks,
        dbg_nfiles=dbg_ncorefiles,
    )

    # Forward iterate over lightcone outputs
    for step, snap_num in zip(timesteps, snapnums):
        if partition_cube.rank == 0:
            _tstep = config.simulation.cosmotools_steps[snap_num]
            print(
                f"Processing step {step} (snapnum {snap_num}, target_step {_tstep})",
                flush=True,
            )
        partition_cube.comm.Barrier()

        # Read LC shell per step
        if partition_cube.rank == 0:
            print(" - Reading lightcone", flush=True)
        lightcone_catalog = lightcone_pattern.replace("#", str(step))
        halo_lc = read_lightcone(partition_cube, Path(lightcone_catalog), sim_np)
        partition_cube.comm.Barrier()

        if partition_cube.rank == 0:
            print(" - Distribute cores at step", flush=True)
        cores_step = distribute_cores_at_step(
            partition_cube, corematrix, snap_num, sim_np, diffmah_host_fields
        )
        # At this point, cores and halos on the lightcone are on the same rank

        ################################################################################
        # for each core, find one halo in the LC
        if partition_cube.rank == 0:
            print(" - Match LC with cores", flush=True)
        # Get all the cores whos parent halo intersects with the lightcone
        mask = np.isin(cores_step["fof_halo_tag_clean"], halo_lc["fof_halo_tag_clean"])
        cores_step = {k: v[mask] for k, v in cores_step.items()}

        # Sort cores by fof_halo_tag (including fragment index), central core first
        # within each halo: the leftmost core of a halo is used as the reference
        # position below (searchsorted), so it has to be the central and not an
        # arbitrary satellite (which happened before with a plain, unstable argsort
        # and displaced ~12% of the centrals from the lightcone halo position).
        s = np.lexsort((1 - cores_step["central"], cores_step["fof_halo_tag"]))
        cores_step = {k: v[s] for k, v in cores_step.items()}

        # Match cores to halos by fof_halo_tag (without fragment index)
        assert np.all(np.diff(halo_lc["fof_halo_tag_clean"]) >= 0)  # check if sorted
        # for each core, find one halo in the LC
        lc_index = np.searchsorted(
            halo_lc["fof_halo_tag_clean"], cores_step["fof_halo_tag_clean"]
        )
        assert np.all(lc_index < len(halo_lc["fof_halo_tag_clean"]))
        assert np.all(lc_index >= 0)
        assert np.all(
            halo_lc["fof_halo_tag_clean"][lc_index] == cores_step["fof_halo_tag_clean"]
        )

        ################################################################################
        # Find central core corresponding to the fragment tag in the lightcone
        if partition_cube.rank == 0:
            print("   - find central core with matching fragment tag", flush=True)
        cores_lc_host_tag = halo_lc["id"][lc_index]
        mask_missing = ~np.isin(cores_lc_host_tag, cores_step["fof_halo_tag"])
        num_missing = np.sum(mask_missing)
        num_total = len(mask_missing)
        if num_missing > 0:
            # Remove missing halos
            cores_step = {k: v[~mask_missing] for k, v in cores_step.items()}
            cores_lc_host_tag = cores_lc_host_tag[~mask_missing]
            lc_index = lc_index[~mask_missing]
        num_missing_global = partition_cube.comm.reduce(num_missing, root=0)
        num_total_global = partition_cube.comm.reduce(num_total, root=0)
        if partition_cube.rank == 0:
            print(
                f"   - missing {num_missing_global} out of {num_total_global} fof_halo_tag",
                flush=True,
            )
        assert np.all(np.isin(cores_lc_host_tag, cores_step["fof_halo_tag"]))

        # Get the actual index of the host core
        cores_lc_host_idx = np.searchsorted(
            cores_step["fof_halo_tag"], cores_lc_host_tag
        )
        assert np.all(
            cores_step["fof_halo_tag"][cores_lc_host_idx] == cores_lc_host_tag
        )

        ################################################################################
        # Calculate distances to the host core
        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - calculate core offsets", flush=True)
        mask_valid = np.ones_like(cores_step["x"], dtype=np.bool_)
        for x in "xyz":
            _dx = (
                cores_step[x] - cores_step[x][cores_lc_host_idx]
                # - cores_step[f"infall_fof_halo_center_{x}"][cores_lc_host_idx]
            )
            _dx[_dx > config.simulation.rl / 2] -= config.simulation.rl
            _dx[_dx < -config.simulation.rl / 2] += config.simulation.rl
            if not np.all(np.abs(_dx) <= 20):
                print("DEBUG:: found large dx", x, _dx[np.abs(_dx) > 20], flush=True)
            mask_valid &= np.abs(_dx) <= 20
            # assert np.all(np.abs(_dx) <= 20)
            cores_step[f"d{x}"] = _dx

        cores_step = {k: v[mask_valid] for k, v in cores_step.items()}
        lc_index = lc_index[mask_valid]
        cores_lc_host_tag = cores_lc_host_tag[mask_valid]

        ################################################################################
        # Handle replications
        if partition_cube.rank == 0:
            print("   - handle replications", flush=True)
        if len(lc_index) > 0:
            _counts = halo_lc["replications_count"][lc_index]
            assert np.all(_counts > 0)
            # replicate each core by the number of replications
            s = np.repeat(np.arange(len(cores_lc_host_tag)), _counts)
            cores_step = {k: v[s] for k, v in cores_step.items()}
            lc_index = lc_index[s]
            # offset each repeated index by 1 (so it points to the next replicated halo in the lightcone)
            lc_index += np.concatenate([np.arange(c) for c in _counts])
            # add replications to cores_step
            cores_step["replication"] = halo_lc["replication"][lc_index]
            cores_step["unique_id"] = compose_unique_id(
                cores_step["replication"], cores_step["fof_halo_tag_clean"]
            )
            # make sure we didn't screw up anything
            assert np.all(
                halo_lc["fof_halo_tag_clean"][lc_index]
                == cores_step["fof_halo_tag_clean"]
            )
            assert np.all(halo_lc["unique_id"][lc_index] == cores_step["unique_id"])
            mask_invalid = halo_lc["id"][lc_index] != cores_lc_host_tag[s]
            if np.any(mask_invalid):
                print(f"DEBUG:: found {np.sum(mask_invalid)} mismatching fof_halo_tag")
                print(
                    halo_lc["id"][lc_index][mask_invalid],
                    cores_lc_host_tag[s][mask_invalid],
                    flush=True,
                )
            cores_step["lc_fof_halo_tag"] = halo_lc["id"][lc_index]
            cores_step["lc_fof_halo_flag"] = mask_invalid
        else:
            cores_step["replication"] = np.empty(0, halo_lc["replication"].dtype)
            cores_step["unique_id"] = np.empty(0, halo_lc["unique_id"].dtype)
            cores_step["lc_fof_halo_tag"] = np.empty(0, halo_lc["id"].dtype)
            cores_step["lc_fof_halo_flag"] = np.empty(0, np.bool_)

        ################################################################################
        # Apply offsets to core positions
        for x in "xyz":
            cores_step[x] = halo_lc[x][lc_index] + cores_step[f"d{x}"]
            cores_step[f"host_{x}"] = halo_lc[x][lc_index]

        ################################################################################
        # Calculate angular coordinates
        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - calculate angular coordinates", flush=True)
        # Calculate angular lightcone coordinates: the usual spherical convention,
        # theta = polar angle from +z in [0, pi], phi = azimuth from +x in [0, 2pi)
        # (phi = mod(arctan2(y, x), 2pi), as diffsky's lightcone_utils). Earlier versions
        # added pi instead, which rotated phi by 180 degrees.
        r = np.sqrt(np.sum([cores_step[x] ** 2 for x in "xyz"], axis=0))
        rhost = np.sqrt(np.sum([cores_step[f"host_{x}"] ** 2 for x in "xyz"], axis=0))
        cores_step["theta"] = np.arccos(cores_step["z"] / r)
        cores_step["phi"] = np.mod(
            np.arctan2(cores_step["y"], cores_step["x"]), 2 * np.pi
        )
        cores_step["host_theta"] = np.arccos(cores_step["host_z"] / rhost)
        cores_step["host_phi"] = np.mod(
            np.arctan2(cores_step["host_y"], cores_step["host_x"]), 2 * np.pi
        )
        cores_step["scale_factor"] = halo_lc["a"][lc_index]

        assert np.all(cores_step["theta"] >= 0)
        assert np.all(cores_step["theta"] <= np.pi)
        assert np.all(cores_step["phi"] >= 0)
        assert np.all(cores_step["phi"] <= 2 * np.pi)
        assert np.all(cores_step["host_theta"] >= 0)
        assert np.all(cores_step["host_theta"] <= np.pi)
        assert np.all(cores_step["host_phi"] >= 0)
        assert np.all(cores_step["host_phi"] <= 2 * np.pi)

        # float rounding can land exactly on 2pi; the partition wants [0, 2pi)
        cores_step["phi"] = np.fmod(cores_step["phi"], 2 * np.pi)
        cores_step["host_phi"] = np.fmod(cores_step["host_phi"], 2 * np.pi)

        ################################################################################
        # Redistribute cores on the lightcone, given by sphere segment
        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - distribute cores on LC", flush=True)
        # distribute cores by the angular position of the host halo
        cores_step = s2_distribute(
            partition_s2,
            cores_step,
            theta_key="host_theta",
            phi_key="host_phi",
        )

        ################################################################################
        # Sort cores by unique_id, with central core first in each group
        sort_key = np.lexsort((1 - cores_step["central"], cores_step["unique_id"]))
        cores_step = {k: v[sort_key] for k, v in cores_step.items()}

        # Build group index: offset and count for each unique_id
        unique_ids, group_offsets, group_counts = np.unique(
            cores_step["unique_id"], return_index=True, return_counts=True
        )
        n_no_central = int(np.sum(cores_step["central"][group_offsets] != 1))
        n_no_central_global = partition_cube.comm.reduce(
            n_no_central, op=MPI.SUM, root=0
        )
        n_groups_global = partition_cube.comm.reduce(
            len(group_offsets), op=MPI.SUM, root=0
        )
        if partition_cube.rank == 0:
            print(
                f"   - {n_no_central_global} / {n_groups_global} groups have no central core",
                flush=True,
            )

        ################################################################################
        # Get additional indices (Andrew)
        if partition_cube.rank == 0:
            print(" - calculate additional indices", flush=True)

        lightcone_dtype = np.dtype([("replication", np.int32), ("tag", np.int64)])
        unique_coretag = np.empty(len(cores_step["core_tag"]), dtype=lightcone_dtype)
        unique_coretag["replication"] = cores_step["replication"]
        unique_coretag["tag"] = cores_step["core_tag"]
        s = np.argsort(unique_coretag)
        assert len(unique_coretag) == len(np.unique(unique_coretag))
        # # make sure we don't have multiple replications on this rank...
        # assert np.all(np.diff(cores_step["core_tag"][s]) > 0)

        for idx_name in ["top_host_tag", "secondary_top_host_tag"]:
            partition_cube.comm.Barrier()
            if partition_cube.rank == 0:
                print(f"   - {idx_name}", flush=True)
            mask = cores_step[idx_name] >= 0
            count_target = int(np.sum(mask))
            # mask[~np.isin(cores_step[idx_name][mask], cores_step["core_tag"])] = False
            mask &= np.isin(cores_step[idx_name], cores_step["core_tag"])
            count_actual = int(np.sum(mask))
            count_lost = count_target - count_actual
            count_lost_global = partition_cube.comm.reduce(
                count_lost, op=MPI.SUM, root=0
            )
            if partition_cube.rank == 0:
                assert count_lost_global is not None
                if count_lost_global > 0:
                    print(
                        f"WARNING: LOST {count_lost_global} INDICES FOR {idx_name}",
                        flush=True,
                    )
            assert np.all(np.isin(cores_step[idx_name][mask], cores_step["core_tag"]))

            unique_coretag_host = np.empty(int(np.sum(mask)), dtype=lightcone_dtype)
            unique_coretag_host["replication"] = cores_step["replication"][mask]
            unique_coretag_host["tag"] = cores_step[idx_name][mask]

            idx = np.searchsorted(unique_coretag[s], unique_coretag_host)
            assert np.all(unique_coretag[s][idx] == unique_coretag_host)

            key = idx_name.replace("tag", "idx")
            _d = np.full(len(cores_step["core_tag"]), -1, dtype=np.int64)
            _d[mask] = s[idx]
            cores_step[key] = _d
            assert np.all(
                cores_step["core_tag"][_d[mask]] == cores_step[idx_name][mask]
            )
            assert np.all(
                cores_step["replication"][_d[mask]] == cores_step["replication"][mask]
            )

        ################################################################################
        # Write cores to HDF5
        if partition_cube.rank == 0:
            print(" - write HDF5", flush=True)

        # Rename fields
        cores_step["lc_halo_x"] = cores_step.pop("host_x")
        cores_step["lc_halo_y"] = cores_step.pop("host_y")
        cores_step["lc_halo_z"] = cores_step.pop("host_z")
        cores_step["lc_halo_phi"] = cores_step.pop("host_phi")
        cores_step["lc_halo_theta"] = cores_step.pop("host_theta")
        cores_step["lc_halo_foftag"] = cores_step.pop("fof_halo_tag_clean")
        # Write cores to file
        output_file = str(output_base) + f"-{step}.{partition_cube.rank}.hdf5"
        output_fields = core_fields + [
            "coreforest_file_idx",
            "coreforest_row_idx",
            "theta",
            "phi",
            "scale_factor",
            "lc_halo_x",
            "lc_halo_y",
            "lc_halo_z",
            "lc_halo_theta",
            "lc_halo_phi",
            "lc_fof_halo_tag",
            "lc_fof_halo_flag",
        ]
        output_fields += ["top_host_tag", "secondary_top_host_tag"]
        output_fields += ["top_host_idx", "secondary_top_host_idx"]
        # essential coreforest fields the reader adds (kept for backwards compatibility)
        output_fields += [
            k for k in ("host_core", "merged", "snapnum") if k in cores_step
        ]
        # diffmah fits of the core and its hosts, host infall shapes, infall indices
        output_fields += [k for k in diffmah_fields if k in cores_step]
        output_fields += [
            f"{h}_{k}"
            for h in ("top_host", "sec_host")
            for k in diffmah_host_fields
            if f"{h}_{k}" in cores_step
        ]
        output_fields += [
            f"top_host_{k}"
            for k in top_host_core_fields
            if f"top_host_{k}" in cores_step
        ]
        output_fields += ["indx_t_ult_inf", "indx_t_pen_inf"]
        with h5py.File(output_file, "w") as f:
            f.attrs["snapnum"] = snap_num
            f.attrs["theta_extent"] = partition_s2.theta_extent
            f.attrs["phi_extent"] = partition_s2.phi_extent
            grp = f.create_group("data")
            for k in output_fields:
                create_dataset(grp, k, cores_step[k], compression_level)
            grp = f.create_group("index")
            create_dataset(grp, "unique_id", unique_ids, compression_level)
            create_dataset(grp, "offset", group_offsets, compression_level)
            create_dataset(grp, "count", group_counts, compression_level)

        partition_s2.comm.Barrier()
