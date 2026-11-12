from typing import Mapping, Union, List, Optional

import h5py
import hdf5plugin  # noqa: F401  registers the Blosc2 filter of compressed coreforests
import numba
import numpy as np

from ..simulations import Simulation

# These fields will always be loaded from the HDF5 files
_essential_fields = ["core_tag", "host_core", "snapnum", "central", "merged"]


@numba.jit(nopython=True)
def _count_coreforest_rows(core_tag):
    count = 0
    prev_core_tag = -1
    ncores = len(core_tag)
    for i in range(ncores):
        if core_tag[i] != prev_core_tag:
            count += 1
            prev_core_tag = core_tag[i]
    return count


@numba.jit(nopython=True)
def _get_corematrix_row(core_tag, row_idx):
    _idx = -1
    prev_core_tag = -1
    ncores = len(core_tag)
    for i in range(ncores):
        if core_tag[i] != prev_core_tag:
            _idx += 1
            prev_core_tag = core_tag[i]
        row_idx[i] = _idx


@numba.jit(nopython=True, parallel=True)
def _get_top_host_row(host_row, top_host_row):
    for i in numba.prange(len(host_row)):
        # for i in range(len(host_row)):
        if host_row[i] < 0:
            continue
        _top_host_row = host_row[i]
        while host_row[_top_host_row] >= 0 and host_row[_top_host_row] != _top_host_row:
            _top_host_row = host_row[_top_host_row]
        top_host_row[i] = _top_host_row


@numba.jit(nopython=True, parallel=True)
def _get_two_top_host_rows(host_row, top_host_row, secondary_top_host_row):
    for i in numba.prange(len(host_row)):
        # for i in range(len(host_row)):
        if host_row[i] < 0:
            continue
        _top_host_row = host_row[i]
        _secondary_top_host_row = -1
        while host_row[_top_host_row] >= 0 and host_row[_top_host_row] != _top_host_row:
            _secondary_top_host_row = _top_host_row
            _top_host_row = host_row[_top_host_row]
        top_host_row[i] = _top_host_row
        secondary_top_host_row[i] = _secondary_top_host_row


def coreforest2matrix(
    forest: Mapping[str, np.ndarray],
    simulation: Simulation,
    *,
    calculate_host_rows: bool = True,
    calculate_secondary_host_row: bool = False,
):
    # first pass: count rows
    nrows = _count_coreforest_rows(forest["core_tag"])
    ncols = len(simulation.cosmotools_steps)

    forest_matrices = {
        k: np.zeros((nrows, ncols), dtype=d.dtype) for k, d in forest.items()
    }

    # second pass: fill in rows
    core_row_idx = np.empty_like(forest["core_tag"], dtype=np.int64)
    core_row_idx[:] = -1
    _get_corematrix_row(forest["core_tag"], core_row_idx)
    assert np.all(core_row_idx >= 0)

    # copy data to matrices
    core_idx = (core_row_idx, forest["snapnum"])
    for k, d in forest.items():
        forest_matrices[k][core_idx] = d

    # Look-up indices
    # TODO: find a more efficient way to find host_rows
    # (should at least parallelize at the root fof level)
    if calculate_host_rows:
        host_row = np.empty((nrows, ncols), dtype=np.int64)
        host_row[:] = -1
        top_host_row = np.empty((nrows, ncols), dtype=np.int64)
        top_host_row[:] = -1

        if calculate_secondary_host_row:
            secondary_top_host_row = np.empty((nrows, ncols), dtype=np.int64)
            secondary_top_host_row[:] = -1

        for s in range(ncols):
            _mask = forest_matrices["core_tag"][:, s] > 0
            assert np.all(forest_matrices["host_core"][_mask, s] > 0)

            core_tag_s = np.argsort(forest_matrices["core_tag"][:, s])
            core_tag_s = core_tag_s[_mask[core_tag_s]]
            core_tag_sorted = forest_matrices["core_tag"][core_tag_s, s]

            _host_row = np.searchsorted(
                core_tag_sorted, forest_matrices["host_core"][_mask, s]
            )
            assert np.all(
                core_tag_sorted[_host_row] == forest_matrices["host_core"][_mask, s]
            )
            host_row[_mask, s] = core_tag_s[_host_row]

            _top_host_row = np.empty_like(host_row[:, s])
            _top_host_row[:] = -1
            if not calculate_secondary_host_row:
                _get_top_host_row(host_row[:, s], _top_host_row)
            else:
                _secondary_top_host_row = np.empty_like(host_row[:, s])
                _secondary_top_host_row[:] = -1
                _get_two_top_host_rows(
                    host_row[:, s], _top_host_row, _secondary_top_host_row
                )
                secondary_top_host_row[:, s] = _secondary_top_host_row
            top_host_row[:, s] = _top_host_row
        forest_matrices["host_row"] = host_row
        forest_matrices["top_host_row"] = top_host_row
        if calculate_secondary_host_row:
            forest_matrices["secondary_top_host_row"] = secondary_top_host_row
    _state = np.empty((nrows, ncols), dtype=np.int16)
    _state[:] = -1
    _state[forest_matrices["core_tag"] > 0] = 1
    _state[forest_matrices["central"] == 1] = 0
    _state[forest_matrices["merged"] == 1] = 2
    forest_matrices["core_state"] = _state

    return forest_matrices


def corematrix_reader(
    filename: str,
    simulation: Union[Simulation, str],
    *,
    nchunks: Optional[int] = None,
    chunknum: Optional[int] = None,
    include_fields: Optional[List[str]] = None,
    calculate_host_rows: bool = True,
    calculate_secondary_host_row: bool = False,
):
    if isinstance(simulation, str):
        if simulation[-4:] == ".cfg":
            simulation = Simulation.parse_config(simulation)
        else:
            simulation = Simulation.simulations[simulation]

    # read index, find start and end of chunk
    with h5py.File(filename) as forest_file:
        ncores = len(forest_file["data"]["core_tag"][:])
        roots = forest_file["index"]["root_idx"][:]
        nroots = len(roots)
        file_end = ncores

    if nchunks is not None:
        chunksize = nroots // nchunks
        if chunknum is None:
            print("Warning: no chunknum specified, reading first chunk")
            chunknum = 0
        if chunknum >= nchunks:
            print(f"invalid chunknum: {chunknum} needs to be smaller than {nchunks}")
        startroot = chunknum * chunksize
        endroot = min(nroots, (chunknum + 1) * chunksize)
        start = roots[startroot]
        if chunknum == nchunks - 1:
            end = file_end
        else:
            end = roots[endroot]
        roots = roots[startroot:endroot]
        roots -= start
    else:
        start = roots[0]
        end = file_end

    # how many rows did we skip?
    if start > 0:
        with h5py.File(filename) as forest_file:
            core_tag = forest_file["data"]["core_tag"][:start]
        skipped_rows = _count_coreforest_rows(core_tag)
        del core_tag
    else:
        skipped_rows = 0

    # read data
    with h5py.File(filename) as forest_file:
        if include_fields is None:
            include_fields = list(forest_file["data"].keys())
        else:
            for k in _essential_fields:
                if k not in include_fields:
                    include_fields.append(k)
        forest_data = {k: forest_file["data"][k][start:end] for k in include_fields}

    # set host_core to itself for centrals
    forest_data["host_core"][forest_data["central"] == 1] = forest_data["core_tag"][
        forest_data["central"] == 1
    ]

    forest_matrices = coreforest2matrix(
        forest_data,
        simulation,
        calculate_host_rows=calculate_host_rows,
        calculate_secondary_host_row=calculate_secondary_host_row,
    )

    forest_matrices["absolute_row_idx"] = np.arange(
        skipped_rows, skipped_rows + len(forest_matrices["core_tag"])
    )
    return forest_matrices
