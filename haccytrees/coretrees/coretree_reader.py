from typing import Mapping, Union, List

import h5py
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


def coreforest2matrix(forest: Mapping[str, np.ndarray], simulation: Simulation):
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
    host_row = np.empty((nrows, ncols), dtype=np.int64)
    host_row[:] = -1
    top_host_row = np.empty((nrows, ncols), dtype=np.int64)
    top_host_row[:] = -1
    for s in range(ncols):
        core_tag_s = np.argsort(forest_matrices["core_tag"][:, s])
        core_tag_s = core_tag_s[forest_matrices["core_tag"][core_tag_s, s] > 0]
        core_tag_sorted = forest_matrices["core_tag"][core_tag_s, s]
        _mask = forest_matrices["host_core"][:, s] > 0
        _host_row = np.searchsorted(
            core_tag_sorted, forest_matrices["host_core"][_mask, s]
        )
        assert np.all(
            core_tag_sorted[_host_row] == forest_matrices["host_core"][_mask, s]
        )
        host_row[_mask, s] = core_tag_s[host_row[_mask, s]]
        _top_host_row = np.empty_like(host_row[:, s])
        _top_host_row[:] = -1
        _get_top_host_row(host_row[:, s], _top_host_row)
        top_host_row[:, s] = _top_host_row
    forest_matrices["host_row"] = host_row
    forest_matrices["top_host_row"] = top_host_row
    _state = np.empty((nrows, ncols), dtype=np.int16)
    _state[:] = -1
    _state[forest_matrices["core_tag"] > 0] = 1
    _state[forest_matrices["central"] == 1] = 0
    _state[forest_matrices["merged"] == 1] = 2
    forest_matrices["core_state"] = _state

    return forest_matrices


def corematrix_reader(
    filename: str, simulation: Union[Simulation, str], include_fields: List[str] = None
):
    if isinstance(simulation, str):
        if simulation[:-4] == ".cfg":
            simulation = Simulation.parse_config(simulation)
        else:
            simulation = Simulation.simulations[simulation]

    with h5py.File(filename) as forest_file:
        if include_fields is None:
            include_fields = list(forest_file["data"].keys())
        else:
            for k in _essential_fields:
                if k not in include_fields:
                    include_fields.append(k)
        forest_data = {k: forest_file["data"][k][:] for k in include_fields}

    # set host_core to itself for centrals
    forest_data["host_core"][forest_data["central"] == 1] = forest_data["core_tag"][
        forest_data["central"] == 1
    ]

    forest_matrices = coreforest2matrix(forest_data, simulation)
    return forest_matrices