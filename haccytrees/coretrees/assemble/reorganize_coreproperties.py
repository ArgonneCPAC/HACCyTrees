from typing import Callable, List

import h5py
import numba
import numpy as np
import pygio
from mpi4py import MPI
from mpipartition import Partition, distribute, exchange

from ...simulations import Simulation
from .coretrees_assembly_config import CoretreesAssemblyConfig
from ...utils.datastores import NumpyStore
from ...utils.timer import Timer
from ...utils.hdf5_utils import create_dataset, COMPRESSION_CHOICES


def _read_data(
    partition: Partition,
    simulation: Simulation,
    coreprops_filename: str,
    read_fields: List[str],
    xyz_fields: List[str] = ["x", "y", "z"],
    logger: Callable = print,
    verbose: bool = True,
):
    with Timer(name="read coreproperties: read genericio", logger=logger):
        coreprops = pygio.read_genericio(
            coreprops_filename,
            read_fields,
            print_stats=verbose > 0,
        )
    # assert np.all(coreprops["fof_halo_tag"] >= 0)
    with Timer(name="read coreproperties: distribute", logger=logger):
        for x in xyz_fields:
            coreprops[x] = np.fmod(coreprops[x] + simulation.rl, simulation.rl)
        coreprops = distribute(
            partition, simulation.rl, coreprops, xyz_fields, verbose=verbose
        )
    return coreprops


@numba.jit(nopython=True, parallel=True)
def _subsort_centrals(coretag, central_start, central_size, host_idx, group_count):
    ncores = len(coretag)
    ncentrals = len(central_start)
    child_count = np.zeros(ncores, dtype=np.int32)
    my_direct_offset = np.zeros(ncores, dtype=np.int32)
    my_total_offset = np.zeros(ncores, dtype=np.int32)

    for i in numba.prange(ncentrals):
        _start = central_start[i]
        _size = central_size[i]
        _end = _start + _size
        # count size of each group
        for j in range(_start, _end):
            # self
            group_count[j] += 1
            # add to host recursively
            _idx = j
            while _idx != _start:
                _host = host_idx[_idx]
                group_count[_host] += 1
                _idx = _host

        # calculate offset of my group among other children of my host
        for j in range(_start + 1, _end):
            my_direct_offset[j] = child_count[host_idx[j]]
            child_count[host_idx[j]] += group_count[j]

        # calculate actual offset
        my_total_offset[_start] = _start
        for j in range(_start + 1, _end):
            my_offset = my_direct_offset[j]
            _idx = j
            while _idx != _start:
                _host = host_idx[_idx]
                my_offset += my_direct_offset[_host] + 1
                _idx = _host
            my_total_offset[j] = my_offset + _start

    return my_total_offset


def _sort_core_roots(coretag, hosttag, foftag, central, extensive_asserts=True):
    # replace hosttag of central with self (by default: 0)
    hosttag = np.copy(hosttag)
    hosttag[central > 0] = coretag[central > 0]
    if extensive_asserts:
        assert np.all(hosttag > 0)
        assert np.all(np.isin(hosttag, coretag))
        assert len(np.unique(foftag)) == np.sum(central)

    # find host index
    s_core = np.argsort(coretag)
    coretag_sorted = coretag[s_core]
    host_idx = np.searchsorted(coretag_sorted, hosttag)
    host_idx = s_core[host_idx]
    if extensive_asserts:
        assert np.all(coretag[host_idx] == hosttag)

    # primary sort: by foftag, with central first
    s_fof = np.lexsort([-central, foftag])
    central_start = np.nonzero(central[s_fof])[0]
    central_size = np.diff(np.append(central_start, len(central)))

    # host_idx for new ordering
    host_idx = np.argsort(s_fof)[host_idx][s_fof]
    if extensive_asserts:
        assert np.all(coretag[s_fof][host_idx] == hosttag[s_fof])

    group_count = np.zeros_like(coretag, dtype=np.int32)
    s = _subsort_centrals(
        coretag[s_fof], central_start, central_size, host_idx, group_count
    )
    if extensive_asserts:
        # check we have the correct sizes for the centrals
        assert np.all(group_count[central_start] == central_size)
        # check it's a permutation
        assert np.all(s >= 0)
        assert np.all(s < len(s))
        assert len(np.unique(s)) == len(s)

    return s_fof[s], host_idx[s], group_count[s]


def reorganize_coreproperties(
    partition: Partition,
    config: CoretreesAssemblyConfig,
):
    total_timer = Timer("total time", logger=None)
    total_timer.start()

    simulation = config.simulation
    steps = simulation.cosmotools_steps

    # Used to print status messages, only from rank 0
    def logger(x, **kwargs):
        kwargs.pop("flush", None)
        partition.comm.Barrier()
        if partition.rank == 0:
            print(x, flush=True, **kwargs)

    if config.compression not in COMPRESSION_CHOICES:
        raise ValueError(
            f"[output] compression = {config.compression!r}, use one of {COMPRESSION_CHOICES}"
        )
    comp = dict(method=config.compression, level=config.compression_level)

    read_fields = [k1 for k1, k2 in config.copy_fields]
    xyz_fields = [
        config.node_position_x,
        config.node_position_y,
        config.node_position_z,
    ]

    assert config.node_position_x in read_fields
    assert config.node_position_y in read_fields
    assert config.node_position_z in read_fields

    # process final step
    step = steps[-1]
    with Timer(name="read coreproperties", logger=logger):
        if "#" in config.input_base:
            coreprops_filename = config.input_base.replace("#", str(step))
        else:
            coreprops_filename = f"{config.input_base}{step}.coreproperties"
        coreprops = _read_data(
            partition=partition,
            simulation=simulation,
            coreprops_filename=coreprops_filename,
            read_fields=read_fields,
            xyz_fields=xyz_fields,
            logger=logger,
            verbose=config.verbose > 1,
        )

    with Timer(name="process final cores", logger=logger):
        tracked_dtypes = {k: d.dtype for k, d in coreprops.items()}
        central_mask = coreprops["central"] == 1
        fof_tags = np.sort(coreprops["fof_halo_tag"][central_mask])

    # communicate foreign cores
    with Timer(name="exchange foreign cores", logger=logger):
        mask_foreign_cores = np.isin(coreprops["fof_halo_tag"], fof_tags, invert=True)
        need_exchange_local = int(np.any(mask_foreign_cores))
        need_exchange_global = partition.comm.allreduce(need_exchange_local, op=MPI.MAX)

        if need_exchange_global:
            coreprops = exchange(
                partition,
                coreprops,
                "fof_halo_tag",
                fof_tags,
                filter_key=lambda idx: idx >= 0,
                verbose=True,
                do_all2all=False,
                replace_notfound_key=None,  # abort if no host found
            )

    # order by fof tag and centrals (for each fof tag, centrals come first)
    with Timer(name="sort cores", logger=logger):
        # s = np.argsort(coreprops["fof_halo_tag"])
        # s = np.lexsort([-coreprops["central"], coreprops["fof_halo_tag"]])
        s, host_idx, group_count = _sort_core_roots(
            coreprops["core_tag"],
            coreprops["host_core"],
            coreprops["fof_halo_tag"],
            coreprops["central"],
        )
        coreprops = {k: d[s] for k, d in coreprops.items()}
        root_idx = np.nonzero(coreprops["central"])[0]

        owned_core_tags = coreprops["core_tag"]
        owned_core_tags_count = len(owned_core_tags)
        owned_core_tags_length = np.ones_like(owned_core_tags, dtype=np.int64)
        owned_core_tags_order_idx = np.argsort(owned_core_tags)
        owned_core_tags_sorted = owned_core_tags[owned_core_tags_order_idx]

    with Timer("temp storage", logger=logger):
        npstore = NumpyStore(partition, temporary_path=f"{config.temporary_path}/tmp")

        coreprops["assign_idx"] = np.arange(owned_core_tags_count)
        npstore[step] = coreprops

    # now every rank has the host cores in its domain plus all its satellite cores
    # walk steps backwards
    for snapnum in range(len(steps) - 2, -1, -1):
        step = steps[snapnum]
        logger(f"\nSTEP {step:3d}\n--------\n")
        with Timer(name="read coreproperties", logger=logger):
            if "#" in config.input_base:
                coreprops_filename = config.input_base.replace("#", str(step))
            else:
                coreprops_filename = f"{config.input_base}{step}.coreproperties"
            coreprops = _read_data(
                partition=partition,
                simulation=simulation,
                coreprops_filename=coreprops_filename,
                read_fields=read_fields,
                xyz_fields=xyz_fields,
                logger=logger,
                verbose=config.verbose > 1,
            )
            new_core_tags_count = len(coreprops["core_tag"])
            new_core_tags_count_global = partition.comm.reduce(
                new_core_tags_count, op=MPI.SUM, root=0
            )

        # communicate foreign cores
        with Timer(name="exchange foreign cores", logger=logger):
            mask_foreign_cores = np.isin(
                coreprops["core_tag"], owned_core_tags, invert=True
            )
            mask_foreign_cores = np.isin(coreprops["fof_halo_tag"], fof_tags, invert=True)
            need_exchange_local = int(np.any(mask_foreign_cores))
            need_exchange_global = partition.comm.allreduce(need_exchange_local, op=MPI.MAX)

            if need_exchange_global:
                coreprops = exchange(
                    partition,
                    coreprops,
                    "core_tag",
                    owned_core_tags,
                    verbose=config.verbose > 1,
                    do_all2all=False,
                    replace_notfound_key=-1,
                )

        # discard orphan cores
        with Timer(name="discard orphan cores", logger=logger):
            # cores that don't belong to any rank will be marked as -1 in previous step
            mask_orphan_cores = coreprops["core_tag"] == -1
            count_orphan_cores = np.sum(mask_orphan_cores)
            count_orphan_cores_global = partition.comm.reduce(
                count_orphan_cores, op=MPI.SUM, root=0
            )
            coreprops = {k: d[~mask_orphan_cores] for k, d in coreprops.items()}
            if partition.rank == 0 and config.verbose:
                discarded_fraction = (
                    count_orphan_cores_global / new_core_tags_count_global
                )
                print(
                    f"Discarding {count_orphan_cores_global} orphan cores "
                    f"({100*discarded_fraction:.3f})%",
                    flush=True,
                )
        with Timer(name="sort cores", logger=logger):
            assert np.all(np.isin(coreprops["core_tag"], owned_core_tags_sorted))
            assign_idx = np.searchsorted(
                owned_core_tags_sorted,
                coreprops["core_tag"],
            )
            assert np.all(assign_idx >= 0)
            assert np.all(assign_idx < owned_core_tags_count)
            assign_idx = owned_core_tags_order_idx[assign_idx]
            assert np.all(owned_core_tags[assign_idx] == coreprops["core_tag"])
            owned_core_tags_length[assign_idx] += 1

        with Timer("temp storage", logger=logger):
            coreprops["assign_idx"] = assign_idx
            npstore[step] = coreprops

    if config.verbose:
        oct_global = partition.comm.gather(owned_core_tags_count, root=0)
        oct_sum_global = partition.comm.gather(np.sum(owned_core_tags_length), root=0)
        if partition.rank == 0:
            for i, (_oct_count, _oct_sum) in enumerate(zip(oct_global, oct_sum_global)):
                print(
                    f"Rank {i:3d}: {_oct_count:8d} cores tracked, "
                    f"{_oct_sum:8d} entries in total",
                    flush=True
                )
        partition.comm.Barrier()

    with Timer(name="write hdf5", logger=logger):
        own_core_offsets = np.insert(np.cumsum(owned_core_tags_length), 0, 0)
        nrows = int(np.sum(owned_core_tags_length))
        # output row of every stored entry, per step: computed once (the temporary store
        # is read once per field and step below, assign_idx only here)
        with Timer(name="write hdf5: row indices", logger=None):
            step_idx = []
            mask = np.zeros(nrows, dtype=np.bool_)
            for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
                step = steps[snapnum]
                _assign = np.asarray(npstore[step]["assign_idx"])
                _idx = own_core_offsets[_assign] + i
                assert np.all(_idx < own_core_offsets[_assign + 1])
                assert np.all(~mask[_idx])
                mask[_idx] = True
                step_idx.append(_idx)
                del _assign
            assert np.all(mask)
            del mask
        logger(f"write hdf5: {nrows} rows on rank 0, row indices done")
        with h5py.File(f"{config.output_base}.{partition.rank}.hdf5", "w") as f:
            g_data = f.create_group("data")
            for kin, kout in config.copy_fields:
                t0 = MPI.Wtime()
                data = np.empty(nrows, dtype=tracked_dtypes[kin])
                for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
                    data[step_idx[i]] = npstore[steps[snapnum]][kin]
                t1 = MPI.Wtime()
                create_dataset(g_data, kout, data, **comp)
                del data
                t2 = MPI.Wtime()
                if partition.rank == 0 and config.verbose:
                    print(f"  {kout}: gather {t1 - t0:.1f} s, write {t2 - t1:.1f} s", flush=True)

            data = np.empty(nrows, dtype=np.int16)
            for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
                data[step_idx[i]] = snapnum
            create_dataset(g_data, "snapnum", data, **comp)
            del data, step_idx

            g_idx = f.create_group("index")
            create_dataset(g_idx, "root_idx", own_core_offsets[root_idx], **comp)

    with Timer("cleanup", logger=logger):
        for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
            step = steps[snapnum]
            npstore.remove(step)

    total_timer.stop()

    if partition.rank == 0:
        print("\n----------------------------------------\n")
        print("Forest creation finished!\n")
        print("Final timers:")
        timer_names = list(Timer.timers.keys())
        timer_names.sort()
        maxtimerlen = max([len(n) for n in timer_names]) + 1
        for n, t in Timer.timers.items():
            print(f"{n:{maxtimerlen}s}: {t:4e}s")
    partition.comm.Barrier()
