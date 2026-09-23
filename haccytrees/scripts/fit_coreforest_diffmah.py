"""Fit diffmah mass-history parameters for every core of a coreforest, one output file per
coreforest file with one row per coreforest matrix row (the order haccytrees'
corematrix_reader / absolute_row_idx uses, and the order diffsky reads them back in).

Output: <output-dir>/subvol_<i>_diffmah_fits.hdf5 with the columns of diffmah's own
fitter (logm0 logtc early_index late_index t_peak loss n_points_per_fit fit_algo) plus
fit_flag (bitmask, see haccytrees.diffmah) and n_iterations.

Devices: on a GPU node run one process per GPU/tile with the vendor's device mask
(CUDA_VISIBLE_DEVICES, ZE_AFFINITY_MASK, ROCR_VISIBLE_DEVICES) and --rank/--nranks
(auto-detected from PMI_RANK, PALS_RANKID, SLURM_PROCID) to split the subvolumes;
on CPUs use --device cpu --workers <cores>.
"""

import os
import time
from pathlib import Path

import click
import h5py
import numpy as np

from haccytrees import Simulation, coretrees
from haccytrees.diffmah import (
    DIFFMAH_FIELDS,
    EXTRA_FIELDS,
    FitConfig,
    cosmic_time,
    default_cpu_config,
    fit_mahs,
    fit_mahs_parallel,
    mah_from_mass,
    set_env,
    cpu_environment,
)


def _auto_rank():
    env = os.environ
    rank = int(
        env.get("PMI_RANK", env.get("PALS_RANKID", env.get("SLURM_PROCID", "0")))
    )
    size = int(
        env.get("PMI_SIZE", env.get("PALS_LOCAL_SIZE", env.get("SLURM_NTASKS", "1")))
    )
    return rank, size


def _parse_subvols(spec: str, nfiles: int) -> list[int]:
    if spec == "all":
        return list(range(nfiles))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


@click.command()
@click.argument("forest_base", type=str)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--simulation", required=True, help="haccytrees simulation name or config"
)
@click.option("--subvolumes", default="all", show_default=True, help="e.g. 0-19,40")
@click.option(
    "--nchunks", default=20, show_default=True, help="root-aligned chunks per file"
)
@click.option(
    "--device", type=click.Choice(["gpu", "cpu"]), default="gpu", show_default=True
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    help="CPU worker processes (--device cpu)",
)
@click.option(
    "--dtype",
    type=click.Choice(["float64", "float32"]),
    default="float64",
    show_default=True,
)
@click.option(
    "--maxiter", default=None, type=int, help="iteration cap [gpu: 300, cpu: 100]"
)
@click.option(
    "--batch",
    default=None,
    type=int,
    help="cores per optimizer call [gpu: 65536, cpu: 256]",
)
@click.option(
    "--rank",
    default=None,
    type=int,
    help="this process' index (auto from PMI/PALS/SLURM)",
)
@click.option("--nranks", default=None, type=int)
@click.option("--mass-field", default="infall_tree_node_mass", show_default=True)
@click.option("--overwrite", is_flag=True)
def cli(
    forest_base,
    output_dir,
    simulation,
    subvolumes,
    nchunks,
    device,
    workers,
    dtype,
    maxiter,
    batch,
    rank,
    nranks,
    mass_field,
    overwrite,
):
    sim = (
        Simulation.simulations[simulation]
        if simulation in Simulation.simulations
        else Simulation.parse_config(simulation)
    )
    tarr = cosmic_time(sim)
    if device == "cpu":
        set_env(cpu_environment())
        cfg = default_cpu_config(
            dtype=dtype,
            **({"maxiter": maxiter} if maxiter else {}),
            **({"batch": batch} if batch else {}),
        )
    else:
        cfg = FitConfig(dtype=dtype, maxiter=maxiter or 300, batch=batch or 65536)
    r0, n0 = _auto_rank()
    rank = r0 if rank is None else rank
    nranks = n0 if nranks is None else nranks

    nfiles = 0
    while Path(f"{forest_base}.{nfiles}.hdf5").exists():
        nfiles += 1
    subvols = _parse_subvols(subvolumes, nfiles)[rank::nranks]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"rank {rank}/{nranks}: {len(subvols)} subvolumes, device {device}, {cfg}",
        flush=True,
    )

    for i in subvols:
        fn = f"{forest_base}.{i}.hdf5"
        out = output_dir / f"subvol_{i}_diffmah_fits.hdf5"
        if out.exists() and not overwrite:
            print(f"  subvol {i}: {out} exists, skipping", flush=True)
            continue
        t0 = time.time()
        parts = []
        nrows = 0
        for c in range(nchunks):
            fm = coretrees.corematrix_reader(
                fn,
                sim,
                nchunks=nchunks,
                chunknum=c,
                include_fields=[mass_field],
                calculate_host_rows=False,
            )
            rows = fm["absolute_row_idx"]
            if len(rows) == 0:  # fewer roots than chunks: early chunks are empty
                continue
            assert rows[0] == nrows, f"non-contiguous chunk {c}: {rows[0]} != {nrows}"
            mahs = mah_from_mass(fm[mass_field], sim.cosmo.h)
            del fm
            res = (
                fit_mahs_parallel(tarr, mahs, cfg, workers)
                if device == "cpu"
                else fit_mahs(tarr, mahs, cfg)
            )
            parts.append(res)
            nrows += len(rows)
            print(
                f"  subvol {i} chunk {c}: {len(rows)} cores, {time.time() - t0:.0f}s elapsed",
                flush=True,
            )
        if not parts:
            print(f"  subvol {i}: no cores, skipping", flush=True)
            continue
        res = {
            k: np.concatenate([p[k] for p in parts])
            for k in DIFFMAH_FIELDS + EXTRA_FIELDS
        }
        tmp = out.with_suffix(".tmp.hdf5")
        with h5py.File(tmp, "w") as f:
            for k, v in res.items():
                f.create_dataset(k, data=v)
            f.attrs["forest_file"] = fn
            f.attrs["simulation"] = sim.name
            f.attrs["config"] = str(cfg)
            f.attrs["device"] = device
        tmp.rename(out)
        print(
            f"  subvol {i}: {nrows} cores in {time.time() - t0:.0f}s -> {out}",
            flush=True,
        )


if __name__ == "__main__":
    cli()
