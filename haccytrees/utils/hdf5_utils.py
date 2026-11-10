"""Compressed HDF5 datasets, with the production setting of reformat-hacc-data.

Blosc2 (through hdf5plugin) with zstd at level 1, byte shuffle, Fletcher32 checksums and
chunks of about 4 MiB along the first axis. BLOSC_NTHREADS is set to 1 before writing, so that
the stored bytes do not depend on the job environment (with more threads the blocks of a chunk
are stored in the order the threads finish). Readers of such files need ``import hdf5plugin``.

hdf5plugin is imported only when a compressed dataset is written.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

CHUNK_BYTES = 4 * 1024 * 1024
CHUNK_MULTIPLE = 1024  # chunk sizes are multiples of 1 KiB
COMPRESSION_CHOICES = ("none", "gzip", "blosc2-zstd", "blosc2-lz4")


def chunk_shape(shape: tuple[int, ...], dtype) -> tuple[int, ...] | None:
    """About CHUNK_BYTES per chunk along the first axis; None for empty datasets."""
    if shape[0] == 0:
        return None
    elem = int(np.dtype(dtype).itemsize * int(np.prod(shape[1:], dtype=np.int64)))
    align = math.lcm(elem, CHUNK_MULTIPLE)
    n = max((CHUNK_BYTES // elem) // align, 1) * align
    n = min(n, shape[0])
    return (n,) + tuple(shape[1:])


def compression_kwargs(
    method: str = "blosc2-zstd", level: int = 1, shape=None, dtype=None
) -> dict[str, Any]:
    """Keyword arguments for h5py ``create_dataset``.

    method: "none", "gzip" (with byte shuffle), "blosc2-zstd" or "blosc2-lz4" (byte shuffle).
    Empty datasets cannot be chunked and are written plain.
    """
    if method not in COMPRESSION_CHOICES:
        raise ValueError(f"unknown compression {method!r}, use one of {COMPRESSION_CHOICES}")
    if method == "none" or (shape is not None and shape[0] == 0):
        return {}
    kw: dict[str, Any] = {"fletcher32": True}
    if shape is not None:
        kw["chunks"] = chunk_shape(tuple(shape), dtype)
    else:
        kw["chunks"] = True
    if method == "gzip":
        kw.update(compression="gzip", compression_opts=level, shuffle=True)
        return kw
    import hdf5plugin

    os.environ["BLOSC_NTHREADS"] = "1"
    cname = method.split("-", 1)[1]
    kw["compression"] = hdf5plugin.Blosc2(
        cname=cname, clevel=level, filters=hdf5plugin.Blosc2.SHUFFLE
    )
    return kw


def create_dataset(group, name: str, data: np.ndarray, method="none", level=1):
    """``group.create_dataset(name, data=data)`` with the given compression."""
    return group.create_dataset(
        name, data=data, **compression_kwargs(method, level, data.shape, data.dtype)
    )
