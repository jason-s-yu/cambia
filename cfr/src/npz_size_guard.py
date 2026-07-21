"""
src/npz_size_guard.py

Decompression-bomb size guard for ``np.load`` of ``runs/``-directory
artifacts (cambia-559, follow-up to the cambia-552 RCE hardening).

Threat model: an attacker with rsync write access to ``runs/`` cannot execute
code via a crafted ``.npz`` (every guarded load site already passes
``allow_pickle=False``, so no pickle opcode ever runs), but can still crash
the runner. An ``.npz`` archive is a zip of ``.npy`` members; each member's
header declares a shape and dtype, and ``numpy.lib.format.read_array``
allocates the array's *full declared size* (``numpy.ndarray(count, dtype)``)
before reading a single byte of data off disk. A member whose header claims a
multi-GB shape but whose actual data is truncated/empty is therefore only a
few KB on disk, yet OOM-kills the process the moment it is opened, regardless
of whether the read that follows would have failed.

Defense: read every member's ``.npy`` header (magic + shape/dtype) directly
off the zip entry, without ever letting numpy allocate or read the array
body, and reject the archive before any member is materialized if declared
sizes are implausible for a legitimate artifact.
"""

import zipfile
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib import format as npy_format

from .cfr.exceptions import NpzSizeGuardError

PathLike = Union[str, Path]

# Largest legitimate single-archive save observed across every ReservoirBuffer
# configuration in cfr/config (src/config.py): capacity up to 2,000,000 rows,
# combined with the largest dims actually used together -- deep_cfr's
# advantage/strategy buffers (INPUT_DIM=222, NUM_ACTIONS=146, has_mask=True) --
# work out to float32 features + targets + bool masks + int64 iterations
# totalling ~3.1 GiB; DESCA's value buffer (input_dim=377) and regret/strategy
# buffers (input_dim=257, target_dim=32) top out lower, ~2.2-2.9 GiB. 32 GiB
# is roughly an order of magnitude above that observed max, leaving headroom
# for capacity growth while still failing loudly well short of exhausting a
# training host's RAM.
RUNS_NPZ_MAX_DECLARED_BYTES = 32 * 1024**3

# Secondary check: a member whose declared size exceeds what DEFLATE can
# possibly inflate from its on-disk compressed footprint must have
# truncated/missing data -- the signature of a bomb whose header alone drives
# np.load's allocation. DEFLATE's compression ratio is bounded at ~1032:1
# (each symbol emits at most 258 bytes from ~2 bits), so a fully-present
# member -- even a degenerate-but-real one like a constant-valued int64
# iterations column -- can never exceed ~1032x; 2048x therefore has zero
# false positives on archives whose data actually exists. Only applied above
# a floor so tiny scalar/metadata members (meta, has_mask, input_dim, ...)
# are never flagged.
_RATIO_CHECK_MIN_DECLARED_BYTES = 1 * 1024 * 1024
_MAX_DECLARED_TO_COMPRESSED_RATIO = 2048


def _read_member_declared_bytes(member, label: str, member_name: str) -> int:
    """Read one .npy member's header and return its declared byte count,
    without reading (or allocating) any array data."""
    version = npy_format.read_magic(member)
    if version == (1, 0):
        shape, _fortran_order, dtype = npy_format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, _fortran_order, dtype = npy_format.read_array_header_2_0(member)
    else:
        # Plain float32/bool/int64 reservoir and agent-data arrays never need
        # format 3.0 (reserved for non-latin1 structured-dtype field names) or
        # beyond. Fail closed rather than skip a header we cannot parse
        # safely.
        raise NpzSizeGuardError(
            f"{label}: archive member {member_name!r} uses unsupported .npy "
            f"format version {version}; refusing to load without validating "
            "its declared size (cambia-559)."
        )
    try:
        count = int(np.multiply.reduce(shape, dtype=np.int64)) if shape else 1
        declared = count * dtype.itemsize
    except (OverflowError, ValueError) as e:
        # Header shape fields are attacker-controlled; a dimension too large
        # for int64 conversion is itself a bomb signature.
        raise NpzSizeGuardError(
            f"{label}: archive member {member_name!r} declares shape {shape!r} "
            f"whose size cannot be computed ({e}); refusing to load as a "
            "likely decompression bomb (cambia-559)."
        ) from e
    if declared < 0:
        raise NpzSizeGuardError(
            f"{label}: archive member {member_name!r} declares shape {shape!r} "
            "whose byte size overflows; refusing to load as a likely "
            "decompression bomb (cambia-559)."
        )
    return declared


def guard_npz_size(path: PathLike, *, label: str = "") -> None:
    """
    Validate every member of an ``.npz`` archive's declared array size BEFORE
    any array is materialized.

    Raises ``NpzSizeGuardError`` (never partially loads) if:
      - any member's declared size is wildly disproportionate to its on-disk
        compressed size (likely a truncated/empty bomb payload), or
      - the sum of all members' declared sizes exceeds
        ``RUNS_NPZ_MAX_DECLARED_BYTES``.

    ``path`` must be a real file path (a zip central directory is required to
    read compressed sizes without inflating each member).
    """
    label = label or str(path)
    total_declared = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info.filename) as member:
                declared_bytes = _read_member_declared_bytes(
                    member, label, info.filename
                )
            total_declared += declared_bytes

            if (
                declared_bytes >= _RATIO_CHECK_MIN_DECLARED_BYTES
                and declared_bytes
                > info.compress_size * _MAX_DECLARED_TO_COMPRESSED_RATIO
            ):
                ratio = declared_bytes / max(info.compress_size, 1)
                raise NpzSizeGuardError(
                    f"{label}: archive member {info.filename!r} declares "
                    f"{declared_bytes} bytes but is only {info.compress_size} "
                    f"bytes compressed on disk (ratio {ratio:.0f}x exceeds the "
                    f"{_MAX_DECLARED_TO_COMPRESSED_RATIO}x cap); refusing to "
                    "load as a likely decompression bomb (cambia-559)."
                )

        if total_declared > RUNS_NPZ_MAX_DECLARED_BYTES:
            raise NpzSizeGuardError(
                f"{label}: archive declares {total_declared} bytes of array "
                f"data across all members, exceeding the "
                f"{RUNS_NPZ_MAX_DECLARED_BYTES}-byte size-guard cap; refusing "
                "to load as a likely decompression bomb (cambia-559)."
            )
