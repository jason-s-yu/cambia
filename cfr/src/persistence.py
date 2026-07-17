# src/persistence.py
from typing import Optional, Dict, Any, List, Tuple, TypeAlias
import os
import json
import logging
import tempfile

import numpy as np

# Assuming InfosetKey is properly imported elsewhere or defined if needed directly
# from .utils import InfosetKey

logger = logging.getLogger(__name__)

# Placeholder if InfosetKey isn't imported directly but used in type hints
# InfosetKey = Any

# Type alias for the reach probability sum dictionary used in CFR+
ReachProbDict: TypeAlias = Dict[Any, float]  # Use Any if InfosetKey not imported

# On-disk format for tabular-CFR agent data. Previously a joblib pickle, which
# executes arbitrary code on load and was the rsync-into-runs RCE vector
# (cambia-552). The current format is a numpy ``.npz`` archive: keys are stored
# as a JSON blob, ragged regret/strategy arrays as a flat float64 buffer plus
# int64 row offsets, reach-prob sums as a parallel float64 array. Loading uses
# ``allow_pickle=False`` so no pickle opcode ever runs. Tabular CFR is legacy;
# there is no back-compat reader for old joblib artifacts (reading them would
# require the unrestricted pickle this hardening removes).
_AGENT_DATA_FORMAT_VERSION = 1

# numpy ``.npz`` files are ZIP archives; every ZIP local-file header starts with
# this magic. Legacy joblib/pickle artifacts do not, so it cleanly discriminates
# a hardened file from a poisoned pickle without unpickling anything.
_NPZ_MAGIC = b"PK\x03\x04"


def atomic_torch_save(obj: Any, path: str) -> None:
    """
    Save a PyTorch object atomically by writing to a temp file then renaming.

    Raises:
        OSError: If the write or rename fails.
    """
    import torch

    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_npz_save(save_fn, path: str) -> None:
    """
    Save a numpy .npz file atomically using a write-then-rename pattern.

    ``save_fn`` is a callable that accepts a file path and writes the .npz file
    there (e.g. ``buffer.save``). The extension ``.npz`` is appended by numpy
    automatically, so we operate on the base path and rename the resulting file.

    Raises:
        OSError: If the write or rename fails.
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_base = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    os.close(fd)
    os.unlink(tmp_base)  # Remove so numpy can write <tmp_base>.npz
    try:
        save_fn(tmp_base)
        tmp_npz = tmp_base + ".npz"
        target_npz = path if path.endswith(".npz") else path + ".npz"
        os.replace(tmp_npz, target_npz)
    except Exception:
        for candidate in [tmp_base + ".npz", tmp_base]:
            try:
                os.unlink(candidate)
            except OSError:
                pass
        raise


def _key_to_jsonable(key: Any) -> List[Any]:
    """Convert an infoset key (``InfosetKey.astuple()`` shape or a plain tuple)
    to a JSON-serializable list. Nested tuple fields (own-hand / opp-belief)
    become lists of ints; scalar fields become ints."""
    out: List[Any] = []
    for elem in key:
        if isinstance(elem, (tuple, list, np.ndarray)):
            out.append([int(x) for x in elem])
        else:
            out.append(int(elem))
    return out


def _jsonable_to_key(item: Any) -> Tuple[Any, ...]:
    """Inverse of ``_key_to_jsonable``: rebuild a hashable tuple key whose nested
    fields are tuples, so ``InfosetKey(*key)`` reconstructs a valid, hashable
    key downstream."""
    return tuple(
        tuple(int(x) for x in elem) if isinstance(elem, list) else int(elem)
        for elem in item
    )


def _encode_ragged(mapping: Dict[Any, Any]) -> Tuple[str, np.ndarray, np.ndarray]:
    """Flatten a dict of ``key -> 1-D float array`` into
    ``(keys_json, flat, offsets)``. ``flat`` concatenates every value; ``offsets``
    has length ``len(keys) + 1`` marking row boundaries, so value ``i`` is
    ``flat[offsets[i]:offsets[i + 1]]``."""
    keys = list(mapping.keys())
    arrays = [np.asarray(mapping[k], dtype=np.float64).ravel() for k in keys]
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    if keys:
        lengths = np.array([a.size for a in arrays], dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)
    else:
        flat = np.zeros(0, dtype=np.float64)
    keys_json = json.dumps([_key_to_jsonable(k) for k in keys])
    return keys_json, flat, offsets


def _decode_ragged(
    keys_json: str, flat: np.ndarray, offsets: np.ndarray
) -> Dict[Tuple[Any, ...], np.ndarray]:
    keys = json.loads(keys_json)
    flat = np.asarray(flat, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    out: Dict[Tuple[Any, ...], np.ndarray] = {}
    for i, raw_key in enumerate(keys):
        out[_jsonable_to_key(raw_key)] = flat[offsets[i] : offsets[i + 1]].copy()
    return out


def _encode_scalar_map(mapping: Dict[Any, Any]) -> Tuple[str, np.ndarray]:
    """Flatten a dict of ``key -> float`` into ``(keys_json, values)``."""
    keys = list(mapping.keys())
    values = np.array([float(mapping[k]) for k in keys], dtype=np.float64)
    keys_json = json.dumps([_key_to_jsonable(k) for k in keys])
    return keys_json, values


def _decode_scalar_map(
    keys_json: str, values: np.ndarray
) -> Dict[Tuple[Any, ...], float]:
    keys = json.loads(keys_json)
    values = np.asarray(values, dtype=np.float64)
    return {_jsonable_to_key(k): float(values[i]) for i, k in enumerate(keys)}


def _npz_scalar_str(arr: np.ndarray) -> str:
    """Read a 0-d unicode array (JSON blob) back to a Python str."""
    return str(np.asarray(arr).item())


def save_agent_data(data_to_save: Dict[str, Any], filepath: str) -> bool:
    """
    Saves the agent's learned data to ``filepath`` as a numpy ``.npz`` archive.

    The archive contains no pickled Python objects: keys are JSON, values are
    numeric arrays. The written path is exactly ``filepath`` (the archive bytes
    are streamed to an open handle, bypassing numpy's automatic ``.npz`` suffix)
    so save and load agree on the path regardless of its extension.

    Returns:
        bool: True if save succeeded, False otherwise.
    """
    if not filepath or not isinstance(filepath, str):
        logger.error(
            "Cannot save agent data: Invalid filepath provided (received: %s).",
            filepath,
        )
        return False

    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(filepath)
        # Handle case where filepath is just a filename (dirname is '')
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        regret_keys, regret_flat, regret_offsets = _encode_ragged(
            data_to_save.get("regret_sum", {})
        )
        strategy_keys, strategy_flat, strategy_offsets = _encode_ragged(
            data_to_save.get("strategy_sum", {})
        )
        reach_keys, reach_values = _encode_scalar_map(
            data_to_save.get("reach_prob_sum", {})
        )
        iteration = int(data_to_save.get("iteration", 0))
        exploit = data_to_save.get("exploitability_results", []) or []
        exploit_json = json.dumps([[int(it), float(val)] for it, val in exploit])

        fields = {
            "format_version": np.array([_AGENT_DATA_FORMAT_VERSION], dtype=np.int64),
            "iteration": np.array([iteration], dtype=np.int64),
            "regret_keys": np.array(regret_keys),
            "regret_flat": regret_flat,
            "regret_offsets": regret_offsets,
            "strategy_keys": np.array(strategy_keys),
            "strategy_flat": strategy_flat,
            "strategy_offsets": strategy_offsets,
            "reach_prob_keys": np.array(reach_keys),
            "reach_prob_values": reach_values,
            "exploitability_results": np.array(exploit_json),
        }

        # Atomic write: stream the archive to a temp handle, then rename. Passing
        # an open file object (not a path) makes numpy write the exact bytes with
        # no ``.npz`` suffix rewriting, so the final path == ``filepath``.
        dir_path = parent_dir or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(fh, **fields)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info("Agent data saved to %s at iteration %s", filepath, iteration)
        return True

    except (OSError, TypeError, ValueError) as e:
        logger.error("Error saving agent data to %s: %s", filepath, e)
        return False


def load_agent_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads the agent's learned data from a hardened ``.npz`` archive.

    Legacy joblib/pickle artifacts are rejected with a clear error rather than
    unpickled: reading them would require the unrestricted pickle this hardening
    removes (cambia-552). Tabular CFR is legacy; retrain or re-export to migrate.

    Returns the loaded dict, or None when the file is absent or empty (fresh
    start). Raises ``ValueError`` on a legacy/foreign-format file.
    """
    if not filepath or not isinstance(filepath, str):
        logger.error(
            "Cannot load agent data: Invalid filepath provided (received: %s).",
            filepath,
        )
        return None

    if not os.path.exists(filepath):
        logger.info("Agent data file not found at %s. Starting fresh.", filepath)
        return None
    if os.path.getsize(filepath) == 0:
        logger.warning(
            "Agent data file found at %s but is empty. Starting fresh.", filepath
        )
        return None

    with open(filepath, "rb") as fh:
        magic = fh.read(4)
    if magic != _NPZ_MAGIC:
        raise ValueError(
            f"Agent data file {filepath!r} is not in the hardened npz format. "
            "Legacy joblib/pickle tabular checkpoints are no longer loadable "
            "(security hardening, cambia-552): reading them would require "
            "unrestricted pickle. Retrain or re-export; tabular CFR is legacy, "
            "prefer the deep PRT-CFR pipeline."
        )

    try:
        with open(filepath, "rb") as fh:
            # allow_pickle=False: no pickle opcode ever executes, even on a
            # crafted archive.
            with np.load(fh, allow_pickle=False) as npz:
                version = (
                    int(npz["format_version"][0]) if "format_version" in npz else 0
                )
                if version != _AGENT_DATA_FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported agent data format version {version} in "
                        f"{filepath!r} (expected {_AGENT_DATA_FORMAT_VERSION})."
                    )
                regret_sum = _decode_ragged(
                    _npz_scalar_str(npz["regret_keys"]),
                    npz["regret_flat"],
                    npz["regret_offsets"],
                )
                strategy_sum = _decode_ragged(
                    _npz_scalar_str(npz["strategy_keys"]),
                    npz["strategy_flat"],
                    npz["strategy_offsets"],
                )
                reach_prob_sum = _decode_scalar_map(
                    _npz_scalar_str(npz["reach_prob_keys"]),
                    npz["reach_prob_values"],
                )
                iteration = int(npz["iteration"][0])
                exploit_raw = json.loads(_npz_scalar_str(npz["exploitability_results"]))

        exploitability_results = [(int(it), float(val)) for it, val in exploit_raw]
        loaded_data: Dict[str, Any] = {
            "regret_sum": regret_sum,
            "strategy_sum": strategy_sum,
            "reach_prob_sum": reach_prob_sum,
            "iteration": iteration,
            "exploitability_results": exploitability_results,
        }
        logger.info(
            "Agent data loaded from %s. Resuming from iteration %d.",
            filepath,
            iteration + 1,  # Log the iteration we are starting
        )
        return loaded_data
    except (OSError, EOFError, KeyError, ValueError) as e:
        logger.error("Error loading agent data from %s: %s", filepath, e)
        return None
