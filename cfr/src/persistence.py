# src/persistence.py
from typing import Optional, Dict, Any, TypeAlias
import os
import logging
import pickle
import tempfile

# Assuming InfosetKey is properly imported elsewhere or defined if needed directly
# from .utils import InfosetKey

logger = logging.getLogger(__name__)

# Placeholder if InfosetKey isn't imported directly but used in type hints
# InfosetKey = Any

# Type alias for the reach probability sum dictionary used in CFR+
ReachProbDict: TypeAlias = Dict[Any, float]  # Use Any if InfosetKey not imported


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


def save_agent_data(data_to_save: Dict[str, Any], filepath: str) -> bool:
    """
    Saves the agent's learned data to a file.

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

        # Save the data
        import joblib
        joblib.dump(data_to_save, filepath)

        iteration = data_to_save.get("iteration", "N/A")
        logger.info("Agent data saved to %s at iteration %s", filepath, iteration)
        return True

    except (OSError, pickle.PicklingError, TypeError) as e:
        logger.error("Error saving agent data to %s: %s", filepath, e)
        return False


def load_agent_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads the agent's learned data from a file."""
    if not filepath or not isinstance(filepath, str):
        logger.error(
            "Cannot load agent data: Invalid filepath provided (received: %s).",
            filepath,
        )
        return None

    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            import joblib
            loaded_data = joblib.load(filepath)
            iteration = loaded_data.get("iteration", 0)
            # Basic validation: Check if expected keys exist
            if (
                "regret_sum" not in loaded_data
                or "strategy_sum" not in loaded_data
                or "reach_prob_sum" not in loaded_data
            ):
                logger.warning(
                    "Loaded data from %s missing expected keys. Check file integrity.",
                    filepath,
                )
            logger.info(
                "Agent data loaded from %s. Resuming from iteration %d.",
                filepath,
                iteration + 1,  # Log the iteration we are starting
            )
            # Return the whole dictionary
            return loaded_data
        if os.path.exists(filepath):  # File exists but is empty
            logger.warning(
                "Agent data file found at %s but is empty. Starting fresh.", filepath
            )
            return None
        else:  # File does not exist
            logger.info("Agent data file not found at %s. Starting fresh.", filepath)
            return None
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
    ) as e:
        logger.error("Error loading agent data from %s: %s", filepath, e)
        return None
