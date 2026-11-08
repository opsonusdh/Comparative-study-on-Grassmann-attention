"""Checkpoint save/load utilities for the refined LoopLM experiment.

The checkpoint format is trusted application data produced by this project.  The
loader therefore prioritizes compatibility with the project's full checkpoint
payload (optimizer state, Python RNG state, NumPy RNG state, etc.) while still
trying ``weights_only=True`` first when possible.

DirectML is supported explicitly.  DirectML is exposed by torch-directml as the
``privateuseone`` backend, and checkpoints are always deserialized on CPU before
the model is moved to the requested execution backend.
"""
from __future__ import annotations

import copy
import dataclasses
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .config import LoopConfig, is_grassmann_attention
from .model import LoopTransformer


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def resolve_device(device: Union[str, torch.device] = "cpu") -> torch.device:
    """Resolve cpu/cuda/xpu/directml/auto to a torch device.

    ``torch-directml`` returns a ``privateuseone:*`` device.  We intentionally
    keep that object intact rather than reconstructing it with ``torch.device``.
    """
    if isinstance(device, torch.device):
        return device

    spec = str(device).strip().lower()

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        try:
            import torch_directml  # type: ignore

            return torch_directml.device()
        except (ImportError, RuntimeError):
            return torch.device("cpu")

    if spec in {"dml", "directml", "privateuseone", "privateuseone:0"}:
        try:
            import torch_directml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DirectML requested, but torch-directml is not installed. "
                "Install it with: pip install torch-directml"
            ) from exc
        return torch_directml.device()

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if spec == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested, but torch.xpu.is_available() is False.")
        return torch.device("xpu")

    if spec == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, xpu, dml."
    )


def is_directml_device(device: Union[str, torch.device]) -> bool:
    """Return True for torch-directml's privateuseone backend."""
    try:
        resolved = resolve_device(device)
    except Exception:
        return str(device).strip().lower().startswith("privateuseone")
    return resolved.type == "privateuseone" or str(resolved).lower().startswith("privateuseone")


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def _capture_rng_state() -> Dict[str, Any]:
    """Capture all RNG streams used by training for exact continuation.

    NumPy's uint32 key array is stored as a plain Python list.  This avoids the
    ``torch.uint32`` safe-unpickling incompatibility seen with some Windows
    PyTorch releases while remaining fully backward compatible in the loader.
    """
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state().cpu(),
        "python": random.getstate(),
        "numpy": (
            np_name,
            np_keys.astype(np.uint32, copy=False).tolist(),
            int(np_pos),
            int(np_has_gauss),
            float(np_cached),
        ),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams captured by ``_capture_rng_state``."""
    if not state:
        return

    if "torch" in state:
        torch_state = state["torch"]
        if not torch.is_tensor(torch_state):
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
        else:
            torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(torch_state.contiguous())

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np_name, np_keys, np_pos, np_has_gauss, np_cached = state["numpy"]
        if torch.is_tensor(np_keys):
            # Legacy checkpoints from the earlier format stored this as
            # torch.uint32.  Convert it back to NumPy before restoring.
            np_keys = np_keys.detach().cpu().numpy()
        np.random.set_state(
            (
                np_name,
                np.asarray(np_keys, dtype=np.uint32),
                int(np_pos),
                int(np_has_gauss),
                float(np_cached),
            )
        )

    if "cuda" in state and torch.cuda.is_available():
        cuda_states = []
        for item in state["cuda"]:
            if not torch.is_tensor(item):
                item = torch.as_tensor(item, dtype=torch.uint8)
            else:
                item = item.detach().cpu().to(dtype=torch.uint8)
            cuda_states.append(item.contiguous())
        torch.cuda.set_rng_state_all(cuda_states)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move tensor-valued optimizer state onto the requested backend."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Union[str, Path],
    model: LoopTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save a complete LoopLM checkpoint."""
    payload: Dict[str, Any] = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dataclasses.asdict(model.cfg),
        "step": int(step),
        "extra": extra or {},
        "rng_state": _capture_rng_state(),
    }
    if optimizer is not None:
        # Optimizer state may live on CUDA/DML.  Save a CPU copy so the file is
        # backend-neutral and can be loaded on another machine/backend.
        # IMPORTANT: optimizer.state_dict() can contain references to the live
        # optimizer state tensors. Never mutate it in-place while preparing a
        # checkpoint, or checkpointing can silently move the live AdamW state
        # to CPU and make the next CUDA step fail with mixed devices.
        opt_state = copy.deepcopy(optimizer.state_dict())
        for state in opt_state.get("state", {}).values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.detach().cpu()
        payload["optimizer_state_dict"] = opt_state

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _load_serialized_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a trusted LoopLM checkpoint, with legacy compatibility.

    We first try the restricted PyTorch loader.  Some older project checkpoints
    contain ``torch.uint32`` and/or NumPy/Python objects that a given PyTorch
    build refuses under ``weights_only=True``.  These are trusted project-owned
    files, so we then fall back to the normal unpickler.  Importantly, we do NOT
    register dtype objects as safe globals: ``add_safe_globals`` expects
    classes/functions, which is the bug that caused the previous failure.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        warnings.warn(
            "Restricted checkpoint loading failed; loading this trusted project checkpoint "
            f"with weights_only=False. Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint on CPU, reconstruct the model, then move it to device.

    This CPU-first path is intentional for DirectML.  It also makes CUDA/XPU/DML
    checkpoint files portable because serialization never targets a backend that
    may not exist on the current machine.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    target_device = resolve_device(device)
    ckpt = _load_serialized_checkpoint(path)

    if "config" not in ckpt:
        raise KeyError(f"Checkpoint {path} does not contain a 'config' entry.")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {path} does not contain 'model_state_dict'.")

    raw_config = dict(ckpt["config"])
    # Before adaptive sparse-global offsets were introduced, grassmann-new
    # checkpoints stored only progressive local offsets. Preserve that exact
    # behavior when reading a checkpoint without the new scheme marker. The
    # trainer separately refuses to resume such a checkpoint under the new
    # adaptive experiment, but inspection/generation remains reproducible.
    if (
        raw_config.get("attention_type") == "grassmann-new"
        and "grassmann_offset_scheme" not in raw_config
    ):
        raw_config["grassmann_offset_scheme"] = "legacy_local_only_v0"
    valid_fields = {f.name for f in dataclasses.fields(LoopConfig)}
    ignored = sorted(set(raw_config) - valid_fields)
    if ignored:
        warnings.warn(
            "Ignoring retired/unknown checkpoint config fields: " + ", ".join(ignored),
            RuntimeWarning,
            stacklevel=2,
        )

    cfg = LoopConfig(**{k: v for k, v in raw_config.items() if k in valid_fields})

    # Always load weights into a CPU model first.  This avoids backend-specific
    # state-dict copy failures, especially on torch-directml/privateuseone.
    model = LoopTransformer(cfg)
    model_state = ckpt["model_state_dict"]
    if is_grassmann_attention(cfg.attention_type):
        # Older Grassmann checkpoints predate the learned absolute token
        # position table added from paper §3.1. Permit exactly that one missing
        # tensor; every other missing or unexpected key remains a hard error.
        incompatible = model.load_state_dict(model_state, strict=False)
        allowed_missing = {"grassmann_positional_embedding.weight"}
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing or unexpected:
            if missing == allowed_missing and not unexpected:
                warnings.warn(
                    "This Grassmann checkpoint predates learned absolute token positions. "
                    "The new grassmann_positional_embedding is freshly initialized; "
                    "its optimizer moments will also start fresh for that parameter.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "Checkpoint/model state mismatch. "
                    f"Missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}"
                )
    else:
        model.load_state_dict(model_state, strict=True)
    model = model.to(target_device)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, target_device)

    if restore_rng:
        _restore_rng_state(ckpt.get("rng_state", {}))

    ckpt["model"] = model
    ckpt["device"] = target_device
    return ckpt
