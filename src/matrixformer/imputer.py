"""Public matrix completion API for one pretrained MatrixFormer network."""

import json
from pathlib import Path

import numpy as np
import torch

from ._inference import Expert
from .artifacts import resolve_checkpoint


def prepare_matrix(matrix):
    matrix = np.asarray(matrix)
    if matrix.dtype.kind not in "iuf":
        raise ValueError("Expected a real numeric matrix with NaN marking missing values")
    if matrix.ndim != 2 or min(matrix.shape) == 0 or np.isinf(matrix).any():
        raise ValueError("Expected a nonempty 2D matrix with finite observations or NaN")
    if not np.isfinite(matrix).any():
        raise ValueError("At least one observed entry is required")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = matrix.astype(np.float64, copy=False)
        if matrix.dtype.kind in "iu" or matrix.dtype.itemsize > 8:
            restored = converted.astype(matrix.dtype, copy=False)
            if not np.array_equal(matrix, restored, equal_nan=True):
                raise ValueError("Observed entries must be exactly representable as float64")
    return converted


class MatrixFormer:
    """One frozen network trained on mixed synthetic matrix-completion tasks."""

    def __init__(self, model):
        self._model = model

    @classmethod
    def from_checkpoint(
        cls, checkpoint, *, config=None, device="cpu", precision="amp_bfloat16",
        max_rows=2000, max_cols=1000,
    ):
        """Load your own training output without downloading or claiming release identity.

        Config defaults to config.json beside the checkpoint. The architecture
        must use the packaged output bins. Unlike from_pretrained, local training
        artifacts are not checked against the published checkpoint hash.
        """
        target_device = validate_inference_options(device, precision, max_rows, max_cols)
        checkpoint = Path(checkpoint)
        config = Path(config) if config is not None else checkpoint.parent / "config.json"
        model_config = json.loads(config.read_text(encoding="utf-8"))["model"]
        if model_config.get("num_outputs") != 5000:
            raise ValueError("The checkpoint must use the packaged 5,000 output bins")
        model = Expert(checkpoint, model_config, device=target_device, precision=precision,
                       max_rows=max_rows, max_cols=max_cols)
        return cls(model)

    @classmethod
    def from_pretrained(
        cls, *, device="cpu", precision="amp_bfloat16", max_rows=2000,
        max_cols=1000, local_dir=None, cache_dir=None, local_files_only=False,
    ):
        """Load frozen weights from the pinned Hub release or an offline directory.

        Context limits are positive integers, or None for the full axis. Inputs
        are normalized per column; this is not the paper's rating-specific recipe.
        local_dir reads existing files only and never performs a network request.
        """
        target_device = validate_inference_options(device, precision, max_rows, max_cols)
        paths = resolve_checkpoint(
            local_dir=local_dir, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        config = json.loads(paths["config"].read_text(encoding="utf-8"))["model"]
        model = Expert(
            paths["checkpoint"], config, device=target_device, precision=precision,
            max_rows=max_rows, max_cols=max_cols,
        )
        return cls(model)

    def impute(self, matrix):
        """Complete a 2D numerical matrix, preserving every observed entry."""
        matrix = prepare_matrix(matrix)
        completed = self._model.impute(matrix.copy(order="K"))
        if not np.isfinite(completed).all():
            raise ValueError("Nonfinite predictions; check input magnitudes and scaling")
        return completed


def validate_inference_options(device, precision, max_rows, max_cols):
    if precision not in {"amp_bfloat16", "float32"}:
        raise ValueError("precision must be 'amp_bfloat16' or 'float32'")
    for name, limit in (("max_rows", max_rows), ("max_cols", max_cols)):
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are CPU and CUDA")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use device='cpu' or install a CUDA-enabled PyTorch")
        with torch.cuda.device(target_device):
            if precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support BF16; use precision='float32'")
    return target_device
