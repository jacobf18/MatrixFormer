"""Public matrix completion API and the observed-error convex blend."""

import json

import numpy as np
import torch

from ._inference import Expert
from .artifacts import resolve_experts


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


def observed_scales(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    observed = np.isfinite(matrix)
    counts = observed.sum(axis=0)
    means = np.nansum(matrix, axis=0) / np.maximum(counts, 1)
    residuals = np.where(observed, matrix - means, 0.0)
    scales = np.sqrt(np.sum(residuals ** 2, axis=0) / np.maximum(counts, 1))
    return np.where(scales > 1e-8, scales, 1.0)


def fit_blend(matrix, general, specialist):
    observed = np.isfinite(matrix)
    target = np.asarray(matrix, dtype=np.float64)
    scales = observed_scales(target)
    general_error = ((general - target) / scales)[observed]
    specialist_error = ((specialist - target) / scales)[observed]
    difference = specialist_error - general_error
    denominator = float(np.dot(difference, difference))
    weight = 0.0
    if denominator > 0:
        weight = float(np.clip(-np.dot(general_error, difference) / denominator, 0.0, 1.0))
    general_mse = float(np.mean(general_error ** 2))
    specialist_mse = float(np.mean(specialist_error ** 2))
    if not np.isfinite([weight, general_mse, specialist_mse]).all():
        raise ValueError("Nonfinite blend loss; check input magnitudes and scaling")
    return {
        "evidence_cells": int(observed.sum()),
        "general_mse": general_mse,
        "specialist_mse": specialist_mse,
        "selected": "specialist" if specialist_mse < general_mse else "general",
        "specialist_weight": weight,
        "fallback": None,
    }


class _PretrainedImputer:
    def __init__(self, experts):
        self._experts = experts

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
        if precision not in {"amp_bfloat16", "float32"}:
            raise ValueError("precision must be 'amp_bfloat16' or 'float32'")
        for name, limit in (("max_rows", max_rows), ("max_cols", max_cols)):
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        target_device = torch.device(device)
        if target_device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are CPU and CUDA")
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA is unavailable; use device='cpu' or install a CUDA-enabled PyTorch"
                )
            with torch.cuda.device(target_device):
                if precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
                    raise ValueError("This GPU does not support BF16; use precision='float32'")
        paths = resolve_experts(
            cls._expert_names, local_dir=local_dir, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        experts = {}
        for name, artifacts in paths.items():
            config = json.loads(artifacts["config"].read_text(encoding="utf-8"))["model"]
            experts[name] = Expert(
                artifacts["checkpoint"], config, device=target_device, precision=precision,
                max_rows=max_rows, max_cols=max_cols,
            )
        return cls(experts)


class MatrixFormerImpute(_PretrainedImputer):
    """The single frozen general-purpose expert."""

    _expert_names = ("impute",)

    def impute(self, matrix):
        """Complete a 2D numerical matrix, preserving every observed entry."""
        matrix = prepare_matrix(matrix)
        completed = self._experts["impute"].impute(matrix.copy(order="K"))
        if not np.isfinite(completed).all():
            raise ValueError("Nonfinite expert predictions; check input magnitudes and scaling")
        return completed


class MatrixFormerPlus(_PretrainedImputer):
    """Two frozen experts with one observed-error-fitted weight per input matrix."""

    _expert_names = ("impute", "recsys")

    def __init__(self, experts):
        super().__init__(experts)
        self.last_decision = None

    def impute(self, matrix):
        """Impute without modifying the input or fitting neural network weights.

        last_decision exposes the specialist weight and in-sample reconstruction
        losses. These are not held-out accuracy estimates.
        """
        self.last_decision = None
        matrix = prepare_matrix(matrix)
        completed, raw = {}, {}
        for name, expert in self._experts.items():
            completed[name], raw[name] = expert.impute(
                matrix.copy(order="K"), return_full=True,
            )
            if not np.isfinite(completed[name]).all() or not np.isfinite(raw[name]).all():
                raise ValueError(f"Nonfinite expert predictions: {name}")
        decision = fit_blend(matrix, raw["impute"], raw["recsys"])
        weight = decision["specialist_weight"]
        result = (1.0 - weight) * completed["impute"] + weight * completed["recsys"]
        observed = np.isfinite(matrix)
        result[observed] = matrix[observed]
        self.last_decision = decision
        return result
