"""Normalization and sparse-support masks for synthetic episodes."""

import math
import random
from contextlib import contextmanager
import numpy as np
import torch

def torch_nansum(values: torch.Tensor, axis: int | tuple[int, ...] | None=None, keepdim: bool=False, dtype: torch.dtype | None=None) -> torch.Tensor:
    """Computes the sum of a tensor, treating NaNs as zero.

    Args:
        x: The input tensor.
        axis: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `axis` retained or not.
        dtype: The desired data type of the returned tensor.

    Returns:
        The sum of the tensor with NaNs treated as zero.
    """
    nan_mask = torch.isnan(values)
    masked_input = torch.where(nan_mask, torch.tensor(0.0, device=values.device, dtype=values.dtype), values)
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)

def torch_nanmean(values: torch.Tensor, axis: int=0, *, return_nanshare: bool=False, include_inf: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.
        return_nanshare: If True, also return the proportion of NaNs.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs. If `return_nanshare` is True,
        returns a tuple containing the mean and the share of NaNs.
    """
    nan_mask = torch.isnan(values)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(values))
    num = torch.where(nan_mask, torch.full_like(values, 0), torch.full_like(values, 1)).sum(axis=axis)
    value = torch.where(nan_mask, torch.full_like(values, 0), values).sum(axis=axis)
    if return_nanshare:
        return (value / num, 1.0 - num / values.shape[axis])
    return value / num.clip(min=1.0)

def torch_nanstd(values: torch.Tensor, axis: int=0) -> torch.Tensor:
    """Computes the standard deviation of a tensor over a given dimension, ignoring NaNs.

    This implementation is designed for stability. It clips the denominator `(num - 1)`
    at a minimum of 1. This prevents division-by-zero errors that would produce `NaN`
    or `inf` when calculating the standard deviation of a single data point.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.

    Returns:
        The standard deviation of the input tensor, ignoring NaNs.
    """
    num = torch.where(torch.isnan(values), torch.full_like(values, 0), torch.full_like(values, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(values), torch.full_like(values, 0), values).sum(axis=axis)
    mean = value / num.clip(min=1.0)
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(axis), values.shape[axis], dim=axis)
    var = torch_nansum(torch.square(mean_broadcast - values), axis=axis) / (num - 1).clip(min=1.0)
    return torch.sqrt(var)

def normalize_data(data: torch.Tensor, *, normalize_positions: int=-1, return_scaling: bool=False, clip: bool=True, std_only: bool=False, mean: torch.Tensor | None=None, std: torch.Tensor | None=None) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize data to mean 0 and std 1 with high numerical stability.

    This function is designed to be robust against several edge cases:
    1.  **Constant Features**: If a feature is constant, its standard deviation (`std`)
        will be 0. This is handled by replacing `std=0` with `1` to prevent `0/0`
        division, effectively mapping constant features to a normalized value of 0.
    2.  **Single-Sample Normalization**: If the normalization is based on a single
        data point, `std` is explicitly set to `1` to prevent undefined behavior.
    3.  **Low-Precision Dtypes**: During the final division, a small epsilon (`1e-16`)
        is added to the denominator. This prevents division by a near-zero `std`,
        which could cause the value to overflow to infinity (`inf`), especially when
        using low-precision dtypes like `torch.float16`.
    4.  **Autograd Compatibility**: All inplace operations (`x[:] = ...`) have been
        replaced with their functional equivalents (`torch.where`, `torch.ones_like`)
        to ensure the computation graph is not corrupted, allowing gradients to be
        computed correctly.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions`
            positions for normalization.
        return_scaling: If True, return the scaling parameters as well (mean, std).
        std_only: If True, only divide by std.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.

    Returns:
        The normalized data tensor, or a tuple containing the data and scaling factors.
    """
    assert (mean is None) == (std is None), 'Either both or none of mean and std must be given'
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)
            std = torch_nanstd(data[:normalize_positions], axis=0)
        else:
            mean = torch_nanmean(data, axis=0)
            std = torch_nanstd(data, axis=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        if len(data) == 1 or normalize_positions == 1:
            std = torch.ones_like(std)
        if std_only:
            mean = torch.zeros_like(mean)
    data = (data - mean) / (std + 1e-16)
    if clip:
        data = torch.clip(data, min=-100, max=100)
    if return_scaling:
        return (data, (mean, std))
    return data

@contextmanager
def preserved_rng(seed=None):
    (python_state, numpy_state) = (random.getstate(), np.random.get_state())
    with torch.random.fork_rng(devices=[]):
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % 2 ** 32)
                torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

def apply_support_mask(source, target, mode='mixed', seed=None):
    if mode not in {'mixed', 'broad_support', 'sparse_query'}:
        raise ValueError(f'Unknown support mode: {mode}')
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError('source and target must be matching B x R x C tensors')
    with preserved_rng(seed):
        if mode == 'mixed' or (mode == 'broad_support' and random.random() < 0.25):
            return (source, target, torch.isnan(source), {'support_episode': 0.0})
        transpose = random.random() < 0.5
        src = source.transpose(1, 2) if transpose else source
        keep = ~torch.isnan(src)
        query_mask = torch.zeros_like(keep)
        for batch_index in range(src.shape[0]):
            (rows, cols) = src.shape[1:]
            n_query = max(1, min(rows - 1, round(rows * random.uniform(0.1, 0.4))))
            query = torch.randperm(rows, device=src.device)[:n_query]
            fraction = math.exp(random.uniform(math.log(0.01), math.log(0.6)))
            support_count = max(1, min(cols - 1, round(cols * fraction)))
            allowed = torch.zeros((n_query, cols), device=src.device, dtype=torch.bool)
            if random.random() < 0.5:
                allowed[:, torch.randperm(cols, device=src.device)[:support_count]] = True
            else:
                for row in range(n_query):
                    allowed[row, torch.randperm(cols, device=src.device)[:support_count]] = True
            keep[batch_index, query] &= allowed
            query_mask[batch_index, query] = True
        if transpose:
            (keep, query_mask) = (keep.transpose(1, 2), query_mask.transpose(1, 2))
        masked = source.float().masked_fill(~keep, float('nan'))
        counts = keep.sum(dim=1, keepdim=True)
        mean = masked.nan_to_num().sum(dim=1, keepdim=True) / counts.clamp_min(1)
        centered = torch.where(keep, masked - mean, 0.0)
        std = (centered.square().sum(dim=1, keepdim=True) / counts.clamp_min(1)).sqrt()
        std = torch.where((counts >= 2) & (std > 0.001), std, 1.0)
        loss_mask = query_mask & ~keep
        if not loss_mask.any():
            loss_mask = ~keep
        stats = {'support_episode': 1.0, 'support_transposed': float(transpose), 'support_query_observed_rate': float(keep[query_mask].float().mean()), 'support_observed_rate': float(keep.float().mean())}
        return ((masked - mean) / std, (target.float() - mean) / std, loss_mask, stats)

def normalize_remaining_support(source, target, keep, variance_df=4.0):
    if not math.isfinite(variance_df) or variance_df <= 0:
        raise ValueError('variance_df must be positive and finite')
    if source.ndim != 3 or source.shape != target.shape or keep.shape != source.shape:
        raise ValueError('Expected matching B x R x C tensors')
    keep = keep.bool() & torch.isfinite(source)
    values = torch.where(keep, source.float(), 0.0)
    counts = keep.sum(1, keepdim=True)
    mean = values.sum(1, keepdim=True) / counts.clamp_min(1)
    centered = torch.where(keep, values - mean, 0.0)
    ss = centered.square().sum(1, keepdim=True)
    pool = ss.sum(2, keepdim=True) / counts.sum(2, keepdim=True).clamp_min(1)
    pool = torch.where(pool > 1e-06, pool, 1.0)
    variance = (ss + variance_df * pool) / (counts + variance_df)
    scale = variance.clamp_min(1e-06).sqrt()
    masked = source.float().masked_fill(~keep, float('nan'))
    return ((masked - mean) / scale, (target.float() - mean) / scale, mean, scale)

def apply_shrunk_support_mask(source, target, mode='mixed', seed=None, variance_df=4.0):
    (masked, normalized, mask, stats) = apply_support_mask(source, target, mode, seed)
    if not stats['support_episode']:
        return (masked, normalized, mask, stats)
    (values, target_values, _, scale) = normalize_remaining_support(source, target, torch.isfinite(masked), variance_df)
    stats = {**stats, 'support_variance_df': variance_df, 'support_scale_min': float(scale.min()), 'support_target_rms': float(target_values[mask].square().mean().sqrt()), 'support_original_target_rms': float(target[mask].square().mean().sqrt())}
    return (values, target_values, mask, stats)
