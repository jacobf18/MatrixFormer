"""Gradient-safe axial microbatching with full attention context."""

from functools import partial

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


def recompute(function, *arguments, enabled):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *arguments, use_reentrant=False)
    return function(*arguments)


def attention_block(hidden, *, block, cls_count, attention_batch_size, cell_batch_size, use_checkpoint):
    def column_attention(current):
        return block.self_attention_between_features(block.norm1(current), rope_start_index=cls_count)

    def row_attention(current):
        return block.self_attention_between_datapoints(
            block.norm2(current).transpose(0, 1), rope_start_index=cls_count,
        ).transpose(0, 1)

    def feed_forward(current):
        normalized = block.norm3(current)
        return block.ff_out(functional.silu(block.ff_gate(normalized)) * block.ff_value(normalized))

    attended = torch.stack([
        torch.cat([recompute(column_attention, matrix[start:start + attention_batch_size], enabled=use_checkpoint)
                   for start in range(0, matrix.shape[0], attention_batch_size)], dim=0)
        for matrix in hidden.unbind(0)
    ], dim=0)
    hidden = hidden + block.feature_residual_scale * attended
    attended = torch.stack([
        torch.cat([recompute(row_attention, matrix[:, start:start + attention_batch_size], enabled=use_checkpoint)
                   for start in range(0, matrix.shape[1], attention_batch_size)], dim=1)
        for matrix in hidden.unbind(0)
    ], dim=0)
    hidden = hidden + block.datapoint_residual_scale * attended
    projected = torch.cat([
        recompute(feed_forward, current, enabled=use_checkpoint)
        for current in hidden.reshape(-1, hidden.shape[-1]).split(cell_batch_size)
    ], dim=0).reshape_as(hidden)
    return hidden + block.ffn_residual_scale * projected


def masked_loss(model, values, targets, mask, loss_function, *, attention_batch_size=32,
                cell_batch_size=4096, encoder_row_batch_size=32, decoder_cell_batch_size=256,
                use_checkpoint=True):
    """Average all queried cells, including incomplete final microbatches.

    Microbatches split independent attention sequences, not context length.
    Hidden grids and gradients remain in memory; this is not constant-memory training.
    """
    if min(attention_batch_size, cell_batch_size, encoder_row_batch_size, decoder_cell_batch_size) < 1:
        raise ValueError("Microbatch sizes must be positive")
    if values.ndim != 3 or targets.shape != values.shape or mask.shape != values.shape:
        raise ValueError("Require matching batch x rows x columns tensors")
    if mask.dtype != torch.bool or not mask.any() or (mask & torch.isfinite(values)).any():
        raise ValueError("Require nonempty hidden-cell queries")
    if not torch.isfinite(targets[mask]).all():
        raise ValueError("Targets must be finite")
    cls_count = model.feature_encoder.num_cls
    encoded_rows = []
    for start in range(0, values.shape[1], encoder_row_batch_size):
        encoded = recompute(model.feature_encoder, values[:, start:start + encoder_row_batch_size],
                            enabled=use_checkpoint)
        encoded_rows.append(encoded if start == 0 else encoded[:, cls_count:])
    hidden = torch.cat(encoded_rows, dim=1)
    del encoded, encoded_rows
    for block in model.transformer_blocks:
        run_block = partial(attention_block, block=block, cls_count=cls_count,
                            attention_batch_size=attention_batch_size, cell_batch_size=cell_batch_size,
                            use_checkpoint=use_checkpoint)
        hidden = recompute(run_block, hidden, enabled=use_checkpoint)
    features = hidden[:, cls_count:, cls_count:][mask]
    selected_targets = targets[mask]

    def decode_loss(current, truth):
        logits = model.decoder(current)
        with torch.autocast(device_type=values.device.type, enabled=False):
            losses = loss_function(logits.float(), truth.float())
        if losses.shape != truth.shape:
            raise ValueError("Loss must return one value per query")
        return losses.sum()

    losses = [recompute(decode_loss, current, truth, enabled=use_checkpoint)
              for current, truth in zip(features.split(decoder_cell_batch_size),
                                        selected_targets.split(decoder_cell_batch_size))]
    return torch.stack(losses).sum() / selected_targets.numel()
