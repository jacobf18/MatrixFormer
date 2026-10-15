"""Memory-bounded predictive means and column-normalized matrix completion."""

import math
import warnings
from importlib.resources import files

import numpy as np
import torch
from torch.nn import functional

from ._model import FrozenModel


def bucket_means(device):
    with files("matrixformer").joinpath("assets/borders.pt").open("rb") as handle:
        borders = torch.load(handle, map_location=device, weights_only=True)
    widths = borders[1:] - borders[:-1]
    means = borders[:-1] + widths / 2
    halfnormal_median = widths.new_tensor(0.6744897501960817)
    means[0] = -(widths[0] / halfnormal_median * math.sqrt(2.0 / math.pi)) + borders[1]
    means[-1] = widths[-1] / halfnormal_median * math.sqrt(2.0 / math.pi) + borders[-2]
    return means


@torch.inference_mode()
def streamed_predictions(
    model, values, means, attention_batch_size=32,
    cell_batch_size=8192, encoder_row_batch_size=32,
):
    batch_size, rows, columns = values.shape
    cls_count = model.feature_encoder.num_cls
    hidden = None
    for start in range(0, rows, encoder_row_batch_size):
        stop = min(start + encoder_row_batch_size, rows)
        encoded = model.feature_encoder(values[:, start:stop])
        if hidden is None:
            hidden = encoded.new_empty(
                batch_size, rows + cls_count, columns + cls_count, encoded.shape[-1],
            )
            hidden[:, :cls_count].copy_(encoded[:, :cls_count])
        hidden[:, cls_count + start:cls_count + stop].copy_(encoded[:, cls_count:])
        del encoded
    for block in model.transformer_blocks:
        for batch in range(batch_size):
            for start in range(0, hidden.shape[1], attention_batch_size):
                current = hidden[batch, start:start + attention_batch_size]
                attended = block.self_attention_between_features(
                    block.norm1(current), rope_start_index=cls_count,
                )
                current.copy_(current + block.feature_residual_scale * attended)
                del attended
            for start in range(0, hidden.shape[2], attention_batch_size):
                current = hidden[batch, :, start:start + attention_batch_size]
                normalized = block.norm2(current).transpose(0, 1)
                if batch_size > 1:
                    normalized = normalized.contiguous()
                attended = block.self_attention_between_datapoints(
                    normalized, rope_start_index=cls_count,
                ).transpose(0, 1)
                current.copy_(current + block.datapoint_residual_scale * attended)
                del normalized, attended
        flat = hidden.view(-1, hidden.shape[-1])
        for start in range(0, flat.shape[0], cell_batch_size):
            current = flat[start:start + cell_batch_size]
            normalized = block.norm3(current)
            projected = functional.silu(block.ff_gate(normalized)) * block.ff_value(normalized)
            projected = block.ff_out(projected)
            current.copy_(current + block.ffn_residual_scale * projected)
            del normalized, projected
    predictions = torch.empty(batch_size, rows, columns, device=values.device, dtype=torch.float32)
    decode_rows = max(2, cell_batch_size // columns)
    for batch in range(batch_size):
        for start in range(0, rows, decode_rows):
            stop = min(start + decode_rows, rows)
            if stop - start == 1 and rows > 1:
                start -= 1
            features = hidden[batch:batch + 1, cls_count + start:cls_count + stop, cls_count:]
            logits = model.decoder(features)
            with torch.autocast(device_type=values.device.type, enabled=False):
                predictions[batch:batch + 1, start:stop] = (
                    torch.softmax(logits.float(), -1) @ means
                )
            del logits
    return predictions


def overlapping_spans(size, limit):
    if limit is None or limit >= size:
        return [(0, size)]
    spans = [(start, start + limit) for start in range(0, size - limit, limit)]
    spans.append((size - limit, size))
    return spans


class Expert:
    """Retain reference parameter flags and run all predictions in inference mode.

    Changing requires_grad alters PyTorch's strided decoder matmul dispatch even
    inside inference_mode, so it can change rounding in the released recipe.
    """

    def __init__(self, checkpoint, config, *, device, precision, max_rows, max_cols):
        self.device = torch.device(device)
        self.precision = precision
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.model = FrozenModel(config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval()
        self.bucket_means = bucket_means(self.device)

    @torch.inference_mode()
    def predict(self, values):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.precision == "amp_bfloat16",
        ):
            return streamed_predictions(self.model, values, self.bucket_means)

    def impute(self, matrix, *, return_full=False):
        matrix = np.asarray(matrix, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(matrix, axis=0)
            scales = np.nanstd(matrix, axis=0)
        means = np.where(np.isnan(means), 0, means)
        scales = np.where(np.isnan(scales), 1, scales)
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("Column normalization overflowed; rescale the observed values")
        normalized = (matrix - means) / (scales + 1e-16)
        totals = np.zeros_like(normalized)
        raw_totals = np.zeros_like(normalized) if return_full else None
        counts = np.zeros(matrix.shape, dtype=np.uint8)
        for row_start, row_stop in overlapping_spans(matrix.shape[0], self.max_rows):
            for col_start, col_stop in overlapping_spans(matrix.shape[1], self.max_cols):
                selection = np.s_[row_start:row_stop, col_start:col_stop]
                block = normalized[selection]
                values = torch.from_numpy(block.copy()).to(
                    device=self.device, dtype=torch.float32,
                ).unsqueeze(0)
                predictions = self.predict(values)[0].cpu().numpy()
                totals[selection] += np.where(np.isnan(block), predictions, block)
                if return_full:
                    raw_totals[selection] += predictions
                counts[selection] += 1
        completed = (totals / counts) * (scales + 1e-16) + means
        completed[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
        if return_full:
            return completed, (raw_totals / counts) * (scales + 1e-16) + means
        return completed
