"""Inference architecture with state-dict names matching the released experts.

Only the frozen Dwip v2-2 configuration is supported. Training-only paths,
attention diagnostics, and unused absolute position embeddings are omitted.
"""

import torch
from torch import nn
from torch.nn import functional


class RotaryEmbedding(nn.Module):
    def __init__(self, dimension, base):
        super().__init__()
        frequencies = 1.0 / (base ** (torch.arange(0, dimension, 2).float() / dimension))
        self.register_buffer("inv_freq", frequencies)

    def forward(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype)
        sine = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype)
        return cosine[None, None], sine[None, None]


def apply_rotary(values, cosine, sine, start, dimension):
    prefix = values[:, :, :start]
    target = values[:, :, start:]
    rotated_values = target[..., :dimension]
    pairs = rotated_values.reshape(*rotated_values.shape[:-1], -1, 2)
    rotated_half = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    rotated = rotated_values * cosine + rotated_half * sine
    tail = torch.cat([rotated, target[..., dimension:]], dim=-1)
    return torch.cat([prefix, tail], dim=2) if start > 0 else tail


class AxialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config["embedding_size"]
        self.heads = config["num_attention_heads"]
        self.head_width = self.width // self.heads
        rotary_dimension = int(self.head_width * config["rope_fraction"])
        self.rotary_dimension = rotary_dimension - rotary_dimension % 2
        self.rope = RotaryEmbedding(self.rotary_dimension, config["rope_base"])
        self.q_proj = nn.Linear(self.width, self.width)
        self.k_proj = nn.Linear(self.width, self.width)
        self.v_proj = nn.Linear(self.width, self.width)
        self.out_proj = nn.Linear(self.width, self.width)

    def split_heads(self, values):
        batch_size, sequence_length, _ = values.shape
        return values.view(
            batch_size, sequence_length, self.heads, self.head_width,
        ).transpose(1, 2)

    def forward(self, values, rope_start_index):
        query = self.split_heads(self.q_proj(values))
        key = self.split_heads(self.k_proj(values))
        value = self.split_heads(self.v_proj(values))
        if rope_start_index < query.size(2):
            cosine, sine = self.rope(query.size(2) - rope_start_index, query.device, query.dtype)
            query = apply_rotary(query, cosine, sine, rope_start_index, self.rotary_dimension)
            key = apply_rotary(key, cosine, sine, rope_start_index, self.rotary_dimension)
        attended = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().view(
            values.size(0), values.size(1), self.width,
        )
        return self.out_proj(attended)


class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config["embedding_size"]
        self.num_cls = config["num_cls"]
        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.width))
        self.missingness_embedding = nn.Embedding(2, self.width)
        self.row_cls_embedding = nn.Embedding(self.num_cls, self.width)
        self.col_cls_embedding = nn.Embedding(self.num_cls, self.width)

    def forward(self, values):
        missing = torch.isnan(values)
        clean = torch.where(missing, torch.zeros_like(values), values)
        embedded_observed = self.observed_linear_layer(clean.unsqueeze(-1))
        batch_size, rows, columns = values.shape
        embedded = torch.where(
            missing.unsqueeze(-1).expand_as(embedded_observed),
            self.mask_token.expand(batch_size, rows, columns, self.width),
            embedded_observed,
        )
        embedded = embedded + self.missingness_embedding(missing.to(torch.long))
        indices = torch.arange(self.num_cls, device=values.device, dtype=torch.long)
        row_tokens = self.row_cls_embedding(indices).to(embedded.dtype)
        column_tokens = self.col_cls_embedding(indices).to(embedded.dtype)
        row_cls = row_tokens[None, None].expand(batch_size, rows, self.num_cls, self.width)
        column_cls = column_tokens[None, :, None].expand(
            batch_size, self.num_cls, columns, self.width,
        )
        row_corner = row_tokens[None].expand(self.num_cls, self.num_cls, self.width)
        column_corner = column_tokens[:, None].expand(self.num_cls, self.num_cls, self.width)
        corner = (row_corner + column_corner)[None].expand(
            batch_size, self.num_cls, self.num_cls, self.width,
        )
        top = torch.cat([corner, column_cls], dim=2)
        bottom = torch.cat([row_cls, embedded], dim=2)
        return torch.cat([top, bottom], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["embedding_size"]
        hidden_width = config["mlp_hidden_size"]
        self.self_attention_between_features = AxialAttention(config)
        self.self_attention_between_datapoints = AxialAttention(config)
        self.norm1 = nn.RMSNorm(width, eps=config["rms_norm_eps"])
        self.norm2 = nn.RMSNorm(width, eps=config["rms_norm_eps"])
        self.norm3 = nn.RMSNorm(width, eps=config["rms_norm_eps"])
        self.ff_gate = nn.Linear(width, hidden_width)
        self.ff_value = nn.Linear(width, hidden_width)
        self.ff_out = nn.Linear(hidden_width, width)
        initial_scale = config["residual_scale_init"]
        self.feature_residual_scale = nn.Parameter(torch.full((1,), initial_scale))
        self.datapoint_residual_scale = nn.Parameter(torch.full((1,), initial_scale))
        self.ffn_residual_scale = nn.Parameter(torch.full((1,), initial_scale))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_width = config["mlp_hidden_size"]
        self.layers = nn.ModuleList([
            nn.Linear(config["embedding_size"], hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, config["num_outputs"]),
        ])

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class FrozenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("use_absolute_positional_embeddings") or not config.get("use_rope"):
            raise ValueError("Only the released RoPE architecture is supported")
        self.feature_encoder = FeatureEncoder(config)
        self.transformer_blocks = nn.ModuleList([
            EncoderBlock(config) for _ in range(config["num_layers"])
        ])
        self.decoder = Decoder(config)
