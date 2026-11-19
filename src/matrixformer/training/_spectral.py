"""Clean spectral signals, noisy measurements, and mixed basis geometry."""

import numpy as np
import torch

class DecayingSpectrumGenerator:
    """Generate full-spectrum matrices and truncated top-k targets."""

    def __init__(self, config: dict):
        self.config = config

    def _sample_int(self, low_key: str, high_key: str) -> int:
        low = int(self.config[low_key])
        high = int(self.config[high_key])
        if high < low:
            raise ValueError(f'Expected {high_key} >= {low_key}, got {high} < {low}')
        return int(np.random.randint(low, high + 1))

    def _sample_orthonormal_matrix(self, n_rows: int, rank: int) -> np.ndarray:
        gaussian = np.random.standard_normal((n_rows, rank))
        (basis, triangular) = np.linalg.qr(gaussian, mode='reduced')
        diag = np.sign(np.diag(triangular))
        diag[diag == 0] = 1.0
        return basis * diag

    def sample_spectrum(self, rank: int) -> tuple[np.ndarray, dict[str, float]]:
        alpha = float(np.random.uniform(self.config.get('spectrum_alpha_low', 0.7), self.config.get('spectrum_alpha_high', 1.0)))
        scale = float(np.random.uniform(self.config.get('spectrum_scale_low', 1.0), self.config.get('spectrum_scale_high', 5.0)))
        head_weight = float(np.random.uniform(self.config.get('spectrum_head_weight_low', 5.0), self.config.get('spectrum_head_weight_high', 10.0)))
        head_exp_scale_low = self.config.get('spectrum_head_exp_scale_low')
        head_exp_scale_high = self.config.get('spectrum_head_exp_scale_high')
        head_handoff_scale_low = self.config.get('spectrum_head_handoff_scale_low')
        head_handoff_scale_high = self.config.get('spectrum_head_handoff_scale_high')
        if head_exp_scale_low is not None and head_exp_scale_high is not None and (head_handoff_scale_low is not None) and (head_handoff_scale_high is not None):
            head_exp_scale = float(np.random.uniform(head_exp_scale_low, head_exp_scale_high))
            head_handoff_scale = float(np.random.uniform(head_handoff_scale_low, head_handoff_scale_high))
            head_decay = head_handoff_scale / max(rank, 1)
        else:
            head_decay = float(np.random.uniform(self.config.get('spectrum_head_decay_low', 0.14), self.config.get('spectrum_head_decay_high', 0.24)))
            head_handoff_scale = max(1.0, head_decay * rank)
            head_exp_scale = max(1.0, 0.35 * head_handoff_scale)
        taper_strength = float(np.random.uniform(self.config.get('spectrum_taper_strength_low', 0.6), self.config.get('spectrum_taper_strength_high', 1.8)))
        taper_power = float(np.random.uniform(self.config.get('spectrum_taper_power_low', 3.5), self.config.get('spectrum_taper_power_high', 6.0)))
        head_exp_power = float(np.random.uniform(self.config.get('spectrum_head_exp_power_low', 1.8), self.config.get('spectrum_head_exp_power_high', 2.4)))
        tail_amplitude = float(np.random.uniform(self.config.get('spectrum_tail_amplitude_low', 0.03), self.config.get('spectrum_tail_amplitude_high', 0.12)))
        component_idx = np.arange(rank, dtype=np.float32)
        normalized_idx = component_idx / max(rank - 1, 1)
        head_component = np.exp(-np.power(component_idx / head_exp_scale, head_exp_power))
        tail_component = tail_amplitude * np.power(1.0 + component_idx, -alpha)
        taper_component = np.exp(-taper_strength * np.power(normalized_idx, taper_power))
        head_mix = 1.0 - np.exp(-head_weight)
        blend = head_mix * np.exp(-component_idx / head_handoff_scale)
        spectrum = (blend * head_component + (1.0 - blend) * tail_component) * taper_component
        spectrum /= spectrum[0]
        spectrum *= scale
        return (spectrum, {'alpha': alpha, 'scale': scale, 'head_weight': head_weight, 'head_decay': head_decay, 'head_scale': head_handoff_scale, 'head_exp_scale': head_exp_scale, 'head_exp_power': head_exp_power, 'head_mix': head_mix, 'tail_amplitude': tail_amplitude, 'taper_strength': taper_strength, 'taper_power': taper_power})

    def _resolve_rank(self, n_rows: int, n_cols: int) -> int:
        rank_low = int(self.config.get('spectrum_rank_low', self.config.get('latent_rank_low', 5)))
        rank_high = int(self.config.get('spectrum_rank_high', self.config.get('latent_rank_high', 30)))
        rank_high = min(rank_high, n_rows, n_cols)
        rank_low = min(rank_low, rank_high)
        return int(np.random.randint(rank_low, rank_high + 1))

    def _resolve_target_rank(self, spectrum: np.ndarray, noise_std: float) -> int:
        explicit_rank = self.config.get('spectrum_target_rank')
        if explicit_rank is not None:
            return max(1, min(int(explicit_rank), len(spectrum)))
        threshold_scale = float(self.config.get('spectrum_target_noise_floor_scale', 1.0))
        threshold = threshold_scale * float(noise_std)
        effective_rank = int(np.sum(spectrum > threshold))
        return max(1, min(effective_rank, len(spectrum)))

    def generate_matrix_bundle(self) -> dict[str, torch.Tensor | np.ndarray | float | int]:
        n_rows = self._sample_int('num_rows_low', 'num_rows_high')
        n_cols = self._sample_int('num_cols_low', 'num_cols_high')
        rank = self._resolve_rank(n_rows, n_cols)
        row_basis = self._sample_orthonormal_matrix(n_rows, rank).astype(np.float32)
        column_basis = self._sample_orthonormal_matrix(n_cols, rank).astype(np.float32)
        (spectrum, spectrum_params) = self.sample_spectrum(rank)
        noise_std = float(np.random.uniform(self.config.get('spectrum_noise_std_low', 0.05), self.config.get('spectrum_noise_std_high', 0.5)))
        target_rank = self._resolve_target_rank(spectrum=spectrum, noise_std=noise_std)
        full_signal = row_basis * spectrum @ column_basis.T
        truncated_signal = row_basis[:, :target_rank] * spectrum[:target_rank] @ column_basis[:, :target_rank].T
        noisy_observation = full_signal + noise_std * np.random.standard_normal((n_rows, n_cols))
        return {'full_matrix': torch.tensor(noisy_observation, dtype=torch.float32), 'target_matrix': torch.tensor(truncated_signal, dtype=torch.float32), 'clean_signal': torch.tensor(full_signal, dtype=torch.float32), 'spectrum': spectrum, 'rank': rank, 'target_rank': target_rank, **spectrum_params, 'noise_std': noise_std}

    def generate_matrix_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        bundle = self.generate_matrix_bundle()
        return (bundle['full_matrix'], bundle['target_matrix'])

class GeometrySpectrumGenerator(DecayingSpectrumGenerator):

    def _sample_orthonormal_matrix(self, n_rows, rank):
        kind = self.config.get('spectrum_basis_kind', 'haar')
        if kind == 'mixed':
            kind = str(np.random.choice(('haar', 'coherent', 'smooth', 'offset')))
        if kind == 'haar':
            return super()._sample_orthonormal_matrix(n_rows, rank)
        if kind == 'coherent':
            strength = float(np.random.uniform(0.25, 2.0))
            row_scale = np.exp(np.random.normal(0.0, strength, (n_rows, 1)))
            features = row_scale * np.random.standard_normal((n_rows, rank))
        elif kind == 'smooth':
            dimensions = int(np.random.randint(1, 6))
            positions = np.random.standard_normal((n_rows, dimensions))
            bandwidth = float(np.exp(np.random.uniform(np.log(0.3), np.log(3.0))))
            frequencies = np.random.standard_normal((dimensions, rank)) / bandwidth
            phases = np.random.uniform(0.0, 2 * np.pi, (1, rank))
            features = np.cos(positions @ frequencies + phases)
            features += 0.03 * np.random.standard_normal((n_rows, rank))
        elif kind == 'offset':
            features = np.random.standard_normal((n_rows, rank))
            features[:, 0] = 1.0 + 0.05 * features[:, 0]
        else:
            raise ValueError(f'Unknown spectral basis geometry: {kind}')
        (basis, triangular) = np.linalg.qr(features, mode='reduced')
        signs = np.sign(np.diag(triangular))
        signs[signs == 0] = 1.0
        return basis * signs
