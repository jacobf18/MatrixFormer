"""The twelve imputation missingness mechanisms."""

import copy
import numpy as np
import torch
from ._neural_masks import MAR_missingness, MAR_block_missingness

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError('Weight dictionary must sum to a positive value')
    return {key: float(value) / total for (key, value) in weights.items()}

def _sample_weighted_key(weights: dict[str, float]) -> str:
    names = list(weights.keys())
    probs = np.asarray([weights[name] for name in names], dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(names, p=probs))

def _safe_logit(prob: float) -> float:
    clipped = min(max(float(prob), 0.0001), 1.0 - 0.0001)
    return float(np.log(clipped / (1.0 - clipped)))

def _sample_profile_p_missing(config: dict) -> float:
    anchor_prob = float(config.get('profile_p_missing_anchor_prob', 0.25))
    anchor_value = float(config.get('profile_p_missing_anchor_value', 0.4))
    if np.random.rand() < anchor_prob:
        return anchor_value
    alpha = float(config.get('profile_p_missing_beta_alpha', 2.0))
    beta_param = float(config.get('profile_p_missing_beta_beta', 2.0))
    low = float(config.get('profile_p_missing_low', 0.05))
    width = float(config.get('profile_p_missing_width', 0.7))
    return low + width * float(np.random.beta(alpha, beta_param))
MISSINGNESS_PROFILES = {'missmix_v1': {'weights': _normalize_weights({'mcar': 0.3, 'mar': 0.2, 'mnar': 0.2, 'mnar_censoring': 0.1, 'mnar_two_phase_subset': 0.1, 'mnar_skip_logic': 0.05, 'mnar_cluster_level': 0.05}), 'p_missing_anchor_prob': 0.25, 'p_missing_anchor_value': 0.4, 'p_missing_beta_alpha': 2.0, 'p_missing_beta_beta': 2.0, 'p_missing_low': 0.05, 'p_missing_width': 0.7}}

class BaseMissingness:

    def __init__(self, config: dict):
        self.config = config

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MCARPattern(BaseMissingness):

    def _induce_missingness(self, values):
        values[torch.rand(*values.shape) < self.config.get('p_missing', 0.6)] = torch.nan
        return values

    def __str__(self):
        return 'MCAR'

class MARPattern(BaseMissingness):

    def _induce_missingness(self, values):
        (probability, (_, n_cols)) = (self.config.get('p_missing', 0.6), values.shape)
        num_predictors = np.random.randint(1, max(2, n_cols // 2))
        predictor_indices = np.random.choice(n_cols, num_predictors, replace=False)
        target_indices = np.setdiff1d(np.arange(n_cols), predictor_indices)
        predictors = values[:, predictor_indices]
        predictors[torch.isnan(predictors)] = 0
        log_odds = predictors @ torch.randn(len(predictor_indices))
        probs = torch.sigmoid(log_odds - torch.quantile(log_odds, 1 - probability))
        for col_idx in target_indices:
            values[torch.rand(len(values)) < probs, col_idx] = torch.nan
        return values

    def __str__(self):
        return 'MAR'

class MNARPattern(BaseMissingness):

    def _induce_missingness(self, values):
        (probability, (_, n_cols)) = (self.config.get('p_missing', 0.6), values.shape)
        target_indices = np.random.choice(n_cols, np.random.randint(1, n_cols + 1), replace=False)
        for col_idx in target_indices:
            col_data = values[:, col_idx]
            log_odds = col_data * np.random.choice([-2, -1, 1, 2])
            if not torch.isnan(log_odds).all():
                beta_0 = -torch.quantile(log_odds[~torch.isnan(log_odds)], 1 - probability)
                probs = torch.sigmoid(log_odds + beta_0)
                values[torch.rand(len(values)) < probs, col_idx] = torch.nan
        return values

    def __str__(self):
        return 'MNAR'

class MNARPolarizationPattern(BaseMissingness):

    def _induce_missingness(self, values):
        threshold = self.config.get('threshold_quantile', 0.25)
        for column_index in range(values.shape[1]):
            col_data = values[:, column_index]
            if len(torch.unique(col_data[~torch.isnan(col_data)])) < 2:
                continue
            low = torch.quantile(col_data[~torch.isnan(col_data)], threshold)
            high = torch.quantile(col_data[~torch.isnan(col_data)], 1 - threshold)
            if low != high:
                values[(col_data > low) & (col_data < high), column_index] = torch.nan
        return values

class MNARSoftPolarizationPattern(BaseMissingness):
    """Induces missingness using a soft polarization pattern."""

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        alpha = self.config.get('soft_polarization_alpha', np.random.uniform(1.1, 5.0))
        epsilon = self.config.get('soft_polarization_epsilon', np.random.uniform(0.0, 0.1))
        for column_index in range(values.shape[1]):
            col_data = values[:, column_index]
            valid_data = col_data[~torch.isnan(col_data)]
            if len(torch.unique(valid_data)) < 2:
                continue
            mu = torch.median(valid_data)
            deviation = torch.abs(col_data - mu) ** alpha
            max_deviation = torch.max(deviation[~torch.isnan(deviation)])
            if max_deviation < 1e-06:
                continue
            normalized_deviation = deviation / max_deviation
            p_obs = epsilon + (1 - 2 * epsilon) * normalized_deviation
            is_missing = torch.rand_like(p_obs) > p_obs
            col_data[is_missing] = torch.nan
            values[:, column_index] = col_data
        return values

class MNARLatentFactorPattern(BaseMissingness):
    """
    Induces missingness where the probability is determined by a low-rank
    dot-product of latent factors for rows and columns.
    p_ij = sigmoid(u_i^T v_j + b_i + c_j)
    """

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (n_rows, n_cols) = values.shape
        rank_low = int(self.config.get('latent_rank_low', 1))
        rank_high = min(int(self.config.get('latent_rank_high', 11)), n_rows, n_cols)
        rank_low = min(rank_low, rank_high)
        rank = np.random.randint(rank_low, rank_high + 1)
        row_factors = torch.randn(n_rows, rank)
        column_factors = torch.randn(n_cols, rank)
        batch_index = torch.randn(n_rows, 1)
        column_index = torch.randn(1, n_cols)
        logits = row_factors @ column_factors.T / np.sqrt(max(rank, 1)) + batch_index + column_index
        target_obs = 1.0 - float(self.config.get('p_missing', 0.6))
        target_obs = min(max(target_obs, 0.0001), 1.0 - 0.0001)
        (low, high) = (-40.0, 40.0)
        for _ in range(40):
            mid = 0.5 * (low + high)
            realized = float(torch.sigmoid(logits + mid).mean())
            if realized < target_obs:
                low = mid
            else:
                high = mid
        p_obs = torch.sigmoid(logits + 0.5 * (low + high))
        mask = torch.rand_like(p_obs) > p_obs
        values[mask] = torch.nan
        return values

class MNARClusterLevelPattern(BaseMissingness):
    """
    Induces missingness based on cluster-level random effects. Rows and columns
    are assigned to clusters, and each cluster has an offset that affects the
    observation probability of all its members.
    """

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (n_rows, n_cols) = values.shape
        num_row_clusters = self.config.get('cluster_level_n_row_clusters', 5)
        num_col_clusters = self.config.get('cluster_level_n_col_clusters', 5)
        tau_r_std = self.config.get('cluster_level_tau_r_std', 1.0)
        tau_c_std = self.config.get('cluster_level_tau_c_std', 1.0)
        epsilon_std = self.config.get('cluster_level_epsilon_std', 1.0)
        row_assignments = torch.randint(0, num_row_clusters, (n_rows,))
        col_assignments = torch.randint(0, num_col_clusters, (n_cols,))
        g_k = torch.randn(num_row_clusters) * tau_r_std
        h_l = torch.randn(num_col_clusters) * tau_c_std
        g_expanded = g_k[row_assignments].unsqueeze(1)
        h_expanded = h_l[col_assignments].unsqueeze(0)
        epsilon_noise = torch.randn(n_rows, n_cols) * epsilon_std
        logits = g_expanded + h_expanded + epsilon_noise
        p_obs = torch.sigmoid(logits)
        mask = torch.rand_like(p_obs) > p_obs
        values[mask] = torch.nan
        return values

class MNARCensoringPattern(BaseMissingness):
    """
    Induces missingness by censoring values that are above or below a certain
    detection threshold
    """

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        censor_quantile = self.config.get('censor_quantile', 0.1)
        for column_index in range(values.shape[1]):
            col_data = values[:, column_index]
            valid_data = col_data[~torch.isnan(col_data)]
            if len(torch.unique(valid_data)) < 2:
                continue
            if np.random.rand() < 0.5:
                cutoff = torch.quantile(valid_data, censor_quantile)
                col_data[col_data < cutoff] = torch.nan
            else:
                cutoff = torch.quantile(valid_data, 1 - censor_quantile)
                col_data[col_data > cutoff] = torch.nan
            values[:, column_index] = col_data
        return values

class MNARTwoPhaseSubsetPattern(BaseMissingness):
    """
    Induces missingness based on a two-phase survey design. A subset of
    "expensive" columns are missing for rows based on their values in
    "cheap" (always observed) columns.
    """

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (n_rows, n_cols) = values.shape
        if n_cols < 2:
            return values
        frac = self.config.get('two_phase_cheap_fraction', 0.5)
        n_cheap = max(1, int(n_cols * frac))
        col_indices = np.random.permutation(n_cols)
        (cheap_cols, expensive_cols) = (col_indices[:n_cheap], col_indices[n_cheap:])
        X_cheap = values[:, cheap_cols]
        weights = torch.randn(n_cheap)
        scores = X_cheap @ weights
        scores = (scores - scores.mean()) / (scores.std() + 1e-06)
        alpha = self.config.get('two_phase_alpha', 0)
        beta = self.config.get('two_phase_beta', 2.0)
        logits = alpha + beta * scores
        p_obs = torch.sigmoid(logits)
        is_missing = torch.rand(n_rows) > p_obs
        values[is_missing, :] = values[is_missing, :].index_fill(1, torch.tensor(expensive_cols), torch.nan)
        return values

class MNARSkipLogicPattern(BaseMissingness):
    """
    Induces missingness based on survey skip-logic, creating deterministic
    nested missingness patterns. Follow-up questions are missing if a
    "gate" question is answered a certain way.
    """

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (_, n_cols) = values.shape
        if n_cols < 2:
            return values
        p_noise = self.config.get('skip_logic_p_noise', 0.95)
        gate_col = np.random.randint(0, n_cols)
        dependent_cols = [column_index for column_index in range(n_cols) if column_index != gate_col]
        gate_condition_met = values[:, gate_col] > torch.median(values[:, gate_col])
        rows_to_mask = ~gate_condition_met
        values[rows_to_mask, :] = values[rows_to_mask, :].index_fill(1, torch.tensor(dependent_cols), torch.nan)
        rows_passed = torch.where(gate_condition_met)[0]
        if len(rows_passed) > 0:
            rand_mask = torch.rand(len(rows_passed), len(dependent_cols)) > p_noise
            values[rows_passed.unsqueeze(1), torch.tensor(dependent_cols).unsqueeze(0)][rand_mask] = torch.nan
        return values

class MARNeuralNetwork(BaseMissingness):
    """
    Induces missingness based on a neural network with random weights.
    This is a simple implementation of the MAR missingness pattern.
    """

    def __init__(self, config: dict):
        self.mar_config = config['mar_config']

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        self.mar_config['N'] = values.shape[0]
        self.mar_config['T'] = values.shape[1]
        self.mar_config['row_neighbor_upper'] = values.shape[0] // 2
        self.mar_config['col_neighbor_upper'] = values.shape[1] // 2
        self.mar = MAR_missingness(self.mar_config)
        propensities = self.mar(values)
        mask = torch.bernoulli(propensities).bool()
        values[mask] = torch.nan
        return values

class MARBlockNeuralNetwork(BaseMissingness):
    """
    Induces missingness based on a neural network with random weights.
    This is a simple implementation of the MAR missingness pattern.
    """

    def __init__(self, config: dict):
        self.mar_block_config = config['mar_block_config']

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        self.mar_block_config['N'] = values.shape[0]
        self.mar_block_config['T'] = values.shape[1]
        self.mar_block_config['row_blocks'] = values.shape[0] // 3
        self.mar_block_config['col_blocks'] = values.shape[1] // 3
        self.mar_block = MAR_block_missingness(self.mar_block_config)
        (propensities, row_cumsum, col_cumsum) = self.mar_block(values)

        def recover_block_missingness(propensity_small, row_cumsum, col_cumsum):
            row_count = row_cumsum[-1]
            column_count = col_cumsum[-1]
            missing_mask = torch.zeros(row_count, column_count)
            for row_index in range(row_cumsum.shape[0] - 1):
                for time_index in range(col_cumsum.shape[0] - 1):
                    missing_mask[row_cumsum[row_index]:row_cumsum[row_index + 1], col_cumsum[time_index]:col_cumsum[time_index + 1]] = torch.bernoulli(propensity_small[row_index, time_index])
            return missing_mask
        mask = recover_block_missingness(propensities, row_cumsum, col_cumsum).bool()
        values[mask] = torch.nan
        return values

class ProfileMNARClusterLevelPattern(MNARClusterLevelPattern):
    """Cluster-level pattern variant with profile-controlled density."""

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (n_rows, n_cols) = values.shape
        num_row_clusters = self.config.get('cluster_level_n_row_clusters', 5)
        num_col_clusters = self.config.get('cluster_level_n_col_clusters', 5)
        tau_r_std = self.config.get('cluster_level_tau_r_std', 1.0)
        tau_c_std = self.config.get('cluster_level_tau_c_std', 1.0)
        epsilon_std = self.config.get('cluster_level_epsilon_std', 1.0)
        row_assignments = torch.randint(0, num_row_clusters, (n_rows,))
        col_assignments = torch.randint(0, num_col_clusters, (n_cols,))
        g_k = torch.randn(num_row_clusters) * tau_r_std
        h_l = torch.randn(num_col_clusters) * tau_c_std
        g_expanded = g_k[row_assignments].unsqueeze(1)
        h_expanded = h_l[col_assignments].unsqueeze(0)
        epsilon_noise = torch.randn(n_rows, n_cols) * epsilon_std
        logit_bias = self.config.get('cluster_level_logit_bias', 0.0)
        logits = g_expanded + h_expanded + epsilon_noise + float(logit_bias)
        p_obs = torch.sigmoid(logits)
        mask = torch.rand_like(p_obs) > p_obs
        values[mask] = torch.nan
        return values

class ProfileMNARTwoPhaseSubsetPattern(MNARTwoPhaseSubsetPattern):
    """Two-phase subset pattern variant with profile-controlled density."""

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (n_rows, n_cols) = values.shape
        if n_cols < 2:
            return values
        frac = self.config.get('two_phase_cheap_fraction', 0.5)
        n_cheap = max(1, int(n_cols * frac))
        col_indices = np.random.permutation(n_cols)
        (cheap_cols, expensive_cols) = (col_indices[:n_cheap], col_indices[n_cheap:])
        X_cheap = values[:, cheap_cols]
        weights = torch.randn(n_cheap)
        scores = X_cheap @ weights
        scores = (scores - scores.mean()) / (scores.std() + 1e-06)
        alpha = self.config.get('two_phase_alpha', _safe_logit(1.0 - float(self.config.get('p_missing', 0.5))))
        beta = self.config.get('two_phase_beta', 2.0)
        logits = alpha + beta * scores
        p_obs = torch.sigmoid(logits)
        is_missing = torch.rand(n_rows) > p_obs
        values[is_missing, :] = values[is_missing, :].index_fill(1, torch.tensor(expensive_cols), torch.nan)
        return values

class ProfileMNARSkipLogicPattern(MNARSkipLogicPattern):
    """Skip-logic pattern variant with profile-controlled gate threshold."""

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        (_, n_cols) = values.shape
        if n_cols < 2:
            return values
        p_noise = self.config.get('skip_logic_p_noise', 0.98)
        gate_quantile = float(self.config.get('skip_logic_gate_quantile', 0.5))
        gate_col = np.random.randint(0, n_cols)
        dependent_cols = [column_index for column_index in range(n_cols) if column_index != gate_col]
        gate_threshold = torch.quantile(values[:, gate_col], gate_quantile)
        gate_condition_met = values[:, gate_col] > gate_threshold
        rows_to_mask = ~gate_condition_met
        values[rows_to_mask, :] = values[rows_to_mask, :].index_fill(1, torch.tensor(dependent_cols), torch.nan)
        rows_passed = torch.where(gate_condition_met)[0]
        if len(rows_passed) > 0:
            rand_mask = torch.rand(len(rows_passed), len(dependent_cols)) > p_noise
            values[rows_passed.unsqueeze(1), torch.tensor(dependent_cols).unsqueeze(0)][rand_mask] = torch.nan
        return values

class ProfileMixedPattern(BaseMissingness):
    """
    Configurable mixed missingness mode that samples a pattern and effective
    missingness density independently for each sample.
    """
    pattern_classes = {'mcar': MCARPattern, 'mar': MARPattern, 'mnar': MNARPattern, 'mnar_censoring': MNARCensoringPattern, 'mnar_two_phase_subset': ProfileMNARTwoPhaseSubsetPattern, 'mnar_skip_logic': ProfileMNARSkipLogicPattern, 'mnar_cluster_level': ProfileMNARClusterLevelPattern}

    def __init__(self, config: dict):
        super().__init__(config)
        profile_name = config.get('missingness_profile_name', 'missmix_v1')
        if profile_name not in MISSINGNESS_PROFILES:
            raise ValueError(f"Unknown missingness profile '{profile_name}'. Available: {list(MISSINGNESS_PROFILES.keys())}")
        self.profile_name = profile_name
        self.profile = copy.deepcopy(MISSINGNESS_PROFILES[profile_name])
        for field in ('p_missing_anchor_prob', 'p_missing_anchor_value', 'p_missing_beta_alpha', 'p_missing_beta_beta', 'p_missing_low', 'p_missing_width'):
            key = f'profile_{field}'
            if key in config:
                self.profile[field] = float(config[key])
        for field in ('p_missing_anchor_prob', 'p_missing_anchor_value', 'p_missing_low', 'p_missing_width'):
            if not 0.0 <= self.profile[field] <= 1.0:
                raise ValueError(f'profile_{field} must be in [0, 1]')
        if self.profile['p_missing_low'] + self.profile['p_missing_width'] > 1.0:
            raise ValueError('Missingness density low + width must be <= 1')
        for field in ('p_missing_beta_alpha', 'p_missing_beta_beta'):
            if not np.isfinite(self.profile[field]) or self.profile[field] <= 0.0:
                raise ValueError(f'profile_{field} must be finite and positive')
        self.weights = dict(self.profile['weights'])

    def _build_pattern_config(self, p_missing: float) -> dict:
        pattern_config = copy.deepcopy(self.config)
        pattern_config['p_missing'] = p_missing
        pattern_config['censor_quantile'] = min(max(p_missing, 0.01), 0.49)
        pattern_config['two_phase_alpha'] = _safe_logit(1.0 - p_missing)
        pattern_config['skip_logic_gate_quantile'] = min(max(p_missing, 0.05), 0.95)
        pattern_config.setdefault('skip_logic_p_noise', 0.98)
        pattern_config['cluster_level_logit_bias'] = _safe_logit(1.0 - p_missing)
        return pattern_config

    def _induce_missingness(self, values: torch.Tensor) -> torch.Tensor:
        p_missing = _sample_profile_p_missing({'profile_p_missing_anchor_prob': self.profile['p_missing_anchor_prob'], 'profile_p_missing_anchor_value': self.profile['p_missing_anchor_value'], 'profile_p_missing_beta_alpha': self.profile['p_missing_beta_alpha'], 'profile_p_missing_beta_beta': self.profile['p_missing_beta_beta'], 'profile_p_missing_low': self.profile['p_missing_low'], 'profile_p_missing_width': self.profile['p_missing_width']})
        pattern_name = _sample_weighted_key(self.weights)
        pattern_cls = self.pattern_classes[pattern_name]
        return pattern_cls(self._build_pattern_config(p_missing))._induce_missingness(values)
