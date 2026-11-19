"""Heterogeneous linear latent factors."""

import numpy as np

class LatentFactorPrior:
    """Generates a complete data matrix using the Latent-Factor Families approach."""

    def __init__(self, config: dict):
        self.config = config
        self.distributions = ['Gaussian', 'Laplace', 't_v', 'Spike+Slab', 'Dirichlet']

    def _sample_latent_vectors(self, num_vectors: int, rank: int) -> np.ndarray:
        latent_matrix = np.zeros((num_vectors, rank))
        for row_index in range(num_vectors):
            choice = np.random.choice(self.distributions)
            if choice == 'Gaussian':
                latent_matrix[row_index, :] = np.random.normal(0, self.config.get('latent_gaussian_sigma', 1.0), size=rank)
            elif choice == 'Laplace':
                latent_matrix[row_index, :] = np.random.laplace(0, self.config.get('latent_laplace_b', 1.0), size=rank)
            elif choice == 't_v':
                df = np.random.randint(self.config.get('latent_t_df_low', 2), self.config.get('latent_t_df_high', 11))
                latent_matrix[row_index, :] = np.random.standard_t(df, size=rank)
            elif choice == 'Spike+Slab':
                is_spike = np.random.rand(rank) < self.config.get('latent_spike_p', 0.5)
                samples = np.random.normal(0, self.config.get('latent_slab_sigma', 3.0), size=rank)
                samples[is_spike] = 0
                latent_matrix[row_index, :] = samples
            elif choice == 'Dirichlet':
                alpha = np.full(rank, self.config.get('latent_dirichlet_alpha', 1.0))
                latent_matrix[row_index, :] = np.random.dirichlet(alpha) - 1 / rank
        return latent_matrix
