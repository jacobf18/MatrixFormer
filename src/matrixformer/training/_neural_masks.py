"""Random neural missingness mechanisms on compact matrices."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

mar_config = {'num_layers_upper': 3, 'hidden_lower': 1, 'hidden_upper': 100, 'activation': 'relu', 'N': 100, 'T': 50, 'row_neighbor_upper': 5, 'col_neighbor_upper': 5, 'seed': 42, 'neighbor_type': 'random'}

class MAR_missingness(nn.Module):
    """
    Neural Net approach for constructing MAR propensities
    Args:
        X: (n_rows, n_cols) numeric matrix; fully observed
        config: dict; configuration for the neural net
    Returns:
        prop: (n_rows, n_cols) numeric matrix; missingness propensities
    """

    def _layer_activation_mixer(self, row_index, time_index):
        """Helper function to interleave layers and activations"""
        mixed = []
        mixed.append(self.input_layers[row_index][time_index])
        mixed.append(self.activation_fn)
        for (column_index, layer) in enumerate(self.layers[row_index][time_index]):
            mixed.append(layer)
            if column_index < len(self.layers[row_index][time_index]) - 1:
                mixed.append(self.activation_fn)
        mixed.append(self.output_layers[row_index][time_index])
        return nn.Sequential(*mixed)

    def __init__(self, config):
        super().__init__()
        self.N = config['N']
        self.T = config['T']
        self.hid_low = config['hidden_lower']
        self.hid_up = config['hidden_upper']
        self.activation = config['activation']
        self.neighbor_type = config['neighbor_type']
        torch.random.default_generator.manual_seed(config['seed'])
        self.num_layers = torch.stack([torch.stack([torch.randint(1, config['num_layers_upper'] + 1, size=(1,)) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.row_neighbor = torch.stack([torch.stack([torch.randint(1, config['row_neighbor_upper'] + 1, size=(1,)) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.col_neighbor = torch.stack([torch.stack([torch.randint(1, config['col_neighbor_upper'] + 1, size=(1,)) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.input_size = torch.stack([torch.stack([(self.row_neighbor[row_index][time_index] + 1) * (self.col_neighbor[row_index][time_index] + 1) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.hid_seq = [[torch.randint(self.hid_low, self.hid_up + 1, size=(self.num_layers[row_index][time_index],)) for time_index in range(self.T)] for row_index in range(self.N)]
        self.input_layers = nn.ModuleList([nn.ModuleList([nn.Linear(self.input_size[row_index][time_index], self.hid_seq[row_index][time_index][0]) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.layers = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(self.hid_seq[row_index][time_index][support_count], self.hid_seq[row_index][time_index][support_count + 1]) for support_count in range(self.num_layers[row_index][time_index] - 1)]) for time_index in range(self.T)]) for row_index in range(self.N)])
        self.output_layers = nn.ModuleList([nn.ModuleList([nn.Linear(self.hid_seq[row_index][time_index][-1], 1) for time_index in range(self.T)]) for row_index in range(self.N)])
        for row_index in range(self.N):
            for time_index in range(self.T):
                nn.init.normal_(self.input_layers[row_index][time_index].weight, mean=0.0, std=1)
                nn.init.normal_(self.input_layers[row_index][time_index].bias, mean=0.0, std=1)
                self.input_layers[row_index][time_index].weight.requires_grad = False
                self.input_layers[row_index][time_index].bias.requires_grad = False
                for layer in self.layers[row_index][time_index]:
                    nn.init.normal_(layer.weight, mean=0.0, std=1)
                    nn.init.normal_(layer.bias, mean=0.0, std=1)
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                nn.init.normal_(self.output_layers[row_index][time_index].weight, mean=0.0, std=1)
                nn.init.normal_(self.output_layers[row_index][time_index].bias, mean=0.0, std=1)
                self.output_layers[row_index][time_index].weight.requires_grad = False
                self.output_layers[row_index][time_index].bias.requires_grad = False
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f'Invalid activation function: {self.activation}')
        self.final_net = nn.ModuleList([nn.ModuleList([self._layer_activation_mixer(row_index, time_index) for time_index in range(self.T)]) for row_index in range(self.N)])

    def get_final_net(self, row_index, time_index):
        return self.final_net[row_index][time_index]

    def forward(self, values):
        (n_rows, n_cols) = values.shape
        assert n_rows == self.N and n_cols == self.T
        propensity_matrix = torch.zeros(n_rows, n_cols)
        for row_index in range(self.N):
            for time_index in range(self.T):
                input_size = self.input_size[row_index][time_index].item()
                row_neighbor = self.row_neighbor[row_index][time_index].item()
                col_neighbor = self.col_neighbor[row_index][time_index].item()
                row_neighbor = min(row_neighbor, n_rows - 1)
                col_neighbor = min(col_neighbor, n_cols - 1)
                row_neighbor = max(row_neighbor, 1)
                col_neighbor = max(col_neighbor, 1)
                final_net = self.get_final_net(row_index, time_index)
                row_neighbor_idx = torch.randint(0, n_rows, size=(row_neighbor,))
                col_neighbor_idx = torch.randint(0, n_cols, size=(col_neighbor,))
                if row_index in row_neighbor_idx:
                    available_rows = torch.tensor([triangular for triangular in range(n_rows) if triangular != row_index and triangular not in row_neighbor_idx])
                    if len(available_rows) > 0:
                        replacement = available_rows[torch.randint(0, len(available_rows), (1,))]
                        row_neighbor_idx[row_neighbor_idx == row_index] = replacement
                if time_index in col_neighbor_idx:
                    available_cols = torch.tensor([column_index for column_index in range(n_cols) if column_index != time_index and column_index not in col_neighbor_idx])
                    if len(available_cols) > 0:
                        replacement = available_cols[torch.randint(0, len(available_cols), (1,))]
                        col_neighbor_idx[col_neighbor_idx == time_index] = replacement
                row_neighbor_idx = torch.cat([row_neighbor_idx, torch.tensor([row_index])])
                col_neighbor_idx = torch.cat([col_neighbor_idx, torch.tensor([time_index])])
                X_neighbor = values[row_neighbor_idx.unsqueeze(1), col_neighbor_idx.unsqueeze(0)]
                X_neighbor = X_neighbor.flatten()
                X_neighbor = X_neighbor.unsqueeze(0)
                propensity = torch.sigmoid(final_net(X_neighbor))
                propensity_matrix[row_index, time_index] = propensity.squeeze()
        return propensity_matrix
mar_block_config = {'N': 100, 'T': 50, 'row_blocks': 10, 'col_blocks': 10, 'convolution_type': 'mean'}

class MAR_block_missingness(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.N = config['N']
        self.T = config['T']
        self.row_blocks = config['row_blocks']
        self.col_blocks = config['col_blocks']
        self.convolution_type = config['convolution_type']
        mar_small_config = {'num_layers_upper': 3, 'hidden_lower': 2, 'hidden_upper': 3, 'activation': 'relu', 'N': self.row_blocks, 'T': self.col_blocks, 'row_neighbor_upper': max(1, self.row_blocks // 2), 'col_neighbor_upper': max(1, self.col_blocks // 2), 'seed': 42, 'neighbor_type': 'random'}
        self.mar_generator = MAR_missingness(mar_small_config)

    def forward(self, values):
        assert values.shape[0] == self.N and values.shape[1] == self.T
        actual_row_blocks = min(self.N, self.row_blocks)
        actual_col_blocks = min(self.T, self.col_blocks)
        row_partition = np.ones(actual_row_blocks, dtype=int)
        col_partition = np.ones(actual_col_blocks, dtype=int)
        remaining_rows = self.N - actual_row_blocks
        remaining_cols = self.T - actual_col_blocks
        if remaining_rows > 0:
            extra_rows = np.random.multinomial(remaining_rows, [1 / actual_row_blocks] * actual_row_blocks)
            row_partition += extra_rows
        if remaining_cols > 0:
            extra_cols = np.random.multinomial(remaining_cols, [1 / actual_col_blocks] * actual_col_blocks)
            col_partition += extra_cols
        assert row_partition.sum() == self.N, f'Row partition sum {row_partition.sum()} != N {self.N}'
        assert col_partition.sum() == self.T, f'Col partition sum {col_partition.sum()} != T {self.T}'
        assert (row_partition > 0).all(), f'Some row blocks are empty: {row_partition}'
        assert (col_partition > 0).all(), f'Some col blocks are empty: {col_partition}'
        row_cumsum = np.concatenate([[0], np.cumsum(row_partition)])
        col_cumsum = np.concatenate([[0], np.cumsum(col_partition)])
        X_small = np.zeros((actual_row_blocks, actual_col_blocks))
        for row_index in range(actual_row_blocks):
            for time_index in range(actual_col_blocks):
                block = values[row_cumsum[row_index]:row_cumsum[row_index + 1], col_cumsum[time_index]:col_cumsum[time_index + 1]]
                if self.convolution_type == 'mean':
                    X_small[row_index, time_index] = block.mean().item()
                if self.convolution_type == 'max':
                    X_small[row_index, time_index] = block.max().item()
        X_small = torch.from_numpy(X_small).to(dtype=torch.float32)
        propensity_small = self.mar_generator(X_small)
        return (propensity_small, row_cumsum, col_cumsum)
