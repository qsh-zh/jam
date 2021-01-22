import torch
import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self, input_size, n_hidden, hidden_size, output_size, is_batchnorm=False
    ):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            if is_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
