from torch import nn

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden, h_dim, out_dim, is_batchnorm=False):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, h_dim))
            if is_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(h_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
