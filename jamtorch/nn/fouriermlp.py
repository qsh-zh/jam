from collections.abc import Iterable

import numpy as np
import torch
from einops import rearrange
from torch import nn

__all__ = ["FourierMLP"]


def check_shape(cur_shape):
    if isinstance(cur_shape, Iterable):
        return tuple(cur_shape)
    elif isinstance(cur_shape, int):
        return tuple(
            [
                cur_shape,
            ]
        )
    else:
        raise NotImplementedError(f"Type {type(cur_shape)} not support")


class FourierMLP(nn.Module):
    def __init__(self, in_shape, out_shape, num_layers=2, channels=128, zero_init=True):
        super().__init__()
        self.in_shape = check_shape(in_shape)
        self.out_shape = check_shape(out_shape)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


if __name__ == "__main__":
    from torchinfo import summary

    for t_in_shape, t_out_shape in [[3, 1], [3, 5], [(3,), 1], [(3, 5), (7, 3)]]:
        net = FourierMLP(t_in_shape, t_out_shape, 2)
        batch_size = 13
        design_in_shpae = check_shape(t_in_shape)
        summary(net, input_size=[(1,), (batch_size, *design_in_shpae)])
        summary(net, input_size=[(batch_size,), (batch_size, *design_in_shpae)])
