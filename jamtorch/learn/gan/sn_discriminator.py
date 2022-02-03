from torch import nn

__all__ = [
    "Discriminator",
]

# pylint: disable=invalid-name, too-many-arguments


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=3,
        pad=1,
        d_spectral_norm=False,
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        ksize=3,
        pad=1,
        d_spectral_norm=False,
        activation=nn.ReLU(),
        downsample=False,
    ):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=ksize, padding=pad
        )
        self.c2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=ksize, padding=pad
        )
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(
        self, df_dim, in_channel=3, d_spectral_norm=False, activation=nn.PReLU()
    ):
        super().__init__()
        self.ch = df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(
            in_channel, self.ch, d_spectral_norm=d_spectral_norm
        )
        self.block2 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=True,
        )
        self.block3 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=False,
        )
        self.block4 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=False,
        )
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
