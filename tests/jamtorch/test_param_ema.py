# only verify ParamEMA works, low efficiency
from jamtorch.io import ParamEMA
import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0, 2.0]))


def main():
    net = Simple()
    ema = ParamEMA(0.5, 8, 1)
    ema.update_parameters(net)
    for i in range(10):
        with torch.no_grad():
            net.a += 1
            ema.update_parameters(net)
        print(f"{i:04d}\t {ema.shadow_params}")

    ema_model = Simple()
    ema.copy_to(ema_model)

    print("ema\t", ema_model.a)
    print("net\t", net.a)


if __name__ == "__main__":
    main()
