import torch

import jamtorch.prototype as jampt
from jammy.utils.gpu import select_gpu

__all__ = [
    "set_best_device",
]
# from gpustat import GPUStatCollection

# import subprocess
# def get_pname(id):
#     p = subprocess.Popen(["ps -o cmd= {}".format(id)], stdout=subprocess.PIPE, shell=True)
#     return str(p.communicate()[0])

# def cuda_process():
#     gpu_stat = GPUStatCollection.new_query()


def set_best_device():
    gpu_id = select_gpu()
    torch.cuda.set_device(gpu_id)
    jampt.set_gpu_mode(True, gpu_id)
    return torch.device(gpu_id)
