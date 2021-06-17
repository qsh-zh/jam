import torch
import numpy as np

def gaussian_kde_log_eval(samples, query):
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std ** 2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)
