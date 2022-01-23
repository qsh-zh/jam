import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from einops import rearrange
from torchvision.utils import make_grid, save_image

__all__ = [
    "show_imgs",
    "show_imgs_traj",
    "save_seperate_imgs",
]


def show_imgs(img_data, nrow=10):
    grid_img = make_grid(img_data, nrow)
    plt.figure(figsize=(nrow, 8))
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0).cpu())


def show_imgs_traj(traj, num_img, num_steps):
    idx = np.linspace(0, len(traj) - 1, num_steps, dtype=int)
    imgs = []
    for cur_idx in idx:
        imgs.append(traj[cur_idx][:num_img])
    imgs = th.cat(imgs)

    imgs = rearrange(imgs, "(n b) ... -> (b n) ...", b=num_img)
    show_imgs(imgs, num_steps)


def save_seperate_imgs(sample, sample_path, cnt):
    batch_size = len(sample)
    for i in range(batch_size):
        save_image((sample[i] + 1.0) / 2, os.path.join(sample_path, f"{cnt:07d}.png"))
        cnt += 1
