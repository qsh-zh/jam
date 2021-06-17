#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imgio.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


import os as os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from . import backend
from .imgproc import dimshuffle


__all__ = [
    "imread",
    "imwrite",
    "imshow",
    "plt2pil",
    "plt2nd",
    "nd2pil",
    "pil2nd",
    "imgstack",
    "savefig",
    "ndimgs_in_row",
]


def imread(path, *, shuffle=False):
    if not osp.exists(path):
        return None
    i = backend.imread(path)
    if i is None:
        return None
    if shuffle:
        return dimshuffle(i, "channel_first")
    return i


def imwrite(path, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, "channel_last")
    backend.imwrite(path, img)


def imshow(title, img, *, shuffle=False):
    if shuffle:
        img = dimshuffle(img, "channel_last")
    backend.imshow(title, img)


def plt2pil(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plt2nd(fig):
    return np.array(plt2pil(fig))

def savefig(fig, fig_name):
    fig_path = fig_name.split("/")
    if len(fig_path) > 1:
        save_path = "/".join(fig_path[:-1])
        if not osp.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    fig.savefig(fig_name)


def imgstack(imgs, dpi=128):
    import matplotlib
    import matplotlib.pyplot as plt

    num_img = len(imgs)
    img_size = imgs[0].size
    with plt.style.context("img"):
        fig, axs = plt.subplots(
            num_img,
            1,
            figsize=(1 * img_size[0] / dpi, num_img * img_size[1] / dpi),
        )
        for cur_img, cur_ax in zip(imgs, axs):
            cur_ax.imshow(cur_img)

    return fig


def ndimgs_in_row(list_x, n, dpi=128, img_size=400):
    """
    args:
    list_x: list of elements that imshow can display
    n: number of the element in a row
    """
    length = len(list_x)
    idxes = np.linspace(0, length - 1, n, dtype=int)
    with plt.style.context("img"):
        fig, axs = plt.subplots(
            1, n, figsize=(n * img_size / dpi, img_size / dpi), dpi=dpi
        )
        for j, idx in enumerate(idxes):
            axs[j].imshow(list_x[idx])
            axs[j].set_title(idx)
    return fig

nd2pil=backend.pil_nd2img
pil2nd=backend.pil_img2nd