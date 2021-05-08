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

from . import backend
from .imgproc import dimshuffle


__all__ = ["imread", "imwrite", "imshow", "fig2img", "imgstack", "savefig"]


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


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def savefig(fig, fig_name):
    fig_path = fig_name.split("/")
    if len(fig_path) > 1:
        save_path = "/".join(fig_path[:-1])
        if not osp.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    fig.savefig(fig_name)


def imgstack(imgs):
    import matplotlib
    import matplotlib.pyplot as plt

    img_size = imgs[0].size
    resolution = 40

    num_img = len(imgs)
    with plt.style.context("img"):
        fig, axs = plt.subplots(
            num_img,
            1,
            figsize=(1 * img_size[0] / resolution, num_img * img_size[1] / resolution),
        )
        for cur_img, cur_ax in zip(imgs, axs):
            cur_ax.imshow(cur_img)

    return fig
