# This is an exmaple to take advantage of trainer
# The design of trainer is a simple model for quick prototype
# For training process, user needs to implement the loss_fn, and other visualizization functions
from jamtorch.trainer import Trainer
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from jammy.image import imread, imwrite
from jammy.logging import Wandb
import jamtorch.prototype as jampt
from omegaconf import OmegaConf
cfg = OmegaConf.load('pixel.yaml')
cfg.wandb.log = False
Wandb.launch(cfg, cfg.wandb.log)
img = imread('position_mlp.jpg').reshape(-1, 3)


def init_model(cfg):
    INPUT = 2 + 2 * cfg.sin_order if cfg.sin_condition else 2
    if cfg.loss == 'mse':
        OUTPUT = 3
    elif cfg.loss == 'nll':
        OUTPUT = 3 * 255
    else:
        raise RuntimeError()
    model = jampt.MLP(INPUT, 3, 64, OUTPUT,
                      is_batchnorm=cfg.is_batchnorm).to(jampt.device)
    return model


def mse_loss_fn(model, feed_dict, is_train):
    data, target = feed_dict
    data, target = data.to(jampt.device), target.to(jampt.device)
    output = model(data)
    loss = F.mse_loss(target, output)
    if is_train:
        return loss, {}, {}
    pred = output.type(torch.long)
    correct = pred.eq(target.view_as(pred)).sum(axis=1) == 3
    correct = correct.sum()
    return loss, {}, {"correct": correct}


def nll_loss_fn(model, feed_dict, is_train):
    data, target = feed_dict
    target.flatten()
    B = data.shape[0]
    output = model(data).view(B * 3, 256)
    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
    if is_train:
        return loss, {}, {}
    pred = output.argmax(dim=1, keepdim=True)
    pixel_mse = F.mse_loss(pred, target.view_as(pred))
    correct = pred.eq(target.view_as(pred)).view(B, -1).sum(axis=1) == 3
    correct = correct.sum()
    return loss, {}, {"pixel_mse": pixel_mse, "correct": correct}


class ImgDataset(Dataset):
    def __init__(self, img_path, is_fourier):
        # ! fixme
        # self.img = jampt.FloatTensor(imread(img_path).astype('float32'))
        self.img = torch.from_numpy(imread(img_path)).type(torch.long)
        self.row = self.img.shape[0]
        self.column = self.img.shape[1]

        rows, columns = np.meshgrid(np.arange(self.column), np.arange(self.row))
        self.mesh_xs = torch.FloatTensor(
            np.stack([columns, rows], axis=2).reshape(-1, 2))
        if is_fourier:
            space = np.stack([columns * np.pi / self.row, rows *
                              np.pi / self.column], axis=2).reshape(-1, 2)
            sin = []
            for i in range(1, 1 + cfg.sin_order):
                sin.append(np.sin(space).T)
            sin = torch.FloatTensor(sin).reshape(2 * cfg.sin_order, -1).T
            self.mesh_xs = torch.cat((self.mesh_xs, sin), dim=1)

        self.pixels = self.img.view(-1, 3)

    def __len__(self):
        return self.row * self.column

    def __getitem__(self, idx):
        return self.mesh_xs[idx], self.pixels[idx]


def init_fn(cfg):
    if cfg.loss == 'mse':
        return mse_loss_fn, mse_val_after
    elif cfg.loss == 'nll':
        return nll_loss_fn, nll_val_after
    else:
        raise RuntimeError


train_kwargs = {'batch_size': img.shape[0]}
test_kwargs = {'batch_size': img.shape[0]}
if cfg.use_cuda:
    jampt.set_gpu_mode(True, 0)
    cuda_kwargs = {'shuffle': True,
                   'pin_memory': True,
                   'num_workers': 6}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
trainset = ImgDataset('position_mlp.jpg', cfg.sin_condition)
valset = ImgDataset('position_mlp.jpg', cfg.sin_condition)


def mse_val_after(trainer, epoch):
    global trainset
    with torch.no_grad():
        output = trainer.model(trainset.mesh_xs.to(jampt.device))
        pred = output.type(torch.long)
        img = pred.view(trainset.row, trainset.column, 3)
        img = jampt.get_numpy(img)
        imwrite(f"mse_{epoch}.jpg", img)


def nll_val_after(trainer, epoch):
    global trainset
    with torch.no_grad():
        output = trainer.model(trainset.mesh_xs).view(-1, 256)
        pred = output.argmax(dim=1, keepdim=True)
        img = pred.view(trainset.row, trainset.column, 3)
        img = jampt.get_numpy(img)
        imwrite(f"nll_{epoch}.jpg", img)


trainloader = DataLoader(trainset, **train_kwargs)
valloader = DataLoader(valset, **test_kwargs)

model = init_model(cfg)
loss_fn, val_after = init_fn(cfg)


optimizer = optim.Adam(model.parameters(), lr=cfg.lr)


trainer = Trainer(model, optimizer, loss_fn)
trainer.register_event("val:after", val_after)
trainer.set_monitor(cfg.wandb.log)

trainer.train(cfg.N_epoch, trainloader, valloader)
