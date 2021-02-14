# This is an exmaple to take advantage of trainer
# The design of trainer is a simple model for quick prototype
# For training process, user needs to implement the loss_fn, and other visualizization functions
from jamtorch.trainer import CfgTrainer, hydpath
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from jammy.image import imread, imwrite
from jammy.logging import Wandb
import jamtorch.prototype as jampt
from torch.optim.lr_scheduler import StepLR
import jamtorch.nn as nn
from omegaconf import OmegaConf
from einops import repeat
import hydra


def init_model(cfg):
    INPUT = 2 * cfg.fourier_dim if cfg.is_fourier else 2
    model = nn.MLP(INPUT, 3, 64, 3, is_batchnorm=cfg.is_batchnorm)
    return model


def mse_loss_fn(model, feed_dict, is_train):
    data, target = feed_dict
    data, target = data.float().to(jampt.device), target.to(jampt.device).float()
    output = model(data)
    loss = F.l1_loss(target, output)
    return loss, {}, {}

class ImgDataset(Dataset):
    def __init__(self, cfg):
        self.img = torch.FloatTensor(imread(hydpath(cfg.img)).copy()) / 255
        self.row = self.img.shape[0]
        self.column = self.img.shape[1]

        rows, columns = np.meshgrid(np.arange(self.column), np.arange(self.row))
        self.mesh_xs = torch.FloatTensor(
            np.stack([columns, rows], axis=2).reshape(-1, 2)
        ) # (row_th, col_th)
        if cfg.is_fourier:
            position = repeat(self.mesh_xs, 'w h -> w (c h)', c=cfg.fourier_dim)
            fre = np.arange(1, cfg.fourier_dim+1)[:,None] * np.array([2*np.pi / self.row, 2*np.pi / self.column])[None, :]
            fre = fre.reshape(1, -1) * 0.26
            self.mesh_xs = torch.sin(position * fre)

        self.pixels = self.img.view(self.column * self.row, 3)
        save_img = (self.pixels *255).view(self.row, self.column, 3)
        imwrite("gt.png", jampt.get_numpy(save_img).astype(np.uint8))

    def __len__(self):
        return self.row * self.column

    def __getitem__(self, idx):
        return self.mesh_xs[idx], self.pixels[idx]

def init_data(cfg):
    train_kwargs = {"batch_size": cfg.batch_size}
    test_kwargs = {"batch_size": cfg.batch_size}
    cuda_kwargs = {"shuffle": True, "pin_memory": True}
    trainset = ImgDataset(cfg)
    valset = ImgDataset(cfg)
    trainloader = DataLoader(trainset, **train_kwargs, **cuda_kwargs)
    valloader = DataLoader(valset, **test_kwargs, **cuda_kwargs)
    return trainloader, valloader

def eval_after_wraper(trainloader):
    trainset = trainloader.dataset
    def mse_val_after(trainer, *args):
        with torch.no_grad():
            output = trainer.ema.model(trainset.mesh_xs.float().to(jampt.device))
            img = output.view(trainset.row, trainset.column, 3)
            img = jampt.get_numpy(img)
            save_img = (img * 255).astype(np.uint8)
            imwrite(f"mse_{trainer.epoch_cnt}.jpg", save_img)
    return mse_val_after

def run(cfg):
    jampt.set_gpu_mode(cfg.cuda)
    trainloader, valloader = init_data(cfg.data)
    eval_after_fn = eval_after_wraper(trainloader)
    model = init_model(cfg.model).to(jampt.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)
    trainer = CfgTrainer(model, optimizer, mse_loss_fn, scheduler, cfg.trainer)
    trainer.register_event("val:after", eval_after_fn)
    trainer.set_monitor(cfg.trainer.wandb.log)
    trainer(trainloader, valloader)

@hydra.main(config_name="pixels.yaml")
def main(cfg):
    Wandb.launch(cfg, cfg.trainer.wandb.log, True)
    OmegaConf.set_struct(cfg, False)
    run(cfg)
    Wandb.finish()


if __name__ == "__main__":
    main()
