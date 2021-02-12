from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from jamtorch.trainer import CfgTrainer, step_lr
from jammy.logging import Wandb
import hydra
from omegaconf import OmegaConf


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# TODO: should consider process after eval
def build_loss(device):
    def loss_fn(model, feed_dict, is_train, device=device):
        data, target = feed_dict
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        if is_train:
            return loss, {}, {}

        pred = output.argmax(dim=1, keepdim=True)
        correct = 1.0 * pred.eq(target.view_as(pred)).sum() / len(pred)
        return loss, {}, {"correct": correct}

    return loss_fn


def run(cfg):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("using CUDA")

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwcfg = {"batch_size": cfg.data.batch_size}
    test_kwcfg = {"batch_size": cfg.data.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwcfg.update(cuda_kwargs)
        test_kwcfg.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_path = osp.join(hydra.utils.get_original_cwd(), "data")
    dataset1 = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwcfg)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwcfg)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)

    trainer = CfgTrainer(model, optimizer, build_loss(device), scheduler, cfg.trainer)
    trainer.set_monitor(cfg.trainer.wandb.log)
    trainer.register_event("epoch:after", step_lr)
    trainer(train_loader, test_loader)


@hydra.main(config_name="mnist.yaml")
def main(cfg):
    Wandb.launch(cfg, cfg.trainer.wandb.log, True)
    OmegaConf.set_struct(cfg, False)
    run(cfg)
    Wandb.finish()


if __name__ == "__main__":
    main()
