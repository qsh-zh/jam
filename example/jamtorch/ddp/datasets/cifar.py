import torchvision
import torch
import jammy.utils.hyd as hyd
import os

def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = torchvision.datasets.CIFAR10(root=hyd.hydpath('/tmp/data/train'), train=True, 
        download=True, transform=transform)
    val = torchvision.datasets.CIFAR10(root=hyd.hydpath('/tmp/data/val'), train=False, 
        download=True, transform=transform)
    return train, val
