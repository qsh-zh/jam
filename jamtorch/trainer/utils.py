import os.path as osp

import torch
import wandb
import jammy.io as io
from jammy.logging import get_logger

__all__ = ["load_checkpoint", "save_checkpoint", "checkpoint_state"]

logger = get_logger()


def wandb_trainer_update(stage, record_dict, step_cnt):
    book = {f"{stage}/{k}": v for k, v in record_dict.items()}
    wandb.log(book, step=step_cnt)


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if osp.isfile(filename):
        logger.critical("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        metrics = checkpoint["metrics"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.critical("==> Done")
        return epoch, it, metrics
    else:
        logger.critical("==> Checkpoint '{}' not found".format(filename))
        return None


def save_checkpoint(state, is_best, filename="checkpoint", bestname="model_best"):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        io.copy(filename, "{}.pth.tar".format(bestname))
        logger.info("Save best up to now: {}".format(filename))


def checkpoint_state(model=None, optimizer=None, metrics=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "metrics": metrics,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }
