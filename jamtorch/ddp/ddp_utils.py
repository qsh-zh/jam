import datetime
import functools
import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

import jammy.comm as comm

__all__ = [
    "is_master",
    "is_dist",
    "master_first",
    "master_only",
    "ddp_runner",
    "ddp_loaders",
    "get_world_size",
    "barrier",
]


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


def is_dist():
    return dist.is_initialized()


def get_world_size():
    if is_dist():
        return dist.get_world_size()
    return 1


def barrier():
    if is_dist():
        dist.barrier()


@contextmanager
def master_first():
    if not is_master():
        barrier()
    yield
    if dist.is_initialized() and is_master():
        barrier()


def master_only(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        if is_master():
            return func(*args, **kwargs)
        return

    return new_func


def ddp_setup(rank, world_size, working_dir, cfg):
    # TODO: discuss the necessarity?
    cfg.rank = rank
    cfg.gpu = rank
    cfg.cwd = working_dir
    cfg.world_size = world_size

    # different random seed for different process
    torch.manual_seed(rank)

    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=cfg.dist.mode, rank=rank, world_size=world_size, timeout=timeout
    )
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def ddp_runner(func):
    @functools.wraps(func)
    def new_fn(rank, n_gpu, working_dir, config):
        ddp_setup(rank, n_gpu, working_dir, config.trainer)
        if config.trainer.dist.adjust_lr:
            if "optimizer" in config:
                config.optimizer.lr = config.optimizer.lr * config.trainer.world_size
        value = func(config)
        cleanup()
        return value

    return new_fn


def ddp_loaders(train_set, val_set, rank=None, world_size=None, **dl_kwargs):
    """instantiate dataloaders, deal with ddp and non-ddp, return None sampler

    :type train_set: torch.utils.data.Dataset
    :type val_set: torch.utils.data.Dataset
    :param rank: [description], defaults to None
    :type rank: [type], optional
    :param world_size: [description], defaults to None
    :type world_size: [type], optional
    :return: train_loader, train_sampler, val_loader, val_sampler
    """
    if not dist.is_initialized():
        # no dist trainining
        if train_set is not None:
            train_loader = torch.utils.data.DataLoader(
                train_set, shuffle=True, **dl_kwargs
            )
        else:
            train_loader = None
        if val_set is not None:
            val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, **dl_kwargs)
        else:
            val_loader = None

        return train_loader, None, val_loader, None

    if train_set is not None:
        train_sampler = torch.utils.data.DistributedSampler(train_set, world_size, rank)
        train_loader = torch.utils.data.DataLoader(
            train_set, sampler=train_sampler, **dl_kwargs
        )

    else:
        train_sampler = None
        train_loader = False

    if val_set is not None:
        val_sampler = torch.utils.data.DistributedSampler(val_set, world_size, rank)
        val_loader = torch.utils.data.DataLoader(
            val_set, sampler=val_sampler, **dl_kwargs
        )
    else:
        val_sampler = None
        val_loader = None

    return train_loader, train_sampler, val_loader, val_sampler


def fast_acc_grad_loss(trainer, loss):
    is_step = trainer.iter_cnt % trainer.ratio_forback != 0
    sync_context = (
        trainer.model.no_sync if dist.is_initialized() and is_step else nullcontext
    )
    # avoid sync gradient
    with sync_context:
        loss.backward()
    if is_step:
        return True

    return False


def prepare_cfg(cfg):
    # find good ports
    free_port = comm.find_free_port()
    cfg.trainer.dist.master_port = str(free_port)
