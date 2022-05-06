#!/usr/bin/env python3
# pylint: disable=global-variable-not-assigned
import functools
import os
import os.path as osp
from subprocess import Popen

import hydra
from omegaconf import DictConfig, OmegaConf

from jammy.logging import get_logger

logger = get_logger()

jam_dk_wrappers = []
dk_args = ["docker", "run", "-d", "--gpus", "all", "--privileged"]
print_args = ["\n", "\t".join(dk_args)]


def jamdk_wrapper(func):
    global jam_dk_wrappers, dk_args, print_args

    @functools.wraps(func)
    def _wrapped_func(*args, **kwargs):
        rtn = func(*args, **kwargs)
        if rtn is not None:
            dk_args.extend(rtn)
            print_args.append("\t".join(rtn))
        return dk_args

    jam_dk_wrappers.append(_wrapped_func)
    return _wrapped_func


def read_local_cfg(cfg):
    cfg_path = hydra.utils.to_absolute_path(cfg.file)
    if not osp.exists(cfg_path):
        return cfg
    logger.info("readding config from: " + cfg_path)
    local_cfg = OmegaConf.load(cfg_path)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, local_cfg, cli_cfg)
    return cfg


@jamdk_wrapper
def check_network(cfg):
    if cfg.network is None:
        return ["--network", "host"]
    return ["--network", cfg.network]


@jamdk_wrapper
def check_memeory(cfg):
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mem_gib = mem_bytes / (1024.0 ** 3)
    if cfg.memory is None:
        cfg.memory = int(mem_gib)
    return ["--shm-size", f"{cfg.memory}G"]


@jamdk_wrapper
def check_x_display(cfg):
    display = os.getenv("DISPLAY", None)
    if display is None or cfg.x is False:
        return None
    return [
        "--volume=/tmp/.X11-unix:/tmp/.X11-unix",
        "--volume=/tmp/.docker.xauth:/tmp/.docker.xauth",
        "-e",
        f"DISPLAY={display}",
    ]


@jamdk_wrapper
def check_ssh_mount(cfg):
    if cfg.ssh:
        ssh_path = osp.expanduser("~/.ssh/")
        return [f"--volume={ssh_path}:{ssh_path}:rw"]
    return None


@jamdk_wrapper
def check_jam_mount(cfg):
    if cfg.jam:
        jam_path = osp.expanduser("~/jam/")
        return [f"--volume={jam_path}:{jam_path}:rw"]
    return None


@jamdk_wrapper
def extra_mount(cfg):
    if cfg.mount is not None:
        rtn = []
        for local_f, dk_f in cfg.mount.items():
            if local_f[0] == "~":
                local_f = osp.expanduser(local_f)
            if local_f[0] != "/":
                local_f = hydra.utils.to_absolute_path(local_f)
            if dk_f is None:
                dk_f = local_f
            else:
                if dk_f[0] == "~":
                    dk_f = osp.expanduser(dk_f)
                if dk_f[0] != "/":
                    dk_f = hydra.utils.to_absolute_path(dk_f)
            rtn.append(f"--volume={local_f}:{dk_f}:rw")
        return rtn
    return None


@jamdk_wrapper
def extra_aug(cfg):
    if "extra" in cfg:
        if cfg.extra is not None:
            rtn = []
            for key, value in cfg.extra.items():
                rtn.append(f"--{key}={value}")
            return rtn
    return None


@jamdk_wrapper
def container_name(cfg):
    if cfg.name is not None:
        return ["--name", cfg.name]
    return None


@jamdk_wrapper
def img(cfg):
    if cfg.img is not None:
        return [f"{cfg.img}"]
    raise RuntimeError


@hydra.main(config_path="conf", config_name="jamdk")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg = read_local_cfg(cfg)
    for item in jam_dk_wrappers:
        item(cfg)
    logger.debug(dk_args)
    logger.info(f"stating {cfg.name}" + "\n".join(print_args))
    Popen(dk_args)  # pylint: disable=consider-using-with


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
