from omegaconf import OmegaConf
import os.path as osp
import hydra

__all__ = ["hydpath"]


def hydpath(input_path=None):
    if input_path is None or len(input_path) == 0:
        return hydra.utils.to_absolute_path(".")
    if input_path[0] == "/":
        return input_path
    if input_path[0] == "~":
        return osp.expanduser(input_path)
    return hydra.utils.to_absolute_path(input_path)
