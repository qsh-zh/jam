import os
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from jammy.event import SimpleEventRegistry
from jammy.utils.enum import JamEnum
from jamtorch.io import attr_dict, save_ckpt,load_ckpt
import os.path as osp
from jamtorch.logging import get_logger

logger = get_logger()

__all__ = ["EvalState", "GeneticTrainer"]

class EvalState(JamEnum):
    NO = 1
    ITER = 2
    EPOCH = 3


class GeneticTrainer:
    def __init__(self, cfg, loss_fn):
        self._device = None
        self.train_set = None
        self.train_loader = None
        self.val_set = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn

        self.epoch_cnt = 0
        self.iter_cnt = 0
        self.best_loss = float("inf")
        self.ckpt_dir = getattr(cfg, "ckpt_dir") or os.getcwd()
        self._states = ["epoch_cnt", "iter_cnt", "best_loss", "ckpt_dir"]

        self._cfg = cfg
        self.eval_epoch = cfg.get("eval_epoch") or 1
        self.eval_iter = cfg.get("eval_iter") or -1
        if "gpu" in self._cfg:
            self.device = cfg["gpu"]


        self._event_manager = SimpleEventRegistry(
            {
                "epoch:start",
                "epoch:end",
                "epoch:before",
                "epoch:after",
                "step:start",
                "step:end",
                "step:summary",
                "forward:before",
                "forward:after",
                "backward:before",
                "backward:after",
                "val:start",
                "val:end",
                "val:step",  # quite strange naming and design
            }
        )

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, gpu):
        if isinstance(gpu, int):
            self._device = torch.device(f"cuda:{gpu}")
        elif isinstance(gpu, str) and len(gpu) < 2:
            self._device = torch.device(f"cuda:{gpu}")
        else:
            self._device = gpu

    def set_dataset(self, train_set, val_set=None, loader_cfg=None):
        self.train_set, self.val_set = train_set, val_set
        if loader_cfg is not None:
            return self.quick_loader(loader_cfg)

    def set_dataloader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def quick_loader(self, loader_cfg):
        if isinstance(loader_cfg, DictConfig):
            loader_cfg = OmegaConf.to_container(loader_cfg, resolve=True)
        if self.train_set is not None:
            self.train_loader = DataLoader(self.train_set, **loader_cfg)
        if self.val_set is not None:
            self.val_loader = DataLoader(self.val_set, **loader_cfg)

    def set_model_optim(self, model, optimizer=None):
        self.model, self.optimizer = model, optimizer
        if self._cfg.resume:
            self.load_ckpt(self._cfg.ckpt)

    def export_env(self):
        states = {key: getattr(self, key) for key in self._states}
        states["epoch_cnt"] +=1
        return states

    def load_env(self, cfg):
        for key, value in cfg.items():
            if value is None:
                continue
            setattr(self, key, value)

    def load_ckpt(self, filename="checkpoint"):
        state = load_ckpt(self.device, filename)
        self._impl_load_ckpt(self, state)
        self.load_env(state["env"])

    def _impl_load_ckpt(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def save_ckpt(self, val_loss=float("inf")):
        is_best = val_loss < self.best_loss
        self.best_loss = min(val_loss, self.best_loss)
        env_state = self.export_env()
        state_dict = {"env": env_state}
        ckpt_path = osp.join(self.ckpt_dir, f"ckpt-{self.iter_cnt}")
        best_path = osp.join(self.ckpt_dir, "best")
        state_dict.update(self._impl_save_ckpt())
        save_ckpt(state_dict, is_best, ckpt_path, best_path)

    def _impl_save_ckpt(self):
        return attr_dict(self, ["model", "optimizer"])

    def register_event(self, name, callback, verbose=True):
        """
        "epoch:start/end/before/after",(trainer)
        "step:start/end",(trainer)
        "step:summary",(trainer, loss, monitors, cmdviz)

        "forward:before",(trainer, feed_dict)
        "forward:after",(trainer, feed_dict, loss, monitors, cmdviz)
        "backward:before",(trainer, feed_dict, loss, monitors, cmdviz)
        "backward:after",(trainer, feed_dict, loss, monitors, cmdviz)

        "val:start/end",(trainer)
        "val:step",(trainer, feed_dict, loss, monitors, cmdviz)
        """

        if verbose:
            logger.info(
                "Register trainer event: name={}, callback={}.".format(
                    name, callback.__module__ + "." + callback.__name__
                )
            )
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._event_manager.trigger(name, *args, **kwargs)

    def eval(self):
        self.model.eval()
        self.trigger_event("val:start", self)
        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):
                loss, monitor, cmdviz_dict = self.loss_fn(
                    self.model, batch, is_train=False
                )
                self.trigger_event("val:step", self, batch, loss, monitor, cmdviz_dict)
        self.trigger_event("val:end", self)

    def eval_state(self):
        if self.val_loader is None:
            return EvalState.NO

        if self.eval_iter > 0 and (self.iter_cnt % self.eval_iter) == 0:
            if self.eval_epoch > 0 and ((self.epoch_cnt + 1) % self.eval_epoch) == 0:
                return EvalState.EPOCH
            return EvalState.ITER

        if self.iter_cnt % len(self.train_loader) == 0:
            if self.eval_epoch > 0 and ((self.epoch_cnt + 1) % self.eval_epoch) == 0:
                return EvalState.EPOCH
        return EvalState.NO

    def train(self):
        self.trigger_event("epoch:start", self)
        for self.epoch_cnt in range(self.epoch_cnt, self._cfg.epochs):
            self.trigger_event("epoch:before", self)
            for _, batch in enumerate(self.train_loader):
                self.iter_cnt += 1
                self.trigger_event("step:start", self)
                self.train_step(batch)
                self.trigger_event("step:end", self)
                state = self.eval_state()
                if state is EvalState.NO:
                    self.monitor_update()
                elif state is EvalState.ITER:
                    self._eval_iter()
                    self.monitor_update()
            if state is EvalState.EPOCH:
                self._eval_epoch()
                self.monitor_update()
            self.trigger_event("epoch:after", self)
        self.trigger_event("epoch:end", self)
        return self.best_loss

    def train_step(self, feed_dict):
        self.trigger_event("forward:before", self, feed_dict)
        loss, monitors, cmdviz_dict = self.loss_fn(self.model, feed_dict, is_train=True)
        self.trigger_event(
            "forward:after", self, feed_dict, loss, monitors, cmdviz_dict
        )

        if loss.requires_grad:
            self.optimizer.zero_grad()
            self.trigger_event(
                "backward:before", self, feed_dict, loss, monitors, cmdviz_dict
            )
            # The design is useful for accumulated loss
            if self.loss_backward(loss):
                self.trigger_event(
                    "backward:after", self, feed_dict, loss, monitors, cmdviz_dict
                )
                self.optimizer.step()
        self.trigger_event("step:summary", self, loss, monitors, cmdviz_dict)

    def _eval_iter(self):
        self.eval()

    def _eval_epoch(self):
        self.eval()

    def loss_backward(self, loss):
        loss.backward()
        return True

    def monitor_update(self):
        pass