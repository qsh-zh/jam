import os
import os.path as osp

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from jammy.event import SimpleEventRegistry
from jammy.utils.enum import JamEnum
from jamtorch.io import attr_dict, load_ckpt, save_ckpt
from jamtorch.logging import get_logger

from . import trainer_fn

logger = get_logger()

__all__ = ["EvalState", "GeneticTrainer"]


class EvalState(JamEnum):
    NO = 1  # pylint: disable= invalid-name
    ITER = 2
    EPOCH = 3


class GeneticTrainer:  # pylint: disable=too-many-instance-attributes
    def __init__(self, cfg, loss_fn):
        self._device = None
        self.train_set = None
        self.train_loader = None
        self.val_set = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = loss_fn

        self.epoch_cnt = 0
        self.iter_cnt = 0
        self.best_loss = float("inf")
        self.ckpt_dir = getattr(cfg, "ckpt_dir") or os.getcwd()
        self._states = ["epoch_cnt", "iter_cnt", "best_loss"]
        self.latest_ckpt = None

        self._cfg = cfg
        self.eval_epoch = cfg.get("eval_epoch") or 1
        self.eval_iter = cfg.get("eval_iter") or -1
        self.ratio_forback = cfg.get("ratio_forback") or 1
        self.use_amp = cfg.get("use_amp") or False
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
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
                "trainer:export",
                "trainer:load",
            }
        )
        if cfg.clip_grad and cfg.clip_grad > 0:
            trainer_fn.register_grad_clip(self, cfg.clip_grad)

    @property
    def mmodel(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

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
            self.quick_loader(loader_cfg)

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
        if next(model.parameters()).device == torch.device("cpu"):
            model = model.to(self.device)
        self.model, self.optimizer = model, optimizer
        # resume will loss one epoch, it is acceptable
        if self._cfg.resume:
            self.load_ckpt(self._cfg.ckpt)

    def export_env(self):
        states = {key: getattr(self, key) for key in self._states}
        return states

    def load_env(self, cfg):
        for key, value in cfg.items():
            if value is None:
                continue
            if key in self._states:
                setattr(self, key, value)

    def load_ckpt(self, filename="checkpoint"):
        state = load_ckpt(self.device, filename)
        self._impl_load_ckpt(state)
        self.load_env(state["env"])
        self.trigger_event("trainer:load", self, state)

    def _impl_load_ckpt(self, state):
        msg_model = self.model.load_state_dict(state["model"])
        logger.critical(f"load model ckpt {msg_model}")
        if "optimizer" in state:
            msg_optimizer = self.optimizer.load_state_dict(state["optimizer"])
            logger.critical(f"load optimizer ckpt {msg_optimizer}")
        else:
            logger.critical("no optimizer in state")
        self._impl_load_amp_scaler(state)

    def _impl_load_amp_scaler(self, state):
        if self.use_amp:
            if "amp_scaler" in state and len(state["amp_scaler"]) == 0:
                self.amp_scaler.load_state_dict(state["amp_scaler"])
            else:
                logger.critical("enabled amp but amp_scaler not found")

    def save_ckpt(self, val_loss=float("inf")):
        is_best = val_loss < self.best_loss
        self.best_loss = min(val_loss, self.best_loss)
        env_state = self.export_env()
        state_dict = {"env": env_state}
        ckpt_path = osp.join(self.ckpt_dir, f"ckpt-{self.iter_cnt}")
        self.latest_ckpt = ckpt_path
        best_path = osp.join(self.ckpt_dir, "best")
        state_dict.update(self._impl_save_ckpt())
        self.trigger_event("trainer:export", self, state_dict)

        save_ckpt(state_dict, is_best, ckpt_path, best_path)

    def _impl_save_ckpt(self):
        if self.use_amp:
            return attr_dict(self, ["model", "optimizer", "amp_scaler"])
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

        "trainer:export",(trainer, state_dict)
        "trainer:load",(trainer, state)
        """

        if verbose:
            logger.info(
                "Register trainer event: name={}, callback={}.".format(
                    name, callback.__module__ + "." + callback.__name__
                )
            )
        else:
            logger.debug(
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
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, monitor, cmdviz_dict = self.loss_fn(
                        self, batch, is_train=False
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
        self.optimizer.zero_grad()
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
        self.eval()
        return self.best_loss

    def train_step(self, feed_dict):
        self.trigger_event("forward:before", self, feed_dict)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss, monitors, cmdviz_dict = self.loss_fn(self, feed_dict, is_train=True)

        self.trigger_event(
            "forward:after", self, feed_dict, loss, monitors, cmdviz_dict
        )

        if loss.requires_grad:
            self.trigger_event(
                "backward:before", self, feed_dict, loss, monitors, cmdviz_dict
            )
            # The design is useful for accumulated loss
            if self.loss_backward(loss):
                self.trigger_event(
                    "backward:after", self, feed_dict, loss, monitors, cmdviz_dict
                )
                self.optimizer_step()
        self.trigger_event("step:summary", self, loss, monitors, cmdviz_dict)

    def _eval_iter(self):
        self.eval()

    def _eval_epoch(self):
        self.eval()

    def optimizer_step(self):
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        self.optimizer.zero_grad()

    def loss_backward(self, loss):
        self.amp_scaler.scale(loss).backward()
        if self.iter_cnt % self.ratio_forback == 0:
            # TODO: unscales the gradient
            self.amp_scaler.unscale_(self.optimizer)
            return True
        return False

    def monitor_update(self):
        pass

    def forward(self, *args, **kwargs):
        if self.model is not None:
            return self.model(*args, **kwargs)
        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
