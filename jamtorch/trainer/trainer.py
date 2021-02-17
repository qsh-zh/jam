import os
import os.path as osp
import time

import torch

from tqdm.auto import tqdm, trange
from jammy.cli.cmdline_viz import CmdLineViz
from jammy.event import SimpleEventRegistry
from jammy.logging import get_logger
from jammy.utils.enum import JamEnum
from jammy.utils.meter import AverageMeter
from jamtorch.utils.meta import as_float

from .monitor import group_prefix
from .trainer_monitor import TrainerMonitor
from .utils import *
from jammy.utils.hyd import *

logger = get_logger()

__all__ = ["Trainer"]


def cuda_time(sync=True):
    if sync:
        torch.cuda.synchronize()
    return time.time()


class EvalState(JamEnum):
    NO = 1
    ITER = 2
    EPOCH = 3


class Trainer:
    def __init__(self, model, optimizer, loss_fn, lr_scheduler=None, cfg=dict()):
        (self.model, self.optimizer, self.loss_fn, self.lr_scheduler,) = (
            model,
            optimizer,
            loss_fn,
            lr_scheduler,
        )
        self._event_manager = SimpleEventRegistry(
            {
                "epoch:start",
                "epoch:finish",
                "epoch:before",
                "epoch:after",
                "step:before",
                "step:after",
                "forward:before",
                "forward:after",
                "backward:before",
                "backward:after",
                "val:before",
                "val:after",
                "val:epoch",  # quite strange naming and design
            }
        )
        self.cmdviz = CmdLineViz()
        self.trainer_monitor = None

        # cache state accross different methods
        self.cur_monitor = dict()
        self.cur_cmdviz = dict()

        # trainer env states
        self._cfg = cfg
        self.load_env(cfg)

    def load_env(self, dict_cfg):
        self.eval_epoch = dict_cfg.get("eval_epoch") or 1
        self.eval_iter = dict_cfg.get("eval_iter") or -1
        self.best_loss = dict_cfg.get("best_loss") or float("inf")
        self.checkpoint_dir = dict_cfg.get("ckptDir") or os.getcwd()
        self.iter_cnt = dict_cfg.get("iter_cnt") or 0
        self.epoch_cnt = dict_cfg.get("epoch_cnt") or 0

    def export_env(self):
        return {
            "eval_epoch": self.eval_epoch,
            "eval_iter": self.eval_iter,
            "best_loss": self.best_loss,
            "ckptDir": self.checkpoint_dir,
            "iter_cnt": self.iter_cnt,
            "epoch_cnt": self.epoch_cnt,
        }

    def set_monitor(self, is_wandb, tblogger=False):
        """
        docstring
        """
        self.trainer_monitor = TrainerMonitor(is_wandb, tblogger)

    def load_ckpt(self, filename="checkpoint"):
        env = load_checkpoint(self.model, self.optimizer, hydpath(filename))
        self.load_env(env)

    def __call__(
        self,
        n_epochs,
        train_loader,
        val_loader,
        trainer_ckpt="checkpoint",
        is_resume=False,
    ):
        """
        resume training or start a new training
        """
        # FIXME
        if is_resume:
            self.load_ckpt(trainer_ckpt)
            start_epoch = self.epoch_cnt + 1
            start_iter = self.iter_cnt + 1
            loss = self.best_loss
        else:
            start_epoch, start_iter, loss = 0, 0, float("inf")
        return self.train(
            n_epochs, train_loader, val_loader, loss, start_iter, start_epoch
        )

    def eval_impl(self, data_loader, **kwargs):
        self.model.eval()
        self.trigger_event("val:before", self, data_loader)

        loss_meter = AverageMeter()
        with tqdm(total=len(data_loader), leave=False, desc="val") as pbar:
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    # FIXME what is the usage of momitot, should be fix the api, make an agreement
                    loss, monitor, cmdviz_dict = self.loss_fn(
                        self.model, batch, is_train=False
                    )
                    # FIXME bug
                    self.trigger_event(
                        "val:epoch", self, batch, loss, monitor, cmdviz_dict
                    )
                    loss_meter.update(as_float(loss))
                    f_cmdviz = as_float(cmdviz_dict)
                    f_cmdviz.update({"loss": loss_meter.val})
                    self.cmdviz.update("eval", f_cmdviz)

                    pbar.update()
                    pbar.set_postfix({"loss": loss_meter.avg})

        self.trigger_event("val:after", self, data_loader)

        self.val_loss_ckpt(loss_meter.avg)
        self.cur_monitor.update(group_prefix("eval", self.cmdviz.meter["eval"].avg))

    def val_loss_ckpt(self, val_loss):
        is_best = val_loss < self.best_loss
        self.best_loss = min(val_loss, self.best_loss)
        if self.checkpoint_dir is not None:
            state = checkpoint_state(self.model, self.optimizer)
            state["env"] = self.export_env()

            save_checkpoint(
                state,
                is_best,
                osp.join(self.checkpoint_dir, "checkpoint"),
            )

    def loss_backward(self, loss):
        loss.backward()
        return True

    def train_step(self, feed_dict):
        self.trigger_event("step:before", self)
        self.trigger_event("forward:before", self, feed_dict)
        loss, monitors, cmdviz_dict = self.loss_fn(self.model, feed_dict, is_train=True)
        self.trigger_event(
            "forward:after", self, feed_dict, loss, monitors, cmdviz_dict
        )
        self.trigger_event(
            "backward:before", self, feed_dict, loss, monitors, cmdviz_dict
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

        return self._train_step_after(loss, monitors, cmdviz_dict)

    def _train_step_after(self, loss, monitors, cmdviz_dict):
        self.trigger_event("step:after", self)

        loss_f = as_float(loss)
        cmdviz_f = as_float(cmdviz_dict)
        cmdviz_f.update({"loss": loss_f})

        monitors = as_float(monitors)
        monitors.update(cmdviz_f)

        self.cur_monitor.update(group_prefix("train", monitors))
        self.cmdviz.update("train", cmdviz_f)
        return loss_f

    def register_event(self, name, callback):
        """
        "epoch:start",(trainer)
        "epoch:finish",(trainer)
        "epoch:before",(trainer)
        "epoch:after",(trainer)
        "step:before",(trainer)
        "step:after",(trainer)
        "forward:before",(trainer, feed_dict)
        "forward:after",(trainer, feed_dict, loss, monitors, cmdviz)
        "backward:before",(trainer, feed_dict, loss, monitors, cmdviz)
        "backward:after",(trainer, feed_dict, loss, monitors, cmdviz)
        "val:before",(trainer, dataloader)
        "val:after",(trainer, dataloader)
        "val:epoch",(trainer, feed_dict, loss, monitors, cmdviz)
        """

        logger.info(
            "Register trainer event: name={}, callback={}.".format(
                name, callback.__module__ + "." + callback.__name__
            )
        )
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._event_manager.trigger(name, *args, **kwargs)

    # TODO: strange API
    def train(
        self,
        n_epochs,
        train_loader,
        test_loader=None,
        best_loss=float("inf"),
        start_it=0,
        start_epoch=0,
    ):
        num_batch = len(train_loader)
        with trange(
            start_epoch, n_epochs, desc="epochs", dynamic_ncols=True
        ) as tbar, tqdm(
            total=num_batch, leave=False, desc="train", dynamic_ncols=True
        ) as pbar:
            self.trigger_event("epoch:start", self)
            for self.epoch_cnt in tbar:
                self.trigger_event("epoch:before", self)
                for batch_cnt, batch in enumerate(train_loader):
                    loss = self.train_step(batch)
                    self.iter_cnt += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=self.iter_cnt, loss=loss))
                    state = self.eval_state(test_loader, batch_cnt, num_batch)
                    if state is EvalState.NO:
                        self.monitor_update()
                    elif state is EvalState.ITER:
                        pbar.close()
                        self.eval_impl(test_loader)
                        self.cmdviz.flush()
                        pbar = tqdm(
                            total=num_batch,
                            leave=False,
                            desc="train",
                            dynamic_ncols=True,
                            initial=batch_cnt + 1,
                        )
                        self.monitor_update()
                pbar.close()
                if state is EvalState.EPOCH:
                    self.eval_impl(test_loader)
                    self.cmdviz.flush()
                    self.monitor_update()

                self.trigger_event("epoch:after", self)
                pbar = tqdm(
                    total=num_batch,
                    leave=False,
                    desc="train",
                    dynamic_ncols=True,
                )

            self.trigger_event("epoch:finish", self)
        pbar.clear()
        return self.best_loss

    def eval_state(self, val_loader, batch_cnt, num_batch):
        if val_loader is None:
            return EvalState.NO
        if self.eval_iter > 0 and ((self.iter_cnt + 1) % self.eval_iter) == 0:
            if self.eval_epoch > 0 and ((self.epoch_cnt + 1) % self.eval_epoch) == 0:
                return EvalState.EPOCH
            return EvalState.ITER
        if batch_cnt + 1 == num_batch:
            if self.eval_epoch > 0 and ((self.epoch_cnt + 1) % self.eval_epoch) == 0:
                return EvalState.EPOCH
        return EvalState.NO

    def monitor_update(self):
        if self.trainer_monitor:
            self.trainer_monitor.update(
                {
                    **self.cur_monitor,
                    "epoch": self.epoch_cnt,
                    "iter": self.iter_cnt,
                }
            )
        self.cur_monitor = dict()
