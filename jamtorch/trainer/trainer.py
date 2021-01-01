import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import tqdm
import time
from jammy.utils.meter import GroupMeters, AverageMeter
from jammy.event import SimpleEventRegistry
from jammy.logging import get_logger
from jammy.cli.cmdline_viz import CmdLineViz

from jamtorch.utils.meta import as_float

from .utils import *
from .trainer_monitor import TrainerMonitor
from .monitor import group_prefix

logger = get_logger()

__all__ = ["Trainer"]


def cuda_time(sync=True):
    if sync:
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            loss_fn,
            lr_scheduler=None,
            bnm_scheduler=None):
        self.model, self._optimizer, self.loss_fn, self.lr_scheduler, self.bnm_scheduler = (
            model, optimizer, loss_fn, lr_scheduler, bnm_scheduler)
        self._event_manager = SimpleEventRegistry({
            'epoch:start', 'epoch:finish',
            'epoch:before', 'epoch:after',
            'step:before', 'step:after',
            'forward:before', 'forward:after',
            'backward:before', 'backward:after',
            'val:before', 'val:after', 'val:epoch',
        })
        self.cmdviz = CmdLineViz()
        self.trainer_monitor = None

        # FIXME!
        self.eval_frequency = -1
        self.checkpoint_dir = os.getcwd()

    def set_monitor(self, is_wandb, tblogger=False):
        """
        docstring
        """
        self.trainer_monitor = TrainerMonitor(is_wandb, tblogger)

    def load_ckpt(self, filename="checkpoint"):
        return load_checkpoint(self.model, self._optimizer, filename)

    def eval_epoch(self, data_loader, epoch, **kwargs):
        self.model.eval()

        loss_meter = AverageMeter()
        with tqdm.tqdm(total=len(data_loader), leave=False, desc="val") as pbar:
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    loss, minitor, cmdviz_dict = self.loss_fn(
                        self.model, batch, is_train=False)
                    # FIXME bug
                    self.trigger_event(
                        "val:epoch", batch, loss, minitor, cmdviz_dict)
                    loss_meter.update(as_float(loss))
                    f_cmdviz = as_float(cmdviz_dict)
                    f_cmdviz.update({"loss": loss_meter.val})
                    self.cmdviz.update("eval", f_cmdviz)


                    pbar.update()
                    pbar.set_postfix({"loss": loss_meter.avg})

        self.trigger_event("val:after", self, epoch)
        # TODO: rigid
        monitor = self.cmdviz.meter["eval"].avg
        self.cmdviz.flush()
        return loss_meter.avg, group_prefix("eval", monitor)

    def train_step(self, feed_dict, measure_time=False, grad_clip=0):
        if hasattr(self.model, 'train_step'):
            return self.model.train_step(self._optimizer, feed_dict)

        metrics = dict()
        self.trigger_event('step:before', self)

        if measure_time:
            end_time = cuda_time()

        self.trigger_event('forward:before', self, feed_dict)
        #! check readme return type of loss_fn
        loss, monitors, cmdviz_dict = self.loss_fn(
            self.model, feed_dict, is_train=True)
        self.trigger_event(
            'forward:after',
            self,
            feed_dict,
            loss,
            monitors,
            cmdviz_dict)

        if measure_time:
            metrics["time/forward"] = cuda_time() - end_time
            end_time = cuda_time(False)

        if measure_time:
            metrics['time/loss'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self._optimizer.zero_grad()
        self.trigger_event(
            'backward:before',
            self,
            feed_dict,
            loss,
            monitors,
            cmdviz_dict)
        if loss.requires_grad:
            loss.backward()
            if grad_clip > 0:
                from torch.nn.utils.clip_grad import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), grad_clip)

        if measure_time:
            metrics['time/backward'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self.trigger_event(
            'backward:after',
            self,
            feed_dict,
            loss,
            monitors,
            cmdviz_dict)
        if loss.requires_grad:
            self._optimizer.step()

        if measure_time:
            metrics['time/optimize'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self.trigger_event('step:after', self, metrics)

        #! FIXME! should be place here
        loss_f = as_float(loss)
        cmdviz_f = as_float(cmdviz_dict)
        cmdviz_f.update({"loss": loss_f})
        self.cmdviz.update("train", cmdviz_f)
        monitors.update(cmdviz_f)
        # think twice on return value
        return loss_f, group_prefix("train", monitors)

    def register_event(self, name, callback):
        logger.info('Register trainer event: name={}, callback={}.'.format(
            name, callback.__module__ + '.' + callback.__name__))
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._event_manager.trigger(name, *args, **kwargs)

    # TODO: strange API
    def train(
            self,
            n_epochs,
            train_loader,
            test_loader=None,
            best_loss=float('inf'),
            start_it=0, start_epoch=0):
        # TODO: need to fix the eval_freqeuncy
        eval_frequency = (
            self.eval_frequency if self.eval_frequency > 0 else len(train_loader))

        it = start_it
        with tqdm.trange(
            start_epoch, n_epochs, desc="epochs", dynamic_ncols=True
        ) as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train", dynamic_ncols=True
        ) as pbar:
            self.trigger_event('epoch:start', self)
            for epoch in tbar:
                self.trigger_event("epoch:before", self)
                for batch in train_loader:
                    loss, monitor = self.train_step(batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it, loss=loss))
                    tbar.refresh()

                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, eval_monitor = self.eval_epoch(test_loader, epoch)
                            monitor.update(eval_monitor)
                            if self.checkpoint_dir is not None:
                                is_best = val_loss < best_loss
                                best_loss = min(val_loss, best_loss)

                                state = checkpoint_state(
                                    self.model, self._optimizer, val_loss, epoch, it)

                                save_checkpoint(
                                    state, is_best, osp.join(
                                        self.checkpoint_dir, "checkpoint"))
                    if self.trainer_monitor:
                        self.trainer_monitor.update({**monitor, "epoch":epoch, "it":it})

                pbar = tqdm.tqdm(
                    total=eval_frequency,
                    leave=False,
                    desc="train",
                    dynamic_ncols=True,
                )
                pbar.set_postfix(dict(total_it=it))
                self.trigger_event("epoch:after", self, epoch)
            self.trigger_event('epoch:finish', self)
        return best_loss