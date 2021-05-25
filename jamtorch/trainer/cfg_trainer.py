from .trainer import Trainer
from jamtorch.io import EMA
import torch.nn as nn

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

__all__ = ["CfgTrainer"]


class CfgTrainer(Trainer):
    def load_env(self, cfg):
        super().load_env(cfg)
        is_ema = cfg.get("ema") or False
        if is_ema:
            self.ema = EMA(
                cfg.get("ema_beta"),
                cfg.get("ema_num_warm"),
                cfg.get("ema_num_every"),
                self.model,
            )
            if cfg.get("ema_state") is not None:
                # if cfg from checkpoint
                self.ema.load_state_dict(cfg.get("ema_state"))
        else:
            self.ema = None
        self.load_fp16(cfg)

    def load_fp16(self, cfg):
        self.fp16 = bool(cfg.get("fp16") or False)
        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, "INSTALL Apex"
        if self.fp16:
            if isinstance(self.model, nn.DataParallel):
                raise RuntimeError("DISABLE fp16 if using nn.DataParallel")

            if self.ema:
                (self.model, self.ema.model), self.optimizer = amp.initialize(
                    [self.model, self.ema.model], self.optimizer, opt_level="O1"
                )
            else:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level="O1"
                )

    def export_env(self):
        basic_state = super().export_env()
        new_state = {
            "ema": bool(self.ema),
            "ema_beta": self.ema.beta if self.ema else 0.9,
            "ema_num_warm": self.ema.num_warm if self.ema else 1,
            "ema_num_every": self.ema.num_every if self.ema else 1,
            "ema_state": self.ema.state_dict() if self.ema else None,
            "fp16": self.fp16,
        }
        return {**basic_state, **new_state}

    def load_cfg(self, cfg):
        self._cfg = cfg
        if cfg.resume:
            self.load_ckpt(cfg.ckpt, cfg.ckpt)
        else:
            self.load_env(cfg)

    def __call__(self, train_loader, val_loader):
        if self._cfg.resume:
            self.load_ckpt(self._cfg.ckpt)
            start_epoch = self.epoch_cnt + 1
            start_iter = self.iter_cnt + 1
            loss = self.best_loss
        else:
            start_epoch, start_iter, loss = 0, 0, float("inf")
        n_epochs = self._cfg.epochs
        return self.train(
            n_epochs, train_loader, val_loader, loss, start_iter, start_epoch
        )

    def train_step(self, feed_dict):
        rtn = super().train_step(feed_dict)
        if self.ema:
            self.ema.update_parameters(self.model)
        return rtn

    def loss_backward(self, loss):
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return True
