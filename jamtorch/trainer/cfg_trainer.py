from .trainer import Trainer
from jamtorch.io import EWA

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

__all__ = ["CfgTrainer"]


class CfgTrainer(Trainer):
    def load_env(self, cfg):
        super().load_env(cfg)
        is_ewa = cfg.get("ewa") or False
        if is_ewa:
            self.ewa = EWA(
                cfg.get("ewa_beta"),
                cfg.get("ewa_num_warm"),
                cfg.get("ewa_num_every"),
                self.model,
            )
            if cfg.get("ewa_state") is not None:
                # if cfg from checkpoint
                self.ewa.load_dict(cfg.get("ewa_state"))
        else:
            self.ewa = None
        self.load_fp16(cfg)

    def load_fp16(self, cfg):
        self.fp16 = bool(cfg.get("fp16") or False)
        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, "INSTALL Apex"
        if self.fp16:
            if self.ewa:
                (self.model, self.ewa.model), self.optimizer = amp.initialize(
                    [self.model, self.ewa.model], self.optimizer, opt_level="O1"
                )
            else:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level="O1"
                )

    def export_env(self):
        basic_state = super().export_env()
        new_state = {
            "ewa": bool(self.ewa),
            "ewa_beta": self.ewa.beta if self.ewa else 0.9,
            "ewa_num_warm": self.ewa.num_warm if self.ewa else 1,
            "ewa_num_every": self.ewa.num_every if self.ewa else 1,
            "ewa_state": self.ewa.dump2dict() if self.ewa else None,
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
        if self.ewa:
            self.ewa.update_model_average(self.model)
        return rtn

    def loss_backward(self, loss):
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return True
