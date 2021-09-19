from .trainer import Trainer
from jamtorch.io import EMA
import torch.nn as nn
import os
import torch.distributed as dist

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

__all__ = ["CfgTrainer"]


class TestTrainer(Trainer):
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
            self.ema = False
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

    def apex_parallel(self, cfg):
        if "WORLD_SIZE" in os.environ:
            cfg.ddp = int(os.environ["WORLD_SIZE"]) > 1
        self.ddp = bool(cfg.get("ddp") or False)
        if self.ddp:
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(cfg.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cudnn_bm = cfg.get("cudnn_bm") or False
        if "cudnn_bm" in cfg:
            torch.backends.cudnn.benchmark = cfg.get("cudnn_bm")

        if self.ddp:
            self.model = DistributedDataParallel(model, delay_allreduce=True)

    # TODO: update a smart verion
    def reduce_tensor_iter(self, iter_dict):
        pass

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= self.world_size
        return rt

    def _train_step_after(self, loss, monitors, cmdviz_dict):
        if not self.ddp:
            return super()._train_step_after(loss, monitors, cmdviz_dict)

        self.trigger_event("step:after", self)

        loss_f = as_float(self.reduce_tensor(loss))
        monitors_f = as_float(self.reduce_tensor_iter(monitors))
        cmdviz_f = as_float(self.reduce_tensor_iter(cmdviz_dict))

        cmdviz_f.update({"loss": loss_f})
        monitors_f.update(cmdviz_f)

        self.cur_monitor.update(group_prefix("train", monitors_f))
        self.cmdviz.update("train", cmdviz_f)
        return loss_f

    # TODO: update me
    def val_loss_ckpt(self, val_loss):
        is_best = val_loss < self.best_loss
        self.best_loss = min(val_loss, self.best_loss)
        if self.checkpoint_dir is not None:
            state = checkpoint_state(self.model, self.optimizer)
            state["env"] = self.export_env()

            # FIXME: CHECK IT
            save_checkpoint(
                state,
                is_best,
                osp.join(self.checkpoint_dir, "checkpoint"),
            )

    # TODO: update me, should adjust the API
    def eval_impl(self, data_loader, **kwargs):
        pass

    # TODO: should update epoch:start trigger, it should be desing as a call_list, auto stack it
    # train_sampler

    # TODO: update ddp env
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
