import torch
from jamtorch.trainer import GeneticTrainer
from omegaconf import DictConfig
import torch.distributed as dist
from jammy.utils.hyd import hyd_instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from jamtorch.utils.meta import is_master
import jammy.utils.hyd as hyd
import jamtorch.trainer.progress_fn as progress_fn
from jamtorch.io import attr_dict
import os.path as osp
import tempfile

from jamtorch.logging import get_logger

logger = get_logger()


class DDPTrainer(GeneticTrainer):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        self.train_sampler, self.val_sampler = None, None
        self.is_master = is_master()
        self.rank = 0 if self.is_master else dist.get_rank()
        self.setup_ddp()

    def set_model_optim(self, model, optimizer):
        # assert isinstance(optimizer_cfg, DictConfig)
        if next(model.parameters()).device == torch.device("cpu"):
            model = model.to(self.device)
        if self._cfg.dist.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[self.rank])
        # instantiate optmiizer after DDP model
        # optimizer = hyd_instantiate(optimizer_cfg, model.parameters())
        super().set_model_optim(model, optimizer)

    def init_model(self):
        if self._cfg.resume:
            return
        checkpoint_path = osp.join(tempfile.gettempdir(), "initial_weights.pt")
        if self.is_master:
            torch.save(self.model.state_dict(), checkpoint_path)

        dist.barrier()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def set_sampler(self, train_sampler, val_sampler):
        self.train_sampler, self.val_sampler = train_sampler, val_sampler

    def setup_ddp(self):
        self.register_event(
            "epoch:before",
            lambda trainer: trainer.train_sampler.set_epoch(self.epoch_cnt),
            False,
        )
        if self.is_master:
            progress_fn.simple_train_bar(self)
            progress_fn.simple_val_bar(self)

    def _impl_load_ckpt(self, state):
        """
        only load model, bypass optimizer
        """
        # The creatation of optimizer needs wait after model
        msg_model = self.model.module.load_state_dict(state["model"])
        logger.critical(f"load model ckpt {msg_model}")
        self._impl_load_amp_scaler(state)

        dist.barrier()

    def save_ckpt(self, val_loss=float("inf")):
        if self.is_master:
            return super().save_ckpt(val_loss)
        return

    def _impl_save_ckpt(self):
        super_ckpt = attr_dict(self, ["model", "amp_scaler"])
        return super_ckpt
