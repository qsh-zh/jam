import os.path as osp

import torch

__all__ = [
    "LossException",
    "NanException",
    "InfException",
    "step_lr",
    "register_grad_clip",
    "trainer_save_cfg",
    "check_loss_error",
]

# pylint: disable=unused-argument
class LossException(Exception):
    pass


class NanException(LossException):
    pass


class InfException(LossException):
    pass


def step_lr(trainer, *args, **kwargs):
    if trainer.lr_scheduler:
        trainer.lr_scheduler.step()
        trainer.cur_monitor.update(
            {
                "lr": trainer.optimizer.param_groups[0]["lr"],
            }
        )


def register_grad_clip(trainer, grad_clip_value):
    def grad_clip_fn(trainer, *args, **kwargs):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip_value,error_if_nonfinite=False)

    trainer.register_event("backward:after", grad_clip_fn, False)


def trainer_save_cfg(trainer, cfg):
    def save_cfg(_trainer, state_dict):
        state_dict["cfg"] = cfg

    trainer.register_event("trainer:export", save_cfg, False)


def check_loss_error(trainer):
    def _deal_error(_trainer):
        model = (
            _trainer.model.module
            if hasattr(_trainer.model, "module")
            else _trainer.model
        )
        from jammy.io import locate_newest_file
        from jammy.logging import get_logger
        from jamtorch.io import load_ckpt

        fckpt = locate_newest_file(_trainer.ckpt_dir, "*.pth")
        state = load_ckpt(_trainer.device, osp.join(_trainer.ckpt_dir, fckpt))
        msg_model = model.load_state_dict(state["model"])
        logger = get_logger()
        trainer_id = trainer.rank if hasattr(trainer, "rank") else 0
        logger.critical(f"{trainer_id}: reload model ckpt {msg_model}")

    def _check_loss_error(_trainer, batch, loss, *args):
        if torch.isnan(loss):
            _deal_error(_trainer)
            raise NanException
        if torch.isinf(loss):
            _deal_error(_trainer)
            raise InfException

    trainer.register_event("forward:after", _check_loss_error, False)
    # trainer.register_event("val:step", _check_loss_error, False)
