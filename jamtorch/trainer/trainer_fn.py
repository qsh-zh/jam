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
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip_value)

    trainer.register_event("backward:after", grad_clip_fn, False)


def trainer_save_cfg(trainer, cfg):
    def save_cfg(_trainer, state_dict):
        state_dict["cfg"] = cfg

    trainer.register_event("trainer:export", save_cfg, False)


def check_loss_error(trainer):
    def _check_loss_error(_trainer, batch, loss, *args):
        if torch.isnan(loss):
            raise NanException
        if torch.isinf(loss):
            raise InfException

    trainer.register_event("forward:after", _check_loss_error, False)
    trainer.register_event("val:step", _check_loss_error, False)
