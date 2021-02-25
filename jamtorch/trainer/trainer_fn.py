import torch

__all__ = ["step_lr", "register_grad_clip"]


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
