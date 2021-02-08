__all__ = ["step_lr", "EMA"]


def step_lr(trainer, *args, **kwargs):
    if trainer.lr_scheduler:
        trainer.lr_scheduler.step()
        trainer.cur_monitor.update(
            {
                "lr": trainer.optimizer.param_groups[0]["lr"],
            }
        )


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
