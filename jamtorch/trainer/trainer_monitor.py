from jammy.utils.meter import GroupMeters
import wandb


class TrainerMonitor:
    def __init__(self, is_wandb=False, tb_logger=False):
        self.is_wandb = is_wandb
        self.counter = GroupMeters()
        if isinstance(tb_logger, bool):
            assert tb_logger == False
            self._tb_logger = None
            self.is_tblogger = False
        else:
            self._tb_logger = tb_logger
            self.is_tblogger = True

    def update(self, updates=None, value=None, **kwargs):
        """
        Example:
            >>> Monitor.update(key, value)
            >>> Monitor.update({key1: value1, key2: value2})
            >>> Monitor.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        if self.is_wandb:
            wandb.log(updates)
        if self.is_tblogger:
            for k, v in updates.items():
                self.counter[k].update(1, 1)
                self._tb_logger.scalar_summary(k, v, self.counter[k].tot_count)

    # TODO: add a method for adding image here! 
    def flush(self):
        if self.is_wandb:
            # TODO:
            pass
        if self.is_tblogger:
            self._tb_logger.flush()

    def close(self):
        if self.is_wandb:
            wandb.finish()
        if self.is_tblogger:
            self._tb_logger.close()