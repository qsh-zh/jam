from jammy.cli.cmdline_viz import CmdLineViz

from . import progress_fn
from .genetic_trainer import GeneticTrainer
from .trainer_monitor import TrainerMonitor


class SimpleTrainer(GeneticTrainer):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        self.cmdviz = CmdLineViz()
        self.trainer_monitor = None
        # cache state accross different methods
        self.cur_monitor = dict()
        self.cur_cmdviz = dict()
        self.set_up()

    def set_up(self):
        progress_fn.simple_train_bar(self)
        progress_fn.simple_val_bar(self)

    def monitor_update(self):
        if self.trainer_monitor:
            self.trainer_monitor.update(
                {
                    **self.cur_monitor,
                    "epoch": self.epoch_cnt,
                    "iter": self.iter_cnt,
                }
            )
        self.cur_monitor = dict()

    def set_monitor(self, is_wandb, tblogger=False):
        """
        docstring
        """
        self.trainer_monitor = TrainerMonitor(is_wandb, tblogger)
