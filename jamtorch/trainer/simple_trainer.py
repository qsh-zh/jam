from .genetic_trainer import GeneticTrainer
from .progress_fn import *
from jammy.cli.cmdline_viz import CmdLineViz
from .trainer_monitor import TrainerMonitor
from jamtorch.io import attr_dict, save_ckpt,load_ckpt
import os.path as osp


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
        self.register_event("epoch:start", train_bar_init, False)
        self.register_event("step:end", batch_bar_update, False)
        self.register_event("epoch:after", batch_bar_new, False)
        self.register_event("epoch:after", epoch_bar_update, False)
        self.register_event("epoch:end", bar_close, False)

        self.register_event("step:summary", simple_step_summary, False)

        self.register_event("val:start", val_start, False)
        self.register_event("val:step", val_step, False)
        self.register_event("val:end", val_end, False)

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

    def eval(self):
        super().eval()
        self.cmdviz.flush()

    def set_monitor(self, is_wandb, tblogger=False):
        """
        docstring
        """
        self.trainer_monitor = TrainerMonitor(is_wandb, tblogger)
