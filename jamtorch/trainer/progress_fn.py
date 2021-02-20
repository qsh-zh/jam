import sys

from jammy.utils.meter import AverageMeter
from jamtorch.utils.meta import as_float
from tqdm.auto import tqdm
from jammy.cli.cmdline_viz import CmdLineViz
from .monitor import group_prefix

__all__ = [
    "set_monitor_viz",
    "train_bar_init",
    "batch_bar_update",
    "epoch_bar_update",
    "batch_bar_new",
    "bar_close",
    "simple_step_summary",
    "val_start",
    "val_step",
    "val_end",
    "simple_train_bar",
    "simple_val_bar",
]


def set_monitor_viz(trainer):
    trainer.cur_monitor = dict()
    trainer.cmdviz = CmdLineViz()


def train_bar_init(trainer):
    trainer.num_batch = len(trainer.train_loader)
    trainer.epoch_bar = tqdm(
        total=trainer._cfg.epochs,
        desc="epochs",
        dynamic_ncols=True,
        initial=trainer.epoch_cnt,
        file=sys.stdout,
    )
    trainer.batch_bar = tqdm(
        total=trainer.num_batch,
        leave=False,
        desc="train",
        dynamic_ncols=True,
        file=sys.stdout,
    )


def batch_bar_update(trainer):
    trainer.batch_bar.update()
    trainer.batch_bar.set_postfix(
        dict(total_it=trainer.iter_cnt, loss=trainer.batch_loss)
    )


def epoch_bar_update(trainer):
    trainer.epoch_bar.update()
    trainer.epoch_bar.set_postfix(
        dict(total_it=trainer.iter_cnt, loss=trainer.batch_loss)
    )


def batch_bar_new(trainer):
    trainer.batch_bar.close()
    trainer.batch_bar = tqdm(
        total=trainer.num_batch,
        leave=False,
        desc="train",
        dynamic_ncols=True,
        file=sys.stdout,
    )


def bar_close(trainer):
    trainer.epoch_bar.close()
    trainer.batch_bar.close()
    sys.stdout.flush()


def val_start(trainer):
    trainer.val_loss_meter = AverageMeter()
    trainer.val_bar = tqdm(
        total=len(trainer.val_loader),
        leave=False,
        desc="val",
        dynamic_ncols=True,
        file=sys.stdout,
    )


def val_step(trainer, batch, loss, monitor, cmdviz_dict):
    trainer.val_loss_meter.update(loss)
    cmdviz_dict.update({"loss": trainer.val_loss_meter.val})
    f_cmdviz = as_float(cmdviz_dict)
    trainer.cmdviz.update("eval", f_cmdviz)

    trainer.val_bar.update()
    trainer.val_bar.set_postfix({"loss": trainer.val_loss_meter.avg.item()})


def val_end(trainer):
    val_loss = trainer.val_loss_meter.avg.item()
    trainer.cur_monitor.update(group_prefix("eval", trainer.cmdviz.meter["eval"].avg))
    trainer.val_bar.close()
    trainer.save_ckpt(val_loss)
    trainer.cmdviz.flush()


def simple_step_summary(trainer, loss, monitors, cmdviz_dict):
    loss_f = as_float(loss)
    cmdviz_f = as_float(cmdviz_dict)
    monitors = as_float(monitors)

    cmdviz_f.update({"loss": loss_f})
    monitors.update(cmdviz_f)

    trainer.batch_loss = loss_f
    trainer.cur_monitor.update(group_prefix("train", monitors))
    trainer.cmdviz.update("train", cmdviz_f)


def simple_train_bar(trainer):
    trainer.register_event("epoch:start", set_monitor_viz, False)
    trainer.register_event("epoch:start", train_bar_init, False)
    trainer.register_event("step:end", batch_bar_update, False)
    trainer.register_event("epoch:after", batch_bar_new, False)
    trainer.register_event("epoch:after", epoch_bar_update, False)
    trainer.register_event("epoch:end", bar_close, False)

    trainer.register_event("step:summary", simple_step_summary, False)


def simple_val_bar(trainer):
    trainer.register_event("epoch:start", set_monitor_viz, False)
    trainer.register_event("val:start", val_start, False)
    trainer.register_event("val:step", val_step, False)
    trainer.register_event("val:end", val_end, False)
