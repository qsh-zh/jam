import functools

import matplotlib.pyplot as plt
import wandb

from .wandb_utils import Wandb

__all__ = ["wandb_plt", "wandb_img"]


def wandb_plt(func):
    @functools.wraps(func)
    def wandb_record(*args, **kwargs):
        fig, caption = func(*args, **kwargs)
        msg = None
        if Wandb.IS_ACTIVE:
            title = (
                caption
                if fig._suptitle is None  # pylint: disable= protected-access
                else fig._suptitle.get_text()  # pylint: disable= protected-access
            )
            msg = {title: wandb.Image(fig, caption=caption)}
            Wandb.log(msg)
        plt.close(fig)
        return msg

    return wandb_record


def wandb_img(title, fpath, caption=None):
    if Wandb.IS_ACTIVE:
        msg = {title: wandb.Image(fpath, caption=caption)}
        wandb.log(msg)
