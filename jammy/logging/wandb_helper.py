import functools

import matplotlib.pyplot as plt
import wandb

from .wandb_utils import Wandb

__all__ = ["wandb_plt"]


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
