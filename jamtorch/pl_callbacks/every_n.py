from abc import abstractmethod

import pytorch_lightning as pl  # pylint: disable=unused-import
from pytorch_lightning import Callback


class EveryN(Callback):
    def __init__(self, every_n):
        self.every_n = every_n

    def on_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.every_n_impl(trainer, pl_module)

    @abstractmethod
    def every_n_impl(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        ...
