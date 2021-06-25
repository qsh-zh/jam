import copy

import torch
import torch.nn as nn

from jamtorch.logging import get_logger

__all__ = ["EMA"]


class EMA:
    def __init__(
        self, beta, num_warm, num_every, model=None, forget_resume=False
    ):  # pylint: disable=too-many-arguments
        self.beta = beta
        self.num_warm = num_warm
        self.num_every = num_every
        self.cnt = 0
        if isinstance(model, nn.DataParallel):
            model = model.module
        self._old_model = copy.deepcopy(model)
        self.forget_resume = forget_resume

    def forward(self, *args, **kwargs):
        return self._old_model(*args, **kwargs)

    def update_parameters(self, new_model):
        if hasattr(new_model, "module"):
            new_model = new_model.module
        self.cnt += 1
        if self.cnt < self.num_warm:
            return
        if self.model is None:
            self.model = new_model
        if self.cnt == self.num_warm:
            self._old_model.load_state_dict(new_model.state_dict())
            return
        with torch.no_grad():
            if self.cnt % self.num_every == 0:
                for old_param, new_param in zip(
                    self.model.parameters(), new_model.parameters()
                ):
                    old_weight, new_weight = old_param.data, new_param.data
                    old_param.data = (
                        old_weight * self.beta + (1 - self.beta) * new_weight
                    )

    @property
    def model(self):
        return self._old_model

    @model.setter
    def model(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
        self._old_model = copy.deepcopy(model)

    def state_dict(self):
        if self._old_model is None:
            model_params = None
        else:
            model_params = self._old_model.state_dict()
        return {
            "cnt": self.cnt,
            "model": model_params,
            "num_warm": self.num_warm,
            "num_every": self.num_every,
            "beta": self.beta,
        }

    def copy_to(self, model):  # pylint: disable= unused-argument
        return self._old_model

    def load_state_dict(self, ema_dict):
        logger = get_logger()
        self.cnt = ema_dict.get("cnt") or 0
        self.num_warm = ema_dict.get("num_warm") or 0
        if not self.forget_resume:
            if self._old_model is not None and ema_dict.get("model") is not None:
                self._old_model.load_state_dict(ema_dict.get("model"))
        self.beta = ema_dict.get("beta") or 0.9
        self.num_every = ema_dict.get("num_every") or 1
        logger.critical("=====> Loading EMA finish =====>")
