import torch
from jamtorch.logging import get_logger
import torch.nn as nn

__all__ = ["ParamEMA"]

logger = get_logger()


class ParamEMA:
    def __init__(self, beta, num_warm, num_every, model=None, forget_resume=False):
        if beta < 0.0 or beta > 1.0:
            raise ValueError("beta must be between 0 and 1")
        self.one_minus_decay = 1 - beta
        self.cnt = 0
        self.num_warm = num_warm
        self.num_every = num_every
        self.shadow_params = None
        self.forget_resume = forget_resume

    def update_parameters(self, new_model):
        if hasattr(new_model, "module"):
            new_model = new_model.module
        self.cnt += 1
        if self.cnt < self.num_warm:
            return
        if self.shadow_params is None:
            logger.debug("Ema finish the initialization")
            self.shadow_params = [
                p.clone().detach().cpu() for p in new_model.parameters()
            ]
        with torch.no_grad():
            if self.cnt % self.num_every == 0:
                for s_param, param in zip(self.shadow_params, new_model.parameters()):
                    s_param.sub_(self.one_minus_decay * (s_param - param.cpu()))

    def copy_to(self, model):
        if hasattr(model, "module"):
            model = model.module
        for s_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(s_param.to(param.device).data)
        return model

    def state_dict(self):
        return {
            "one_minus_decay": self.one_minus_decay,
            "model": self.shadow_params,
            "num_warm": self.num_warm,
            "num_every": self.num_every,
            "cnt": self.cnt,
        }

    def load_state_dict(self, ema_dict):
        self.cnt = ema_dict.get("cnt") or 0
        self.num_warm = ema_dict.get("num_warm") or 0
        self.num_every = ema_dict.get("num_every") or 1
        if not self.forget_resume:
            self.shadow_params = ema_dict.get("model") or []
            if self.shadow_params:
                self.shadow_params = [param.cpu() for param in ema_dict.get("model")]
            # for param in self.shadow_params:
            # param.cpu()
        self.one_minus_decay = ema_dict.get("one_minus_decay") or 0.1
        logger.critical(
            f"=====> Loading EMA finish =====> with forget_resume: {self.forget_resume}"
        )
