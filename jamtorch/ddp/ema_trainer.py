from .ddp_trainer import DDPTrainer
from .fns import setup_ema

class EMATrainer(DDPTrainer):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        setup_ema(cfg, self)

    def _impl_load_ckpt(self, state):
        if self.ema and "ema" in state:
            if state["ema"]:
                self.ema.load_state_dict(state["ema"])
        super()._impl_load_ckpt(state)

    def _impl_save_ckpt(self):
        rtn = {}
        if self.ema:
            rtn["ema"] = self.ema.state_dict()
        return {**super()._impl_save_ckpt(), **rtn}
